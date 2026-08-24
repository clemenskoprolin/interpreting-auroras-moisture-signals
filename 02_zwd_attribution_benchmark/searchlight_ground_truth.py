"""
Ground-truth causal effects for the ZWD searchlight benchmark.

For each searchlight mask r we compute

    zwd_plus  = zwd_t1 + magnitude * sigma_zwd * mask_r
    zwd_minus = zwd_t1 - magnitude * sigma_zwd * mask_r
    both clipped to the normalisation range [mu - 4 sigma, mu + 4 sigma].

    f(x) = target scalar of q at 850 hPa at t1+12h
    G_r  = |f(zwd_plus) - f(zwd_minus)| / 2    (magnitude)
    S_r  =  (f(zwd_plus) - f(zwd_minus)) / 2   (signed)

`magnitude` is passed in from the CLI (default 1.0 = ±1 sigma).

Only init-time t1 ZWD is perturbed; the t0 slice stays at the original
value.  All other variables (atmos, other surf, static) are untouched.

The ground-truth evaluation is expensive (2 forward passes per mask),
so we parallelise over SLURM ranks via a file-based barrier — same
pattern as RISE / ViT-CX in SCLIB.
"""

from __future__ import annotations

import dataclasses
import os
import time as _time
from dataclasses import dataclass

import numpy as np
import torch

from searchlight_tasks import MaskSpec, gaussian_mask, SCALES


def _rollout_final_pred(model, batch, steps: int):
    """Return the model prediction after `steps` autoregressive rollouts.

    Each step of the Aurora ZWD model is 6 h, so steps=1 -> +6h, steps=12 -> +72h.
    For steps==1 the single forward pass is done without input preprocessing
    (matches the prior behavior of compute_ground_truth). For steps>=2 the
    standard Aurora rollout is performed (type-cast + crop + to-device), and
    predictions are fed back as the next timestep's t1.
    """
    if steps < 1:
        raise ValueError(f"rollout steps must be >= 1, got {steps}")
    if steps == 1:
        pred = model.forward(batch)
        if isinstance(pred, tuple):
            pred = pred[0]
        return pred

    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if model.use_resolution_specific_patch_tokenizers:
        patch_size = model.patch_tokenizer_identifier.get_patch_size(
            batch.metadata.grid_resolution
        )
    else:
        patch_size = model.patch_size
    batch = batch.crop(patch_size=patch_size)
    batch = batch.to(p.device)

    pred = None
    for _ in range(steps):
        pred = model.forward(batch)
        if isinstance(pred, tuple):
            pred = pred[0]
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
    return pred


# ------------------------------------------------------------------
# Perturbation operator
# ------------------------------------------------------------------
def perturb_zwd(
    zwd_actual_1_2_H_W: torch.Tensor,
    mask_H_W: np.ndarray,
    sign: float,
    magnitude: float,
    zwd_loc: float,
    zwd_scale: float,
    timestep_idx: int = 1,
) -> torch.Tensor:
    """Return a perturbed copy of the ZWD tensor.

    By default only the t1 slice is modified; pass `timestep_idx=0` to
    perturb t0 instead.

    Args:
        zwd_actual_1_2_H_W: float32 tensor (1, 2, H, W) of ZWD values.
        mask_H_W: float32 np array (H, W) in [0, 1] (Gaussian).
        sign: +1 for plus-perturbation, -1 for minus-perturbation.
        magnitude: perturbation amplitude in units of sigma_zwd.
        zwd_loc: normalisation mean (for clipping).
        zwd_scale: normalisation std.
        timestep_idx: which history slice to perturb. `1` is the searchlight
            benchmark default (`t1`), `0` allows representation-tracing
            comparisons against `t0`.

    Returns:
        float32 tensor (1, 2, H, W), clipped to [mu - 4 sigma, mu + 4 sigma].
    """
    if timestep_idx not in (0, 1):
        raise ValueError(f"timestep_idx must be 0 or 1, got {timestep_idx}")

    out = zwd_actual_1_2_H_W.clone()
    delta = sign * magnitude * zwd_scale * torch.from_numpy(mask_H_W).to(out.dtype)
    out[0, timestep_idx] = out[0, timestep_idx] + delta

    lo = zwd_loc - 4.0 * zwd_scale
    hi = zwd_loc + 4.0 * zwd_scale
    out.clamp_(min=lo, max=hi)
    return out


# ------------------------------------------------------------------
# Baseline: Gaussian-blurred ZWD (shared across IG, RISE, ViT-CX)
# ------------------------------------------------------------------
def smoothed_zwd_baseline(
    zwd_actual_1_2_H_W: torch.Tensor,
    sigma_deg: float,
) -> torch.Tensor:
    """Return a CPU float32 tensor (1, 2, H, W) of spatially-smoothed ZWD.

    Uses scipy.ndimage.gaussian_filter with 'wrap' along longitude and
    'reflect' along latitude.  Both timesteps are smoothed independently.
    The grid resolution is assumed to be 0.25 degree/pixel.
    """
    from scipy.ndimage import gaussian_filter

    arr = zwd_actual_1_2_H_W.detach().cpu().numpy().copy()
    sigma_pix = sigma_deg / 0.25
    for t in range(arr.shape[1]):
        arr[0, t] = gaussian_filter(
            arr[0, t], sigma=sigma_pix, mode=("reflect", "wrap")
        )
    return torch.from_numpy(arr).float()


# ------------------------------------------------------------------
# Ground-truth evaluation
# ------------------------------------------------------------------
@dataclass
class GTResult:
    mask_keys: list[str]
    G: np.ndarray          # (n_masks,) magnitudes
    S: np.ndarray          # (n_masks,) signed
    f_plus: np.ndarray     # (n_masks,)
    f_minus: np.ndarray    # (n_masks,)


def compute_ground_truth(
    *,
    masks: list[MaskSpec],
    case_data,
    device,
    model,
    target_fns: dict,
    lat_vals,
    lon_vals,
    magnitude: float,
    make_batch_with_zwd,
    rank: int,
    world_size: int,
    tmp_dir: str,
    verbose: bool = True,
    rollout_steps: int = 1,
) -> dict:
    """Compute G_r and S_r for every mask, for one or more target functions.

    `target_fns` is a dict `{mode_name: callable(pred) -> Tensor}`; each scalar
    target is evaluated on the same forward passes (no extra GPU cost).

    Returns (on rank 0) `{mode_name: GTResult}`. Other ranks return empty dict.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    zwd_actual = case_data.surf_cpu["zwd"]  # (1, 2, H, W)
    zwd_loc = case_data.zwd_loc
    zwd_scale_val = case_data.zwd_scale

    mode_names = list(target_fns.keys())

    n_masks = len(masks)
    per_rank = n_masks // world_size
    extra = n_masks % world_size
    my_n = per_rank + (1 if rank < extra else 0)
    my_start = rank * per_rank + min(rank, extra)
    my_end = my_start + my_n
    my_masks = masks[my_start:my_end]

    my_keys: list[str] = []
    my_fp: dict[str, list[float]] = {mode: [] for mode in mode_names}
    my_fm: dict[str, list[float]] = {mode: [] for mode in mode_names}

    for i, spec in enumerate(my_masks):
        sigma = SCALES[spec.scale].sigma_deg
        mask_np = gaussian_mask(spec, sigma, lat_vals, lon_vals)

        zwd_plus = perturb_zwd(
            zwd_actual, mask_np, +1.0, magnitude, zwd_loc, zwd_scale_val
        )
        zwd_minus = perturb_zwd(
            zwd_actual, mask_np, -1.0, magnitude, zwd_loc, zwd_scale_val
        )

        with torch.no_grad():
            batch_p = make_batch_with_zwd(zwd_plus)
            pred_p = _rollout_final_pred(model, batch_p, rollout_steps)
            for mode, fn in target_fns.items():
                my_fp[mode].append(float(fn(pred_p).item()))
            del batch_p, pred_p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_m = make_batch_with_zwd(zwd_minus)
            pred_m = _rollout_final_pred(model, batch_m, rollout_steps)
            for mode, fn in target_fns.items():
                my_fm[mode].append(float(fn(pred_m).item()))
            del batch_m, pred_m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        my_keys.append(spec.key)

        if verbose and ((i + 1) % 5 == 0 or i == my_n - 1):
            print(f"        [Rank {rank}] GT {i + 1}/{my_n} masks "
                  f"({len(mode_names)} target{'s' if len(mode_names) > 1 else ''})")

    for mode in mode_names:
        np.save(os.path.join(tmp_dir, f"_gt_fp_{mode}_r{rank}.npy"),
                np.asarray(my_fp[mode]))
        np.save(os.path.join(tmp_dir, f"_gt_fm_{mode}_r{rank}.npy"),
                np.asarray(my_fm[mode]))
    with open(os.path.join(tmp_dir, f"_gt_keys_r{rank}.txt"), "w") as f:
        f.write("\n".join(my_keys))
    with open(os.path.join(tmp_dir, f"_gt_done_r{rank}"), "w") as f:
        f.write("done")

    if rank != 0:
        return {}

    for r in range(1, world_size):
        marker = os.path.join(tmp_dir, f"_gt_done_r{r}")
        waited = 0
        while not os.path.exists(marker):
            _time.sleep(2)
            waited += 2
            if waited > 3600:
                print(f"  WARNING: GT rank {r} did not finish in time!")
                break

    all_keys: list[str] = []
    all_fp: dict[str, list[float]] = {mode: [] for mode in mode_names}
    all_fm: dict[str, list[float]] = {mode: [] for mode in mode_names}
    for r in range(world_size):
        with open(os.path.join(tmp_dir, f"_gt_keys_r{r}.txt"), "r") as f:
            keys = [ln for ln in f.read().splitlines() if ln]
        all_keys.extend(keys)
        for mode in mode_names:
            fp = np.load(os.path.join(tmp_dir, f"_gt_fp_{mode}_r{r}.npy"))
            fm = np.load(os.path.join(tmp_dir, f"_gt_fm_{mode}_r{r}.npy"))
            all_fp[mode].extend(fp.tolist())
            all_fm[mode].extend(fm.tolist())

    for r in range(world_size):
        for fname in (
            f"_gt_keys_r{r}.txt", f"_gt_done_r{r}",
        ):
            p = os.path.join(tmp_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        for mode in mode_names:
            for fname in (
                f"_gt_fp_{mode}_r{r}.npy", f"_gt_fm_{mode}_r{r}.npy",
            ):
                p = os.path.join(tmp_dir, fname)
                if os.path.exists(p):
                    os.remove(p)

    results: dict[str, GTResult] = {}
    for mode in mode_names:
        fp_arr = np.asarray(all_fp[mode], dtype=np.float64)
        fm_arr = np.asarray(all_fm[mode], dtype=np.float64)
        G = np.abs(fp_arr - fm_arr) / 2.0
        S = (fp_arr - fm_arr) / 2.0
        results[mode] = GTResult(
            mask_keys=all_keys, G=G, S=S, f_plus=fp_arr, f_minus=fm_arr
        )
    return results
