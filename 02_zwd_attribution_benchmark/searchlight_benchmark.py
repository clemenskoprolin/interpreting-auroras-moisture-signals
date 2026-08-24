"""
ZWD Searchlight Benchmark — Phase 2 real-task benchmark
=========================================================

Compares saliency / IG / RISE / ViT-CX on a physically-grounded
causal task: does each method recover the regions where a +/- sigma
local perturbation of init-time ZWD actually changes a downstream
q850 target scalar?

Ground truth is computed by direct forward passes over symmetric
Gaussian searchlight masks. See README.md for reproduced configurations.

Usage
-----
    python 02_zwd_attribution_benchmark/searchlight_benchmark.py \\
        --methods saliency ig rise vit_cx \\
        --targets ticino california japan \\
        --scales local synoptic \\
        --output-dir results/zwd_attribution_benchmark/6h_box
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# --- Path wiring ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from aurora import Aurora  # noqa: E402

from searchlight_data import (  # noqa: E402
    CaseData, CHECKPOINT_PATH, SURF_VARS, ATMOS_VARS,
    load_case, make_batch,
)
from searchlight_tasks import (  # noqa: E402
    TARGETS, SCALES, Case, default_cases, load_cases_from_json,
    MaskSpec, generate_mask_centers, gaussian_mask, cos_lat_weights,
    box_indices, nearest_gridpoint_indices, TargetRegion,
)
from searchlight_ground_truth import (  # noqa: E402
    compute_ground_truth, perturb_zwd, smoothed_zwd_baseline,
)
from searchlight_metrics import evaluate  # noqa: E402
from searchlight_report import (  # noqa: E402
    save_ground_truth, save_method_result,
    plot_case_scatter,
)
from generate_leaderboard import write_leaderboard, collect_rows  # noqa: E402

from xia_methods.saliency import saliency as _xia_saliency  # noqa: E402
from xia_methods.smoothgrad import smoothgrad as _xia_smoothgrad  # noqa: E402
from xia_methods.ig import integrated_gradients as _xia_ig  # noqa: E402
from xia_methods.rise import (  # noqa: E402
    accumulate_rise_with_stats,
    normalize_rise_covariance,
)
from xia_methods.vit_cx import (  # noqa: E402
    extract_feature_map, cluster_features,
    score_clusters, aggregate_and_upsample,
)

INPUT_H, INPUT_W = 721, 1440
TARGET_LEVEL_HPA = 850

RISE_N_MASKS_DEFAULT = 200
RISE_CELLS_H = 18
RISE_CELLS_W = 36
RISE_P = 0.5
VIT_CX_STAGE_DEFAULT = 2
VIT_CX_DIST_THRESH = 0.08


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="ZWD Searchlight Benchmark")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of target short names (ticino, california, japan).")
    p.add_argument("--methods", nargs="+",
                   default=["saliency", "ig", "rise", "vit_cx"])
    p.add_argument("--scales", nargs="+", default=["local", "synoptic"])
    p.add_argument("--cases", type=str, default="auto",
                   help="'auto' for built-in default_cases, or path to cases.json.")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--target-mode", choices=("box", "point"), default="box",
                   help="Target scalar: 'box' = mean q850 over the target box; "
                        "'point' = q850 at the nearest grid point to the target "
                        "box center.")
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--smoothgrad-n-samples", type=int, default=16,
                   help="Number of noisy samples to average for SmoothGrad.")
    p.add_argument("--smoothgrad-noise-sigma-frac", type=float, default=0.15,
                   help="Gaussian noise sigma as a fraction of ZWD std on the "
                        "case domain.")
    p.add_argument("--rise-n-masks", type=int, default=RISE_N_MASKS_DEFAULT)
    p.add_argument("--rise-cells-h", type=int, default=RISE_CELLS_H,
                   help="RISE coarse grid height (cells along latitude).")
    p.add_argument("--rise-cells-w", type=int, default=RISE_CELLS_W,
                   help="RISE coarse grid width (cells along longitude).")
    p.add_argument("--vit-cx-stage", type=int, default=VIT_CX_STAGE_DEFAULT)
    p.add_argument("--vit-cx-n-clusters", type=int, default=None,
                   help="Fixed ViT-CX cluster budget (one occlusion forward per "
                        "cluster). When set, overrides --vit-cx-dist-thresh.")
    p.add_argument("--vit-cx-dist-thresh", type=float, default=VIT_CX_DIST_THRESH,
                   help="Cosine-distance threshold for ViT-CX clustering.")
    p.add_argument("--vit-cx-no-smoothing", action="store_true",
                   help="Disable post-smoothing of the upsampled ViT-CX map.")
    p.add_argument("--magnitude", type=float, default=1.0,
                   help="Perturbation amplitude in units of sigma_zwd.")
    p.add_argument("--lead-time-hours", type=int, default=72,
                   help="Forecast lead time for the GT target scalar. Each Aurora "
                        "rollout step is 6 h, so this must be a positive multiple "
                        "of 6 (6=+6h, 12=+12h, 72=+72h, ...). NOTE: XAI methods "
                        "currently still use a single-step forward; for a fair "
                        "XAI-vs-GT comparison at lead > 6h, use --gt-only for now "
                        "and wait for the XAI-rollout wiring.")
    p.add_argument("--gt-only", action="store_true",
                   help="Compute and save ground truth only; skip all XAI methods.")
    p.add_argument("--no-gt", action="store_true",
                   help="Skip ground truth computation entirely; only run XAI methods. "
                        "Useful when you only need the attribution arrays (no scoring).")
    p.add_argument("--gt-mask-dir", type=str, default=None,
                   help="Optional: path to a previous output directory whose ground-truth "
                        "JSON files can be reused instead of recomputing. Cached files are "
                        "validated per (case, scale, mode) on case id, target location, "
                        "and mask keys; any missing or invalid combination is recomputed "
                        "while the valid ones are reused as-is.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true",
                   help="Smoke test: one target, one case, one scale, saliency only.")
    return p.parse_args()


# ===================================================================
# Model setup
# ===================================================================
def setup_model(device) -> Aurora:
    model = Aurora(
        surf_vars=SURF_VARS,
        static_vars=("lsm", "z", "slt"),
        atmos_vars=ATMOS_VARS,
        encoder_depths=(6, 10, 8),
        encoder_num_heads=(8, 16, 32),
        decoder_depths=(8, 10, 6),
        decoder_num_heads=(32, 16, 8),
        embed_dim=512,
        num_heads=16,
        autocast=True,
        use_lora=False,
        num_ensemble=1,
        encoder_activation_checkpointing=False,
    )
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = {k[4:]: v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    for layer in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        for blk in layer.blocks:
            orig = blk.forward
            def _make_cp(fn):
                def _cp(*a, **kw):
                    # Preserve autocast state during recomputation. The
                    # non-reentrant variant triggers CheckpointError in rollout
                    # backward because recomputed tensor metadata can differ.
                    if kw:
                        return torch_checkpoint(
                            lambda *args: fn(*args, **kw),
                            *a,
                            use_reentrant=True,
                        )
                    return torch_checkpoint(fn, *a, use_reentrant=True)
                return _cp
            blk.forward = _make_cp(orig)

    _enable_decoder_rollout_checkpointing(model)

    # Checkpoint encoder.level_agg (PerceiverResampler) to avoid saving a
    # ~126 GiB intermediate during backward on multi-step rollouts.
    # Perceiver3DEncoder has no built-in extensive_checkpointing, so we wrap here.
    for enc in [model.encoder] + list(getattr(model, 'encoders', [])):
        if hasattr(enc, 'level_agg'):
            orig_la = enc.level_agg.forward
            def _make_la_cp(fn):
                def _cp(*a, **kw):
                    # use_reentrant=True preserves autocast state during
                    # recomputation; use_reentrant=False causes CheckpointError
                    # when bfloat16 autocast changes tensor metadata on rerun.
                    if kw:
                        return torch_checkpoint(
                            lambda *args: fn(*args, **kw),
                            *a,
                            use_reentrant=True,
                        )
                    return torch_checkpoint(fn, *a, use_reentrant=True)
                return _cp
            enc.level_agg.forward = _make_la_cp(orig_la)

    print("ZWD-augmented Aurora loaded, frozen, grad-checkpointed.")
    return model


def _forward(model, batch):
    """Call model.forward and strip the (pred, std, preds) tuple."""
    out = model.forward(batch)
    if isinstance(out, tuple):
        out = out[0]
    return out


def rollout_mean(model, batch, steps: int):
    """Yield mean predictions from an autoregressive rollout.

    This is a compatibility wrapper around Aurora's rollout mechanics for the
    modified ZWD model used here. The model returns `(pred, std, preds)` for a
    one-step forward pass; this helper keeps the existing single-step behavior
    unchanged and only activates when a caller explicitly requests a rollout.

    It yields the same mean-prediction `Batch` shape as `_forward`, so existing
    target functions that expect one-step predictions can be reused unchanged at
    each rollout horizon.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if steps == 0:
        return

    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if model.use_resolution_specific_patch_tokenizers:
        patch_size = model.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
    else:
        patch_size = model.patch_size
    batch = batch.crop(patch_size=patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = _forward(model, batch)
        yield pred
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


def _gpu_sync_and_gc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================================
# Rollout infrastructure for gradient-based XAI methods
# ===================================================================
def _enable_decoder_rollout_checkpointing(model) -> None:
    """Wrap Aurora's decoder.level_decoder.forward with activation checkpointing
    when grad is enabled and an input requires grad. No-op otherwise, so this
    is safe to apply even for single-step / no-grad runs.

    Required to keep multi-step rollout backward passes in 0.25-deg + 13-level
    memory budget. Pattern matches searchlight_rollout_saliency.py.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return
    level_decoder = getattr(decoder, "level_decoder", None)
    if level_decoder is None:
        return
    if getattr(level_decoder, "_rollout_checkpoint_wrapped", False):
        return

    orig_forward = level_decoder.forward

    def _checkpointed_forward(latents, x):
        if torch.is_grad_enabled() and (
            getattr(latents, "requires_grad", False)
            or getattr(x, "requires_grad", False)
        ):
            # Match the Swin block / level_agg wrappers so autocast state is
            # preserved across recomputation in backward.
            return torch_checkpoint(orig_forward, latents, x, use_reentrant=True)
        return orig_forward(latents, x)

    level_decoder.forward = _checkpointed_forward
    level_decoder._rollout_checkpoint_wrapped = True


def _saved_tensors_cpu_context(enabled: bool):
    """Return torch.autograd.graph.save_on_cpu() if enabled and available,
    else a nullcontext. Offloads saved activations to CPU during backward to
    survive multi-step rollout with gradients at 0.25-deg resolution."""
    from contextlib import nullcontext

    if not enabled:
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    if graph_mod is None or not hasattr(graph_mod, "save_on_cpu"):
        return nullcontext()
    save_on_cpu = graph_mod.save_on_cpu
    try:
        return save_on_cpu(pin_memory=torch.cuda.is_available())
    except TypeError:
        return save_on_cpu()


class _RolloutForwardWrapper:
    """Thin wrapper that makes ``model.forward(batch)`` run an ``N``-step
    autoregressive rollout and return the final prediction.

    All other attribute access (reads and writes, e.g. ``model.autocast``) is
    delegated to the wrapped model, so the xia_methods can be passed this
    wrapper unchanged.
    """

    def __init__(self, model, steps: int):
        if steps < 1:
            raise ValueError(f"rollout steps must be >= 1, got {steps}")
        object.__setattr__(self, "_wrapped_model", model)
        object.__setattr__(self, "_rollout_steps", int(steps))

    def forward(self, batch):
        model = object.__getattribute__(self, "_wrapped_model")
        steps = object.__getattribute__(self, "_rollout_steps")
        if steps == 1:
            return _forward(model, batch)
        final = None
        for pred in rollout_mean(model, batch, steps):
            final = pred
        if final is None:
            raise RuntimeError(
                f"rollout_mean yielded no predictions for steps={steps}"
            )
        return final

    def __call__(self, batch):
        return self.forward(batch)

    def __getattr__(self, name):
        model = object.__getattribute__(self, "_wrapped_model")
        return getattr(model, name)

    def __setattr__(self, name, value):
        model = object.__getattribute__(self, "_wrapped_model")
        setattr(model, name, value)


# ===================================================================
# Target function
# ===================================================================
def _q850_level_index(case: CaseData) -> int:
    matches = np.where(np.asarray(case.pressure_levels) == TARGET_LEVEL_HPA)[0]
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one {TARGET_LEVEL_HPA} hPa level, got {matches.tolist()}"
        )
    return int(matches[0])


def _q850_box_mean(
    q: torch.Tensor,
    level_idx: int,
    lat_imin: int,
    lat_imax: int,
    lon_imin: int,
    lon_imax: int,
) -> torch.Tensor:
    return q[0, 0, level_idx,
             lat_imin:lat_imax + 1,
             lon_imin:lon_imax + 1].mean() * 1e3


def _q850_point_value(
    q: torch.Tensor,
    level_idx: int,
    lat_idx: int,
    lon_idx: int,
) -> torch.Tensor:
    return q[0, 0, level_idx, lat_idx, lon_idx] * 1e3


def make_q850_target(case: CaseData, target_short: str, target_mode: str):
    """Return a differentiable scalar fn plus JSON-safe metadata."""
    target = TARGETS[target_short]
    level_idx = _q850_level_index(case)

    if target_mode == "box":
        lat_imin, lat_imax, lon_imin, lon_imax = box_indices(
            target, case.lat_vals, case.lon_vals
        )

        def target_fn(pred):
            q = pred.atmos_vars["q"].float()
            return _q850_box_mean(
                q, level_idx, lat_imin, lat_imax, lon_imin, lon_imax
            )

        meta = {
            "target_short": target_short,
            "target_mode": "box",
            "level_hpa": TARGET_LEVEL_HPA,
            "selection": "box_mean",
            "center_lat": float(target.center_lat),
            "center_lon": float(target.center_lon),
            "box_lat": [float(target.box_lat[0]), float(target.box_lat[1])],
            "box_lon": [float(target.box_lon[0]), float(target.box_lon[1])],
            "lat_idx_min": lat_imin,
            "lat_idx_max": lat_imax,
            "lon_idx_min": lon_imin,
            "lon_idx_max": lon_imax,
        }
        return target_fn, meta

    if target_mode == "point":
        lat_idx, lon_idx = nearest_gridpoint_indices(
            case.lat_vals, case.lon_vals, target.center_lat, target.center_lon
        )
        actual_lat = float(case.lat_vals[lat_idx])
        actual_lon = float(case.lon_vals[lon_idx])

        def target_fn(pred):
            q = pred.atmos_vars["q"].float()
            return _q850_point_value(q, level_idx, lat_idx, lon_idx)

        meta = {
            "target_short": target_short,
            "target_mode": "point",
            "level_hpa": TARGET_LEVEL_HPA,
            "selection": "nearest_gridpoint_to_box_center",
            "requested_lat": float(target.center_lat),
            "requested_lon": float(target.center_lon),
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "lat": actual_lat,
            "lon": actual_lon,
        }
        return target_fn, meta

    raise ValueError(f"Unknown target mode: {target_mode!r}")


def make_q850_box_target(case: CaseData, target_short: str):
    """Backward-compatible wrapper for box-mode callers in downstream scripts."""
    target_fn, meta = make_q850_target(case, target_short, "box")
    box = (
        meta["lat_idx_min"],
        meta["lat_idx_max"],
        meta["lon_idx_min"],
        meta["lon_idx_max"],
    )
    return target_fn, box


def _format_target_meta(target_meta: dict) -> str:
    if target_meta["target_mode"] == "box":
        return (
            "target box (lat idx "
            f"{target_meta['lat_idx_min']}:{target_meta['lat_idx_max']}, lon idx "
            f"{target_meta['lon_idx_min']}:{target_meta['lon_idx_max']})"
        )
    return (
        "target point "
        f"(lat_idx={target_meta['lat_idx']}, lon_idx={target_meta['lon_idx']}, "
        f"lat={target_meta['lat']:.2f}, lon={target_meta['lon']:.2f}; "
        f"requested center={target_meta['requested_lat']:.2f},"
        f"{target_meta['requested_lon']:.2f})"
    )


def _output_case_id(base_case_id: str, target_mode: str) -> str:
    if target_mode == "box":
        return base_case_id
    return f"{base_case_id}__{target_mode}"


# ===================================================================
# Contrastive target functions (remote-box / global-mean)
# ===================================================================
def _remote_box_indices(
    target, lat_vals: np.ndarray, lon_vals: np.ndarray
) -> tuple[int, int, int, int]:
    """Antipodal reference box: same lat span, lon shifted by +180 deg (mod 360).

    Returns indices using a contiguous lon slice. If the shifted range would
    wrap across 0/360, we nudge it to a non-wrapping window of the same width
    near the same antipodal longitude — purely to keep indexing contiguous;
    the fairness guarantee (GT-independence) is unaffected.
    """
    lo_s, lo_n = target.box_lat
    lo_w, lo_e = target.box_lon
    width = lo_e - lo_w

    center_lon = 0.5 * (lo_w + lo_e)
    remote_center = (center_lon + 180.0) % 360.0
    rem_w = remote_center - 0.5 * width
    rem_e = remote_center + 0.5 * width

    # Keep contiguous: if wrap, slide into the nearest valid interior window.
    if rem_w < 0.0:
        rem_w, rem_e = 0.0, width
    if rem_e > 360.0:
        rem_w, rem_e = 360.0 - width, 360.0

    if lat_vals[0] > lat_vals[-1]:
        lat_imin = int(np.where(lat_vals <= lo_n)[0][0])
        lat_imax = int(np.where(lat_vals >= lo_s)[0][-1])
    else:
        lat_imin = int(np.where(lat_vals >= lo_s)[0][0])
        lat_imax = int(np.where(lat_vals <= lo_n)[0][-1])
    lon_imin = int(np.where(lon_vals >= rem_w)[0][0])
    lon_imax = int(np.where(lon_vals <= rem_e)[0][-1])
    return lat_imin, lat_imax, lon_imin, lon_imax


def make_contrast_target_fn(
    case, target_short: str, mode: str, device, target_mode: str = "box"
):
    """Build a contrastive scalar target_fn(pred) -> (f_target - f_reference).

    mode = "remote": f_reference = either an antipodal box mean (box mode) or
           the antipodal point value (point mode).
    mode = "global": f_reference = cos-lat-weighted global mean of q@850.

    The ±1e3 unit conversion used by the target is applied to both sides so
    the contrast is in g/kg as well.
    """
    from searchlight_tasks import TARGETS

    target = TARGETS[target_short]
    lat_vals = case.lat_vals
    lon_vals = case.lon_vals
    level_idx = _q850_level_index(case)

    if target_mode == "box":
        lat_imin, lat_imax, lon_imin, lon_imax = box_indices(target, lat_vals, lon_vals)

        def f_tgt(q):
            return _q850_box_mean(q, level_idx, lat_imin, lat_imax, lon_imin, lon_imax)

    elif target_mode == "point":
        lat_idx, lon_idx = nearest_gridpoint_indices(
            lat_vals, lon_vals, target.center_lat, target.center_lon
        )

        def f_tgt(q):
            return _q850_point_value(q, level_idx, lat_idx, lon_idx)

    else:
        raise ValueError(f"Unknown target mode: {target_mode!r}")

    if mode == "remote":
        if target_mode == "box":
            r_lat_imin, r_lat_imax, r_lon_imin, r_lon_imax = _remote_box_indices(
                target, lat_vals, lon_vals
            )

            def f_ref(q):
                return _q850_box_mean(
                    q, level_idx, r_lat_imin, r_lat_imax, r_lon_imin, r_lon_imax
                )

        else:
            r_lat_idx, r_lon_idx = nearest_gridpoint_indices(
                lat_vals, lon_vals, target.center_lat, target.center_lon + 180.0
            )

            def f_ref(q):
                return _q850_point_value(q, level_idx, r_lat_idx, r_lon_idx)

        def target_fn(pred):
            q = pred.atmos_vars["q"].float()
            return f_tgt(q) - f_ref(q)

        return target_fn

    if mode == "global":
        _cos_w_cache: list = []  # lazily populated from first pred shape

        def target_fn(pred):
            q = pred.atmos_vars["q"].float()
            q_h = q.shape[-2]
            if not _cos_w_cache:
                cw = torch.from_numpy(
                    cos_lat_weights(lat_vals[:q_h], q.shape[-1])
                ).to(q.device).float()
                _cos_w_cache.append(cw)
            cos_w = _cos_w_cache[0]
            f_t = f_tgt(q)
            f_ref = (q[0, 0, level_idx] * cos_w).sum() / cos_w.sum() * 1e3
            return f_t - f_ref

        return target_fn

    raise ValueError(f"Unknown contrast mode: {mode!r}")


# ===================================================================
# Method -> GT mode mapping
# ===================================================================
def _method_gt_mode(method_name: str) -> str:
    """Return the contrast mode whose GT this method should be scored against.

    Plain methods (saliency, ig, rise, vit_cx) are scored against the plain
    f_tgt GT. Each contrastive_*_remote/global is scored against a GT computed
    from the matching contrastive scalar f_tgt - f_ref.
    """
    if method_name.endswith("_remote"):
        return "remote"
    if method_name.endswith("_global"):
        return "global"
    return "plain"


# ===================================================================
# Attribution reductions (1,2,H,W) -> (H, W) for ZWD surface var
# ===================================================================
def _reduce_zwd_attr(attr_1_2_H_W: np.ndarray) -> np.ndarray:
    """Take the t1 timestep (the only one we perturb in ground truth).

    Returns signed (H, W) float32.  Keep-sign so that signed-Spearman is
    meaningful.  For magnitude-pooled metrics, the caller applies |.|.
    """
    return attr_1_2_H_W[0, 1].astype(np.float32)


# ===================================================================
# Saliency
# ===================================================================
def run_saliency(
    model, case: CaseData, target_fn, device, rollout_steps: int = 1,
) -> np.ndarray:
    def batch_fn(requires_grad=False):
        return make_batch(
            case, device,
            requires_grad_surf=("zwd",) if requires_grad else (),
        )

    model_fwd = (
        _RolloutForwardWrapper(model, rollout_steps)
        if rollout_steps > 1 else model
    )
    with _saved_tensors_cpu_context(rollout_steps > 1):
        result = _xia_saliency(
            model=model_fwd,
            batch_fn=batch_fn,
            target_fn=target_fn,
            atmos_var_names=(),
            surf_var_names=("zwd",),
            device=device,
        )
    g = result["grads"]["zwd"]   # (1, 2, H, W) signed
    if g is None:
        raise RuntimeError("saliency: no gradient returned for ZWD")
    return _reduce_zwd_attr(g)


# ===================================================================
# SmoothGrad (surface-ZWD)
# ===================================================================
def run_smoothgrad(
    model, case: CaseData, target_fn, device,
    n_samples: int, noise_sigma_frac: float, seed: int,
) -> np.ndarray:
    """SmoothGrad over ZWD: average gradient across n_samples noised copies.

    Noise sigma = noise_sigma_frac * std(ZWD) on the case domain, so the
    scale is case-relative.
    """
    zwd_actual_cpu = case.surf_cpu["zwd"]  # (1, 2, H, W) float32
    sigma = noise_sigma_frac * float(zwd_actual_cpu.std().item())
    generator = torch.Generator().manual_seed(seed)

    def batch_fn(requires_grad=False):
        if requires_grad:
            noise = torch.randn(
                zwd_actual_cpu.shape,
                generator=generator,
                dtype=zwd_actual_cpu.dtype,
            ) * sigma
            zwd_noised = zwd_actual_cpu + noise
            return make_batch(
                case, device,
                requires_grad_surf=("zwd",),
                zwd_override=zwd_noised,
            )
        return make_batch(case, device)

    result = _xia_smoothgrad(
        model=model,
        batch_fn=batch_fn,
        target_fn=target_fn,
        atmos_var_names=(),
        surf_var_names=("zwd",),
        device=device,
        n_samples=n_samples,
    )
    g = result["grads"]["zwd"]
    if g is None:
        raise RuntimeError("smoothgrad: no gradient returned for ZWD")
    return _reduce_zwd_attr(g)


# ===================================================================
# Integrated Gradients (surface-ZWD, smoothed baseline)
# ===================================================================
def run_ig(
    model, case: CaseData, target_fn, device,
    zwd_baseline_cpu: torch.Tensor, n_steps: int,
    rollout_steps: int = 1,
) -> np.ndarray:
    zwd_actual_cpu = case.surf_cpu["zwd"]     # (1, 2, H, W)
    zwd_delta_cpu = zwd_actual_cpu - zwd_baseline_cpu

    def batch_fn(alpha, requires_grad=False):
        zwd_interp = (zwd_baseline_cpu + alpha * zwd_delta_cpu).clone()
        return make_batch(
            case, device,
            requires_grad_surf=("zwd",) if requires_grad else (),
            zwd_override=zwd_interp,
        )

    surf_actual = {"zwd": zwd_actual_cpu}
    surf_baseline = {"zwd": zwd_baseline_cpu}

    model_fwd = (
        _RolloutForwardWrapper(model, rollout_steps)
        if rollout_steps > 1 else model
    )
    with _saved_tensors_cpu_context(rollout_steps > 1):
        result = _xia_ig(
            model=model_fwd,
            batch_fn=batch_fn,
            target_fn=target_fn,
            surf_actual=surf_actual,
            surf_baseline=surf_baseline,
            surf_var_names=("zwd",),
            device=device,
            n_steps=n_steps,
        )
    ig = result["ig"]["zwd"]   # (1, 2, H, W) signed
    return _reduce_zwd_attr(ig)


# ===================================================================
# ZWD-only masked batch helper
# ===================================================================
def _make_zwd_masked_batch(
    case: CaseData,
    device,
    mask_H_W: np.ndarray,
    zwd_baseline_cpu: torch.Tensor,
):
    """Build a Batch where only zwd_t1 is partially replaced by a baseline.

    Mask semantics: mask=1 keep original, mask=0 replace with baseline.
    Only the t1 timestep of ZWD is blended; t0 is left at the original
    actual value.  Every other variable is at actual.
    """
    zwd_actual = case.surf_cpu["zwd"]        # (1, 2, H, W)
    zwd_bl = zwd_baseline_cpu                # (1, 2, H, W)

    m = torch.from_numpy(mask_H_W.astype(np.float32))  # (H, W)
    blended_t1 = zwd_actual[0, 1] * m + zwd_bl[0, 1] * (1.0 - m)

    zwd_override = zwd_actual.clone()
    zwd_override[0, 1] = blended_t1

    return make_batch(case, device, zwd_override=zwd_override)


# ===================================================================
# RISE — ZWD-only, scale-dependent baseline
# ===================================================================
def run_rise(
    model, case: CaseData, target_fn, device,
    zwd_baseline_cpu: torch.Tensor,
    n_masks: int,
    cells_h: int, cells_w: int,
    seed: int, rank: int, world_size: int, tmp_dir: str,
) -> np.ndarray:
    per_rank = n_masks // world_size
    extra = n_masks % world_size
    my_n = per_rank + (1 if rank < extra else 0)
    my_start = rank * per_rank + min(rank, extra)

    def scorer_fn(mask_np):
        batch = _make_zwd_masked_batch(case, device, mask_np, zwd_baseline_cpu)
        with torch.no_grad():
            pred = _forward(model, batch)
            val = float(target_fn(pred).item())
        del batch, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val

    print(f"        [Rank {rank}] RISE: {my_n}/{n_masks} masks "
          f"(cells {cells_h}x{cells_w}, covariance-centered)")
    (
        my_sal,
        my_mask_sum,
        my_mask_sq_sum,
        my_score_sum,
        my_seen,
    ) = accumulate_rise_with_stats(
        scorer_fn=scorer_fn,
        n=my_n,
        cells_h=cells_h, cells_w=cells_w,
        H=INPUT_H, W=INPUT_W,
        p=RISE_P, seed=seed, start_idx=my_start,
        verbose=True, rank=rank,
    )

    if world_size == 1:
        raw = normalize_rise_covariance(
            my_sal, my_mask_sum, my_mask_sq_sum, my_score_sum, my_seen
        )
        return raw.astype(np.float32)

    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, f"_rise_sal_r{rank}.npy"), my_sal)
    np.save(os.path.join(tmp_dir, f"_rise_msk_r{rank}.npy"), my_mask_sum)
    np.save(os.path.join(tmp_dir, f"_rise_msk2_r{rank}.npy"), my_mask_sq_sum)
    np.save(
        os.path.join(tmp_dir, f"_rise_stats_r{rank}.npy"),
        np.array([my_score_sum, my_seen], dtype=np.float64),
    )
    with open(os.path.join(tmp_dir, f"_rise_done_r{rank}"), "w") as f:
        f.write("done")

    if rank != 0:
        return np.zeros((INPUT_H, INPUT_W), dtype=np.float32)

    for r in range(1, world_size):
        marker = os.path.join(tmp_dir, f"_rise_done_r{r}")
        waited = 0
        while not os.path.exists(marker):
            time.sleep(2)
            waited += 2
            if waited > 3600:
                print(f"  WARNING: RISE rank {r} did not finish in time!")
                break

    total_sal = np.zeros((INPUT_H, INPUT_W), dtype=np.float64)
    total_msk = np.zeros((INPUT_H, INPUT_W), dtype=np.float64)
    total_msk2 = np.zeros((INPUT_H, INPUT_W), dtype=np.float64)
    total_score = 0.0
    total_seen = 0
    for r in range(world_size):
        total_sal += np.load(os.path.join(tmp_dir, f"_rise_sal_r{r}.npy"))
        total_msk += np.load(os.path.join(tmp_dir, f"_rise_msk_r{r}.npy"))
        total_msk2 += np.load(os.path.join(tmp_dir, f"_rise_msk2_r{r}.npy"))
        stats = np.load(os.path.join(tmp_dir, f"_rise_stats_r{r}.npy"))
        total_score += float(stats[0])
        total_seen += int(stats[1])

    raw = normalize_rise_covariance(
        total_sal, total_msk, total_msk2, total_score, total_seen
    )

    for r in range(world_size):
        for f in (
            f"_rise_sal_r{r}.npy", f"_rise_msk_r{r}.npy",
            f"_rise_msk2_r{r}.npy", f"_rise_stats_r{r}.npy",
            f"_rise_done_r{r}",
        ):
            p = os.path.join(tmp_dir, f)
            if os.path.exists(p):
                os.remove(p)

    return raw.astype(np.float32)


# ===================================================================
# ViT-CX — option (a): standard clustering, ZWD-only occlusion
# ===================================================================
# Aurora stage grid table from Swin3DTransformerBackbone.get_encoder_specs:
#   stage 0: 4 x 180 x 360 = 259200
#   stage 1: 4 x  90 x 180 =  64800
#   stage 2: 4 x  45 x  90 =  16200
#
# The product-only assertion below would not catch a swapped H/W layout, so keep
# these dimensions in (latent_levels, latitude_tokens, longitude_tokens) order.
STAGE_GRID = {0: (4, 180, 360), 1: (4, 90, 180), 2: (4, 45, 90)}


def run_vit_cx(
    model, case: CaseData, target_fn, device,
    zwd_baseline_cpu: torch.Tensor,
    hook_stage: int, n_clusters: int | None, distance_threshold: float,
    smooth_sigma: tuple[float, float] | None,
    rank: int, world_size: int, tmp_dir: str,
) -> np.ndarray:
    # --- Step 1: feature extraction + original score ---
    feat_store: dict = {}

    def _feat_hook(module, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        feat_store["feat"] = t.detach().cpu().float()

    handle = model.backbone.encoder_layers[hook_stage].blocks[-1] \
        .register_forward_hook(_feat_hook)

    batch_orig = make_batch(case, device)
    with torch.no_grad():
        pred_orig = _forward(model, batch_orig)
        orig_val = float(target_fn(pred_orig).item())
    handle.remove()
    del batch_orig, pred_orig
    _gpu_sync_and_gc()

    num_levels, feat_H, feat_W = STAGE_GRID[hook_stage]
    feat_tensor = feat_store["feat"]
    _, L, _ = feat_tensor.shape
    assert L == num_levels * feat_H * feat_W, (
        f"Stage {hook_stage} token count mismatch: got L={L}, expected "
        f"{num_levels}x{feat_H}x{feat_W}={num_levels*feat_H*feat_W}"
    )
    feature_map = extract_feature_map(feat_tensor, num_levels, feat_H, feat_W)
    cluster_kwargs = (
        {"n_clusters": n_clusters, "distance_threshold": None}
        if n_clusters is not None
        else {"n_clusters": None, "distance_threshold": distance_threshold}
    )
    labels, patch_masks, actual_n_clusters = cluster_features(
        feature_map, **cluster_kwargs
    )

    # --- Step 2: rank split ---
    per_rank = actual_n_clusters // world_size
    extra = actual_n_clusters % world_size
    my_n = per_rank + (1 if rank < extra else 0)
    my_start = rank * per_rank + min(rank, extra)
    my_indices = list(range(my_start, my_start + my_n))

    cluster_desc = (
        f"fixed_budget={n_clusters}"
        if n_clusters is not None
        else f"threshold={distance_threshold}"
    )
    print(f"        [Rank {rank}] ViT-CX: {my_n}/{actual_n_clusters} clusters "
          f"at stage {hook_stage} ({feat_H}x{feat_W}, {cluster_desc}, "
          f"smooth_sigma={smooth_sigma})")

    # ViT-CX occlusion semantics: 1 = occlude (replace with baseline).
    # _make_zwd_masked_batch uses 1 = keep → invert.
    def scorer_fn(occlusion_mask_hw):
        keep_mask = 1.0 - occlusion_mask_hw
        batch = _make_zwd_masked_batch(case, device, keep_mask, zwd_baseline_cpu)
        with torch.no_grad():
            pred = _forward(model, batch)
            masked_val = float(target_fn(pred).item())
        del batch, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return orig_val - masked_val

    my_sal, my_wgt = score_clusters(
        scorer_fn=scorer_fn,
        patch_masks=patch_masks,
        feat_H=feat_H, feat_W=feat_W,
        H=INPUT_H, W=INPUT_W,
        cluster_indices=my_indices,
        verbose=True, rank=rank,
    )

    if world_size == 1:
        return aggregate_and_upsample(
            my_sal, my_wgt, H=INPUT_H, W=INPUT_W,
            smooth_sigma=smooth_sigma,
        ).astype(np.float32)

    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, f"_vcx_sal_r{rank}.npy"), my_sal)
    np.save(os.path.join(tmp_dir, f"_vcx_wgt_r{rank}.npy"), my_wgt)
    with open(os.path.join(tmp_dir, f"_vcx_done_r{rank}"), "w") as f:
        f.write("done")

    if rank != 0:
        return np.zeros((INPUT_H, INPUT_W), dtype=np.float32)

    for r in range(1, world_size):
        marker = os.path.join(tmp_dir, f"_vcx_done_r{r}")
        waited = 0
        while not os.path.exists(marker):
            time.sleep(2)
            waited += 2
            if waited > 7200:
                print(f"  WARNING: ViT-CX rank {r} did not finish in time!")
                break

    total_sal = np.zeros((feat_H, feat_W), dtype=np.float64)
    total_wgt = np.zeros((feat_H, feat_W), dtype=np.float64)
    for r in range(world_size):
        total_sal += np.load(os.path.join(tmp_dir, f"_vcx_sal_r{r}.npy"))
        total_wgt += np.load(os.path.join(tmp_dir, f"_vcx_wgt_r{r}.npy"))

    for r in range(world_size):
        for f in (f"_vcx_sal_r{r}.npy", f"_vcx_wgt_r{r}.npy",
                  f"_vcx_done_r{r}"):
            p = os.path.join(tmp_dir, f)
            if os.path.exists(p):
                os.remove(p)

    return aggregate_and_upsample(
        total_sal, total_wgt, H=INPUT_H, W=INPUT_W,
        smooth_sigma=smooth_sigma,
    ).astype(np.float32)


# ===================================================================
# GT cache loader
# ===================================================================
def _try_load_gt_from_cache(
    gt_mask_dir: str,
    case_id: str,
    scale_name: str,
    mode: str,
    masks,
    target_meta: dict,
) -> "GTResult | None":
    """Try to load pre-computed GT from a previous output directory.

    Validates that the cached file matches on: case id (encodes init time),
    scale, number of masks, all mask keys (encodes center locations), and
    target center coordinates. Returns None and prints a warning on any
    mismatch or missing file.
    """
    from searchlight_ground_truth import GTResult

    fname = "ground_truth.json" if mode == "plain" else f"ground_truth_{mode}.json"
    path = os.path.join(gt_mask_dir, "per_case", case_id, scale_name, fname)

    if not os.path.exists(path):
        return None

    with open(path) as fh:
        data = json.load(fh)

    # Case id and scale
    if data.get("case_id") != case_id:
        print(f"    [GT cache] MISMATCH case_id: cached={data.get('case_id')!r} "
              f"vs expected={case_id!r} in {path}")
        return None
    if data.get("scale") != scale_name:
        print(f"    [GT cache] MISMATCH scale: cached={data.get('scale')!r} "
              f"vs expected={scale_name!r} in {path}")
        return None

    # Mask count
    cached_masks = data.get("masks", [])
    if len(cached_masks) != len(masks):
        print(f"    [GT cache] MISMATCH n_masks: cached={len(cached_masks)} "
              f"vs expected={len(masks)} in {path}")
        return None

    # Mask keys (key encodes center_lat, center_lon, scale)
    cached_keys = [m["key"] for m in cached_masks]
    expected_keys = [m.key for m in masks]
    if cached_keys != expected_keys:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(cached_keys, expected_keys)) if a != b),
            None,
        )
        print(f"    [GT cache] MISMATCH mask keys (first diff at index "
              f"{first_diff}) in {path}")
        return None

    # Target location
    cached_tmeta = data.get("target_meta", {})
    if target_meta.get("target_mode") == "box":
        for field in ("center_lat", "center_lon"):
            cached_v = cached_tmeta.get(field)
            expected_v = target_meta.get(field)
            if cached_v is None or expected_v is None or abs(cached_v - expected_v) > 1e-6:
                print(f"    [GT cache] MISMATCH target {field}: "
                      f"cached={cached_v} vs expected={expected_v} in {path}")
                return None
    elif target_meta.get("target_mode") == "point":
        for field in ("lat_idx", "lon_idx"):
            if cached_tmeta.get(field) != target_meta.get(field):
                print(f"    [GT cache] MISMATCH target {field}: "
                      f"cached={cached_tmeta.get(field)} vs "
                      f"expected={target_meta.get(field)} in {path}")
                return None

    G = np.array(data["G"], dtype=np.float64)
    S = np.array(data["S"], dtype=np.float64)
    f_plus = np.array(data["f_plus"], dtype=np.float64)
    f_minus = np.array(data["f_minus"], dtype=np.float64)
    return GTResult(
        mask_keys=[m["key"] for m in cached_masks],
        G=G, S=S, f_plus=f_plus, f_minus=f_minus,
    )


# ===================================================================
# Main
# ===================================================================
def run_benchmark(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    t_start = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"START rank={rank}/{world_size} device={device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.lead_time_hours <= 0 or args.lead_time_hours % 6 != 0:
        raise ValueError(
            f"--lead-time-hours must be a positive multiple of 6, "
            f"got {args.lead_time_hours}"
        )
    rollout_steps = args.lead_time_hours // 6
    print(f"Lead time: +{args.lead_time_hours}h ({rollout_steps} rollout step"
          f"{'s' if rollout_steps != 1 else ''})")
    if args.gt_only:
        print("GT-only mode: XAI methods will be skipped.")
    if rollout_steps > 1 and not args.gt_only:
        _rollout_ready = {
            "saliency", "ig",
            "contrastive_saliency_remote", "contrastive_saliency_global",
            "contrastive_ig_remote", "contrastive_ig_global",
        }
        _not_ready = [m for m in args.methods if m not in _rollout_ready]
        if _not_ready:
            print(f"WARNING: rollout_steps={rollout_steps} is supported for "
                  f"saliency / ig / contrastive_*. The following requested "
                  f"methods still do a 1-step forward and will NOT reflect the "
                  f"requested lead time: {_not_ready}")

    # Cases + target filter
    if args.cases == "auto":
        cases = default_cases()
    else:
        cases = load_cases_from_json(args.cases)
    if args.targets:
        cases = [c for c in cases if c.target in set(args.targets)]
    if args.debug:
        cases = cases[:1]
        args.methods = ["saliency"]
        args.scales = ["local"]

    scales = [SCALES[s] for s in args.scales]

    results_root = os.environ.get(
        "AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")
    )
    output_dir = args.output_dir or os.path.join(
        results_root, "zwd_attribution_benchmark", "searchlight_v1"
    )
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump({
                "methods": args.methods,
                "scales": [s.name for s in scales],
                "target_mode": args.target_mode,
                "cases": [
                    {"target": c.target,
                     "init_time": c.init_time.isoformat(),
                     "role": c.role}
                    for c in cases
                ],
                "ig_steps": args.ig_steps,
                "rise_n_masks": args.rise_n_masks,
                "rise_cells_h": args.rise_cells_h,
                "rise_cells_w": args.rise_cells_w,
                "vit_cx_stage": args.vit_cx_stage,
                "vit_cx_n_clusters": args.vit_cx_n_clusters,
                "vit_cx_dist_thresh": args.vit_cx_dist_thresh,
                "vit_cx_no_smoothing": args.vit_cx_no_smoothing,
                "magnitude": args.magnitude,
                "lead_time_hours": args.lead_time_hours,
                "gt_only": args.gt_only,
                "seed": args.seed,
            }, f, indent=2)

    print(f"Output: {output_dir}")
    print(f"Cases: {[c.case_id for c in cases]}")
    print(f"Methods: {args.methods}")
    print(f"Scales: {[s.name for s in scales]}")
    print(f"Target mode: {args.target_mode}")
    print(f"Magnitude (± sigma_zwd): {args.magnitude}")

    model = setup_model(device)
    tmp_root = os.environ.get(
        "AURORA_XAI_TMP_DIR", os.path.join(_ROOT, ".tmp", "searchlight")
    )
    os.makedirs(tmp_root, exist_ok=True)

    all_rows: list[dict] = []

    for case in cases:
        print(f"\n{'='*60}")
        case_id = _output_case_id(case.case_id, args.target_mode)
        print(f"Case: {case.case_id}")
        if case_id != case.case_id:
            print(f"  output case id: {case_id}")
        case_data = load_case(case.init_time)
        target_fn, target_meta = make_q850_target(
            case_data, case.target, args.target_mode
        )
        print(f"  {_format_target_meta(target_meta)}")

        zwd_actual = case_data.surf_cpu["zwd"]
        lat_vals = case_data.lat_vals
        lon_vals = case_data.lon_vals
        cos_w = cos_lat_weights(lat_vals, INPUT_W)

        def make_batch_with_zwd(zwd_override: torch.Tensor):
            return make_batch(case_data, device, zwd_override=zwd_override)

        def target_fn_float(pred):
            return target_fn(pred)

        # --- Per-scale loop ---
        for scale in scales:
            print(f"\n  Scale: {scale.name} (sigma={scale.sigma_deg}°)")
            target = TARGETS[case.target]
            masks = generate_mask_centers(target, scale)
            print(f"    {len(masks)} total masks "
                  f"({sum(1 for m in masks if m.role=='near')} near / "
                  f"{sum(1 for m in masks if m.role=='remote')} remote)")

            # Determine which GT modes are needed by the requested methods.
            modes_needed: set[str] = set()
            for m_name in args.methods:
                modes_needed.add(_method_gt_mode(m_name))

            target_fns: dict = {}
            if "plain" in modes_needed:
                target_fns["plain"] = target_fn_float
            if "remote" in modes_needed:
                target_fns["remote"] = make_contrast_target_fn(
                    case_data, case.target, "remote", device,
                    target_mode=args.target_mode,
                )
            if "global" in modes_needed:
                target_fns["global"] = make_contrast_target_fn(
                    case_data, case.target, "global", device,
                    target_mode=args.target_mode,
                )

            # --- Ground truth (one pass, multiple targets) ---
            # Try loading from a pre-computed cache directory first, per-mode.
            # Modes found in the cache are reused; missing modes are computed.
            # All ranks perform the same validation so no synchronisation is
            # needed.
            gt_by_mode: dict = {}
            cached_modes: list = []
            missing_modes = [] if args.no_gt else list(modes_needed)
            if rank == 0 and args.no_gt:
                print("    --no-gt: skipping ground truth computation.")
            if args.gt_mask_dir and not args.no_gt:
                missing_modes = []
                for _mode in modes_needed:
                    _r = _try_load_gt_from_cache(
                        args.gt_mask_dir, case_id, scale.name,
                        _mode, masks, target_meta,
                    )
                    if _r is None:
                        missing_modes.append(_mode)
                    else:
                        gt_by_mode[_mode] = _r
                        cached_modes.append(_mode)
                if rank == 0 and cached_modes:
                    print(f"    GT loaded from cache: {args.gt_mask_dir} "
                          f"(modes: {sorted(cached_modes)})")
                if rank == 0 and missing_modes:
                    print(f"    GT will be computed for missing modes: "
                          f"{sorted(missing_modes)}")

            if missing_modes:
                missing_target_fns = {m: target_fns[m] for m in missing_modes}
                gt_tmp = os.path.join(
                    tmp_root, f"gt_{case_id}_{scale.name}_{args.lead_time_hours}h"
                )
                gt_t0 = time.time()
                computed = compute_ground_truth(
                    masks=masks,
                    case_data=case_data,
                    device=device,
                    model=model,
                    target_fns=missing_target_fns,
                    lat_vals=lat_vals,
                    lon_vals=lon_vals,
                    magnitude=args.magnitude,
                    make_batch_with_zwd=make_batch_with_zwd,
                    rank=rank,
                    world_size=world_size,
                    tmp_dir=gt_tmp,
                    rollout_steps=rollout_steps,
                )
                gt_by_mode.update(computed)
                print(f"    GT done in {time.time() - gt_t0:.1f}s "
                      f"(modes: {sorted(missing_target_fns.keys())})")

            # Scale-dependent smoothed baseline (shared across IG/RISE/ViT-CX)
            zwd_baseline_cpu = smoothed_zwd_baseline(zwd_actual, scale.sigma_deg)

            # Precompute mask arrays once on rank 0 for pooling / plotting.
            if rank == 0:
                mask_arrays = [
                    gaussian_mask(m, scale.sigma_deg, lat_vals, lon_vals)
                    for m in masks
                ]
                for mode_name, gt in gt_by_mode.items():
                    save_ground_truth(
                        output_dir=output_dir,
                        case_id=case_id,
                        base_case_id=case.case_id,
                        scale=scale.name,
                        masks=masks,
                        G=gt.G, S=gt.S,
                        f_plus=gt.f_plus, f_minus=gt.f_minus,
                        mode=mode_name,
                        target_mode=args.target_mode,
                        target_meta=target_meta,
                    )

            if args.gt_only:
                print("    --gt-only: skipping all XAI methods for this scale.")
                _gpu_sync_and_gc()
                continue

            # --- Methods ---
            _gradient_methods = {
                "saliency", "smoothgrad", "ig",
                "contrastive_saliency_remote", "contrastive_saliency_global",
                "contrastive_ig_remote", "contrastive_ig_global",
            }
            for method_name in args.methods:
                print(f"\n    Method: {method_name}")
                t_m = time.time()

                # Gradient-based methods: only rank 0 does the backward pass.
                # Other ranks skip — attr is only needed on rank 0 anyway.
                if method_name in _gradient_methods and rank != 0:
                    _gpu_sync_and_gc()
                    continue

                if method_name == "saliency":
                    attr = run_saliency(
                        model, case_data, target_fn, device,
                        rollout_steps=rollout_steps,
                    )
                elif method_name == "smoothgrad":
                    attr = run_smoothgrad(
                        model, case_data, target_fn, device,
                        n_samples=args.smoothgrad_n_samples,
                        noise_sigma_frac=args.smoothgrad_noise_sigma_frac,
                        seed=args.seed,
                    )
                elif method_name == "ig":
                    attr = run_ig(
                        model, case_data, target_fn, device,
                        zwd_baseline_cpu=zwd_baseline_cpu,
                        n_steps=args.ig_steps,
                        rollout_steps=rollout_steps,
                    )
                elif method_name in (
                    "contrastive_saliency_remote",
                    "contrastive_saliency_global",
                ):
                    mode = "remote" if method_name.endswith("_remote") else "global"
                    c_target = make_contrast_target_fn(
                        case_data, case.target, mode, device,
                        target_mode=args.target_mode,
                    )
                    attr = run_saliency(
                        model, case_data, c_target, device,
                        rollout_steps=rollout_steps,
                    )
                elif method_name in (
                    "contrastive_ig_remote",
                    "contrastive_ig_global",
                ):
                    mode = "remote" if method_name.endswith("_remote") else "global"
                    c_target = make_contrast_target_fn(
                        case_data, case.target, mode, device,
                        target_mode=args.target_mode,
                    )
                    attr = run_ig(
                        model, case_data, c_target, device,
                        zwd_baseline_cpu=zwd_baseline_cpu,
                        n_steps=args.ig_steps,
                        rollout_steps=rollout_steps,
                    )
                elif method_name == "rise":
                    attr = run_rise(
                        model, case_data, target_fn, device,
                        zwd_baseline_cpu=zwd_baseline_cpu,
                        n_masks=args.rise_n_masks,
                        cells_h=args.rise_cells_h,
                        cells_w=args.rise_cells_w,
                        seed=args.seed,
                        rank=rank, world_size=world_size,
                        tmp_dir=os.path.join(
                            tmp_root, f"rise_{case_id}_{scale.name}"
                        ),
                    )
                elif method_name == "vit_cx":
                    attr = run_vit_cx(
                        model, case_data, target_fn, device,
                        zwd_baseline_cpu=zwd_baseline_cpu,
                        hook_stage=args.vit_cx_stage,
                        n_clusters=args.vit_cx_n_clusters,
                        distance_threshold=args.vit_cx_dist_thresh,
                        smooth_sigma=(
                            None if args.vit_cx_no_smoothing else (2.0, 4.0)
                        ),
                        rank=rank, world_size=world_size,
                        tmp_dir=os.path.join(
                            tmp_root, f"vcx_{case_id}_{scale.name}"
                        ),
                    )
                else:
                    print(f"      Unknown method: {method_name}, skipping")
                    continue

                runtime_s = time.time() - t_m
                print(f"      completed in {runtime_s:.1f}s")

                if rank != 0:
                    _gpu_sync_and_gc()
                    continue

                # When --no-gt: save only the attribution array, skip scoring.
                if args.no_gt:
                    out_dir = os.path.join(
                        output_dir, "per_case", case_id, scale.name
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    import numpy as _np
                    _np.save(
                        os.path.join(out_dir, f"{method_name}_attr.npy"),
                        attr.astype(_np.float32),
                    )
                    print(f"      saved {method_name}_attr.npy (no-gt mode)")
                    _gpu_sync_and_gc()
                    continue

                gt_mode = _method_gt_mode(method_name)
                gt = gt_by_mode[gt_mode]

                metrics = evaluate(
                    method=method_name,
                    case_id=case_id,
                    scale=scale.name,
                    attr_map=attr,
                    masks=masks,
                    mask_arrays=mask_arrays,
                    G=gt.G,
                    S=gt.S,
                    cos_lat_w=cos_w,
                    signed=(
                        method_name in ("saliency", "smoothgrad", "ig")
                        or method_name.startswith("contrastive_saliency_")
                        or method_name.startswith("contrastive_ig_")
                    ),
                )

                save_method_result(
                    output_dir=output_dir,
                    case_id=case_id,
                    base_case_id=case.case_id,
                    scale=scale.name,
                    method=method_name,
                    attr_map=attr,
                    metrics=metrics,
                    runtime_s=runtime_s,
                    target_mode=args.target_mode,
                    target_meta=target_meta,
                )
                plot_case_scatter(
                    output_dir=output_dir,
                    case_id=case_id,
                    scale=scale.name,
                    method=method_name,
                    pooled_A_mag=metrics.pooled_A_mag,
                    G=gt.G,
                    masks=masks,
                )

                all_rows.append({
                    "method": method_name,
                    "target_mode": args.target_mode,
                    "case_id": case_id,
                    "base_case_id": case.case_id,
                    "scale": scale.name,
                    "rho_mag": metrics.rho_mag,
                    "rho_signed": metrics.rho_signed,
                    "ndcg_at_10": metrics.ndcg_at_10,
                    "top10_recall": metrics.top10_recall,
                    "remote_gap": metrics.remote_gap,
                    "n_masks": metrics.n_masks,
                    "n_remote": metrics.n_remote,
                    "runtime_s": runtime_s,
                })

                print(f"      rho_mag={metrics.rho_mag:.3f} "
                      f"rho_signed={metrics.rho_signed:.3f} "
                      f"ndcg@10={metrics.ndcg_at_10:.3f} "
                      f"top10={metrics.top10_recall:.3f} "
                      f"remote_gap={metrics.remote_gap:.3e}")
                _gpu_sync_and_gc()

        _gpu_sync_and_gc()

    if rank == 0:
        # Merge with any pre-existing per_case/*/_metrics.json from previous
        # runs so a partial re-run (e.g. adding only contrastive variants)
        # still produces a complete leaderboard covering all methods present
        # on disk.
        try:
            merged_rows = collect_rows(output_dir)
        except FileNotFoundError:
            merged_rows = all_rows
        write_leaderboard(output_dir=output_dir, rows=merged_rows)
        print(f"\nLeaderboard written to {output_dir}/leaderboard.csv "
              f"({len(merged_rows)} rows from disk)")

    elapsed = time.time() - t_start
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"END rank={rank} elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    run_benchmark(parse_args())
