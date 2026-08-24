"""
Activation-patching experiment through Aurora's Swin U-Net.

For each (case, contrast, lead, patch_site, patch_region) combination:
  1. Cache hidden states at the site from a source run and a base run
     (both using only rollout step 0).
  2. Re-run the base rollout (N steps for lead > 6h) with the site's
     hidden state replaced by a soft blend of base and source at step 0;
     steps 1..N propagate naturally.
  3. Compute
       recovery = (score_patched - score_base) / (score_source - score_base)

Contrasts
---------
  residual_true_qhat   source=actual ZWD,           base=IWV-qhat ZWD
  plus_hotspot_actual  source=actual + 1σ hotspot,  base=actual
  plus_low_near_actual source=actual + 1σ low_near, base=actual

Patch sites (8)
---------------
  enc_s0_skip, enc_s1_skip, enc_s2_bottleneck
  dec_s0_pre_skip
  dec_s1_pre_skip, dec_s1_post_skip
  dec_s2_pre_concat, dec_s2_post_concat

Patch regions (7)
-----------------
  whole, target_box, mountain_near, upstream,
  hotspot_gaussian, low_near_gaussian, remote_control

Leads: 6h (1 step), 12h (2 steps), 24h (4 steps).
ZWD perturbations affect only t1 (timestep_idx=1).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup  (same as trace_representations.py)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_COND_DIR = os.path.join(_ROOT, "04_zwd_counterfactual_interventions")

for _p in (_HERE, _ROOT, _SEARCHLIGHT_DIR, _COND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_LEVELS = 4  # C dimension in Aurora's Swin token grid (confirmed at runtime)

# Spatial (H_tok, W_tok) per patch site, from get_encoder_specs with patch_res=(4,180,360).
# Encoder stages: s0=(4,180,360), s1=(4,90,180), s2=(4,45,90).
# Decoder reverses: d0 in=(4,45,90) out=(4,90,180), d1 in=(4,90,180) out=(4,180,360), d2=(4,180,360).
PATCH_SITE_SPATIAL: dict[str, tuple[int, int]] = {
    "enc_s0_skip":        (180, 360),  # all_enc_res[0]: (4,180,360)
    "enc_s1_skip":        (90,  180),  # all_enc_res[1]: (4,90,180)
    "enc_s2_bottleneck":  (45,  90),   # all_enc_res[2]: (4,45,90)
    "dec_s0_pre_skip":    (45,  90),   # enters dec layer 0 at all_enc_res[2]
    "dec_s1_pre_skip":    (90,  180),  # enters dec layer 1 at all_enc_res[1]
    "dec_s1_post_skip":   (180, 360),  # after PatchSplitting upsample + enc_s0 skip
    "dec_s2_pre_concat":  (180, 360),  # enters dec layer 2 at all_enc_res[0]
    "dec_s2_post_concat": (180, 360),  # after cat with enc_s0 skip
}

PATCH_SITES = list(PATCH_SITE_SPATIAL.keys())

PATCH_REGIONS = [
    "whole",
    "target_box",
    "mountain_near",
    "upstream",
    "hotspot_gaussian",
    "low_near_gaussian",
    "remote_control",
]

CONTRASTS = [
    "residual_true_qhat",
    "residual_true_zerozwd",
    "plus_hotspot_actual",
    "plus_low_near_actual",
]

LEADS_HOURS = [6, 12, 24]

# Gaussian sigma for saliency-based hotspot / low_near selection (synoptic scale)
_SALIENCY_SIGMA_DEG = 6.0

DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "activation_patch_zwd",
)
DEFAULT_CASES_JSON = os.path.join(
    os.path.dirname(_HERE), "cases_activation_patch_8.json"
)


# ---------------------------------------------------------------------------
# Pure helpers  (no Aurora import — importable on login node)
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-30 else float("nan")


def _downsample_mask(mask_hw: np.ndarray, tok_h: int, tok_w: int) -> np.ndarray:
    """Area-average the full-resolution mask to (tok_h, tok_w) token grid.

    Prefers exact tile-average when strides are integers (avoids scipy zoom
    boundary artefacts).  Falls back to scipy.ndimage.zoom for non-integer
    strides (e.g. 721 → 180).
    """
    full_h, full_w = mask_hw.shape
    if full_h == tok_h and full_w == tok_w:
        return mask_hw.astype(np.float32)

    sh = full_h / tok_h
    sw = full_w / tok_w
    if sh == int(sh) and sw == int(sw):
        # Exact integer strides — tile-average (always correct, no artefacts)
        sh_i, sw_i = int(sh), int(sw)
        return (
            mask_hw[: tok_h * sh_i, : tok_w * sw_i]
            .reshape(tok_h, sh_i, tok_w, sw_i)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )

    try:
        from scipy.ndimage import zoom  # type: ignore[import]
        factor_h = tok_h / full_h
        factor_w = tok_w / full_w
        out = zoom(mask_hw.astype(np.float64), (factor_h, factor_w), order=1)
        return np.clip(out, 0.0, None).astype(np.float32)
    except ImportError:
        pass

    # Last-resort: crop to nearest exact-divisible size and tile-average
    sh_i, sw_i = max(1, full_h // tok_h), max(1, full_w // tok_w)
    return (
        mask_hw[: tok_h * sh_i, : tok_w * sw_i]
        .reshape(tok_h, sh_i, tok_w, sw_i)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def _apply_patch(
    base_hidden: torch.Tensor,
    src_hidden: torch.Tensor,
    mask_Ntok_1: torch.Tensor,
) -> torch.Tensor:
    """Soft blend: base * (1 - mask) + src * mask.

    All three tensors must be on the same device and dtype.
    mask_Ntok_1 shape: (N_tok, 1)  — broadcasts over the D dimension.
    base_hidden / src_hidden shape: (batch, N_tok, D)
    """
    return base_hidden * (1.0 - mask_Ntok_1) + src_hidden * mask_Ntok_1


def _mask_to_token_tensor(
    mask_hw: np.ndarray,
    site: str,
    full_h: int,
    full_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Downsample spatial mask to (N_tok, 1) tensor for the given patch site.

    Replicates the 2-D spatial mask across all LATENT_LEVELS token-depth
    levels so the same spatial region is patched at every pressure level.
    """
    tok_h, tok_w = PATCH_SITE_SPATIAL[site]
    # Crop to the same height the model sees (Aurora crops to patch_size multiple)
    crop_h = (full_h // 4) * 4  # patch_size = 4
    mask_cropped = mask_hw[:crop_h, :]

    mask_tok = _downsample_mask(mask_cropped, tok_h, tok_w)  # (tok_h, tok_w)
    # Replicate across LATENT_LEVELS
    mask_chw = np.broadcast_to(mask_tok[np.newaxis], (LATENT_LEVELS, tok_h, tok_w)).copy()
    mask_flat = mask_chw.reshape(-1, 1)  # (N_tok, 1)
    return torch.tensor(mask_flat, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Hook classes
# ---------------------------------------------------------------------------

class _CaptureHook:
    """Forward-output hook that stores the module's output tensor."""

    def __init__(self) -> None:
        self.captured: torch.Tensor | None = None

    def __call__(self, module: Any, inp: Any, output: Any) -> None:
        # Store CPU fp16 copy to save VRAM
        if isinstance(output, torch.Tensor):
            self.captured = output.detach().cpu().half().contiguous()
        else:
            # Some decoder layers return tuple; take first element
            self.captured = output[0].detach().cpu().half().contiguous()


class _PreCaptureHook:
    """Forward-pre hook that stores the first input argument."""

    def __init__(self) -> None:
        self.captured: torch.Tensor | None = None

    def __call__(self, module: Any, args: tuple) -> None:
        if args:
            self.captured = args[0].detach().cpu().half().contiguous()


class _PatchHook:
    """Forward-output hook: patches output at rollout step 0 only."""

    def __init__(
        self,
        base_act: torch.Tensor,  # (1, N_tok, D) cpu half
        src_act: torch.Tensor,
        mask_Ntok_1: torch.Tensor,  # (N_tok, 1) cpu float
    ) -> None:
        self.base_act = base_act
        self.src_act = src_act
        self.mask_Ntok_1 = mask_Ntok_1
        self._calls = 0

    def __call__(self, module: Any, inp: Any, output: Any) -> Any:
        self._calls += 1
        if self._calls > 1:
            return output  # pass-through for steps 1+
        tgt = output if isinstance(output, torch.Tensor) else output[0]
        base = self.base_act.to(device=tgt.device, dtype=tgt.dtype)
        src = self.src_act.to(device=tgt.device, dtype=tgt.dtype)
        mask = self.mask_Ntok_1.to(device=tgt.device, dtype=tgt.dtype)
        patched = _apply_patch(base, src, mask)
        if isinstance(output, torch.Tensor):
            return patched
        return (patched,) + output[1:]


class _PrePatchHook:
    """Forward-pre hook: patches first input argument at rollout step 0 only."""

    def __init__(
        self,
        base_act: torch.Tensor,
        src_act: torch.Tensor,
        mask_Ntok_1: torch.Tensor,
    ) -> None:
        self.base_act = base_act
        self.src_act = src_act
        self.mask_Ntok_1 = mask_Ntok_1
        self._calls = 0

    def __call__(self, module: Any, args: tuple) -> tuple | None:
        self._calls += 1
        if self._calls > 1 or not args:
            return None
        x = args[0]
        base = self.base_act.to(device=x.device, dtype=x.dtype)
        src = self.src_act.to(device=x.device, dtype=x.dtype)
        mask = self.mask_Ntok_1.to(device=x.device, dtype=x.dtype)
        patched = _apply_patch(base, src, mask)
        return (patched,) + args[1:]


# ---------------------------------------------------------------------------
# Site module lookup
# ---------------------------------------------------------------------------

def _site_module_is_pre(backbone: Any, site: str) -> tuple[Any, bool]:
    """Return (module, is_pre_hook) for the given patch site name."""
    _enc = backbone.encoder_layers
    _dec = backbone.decoder_layers
    mapping = {
        "enc_s0_skip":        (_enc[0], False),
        "enc_s1_skip":        (_enc[1], False),
        "enc_s2_bottleneck":  (_enc[2], False),
        "dec_s0_pre_skip":    (_dec[0], True),
        "dec_s1_pre_skip":    (_dec[1], True),
        "dec_s1_post_skip":   (_dec[1], False),
        "dec_s2_pre_concat":  (_dec[2], True),
        "dec_s2_post_concat": (_dec[2], False),
    }
    if site not in mapping:
        raise ValueError(f"Unknown patch site: {site!r}")
    return mapping[site]


# ---------------------------------------------------------------------------
# Backbone-level capture/patch wrapper
# ---------------------------------------------------------------------------

def _capture_site_value(site: str, store: dict[str, torch.Tensor], value: torch.Tensor) -> None:
    store[site] = value.detach().cpu().half().contiguous()


def _maybe_patch_site(
    site: str,
    value: torch.Tensor,
    patch_site: str | None,
    patch_enabled: bool,
    base_act: torch.Tensor | None,
    src_act: torch.Tensor | None,
    mask_Ntok_1: torch.Tensor | None,
) -> torch.Tensor:
    if not patch_enabled or site != patch_site:
        return value
    if base_act is None or src_act is None or mask_Ntok_1 is None:
        raise RuntimeError(f"Missing patch tensors for site {site!r}")
    base = base_act.to(device=value.device, dtype=value.dtype)
    src = src_act.to(device=value.device, dtype=value.dtype)
    mask = mask_Ntok_1.to(device=value.device, dtype=value.dtype)
    return _apply_patch(base, src, mask)


def _make_backbone_forward_wrapper(
    backbone: Any,
    *,
    capture_store: dict[str, torch.Tensor] | None = None,
    patch_site: str | None = None,
    base_act: torch.Tensor | None = None,
    src_act: torch.Tensor | None = None,
    mask_Ntok_1: torch.Tensor | None = None,
):
    original_forward = backbone.forward
    lead_time_expansion_fn = original_forward.__globals__["lead_time_expansion"]
    patch_state = {"done": False}

    def wrapped_forward(
        x: torch.Tensor,
        lead_time: Any,
        rollout_step: int,
        patch_res: tuple[int, int, int],
        is_global_observation: bool = True,
    ) -> torch.Tensor:
        patch_now = (patch_site is not None) and (rollout_step == 0) and (not patch_state["done"])

        _msg = "Input shape does not match patch size."
        assert x.shape[1] == patch_res[0] * patch_res[1] * patch_res[2], _msg

        _msg = f"Patch height ({patch_res[0]}) must be divisible by ws[0] ({backbone.window_size[0]})"
        assert patch_res[0] % backbone.window_size[0] == 0, _msg

        all_enc_res, padded_outs = backbone.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        c = backbone.time_mlp(lead_time_expansion_fn(lead_times, backbone.embed_dim).to(dtype=x.dtype))

        skips = []
        enc_sites = ["enc_s0_skip", "enc_s1_skip", "enc_s2_bottleneck"]
        dec_pre_sites = ["dec_s0_pre_skip", "dec_s1_pre_skip", "dec_s2_pre_concat"]
        for i, layer in enumerate(backbone.encoder_layers):
            x, x_unscaled = layer(
                x, c, all_enc_res[i], rollout_step=rollout_step, is_global_observation=is_global_observation
            )
            enc_site = enc_sites[i]
            enc_value = x if x_unscaled is None else x_unscaled
            enc_value = _maybe_patch_site(enc_site, enc_value, patch_site, patch_now, base_act, src_act, mask_Ntok_1)
            if x_unscaled is None:
                x = enc_value
            else:
                x_unscaled = enc_value
            if capture_store is not None:
                _capture_site_value(enc_site, capture_store, enc_value)
            skips.append(enc_value)

        for i, layer in enumerate(backbone.decoder_layers):
            index = backbone.num_decoder_layers - i - 1
            dec_pre_site = dec_pre_sites[i]
            x = _maybe_patch_site(dec_pre_site, x, patch_site, patch_now, base_act, src_act, mask_Ntok_1)
            if capture_store is not None:
                _capture_site_value(dec_pre_site, capture_store, x)

            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
                is_global_observation=is_global_observation,
            )

            if 0 < i < backbone.num_decoder_layers - 1:
                x = x + skips[index - 1]
                dec_post_site = "dec_s1_post_skip"
                x = _maybe_patch_site(dec_post_site, x, patch_site, patch_now, base_act, src_act, mask_Ntok_1)
                if capture_store is not None:
                    _capture_site_value(dec_post_site, capture_store, x)
            elif i == backbone.num_decoder_layers - 1:
                x = torch.cat([x, skips[0]], dim=-1)
                dec_post_site = "dec_s2_post_concat"
                x = _maybe_patch_site(dec_post_site, x, patch_site, patch_now, base_act, src_act, mask_Ntok_1)
                if capture_store is not None:
                    _capture_site_value(dec_post_site, capture_store, x)

        if patch_now:
            patch_state["done"] = True
        return x

    return original_forward, wrapped_forward


@contextmanager
def _temporary_backbone_forward(
    backbone: Any,
    *,
    capture_store: dict[str, torch.Tensor] | None = None,
    patch_site: str | None = None,
    base_act: torch.Tensor | None = None,
    src_act: torch.Tensor | None = None,
    mask_Ntok_1: torch.Tensor | None = None,
):
    original_forward, wrapped_forward = _make_backbone_forward_wrapper(
        backbone,
        capture_store=capture_store,
        patch_site=patch_site,
        base_act=base_act,
        src_act=src_act,
        mask_Ntok_1=mask_Ntok_1,
    )
    backbone.forward = wrapped_forward
    try:
        yield
    finally:
        backbone.forward = original_forward


# ---------------------------------------------------------------------------
# Activation caching
# ---------------------------------------------------------------------------

def _cache_all_sites(
    model: Any,
    batch: Any,
    target_fn: Any,
    backbone: Any,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run one forward pass and capture activations at all 8 patch sites.

    Returns (act_dict, score).
    act_dict keys are patch site names; values are cpu-half tensors.
    """
    act_dict: dict[str, torch.Tensor] = {}
    with _temporary_backbone_forward(backbone, capture_store=act_dict):
        with torch.no_grad():
            out = model.forward(batch)
            pred = out[0] if isinstance(out, tuple) else out
            score = float(target_fn(pred).item())

    _gpu_sync_and_gc()

    missing = [site for site in PATCH_SITES if site not in act_dict]
    if missing:
        raise RuntimeError(f"No activation captured at sites: {missing}")
    return act_dict, score


# ---------------------------------------------------------------------------
# Patched rollout
# ---------------------------------------------------------------------------

def _run_patched_rollout(
    model: Any,
    base_batch: Any,
    backbone: Any,
    site: str,
    base_act: torch.Tensor,
    src_act: torch.Tensor,
    mask_Ntok_1: torch.Tensor,
    target_fn: Any,
    lead_steps: int,
) -> float:
    """Run N-step rollout from base_batch, patching site only at step 0.

    Returns the target score from the final rollout step.
    """
    import dataclasses

    with _temporary_backbone_forward(
        backbone,
        patch_site=site,
        base_act=base_act,
        src_act=src_act,
        mask_Ntok_1=mask_Ntok_1,
    ):
        with torch.no_grad():
            if lead_steps == 1:
                batch = _prepare_batch_for_model(model, base_batch)
                out = model.forward(batch)
                pred = out[0] if isinstance(out, tuple) else out
            else:
                batch = _prepare_batch_for_model(model, base_batch)
                pred = None
                for _ in range(lead_steps):
                    out = model.forward(batch)
                    pred = out[0] if isinstance(out, tuple) else out
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
        score = float(target_fn(pred).item())

    _gpu_sync_and_gc()
    return score


def _run_baseline_rollout(
    model: Any,
    batch: Any,
    target_fn: Any,
    lead_steps: int,
) -> float:
    """Run N-step rollout without any patching and return the target score."""
    import dataclasses

    with torch.no_grad():
        b = _prepare_batch_for_model(model, batch)
        if lead_steps == 1:
            out = model.forward(b)
            pred = out[0] if isinstance(out, tuple) else out
        else:
            pred = None
            for _ in range(lead_steps):
                out = model.forward(b)
                pred = out[0] if isinstance(out, tuple) else out
                b = dataclasses.replace(
                    pred,
                    surf_vars={
                        k: torch.cat([b.surf_vars[k][:, 1:], v], dim=1)
                        for k, v in pred.surf_vars.items()
                    },
                    atmos_vars={
                        k: torch.cat([b.atmos_vars[k][:, 1:], v], dim=1)
                        for k, v in pred.atmos_vars.items()
                    },
                )
    score = float(target_fn(pred).item())
    _gpu_sync_and_gc()
    return score


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _prepare_batch_for_model(model: Any, batch: Any) -> Any:
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if model.use_resolution_specific_patch_tokenizers:
        ps = model.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
    else:
        ps = model.patch_size
    return batch.crop(patch_size=ps).to(p.device)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_sync_and_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Saliency-based region selection (hotspot / low_near / remote)
# ---------------------------------------------------------------------------

@dataclass
class _RegionSpec:
    kind: str           # hotspot / low_near / remote
    center_lat: float
    center_lon: float
    pooled_saliency: float


def _select_gaussian_regions(
    model: Any,
    actual_batch: Any,
    case: Any,
    case_obj: Any,
    target_fn: Any,
    device: torch.device,
) -> tuple[_RegionSpec, _RegionSpec, _RegionSpec]:
    """Compute 6h ZWD saliency and select hotspot, low_near, remote regions.

    Reuses the same near-mask grid as the searchlight benchmark.
    Falls back to geometry-based defaults if saliency fails.
    """
    from searchlight_tasks import (
        TARGETS, SCALES, generate_mask_centers, gaussian_mask, cos_lat_weights,
        great_circle_km,
    )
    from searchlight_data import make_batch as _make_batch

    target = TARGETS[case_obj.target]
    scale = SCALES["synoptic"]
    masks_specs = generate_mask_centers(target, scale)
    lat_vals = case.lat_vals
    lon_vals = case.lon_vals

    try:
        from xia_methods.saliency import saliency as _xia_saliency  # type: ignore[import]

        def _batch_fn(requires_grad: bool):
            return _make_batch(
                case, device,
                requires_grad_surf=("zwd",) if requires_grad else (),
            )

        result = _xia_saliency(model, _batch_fn, target_fn, device, surf_var_names=("zwd",))
        grad_zwd = result["grads"].get("zwd")
        if grad_zwd is None:
            raise ValueError("ZWD gradient not found in saliency result.")
        saliency_hw = np.abs(grad_zwd[0, 1])  # t1 slice, (H, W)
    except Exception as exc:
        print(f"  WARNING: saliency failed ({exc}), using uniform saliency.")
        saliency_hw = np.ones((len(lat_vals), len(lon_vals)), dtype=np.float32)

    _gpu_sync_and_gc()

    # Pool saliency per mask
    cos_w = cos_lat_weights(lat_vals, len(lon_vals))
    near_specs = [m for m in masks_specs if m.role == "near"]
    remote_specs = [m for m in masks_specs if m.role == "remote"]

    def _pool(spec) -> float:
        g = gaussian_mask(spec, _SALIENCY_SIGMA_DEG, lat_vals, lon_vals)
        w = g * cos_w
        ws = float(w.sum())
        return float((saliency_hw * w).sum() / ws) if ws > 1e-10 else 0.0

    near_pooled = np.array([_pool(m) for m in near_specs])
    remote_pooled = np.array([_pool(m) for m in remote_specs]) if remote_specs else np.zeros(1)

    if near_pooled.size == 0:
        # Absolute fallback: use target center as hotspot, offset as low_near
        hs = _RegionSpec("hotspot", target.center_lat, target.center_lon, float("nan"))
        ln = _RegionSpec("low_near", target.center_lat + 5.0, target.center_lon, float("nan"))
        rm = _RegionSpec("remote", target.center_lat, (target.center_lon + 180.0) % 360.0, float("nan"))
        return hs, ln, rm

    hotspot_i = int(np.argmax(near_pooled))
    hotspot_spec = near_specs[hotspot_i]
    hotspot_dist = float(great_circle_km(
        target.center_lat, target.center_lon,
        hotspot_spec.center_lat, hotspot_spec.center_lon,
    ))

    others = [(i, m) for i, m in enumerate(near_specs) if i != hotspot_i]
    if others:
        low_cut = float(np.quantile(near_pooled[[i for i, _ in others]], 0.25))
        low_pool_idx = [i for i, m in others if near_pooled[i] <= low_cut]
        if not low_pool_idx:
            low_pool_idx = [i for i, _ in others]
        dists = np.array([
            float(great_circle_km(target.center_lat, target.center_lon,
                                  near_specs[i].center_lat, near_specs[i].center_lon))
            for i in low_pool_idx
        ])
        best_low = low_pool_idx[int(np.argmin(np.abs(dists - hotspot_dist)))]
        low_near_spec = near_specs[best_low]
    else:
        low_near_spec = hotspot_spec  # degenerate; shouldn't happen in practice

    remote_i = int(np.argmin(remote_pooled)) if remote_specs else 0
    remote_spec = remote_specs[remote_i] if remote_specs else hotspot_spec

    hs = _RegionSpec("hotspot", hotspot_spec.center_lat, hotspot_spec.center_lon,
                     float(near_pooled[hotspot_i]))
    ln = _RegionSpec("low_near", low_near_spec.center_lat, low_near_spec.center_lon,
                     float(near_pooled[low_near_spec.mask_id if hasattr(low_near_spec, 'mask_id') else 0]))
    rm = _RegionSpec("remote", remote_spec.center_lat, remote_spec.center_lon,
                     float(remote_pooled[remote_i]) if remote_specs else float("nan"))

    return hs, ln, rm


# ---------------------------------------------------------------------------
# Patch region mask construction
# ---------------------------------------------------------------------------

def _build_region_masks(
    case: Any,
    case_obj: Any,
    hotspot: _RegionSpec,
    low_near: _RegionSpec,
    remote: _RegionSpec,
) -> dict[str, np.ndarray]:
    """Build (H, W) float32 masks for each patch region.

    Reuses Stage-E mask logic from conditional_data.build_localized_residual_masks.
    """
    from conditional_data import build_localized_residual_masks  # type: ignore[import]
    from searchlight_tasks import MaskSpec, gaussian_mask  # type: ignore[import]

    stage_e = build_localized_residual_masks(case, case_obj)
    lat_vals = case.lat_vals
    lon_vals = case.lon_vals
    H, W = len(lat_vals), len(lon_vals)

    def _to_f32(t: Any) -> np.ndarray:
        if hasattr(t, "numpy"):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    # Gaussian masks for saliency-selected regions
    def _gauss(clat: float, clon: float) -> np.ndarray:
        spec = MaskSpec(
            scale="synoptic", role="near",
            center_lat=clat, center_lon=clon, mask_id=0,
        )
        return gaussian_mask(spec, _SALIENCY_SIGMA_DEG, lat_vals, lon_vals)

    return {
        "whole":            np.ones((H, W), dtype=np.float32),
        "target_box":       _to_f32(stage_e["target_box"]),
        "mountain_near":    _to_f32(stage_e["mountain_near"]),
        "upstream":         _to_f32(stage_e["upstream"]),
        "hotspot_gaussian": _gauss(hotspot.center_lat, hotspot.center_lon),
        "low_near_gaussian": _gauss(low_near.center_lat, low_near.center_lon),
        "remote_control":   _to_f32(stage_e["remote_control"]),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        open(path, "w").close()
        return
    # Union of all field names (in insertion order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_recovery_heatmap(
    path: str,
    site_names: list[str],
    region_names: list[str],
    recovery_matrix: np.ndarray,  # (n_sites, n_regions)
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(region_names) * 1.3),
                                    max(4, len(site_names) * 0.8)))
    im = ax.imshow(recovery_matrix, vmin=-0.5, vmax=1.5, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(region_names)))
    ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(site_names)))
    ax.set_yticklabels(site_names, fontsize=8)
    for i in range(len(site_names)):
        for j in range(len(region_names)):
            v = recovery_matrix[i, j]
            txt = f"{v:.2f}" if not math.isnan(v) else "–"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.8 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="recovery")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def _plot_pre_post_skip(
    path: str,
    stage: str,
    pre_data: dict[str, np.ndarray],
    post_data: dict[str, np.ndarray],
    region_names: list[str],
) -> None:
    """Bar chart comparing pre-skip vs post-skip recovery for a decoder stage."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})")
        return

    pre_key = f"{stage}_pre_skip" if "s1" in stage else f"{stage}_pre_concat"
    post_key = f"{stage}_post_skip" if "s1" in stage else f"{stage}_post_concat"

    xs = np.arange(len(region_names))
    fig, ax = plt.subplots(figsize=(max(7, len(region_names) * 1.1), 4))
    pre_vals = np.array([pre_data.get(r, float("nan")) for r in region_names])
    post_vals = np.array([post_data.get(r, float("nan")) for r in region_names])
    ax.bar(xs - 0.2, pre_vals, 0.35, label=pre_key, alpha=0.85)
    ax.bar(xs + 0.2, post_vals, 0.35, label=post_key, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("recovery")
    ax.set_title(f"Pre vs post skip recovery — {stage}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color="gray", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Activation-patching experiment through Aurora's Swin U-Net."
    )
    p.add_argument("--cases", type=str, default=DEFAULT_CASES_JSON,
                   help="Path to cases JSON file.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--contrasts", nargs="+", default=list(CONTRASTS),
                   choices=CONTRASTS)
    p.add_argument("--sites", nargs="+", default=list(PATCH_SITES),
                   choices=PATCH_SITES)
    p.add_argument("--regions", nargs="+", default=list(PATCH_REGIONS),
                   choices=PATCH_REGIONS)
    p.add_argument("--leads", nargs="+", type=int, default=list(LEADS_HOURS),
                   choices=LEADS_HOURS)
    p.add_argument("--magnitude", type=float, default=1.0,
                   help="ZWD perturbation amplitude in units of sigma_zwd.")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: one case, 6h only, two sites, residual contrast.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from searchlight_benchmark import setup_model, make_q850_box_target  # noqa: E402
    from searchlight_data import load_case, make_batch  # noqa: E402
    from searchlight_tasks import load_cases_from_json  # noqa: E402
    from searchlight_ground_truth import perturb_zwd  # noqa: E402
    from conditional_data import make_qhat_zwd  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_cases_from_json(args.cases)
    if not cases:
        raise ValueError("No cases found in cases JSON.")

    sites = args.sites
    regions = args.regions
    contrasts = args.contrasts
    leads = args.leads
    magnitude = args.magnitude

    if args.smoke:
        print("** SMOKE TEST mode **")
        cases = cases[:1]
        leads = [6]
        contrasts = ["residual_true_qhat", "residual_true_zerozwd"]
        sites = ["dec_s2_post_concat", "enc_s2_bottleneck", "enc_s0_skip"]
        regions = ["whole", "hotspot_gaussian", "remote_control"]

    _ensure_dir(args.output_dir)
    _write_json(os.path.join(args.output_dir, "config.json"), {
        "cases_file": args.cases,
        "output_dir": args.output_dir,
        "contrasts": contrasts,
        "sites": sites,
        "regions": regions,
        "leads": leads,
        "magnitude": magnitude,
        "smoke": args.smoke,
        "patch_site_spatial": PATCH_SITE_SPATIAL,
        "cases": [
            {"target": c.target, "init_time": c.init_time.isoformat(), "role": c.role}
            for c in cases
        ],
    })

    model = setup_model(device)
    backbone = model.backbone

    all_score_rows: list[dict[str, Any]] = []

    for case_obj in cases:
        cid = case_obj.case_id
        print(f"\n=== Case {cid} ===")
        case = load_case(case_obj.init_time)
        target_fn, _box = make_q850_box_target(case, case_obj.target)
        actual_batch = make_batch(case, device)
        qhat_zwd = make_qhat_zwd(case)

        full_H = len(case.lat_vals)
        full_W = len(case.lon_vals)

        # Select hotspot / low_near / remote via 6h saliency (once per case)
        print("  selecting hotspot / low_near via saliency ...")
        hotspot, low_near, remote = _select_gaussian_regions(
            model, actual_batch, case, case_obj, target_fn, device
        )
        print(f"  hotspot  lat={hotspot.center_lat:.2f} lon={hotspot.center_lon:.2f} "
              f"sal={hotspot.pooled_saliency:.4f}")
        print(f"  low_near lat={low_near.center_lat:.2f} lon={low_near.center_lon:.2f} "
              f"sal={low_near.pooled_saliency:.4f}")

        # Build all region masks at full resolution
        region_masks_hw = _build_region_masks(
            case, case_obj, hotspot, low_near, remote
        )
        # Restrict to requested regions
        region_masks_hw = {r: region_masks_hw[r] for r in regions if r in region_masks_hw}

        # Contrast definitions: (src_zwd, base_zwd)  (None = actual ZWD)
        hotspot_mask_hw = region_masks_hw.get("hotspot_gaussian",
                          np.ones((full_H, full_W), dtype=np.float32))
        low_near_mask_hw = region_masks_hw.get("low_near_gaussian",
                           np.ones((full_H, full_W), dtype=np.float32))

        plus_hotspot_zwd = perturb_zwd(
            zwd_actual_1_2_H_W=case.surf_cpu["zwd"],
            mask_H_W=hotspot_mask_hw,
            sign=+1.0, magnitude=magnitude,
            zwd_loc=case.zwd_loc, zwd_scale=case.zwd_scale,
            timestep_idx=1,
        )
        plus_low_near_zwd = perturb_zwd(
            zwd_actual_1_2_H_W=case.surf_cpu["zwd"],
            mask_H_W=low_near_mask_hw,
            sign=+1.0, magnitude=magnitude,
            zwd_loc=case.zwd_loc, zwd_scale=case.zwd_scale,
            timestep_idx=1,
        )

        # Spatially uniform ZWD at the per-case mean — no spatial ZWD structure at all.
        zero_zwd = torch.full_like(case.surf_cpu["zwd"], case.zwd_loc)

        contrast_batches: dict[str, tuple[Any, Any]] = {
            "residual_true_qhat":   (make_batch(case, device),
                                     make_batch(case, device, zwd_override=qhat_zwd)),
            "residual_true_zerozwd": (make_batch(case, device),
                                      make_batch(case, device, zwd_override=zero_zwd)),
            "plus_hotspot_actual":  (make_batch(case, device, zwd_override=plus_hotspot_zwd),
                                     make_batch(case, device)),
            "plus_low_near_actual": (make_batch(case, device, zwd_override=plus_low_near_zwd),
                                     make_batch(case, device)),
        }

        for contrast in contrasts:
            if contrast not in contrast_batches:
                continue
            src_batch, base_batch_c = contrast_batches[contrast]

            print(f"\n  --- Contrast: {contrast} ---")

            # Cache step-0 activations for source and base
            print("    caching source activations ...")
            src_acts, _ = _cache_all_sites(model, src_batch, target_fn, backbone)
            print("    caching base activations ...")
            base_acts, _ = _cache_all_sites(model, base_batch_c, target_fn, backbone)

            for lead_h in leads:
                lead_steps = lead_h // 6
                print(f"    lead={lead_h}h (steps={lead_steps})")

                # Unpatched scores
                score_base = _run_baseline_rollout(model, base_batch_c, target_fn, lead_steps)
                score_source = _run_baseline_rollout(model, src_batch, target_fn, lead_steps)
                delta_sb = score_source - score_base
                print(f"      base={score_base:.5f}  source={score_source:.5f}  "
                      f"delta={delta_sb:.5f}")

                for site in sites:
                    tok_h, tok_w = PATCH_SITE_SPATIAL[site]
                    base_act = base_acts[site]
                    src_act = src_acts[site]

                    for region_name, mask_hw in region_masks_hw.items():
                        mask_tok = _mask_to_token_tensor(
                            mask_hw, site, full_H, full_W, device=torch.device("cpu"), dtype=torch.float32
                        )

                        score_patched = _run_patched_rollout(
                            model=model,
                            base_batch=base_batch_c,
                            backbone=backbone,
                            site=site,
                            base_act=base_act,
                            src_act=src_act,
                            mask_Ntok_1=mask_tok,
                            target_fn=target_fn,
                            lead_steps=lead_steps,
                        )

                        recovery = _safe_div(score_patched - score_base, delta_sb)
                        delta_patched_minus_base = score_patched - score_base

                        row: dict[str, Any] = {
                            "case_id": cid,
                            "target": case_obj.target,
                            "role": case_obj.role,
                            "contrast": contrast,
                            "lead_h": lead_h,
                            "patch_site": site,
                            "patch_region": region_name,
                            "score_base": score_base,
                            "score_source": score_source,
                            "score_patched": score_patched,
                            "delta_source_minus_base": delta_sb,
                            "delta_patched_minus_base": delta_patched_minus_base,
                            "recovery": recovery,
                            "hotspot_lat": hotspot.center_lat,
                            "hotspot_lon": hotspot.center_lon,
                            "low_near_lat": low_near.center_lat,
                            "low_near_lon": low_near.center_lon,
                        }
                        all_score_rows.append(row)

                        print(f"      [{site}][{region_name}] "
                              f"patched={score_patched:.5f} recovery={recovery:.3f}")

        _gpu_sync_and_gc()

    # --- Write main output CSVs ---
    _write_csv(os.path.join(args.output_dir, "activation_patch_scores.csv"), all_score_rows)

    # Site summary: mean recovery per (contrast, lead, site) across cases and regions
    _write_summary_csvs(args.output_dir, all_score_rows)

    # Plots
    if not args.skip_plots:
        _write_plots(args.output_dir, all_score_rows, contrasts, leads, sites, regions)

    print(f"\nDone. Results in: {args.output_dir}")


def _write_summary_csvs(output_dir: str, rows: list[dict[str, Any]]) -> None:
    from collections import defaultdict

    # Site summary
    site_agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["contrast"], r["lead_h"], r["patch_site"])
        v = r["recovery"]
        if not math.isnan(v):
            site_agg[key].append(v)

    site_rows = []
    for (contrast, lead_h, site), vals in sorted(site_agg.items()):
        site_rows.append({
            "contrast": contrast,
            "lead_h": lead_h,
            "patch_site": site,
            "n_obs": len(vals),
            "mean_recovery": float(np.mean(vals)),
            "median_recovery": float(np.median(vals)),
            "std_recovery": float(np.std(vals)),
        })
    _write_csv(os.path.join(output_dir, "activation_patch_site_summary.csv"), site_rows)

    # Region summary
    region_agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["contrast"], r["lead_h"], r["patch_region"])
        v = r["recovery"]
        if not math.isnan(v):
            region_agg[key].append(v)

    region_rows = []
    for (contrast, lead_h, region), vals in sorted(region_agg.items()):
        region_rows.append({
            "contrast": contrast,
            "lead_h": lead_h,
            "patch_region": region,
            "n_obs": len(vals),
            "mean_recovery": float(np.mean(vals)),
            "median_recovery": float(np.median(vals)),
            "std_recovery": float(np.std(vals)),
        })
    _write_csv(os.path.join(output_dir, "activation_patch_region_summary.csv"), region_rows)


def _write_plots(
    output_dir: str,
    rows: list[dict[str, Any]],
    contrasts: list[str],
    leads: list[int],
    sites: list[str],
    regions: list[str],
) -> None:
    from collections import defaultdict

    # Heatmaps: recovery by site × region, for each (contrast, lead)
    for contrast in contrasts:
        for lead_h in leads:
            subset = [r for r in rows if r["contrast"] == contrast and r["lead_h"] == lead_h]
            if not subset:
                continue

            # Mean recovery across cases
            site_region_vals: dict[tuple, list[float]] = defaultdict(list)
            for r in subset:
                v = r["recovery"]
                if not math.isnan(v):
                    site_region_vals[(r["patch_site"], r["patch_region"])].append(v)

            matrix = np.full((len(sites), len(regions)), float("nan"))
            for i, site in enumerate(sites):
                for j, region in enumerate(regions):
                    vals = site_region_vals.get((site, region), [])
                    if vals:
                        matrix[i, j] = float(np.mean(vals))

            fname = f"heatmap_{contrast}_lead{lead_h}h.png"
            _plot_recovery_heatmap(
                path=os.path.join(output_dir, fname),
                site_names=sites,
                region_names=regions,
                recovery_matrix=matrix,
                title=f"Recovery heatmap — {contrast} — lead={lead_h}h",
            )

    # Pre/post skip comparison for dec_s1 and dec_s2
    for stage, pre_key, post_key in [
        ("dec_s1", "dec_s1_pre_skip", "dec_s1_post_skip"),
        ("dec_s2", "dec_s2_pre_concat", "dec_s2_post_concat"),
    ]:
        if pre_key not in sites or post_key not in sites:
            continue
        for contrast in contrasts:
            for lead_h in leads:
                subset = [
                    r for r in rows
                    if r["contrast"] == contrast and r["lead_h"] == lead_h
                    and r["patch_site"] in (pre_key, post_key)
                ]
                if not subset:
                    continue
                pre_data: dict[str, list[float]] = defaultdict(list)
                post_data: dict[str, list[float]] = defaultdict(list)
                for r in subset:
                    v = r["recovery"]
                    if math.isnan(v):
                        continue
                    if r["patch_site"] == pre_key:
                        pre_data[r["patch_region"]].append(v)
                    else:
                        post_data[r["patch_region"]].append(v)
                _plot_pre_post_skip(
                    path=os.path.join(output_dir,
                                      f"preskip_{stage}_{contrast}_lead{lead_h}h.png"),
                    stage=stage,
                    pre_data={k: float(np.mean(v)) for k, v in pre_data.items()},
                    post_data={k: float(np.mean(v)) for k, v in post_data.items()},
                    region_names=regions,
                )


if __name__ == "__main__":
    main()
