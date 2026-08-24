"""Shared constants and low-level helpers for the local compute layer.

This module holds the pieces used by more than one of ``compute`` (single-frame
+ batch method runners), ``rollout`` (autoregressive rollout) and
``overlay_compute`` (raw ERA5 overlays).  Keeping them here lets those three
siblings import from one place without forming an import cycle.
"""

from __future__ import annotations

import gc
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

# ``torch`` is imported lazily inside the helpers that need it, so that modules
# importing only the torch-free helpers (e.g. ``overlay_compute``, which uses
# ``_parse_init_time``) keep working in a torch-free ``[server]`` install.

from .data import (
    ATMOS_VARS,
    AURORA_LEVELS,
    SFC_LEVEL_KEY,
    SURF_VARS,
    CaseData,
    atmos_level_key,
    atmos_level_label,
    make_batch,
)


# ── Constants ─────────────────────────────────────────────────────────────────

RISE_N_MASKS_DEFAULT = 1200
RISE_CELLS_H = 400
RISE_CELLS_W = 800
RISE_P = 0.5
VIT_CX_STAGE_DEFAULT = 1
VIT_CX_N_CLUSTERS = 4096  # fixed cluster budget = #occlusion forwards per variable
IG_BASELINE_SIGMA_DEG = 2.5   # Gaussian sigma for the IG smoothed baseline
INPUT_H, INPUT_W = 721, 1440
STAGE_GRID = {0: (4, 180, 360), 1: (4, 90, 180), 2: (4, 45, 90)}

# RISE and ViT-CX perturb a full (H, W) spatial mask that is applied to every
# pressure level at once, so their attribution is identical across the whole
# vertical column.  Rather than emit 13 redundant copies (one per level), they
# store the map once under this single layer.  The viewer requires a valid
# "sfc"/"z-{N}" level id and replicates a single visible level through the whole
# vertical column when rendering, so z-0 reproduces the previous all-levels look
# while collapsing the level selector to one entry.
COLLAPSED_LEVEL_KEY = atmos_level_key(AURORA_LEVELS[0])  # "z-0"
COLLAPSED_LAYER_LABELS = {
    "rise": "RISE attribution",
    "vit_cx": "ViT-CX attribution",
}


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _parse_init_time(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp into a naive UTC datetime."""
    ts = timestamp.rstrip("Z").replace("+00:00", "")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        raise ValueError(
            f"Cannot parse timestamp {timestamp!r}.  "
            "Expected ISO-8601, e.g. '2024-03-20T00:00:00Z'."
        )


def _format_timestamp(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


# ── Attribution helpers ───────────────────────────────────────────────────────

def _split_atmos_levels(
    grad_1_2_L_H_W: np.ndarray,
    pressure_levels: tuple,
    keep_levels: Optional[set[int]] = None,
) -> dict[str, np.ndarray]:
    """Slice an (1, 2, L, H, W) atmos gradient array into per-level dicts.

    Takes the t1 (second) timestep, which is the one perturbed by ground-truth.
    Returns ``{"z-{N}": (H, W) float32, ...}`` (vertical-order keys, see
    ``data.atmos_level_key``).  When ``keep_levels`` is given, only those
    pressure levels (hPa) are emitted; ``None`` emits every level.
    """
    result: dict[str, np.ndarray] = {}
    for l_idx, hpa in enumerate(pressure_levels):
        if keep_levels is not None and hpa not in keep_levels:
            continue
        result[atmos_level_key(hpa)] = grad_1_2_L_H_W[0, 1, l_idx].astype(np.float32)
    return result


def _detect_diverging(attributions: dict[str, dict[str, np.ndarray]]) -> bool:
    """Return True if attribution maps contain significant negative values."""
    parts = []
    for levels in attributions.values():
        for arr in levels.values():
            parts.append(arr.ravel())
    if not parts:
        return False
    flat = np.concatenate(parts)
    max_abs = float(np.abs(flat).max()) or 1.0
    neg_frac = float((flat < -0.005 * max_abs).mean())
    return neg_frac > 0.05


def _smooth_tensor(t: torch.Tensor, sigma_deg: float) -> torch.Tensor:
    """Apply per-spatial-dimension Gaussian smoothing to a CPU tensor.

    Sigma is specified in degrees of latitude; the grid is 0.25°/cell.
    Latitude uses ``mode='reflect'``, longitude uses ``mode='wrap'``
    (periodic boundary) to avoid edge artefacts.
    """
    import torch

    sigma_cells = sigma_deg / 0.25
    t_np = t.cpu().numpy()
    shape = t_np.shape
    smoothed = np.zeros_like(t_np)

    if t_np.ndim == 4:  # (1, 2, H, W) — surface
        for b in range(shape[0]):
            for ti in range(shape[1]):
                smoothed[b, ti] = gaussian_filter(
                    t_np[b, ti],
                    sigma=[sigma_cells, sigma_cells],
                    mode=["reflect", "wrap"],
                )
    elif t_np.ndim == 5:  # (1, 2, L, H, W) — atmos
        for b in range(shape[0]):
            for ti in range(shape[1]):
                for lev in range(shape[2]):
                    smoothed[b, ti, lev] = gaussian_filter(
                        t_np[b, ti, lev],
                        sigma=[sigma_cells, sigma_cells],
                        mode=["reflect", "wrap"],
                    )
    else:
        raise ValueError(f"Unexpected tensor shape for smoothing: {shape}")

    return torch.tensor(smoothed, dtype=t.dtype)


def _gpu_sync_and_gc() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Perturbation helpers ──────────────────────────────────────────────────────

def _make_masked_batch(
    case: CaseData,
    device: str,
    mask_hw: np.ndarray,
    var_name: str,
    baseline_cpu: torch.Tensor,
    var_type: str,
):
    """Build a Batch where var_name_t1 is partially replaced by the baseline.

    mask_hw semantics: 1 = keep original, 0 = replace with baseline.
    Only the t1 timestep is blended; t0 is left at the actual value.

    Shared by the perturbation-based runners (RISE and ViT-CX).
    """
    import torch

    m = torch.from_numpy(mask_hw.astype(np.float32))
    if var_type == "surf":
        actual = case.surf_cpu[var_name]       # (1, 2, H, W)
        bl = baseline_cpu                       # (1, 2, H, W)
        override = actual.clone()
        override[0, 1] = actual[0, 1] * m + bl[0, 1] * (1.0 - m)
        return make_batch(case, device, surf_overrides={var_name: override})
    elif var_type == "atmos":
        actual = case.atmos_cpu[var_name]      # (1, 2, L, H, W)
        bl = baseline_cpu                       # (1, 2, L, H, W)
        override = actual.clone()
        for lev in range(actual.shape[2]):
            override[0, 1, lev] = actual[0, 1, lev] * m + bl[0, 1, lev] * (1.0 - m)
        return make_batch(case, device, atmos_overrides={var_name: override})
    else:
        raise ValueError(f"Unknown var_type {var_type!r}; expected 'surf' or 'atmos'")


# ── Progress / labelling helpers ──────────────────────────────────────────────

def _estimate_total_units(
    method: str,
    *,
    n_frames: int,
    n_vars: int,
    options: dict,
) -> Optional[int]:
    n_frames = max(1, int(n_frames))
    n_vars = max(1, int(n_vars))
    if method == "saliency":
        return n_frames
    if method == "ig":
        return n_frames * int(options.get("n_steps", 32))
    if method == "rise":
        return n_frames * n_vars * int(options.get("n_masks", RISE_N_MASKS_DEFAULT))
    if method == "vit_cx":
        return None
    return None


def _split_input_vars(input_vars: list[str]) -> tuple[list[str], list[str]]:
    atmos_vars = [v for v in input_vars if v in ATMOS_VARS]
    surf_vars = [v for v in input_vars if v in SURF_VARS]
    unknown = [v for v in input_vars if v not in ATMOS_VARS and v not in SURF_VARS]
    if unknown:
        raise ValueError(
            f"Unknown input variable(s): {unknown}. "
            f"ATMOS_VARS={list(ATMOS_VARS)}, SURF_VARS={list(SURF_VARS)}"
        )
    if not atmos_vars and not surf_vars:
        raise ValueError("input_vars is empty - nothing to attribute.")
    return atmos_vars, surf_vars


def _layer_labels_for(
    case: CaseData,
    attributions: dict[str, dict[str, np.ndarray]],
    method: Optional[str] = None,
) -> dict[str, str]:
    present_levels = {lvl for levels in attributions.values() for lvl in levels}
    candidate_labels = {SFC_LEVEL_KEY: "Surface"}
    for hpa in case.pressure_levels:
        candidate_labels[atmos_level_key(hpa)] = atmos_level_label(hpa)
    # RISE / ViT-CX collapse their column-uniform map onto COLLAPSED_LEVEL_KEY;
    # give that layer a method-specific name instead of the pressure-level label
    # it would otherwise inherit (e.g. "1000 hPa").
    collapsed_label = COLLAPSED_LAYER_LABELS.get(method)
    if collapsed_label is not None:
        candidate_labels[COLLAPSED_LEVEL_KEY] = collapsed_label
    return {k: v for k, v in candidate_labels.items() if k in present_levels}
