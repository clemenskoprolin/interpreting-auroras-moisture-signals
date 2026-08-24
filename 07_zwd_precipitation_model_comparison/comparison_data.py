"""
comparison_data.py — Data loading for 07_zwd_precipitation_model_comparison.

Reuses searchlight_data.load_case() for ERA5 base data and adds helpers for:
  - Loading MSWEP precipitation at t0/t1 via MSWEPReader
  - Building Aurora Batch objects for precip models (with/without ZWD)
  - Building Aurora Metadata for precip models
  - Small file-system helpers (_ensure_dir, _write_csv, _append_csv, _write_json)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path wiring — make searchlight_data importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_P_CORR_DIR = os.path.join(_ROOT, "06_precipitation_moisture_relationships")

for _p in (_SEARCHLIGHT_DIR, _P_CORR_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from searchlight_data import (  # noqa: E402
    CaseData, ATMOS_VARS,
    load_case,
)
from comparison_config import (  # noqa: E402
    ModelSpec, MSWEP_STORE_PATH, PRECIP_VAR,
)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write rows (list of dicts) to a CSV, overwriting any existing file."""
    if not rows:
        return
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: str, rows: list[dict]) -> None:
    """Append rows to a CSV; write header if file doesn't exist yet."""
    if not rows:
        return
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, data) -> None:
    """Write data as pretty-printed JSON."""
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# MSWEP precipitation loader
# ---------------------------------------------------------------------------

def load_precip_for_case(
    init_time: datetime,
    mswep_reader,
    *,
    lat_indices: Optional[np.ndarray] = None,
    lon_indices: Optional[np.ndarray] = None,
) -> dict[str, Optional[np.ndarray]]:
    """Load MSWEP precipitation at t0 (init_time - 6h) and t1 (init_time).

    Args:
        init_time: The init time (t1), a datetime object.
        mswep_reader: An MSWEPReader instance (from 06_precipitation_moisture_relationships/common.py).
        lat_indices: Optional subset of latitude indices to load.
        lon_indices: Optional subset of longitude indices to load.

    Returns:
        dict with keys "t0" and "t1", each an (H, W) float32 array or None
        if the timestamp is missing from the store.
    """
    t1 = pd.Timestamp(init_time)
    t0 = pd.Timestamp(init_time - timedelta(hours=6))

    arr_t0 = mswep_reader.read_timestamp(t0, lat_indices=lat_indices, lon_indices=lon_indices)
    arr_t1 = mswep_reader.read_timestamp(t1, lat_indices=lat_indices, lon_indices=lon_indices)

    return {"t0": arr_t0, "t1": arr_t1}


# ---------------------------------------------------------------------------
# Batch / Metadata builders for precip models
# ---------------------------------------------------------------------------

def _cpu_tensor(arr) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=torch.float32)


def build_precip_metadata(
    case: CaseData,
    model_spec: ModelSpec,
    device,
) -> "Metadata":
    """Build Aurora Metadata for a precip model, placed on device.

    This is analogous to searchlight_data.build_metadata but uses the
    model_spec to determine the dataset_name and lead_time.
    """
    from aurora import Metadata

    lats = torch.tensor(case.lat_vals, dtype=torch.float32, device=device)
    lons = torch.tensor(case.lon_vals, dtype=torch.float32, device=device)

    loc_tensors = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in case.locations.items()
    }
    scale_tensors = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in case.scales.items()
    }

    return Metadata(
        dataset_name=model_spec.dataset_name,
        lat=lats,
        lon=lons,
        time=(case.init_time,),
        atmos_levels=case.pressure_levels,
        locations=loc_tensors,
        scales=scale_tensors,
        grid_resolution=0.25,
        is_global_observation=True,
        atmos_levels_output=case.pressure_levels,
        lead_time_seconds=timedelta(hours=6).total_seconds(),
    )


def build_precip_batch(
    case: CaseData,
    model_spec: ModelSpec,
    device,
    *,
    precip_override: Optional[torch.Tensor] = None,
    zwd_override: Optional[torch.Tensor] = None,
    atmos_override: Optional[dict[str, torch.Tensor]] = None,
    requires_grad_surf: tuple[str, ...] = (),
    requires_grad_atmos: tuple[str, ...] = (),
) -> "Batch":
    """Build an Aurora Batch for the given precip model spec.

    The model_spec.surf_vars list determines which surface variables are
    included; the function handles both precip_zwd (with ZWD) and
    precip_only (without ZWD) variants.

    For ZWD: uses case.surf_cpu["zwd"], or zwd_override if provided.
    For precipitation: uses case.surf_cpu[PRECIP_VAR] if available, or
        precip_override. If neither exists, fills with zeros.

    Args:
        case: CaseData loaded by searchlight_data.load_case().
        model_spec: ModelSpec describing the target model's surf_vars.
        device: torch device for the batch.
        precip_override: (1, 2, H, W) CPU float32 tensor for precipitation.
        zwd_override: (1, 2, H, W) CPU float32 tensor for ZWD.
        atmos_override: var -> (1, 2, L, H, W) CPU float32 tensor, replacing
            that atmospheric variable (e.g. a q field perturbed at 850 hPa).
        requires_grad_surf: surface vars for which requires_grad=True.
        requires_grad_atmos: atmos vars for which requires_grad=True.
    """
    from aurora import Batch

    H = case.lat_vals.shape[0]
    W = case.lon_vals.shape[0]

    surf_dev: dict[str, torch.Tensor] = {}
    for k in model_spec.surf_vars:
        if k == "zwd":
            if zwd_override is not None:
                v = zwd_override.clone().to(device)
            elif "zwd" in case.surf_cpu:
                v = case.surf_cpu["zwd"].clone().to(device)
            else:
                # ZWD not available — fill with zeros (should not happen in normal use)
                v = torch.zeros(1, 2, H, W, dtype=torch.float32, device=device)
        elif k == PRECIP_VAR:
            if precip_override is not None:
                v = precip_override.clone().to(device)
            elif PRECIP_VAR in case.surf_cpu:
                v = case.surf_cpu[PRECIP_VAR].clone().to(device)
            else:
                # Precipitation not pre-loaded in case — fill with zeros
                # Caller should provide precip_override when using precip models
                v = torch.zeros(1, 2, H, W, dtype=torch.float32, device=device)
        else:
            v = case.surf_cpu[k].clone().to(device)

        if k in requires_grad_surf:
            v.requires_grad_(True)
        surf_dev[k] = v

    atmos_dev: dict[str, torch.Tensor] = {}
    for k in ATMOS_VARS:
        if atmos_override is not None and k in atmos_override:
            v = atmos_override[k].clone().to(device)
        else:
            v = case.atmos_cpu[k].clone().to(device)
        if k in requires_grad_atmos:
            v.requires_grad_(True)
        atmos_dev[k] = v

    return Batch(
        surf_vars=surf_dev,
        static_vars={k: v.to(device) for k, v in case.static_cpu.items()},
        atmos_vars=atmos_dev,
        metadata=build_precip_metadata(case, model_spec, device),
    )


def inject_precip_into_case(
    case: CaseData,
    precip_t0: np.ndarray,
    precip_t1: np.ndarray,
) -> CaseData:
    """Return a new CaseData with MSWEP precipitation added to surf_cpu.

    The precipitation is stored as (1, 2, H, W) float32 under the
    PRECIP_VAR key, ready for use in build_precip_batch().

    Args:
        case: Original CaseData (unmodified).
        precip_t0: (H, W) precipitation array at t0.
        precip_t1: (H, W) precipitation array at t1.

    Returns:
        New CaseData with PRECIP_VAR added to surf_cpu.
    """
    import dataclasses

    precip_np = np.stack([precip_t0, precip_t1])   # (2, H, W)
    precip_tensor = _cpu_tensor(precip_np)[None]    # (1, 2, H, W)

    new_surf_cpu = dict(case.surf_cpu)
    new_surf_cpu[PRECIP_VAR] = precip_tensor

    return dataclasses.replace(case, surf_cpu=new_surf_cpu)
