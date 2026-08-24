"""Data loading for geoxplain_aurora_adapter.

Provides ``load_case(init_time)`` → ``CaseData`` and ``make_batch()``.

This targets the **public Microsoft Aurora** model: standard ERA5 surface and
atmospheric variables only (no ZWD).  The model carries its own built-in
normalisation, so no ``locations`` / ``scales`` are loaded here.

All data (atmospheric, surface, and static fields) is loaded from
WeatherBench2 zarr stores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..serving.config import (
    DEFAULT_WEATHERBENCH2_PATHS,
    get_config_section,
    normalize_weatherbench2_path,
)
from ..schema.metadata import AURORA_LEVELS

# ``torch`` is imported lazily inside the functions that need it (``_cpu_tensor``,
# ``build_metadata``).  Reading raw ERA5 fields (``load_overlay_field``) is pure
# numpy/xarray, so the overlay path can run in a torch-free ``[server]`` install.


# ── Paths ────────────────────────────────────────────────────────────────────
def _expand_path(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    return normalize_weatherbench2_path(expanded)


def _split_paths(value: str) -> tuple[str, ...]:
    normalized = value.replace("\n", ";")
    if ";" in normalized:
        parts = normalized.split(";")
    elif os.pathsep != ":" or "://" not in normalized:
        parts = normalized.split(os.pathsep)
    else:
        parts = [normalized]
    return tuple(_expand_path(part.strip()) for part in parts if part.strip())


def _configured_wb2_paths() -> tuple[str, ...]:
    data_cfg = get_config_section("data")
    env_value = os.environ.get("GEOXPLAIN_AURORA_ADAPTER_WB2_PATHS")
    if env_value:
        return _split_paths(env_value)
    value = data_cfg.get("weatherbench2_paths") or data_cfg.get("wb2_paths")
    if isinstance(value, str):
        return _split_paths(value)
    if isinstance(value, list):
        return tuple(_expand_path(str(path)) for path in value if str(path).strip())
    return DEFAULT_WEATHERBENCH2_PATHS


WB2_PATHS = _configured_wb2_paths()
_WB2_DATASET_CACHE: dict[str, xr.Dataset] = {}
_WB2_ATMOS = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}
_WB2_SURF = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

SURF_VARS = ("2t", "10u", "10v", "msl")
ATMOS_VARS = ("z", "u", "v", "t", "q")

# ── Decoupled vertical layer keys (`z-{N}`) ──────────────────────────────────
# Result files key layers by a generic vertical order rather than Aurora's
# pressure levels (see ``result.py``).  ``AURORA_LEVELS`` is in descending
# pressure / ascending altitude order, so its index doubles as ``N``: higher
# altitude → higher ``N`` → higher in the visualization.  ``1000 hPa -> z-0``,
# ``50 hPa -> z-12``.  Surface fields keep the reserved key ``"sfc"``.
SFC_LEVEL_KEY = "sfc"
HPA_TO_Z = {hpa: i for i, hpa in enumerate(AURORA_LEVELS)}


def atmos_level_key(hpa: int) -> str:
    """Vertical key (``"z-{N}"``) for an atmospheric pressure level."""
    return f"z-{HPA_TO_Z[hpa]}"


def atmos_level_label(hpa: int) -> str:
    """Human-readable label carried in ``layer_labels`` for a pressure level."""
    return f"{hpa} hPa"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_remote_path(path: str) -> bool:
    return "://" in path


def _open_wb2_store(path: str) -> xr.Dataset:
    path = normalize_weatherbench2_path(path)
    cached = _WB2_DATASET_CACHE.get(path)
    if cached is not None:
        return cached

    kwargs = {"storage_options": {"token": "anon"}} if path.startswith("gs://") else {}
    try:
        ds = xr.open_zarr(path, **kwargs)
    except ImportError as exc:
        if path.startswith("gs://"):
            raise RuntimeError(
                "Reading WeatherBench2 from gs:// requires gcsfs. Install "
                "the bundled adapter from its source directory with the "
                "[gpu] or [server] extra."
            ) from exc
        raise
    _WB2_DATASET_CACHE[path] = ds
    return ds


def _format_timestamp(value) -> str:
    """Format a datetime-like value consistently in data-coverage errors."""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, AttributeError):
        return str(value)


def _describe_time_index(index) -> str:
    """Return a compact coverage/cadence description for a time index."""
    if len(index) == 0:
        return "an empty time axis"
    start = min(index)
    end = max(index)
    cadence = ""
    if len(index) > 1:
        try:
            seconds = (index[1] - index[0]).total_seconds()
        except (TypeError, ValueError, AttributeError):
            seconds = None
        if seconds is not None and seconds > 0:
            if seconds % 3600 == 0:
                cadence = f", {int(seconds // 3600)}-hour cadence"
            elif seconds % 60 == 0:
                cadence = f", {int(seconds // 60)}-minute cadence"
    return (
        f"{_format_timestamp(start)} through {_format_timestamp(end)}"
        f"{cadence}"
    )


def _find_store(ts: "pd.Timestamp"):
    """Return the first WeatherBench2 store whose time axis contains ``ts``."""
    coverage: list[str] = []
    for path in WB2_PATHS:
        if not _is_remote_path(path) and not os.path.exists(path):
            coverage.append(f"  - {path}: path does not exist")
            continue
        ds = _open_wb2_store(path)
        idx = pd.DatetimeIndex(ds.time.values)
        if ts in idx:
            return ds
        coverage.append(f"  - {path}: {_describe_time_index(idx)}")
    details = "\n".join(coverage) if coverage else "  - no stores are configured"
    raise RuntimeError(
        "WeatherBench2 data does not contain the exact timestamp "
        f"{_format_timestamp(ts)}.\n"
        f"Configured store coverage:\n{details}\n"
        "Choose a timestamp present in one of these stores, or configure another "
        "compatible store via [data].weatherbench2_paths or "
        "GEOXPLAIN_AURORA_ADAPTER_WB2_PATHS. Exact timestamps are required; "
        "the adapter will not silently substitute nearest-time weather data."
    )


def _load_from_wb2(t0: datetime, t1: datetime):
    ts0 = pd.Timestamp(t0)
    ts1 = pd.Timestamp(t1)

    ds0 = _find_store(ts0)
    ds1 = _find_store(ts1)

    def _sel(ds, ts):
        return ds.sel(time=ts)

    s0 = _sel(ds0, ts0)
    s1 = _sel(ds1, ts1)

    atmos_cpu = {}
    for short, long in _WB2_ATMOS.items():
        t0_arr = s0[long].sel(level=list(AURORA_LEVELS)).values
        t1_arr = s1[long].sel(level=list(AURORA_LEVELS)).values
        atmos_cpu[short] = _cpu_tensor(np.stack([t0_arr, t1_arr]))[None]

    surf_cpu = {}
    for short, long in _WB2_SURF.items():
        t0_arr = s0[long].values
        t1_arr = s1[long].values
        surf_cpu[short] = _cpu_tensor(np.stack([t0_arr, t1_arr]))[None]

    lat_vals = ds0["latitude"].values
    lon_vals = ds0["longitude"].values
    pressure_levels = tuple(AURORA_LEVELS)

    return atmos_cpu, surf_cpu, lat_vals, lon_vals, pressure_levels


def _cpu_tensor(arr):
    import torch
    return torch.tensor(np.asarray(arr), dtype=torch.float32)


def _load_static_from_wb2() -> dict:
    """Load static fields (lsm, z, slt) from the first WB2 store that has them."""
    for path in WB2_PATHS:
        if not _is_remote_path(path) and not os.path.exists(path):
            continue
        ds = _open_wb2_store(path)
        if "land_sea_mask" in ds and "geopotential_at_surface" in ds and "soil_type" in ds:
            return {
                "lsm": _cpu_tensor(ds["land_sea_mask"].values),
                "z":   _cpu_tensor(ds["geopotential_at_surface"].values),
                "slt": _cpu_tensor(ds["soil_type"].values),
            }
    raise RuntimeError("No WeatherBench2 store contains the required static fields (land_sea_mask, geopotential_at_surface, soil_type).")


# ── CaseData ─────────────────────────────────────────────────────────────────

@dataclass
class CaseData:
    """Everything needed to build a Batch for one init time."""
    init_time: datetime
    atmos_cpu: dict          # var -> (1, 2, L, H, W) CPU float32
    surf_cpu: dict           # var -> (1, 2, H, W) CPU float32
    static_cpu: dict         # var -> (H, W) CPU float32
    lat_vals: np.ndarray     # (H,)
    lon_vals: np.ndarray     # (W,)
    pressure_levels: tuple


def load_case(init_time: datetime) -> CaseData:
    """Load ERA5 data for a given init time.

    ``init_time`` is the second (t1) of the two input timesteps; t0 is
    ``init_time - 6h``.
    """
    if not isinstance(init_time, datetime):
        raise TypeError(f"init_time must be datetime, got {type(init_time)}")

    t1 = init_time
    t0 = init_time - timedelta(hours=6)

    atmos_cpu, surf_cpu, lat_vals, lon_vals, pressure_levels = _load_from_wb2(t0, t1)
    static_cpu = _load_static_from_wb2()

    return CaseData(
        init_time=init_time,
        atmos_cpu=atmos_cpu,
        surf_cpu=surf_cpu,
        static_cpu=static_cpu,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        pressure_levels=pressure_levels,
    )


def load_overlay_field(
    init_time: datetime,
    variable: str,
    level: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one ERA5 field at the t1 (init) timestep as numpy — no torch.

    Returns ``(data (H, W) float32, lat_vals (H,), lon_vals (W,))``.

    Overlays only need the raw field, not a model batch, so this reads it
    straight from the WeatherBench2 store using numpy/xarray alone.  That lets
    the overlay path run in a torch-free ``[server]`` install (the sbatch
    login-node listener).  The returned values match the t1 slice ``load_case``
    would produce for the same variable and level.
    """
    if not isinstance(init_time, datetime):
        raise TypeError(f"init_time must be datetime, got {type(init_time)}")

    ts1 = pd.Timestamp(init_time)
    ds = _find_store(ts1)
    s1 = ds.sel(time=ts1)

    if variable in _WB2_ATMOS:
        if level is None:
            raise ValueError(f"Atmospheric variable {variable!r} requires a level (hPa).")
        arr = s1[_WB2_ATMOS[variable]].sel(level=level).values
    elif variable in _WB2_SURF:
        arr = s1[_WB2_SURF[variable]].values
    else:
        raise ValueError(
            f"Unknown overlay variable {variable!r}. "
            f"ATMOS_VARS={list(ATMOS_VARS)}, SURF_VARS={list(SURF_VARS)}"
        )

    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    return np.asarray(arr, dtype=np.float32), lat_vals, lon_vals


def build_metadata(case: CaseData, device) -> "Metadata":  # noqa: F821
    """Build an Aurora ``Metadata`` for this case, placed on ``device``.

    Matches the minimal metadata used by the base-model experiments
    (``aurora-experiments/base_temperature``): only ``lat``, ``lon``,
    ``time`` and ``atmos_levels``.  The public Aurora model supplies its own
    normalisation, so no ``locations`` / ``scales`` are passed.
    """
    import torch
    from aurora import Metadata  # type: ignore[import]

    lats = torch.tensor(case.lat_vals, dtype=torch.float32, device=device)
    lons = torch.tensor(case.lon_vals, dtype=torch.float32, device=device)
    return Metadata(
        lat=lats,
        lon=lons,
        time=(case.init_time,),
        atmos_levels=case.pressure_levels,
    )


def make_batch(
    case: CaseData,
    device,
    *,
    requires_grad_surf: tuple = (),
    requires_grad_atmos: tuple = (),
    surf_overrides: Optional[dict] = None,
    atmos_overrides: Optional[dict] = None,
) -> "Batch":  # noqa: F821
    """Construct an Aurora Batch from case data.

    Parameters
    ----------
    case:               CaseData from ``load_case()``.
    device:             torch device.
    requires_grad_surf: Surface-var names to mark as leaf tensors with grad.
    requires_grad_atmos:Atmos-var names to mark as leaf tensors with grad.
    surf_overrides:     Optional ``{var_name: cpu_tensor}`` dict replacing
                        individual surface variables before moving to device.
    atmos_overrides:    Optional ``{var_name: cpu_tensor}`` dict replacing
                        individual atmospheric variables before moving to
                        device.
    """
    from aurora import Batch  # type: ignore[import]

    surf_dev: dict[str, torch.Tensor] = {}
    for k in SURF_VARS:
        if surf_overrides and k in surf_overrides:
            v = surf_overrides[k].clone().to(device)
        else:
            v = case.surf_cpu[k].clone().to(device)
        if k in requires_grad_surf:
            v.requires_grad_(True)
        surf_dev[k] = v

    atmos_dev: dict[str, torch.Tensor] = {}
    for k in ATMOS_VARS:
        if atmos_overrides and k in atmos_overrides:
            v = atmos_overrides[k].clone().to(device)
        else:
            v = case.atmos_cpu[k].clone().to(device)
        if k in requires_grad_atmos:
            v.requires_grad_(True)
        atmos_dev[k] = v

    return Batch(
        surf_vars=surf_dev,
        static_vars={k: v.to(device) for k, v in case.static_cpu.items()},
        atmos_vars=atmos_dev,
        metadata=build_metadata(case, device),
    )
