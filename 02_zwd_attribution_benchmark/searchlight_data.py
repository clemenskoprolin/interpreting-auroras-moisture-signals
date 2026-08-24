"""
Data loading for the ZWD searchlight benchmark.

Provides a single entry point `load_case(init_time)` that returns a dict
with everything needed to build an Aurora Batch for the ZWD-augmented
Aurora model. The Aurora fork that supports the added channels must be
installed separately.

Loading priority for ERA5 atmospheric/surface data:
    1. Dated .nc files:  era5_atmos_{YYYY-MM-DD}.nc / era5_surface_{YYYY-MM-DD}.nc
    2. Legacy .nc files: era5_atmos.nc / era5_surface.nc  (2020-01-01 snapshot)
    3. WeatherBench2 zarr stores (automatic fallback for any date 2020-2025)

The WB2 fallback means no manual data preparation is needed for new dates;
load_case() will pull the right timesteps directly from the zarr stores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import xarray as xr


# ------------------------------------------------------------------
# Paths: portable environment overrides with original CSCS fallbacks.
# ------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("AURORA_ERA5_CACHE_DIR", os.path.join(_ROOT, "data"))
ZWD_ZARR_PATH = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)

# WeatherBench2 stores used as fallback when no .nc file exists for a date
_DEFAULT_WB2_PATHS = (
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
WB2_PATHS = tuple(
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_DEFAULT_WB2_PATHS)
    ).split(os.pathsep) if path
)
# WB2 long-name -> Aurora short-name mapping
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
AURORA_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
NORM_STATS_PATH = os.environ.get(
    "AURORA_NORMALIZATION_STATS",
    os.path.join(_ROOT, "config", "normalization_stats_1979_2021.json"),
)
CHECKPOINT_PATH = os.environ.get(
    "AURORA_ZWD_CHECKPOINT",
    "/capstor/store/cscs/swissai/a122/ltrentini/checkpoints_xAI/"
    "zwd_lw_scheduler/model_ckpt-step=4300-loss_train=0.06.ckpt",
)

SURF_VARS = ("2t", "10u", "10v", "msl", "zwd")
ATMOS_VARS = ("z", "u", "v", "t", "q")


def _era5_nc_paths(init_time: datetime):
    """Return (atmos_path, surface_path) for .nc files, or (None, None) if absent."""
    date_str = init_time.strftime("%Y-%m-%d")
    dated_atmos = os.path.join(DATA_DIR, f"era5_atmos_{date_str}.nc")
    dated_surf  = os.path.join(DATA_DIR, f"era5_surface_{date_str}.nc")
    legacy_atmos = os.path.join(DATA_DIR, "era5_atmos.nc")
    legacy_surf  = os.path.join(DATA_DIR, "era5_surface.nc")

    # Prefer dated, fall back to legacy only if it matches the date (2020-01-01)
    if os.path.exists(dated_atmos) and os.path.exists(dated_surf):
        return dated_atmos, dated_surf
    if init_time.date() == pd.Timestamp("2020-01-01").date() and \
            os.path.exists(legacy_atmos) and os.path.exists(legacy_surf):
        return legacy_atmos, legacy_surf
    return None, None


def _load_from_wb2(t0: datetime, t1: datetime):
    """Load atmos_cpu and surf_cpu directly from WeatherBench2 zarr stores."""
    ts0 = pd.Timestamp(t0)
    ts1 = pd.Timestamp(t1)

    # Find a store that contains both timestamps (or each individually)
    def _find_store(ts):
        for path in WB2_PATHS:
            if not os.path.exists(path):
                continue
            ds = xr.open_zarr(path)
            idx = pd.DatetimeIndex(ds.time.values)
            if ts in idx:
                return ds
        raise RuntimeError(
            f"Timestamp {ts} not found in any WeatherBench2 store."
        )

    ds0 = _find_store(ts0)
    ds1 = _find_store(ts1)

    def _sel(ds, ts):
        return ds.sel(time=ts)

    s0 = _sel(ds0, ts0)
    s1 = _sel(ds1, ts1)

    # Atmospheric: (1, 2, L, H, W)
    atmos_cpu = {}
    for short, long in _WB2_ATMOS.items():
        t0_arr = s0[long].sel(level=list(AURORA_LEVELS)).values  # (L, H, W)
        t1_arr = s1[long].sel(level=list(AURORA_LEVELS)).values
        atmos_cpu[short] = _cpu_tensor(np.stack([t0_arr, t1_arr]))[None]

    # Surface: (1, 2, H, W)
    surf_cpu = {}
    for short, long in _WB2_SURF.items():
        t0_arr = s0[long].values  # (H, W)
        t1_arr = s1[long].values
        surf_cpu[short] = _cpu_tensor(np.stack([t0_arr, t1_arr]))[None]

    lat_vals = ds0["latitude"].values
    lon_vals = ds0["longitude"].values
    pressure_levels = tuple(AURORA_LEVELS)

    return atmos_cpu, surf_cpu, lat_vals, lon_vals, pressure_levels


def _cpu_tensor(arr) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=torch.float32)


@dataclass
class CaseData:
    """Everything needed to build a Batch for one init time."""
    init_time: datetime
    atmos_cpu: dict          # var -> (1, 2, L, H, W) CPU float32
    surf_cpu: dict           # var -> (1, 2, H, W) CPU float32
    static_cpu: dict         # var -> (H, W) CPU float32
    lat_vals: np.ndarray     # (H,)
    lon_vals: np.ndarray     # (W,)
    pressure_levels: tuple[int, ...]
    locations: dict          # normalization locations (all keys, python floats)
    scales: dict             # normalization scales

    @property
    def zwd_loc(self) -> float:
        return float(self.locations["zwd"])

    @property
    def zwd_scale(self) -> float:
        return float(self.scales["zwd"])


def load_normalization_stats() -> tuple[dict, dict]:
    with open(NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    return stats["locations"], stats["scales"]


def load_case(init_time: datetime) -> CaseData:
    """Load ERA5 + ZWD data for a given init time.

    init_time is interpreted as the second (t1) of the two input timesteps.
    t0 is init_time - 6h.  This matches the 6h fine-tuned ZWD model cadence.
    """
    if not isinstance(init_time, datetime):
        raise TypeError(f"init_time must be datetime, got {type(init_time)}")

    t1 = init_time
    t0 = init_time - timedelta(hours=6)

    atmos_path, surf_path = _era5_nc_paths(init_time)

    if atmos_path is not None:
        # Fast path: pre-existing .nc files
        ds_atmos = xr.open_dataset(atmos_path)
        ds_surface = xr.open_dataset(surf_path)
        lat_vals = ds_atmos.latitude.values
        lon_vals = ds_atmos.longitude.values
        pressure_levels = tuple(int(p) for p in ds_atmos.pressure_level.values)
        atmos_cpu = {
            k: _cpu_tensor(np.stack([ds_atmos[k].values[0], ds_atmos[k].values[1]])[None])
            for k in ATMOS_VARS
        }
        surf_cpu = {
            "2t":  _cpu_tensor(ds_surface["t2m"].values)[None],
            "10u": _cpu_tensor(ds_surface["u10"].values)[None],
            "10v": _cpu_tensor(ds_surface["v10"].values)[None],
            "msl": _cpu_tensor(ds_surface["msl"].values)[None],
        }
    else:
        # Otherwise: load directly from WeatherBench2 zarr stores
        print(f"  Loading {init_time.date()} from WeatherBench2 zarr...")
        atmos_cpu, surf_cpu, lat_vals, lon_vals, pressure_levels = _load_from_wb2(t0, t1)

    # ZWD from zarr at (t0, t1)
    ds_zwd = xr.open_zarr(ZWD_ZARR_PATH)
    zwd_var = "zenith_wet_delay"
    try:
        zwd_t0 = ds_zwd[zwd_var].sel(time=pd.Timestamp(t0)).values
        zwd_t1 = ds_zwd[zwd_var].sel(time=pd.Timestamp(t1)).values
    except KeyError as e:
        raise RuntimeError(
            f"ZWD zarr does not contain time(s) {t0} / {t1}: {e}"
        ) from e
    zwd_np = np.stack([zwd_t0, zwd_t1])
    surf_cpu["zwd"] = _cpu_tensor(zwd_np)[None]

    ds_static = xr.open_dataset(os.path.join(DATA_DIR, "era5_static.nc"))
    static_cpu = {
        "lsm": _cpu_tensor(ds_static["lsm"].values[0]),
        "z":   _cpu_tensor(ds_static["z"].values[0]),
        "slt": _cpu_tensor(ds_static["slt"].values[0]),
    }

    locations, scales = load_normalization_stats()

    return CaseData(
        init_time=init_time,
        atmos_cpu=atmos_cpu,
        surf_cpu=surf_cpu,
        static_cpu=static_cpu,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        pressure_levels=pressure_levels,
        locations=locations,
        scales=scales,
    )


def build_metadata(case: CaseData, device) -> "Metadata":
    """Build an Aurora `Metadata` for this case, placed on `device`."""
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
        dataset_name="era5_zwd",
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


def make_batch(
    case: CaseData,
    device,
    *,
    requires_grad_surf: tuple[str, ...] = (),
    requires_grad_atmos: tuple[str, ...] = (),
    zwd_override: torch.Tensor | None = None,
    q_override: torch.Tensor | None = None,
    tp_mswep_override: torch.Tensor | None = None,
) -> "Batch":
    """Construct an Aurora Batch from case data.

    Args:
        case: CaseData returned by load_case().
        device: torch device.
        requires_grad_surf: surface-var names to mark as leaf-with-grad.
        requires_grad_atmos: atmos-var names to mark as leaf-with-grad.
        zwd_override: optional CPU float32 tensor of shape (1, 2, H, W) that
            replaces case.surf_cpu["zwd"] before moving to device.
        q_override: optional CPU float32 tensor of shape (1, 2, 13, H, W) that
            replaces case.atmos_cpu["q"] before moving to device.
        tp_mswep_override: optional CPU float32 tensor of shape (1, 2, H, W) that
            replaces case.surf_cpu["tp_mswep"] before moving to device.
    """
    from aurora import Batch

    surf_dev: dict[str, torch.Tensor] = {}
    for k in SURF_VARS:
        if k == "zwd" and zwd_override is not None:
            v = zwd_override.clone().to(device)
        elif k == "tp_mswep" and tp_mswep_override is not None:
            v = tp_mswep_override.clone().to(device)
        else:
            v = case.surf_cpu[k].clone().to(device)
        if k in requires_grad_surf:
            v.requires_grad_(True)
        surf_dev[k] = v

    atmos_dev: dict[str, torch.Tensor] = {}
    for k in ATMOS_VARS:
        if k == "q" and q_override is not None:
            v = q_override.clone().to(device)
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
