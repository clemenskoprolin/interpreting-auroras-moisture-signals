"""
Shared helpers for ZWD/proxy correlation diagnostics.

The scripts in this directory are CPU/data-only diagnostics. They reuse the
target boxes and geometry helpers from the searchlight benchmark, but do not
load Aurora or run model inference.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEARCHLIGHT_DIR = os.path.join(ROOT, "02_zwd_attribution_benchmark")

for path in (SEARCHLIGHT_DIR, ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from searchlight_tasks import (  # noqa: E402
    SCALES,
    TARGETS,
    MaskSpec,
    cos_lat_weights,
    gaussian_mask,
    great_circle_km,
)


RESULTS_ROOT = os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(ROOT, "results"))
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_ROOT, "zwd_correlation_diagnostics")

DEFAULT_ZWD_PATH = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)
_FALLBACK_WB_PATHS = (
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
DEFAULT_WB_PATHS = tuple(
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_FALLBACK_WB_PATHS)
    ).split(os.pathsep) if path
)
AURORA_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

WB2_ATMOS = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}
WB2_SURF = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}


@dataclass(frozen=True)
class CaseKey:
    target: str
    init_time: datetime
    role: str
    target_mode: str | None = None

    @property
    def base_case_id(self) -> str:
        return f"{self.target}_{self.init_time.strftime('%Y%m%d%H')}_{self.role}"

    @property
    def case_id(self) -> str:
        if self.target_mode and self.target_mode != "box":
            return f"{self.base_case_id}__{self.target_mode}"
        return self.base_case_id


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: object, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    print(f"Saved: {path}")


def to_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def norm_lon_360(values) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=np.float64), 360.0)


def subset_lon_mask(lon_vals, west: float, east: float) -> np.ndarray:
    lon_vals = norm_lon_360(lon_vals)
    west = west % 360.0
    east = east % 360.0
    if west <= east:
        return (lon_vals >= west) & (lon_vals <= east)
    # west > east means the range wraps across 0°/360° (e.g. W=350°, E=10°)
    return (lon_vals >= west) | (lon_vals <= east)


def filter_times(
    ds: xr.Dataset,
    time_dim: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    candidate_hours: set[int] | None,
) -> xr.Dataset:
    times = pd.DatetimeIndex(ds[time_dim].values)
    mask = times >= start_ts
    if end_ts is not None:
        mask &= times <= end_ts
    if candidate_hours:
        mask &= times.hour.isin(sorted(candidate_hours))
    return ds.isel({time_dim: np.where(mask)[0]})


def select_time_if_present(ds: xr.Dataset, ts: pd.Timestamp) -> xr.Dataset | None:
    times = pd.DatetimeIndex(ds.time.values)
    if ts not in times:
        return None
    return ds.sel(time=ts)


def open_wb_store_for_time(ts: pd.Timestamp, wb_paths: Iterable[str]) -> xr.Dataset:
    for path in wb_paths:
        if not os.path.exists(path):
            continue
        ds = xr.open_zarr(path)
        if ts in pd.DatetimeIndex(ds.time.values):
            return ds
    raise RuntimeError(f"Timestamp {ts} not found in any WeatherBench2 store.")


def target_box_subset(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_key: str,
) -> xr.DataArray:
    target = TARGETS[target_key]
    lat_vals = da[lat_name].values
    lon_vals = norm_lon_360(da[lon_name].values)

    lat_mask = (lat_vals >= target.box_lat[0]) & (lat_vals <= target.box_lat[1])
    lon_mask = subset_lon_mask(lon_vals, target.box_lon[0], target.box_lon[1])
    return da.sel(
        {
            lat_name: da[lat_name].values[lat_mask],
            lon_name: da[lon_name].values[lon_mask],
        }
    )


def weighted_box_mean(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_key: str,
) -> xr.DataArray:
    sub = target_box_subset(da, lat_name, lon_name, target_key)
    weights = xr.DataArray(
        np.cos(np.deg2rad(sub[lat_name].values)).clip(min=0.0),
        dims=(lat_name,),
        coords={lat_name: sub[lat_name].values},
    )
    return sub.weighted(weights).mean(dim=(lat_name, lon_name))


def disk_mask_da(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_key: str,
    radius_km: float,
) -> xr.DataArray:
    target = TARGETS[target_key]
    lat_vals = da[lat_name].values
    lon_vals = norm_lon_360(da[lon_name].values)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    mask = great_circle_km(
        target.center_lat,
        target.center_lon,
        lat_grid,
        lon_grid,
    ) <= radius_km
    return xr.DataArray(
        mask,
        dims=(lat_name, lon_name),
        coords={lat_name: da[lat_name].values, lon_name: da[lon_name].values},
    )


def weighted_region_mean(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_key: str,
    region: str,
    disk_radius_km: float,
) -> xr.DataArray:
    if region == "box":
        return weighted_box_mean(da, lat_name, lon_name, target_key)

    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values
    weights_np = np.cos(np.deg2rad(lat_vals)).clip(min=0.0)[:, None]
    weights_np = np.broadcast_to(weights_np, (len(lat_vals), len(lon_vals))).astype(np.float32)

    if region == "disk":
        mask = disk_mask_da(da, lat_name, lon_name, target_key, disk_radius_km)
        weights_np = np.where(mask.values, weights_np, 0.0)
    elif region != "global":
        raise ValueError(f"Unknown region: {region!r}")

    weights = xr.DataArray(
        weights_np,
        dims=(lat_name, lon_name),
        coords={lat_name: da[lat_name].values, lon_name: da[lon_name].values},
    )
    return da.weighted(weights).mean(dim=(lat_name, lon_name))


def region_quantile(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_key: str,
    region: str,
    q: float,
    disk_radius_km: float,
) -> xr.DataArray:
    if region == "box":
        sub = target_box_subset(da, lat_name, lon_name, target_key)
        return sub.quantile(q, dim=(lat_name, lon_name), skipna=True)
    if region == "disk":
        mask = disk_mask_da(da, lat_name, lon_name, target_key, disk_radius_km)
        return da.where(mask).chunk({lat_name: -1, lon_name: -1}).quantile(q, dim=(lat_name, lon_name), skipna=True)
    if region == "global":
        return da.chunk({lat_name: -1, lon_name: -1}).quantile(q, dim=(lat_name, lon_name), skipna=True)
    raise ValueError(f"Unknown region: {region!r}")


def pressure_layer_weights(levels_hpa: Iterable[int | float]) -> np.ndarray:
    """Return positive pressure-layer weights for sparse pressure levels."""
    levels = np.asarray(list(levels_hpa), dtype=np.float64)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("Need at least two pressure levels for column weights.")

    # Work in descending pressure order, e.g. 1000, 925, ..., 50 hPa.
    order = np.argsort(levels)[::-1]
    sorted_levels = levels[order]
    mids = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
    # Extrapolate outermost bounds by half the adjacent gap so every level gets
    # a symmetric layer; clamp the top bound to avoid going below 0 hPa.
    bottom = sorted_levels[0] + 0.5 * (sorted_levels[0] - sorted_levels[1])
    top = max(0.0, sorted_levels[-1] - 0.5 * (sorted_levels[-2] - sorted_levels[-1]))
    bounds = np.concatenate([[bottom], mids, [top]])
    weights_sorted = bounds[:-1] - bounds[1:]
    weights = np.empty_like(weights_sorted)
    weights[order] = weights_sorted
    return weights.astype(np.float64)


def pressure_weighted_mean(da: xr.DataArray, levels: list[int]) -> xr.DataArray:
    weights = pressure_layer_weights(levels)
    w = xr.DataArray(weights, dims=("level",), coords={"level": levels})
    return da.weighted(w).mean(dim="level")


def low_level_mean(da: xr.DataArray, available_levels: list[int]) -> xr.DataArray:
    low_levels = [lev for lev in (1000, 925, 850) if lev in available_levels]
    if not low_levels:
        raise ValueError("No low-level pressure levels available.")
    return da.sel(level=low_levels).mean(dim="level")


def pairwise_corr(a, b, method: str) -> float:
    df = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return float("nan")
    if df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan")
    if method == "pearson":
        return float(df["a"].corr(df["b"], method="pearson"))
    if method == "spearman":
        return float(df["a"].corr(df["b"], method="spearman"))
    raise ValueError(f"Unknown correlation method: {method!r}")


def month_zscore(s: pd.Series, group_key=None) -> pd.Series:
    if group_key is None:
        groups = s.groupby(s.index.month)
    else:
        groups = s.groupby([group_key, s.index.month])

    def _z(x: pd.Series) -> pd.Series:
        std = float(x.std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
        return (x - float(x.mean())) / std

    return groups.transform(_z)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    out["month"] = idx.month
    out["year"] = idx.year
    out["season"] = [season_name(m) for m in idx.month]
    return out


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def parse_case_id(case_id: str) -> CaseKey:
    target_mode = None
    base = case_id
    if "__" in case_id:
        base, target_mode = case_id.split("__", 1)
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse case_id: {case_id!r}")
    target = parts[0]
    init_time = datetime.strptime(parts[1], "%Y%m%d%H")
    role = "_".join(parts[2:])
    return CaseKey(target=target, init_time=init_time, role=role, target_mode=target_mode)


def mask_from_gt_dict(mask: dict, scale_name: str) -> MaskSpec:
    role = mask.get("role", "near")
    key = mask.get("key", f"{scale_name}_{role}_000")
    try:
        mask_id = int(key.split("_")[-1])
    except Exception:
        mask_id = 0
    return MaskSpec(
        scale=scale_name,
        role=role,
        center_lat=float(mask["center_lat"]),
        center_lon=float(mask["center_lon"]),
        mask_id=mask_id,
    )


def pool_field_over_mask(
    field_hw: np.ndarray,
    mask_hw: np.ndarray,
    cos_weights_hw: np.ndarray,
) -> float:
    field = np.asarray(field_hw, dtype=np.float64)
    weights = np.asarray(mask_hw, dtype=np.float64) * np.asarray(cos_weights_hw, dtype=np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    return float(np.sum(field[finite] * weights[finite]) / np.sum(weights[finite]))


def global_weighted_zscore(field_hw: np.ndarray, lat_vals: np.ndarray) -> np.ndarray:
    field = np.asarray(field_hw, dtype=np.float64)
    weights = cos_lat_weights(np.asarray(lat_vals), field.shape[1]).astype(np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return np.full_like(field, np.nan, dtype=np.float64)
    mean = np.sum(field[finite] * weights[finite]) / np.sum(weights[finite])
    var = np.sum(((field[finite] - mean) ** 2) * weights[finite]) / np.sum(weights[finite])
    std = float(np.sqrt(max(var, 0.0)))
    if std < 1e-12:
        return np.zeros_like(field, dtype=np.float64)
    return (field - mean) / std
