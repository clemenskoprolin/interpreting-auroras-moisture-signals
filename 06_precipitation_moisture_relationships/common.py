"""
Shared helpers for precipitation/ZWD/humidity correlation diagnostics.

This package intentionally reuses the target geometry and generic statistics
helpers from 03_zwd_moisture_relationships, but keeps MSWEP access separate: the
MSWEP Zarr store is read through a lightweight direct chunk reader because
opening it with xarray can be very slow on the shared filesystem.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZWD_DIAG_DIR = os.path.join(ROOT, "03_zwd_moisture_relationships")

for path in (ROOT, ZWD_DIAG_DIR):
    if path not in sys.path:
        sys.path.append(path)


def _load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ZWD_COMMON = _load_module_from_path(
    "_zwd_correlation_common",
    os.path.join(ZWD_DIAG_DIR, "common.py"),
)
_ZWD_MODULE_CACHE: dict[str, object] = {}


def load_zwd_diag_module(name: str):
    """Load a sibling 03_zwd_moisture_relationships module under a private name."""
    if name in _ZWD_MODULE_CACHE:
        return _ZWD_MODULE_CACHE[name]
    module = _load_module_from_path(
        f"_zwd_correlation_{name}",
        os.path.join(ZWD_DIAG_DIR, f"{name}.py"),
    )
    _ZWD_MODULE_CACHE[name] = module
    return module


# Re-export the generic ZWD diagnostic helpers used by this package.
AURORA_LEVELS = _ZWD_COMMON.AURORA_LEVELS
CaseKey = _ZWD_COMMON.CaseKey
DEFAULT_WB_PATHS = _ZWD_COMMON.DEFAULT_WB_PATHS
DEFAULT_ZWD_PATH = _ZWD_COMMON.DEFAULT_ZWD_PATH
SCALES = _ZWD_COMMON.SCALES
MaskSpec = _ZWD_COMMON.MaskSpec
RESULTS_ROOT = _ZWD_COMMON.RESULTS_ROOT
TARGETS = _ZWD_COMMON.TARGETS
WB2_ATMOS = _ZWD_COMMON.WB2_ATMOS
WB2_SURF = _ZWD_COMMON.WB2_SURF
add_time_columns = _ZWD_COMMON.add_time_columns
cos_lat_weights = _ZWD_COMMON.cos_lat_weights
ensure_dir = _ZWD_COMMON.ensure_dir
filter_times = _ZWD_COMMON.filter_times
gaussian_mask = _ZWD_COMMON.gaussian_mask
global_weighted_zscore = _ZWD_COMMON.global_weighted_zscore
great_circle_km = _ZWD_COMMON.great_circle_km
low_level_mean = _ZWD_COMMON.low_level_mean
mask_from_gt_dict = _ZWD_COMMON.mask_from_gt_dict
month_zscore = _ZWD_COMMON.month_zscore
norm_lon_360 = _ZWD_COMMON.norm_lon_360
open_wb_store_for_time = _ZWD_COMMON.open_wb_store_for_time
pairwise_corr = _ZWD_COMMON.pairwise_corr
parse_case_id = _ZWD_COMMON.parse_case_id
pool_field_over_mask = _ZWD_COMMON.pool_field_over_mask
region_quantile = _ZWD_COMMON.region_quantile
pressure_weighted_mean = _ZWD_COMMON.pressure_weighted_mean
save_json = _ZWD_COMMON.save_json
select_time_if_present = _ZWD_COMMON.select_time_if_present
season_name = _ZWD_COMMON.season_name
subset_lon_mask = _ZWD_COMMON.subset_lon_mask
target_box_subset = _ZWD_COMMON.target_box_subset
to_jsonable = _ZWD_COMMON.to_jsonable
weighted_region_mean = _ZWD_COMMON.weighted_region_mean


DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_ROOT, "p_correlation_diagnostics")
DEFAULT_PRECIP_PATH = os.environ.get(
    "AURORA_MSWEP_DATA",
    "/capstor/store/cscs/swissai/a122/hydrological_data/"
    "MSWEP-v280-720x1440-6h_acc_3h_sampling.zarr",
)
DEFAULT_PRECIP_VAR = "total_precipitation_MSWEP"
PRECIP_CHECKPOINT_CONTEXT = {
    "precip_zwd_model": os.environ.get("AURORA_PRECIP_ZWD_CHECKPOINT", (
        "/capstor/store/cscs/swissai/a122/ltrentini/checkpoints_xAI/"
        "precip_small/model_ckpt-step=6200-loss_train=0.06.ckpt"
    )),
    "precip_only_model": os.environ.get("AURORA_PRECIP_ONLY_CHECKPOINT", (
        "/capstor/store/cscs/swissai/a122/ltrentini/checkpoints_xAI/"
        "precip_small_without_zwd/model_ckpt-step=7000-loss_train=0.07.ckpt"
    )),
}


@dataclass(frozen=True)
class ZarrArrayMeta:
    path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    order: str
    dimension_separator: str
    fill_value: object
    compressor: object


class MSWEPReader:
    """Small direct reader for the MSWEP precipitation Zarr v2 store."""

    def __init__(self, store_path: str = DEFAULT_PRECIP_PATH, var_name: str = DEFAULT_PRECIP_VAR):
        self.store_path = store_path
        self.var_name = var_name
        if not os.path.isdir(store_path):
            raise FileNotFoundError(f"MSWEP store does not exist: {store_path}")

        self._metas = {
            "time": self._load_meta("time"),
            "latitude": self._load_meta("latitude"),
            "longitude": self._load_meta("longitude"),
            var_name: self._load_meta(var_name),
        }
        self.latitude = self._read_1d_array("latitude").astype(np.float64)
        self.longitude = self._read_1d_array("longitude").astype(np.float64)
        raw_time = self._read_1d_array("time").astype(np.int64)
        self.time = self._decode_time(raw_time)

    @property
    def precip_meta(self) -> ZarrArrayMeta:
        return self._metas[self.var_name]

    def _load_meta(self, name: str) -> ZarrArrayMeta:
        path = os.path.join(self.store_path, name)
        with open(os.path.join(path, ".zarray")) as f:
            zarray = json.load(f)
        compressor = None
        if zarray.get("compressor") is not None:
            from numcodecs import get_codec

            compressor = get_codec(zarray["compressor"])
        return ZarrArrayMeta(
            path=path,
            shape=tuple(int(x) for x in zarray["shape"]),
            chunks=tuple(int(x) for x in zarray["chunks"]),
            dtype=np.dtype(zarray["dtype"]),
            order=zarray.get("order", "C"),
            dimension_separator=zarray.get("dimension_separator", "."),
            fill_value=zarray.get("fill_value"),
            compressor=compressor,
        )

    def _chunk_key(self, meta: ZarrArrayMeta, chunk_indices: tuple[int, ...]) -> str:
        sep = meta.dimension_separator
        if sep == "/":
            return os.path.join(*(str(i) for i in chunk_indices))
        return sep.join(str(i) for i in chunk_indices)

    def _fill_chunk(self, meta: ZarrArrayMeta) -> np.ndarray:
        fill = meta.fill_value
        if fill == "NaN":
            fill = np.nan
        if fill is None:
            fill = 0 if not np.issubdtype(meta.dtype, np.floating) else np.nan
        return np.full(meta.chunks, fill, dtype=meta.dtype)

    def _read_chunk(self, meta: ZarrArrayMeta, chunk_indices: tuple[int, ...]) -> np.ndarray:
        chunk_path = os.path.join(meta.path, self._chunk_key(meta, chunk_indices))
        if not os.path.exists(chunk_path):
            return self._fill_chunk(meta)
        with open(chunk_path, "rb") as f:
            raw = f.read()
        if meta.compressor is not None:
            raw = meta.compressor.decode(raw)
        arr = np.frombuffer(raw, dtype=meta.dtype)
        expected = int(np.prod(meta.chunks))
        if arr.size < expected:
            out = self._fill_chunk(meta).reshape(-1)
            out[: arr.size] = arr
            arr = out
        elif arr.size > expected:
            arr = arr[:expected]
        return arr.reshape(meta.chunks, order=meta.order)

    def _read_1d_array(self, name: str) -> np.ndarray:
        meta = self._metas[name]
        if len(meta.shape) != 1:
            raise ValueError(f"{name!r} is not one-dimensional")
        out = np.empty(meta.shape[0], dtype=meta.dtype)
        n_chunks = math.ceil(meta.shape[0] / meta.chunks[0])
        for ci in range(n_chunks):
            start = ci * meta.chunks[0]
            end = min(start + meta.chunks[0], meta.shape[0])
            chunk = self._read_chunk(meta, (ci,))
            out[start:end] = chunk[: end - start]
        return out

    def _decode_time(self, raw_time: np.ndarray) -> pd.DatetimeIndex:
        attrs_path = os.path.join(self.store_path, "time", ".zattrs")
        with open(attrs_path) as f:
            attrs = json.load(f)
        units = attrs.get("units", "")
        match = re.match(r"hours since (.+)", units)
        if match is None:
            raise ValueError(f"Unsupported MSWEP time units: {units!r}")
        base = pd.Timestamp(match.group(1))
        return pd.DatetimeIndex(base + pd.to_timedelta(raw_time, unit="h"))

    def selected_time_indices(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        candidate_hours: Iterable[int] | None,
    ) -> np.ndarray:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end is not None else None
        mask = self.time >= start_ts
        if end_ts is not None:
            mask &= self.time <= end_ts
        if candidate_hours:
            mask &= self.time.hour.isin(sorted(set(int(h) for h in candidate_hours)))
        return np.where(mask)[0]

    def index_for_timestamp(self, ts: pd.Timestamp) -> int | None:
        matches = np.where(self.time == pd.Timestamp(ts))[0]
        if len(matches) == 0:
            return None
        return int(matches[0])

    def read_subset(
        self,
        time_index: int,
        lat_indices: np.ndarray | None = None,
        lon_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        meta = self.precip_meta
        if len(meta.shape) != 3:
            raise ValueError(f"Expected 3D precipitation array, got shape {meta.shape}")
        n_time, n_lat, n_lon = meta.shape
        if not (0 <= int(time_index) < n_time):
            raise IndexError(f"time index out of bounds: {time_index}")

        if lat_indices is None:
            lat_indices = np.arange(n_lat, dtype=np.int64)
        else:
            lat_indices = np.asarray(lat_indices, dtype=np.int64)
        if lon_indices is None:
            lon_indices = np.arange(n_lon, dtype=np.int64)
        else:
            lon_indices = np.asarray(lon_indices, dtype=np.int64)

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            return np.empty((len(lat_indices), len(lon_indices)), dtype=np.float32)

        out = np.full((len(lat_indices), len(lon_indices)), np.nan, dtype=np.float32)
        ct, clat, clon = meta.chunks
        t_chunk = int(time_index) // ct
        local_t = int(time_index) - t_chunk * ct

        lat_chunks = np.unique(lat_indices // clat)
        lon_chunks = np.unique(lon_indices // clon)
        for lat_chunk in lat_chunks:
            lat_start = int(lat_chunk) * clat
            lat_end = min(lat_start + clat, n_lat)
            lat_mask = (lat_indices >= lat_start) & (lat_indices < lat_end)
            lat_global = lat_indices[lat_mask]
            lat_out = np.where(lat_mask)[0]

            for lon_chunk in lon_chunks:
                lon_start = int(lon_chunk) * clon
                lon_end = min(lon_start + clon, n_lon)
                lon_mask = (lon_indices >= lon_start) & (lon_indices < lon_end)
                lon_global = lon_indices[lon_mask]
                lon_out = np.where(lon_mask)[0]

                chunk = self._read_chunk(meta, (t_chunk, int(lat_chunk), int(lon_chunk)))
                plane = chunk[
                    local_t,
                    : lat_end - lat_start,
                    : lon_end - lon_start,
                ].astype(np.float32, copy=False)
                out[np.ix_(lat_out, lon_out)] = plane[
                    np.ix_(lat_global - lat_start, lon_global - lon_start)
                ]
        return out

    def read_timestamp(
        self,
        ts: pd.Timestamp,
        lat_indices: np.ndarray | None = None,
        lon_indices: np.ndarray | None = None,
    ) -> np.ndarray | None:
        idx = self.index_for_timestamp(pd.Timestamp(ts))
        if idx is None:
            return None
        return self.read_subset(idx, lat_indices=lat_indices, lon_indices=lon_indices)


def target_region_mask(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    target_key: str,
    region: str,
    disk_radius_km: float,
) -> np.ndarray:
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target {target_key!r}")
    target = TARGETS[target_key]
    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    lon_vals = norm_lon_360(lon_vals)

    if region == "box":
        lat_mask = (lat_vals >= target.box_lat[0]) & (lat_vals <= target.box_lat[1])
        lon_mask = subset_lon_mask(lon_vals, target.box_lon[0], target.box_lon[1])
        return lat_mask[:, None] & lon_mask[None, :]
    if region == "disk":
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        return (
            great_circle_km(target.center_lat, target.center_lon, lat_grid, lon_grid)
            <= disk_radius_km
        )
    if region == "global":
        return np.ones((len(lat_vals), len(lon_vals)), dtype=bool)
    raise ValueError(f"Unknown region: {region!r}")


def weighted_masked_mean(values: np.ndarray, lat_vals: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    weights = np.cos(np.deg2rad(np.asarray(lat_vals, dtype=np.float64))).clip(min=0.0)[:, None]
    weights = np.broadcast_to(weights, values.shape)
    finite = np.isfinite(values) & mask & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    return float(np.sum(values[finite] * weights[finite]) / np.sum(weights[finite]))


def masked_quantile(values: np.ndarray, mask: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & np.asarray(mask, dtype=bool)
    if not finite.any():
        return float("nan")
    return float(np.nanquantile(values[finite], q))
