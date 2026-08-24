"""
select_precipitation_cases.py — Case selection for 07_zwd_precipitation_model_comparison.

Standalone script (no Aurora import) that reads MSWEP and selects
precipitation events per target region.

Scoring: score = z(p_box_mean) + 0.5 * z(p_disk_p90)
  where z = z-score across all candidate times for the target.

Selection: 4 events per target, at least 30 days apart (fallback: 14 days).

Usage:
    source .venv/bin/activate
    python 07_zwd_precipitation_model_comparison/select_precipitation_cases.py \\
        --output cases_precipitation.json \\
        --year 2020 \\
        --n-per-target 4 \\
        --min-sep-days 30 \\
        --fallback-sep-days 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring — no Aurora, use venv311-xarray
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_P_CORR_DIR = os.path.join(_ROOT, "06_precipitation_moisture_relationships")

for _p in (_P_CORR_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_mswep():
    """Import MSWEPReader and DEFAULT_PRECIP_PATH from 06_precipitation_moisture_relationships."""
    from common import MSWEPReader, DEFAULT_PRECIP_PATH
    return MSWEPReader, DEFAULT_PRECIP_PATH


def _import_targets():
    """Import TARGETS from searchlight_tasks (no Aurora)."""
    _SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
    if _SEARCHLIGHT_DIR not in sys.path:
        sys.path.insert(0, _SEARCHLIGHT_DIR)
    from searchlight_tasks import TARGETS, EARTH_RADIUS_KM, great_circle_km
    return TARGETS, EARTH_RADIUS_KM, great_circle_km


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_precip_mean(
    precip: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    box_lat: tuple[float, float],
    box_lon: tuple[float, float],
) -> float:
    """Cosine-lat-weighted box mean of a (H, W) precipitation array."""
    lo_s, lo_n = box_lat
    lo_w, lo_e = box_lon

    if lat_vals[0] > lat_vals[-1]:
        lat_mask = (lat_vals <= lo_n) & (lat_vals >= lo_s)
    else:
        lat_mask = (lat_vals >= lo_s) & (lat_vals <= lo_n)

    # Normalize longitude to 0..360
    lon_norm = lon_vals % 360.0
    if lo_w <= lo_e:
        lon_mask = (lon_norm >= lo_w) & (lon_norm <= lo_e)
    else:
        # Wrapping case
        lon_mask = (lon_norm >= lo_w) | (lon_norm <= lo_e)

    sub = precip[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
    sub_lat = lat_vals[lat_mask]
    cos_w = np.cos(np.radians(sub_lat)).clip(min=0.0)[:, None]
    total_w = cos_w.sum() * sub.shape[1]
    if total_w == 0.0 or sub.size == 0:
        return float("nan")
    return float((sub * cos_w).sum() / total_w)


def _disk_p90(
    precip: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    disk_km: float = 1000.0,
) -> float:
    """90th percentile of precipitation within a disk of radius disk_km."""
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    # Use simple lat-lon approximation for speed
    dlat = lat_grid - center_lat
    dlon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    dlon_adj = dlon * np.cos(np.radians(center_lat))
    dist = np.sqrt(dlat ** 2 + dlon_adj ** 2) * 111.0   # rough km

    sub = precip[dist <= disk_km]
    if sub.size == 0:
        return float("nan")
    return float(np.nanpercentile(sub, 90))


# ---------------------------------------------------------------------------
# Case selection algorithm
# ---------------------------------------------------------------------------

def _zscore(arr: np.ndarray) -> np.ndarray:
    """Element-wise z-score; returns zeros if std=0."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _greedy_diverse(
    times: pd.DatetimeIndex,
    scores: np.ndarray,
    n: int,
    min_sep_days: int,
) -> list[int]:
    """Greedy selection of n highest-scoring times with minimum separation.

    Returns list of indices into times (sorted by score descending).
    """
    order = np.argsort(-scores)
    selected_times: list[pd.Timestamp] = []
    selected_indices: list[int] = []

    for idx in order:
        t = times[idx]
        too_close = any(
            abs((t - s).days) < min_sep_days for s in selected_times
        )
        if not too_close:
            selected_times.append(t)
            selected_indices.append(int(idx))
        if len(selected_indices) >= n:
            break

    return selected_indices


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def _precompute_target_masks(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    target,
    disk_km: float,
) -> dict:
    """Precompute spatial masks and indices for a target region (once per target).

    Returns dict with:
      - disk_lat_idx, disk_lon_idx: bounding-box indices for reading from zarr
      - box_lat_idx, box_lon_idx: lat/lon indices for box mean (relative to disk bbox)
      - disk_mask: boolean (len_disk_lat, len_disk_lon) within disk_km
      - cos_w_box: cosine-lat weights for box mean, shape (n_box_lat, 1)
    """
    # ---- Disk bounding box ----
    dlat_deg = disk_km / 111.0
    dlon_deg = disk_km / (111.0 * max(np.cos(np.radians(target.center_lat)), 0.05))

    lat_min_d = target.center_lat - dlat_deg
    lat_max_d = target.center_lat + dlat_deg
    lon_min_d = target.center_lon - dlon_deg
    lon_max_d = target.center_lon + dlon_deg

    if lat_vals[0] > lat_vals[-1]:  # descending
        disk_lat_idx = np.where((lat_vals <= lat_max_d) & (lat_vals >= lat_min_d))[0]
    else:
        disk_lat_idx = np.where((lat_vals >= lat_min_d) & (lat_vals <= lat_max_d))[0]

    lon_norm = lon_vals % 360.0
    lon_min_d_n = lon_min_d % 360.0
    lon_max_d_n = lon_max_d % 360.0
    if lon_min_d_n <= lon_max_d_n:
        disk_lon_idx = np.where((lon_norm >= lon_min_d_n) & (lon_norm <= lon_max_d_n))[0]
    else:
        disk_lon_idx = np.where((lon_norm >= lon_min_d_n) | (lon_norm <= lon_max_d_n))[0]

    if disk_lat_idx.size == 0 or disk_lon_idx.size == 0:
        return None

    sub_lat = lat_vals[disk_lat_idx]
    sub_lon = lon_vals[disk_lon_idx]

    # ---- Disk mask within bounding box ----
    lat_g, lon_g = np.meshgrid(sub_lat, sub_lon, indexing="ij")
    dlat = lat_g - target.center_lat
    dlon = ((lon_g - target.center_lon + 180.0) % 360.0) - 180.0
    dlon_adj = dlon * np.cos(np.radians(target.center_lat))
    dist = np.sqrt(dlat ** 2 + dlon_adj ** 2) * 111.0
    disk_mask = dist <= disk_km  # (n_lat, n_lon) bool

    # ---- Box mask within bounding box (for box mean) ----
    lo_s, lo_n = target.box_lat
    lo_w, lo_e = target.box_lon
    if lat_vals[0] > lat_vals[-1]:
        box_lat_mask_sub = (sub_lat <= lo_n) & (sub_lat >= lo_s)
    else:
        box_lat_mask_sub = (sub_lat >= lo_s) & (sub_lat <= lo_n)

    sub_lon_norm = sub_lon % 360.0
    lo_w_n = lo_w % 360.0
    lo_e_n = lo_e % 360.0
    if lo_w_n <= lo_e_n:
        box_lon_mask_sub = (sub_lon_norm >= lo_w_n) & (sub_lon_norm <= lo_e_n)
    else:
        box_lon_mask_sub = (sub_lon_norm >= lo_w_n) | (sub_lon_norm <= lo_e_n)

    box_lat_idx_sub = np.where(box_lat_mask_sub)[0]
    box_lon_idx_sub = np.where(box_lon_mask_sub)[0]

    # Cosine-lat weights for box
    cos_w_box = None
    if box_lat_idx_sub.size > 0:
        cos_w_box = np.cos(np.radians(sub_lat[box_lat_idx_sub])).clip(min=0.0)[:, None]

    return {
        "disk_lat_idx": disk_lat_idx,
        "disk_lon_idx": disk_lon_idx,
        "disk_mask": disk_mask,
        "box_lat_idx_sub": box_lat_idx_sub,
        "box_lon_idx_sub": box_lon_idx_sub,
        "cos_w_box": cos_w_box,
        "sub_lat": sub_lat,
    }


def _fast_box_mean(sub_precip: np.ndarray, masks: dict) -> float:
    """Compute cosine-weighted box mean from a sub-region array."""
    bi = masks["box_lat_idx_sub"]
    bj = masks["box_lon_idx_sub"]
    cos_w = masks["cos_w_box"]
    if bi.size == 0 or bj.size == 0 or cos_w is None:
        return float("nan")
    box = sub_precip[np.ix_(bi, bj)]
    total_w = cos_w.sum() * box.shape[1]
    if total_w < 1e-12:
        return float("nan")
    return float((box * cos_w).sum() / total_w)


def _fast_disk_p90(sub_precip: np.ndarray, masks: dict) -> float:
    """Compute p90 within disk mask from a sub-region array."""
    vals = sub_precip[masks["disk_mask"]]
    if vals.size == 0:
        return float("nan")
    return float(np.nanpercentile(vals, 90))


def select_cases(
    year: int = 2020,
    n_per_target: int = 4,
    min_sep_days: int = 30,
    fallback_sep_days: int = 14,
    candidate_hours: tuple[int, ...] = (0, 6, 12, 18),
    disk_km: float = 1000.0,
    mswep_store_path: str | None = None,
) -> list[dict]:
    """Run the full case selection pipeline.

    Returns:
        List of dicts with keys:
          target, init_time (ISO str), score, p_box_mean, p_disk_p90,
          min_days_sep, fallback_used
    """
    TARGETS, EARTH_RADIUS_KM, great_circle_km = _import_targets()
    MSWEPReader, DEFAULT_PRECIP_PATH = _import_mswep()

    store_path = mswep_store_path or DEFAULT_PRECIP_PATH
    print(f"Opening MSWEP store: {store_path}")
    reader = MSWEPReader(store_path)

    lat_vals = reader.latitude
    lon_vals = reader.longitude

    # Filter to candidate year
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    t_indices = reader.selected_time_indices(start, end, candidate_hours)
    candidate_times = reader.time[t_indices]
    print(f"  {len(t_indices)} candidate times in {year} at hours {candidate_hours}")

    results = []

    for target_key, target in TARGETS.items():
        print(f"\nProcessing target: {target_key} ({target.name})", flush=True)

        # Precompute masks once per target (avoids 1461x meshgrid recreation)
        masks = _precompute_target_masks(lat_vals, lon_vals, target, disk_km)
        if masks is None:
            print(f"  [{target_key}] WARNING: empty disk mask, skipping")
            continue

        n_disk_lat = masks["disk_lat_idx"].size
        n_disk_lon = masks["disk_lon_idx"].size
        print(f"  [{target_key}] disk bbox: {n_disk_lat}×{n_disk_lon} grid points", flush=True)

        box_means = []
        disk_p90s = []

        for i, tidx in enumerate(t_indices):
            if i % 300 == 0:
                print(f"  [{target_key}] {i}/{len(t_indices)} ...", flush=True)
            # Read only the disk bounding-box region — far fewer zarr chunk reads
            sub = reader.read_subset(
                int(tidx),
                lat_indices=masks["disk_lat_idx"],
                lon_indices=masks["disk_lon_idx"],
            )
            if sub is None:
                box_means.append(float("nan"))
                disk_p90s.append(float("nan"))
                continue
            box_means.append(_fast_box_mean(sub, masks))
            disk_p90s.append(_fast_disk_p90(sub, masks))

        box_means = np.asarray(box_means, dtype=np.float64)
        disk_p90s = np.asarray(disk_p90s, dtype=np.float64)

        box_means = np.asarray(box_means, dtype=np.float64)
        disk_p90s = np.asarray(disk_p90s, dtype=np.float64)

        # z-score and combine
        z_box = _zscore(box_means)
        z_p90 = _zscore(disk_p90s)
        scores = z_box + 0.5 * z_p90

        # Greedy selection with min_sep_days
        selected = _greedy_diverse(candidate_times, scores, n_per_target, min_sep_days)
        fallback_used = False

        if len(selected) < n_per_target:
            print(f"  [{target_key}] Only {len(selected)}/{n_per_target} found "
                  f"with {min_sep_days}d sep, retrying with {fallback_sep_days}d")
            selected = _greedy_diverse(candidate_times, scores, n_per_target, fallback_sep_days)
            fallback_used = len(selected) < n_per_target or True  # flag it either way
            if len(selected) < n_per_target:
                print(f"  WARNING: [{target_key}] Only {len(selected)}/{n_per_target} "
                      f"found even with {fallback_sep_days}d sep")

        print(f"  [{target_key}] Selected {len(selected)} cases:")
        for idx in selected:
            t = candidate_times[idx]
            sc = float(scores[idx])
            print(f"    {t.isoformat()}  score={sc:.3f}  "
                  f"box_mean={box_means[idx]:.3f}  p90={disk_p90s[idx]:.3f}")
            results.append({
                "target": target_key,
                "init_time": t.isoformat(),
                "score": round(sc, 4),
                "p_box_mean": round(float(box_means[idx]), 4),
                "p_disk_p90": round(float(disk_p90s[idx]), 4),
                "min_days_sep": min_sep_days if not fallback_used else fallback_sep_days,
                "fallback_used": fallback_used,
                "role": "strong",
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Select precipitation events for 07_zwd_precipitation_model_comparison"
    )
    p.add_argument("--output", type=str, default="cases_precipitation.json",
                   help="Output JSON file path")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--n-per-target", type=int, default=4)
    p.add_argument("--min-sep-days", type=int, default=30)
    p.add_argument("--fallback-sep-days", type=int, default=14)
    p.add_argument("--disk-km", type=float, default=1000.0,
                   help="Disk radius for computing p90 score (km)")
    p.add_argument("--mswep-store", type=str, default=None,
                   help="Path to MSWEP zarr store (default from common.py)")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of target names to process (default: all 22)")
    return p.parse_args()


def main():
    args = parse_args()

    cases = select_cases(
        year=args.year,
        n_per_target=args.n_per_target,
        min_sep_days=args.min_sep_days,
        fallback_sep_days=args.fallback_sep_days,
        mswep_store_path=args.mswep_store,
        disk_km=args.disk_km,
    )

    # Filter to requested targets if specified
    if args.targets:
        target_set = set(args.targets)
        cases = [c for c in cases if c["target"] in target_set]

    out_path = args.output
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"\nWrote {len(cases)} cases to {out_path}")

    # Summary table
    from collections import defaultdict
    per_target = defaultdict(int)
    for c in cases:
        per_target[c["target"]] += 1
    for t, n in sorted(per_target.items()):
        print(f"  {t}: {n} cases")


if __name__ == "__main__":
    main()
