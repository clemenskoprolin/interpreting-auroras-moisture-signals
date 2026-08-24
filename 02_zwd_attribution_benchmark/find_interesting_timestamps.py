"""
Find candidate benchmark timestamps for the ZWD searchlight benchmark.

This script performs a cheap, data-only prescreen over the multi-year ZWD and
WeatherBench2 archives and proposes two timestamps per target region:

    - one "strong" moisture event
    - one "secondary" still-relevant event

Selection is based on a month-normalized composite score:

    score =
        zscore_monthly(zwd_box_mean)
      + 0.75 * zscore_monthly(q850_box_mean)
      + 0.50 * zscore_monthly(zwd_disk_p90)

where:
    zwd_box_mean  = area-weighted mean ZWD inside the target box
    q850_box_mean = area-weighted mean ERA5 q at 850 hPa inside the target box
    zwd_disk_p90  = 90th percentile ZWD inside a 1000 km disk around the
                    target-box center

The output is:
    1. a benchmark-ready cases JSON manifest
    2. one CSV per target with the ranked candidate timestamps
    3. one JSON summary with the selected cases and top-k tables

Example:
    python 02_zwd_attribution_benchmark/find_interesting_timestamps.py \
        --targets ticino california japan \
        --start 2020-01-01T00:00:00 \
        --output-dir results/searchlight_case_selection
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from searchlight_tasks import TARGETS, great_circle_km  # noqa: E402


_FALLBACK_WB_PATHS = (
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
DEFAULT_ZWD_PATH = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)
DEFAULT_WB_PATHS = tuple(
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_FALLBACK_WB_PATHS)
    ).split(os.pathsep) if path
)
DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "results", "searchlight_case_selection")


@dataclass(frozen=True)
class TargetStats:
    target: str
    data: pd.DataFrame


def parse_args():
    p = argparse.ArgumentParser(
        description="Find interesting timestamps for the ZWD searchlight benchmark."
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=["ticino", "california", "japan"],
        help="Subset of target short names.",
    )
    p.add_argument(
        "--start",
        type=str,
        default="2020-01-01T00:00:00",
        help="Earliest timestamp to consider (ISO format).",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional latest timestamp to consider (ISO format).",
    )
    p.add_argument(
        "--candidate-hours",
        type=int,
        nargs="+",
        default=[0, 12],
        help="Only score timestamps whose hour is in this set.",
    )
    p.add_argument(
        "--min-gap-days",
        type=int,
        default=30,
        help="Minimum spacing between selected timestamps for the same target.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many top-ranked candidates to save per target.",
    )
    p.add_argument(
        "--cases-per-target",
        type=int,
        default=2,
        help=(
            "How many ranked cases to write to cases.json per target. "
            "The default 2 preserves the strong/secondary selection; "
            "larger values use the top ranked timestamps subject to --min-gap-days."
        ),
    )
    p.add_argument(
        "--select",
        choices=("top", "bottom", "median"),
        default="top",
        help=(
            "Which end of the composite-score ranking to draw cases from. "
            "'top' (default) = moist, high-ZWD extremes; 'bottom' = dry/low-ZWD; "
            "'median' = typical conditions. Use non-default values to build "
            "contrast sets against the moisture-biased default selection."
        ),
    )
    p.add_argument(
        "--disk-radius-km",
        type=float,
        default=1000.0,
        help="Radius of the broader ZWD disk feature.",
    )
    p.add_argument(
        "--zwd-path",
        type=str,
        default=DEFAULT_ZWD_PATH,
        help="Path to the ZWD zarr store.",
    )
    p.add_argument(
        "--wb-paths",
        nargs="+",
        default=list(DEFAULT_WB_PATHS),
        help="One or more WeatherBench2 zarr stores containing specific_humidity.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/CSV outputs.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers for processing targets (default: 3).",
    )
    return p.parse_args()


def _norm_lon_360(values):
    arr = np.asarray(values)
    return np.mod(arr, 360.0)


def _subset_lon_mask(lon_vals, west, east):
    lon_vals = _norm_lon_360(lon_vals)
    west = west % 360.0
    east = east % 360.0
    if west <= east:
        return (lon_vals >= west) & (lon_vals <= east)
    return (lon_vals >= west) | (lon_vals <= east)


def _weighted_box_mean(da, lat_name, lon_name, box_lat, box_lon):
    lat_vals = da[lat_name].values
    lon_vals = _norm_lon_360(da[lon_name].values)

    lat_mask = (lat_vals >= box_lat[0]) & (lat_vals <= box_lat[1])
    lon_mask = _subset_lon_mask(lon_vals, box_lon[0], box_lon[1])

    sub = da.sel(
        {
            lat_name: da[lat_name].values[lat_mask],
            lon_name: da[lon_name].values[lon_mask],
        }
    )
    weights = xr.DataArray(
        np.cos(np.deg2rad(sub[lat_name].values)),
        dims=(lat_name,),
        coords={lat_name: sub[lat_name].values},
    )
    return sub.weighted(weights).mean(dim=(lat_name, lon_name))


def _disk_mask(lat_vals, lon_vals, center_lat, center_lon, radius_km):
    lat_grid, lon_grid = np.meshgrid(lat_vals, _norm_lon_360(lon_vals), indexing="ij")
    d = great_circle_km(center_lat, center_lon, lat_grid, lon_grid)
    return d <= radius_km


def _monthly_zscore(series):
    series = pd.Series(series)
    groups = series.groupby(series.index.month)

    def _z(x):
        std = float(x.std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
        return (x - float(x.mean())) / std

    return groups.transform(_z)


def _filter_times(ds, time_dim, start_ts, end_ts, candidate_hours):
    """Pre-filter a dataset's time dimension before any spatial computation."""
    times = pd.DatetimeIndex(ds[time_dim].values)
    mask = times >= start_ts
    if end_ts is not None:
        mask &= times <= end_ts
    mask &= times.hour.isin(candidate_hours)
    return ds.isel({time_dim: np.where(mask)[0]})


def _load_q850_series_for_target(wb_paths, target, start_ts, end_ts, candidate_hours):
    pieces = []
    for path in wb_paths:
        if not os.path.exists(path):
            continue
        print(f"  [q850] Opening WB2: {path}", flush=True)
        t0 = time.time()
        ds = xr.open_zarr(path)
        if "specific_humidity" not in ds:
            print(f"  [q850] No specific_humidity in {path}, skipping", flush=True)
            continue

        # Filter time FIRST — avoids loading the full multi-decade series
        print(f"  [q850] Filtering time range ({start_ts} onward, hours={candidate_hours})...", flush=True)
        ds_filt = _filter_times(ds, "time", start_ts, end_ts, candidate_hours)
        n_times = ds_filt.sizes["time"]
        print(f"  [q850] {n_times} timestamps after filter. Selecting level=850 and box mean...", flush=True)

        q850 = ds_filt["specific_humidity"].sel(level=850)
        q850_box = _weighted_box_mean(
            q850, "latitude", "longitude", target.box_lat, target.box_lon
        )

        print(f"  [q850] Computing ({n_times} steps)...", flush=True)
        s = q850_box.compute().to_series()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        pieces.append(s.rename("q850_box_mean"))
        print(f"  [q850] Done ({len(s)} timesteps) in {time.time()-t0:.0f}s", flush=True)

    if not pieces:
        raise RuntimeError("No WeatherBench2 series could be loaded for q850.")

    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _load_zwd_features_for_target(zwd_path, target, start_ts, end_ts, candidate_hours, disk_radius_km):
    if not os.path.exists(zwd_path):
        raise RuntimeError("ZWD path does not exist: %s" % zwd_path)

    print(f"  [zwd] Opening ZWD zarr: {zwd_path}", flush=True)
    t0 = time.time()
    ds = xr.open_zarr(zwd_path)
    var = "zenith_wet_delay"
    if var not in ds:
        raise RuntimeError("ZWD store does not contain %r" % var)

    # Filter time FIRST
    print(f"  [zwd] Filtering time range...", flush=True)
    ds_filt = _filter_times(ds, "time", start_ts, end_ts, candidate_hours)
    n_times = ds_filt.sizes["time"]
    print(f"  [zwd] {n_times} timestamps after filter.", flush=True)

    zwd = ds_filt[var]

    print(f"  [zwd] Computing box mean ({n_times} steps)...", flush=True)
    zwd_box = _weighted_box_mean(
        zwd, "latitude", "longitude", target.box_lat, target.box_lon
    )

    lat_vals = ds[var]["latitude"].values
    lon_vals = ds[var]["longitude"].values
    print(f"  [zwd] Computing disk mask (r={disk_radius_km}km)...", flush=True)
    region_mask = _disk_mask(
        lat_vals, lon_vals, target.center_lat, target.center_lon, disk_radius_km
    )
    region_mask_da = xr.DataArray(
        region_mask,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_vals, "longitude": ds[var]["longitude"].values},
    )
    print(f"  [zwd] Computing disk p90 ({n_times} steps)...", flush=True)
    # xarray's dask-backed quantile needs core dimensions to live in a single
    # chunk. Keep time chunked, but combine the spatial dimensions per block.
    zwd_masked = zwd.where(region_mask_da).chunk({"latitude": -1, "longitude": -1})
    zwd_disk_p90 = zwd_masked.quantile(
        0.9, dim=("latitude", "longitude"), skipna=True
    )

    print(f"  [zwd] Loading to memory...", flush=True)
    box_s = zwd_box.compute().to_series()
    p90_s = zwd_disk_p90.compute().to_series()

    if not isinstance(box_s.index, pd.DatetimeIndex):
        box_s.index = pd.to_datetime(box_s.index)
    if not isinstance(p90_s.index, pd.DatetimeIndex):
        p90_s.index = pd.to_datetime(p90_s.index)

    df = pd.concat(
        [box_s.rename("zwd_box_mean"), p90_s.rename("zwd_disk_p90")],
        axis=1,
    ).sort_index()
    print(f"  [zwd] Done ({len(df)} timesteps) in {time.time()-t0:.0f}s", flush=True)
    return df


def build_target_table(args, target_key):
    target = TARGETS[target_key]
    t_total = time.time()
    print(f"\n[{target_key}] Starting (pid={os.getpid()})", flush=True)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end else None

    print(f"[{target_key}] Loading q850...", flush=True)
    q_series = _load_q850_series_for_target(
        args.wb_paths, target, start_ts, end_ts, set(args.candidate_hours)
    )
    print(f"[{target_key}] q850 done: {len(q_series)} rows", flush=True)

    print(f"[{target_key}] Loading ZWD features...", flush=True)
    zwd_df = _load_zwd_features_for_target(
        args.zwd_path, target, start_ts, end_ts, set(args.candidate_hours),
        args.disk_radius_km,
    )
    print(f"[{target_key}] ZWD done: {len(zwd_df)} rows", flush=True)

    df = pd.concat([zwd_df, q_series], axis=1, join="inner").dropna()
    if df.empty:
        raise RuntimeError("No common timestamps found for target %s" % target_key)

    df["month"] = df.index.month
    df["z_zwd_box_mean"] = _monthly_zscore(df["zwd_box_mean"])
    df["z_q850_box_mean"] = _monthly_zscore(df["q850_box_mean"])
    df["z_zwd_disk_p90"] = _monthly_zscore(df["zwd_disk_p90"])
    df["score"] = (
        df["z_zwd_box_mean"]
        + 0.75 * df["z_q850_box_mean"]
        + 0.50 * df["z_zwd_disk_p90"]
    )

    df = df.sort_values("score", ascending=False)
    elapsed = time.time() - t_total
    print(f"[{target_key}] DONE in {elapsed:.0f}s — top score={df['score'].iloc[0]:.3f} at {df.index[0]}", flush=True)
    return TargetStats(target=target_key, data=df)


def _build_target_table_worker(args_and_key):
    """Worker function for multiprocessing."""
    args, target_key = args_and_key
    return build_target_table(args, target_key)


def _pick_cases(df, min_gap_days):
    selected = []
    min_gap = pd.Timedelta(days=min_gap_days)

    for ts, row in df.iterrows():
        if not selected:
            selected.append((ts, row, "strong"))
            continue
        strong_ts = selected[0][0]
        if abs(ts - strong_ts) < min_gap:
            continue
        if ts.month == strong_ts.month:
            continue
        selected.append((ts, row, "secondary"))
        break

    return selected


def _pick_ranked_cases(df, n_cases, min_gap_days):
    """Pick up to n_cases high-scoring timestamps with temporal separation."""
    selected = []
    min_gap = pd.Timedelta(days=min_gap_days)

    for ts, row in df.iterrows():
        if any(abs(ts - prev_ts) < min_gap for prev_ts, _, _ in selected):
            continue
        idx = len(selected) + 1
        if idx == 1:
            role = "strong"
        elif idx == 2:
            role = "secondary"
        else:
            role = "top%02d" % idx
        selected.append((ts, row, role))
        if len(selected) >= n_cases:
            break

    return selected


def _write_target_csv(output_dir, target_key, df):
    path = os.path.join(output_dir, "%s_candidates.csv" % target_key)
    cols = [
        "zwd_box_mean",
        "q850_box_mean",
        "zwd_disk_p90",
        "z_zwd_box_mean",
        "z_q850_box_mean",
        "z_zwd_disk_p90",
        "score",
    ]
    out = df[cols].copy()
    out.index.name = "init_time"
    out.to_csv(path)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t_main = time.time()
    print(f"Starting find_interesting_timestamps (pid={os.getpid()})", flush=True)
    print(f"Targets: {args.targets}", flush=True)
    print(f"Start: {args.start}, candidate hours: {args.candidate_hours}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    for target_key in args.targets:
        if target_key not in TARGETS:
            raise ValueError("Unknown target %r" % target_key)

    n_workers = min(args.workers, len(args.targets))

    if n_workers > 1:
        print(f"\nRunning {n_workers} targets in parallel...", flush=True)
        with multiprocessing.Pool(processes=n_workers) as pool:
            target_tables = pool.map(
                _build_target_table_worker,
                [(args, k) for k in args.targets],
            )
    else:
        target_tables = [build_target_table(args, k) for k in args.targets]

    # Draw from the requested end of the composite-score ranking. The default
    # "top" selects moist, high-ZWD extremes; every case set built before
    # 2026-07-22 used it, which makes those sets a moisture-biased sample.
    # "bottom" / "median" give dry and typical contrast sets.
    if args.select != "top":
        reordered = []
        for table in target_tables:
            d = table.data
            if args.select == "bottom":
                d = d.sort_values("score", ascending=True)
            else:  # median: nearest the median composite score
                d = d.reindex(d["score"].sub(d["score"].median()).abs()
                              .sort_values().index)
            reordered.append(TargetStats(target=table.target, data=d))
        target_tables = reordered

    # Write CSVs
    for table in target_tables:
        _write_target_csv(args.output_dir, table.target, table.data.head(args.top_k))

    cases = []
    summary = {"targets": {}, "selection_rules": {
        "start": args.start,
        "end": args.end,
        "candidate_hours": args.candidate_hours,
        "min_gap_days": args.min_gap_days,
        "cases_per_target": args.cases_per_target,
        "disk_radius_km": args.disk_radius_km,
        "score_formula": (
            "z_zwd_box_mean + 0.75*z_q850_box_mean + 0.50*z_zwd_disk_p90"
        ),
    }}

    for table in target_tables:
        if args.cases_per_target <= 2:
            picked = _pick_cases(table.data, args.min_gap_days)
        else:
            picked = _pick_ranked_cases(
                table.data,
                min(args.cases_per_target, args.top_k),
                args.min_gap_days,
            )
        if len(picked) < min(args.cases_per_target, 2):
            raise RuntimeError(
                "Could not find enough cases for target %s (got %d, requested %d)"
                % (table.target, len(picked), args.cases_per_target)
            )

        top_rows = []
        for ts, row in table.data.head(args.top_k).iterrows():
            top_rows.append({
                "init_time": ts.isoformat(),
                "score": float(row["score"]),
                "zwd_box_mean": float(row["zwd_box_mean"]),
                "q850_box_mean": float(row["q850_box_mean"]),
                "zwd_disk_p90": float(row["zwd_disk_p90"]),
            })

        chosen = []
        for ts, row, role in picked:
            cases.append({
                "target": table.target,
                "init_time": ts.isoformat(),
                "role": role,
            })
            chosen.append({
                "role": role,
                "init_time": ts.isoformat(),
                "score": float(row["score"]),
                "zwd_box_mean": float(row["zwd_box_mean"]),
                "q850_box_mean": float(row["q850_box_mean"]),
                "zwd_disk_p90": float(row["zwd_disk_p90"]),
            })

        summary["targets"][table.target] = {
            "selected": chosen,
            "top_candidates": top_rows,
        }

    cases_path = os.path.join(args.output_dir, "cases.json")
    with open(cases_path, "w") as f:
        json.dump(cases, f, indent=2)

    summary_path = os.path.join(args.output_dir, "selection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_main
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    print("\nWrote:")
    print("  ", cases_path)
    print("  ", summary_path)
    for target_key in args.targets:
        print("  ", os.path.join(args.output_dir, "%s_candidates.csv" % target_key))

    # Print summary
    print("\n=== Selected Cases ===")
    for target_key, info in summary["targets"].items():
        for c in info["selected"]:
            print(f"  {target_key} [{c['role']}]: {c['init_time']}  score={c['score']:.3f}  zwd_box={c['zwd_box_mean']:.4f}  q850={c['q850_box_mean']:.6f}")


if __name__ == "__main__":
    main()
