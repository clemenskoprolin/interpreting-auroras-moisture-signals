#!/usr/bin/env python3
"""
Build paired precipitation/ZWD/humidity time series for target regions.

The precipitation feature is MSWEP 6-hour accumulated precipitation sampled
at 3-hour intervals. ZWD and humidity proxies are reused from the existing ZWD
correlation diagnostics so the resulting CSVs can be consumed by the P-focused
correlation stage.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from common import (
    AURORA_LEVELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRECIP_PATH,
    DEFAULT_PRECIP_VAR,
    DEFAULT_WB_PATHS,
    DEFAULT_ZWD_PATH,
    MSWEPReader,
    PRECIP_CHECKPOINT_CONTEXT,
    TARGETS,
    add_time_columns,
    ensure_dir,
    load_zwd_diag_module,
    masked_quantile,
    save_json,
    target_region_mask,
    weighted_masked_mean,
)


_ZWD_BUILD = load_zwd_diag_module("build_timeseries")
load_zwd_features = _ZWD_BUILD.load_zwd_features
load_wb_features = _ZWD_BUILD.load_wb_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=["ticino", "california", "japan"])
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--end", default="2020-12-31T00:00:00")
    p.add_argument(
        "--candidate-hours",
        type=int,
        nargs="+",
        default=[0, 12],
        help="UTC hours to keep. MSWEP itself is sampled every 3 hours.",
    )
    p.add_argument("--regions", nargs="+", default=["box", "disk"],
                   choices=["box", "disk", "global"])
    p.add_argument("--disk-radius-km", type=float, default=1000.0)
    p.add_argument("--levels", type=int, nargs="+", default=list(AURORA_LEVELS))
    p.add_argument(
        "--atmos-vars",
        nargs="+",
        default=["q"],
        choices=["q"],
        help="Humidity proxy variables to extract from WeatherBench2.",
    )
    p.add_argument(
        "--include-column",
        action="store_true",
        default=True,
        help="Also compute pressure-weighted and low-level humidity aggregates.",
    )
    p.add_argument(
        "--no-column",
        action="store_false",
        dest="include_column",
        help="Disable column and low-level humidity aggregate proxies.",
    )
    p.add_argument("--precip-path", default=DEFAULT_PRECIP_PATH)
    p.add_argument("--precip-var", default=DEFAULT_PRECIP_VAR)
    p.add_argument("--zwd-path", default=DEFAULT_ZWD_PATH)
    p.add_argument("--wb-paths", nargs="+", default=list(DEFAULT_WB_PATHS))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _feature_name(region: str, suffix: str) -> str:
    return f"p_{region}_{suffix}"


def _region_masks(
    reader: MSWEPReader,
    target_key: str,
    regions: list[str],
    disk_radius_km: float,
) -> dict[str, np.ndarray]:
    return {
        region: target_region_mask(
            reader.latitude,
            reader.longitude,
            target_key,
            region,
            disk_radius_km,
        )
        for region in regions
    }


def _subset_indices(masks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        union |= mask
    lat_has = union.any(axis=1)
    lon_has = union.any(axis=0)
    return np.where(lat_has)[0], np.where(lon_has)[0]


def load_precip_features(
    args: argparse.Namespace,
    target_key: str,
    reader: MSWEPReader | None = None,
) -> pd.DataFrame:
    reader = reader or MSWEPReader(args.precip_path, args.precip_var)
    time_indices = reader.selected_time_indices(args.start, args.end, args.candidate_hours)
    if len(time_indices) == 0:
        raise RuntimeError("No MSWEP timestamps remain after filtering.")

    masks = _region_masks(reader, target_key, args.regions, args.disk_radius_km)
    lat_idx, lon_idx = _subset_indices(masks)
    cropped_masks = {
        region: mask[np.ix_(lat_idx, lon_idx)]
        for region, mask in masks.items()
    }
    cropped_lats = reader.latitude[lat_idx]

    rows: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    print(
        f"  [mswep] Reading {len(time_indices)} timestamps from {args.precip_path}",
        flush=True,
    )
    for pos, time_idx in enumerate(time_indices):
        field = reader.read_subset(int(time_idx), lat_indices=lat_idx, lon_indices=lon_idx)
        row: dict[str, float] = {}
        for region, mask in cropped_masks.items():
            row[_feature_name(region, "mean")] = weighted_masked_mean(field, cropped_lats, mask)
            row[_feature_name(region, "p90")] = masked_quantile(field, mask, 0.90)
        rows.append(row)
        index.append(reader.time[int(time_idx)])
        if (pos + 1) % 100 == 0:
            print(f"  [mswep] {pos + 1}/{len(time_indices)} timestamps", flush=True)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index)).sort_index()
    print(f"  [mswep] {len(df)} rows, {len(df.columns)} features", flush=True)
    return df


def _humidity_args(args: argparse.Namespace) -> argparse.Namespace:
    include_column = bool(args.include_column and len(args.levels) >= 2)
    return argparse.Namespace(
        start=args.start,
        end=args.end,
        candidate_hours=args.candidate_hours,
        regions=args.regions,
        disk_radius_km=args.disk_radius_km,
        levels=args.levels,
        atmos_vars=args.atmos_vars,
        surface_vars=[],
        include_column=include_column,
        include_wind=False,
        wb_paths=args.wb_paths,
    )


def build_target_timeseries(
    args: argparse.Namespace,
    target_key: str,
    reader: MSWEPReader | None = None,
) -> pd.DataFrame:
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target {target_key!r}")

    t0 = time.time()
    print(f"\n[{target_key}] Building precipitation time series", flush=True)
    p_df = load_precip_features(args, target_key, reader=reader)
    zwd_df = load_zwd_features(args, target_key)
    q_df = load_wb_features(_humidity_args(args), target_key)
    coord_cols = ["level", "quantile"]
    zwd_df = zwd_df.drop(columns=[c for c in coord_cols if c in zwd_df.columns])
    q_df = q_df.drop(columns=[c for c in coord_cols if c in q_df.columns])

    df = pd.concat([p_df, zwd_df, q_df], axis=1, join="inner").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if df.empty:
        raise RuntimeError(f"No common timestamps found for target {target_key}")

    df.index.name = "init_time"
    df.insert(0, "target", target_key)
    df = add_time_columns(df)

    print(
        f"[{target_key}] {len(df)} common timestamps, {len(df.columns) - 4} numeric features "
        f"in {(time.time() - t0):.0f}s",
        flush=True,
    )
    return df


def run(args: argparse.Namespace) -> list[str]:
    out_dir = ensure_dir(args.output_dir)
    ts_dir = ensure_dir(os.path.join(out_dir, "timeseries"))

    save_json(
        {
            "targets": args.targets,
            "start": args.start,
            "end": args.end,
            "candidate_hours": args.candidate_hours,
            "regions": args.regions,
            "disk_radius_km": args.disk_radius_km,
            "levels": args.levels,
            "atmos_vars": args.atmos_vars,
            "include_column": args.include_column,
            "precip_path": args.precip_path,
            "precip_var": args.precip_var,
            "zwd_path": args.zwd_path,
            "wb_paths": args.wb_paths,
            "model_checkpoint_context": PRECIP_CHECKPOINT_CONTEXT,
            "note": "Data-only diagnostics; precipitation checkpoints are documented but not loaded.",
        },
        os.path.join(out_dir, "timeseries_config.json"),
    )

    reader = MSWEPReader(args.precip_path, args.precip_var)
    paths: list[str] = []
    all_frames: list[pd.DataFrame] = []
    for target_key in args.targets:
        df = build_target_timeseries(args, target_key, reader=reader)
        path = os.path.join(ts_dir, f"{target_key}_timeseries.csv")
        df.to_csv(path)
        print(f"Saved: {path}", flush=True)
        paths.append(path)
        all_frames.append(df)

    combined = pd.concat(all_frames).sort_index()
    combined_path = os.path.join(ts_dir, "all_targets_timeseries.csv")
    combined.to_csv(combined_path)
    print(f"Saved: {combined_path}", flush=True)
    paths.append(combined_path)
    return paths


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
