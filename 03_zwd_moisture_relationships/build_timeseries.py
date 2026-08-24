#!/usr/bin/env python3
"""
Build paired ZWD/proxy time series for target regions.

This is the data extraction stage. It reads ZWD from the ZWDX Zarr store and
proxy variables from WeatherBench2, then writes one CSV per target region.
The downstream script `correlate_timeseries.py` consumes these CSVs.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from common import (
    AURORA_LEVELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WB_PATHS,
    DEFAULT_ZWD_PATH,
    TARGETS,
    WB2_ATMOS,
    WB2_SURF,
    add_time_columns,
    ensure_dir,
    filter_times,
    low_level_mean,
    pressure_weighted_mean,
    region_quantile,
    save_json,
    weighted_region_mean,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=["ticino", "california", "japan"])
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--end", default=None)
    p.add_argument(
        "--candidate-hours",
        type=int,
        nargs="+",
        default=[0, 12],
        help="UTC hours to keep. Use 0 6 12 18 for 6-hour lag diagnostics.",
    )
    p.add_argument("--regions", nargs="+", default=["box", "disk"],
                   choices=["box", "disk", "global"])
    p.add_argument("--disk-radius-km", type=float, default=1000.0)
    p.add_argument("--levels", type=int, nargs="+", default=list(AURORA_LEVELS))
    p.add_argument(
        "--atmos-vars",
        nargs="+",
        default=["q", "t"],
        choices=sorted(WB2_ATMOS),
        help="Atmospheric proxy variables to extract by level.",
    )
    p.add_argument(
        "--surface-vars",
        nargs="+",
        default=["2t", "msl"],
        choices=sorted(WB2_SURF),
        help="Surface proxy variables to extract.",
    )
    p.add_argument(
        "--include-column",
        action="store_true",
        default=True,
        help="Also compute pressure-weighted column and low-level means for q/t.",
    )
    p.add_argument(
        "--no-column",
        action="store_false",
        dest="include_column",
        help="Disable column and low-level aggregate proxies.",
    )
    p.add_argument(
        "--include-wind",
        action="store_true",
        default=True,
        help="Compute wind speed proxies from u/v when both are available.",
    )
    p.add_argument(
        "--no-wind",
        action="store_false",
        dest="include_wind",
        help="Disable derived wind speed proxies.",
    )
    p.add_argument("--zwd-path", default=DEFAULT_ZWD_PATH)
    p.add_argument("--wb-paths", nargs="+", default=list(DEFAULT_WB_PATHS))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _to_frame(features: dict[str, xr.DataArray]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    ds = xr.Dataset(features)
    df = ds.compute().to_dataframe()
    # xarray produces a MultiIndex when the Dataset retains extra coordinates
    # (e.g. a leftover "level" coord from a sel). Flatten it back to a plain
    # time index.
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if "time" not in df:
            raise RuntimeError("Expected a time column after xarray conversion.")
        df = df.set_index(pd.to_datetime(df["time"])).drop(columns=["time"])
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _feature_name(var: str, region: str, *, level: int | None = None, suffix: str = "mean") -> str:
    parts = [var]
    if level is not None:
        parts.append(str(level))
    parts.extend([region, suffix])
    return "_".join(parts)


def load_zwd_features(args: argparse.Namespace, target_key: str) -> pd.DataFrame:
    if not os.path.exists(args.zwd_path):
        raise RuntimeError(f"ZWD path does not exist: {args.zwd_path}")

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end else None
    candidate_hours = set(args.candidate_hours) if args.candidate_hours else None

    print(f"  [zwd] Opening {args.zwd_path}", flush=True)
    ds = xr.open_zarr(args.zwd_path)
    var = "zenith_wet_delay"
    if var not in ds:
        raise RuntimeError(f"ZWD store does not contain {var!r}")

    ds = filter_times(ds, "time", start_ts, end_ts, candidate_hours)
    if ds.sizes.get("time", 0) == 0:
        raise RuntimeError("No ZWD timestamps remain after filtering.")

    zwd = ds[var]
    features: dict[str, xr.DataArray] = {}
    for region in args.regions:
        features[f"zwd_{region}_mean"] = weighted_region_mean(
            zwd, "latitude", "longitude", target_key, region, args.disk_radius_km
        )
        if region in {"disk", "global"}:
            features[f"zwd_{region}_p90"] = region_quantile(
                zwd, "latitude", "longitude", target_key, region, 0.90, args.disk_radius_km
            )

    df = _to_frame(features)
    print(f"  [zwd] {len(df)} rows, {len(df.columns)} features", flush=True)
    return df


def _available_levels(ds: xr.Dataset, requested: Iterable[int]) -> list[int]:
    if "level" not in ds.coords:
        return []
    have = {int(x) for x in ds.level.values}
    return [int(x) for x in requested if int(x) in have]


def _add_region_features(
    features: dict[str, xr.DataArray],
    da: xr.DataArray,
    var_name: str,
    target_key: str,
    regions: list[str],
    disk_radius_km: float,
    *,
    level: int | None = None,
    suffix: str = "mean",
) -> None:
    for region in regions:
        name = _feature_name(var_name, region, level=level, suffix=suffix)
        features[name] = weighted_region_mean(
            da, "latitude", "longitude", target_key, region, disk_radius_km
        ).drop_vars("level", errors="ignore")


def load_wb_features(args: argparse.Namespace, target_key: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end else None
    candidate_hours = set(args.candidate_hours) if args.candidate_hours else None
    pieces: list[pd.DataFrame] = []

    for path in args.wb_paths:
        if not os.path.exists(path):
            print(f"  [wb2] Missing, skipping: {path}", flush=True)
            continue

        print(f"  [wb2] Opening {path}", flush=True)
        ds = xr.open_zarr(path)
        ds = filter_times(ds, "time", start_ts, end_ts, candidate_hours)
        n_times = ds.sizes.get("time", 0)
        if n_times == 0:
            print("  [wb2] No timestamps in requested range, skipping", flush=True)
            continue

        levels = _available_levels(ds, args.levels)
        features: dict[str, xr.DataArray] = {}

        for short in args.atmos_vars:
            long = WB2_ATMOS[short]
            if long not in ds or not levels:
                print(f"  [wb2] Missing atmos var {long!r}, skipping", flush=True)
                continue

            da_all = ds[long].sel(level=levels)
            for level in levels:
                _add_region_features(
                    features,
                    da_all.sel(level=level),
                    short,
                    target_key,
                    args.regions,
                    args.disk_radius_km,
                    level=level,
                )

            if args.include_column and short in {"q", "t"}:
                _add_region_features(
                    features,
                    pressure_weighted_mean(da_all, levels),
                    f"{short}_column",
                    target_key,
                    args.regions,
                    args.disk_radius_km,
                )
                _add_region_features(
                    features,
                    low_level_mean(da_all, levels),
                    f"{short}_low",
                    target_key,
                    args.regions,
                    args.disk_radius_km,
                )

        if args.include_wind:
            u_name = WB2_ATMOS["u"]
            v_name = WB2_ATMOS["v"]
            if u_name in ds and v_name in ds and levels:
                u = ds[u_name].sel(level=levels)
                v = ds[v_name].sel(level=levels)
                wind = np.hypot(u, v)
                for level in levels:
                    _add_region_features(
                        features,
                        wind.sel(level=level),
                        "wind",
                        target_key,
                        args.regions,
                        args.disk_radius_km,
                        level=level,
                    )
                if args.include_column:
                    _add_region_features(
                        features,
                        low_level_mean(wind, levels),
                        "wind_low",
                        target_key,
                        args.regions,
                        args.disk_radius_km,
                    )
            else:
                print("  [wb2] u/v unavailable, skipping wind proxies", flush=True)

        for short in args.surface_vars:
            long = WB2_SURF[short]
            if long not in ds:
                print(f"  [wb2] Missing surface var {long!r}, skipping", flush=True)
                continue
            _add_region_features(
                features,
                ds[long],
                short,
                target_key,
                args.regions,
                args.disk_radius_km,
            )

        print(f"  [wb2] Computing {len(features)} features over {n_times} times...", flush=True)
        df = _to_frame(features)
        print(f"  [wb2] Done: {len(df)} rows", flush=True)
        pieces.append(df)

    if not pieces:
        raise RuntimeError("No WeatherBench2 features could be loaded.")

    out = pd.concat(pieces).sort_index()
    # The WB2 Zarr stores can overlap in time; keep the first occurrence.
    out = out[~out.index.duplicated(keep="first")]
    return out


def build_target_timeseries(args: argparse.Namespace, target_key: str) -> pd.DataFrame:
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target {target_key!r}")

    t0 = time.time()
    print(f"\n[{target_key}] Building time series", flush=True)
    zwd_df = load_zwd_features(args, target_key)
    wb_df = load_wb_features(args, target_key)

    df = pd.concat([zwd_df, wb_df], axis=1, join="inner").sort_index()
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
            "surface_vars": args.surface_vars,
            "include_column": args.include_column,
            "include_wind": args.include_wind,
            "zwd_path": args.zwd_path,
            "wb_paths": args.wb_paths,
        },
        os.path.join(out_dir, "timeseries_config.json"),
    )

    paths: list[str] = []
    all_frames: list[pd.DataFrame] = []
    for target_key in args.targets:
        df = build_target_timeseries(args, target_key)
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
