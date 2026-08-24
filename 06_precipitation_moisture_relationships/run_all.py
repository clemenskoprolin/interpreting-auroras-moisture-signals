#!/usr/bin/env python3
"""
Convenience runner for the precipitation correlation diagnostics.

It runs:
  1. build_timeseries.py
  2. build_supplemental_precip_timeseries.py for 06/18 UTC outcomes
  3. correlate_timeseries.py at 0, 6, 12, 18, and 24 h lags
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import build_supplemental_precip_timeseries
import build_timeseries
import correlate_timeseries
from common import (
    AURORA_LEVELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRECIP_PATH,
    DEFAULT_PRECIP_VAR,
    DEFAULT_WB_PATHS,
    DEFAULT_ZWD_PATH,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--targets", nargs="+", default=["ticino", "california", "japan"])
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--end", default="2020-12-31T00:00:00")
    p.add_argument("--candidate-hours", type=int, nargs="+", default=[0, 12])
    p.add_argument("--regions", nargs="+", default=["box", "disk"],
                   choices=["box", "disk", "global"])
    p.add_argument("--disk-radius-km", type=float, default=1000.0)
    p.add_argument("--levels", type=int, nargs="+", default=list(AURORA_LEVELS))
    p.add_argument("--no-column", action="store_false", dest="include_column", default=True)

    p.add_argument("--lag-hours", type=int, nargs="+", default=[0, 6, 12, 18, 24])
    p.add_argument("--transforms", nargs="+", default=["raw", "monthly_z"],
                   choices=["raw", "monthly_z"])
    p.add_argument("--driver-columns", nargs="+", default=None)
    p.add_argument("--min-n", type=int, default=12)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--no-plots", action="store_true")

    p.add_argument("--precip-path", default=DEFAULT_PRECIP_PATH)
    p.add_argument("--precip-var", default=DEFAULT_PRECIP_VAR)
    p.add_argument("--zwd-path", default=DEFAULT_ZWD_PATH)
    p.add_argument("--wb-paths", nargs="+", default=list(DEFAULT_WB_PATHS))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    build_timeseries.run(argparse.Namespace(
        targets=args.targets,
        start=args.start,
        end=args.end,
        candidate_hours=args.candidate_hours,
        regions=args.regions,
        disk_radius_km=args.disk_radius_km,
        levels=args.levels,
        atmos_vars=["q"],
        include_column=args.include_column,
        precip_path=args.precip_path,
        precip_var=args.precip_var,
        zwd_path=args.zwd_path,
        wb_paths=args.wb_paths,
        output_dir=args.output_dir,
    ))

    supplemental_path = os.path.join(
        args.output_dir, "timeseries", "supplemental_precip_6h_timeseries.csv"
    )
    start = (pd.Timestamp(args.start) + pd.Timedelta(hours=6)).isoformat()
    end = (pd.Timestamp(args.end) + pd.Timedelta(hours=18)).isoformat()
    build_supplemental_precip_timeseries.run(argparse.Namespace(
        targets=args.targets,
        start=start,
        end=end,
        candidate_hours=[6, 18],
        regions=args.regions,
        disk_radius_km=args.disk_radius_km,
        precip_path=args.precip_path,
        precip_var=args.precip_var,
        output=supplemental_path,
    ))

    corr_args = argparse.Namespace(
        input_dir=os.path.join(args.output_dir, "timeseries"),
        output_dir=args.output_dir,
        driver_columns=args.driver_columns,
        lag_hours=args.lag_hours,
        transforms=args.transforms,
        min_n=args.min_n,
        top_k=args.top_k,
        no_plots=args.no_plots,
        supplemental_precip_file=supplemental_path,
    )
    correlate_timeseries.run(corr_args)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
