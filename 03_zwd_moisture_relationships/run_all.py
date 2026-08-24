#!/usr/bin/env python3
"""
Convenience runner for the ZWD correlation diagnostics.

It runs:
  1. build_timeseries.py
  2. correlate_timeseries.py
"""

from __future__ import annotations

import argparse
import os

import build_timeseries
import correlate_timeseries
from common import (
    AURORA_LEVELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WB_PATHS,
    DEFAULT_ZWD_PATH,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--targets", nargs="+", default=["ticino", "california", "japan"])
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--end", default=None)
    p.add_argument("--candidate-hours", type=int, nargs="+", default=[0, 12])
    p.add_argument("--regions", nargs="+", default=["box", "disk"],
                   choices=["box", "disk", "global"])
    p.add_argument("--disk-radius-km", type=float, default=1000.0)
    p.add_argument("--levels", type=int, nargs="+", default=list(AURORA_LEVELS))
    p.add_argument("--atmos-vars", nargs="+", default=["q", "t"])
    p.add_argument("--surface-vars", nargs="+", default=["2t", "msl"])
    p.add_argument("--no-column", action="store_false", dest="include_column", default=True)
    p.add_argument("--no-wind", action="store_false", dest="include_wind", default=True)

    p.add_argument("--lag-hours", type=int, nargs="+", default=[0, 12, 24])
    p.add_argument("--transforms", nargs="+", default=["raw", "monthly_z"],
                   choices=["raw", "monthly_z"])
    p.add_argument("--zwd-columns", nargs="+", default=None)
    p.add_argument("--min-n", type=int, default=12)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--no-plots", action="store_true")

    p.add_argument("--zwd-path", default=DEFAULT_ZWD_PATH)
    p.add_argument("--wb-paths", nargs="+", default=list(DEFAULT_WB_PATHS))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    build_timeseries.run(args)

    corr_args = argparse.Namespace(
        input_dir=os.path.join(args.output_dir, "timeseries"),
        output_dir=args.output_dir,
        zwd_columns=args.zwd_columns,
        lag_hours=args.lag_hours,
        transforms=args.transforms,
        min_n=args.min_n,
        top_k=args.top_k,
        no_plots=args.no_plots,
    )
    correlate_timeseries.run(corr_args)

def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
