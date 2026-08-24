#!/usr/bin/env python3
"""Build precipitation-only timestamps needed for intermediate moisture leads.

The main paired time series uses 00/12 UTC moisture origins.  Precipitation at
06/18 UTC is therefore required to evaluate 6 h and 18 h moisture leads without
changing the set of moisture-origin times or reloading ZWD and humidity.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from build_timeseries import load_precip_features
from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRECIP_PATH,
    DEFAULT_PRECIP_VAR,
    MSWEPReader,
    ensure_dir,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["ticino", "california", "japan"])
    parser.add_argument("--start", default="2020-01-01T06:00:00")
    parser.add_argument("--end", default="2020-12-31T18:00:00")
    parser.add_argument("--candidate-hours", type=int, nargs="+", default=[6, 18])
    parser.add_argument("--regions", nargs="+", default=["box", "disk"])
    parser.add_argument("--disk-radius-km", type=float, default=1000.0)
    parser.add_argument("--precip-path", default=DEFAULT_PRECIP_PATH)
    parser.add_argument("--precip-var", default=DEFAULT_PRECIP_VAR)
    parser.add_argument(
        "--output",
        default=os.path.join(
            DEFAULT_OUTPUT_DIR,
            "timeseries",
            "supplemental_precip_6h_timeseries.csv",
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> str:
    reader = MSWEPReader(args.precip_path, args.precip_var)
    frames = []
    for target in args.targets:
        frame = load_precip_features(args, target, reader=reader)
        frame.insert(0, "target", target)
        frames.append(frame)

    combined = pd.concat(frames).sort_index()
    combined.index.name = "init_time"
    ensure_dir(os.path.dirname(args.output))
    combined.to_csv(args.output)
    print(f"Saved: {args.output} ({len(combined)} rows)")

    config_path = os.path.splitext(args.output)[0] + "_config.json"
    save_json(
        {
            "targets": args.targets,
            "start": args.start,
            "end": args.end,
            "candidate_hours": args.candidate_hours,
            "regions": args.regions,
            "disk_radius_km": args.disk_radius_km,
            "precip_path": args.precip_path,
            "precip_var": args.precip_var,
            "purpose": "Precipitation outcomes for 6 h and 18 h moisture leads.",
        },
        config_path,
    )
    return args.output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
