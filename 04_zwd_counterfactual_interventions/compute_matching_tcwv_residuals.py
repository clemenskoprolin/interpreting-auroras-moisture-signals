"""Compute the native-TCWV ZWD residual used by the TCWV intervention.

For every case, this reproduces ``make_qhat_tcwv_zwd`` without importing
Aurora or torch:

1. fit a global OLS relation ZWD = a * TCWV + b independently at t0 and t1;
2. clip the reconstructed ZWD to the model's +/-4-sigma input range;
3. calculate ZWD_true - ZWD_hat_TCWV inside the target box; and
4. average the signed box mean over the two input times.

The script is CPU/data-only and does not import Aurora.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from check_iwv_baseline_variants import (  # noqa: E402
    NORM_STATS_PATH,
    WB2_PATHS,
    ZWD_ZARR_PATH,
    box_mask,
    fit_spatial_qhat,
)


def _open_tcwv_stores() -> list[tuple[xr.Dataset, pd.Timestamp, pd.Timestamp]]:
    """Open each TCWV-bearing WB2 store once and record its time coverage."""
    stores = []
    for path in WB2_PATHS:
        ds = xr.open_zarr(path)
        if "total_column_water_vapour" not in ds.data_vars:
            continue
        stores.append(
            (ds, pd.Timestamp(ds.time.values[0]), pd.Timestamp(ds.time.values[-1]))
        )
    if not stores:
        raise RuntimeError("No WeatherBench2 store contains native TCWV")
    return stores


def _tcwv_at(
    stores: list[tuple[xr.Dataset, pd.Timestamp, pd.Timestamp]],
    ts: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TCWV and its latitude/longitude coordinates at one time."""
    for ds, start, end in stores:
        if start <= ts <= end:
            field = ds["total_column_water_vapour"].sel(time=ts)
            return (
                field.values.astype(np.float32),
                field.latitude.values,
                field.longitude.values,
            )
    raise FileNotFoundError(f"No native-TCWV store covers {ts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--effects",
        default=os.path.join(_HERE, "results", "stratified_decomposition_combined228.csv"),
        help="Combined intervention table defining the cases to process.",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_HERE, "results", "tcwv_residuals_228cases.csv"),
    )
    args = ap.parse_args()

    effects = pd.read_csv(args.effects)
    cases = effects.drop_duplicates("case_key").copy()
    if len(cases) != effects["case_key"].nunique():
        raise RuntimeError("Failed to construct one row per case")

    keep = ["target", "case_id", "case_key", "init_time", "stratum", "terrain"]
    missing = [column for column in keep if column not in cases]
    if missing:
        raise ValueError(f"Effects table lacks required columns: {missing}")
    cases = cases[keep].reset_index(drop=True)

    # Several targets may share an input time. Fit the global reconstruction
    # once per unique timestamp and then evaluate every corresponding box.
    events: dict[pd.Timestamp, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, case in cases.iterrows():
        t1 = pd.Timestamp(case["init_time"])
        events[t1 - pd.Timedelta(hours=6)].append((idx, "t0", case["target"]))
        events[t1].append((idx, "t1", case["target"]))

    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)
    zwd_loc = float(stats["locations"]["zwd"])
    zwd_scale = float(stats["scales"]["zwd"])
    lo, hi = zwd_loc - 4.0 * zwd_scale, zwd_loc + 4.0 * zwd_scale

    tcwv_stores = _open_tcwv_stores()
    zwd_ds = xr.open_zarr(ZWD_ZARR_PATH)
    values: dict[tuple[int, str], dict[str, float]] = {}
    box_masks: dict[tuple[str, int, int], np.ndarray] = {}

    for number, (ts, requests) in enumerate(sorted(events.items()), 1):
        tcwv, lat, lon = _tcwv_at(tcwv_stores, ts)
        zwd = (
            zwd_ds["zenith_wet_delay"].sel(time=ts).values.astype(np.float32)
        )
        a, b, qhat, _ = fit_spatial_qhat(zwd, tcwv)
        qhat = np.clip(qhat, lo, hi)
        residual = zwd - qhat

        for idx, timestep, target in requests:
            mask_key = (target, len(lat), len(lon))
            if mask_key not in box_masks:
                box_masks[mask_key] = box_mask(lat, lon, target)
            box_values = residual[box_masks[mask_key]]
            values[(idx, timestep)] = {
                "mean": float(np.mean(box_values)),
                "rms": float(np.sqrt(np.mean(box_values**2))),
                "a": a,
                "b": b,
            }

        if number % 25 == 0 or number == len(events):
            print(f"processed {number}/{len(events)} unique input times", flush=True)

    rows = []
    for idx, case in cases.iterrows():
        t0, t1 = values[(idx, "t0")], values[(idx, "t1")]
        row = case.to_dict()
        row.update(
            {
                "tcwv_resid_mean_t0": t0["mean"],
                "tcwv_resid_mean_t1": t1["mean"],
                "tcwv_resid_mean": 0.5 * (t0["mean"] + t1["mean"]),
                "tcwv_resid_rms_t0": t0["rms"],
                "tcwv_resid_rms_t1": t1["rms"],
                "tcwv_resid_rms": np.sqrt(0.5 * (t0["rms"] ** 2 + t1["rms"] ** 2)),
                "tcwv_slope_t0": t0["a"],
                "tcwv_slope_t1": t1["a"],
                "tcwv_intercept_t0": t0["b"],
                "tcwv_intercept_t1": t1["b"],
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} cases to {args.out}")


if __name__ == "__main__":
    main()
