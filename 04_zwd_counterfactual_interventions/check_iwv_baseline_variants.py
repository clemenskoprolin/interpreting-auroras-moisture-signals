"""
check_iwv_baseline_variants.py — offline (no-GPU, no-Aurora) comparison of
IWV baselines used to build the `qhat` counterfactual ZWD field.

Question: the Stage-A statistic is  S(zwd_true) - S(qhat),  where
    qhat = a * IWV + b
and IWV is currently a 13-level trapezoid integral of q over the Aurora
pressure levels (conditional_data.compute_iwv). That integral includes
levels that lie *below ground* over high terrain, i.e. exactly where the
reported mountain effect is largest. This script asks whether swapping in
a better IWV baseline would move qhat enough to matter.

Three baselines are compared, all fed through the identical global OLS
(fit_spatial_qhat) and the identical +-4 sigma clip:

    iwv13        13-level trapezoid, no surface mask   (current behaviour)
    iwv13_sp     same, but levels with p > surface_pressure are dropped
    tcwv         ERA5 native total_column_water_vapour (137-level)

The decision metric is

    ratio = RMS(qhat_variant - qhat_iwv13) / RMS(zwd_true - qhat_iwv13)

evaluated inside the q850 target box. The denominator is the residual that
*is* the Stage-A effect. ratio << 1 means the baseline swap is small
compared with the signal and the published Stage-A numbers stand; ratio
approaching or exceeding 1 means the counterfactual itself would change
materially and Aurora has to be re-run.

Pure xarray/numpy; does not import torch or Aurora.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_tasks import TARGETS  # noqa: E402  (numpy-only module)

G = 9.80665
AURORA_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
ZWD_ZARR_PATH = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)
_FALLBACK_WB2_PATHS = (
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
WB2_PATHS = tuple(
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_FALLBACK_WB2_PATHS)
    ).split(os.pathsep) if path
)
NORM_STATS_PATH = os.environ.get(
    "AURORA_NORMALIZATION_STATS",
    os.path.join(_ROOT, "config", "normalization_stats_1979_2021.json"),
)

# Terrain groups, matching the Stage-A table in README.md.
GROUPS = {
    "andes": "major_mtn", "himalayas": "major_mtn", "rockies": "major_mtn",
    "cascades_sierra": "major_mtn", "caucasus": "major_mtn",
    "ticino": "mid_mtn", "alps_east": "mid_mtn", "atlas": "mid_mtn",
    "valais": "mid_mtn", "pyrenees": "mid_mtn", "new_zealand_alps": "mid_mtn",
    "ganges_plain": "mtn_adjacent_flat", "sahara_plain": "mtn_adjacent_flat",
    "great_plains": "mtn_adjacent_flat",
    "california": "flat", "pampas": "flat", "pacific_nw_coast": "flat",
    "aquitaine_basin": "flat", "canterbury_plain": "flat", "netherlands": "flat",
    "japan": "flat", "caspian_lowland": "flat",
}
DEFAULT_TARGETS = [
    # major mountain (largest reported effect)
    "andes", "himalayas", "rockies", "cascades_sierra", "caucasus",
    # mid mountain
    "ticino", "valais",
    # mountain-adjacent flat
    "ganges_plain",
    # flat controls (near-zero / sign-unstable effect)
    "netherlands", "japan",
]


# --- baselines ---------------------------------------------------------------

def compute_iwv(q_L_H_W: np.ndarray, levels: tuple, sp_hw: np.ndarray | None = None) -> np.ndarray:
    """Column water vapour (kg/m^2) by trapezoid integration over `levels`.

    Verbatim port of conditional_data.compute_iwv, plus an optional surface
    mask: when `sp_hw` (Pa) is given, sub-surface levels are excluded by
    clamping the integration bounds to the local surface pressure.
    """
    p = np.array(levels, dtype=np.float64) * 100.0  # Pa
    sort_idx = np.argsort(p)[::-1]                  # surface -> top
    p_s = p[sort_idx]
    q_s = q_L_H_W[sort_idx].astype(np.float64)

    if sp_hw is None:
        dp = np.abs(np.diff(p_s))
        q_mid = 0.5 * (q_s[:-1] + q_s[1:])
        return (np.sum(q_mid * dp[:, None, None], axis=0) / G).astype(np.float32)

    # Surface-aware: clamp every level to <= sp, so layers fully below ground
    # collapse to zero thickness and the lowest straddling layer is truncated.
    sp = sp_hw.astype(np.float64)[None, :, :]
    p_eff = np.minimum(p_s[:, None, None], sp)
    dp = np.abs(np.diff(p_eff, axis=0))
    q_mid = 0.5 * (q_s[:-1] + q_s[1:])
    return (np.sum(q_mid * dp, axis=0) / G).astype(np.float32)


def fit_spatial_qhat(zwd_hw: np.ndarray, iwv_hw: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Global OLS ZWD ~ a*IWV + b. Port of conditional_data.fit_spatial_qhat, plus R^2."""
    zwd_flat = zwd_hw.ravel().astype(np.float64)
    iwv_flat = iwv_hw.ravel().astype(np.float64)
    valid = np.isfinite(zwd_flat) & np.isfinite(iwv_flat) & (iwv_flat >= 0.0)

    if valid.sum() < 1000:
        return 0.0, float(np.nanmean(zwd_hw)), zwd_hw.copy(), float("nan")

    X = np.stack([iwv_flat[valid], np.ones(valid.sum())], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, zwd_flat[valid], rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    pred = a * iwv_flat[valid] + b
    ss_res = float(np.sum((zwd_flat[valid] - pred) ** 2))
    ss_tot = float(np.sum((zwd_flat[valid] - zwd_flat[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return a, b, (a * iwv_hw + b).astype(np.float32), r2


# --- data ---------------------------------------------------------------------

REQUIRED_VARS = ("specific_humidity", "total_column_water_vapour", "surface_pressure")


def _open_store(ts: pd.Timestamp) -> xr.Dataset:
    """First WB2 store that covers `ts` AND carries every required variable.

    Note: weatherbench2_2024_2025.zarr has neither total_column_water_vapour
    nor surface_pressure, so 2024+ cases cannot be checked against native
    tcwv without sourcing that field elsewhere.
    """
    for path in WB2_PATHS:
        try:
            ds = xr.open_zarr(path)
        except Exception:
            continue
        if not (pd.Timestamp(ds.time.values[0]) <= ts <= pd.Timestamp(ds.time.values[-1])):
            continue
        missing = [v for v in REQUIRED_VARS if v not in ds.data_vars]
        if missing:
            raise KeyError(f"{ts}: store {os.path.basename(path)} lacks {missing}")
        try:
            return ds.sel(time=ts)
        except KeyError:
            continue
    raise FileNotFoundError(f"No WB2 store covers {ts}")


def load_timestep(ts: pd.Timestamp, ds_zwd: xr.Dataset) -> dict:
    s = _open_store(ts)
    q = s["specific_humidity"].sel(level=list(AURORA_LEVELS)).values.astype(np.float32)
    tcwv = s["total_column_water_vapour"].values.astype(np.float32)
    sp = s["surface_pressure"].values.astype(np.float32)
    zwd = ds_zwd["zenith_wet_delay"].sel(time=ts).values.astype(np.float32)
    return {
        "q": q, "tcwv": tcwv, "sp": sp, "zwd": zwd,
        "lat": s["latitude"].values, "lon": s["longitude"].values,
    }


def box_mask(lat: np.ndarray, lon: np.ndarray, target: str) -> np.ndarray:
    tr = TARGETS[target]
    lat_ok = (lat >= tr.box_lat[0]) & (lat <= tr.box_lat[1])
    lon360 = np.mod(lon, 360.0)
    w, e = np.mod(tr.box_lon[0], 360.0), np.mod(tr.box_lon[1], 360.0)
    lon_ok = (lon360 >= w) & (lon360 <= e) if w <= e else (lon360 >= w) | (lon360 <= e)
    return lat_ok[:, None] & lon_ok[None, :]


def _rms(x: np.ndarray, m: np.ndarray | None = None) -> float:
    v = x[m] if m is not None else x
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


# --- main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=[
        os.path.join(_HERE, "cases_stratified96.json"),
        os.path.join(_HERE, "cases_stratified_extension84.json"),
        os.path.join(_HERE, "cases_stratified_flat_extension48.json"),
    ])
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--n-per-target", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "iwv_baseline_variants.csv"))
    args = ap.parse_args()

    cases = []
    for path in args.cases:
        with open(path) as f:
            cases.extend(json.load(f))

    # 2024+ cases have no native tcwv in the WB2 stores — skip them here.
    n_2024 = sum(1 for c in cases if pd.Timestamp(c["init_time"]).year >= 2024)
    cases = [c for c in cases if pd.Timestamp(c["init_time"]).year < 2024]
    if n_2024:
        print(f"skipping {n_2024} case(s) from 2024+: no tcwv/surface_pressure "
              f"in weatherbench2_2024_2025.zarr")

    selected = []
    for tgt in args.targets:
        hits = [c for c in cases if c["target"] == tgt][: args.n_per_target]
        selected.extend(hits)
    print(f"{len(selected)} cases over {len(args.targets)} targets "
          f"({2 * len(selected)} timesteps)\n", flush=True)

    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)
    zwd_loc = float(stats["locations"]["zwd"])
    zwd_scale = float(stats["scales"]["zwd"])
    lo, hi = zwd_loc - 4 * zwd_scale, zwd_loc + 4 * zwd_scale

    ds_zwd = xr.open_zarr(ZWD_ZARR_PATH)
    rows = []

    for ci, c in enumerate(selected, 1):
        tgt = c["target"]
        t1 = pd.Timestamp(datetime.fromisoformat(c["init_time"]))
        t0 = t1 - timedelta(hours=6)
        print(f"[{ci}/{len(selected)}] {tgt} {t1.date()} {t1.hour:02d}Z", flush=True)

        for ti, ts in enumerate((t0, t1)):
            d = load_timestep(ts, ds_zwd)
            zwd = d["zwd"]
            bm = box_mask(d["lat"], d["lon"], tgt)

            iwv13 = compute_iwv(d["q"], AURORA_LEVELS)
            iwv_sp = compute_iwv(d["q"], AURORA_LEVELS, sp_hw=d["sp"])
            tcwv = d["tcwv"]

            qhats, r2s, coefs = {}, {}, {}
            for name, field in (("iwv13", iwv13), ("iwv13_sp", iwv_sp), ("tcwv", tcwv)):
                a, b, qh, r2 = fit_spatial_qhat(zwd, field)
                qhats[name] = np.clip(qh, lo, hi)
                r2s[name] = r2
                coefs[name] = (a, b)

            base = qhats["iwv13"]
            resid = zwd - base   # the field whose effect Stage A measures

            row = {
                "case_id": f"{tgt}_{t1.strftime('%Y%m%d%H')}_{c.get('role', '?')}",
                "target": tgt, "group": GROUPS.get(tgt, "?"),
                "timestep": f"t{ti}", "time": str(ts),
                "r2_iwv13": r2s["iwv13"], "r2_iwv13_sp": r2s["iwv13_sp"], "r2_tcwv": r2s["tcwv"],
                "a_iwv13": coefs["iwv13"][0], "a_tcwv": coefs["tcwv"][0],
                "rms_resid_box": _rms(resid, bm),
                "rms_resid_global": _rms(resid),
                "iwv13_minus_tcwv_box": float(np.mean((iwv13 - tcwv)[bm])),
                "iwv13_sp_minus_tcwv_box": float(np.mean((iwv_sp - tcwv)[bm])),
            }
            for name in ("iwv13_sp", "tcwv"):
                diff = qhats[name] - base
                row[f"rms_dqhat_{name}_box"] = _rms(diff, bm)
                row[f"rms_dqhat_{name}_global"] = _rms(diff)
                row[f"ratio_{name}_box"] = _rms(diff, bm) / row["rms_resid_box"]
                row[f"ratio_{name}_global"] = _rms(diff) / row["rms_resid_global"]
                # The residual that a Stage-A rerun with this baseline would
                # actually probe: shrinkage < 1 means less "signal beyond IWV"
                # survives, i.e. part of the published effect was integral error.
                resid_v = zwd - qhats[name]
                row[f"rms_resid_{name}_box"] = _rms(resid_v, bm)
                row[f"shrink_{name}_box"] = _rms(resid_v, bm) / row["rms_resid_box"]
                row[f"shrink_{name}_global"] = _rms(resid_v) / row["rms_resid_global"]
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== per-group means (box = inside the q850 target box) ===")
    agg = df.groupby("group").agg({
        "r2_iwv13": "mean", "r2_iwv13_sp": "mean", "r2_tcwv": "mean",
        "rms_resid_box": "mean", "rms_resid_iwv13_sp_box": "mean", "rms_resid_tcwv_box": "mean",
        "shrink_iwv13_sp_box": "mean", "shrink_tcwv_box": "mean",
        "ratio_iwv13_sp_box": "mean", "ratio_tcwv_box": "mean",
        "iwv13_minus_tcwv_box": "mean", "iwv13_sp_minus_tcwv_box": "mean",
    }).round(4)
    print(agg.to_string())

    print("\n=== per-target means ===")
    print(df.groupby(["group", "target"]).agg({
        "rms_resid_box": "mean", "rms_resid_tcwv_box": "mean", "shrink_tcwv_box": "mean",
        "shrink_iwv13_sp_box": "mean", "ratio_tcwv_box": "mean",
        "r2_iwv13": "mean", "r2_tcwv": "mean",
    }).round(4).to_string())

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
