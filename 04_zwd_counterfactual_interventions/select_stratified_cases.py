"""
select_stratified_cases.py — moisture-stratified case selection for the ZWD
conditional-mechanism experiment.

Every case set built before 2026-07-22 was drawn from the top of

    score = z(zwd_box_mean) + 0.75*z(q850_box_mean) + 0.50*z(zwd_disk_p90)

which samples only the moist, high-ZWD tail. Two problems with that for the
present question:

  * it selects on ZWD, i.e. conditions on the very variable the counterfactual
    perturbs, and
  * it uses q850, which over high terrain is largely below ground.

This script instead stratifies on ERA5 native total_column_water_vapour, an
independent 137-level product that is neither perturbed nor terrain-degraded.
Strata are defined on the MONTHLY percentile of box-mean TCWV, so "humid" means
humid for that target in that month rather than merely tropical or summer:

    low       10-30
    typical   40-60
    humid     70-90
    extreme   95-99

Precipitation is recorded as a SECOND label (not a stratum): humid and heavily
precipitating are related but not equivalent, so the analysis can cross the two
into dry/non-raining, humid/non-raining, humid/moderate-rain, extreme-rain.

For each selected case the corrected ZWD residual amplitude and signed mean are
recorded too, so the intended regression

    delta ~ residual_mean + residual_variability + tcwv_pct + tcwv_pct:residual

can distinguish:
  1. ZWD matters only because moist cases carry larger perturbations,
  2. Aurora is more sensitive per mm of residual in moist events,
  3. sensitivity is flat but the residual itself grows in extremes.

CPU/data-only; does not import torch or Aurora.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "02_zwd_attribution_benchmark"))

from searchlight_tasks import TARGETS  # noqa: E402
from check_iwv_baseline_variants import (  # noqa: E402
    AURORA_LEVELS, NORM_STATS_PATH, ZWD_ZARR_PATH,
    compute_iwv, fit_spatial_qhat, _open_store, WB2_PATHS,
)

STRATA = {
    "low": (10.0, 30.0),
    "typical": (40.0, 60.0),
    "humid": (70.0, 90.0),
    "extreme": (95.0, 99.0),
}


def box_mask(lat, lon, target):
    tr = TARGETS[target]
    la = (lat >= tr.box_lat[0]) & (lat <= tr.box_lat[1])
    l3 = np.mod(lon, 360.0)
    w, e = np.mod(tr.box_lon[0], 360.0), np.mod(tr.box_lon[1], 360.0)
    lo = (l3 >= w) & (l3 <= e) if w <= e else (l3 >= w) | (l3 <= e)
    return la, lo


def series_for_target(target: str, start: str, end: str, hours=(0, 12)) -> pd.DataFrame:
    """Box-mean TCWV and total precipitation over the period (surface fields only)."""
    pieces = []
    for path in WB2_PATHS:
        try:
            ds = xr.open_zarr(path)
        except Exception:
            continue
        if "total_column_water_vapour" not in ds.data_vars:
            continue
        t = pd.to_datetime(ds.time.values)
        sel = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & np.isin(t.hour, hours)
        if not sel.any():
            continue
        sub = ds.isel(time=np.where(sel)[0])
        la, lo = box_mask(sub.latitude.values, sub.longitude.values, target)
        coslat = np.cos(np.deg2rad(sub.latitude.values[la]))
        wts = xr.DataArray(coslat, dims=["latitude"])

        tc = sub["total_column_water_vapour"].isel(
            latitude=np.where(la)[0], longitude=np.where(lo)[0])
        tc = tc.weighted(wts).mean(dim=("latitude", "longitude")).compute().to_series()
        df = pd.DataFrame({"tcwv": tc})

        if "total_precipitation" in sub.data_vars:
            tp = sub["total_precipitation"].isel(
                latitude=np.where(la)[0], longitude=np.where(lo)[0])
            df["precip"] = tp.weighted(wts).mean(
                dim=("latitude", "longitude")).compute().to_series()
        else:
            df["precip"] = np.nan
        pieces.append(df)
        print(f"  [{target}] {os.path.basename(path)}: {len(df)} steps", flush=True)

    if not pieces:
        raise RuntimeError(f"no TCWV series for {target}")
    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated()]
    # Monthly percentile: humid FOR THIS TARGET IN THIS MONTH.
    out["month"] = out.index.month
    out["tcwv_pct"] = out.groupby("month")["tcwv"].rank(pct=True) * 100.0
    out["precip_pct"] = out.groupby("month")["precip"].rank(pct=True) * 100.0
    return out


def series_for_targets(
    targets: list[str],
    start: str,
    end: str,
    hours=(0, 12),
    include_precip: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load several target series in one shared dask graph per WB2 store.

    The WB2 time chunks are much larger than these target boxes. Computing each
    target independently can therefore reread the same chunks many times. A
    single Dataset graph lets dask share those reads across all target means.
    """
    pieces: dict[str, list[pd.DataFrame]] = {target: [] for target in targets}
    for path in WB2_PATHS:
        try:
            ds = xr.open_zarr(path)
        except Exception:
            continue
        if "total_column_water_vapour" not in ds.data_vars:
            continue
        t = pd.to_datetime(ds.time.values)
        sel = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & np.isin(t.hour, hours)
        if not sel.any():
            continue
        sub = ds.isel(time=np.where(sel)[0])

        means: dict[str, xr.DataArray] = {}
        for target in targets:
            la, lo = box_mask(sub.latitude.values, sub.longitude.values, target)
            coslat = np.cos(np.deg2rad(sub.latitude.values[la]))
            wts = xr.DataArray(coslat, dims=["latitude"])
            indexers = {
                "latitude": np.where(la)[0],
                "longitude": np.where(lo)[0],
            }
            tc = sub["total_column_water_vapour"].isel(**indexers)
            means[f"{target}__tcwv"] = tc.weighted(wts).mean(
                dim=("latitude", "longitude"))
            if include_precip and "total_precipitation" in sub.data_vars:
                tp = sub["total_precipitation"].isel(**indexers)
                means[f"{target}__precip"] = tp.weighted(wts).mean(
                    dim=("latitude", "longitude"))

        computed = xr.Dataset(means).compute()
        for target in targets:
            tc = computed[f"{target}__tcwv"].to_series()
            frame = pd.DataFrame({"tcwv": tc})
            precip_key = f"{target}__precip"
            frame["precip"] = (
                computed[precip_key].to_series() if precip_key in computed
                else np.nan
            )
            pieces[target].append(frame)
        print(f"  [batched] {os.path.basename(path)}: "
              f"{computed.sizes.get('time', 0)} steps x {len(targets)} targets",
              flush=True)

    out: dict[str, pd.DataFrame] = {}
    for target, target_pieces in pieces.items():
        if not target_pieces:
            raise RuntimeError(f"no TCWV series for {target}")
        frame = pd.concat(target_pieces).sort_index()
        frame = frame[~frame.index.duplicated()]
        frame["month"] = frame.index.month
        frame["tcwv_pct"] = frame.groupby("month")["tcwv"].rank(pct=True) * 100.0
        frame["precip_pct"] = frame.groupby("month")["precip"].rank(pct=True) * 100.0
        out[target] = frame
    return out


def _series_for_target_worker(
    work: tuple[str, str, str, tuple[int, ...]],
) -> tuple[str, pd.DataFrame]:
    """Process-pool wrapper for the I/O-bound multi-target selection pass."""
    target, start, end, hours = work
    return target, series_for_target(target, start, end, hours)


def pick_spaced(idx: pd.DatetimeIndex, n: int, min_gap_days: int) -> list[pd.Timestamp]:
    """Greedy spread: take candidates in order, enforcing a minimum separation."""
    chosen: list[pd.Timestamp] = []
    for ts in idx:
        if all(abs((ts - c).days) >= min_gap_days for c in chosen):
            chosen.append(ts)
        if len(chosen) == n:
            break
    return chosen


_CLIP: tuple[float, float] | None = None
_ZWD_DS: xr.Dataset | None = None


def _zwd_clip() -> tuple[float, float]:
    global _CLIP
    if _CLIP is None:
        with open(NORM_STATS_PATH) as f:
            st = json.load(f)
        loc, scale = float(st["locations"]["zwd"]), float(st["scales"]["zwd"])
        _CLIP = (loc - 4 * scale, loc + 4 * scale)
    return _CLIP


def residual_stats(ts: pd.Timestamp, target: str) -> dict:
    """Corrected (sp-masked) ZWD residual amplitude and signed mean in the box."""
    global _ZWD_DS
    lo_c, hi_c = _zwd_clip()
    if _ZWD_DS is None:
        _ZWD_DS = xr.open_zarr(ZWD_ZARR_PATH)

    s = _open_store(ts)
    q = s["specific_humidity"].sel(level=list(AURORA_LEVELS)).values.astype(np.float32)
    sp = s["surface_pressure"].values.astype(np.float32)
    zwd = _ZWD_DS["zenith_wet_delay"].sel(time=ts).values.astype(np.float32)

    iwv = compute_iwv(q, AURORA_LEVELS, sp_hw=sp)
    # NB check_iwv_baseline_variants.fit_spatial_qhat returns (a, b, qhat, r2);
    # the conditional_data version returns only three values.
    _, _, qhat, _ = fit_spatial_qhat(zwd, iwv)
    r = zwd - np.clip(qhat, lo_c, hi_c)

    la, lo = box_mask(s.latitude.values, s.longitude.values, target)
    rb = r[np.ix_(la, lo)]
    return {
        "resid_rms": float(np.sqrt(np.mean(rb ** 2))),
        "resid_mean": float(np.mean(rb)),
        "resid_std": float(np.std(rb)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2023-12-31T23:00")
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--min-gap-days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1,
                    help="Target-series worker processes (default: 1).")
    ap.add_argument("--batch-targets", action="store_true",
                    help="Compute all target boxes in one shared graph per store.")
    ap.add_argument("--skip-precip", action="store_true",
                    help="Do not load the optional four-year precipitation label.")
    ap.add_argument("--out-cases", default=os.path.join(_HERE, "cases_stratified.json"))
    ap.add_argument("--out-meta", default=os.path.join(_HERE, "results", "cases_stratified_meta.csv"))
    ap.add_argument("--skip-residuals", action="store_true",
                    help="Skip the per-case residual pass (much faster, but the "
                         "regression then has no residual covariates).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cases, meta = [], []

    work = [(target, args.start, args.end, (0, 12)) for target in args.targets]
    n_workers = min(max(args.workers, 1), len(work))
    if args.batch_targets:
        print(f"Loading {len(work)} target series in shared store graphs...",
              flush=True)
        series = series_for_targets(
            args.targets, args.start, args.end,
            include_precip=not args.skip_precip,
        )
    elif n_workers > 1:
        print(f"Loading {len(work)} target series with {n_workers} workers...",
              flush=True)
        with multiprocessing.Pool(processes=n_workers) as pool:
            series = dict(pool.map(_series_for_target_worker, work))
    else:
        series = {
            target: series_for_target(target, args.start, args.end)
            for target in args.targets
        }

    for target in args.targets:
        print(f"\n=== {target} ===", flush=True)
        s = series[target]
        for stratum, (p_lo, p_hi) in STRATA.items():
            band = s[(s.tcwv_pct >= p_lo) & (s.tcwv_pct <= p_hi)]
            if band.empty:
                print(f"  [{target}] {stratum}: EMPTY", flush=True)
                continue
            # Shuffle so the pick is not biased toward the start of the record,
            # then enforce spacing.
            order = rng.permutation(len(band))
            shuffled = band.iloc[order]
            picks = pick_spaced(shuffled.index, args.per_stratum, args.min_gap_days)
            for ts in picks:
                row = s.loc[ts]
                cases.append({
                    "target": target,
                    "init_time": pd.Timestamp(ts).isoformat(),
                    "role": stratum,
                })
                meta.append({
                    "target": target, "init_time": pd.Timestamp(ts).isoformat(),
                    "stratum": stratum, "tcwv": float(row.tcwv),
                    "tcwv_pct": float(row.tcwv_pct), "precip": float(row.precip),
                    "precip_pct": float(row.precip_pct), "month": int(row.month),
                })
            print(f"  [{target}] {stratum}: {len(picks)} cases "
                  f"(TCWV {band.tcwv.min():.1f}-{band.tcwv.max():.1f} kg/m2)", flush=True)

    df = pd.DataFrame(meta)
    if not args.skip_residuals:
        print(f"\nComputing ZWD residual stats for {len(df)} cases "
              f"(both input timesteps)...", flush=True)
        stats = []
        for i, row in enumerate(df.itertuples(), 1):
            t1 = pd.Timestamp(row.init_time)
            t0 = t1 - pd.Timedelta(hours=6)
            try:
                a, b = residual_stats(t0, row.target), residual_stats(t1, row.target)
                stats.append({k: 0.5 * (a[k] + b[k]) for k in a})
            except Exception as e:
                print(f"  [{i}] {row.target} {t1} FAILED {type(e).__name__}", flush=True)
                stats.append({"resid_rms": np.nan, "resid_mean": np.nan, "resid_std": np.nan})
            if i % 10 == 0:
                print(f"  {i}/{len(df)}", flush=True)
        df = pd.concat([df, pd.DataFrame(stats)], axis=1)

    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    df.to_csv(args.out_meta, index=False)
    with open(args.out_cases, "w") as f:
        json.dump(cases, f, indent=1)

    print(f"\n{len(cases)} cases over {len(args.targets)} targets")
    pd.set_option("display.width", 200)
    cols = ["tcwv", "tcwv_pct", "precip_pct"] + (
        [] if args.skip_residuals else ["resid_rms", "resid_mean"])
    print(df.groupby("stratum")[cols].mean().round(3).to_string())
    print(f"\nwrote {args.out_cases} and {args.out_meta}")


if __name__ == "__main__":
    main()
