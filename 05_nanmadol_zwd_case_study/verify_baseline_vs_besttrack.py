"""Verify the Nanmadol baseline rollout against the IBTrACS best track.

CPU-only: reads the already-computed baseline track CSV and the curated
IBTrACS subset, and (optionally) the ERA5 analysis at init time to separate
Aurora's forecast error from the intensity error it inherits from ERA5.

Usage:
    source .venv/bin/activate
    python 05_nanmadol_zwd_case_study/verify_baseline_vs_besttrack.py [--era5]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

BEST_TRACK = os.path.join(_HERE, "data", "nanmadol_ibtracs.csv")
BASELINE = os.path.join(
    _REPO, "results", "zwd_tc_case_study",
    "nanmadol_2022091712_zwd_tc_case_study",
    "storm_centered_xai", "baseline_track.csv",
)
OUT_CSV = os.path.join(
    _REPO, "results", "zwd_tc_case_study",
    "nanmadol_2022091712_zwd_tc_case_study",
    "baseline_vs_besttrack.csv",
)
ERA5_ZARR = os.environ.get(
    "AURORA_ERA5_2022_2023",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
)
INIT_TIME = "2022-09-17T12:00"


def great_circle_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_best_track() -> pd.DataFrame:
    raw = pd.read_csv(BEST_TRACK, parse_dates=["ISO_TIME"])
    pres = raw["TOKYO_PRES"].where(raw["TOKYO_PRES"].notna(), raw["WMO_PRES"])
    wind = raw["TOKYO_WIND"].where(raw["TOKYO_WIND"].notna(), raw["WMO_WIND"])
    return pd.DataFrame({
        "valid_time": raw["ISO_TIME"],
        "lat": raw["LAT"].astype(float),
        "lon": raw["LON"].astype(float) % 360.0,
        "pres": pres.astype(float),
        "wind_kt": wind.astype(float),
    })


def era5_min_msl() -> float | None:
    try:
        import xarray as xr
    except ImportError:
        print("  (xarray unavailable — skipping ERA5 check)")
        return None
    ds = xr.open_zarr(ERA5_ZARR)
    name = next(c for c in ds.data_vars if "mean_sea" in c or c == "msl")
    da = ds[name].sel(time=INIT_TIME)
    lat = da.latitude.values
    sl = slice(32, 22) if lat[0] > lat[-1] else slice(22, 32)
    sub = da.sel(latitude=sl, longitude=slice(127, 137)).load()
    return float(sub.min()) / 100.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--era5", action="store_true",
                   help="Also read the ERA5 analysis at init (slow, needs zarr)")
    args = p.parse_args()

    bt = load_best_track()
    fc = pd.read_csv(BASELINE, parse_dates=["valid_time"])
    m = fc.merge(bt, on="valid_time", how="left")

    m["track_err_km"] = great_circle_km(m.center_lat, m.center_lon, m.lat, m.lon)
    m["pres_err_hpa"] = m.min_msl_hpa - m.pres
    m["obs_wind_ms"] = m.wind_kt * 0.514444
    m["wind_err_ms"] = m.max_wind10_ms - m.obs_wind_ms

    cols = ["lead_hours", "center_lat", "lat", "center_lon", "lon",
            "track_err_km", "min_msl_hpa", "pres", "pres_err_hpa",
            "max_wind10_ms", "obs_wind_ms", "wind_err_ms"]
    print(m[cols].round(1).to_string(index=False))

    valid = m.dropna(subset=["track_err_km"])
    early = valid[valid.lead_hours <= 36]
    late = valid[valid.lead_hours > 36]
    print(f"\nTrack error: <=36h mean {early.track_err_km.mean():.0f} km "
          f"(max {early.track_err_km.max():.0f}); "
          f">36h mean {late.track_err_km.mean():.0f} km "
          f"(max {late.track_err_km.max():.0f})")
    print(f"Pressure bias: mean {m.pres_err_hpa.mean():+.1f} hPa, "
          f"at +6h {m.pres_err_hpa.iloc[0]:+.1f} hPa")
    print(f"Wind bias: mean {m.wind_err_ms.mean():+.1f} m/s")

    if args.era5:
        e = era5_min_msl()
        if e is not None:
            obs0 = float(bt.loc[bt.valid_time == pd.Timestamp(INIT_TIME), "pres"].iloc[0])
            print(f"\nERA5 min MSL at init: {e:.1f} hPa vs best track {obs0:.1f} hPa "
                  f"({e - obs0:+.1f} hPa)")
            print(f"Aurora at +6h: {m.min_msl_hpa.iloc[0]:.1f} hPa "
                  f"({m.min_msl_hpa.iloc[0] - e:+.1f} hPa vs its own initial state)")

    m[cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
