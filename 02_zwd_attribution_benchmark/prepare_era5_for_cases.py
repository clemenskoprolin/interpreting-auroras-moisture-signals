"""
Prepare dated ERA5 .nc files for searchlight benchmark cases.

Reads atmospheric and surface data from WeatherBench2 zarr stores and
saves era5_atmos_YYYY-MM-DD.nc and era5_surface_YYYY-MM-DD.nc for each
unique date implied by the cases in a cases.json manifest.

Handles the t0/t1 logic: for a case with init_time T, we need data at
T - 12h (t0) and T (t1) — two consecutive 12h-spaced snapshots.

The output files are compatible with searchlight_data.load_case():
    era5_atmos_YYYY-MM-DD.nc:   dimensions (time=2, level=13, lat, lon)
        variables: z, u, v, t, q
    era5_surface_YYYY-MM-DD.nc: dimensions (time=2, lat, lon)
        variables: t2m, u10, v10, msl

Usage:
    source .venv/bin/activate
    python 02_zwd_attribution_benchmark/prepare_era5_for_cases.py \\
        --cases 02_zwd_attribution_benchmark/cases_6h_eventdates.json \\
        --output-dir data

The script is safe to re-run; it skips dates whose files already exist.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

# Pressure levels Aurora expects
AURORA_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# WeatherBench2 stores (same as find_interesting_timestamps.py)
_FALLBACK_WB2_PATHS = [
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
]
WB2_PATHS = [
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_FALLBACK_WB2_PATHS)
    ).split(os.pathsep) if path
]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WB2 → ERA5 short-name mapping for atmospheric vars
ATMOS_MAP = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}

# WB2 → ERA5 short-name mapping for surface vars
SURF_MAP = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare dated ERA5 .nc files from WeatherBench2 for searchlight cases."
    )
    p.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Path to cases.json produced by find_interesting_timestamps.py.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("AURORA_ERA5_CACHE_DIR", os.path.join(_ROOT, "data")),
        help="Directory to write dated .nc files (same as searchlight_data.DATA_DIR).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip dates whose .nc files already exist (default: True).",
    )
    return p.parse_args()


def _open_wb2_stores() -> list[xr.Dataset]:
    stores = []
    for path in WB2_PATHS:
        if os.path.exists(path):
            print(f"  Opening WB2: {path}")
            try:
                stores.append(xr.open_zarr(path))
            except Exception as e:
                print(f"  WARNING: could not open {path}: {e}")
    if not stores:
        raise RuntimeError("No WeatherBench2 stores found.")
    return stores


def _find_timestamps_in_stores(
    stores: list[xr.Dataset],
    timestamps: list[pd.Timestamp],
) -> dict[pd.Timestamp, xr.Dataset]:
    """Return a mapping from each timestamp to the store that contains it."""
    result: dict[pd.Timestamp, xr.Dataset] = {}
    for ts in timestamps:
        for ds in stores:
            ts_index = pd.DatetimeIndex(ds.time.values)
            if ts in ts_index:
                result[ts] = ds
                break
        if ts not in result:
            raise RuntimeError(
                f"Timestamp {ts} not found in any WeatherBench2 store. "
                f"Available stores cover: "
                + ", ".join(
                    f"{pd.Timestamp(ds.time.values[0])} – {pd.Timestamp(ds.time.values[-1])}"
                    for ds in stores
                )
            )
    return result


def _extract_atmos(ds: xr.Dataset, timestamps: list[pd.Timestamp]) -> xr.Dataset:
    """Extract atmospheric variables at `timestamps` and the required pressure levels."""
    ts_pd = pd.DatetimeIndex(timestamps)
    # Use .sel with method=None (exact match) on the time index
    slices = []
    for ts in ts_pd:
        slices.append(ds.sel(time=ts))
    combined = xr.concat(slices, dim="time")
    combined["time"] = pd.DatetimeIndex(timestamps)

    out_vars = {}
    for short_name, wb2_name in ATMOS_MAP.items():
        arr = combined[wb2_name].sel(level=AURORA_LEVELS)
        out_vars[short_name] = arr.rename({"level": "pressure_level"})

    return xr.Dataset(out_vars)


def _extract_surf(ds: xr.Dataset, timestamps: list[pd.Timestamp]) -> xr.Dataset:
    """Extract surface variables at `timestamps`."""
    ts_pd = pd.DatetimeIndex(timestamps)
    slices = []
    for ts in ts_pd:
        slices.append(ds.sel(time=ts))
    combined = xr.concat(slices, dim="time")
    combined["time"] = pd.DatetimeIndex(timestamps)

    out_vars = {}
    for short_name, wb2_name in SURF_MAP.items():
        if wb2_name not in combined:
            raise RuntimeError(f"Variable {wb2_name!r} not in WB2 store")
        out_vars[short_name] = combined[wb2_name]

    return xr.Dataset(out_vars)


def prepare_date(
    init_time: datetime,
    ts_to_store: dict[pd.Timestamp, xr.Dataset],
    output_dir: str,
    skip_existing: bool,
) -> None:
    """Prepare and save dated .nc files for a single init_time."""
    date_str = init_time.strftime("%Y-%m-%d")
    t1 = init_time
    t0 = init_time - timedelta(hours=12)

    atmos_path = os.path.join(output_dir, f"era5_atmos_{date_str}.nc")
    surf_path = os.path.join(output_dir, f"era5_surface_{date_str}.nc")

    if skip_existing and os.path.exists(atmos_path) and os.path.exists(surf_path):
        print(f"  [SKIP] {date_str}: files already exist")
        return

    ts0 = pd.Timestamp(t0)
    ts1 = pd.Timestamp(t1)

    print(f"  Processing {date_str}: t0={t0}, t1={t1}")

    # Both timesteps must be in the same or different stores
    ds0 = ts_to_store[ts0]
    ds1 = ts_to_store[ts1]

    # It's fastest to load t0 and t1 separately then concat
    print(f"    Loading t0={ts0} from {ds0.encoding.get('source', '?')[:50]}")
    atmos_t0 = _extract_atmos(ds0, [ts0])
    surf_t0 = _extract_surf(ds0, [ts0])

    print(f"    Loading t1={ts1} from {ds1.encoding.get('source', '?')[:50]}")
    atmos_t1 = _extract_atmos(ds1, [ts1])
    surf_t1 = _extract_surf(ds1, [ts1])

    atmos_combined = xr.concat([atmos_t0, atmos_t1], dim="time")
    surf_combined = xr.concat([surf_t0, surf_t1], dim="time")

    print(f"    Writing {atmos_path}")
    atmos_combined.load().to_netcdf(atmos_path)
    print(f"    Writing {surf_path}")
    surf_combined.load().to_netcdf(surf_path)
    print(f"  Done {date_str}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading cases from: {args.cases}")
    with open(args.cases) as f:
        raw_cases = json.load(f)

    # Collect unique (init_time, t0) pairs
    all_init_times = set()
    for entry in raw_cases:
        it = datetime.fromisoformat(entry["init_time"])
        all_init_times.add(it)

    # Collect all timestamps we need (t0 and t1 for each init_time)
    all_timestamps: set[pd.Timestamp] = set()
    for it in all_init_times:
        all_timestamps.add(pd.Timestamp(it))
        all_timestamps.add(pd.Timestamp(it - timedelta(hours=12)))

    print(f"\nNeed {len(all_timestamps)} timestamps across {len(all_init_times)} case dates:")
    for ts in sorted(all_timestamps):
        print(f"  {ts}")

    print("\nOpening WeatherBench2 stores...")
    stores = _open_wb2_stores()

    print("\nMapping timestamps to stores...")
    ts_to_store = _find_timestamps_in_stores(stores, sorted(all_timestamps))
    print(f"  All {len(ts_to_store)} timestamps found.")

    print(f"\nOutput dir: {args.output_dir}")
    for it in sorted(all_init_times):
        prepare_date(it, ts_to_store, args.output_dir, args.skip_existing)

    print("\nAll done.")


if __name__ == "__main__":
    main()
