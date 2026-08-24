# Third Party Notices

The GeoXplain Aurora Adapter source code is licensed under the MIT License.
This file records third-party material that may be installed or
used alongside the adapter. These notices are informational and do not change
the project license.

## Direct Dependencies

The adapter depends on Python packages that are installed separately by the
user's package manager and retain their own licenses. Direct runtime and
optional dependencies include:

- Microsoft Aurora, MIT License.
- PyTorch, BSD-style License.
- NumPy and SciPy, BSD-style Licenses.
- xarray, Apache License 2.0.
- pandas, BSD 3-Clause License.
- netCDF4, MIT License.
- scikit-image and scikit-learn, BSD 3-Clause Licenses.
- FastAPI, MIT License.
- Uvicorn, BSD 3-Clause License.
- HTTPX, BSD 3-Clause License.
- msgpack, Apache License 2.0.
- tomli, MIT License.

## Models, Data, and Execution Environments

This adapter can load Microsoft Aurora models and WeatherBench-style or ERA5
weather data. The MIT License for the adapter applies to the adapter code
unless otherwise stated. It does not grant rights to third-party model weights,
weather datasets, input files, derived forecasts, attribution result bundles, or
the SLURM/HPC environment where the adapter is run.

Review the applicable source licenses and terms before redistributing model
weights, datasets, generated `.xia.npz` / `.overlay.npz` bundles, or outputs
that incorporate third-party data.
