"""
conditional_data.py — ZWD replacement builders and I/O helpers.

Provides:
  - I/O helpers (_ensure_dir, _write_csv, _write_json, _append_csv)
  - IWV computation and spatial qhat regression
  - All six ZWD replacement modes
  - Reference-frame loading from the ZWD zarr
  - Replacement quality diagnostics
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_data import CaseData, ZWD_ZARR_PATH  # noqa: E402
from searchlight_tasks import (  # noqa: E402
    Case, TARGETS, SCALES, generate_mask_centers, great_circle_km,
)

G = 9.80665            # m/s²
ZWD_SCALE_GLOBAL = 98.5413  # normalization std (mm), from normalization_stats_1979_2021.json
N_REF_MAX = 50


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _append_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not os.path.exists(path)
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ─── IWV and qhat ────────────────────────────────────────────────────────────

def compute_iwv(
    q_L_H_W: np.ndarray,
    pressure_levels: tuple,
    sp_hw: np.ndarray | None = None,
) -> np.ndarray:
    """Column-integrated water vapor (kg/m²) via trapezoid integration.

    q in kg/kg, pressure_levels in hPa. Returns (H, W) float32.

    Without `sp_hw` the integral runs over all levels, including those that
    lie below ground over high terrain (ERA5 extrapolates q there). That adds
    ~8 kg/m² of fictitious moisture over major mountain ranges and biases the
    ZWD~IWV regression exactly where the Stage-A effect is largest — see
    check_iwv_baseline_variants.py and its output table.

    Pass `sp_hw` (surface pressure in Pa) to clamp the integration bounds to
    the local surface: layers fully below ground collapse to zero thickness
    and the lowest straddling layer is truncated at the surface. This tracks
    ERA5's native total_column_water_vapour to within ~0.15 kg/m².
    """
    p = np.array(pressure_levels, dtype=np.float64) * 100.0  # Pa
    sort_idx = np.argsort(p)[::-1]  # surface → top
    p_s = p[sort_idx]
    q_s = q_L_H_W[sort_idx].astype(np.float64)
    q_mid = 0.5 * (q_s[:-1] + q_s[1:])

    if sp_hw is None:
        dp = np.abs(np.diff(p_s))
        iwv = np.sum(q_mid * dp[:, None, None], axis=0) / G
    else:
        p_eff = np.minimum(p_s[:, None, None], sp_hw.astype(np.float64)[None, :, :])
        iwv = np.sum(q_mid * np.abs(np.diff(p_eff, axis=0)), axis=0) / G

    return iwv.astype(np.float32)


def load_wb2_surface_field(case: CaseData, var: str) -> np.ndarray:
    """(2, H, W) WB2 surface field for the case's two input timesteps.

    Not carried on CaseData (Aurora's Batch takes neither `sp` nor `tcwv`), so
    it is read straight from the WB2 store. Raises for 2024+ cases:
    weatherbench2_2024_2025 carries neither surface_pressure nor
    total_column_water_vapour.
    """
    from searchlight_data import WB2_PATHS  # local import: avoids cycles

    times = [
        pd.Timestamp(case.init_time - timedelta(hours=6)),
        pd.Timestamp(case.init_time),
    ]
    out = []
    for ts in times:
        found = None
        for path in WB2_PATHS:
            try:
                ds = xr.open_zarr(path)
            except Exception:
                continue
            if not (pd.Timestamp(ds.time.values[0]) <= ts <= pd.Timestamp(ds.time.values[-1])):
                continue
            if var not in ds.data_vars:
                raise KeyError(
                    f"{ts}: store {os.path.basename(path)} has no {var}; "
                    f"this baseline cannot be built for this date."
                )
            found = ds[var].sel(time=ts).values.astype(np.float32)
            break
        if found is None:
            raise FileNotFoundError(f"No WB2 store covers {ts}")
        out.append(found)
    return np.stack(out)


def load_surface_pressure(case: CaseData) -> np.ndarray:
    """(2, H, W) surface pressure in Pa for the case's two input timesteps."""
    return load_wb2_surface_field(case, "surface_pressure")


def fit_spatial_qhat(
    zwd_hw: np.ndarray,
    iwv_hw: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Global OLS ZWD ~ a·IWV + b across all valid gridpoints.

    Returns (a, b, qhat_hw).
    """
    zwd_flat = zwd_hw.ravel().astype(np.float64)
    iwv_flat = iwv_hw.ravel().astype(np.float64)
    valid = np.isfinite(zwd_flat) & np.isfinite(iwv_flat) & (iwv_flat >= 0.0)

    if valid.sum() < 1000:
        print("  WARNING: few valid gridpoints for qhat regression; copying ZWD.")
        return 0.0, float(np.nanmean(zwd_hw)), zwd_hw.copy()

    X = np.stack([iwv_flat[valid], np.ones(valid.sum())], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, zwd_flat[valid], rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    except np.linalg.LinAlgError:
        a, b = 0.0, float(np.nanmean(zwd_hw))

    return a, b, (a * iwv_hw + b).astype(np.float32)


def _linear_feature_qhat(
    zwd_hw: np.ndarray,
    feature_hws: list[np.ndarray],
    *,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit/predict ZWD from standardized feature maps using OLS."""
    y = zwd_hw.ravel().astype(np.float64)
    X_cols = [np.asarray(f, dtype=np.float64).ravel() for f in feature_hws]
    valid = np.isfinite(y)
    for col in X_cols:
        valid &= np.isfinite(col)
    if sample_mask is not None:
        valid &= np.asarray(sample_mask, dtype=bool).ravel()

    if valid.sum() < 1000:
        _, _, qhat = fit_spatial_qhat(zwd_hw, feature_hws[0])
        return qhat

    X_fit = np.stack([col[valid] for col in X_cols], axis=1)
    mu = X_fit.mean(axis=0)
    sig = X_fit.std(axis=0)
    sig[sig < 1e-12] = 1.0
    X_fit = (X_fit - mu) / sig
    X_fit = np.concatenate([X_fit, np.ones((X_fit.shape[0], 1))], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X_fit, y[valid], rcond=None)

    X_all = np.stack(X_cols, axis=1)
    X_all = (X_all - mu) / sig
    X_all = np.concatenate([X_all, np.ones((X_all.shape[0], 1))], axis=1)
    return (X_all @ coeffs).reshape(zwd_hw.shape).astype(np.float32)


def _latlon_mesh(case: CaseData) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(case.lat_vals, case.lon_vals, indexing="ij")


def _target_distance_km(case: CaseData, target_key: str) -> np.ndarray:
    target = TARGETS[target_key]
    lat_grid, lon_grid = _latlon_mesh(case)
    return great_circle_km(target.center_lat, target.center_lon, lat_grid, lon_grid)


def _target_disk_weight(
    case: CaseData,
    target_key: str,
    *,
    inner_km: float = 1500.0,
    outer_km: float = 3000.0,
) -> np.ndarray:
    """Smooth 1→0 weight around a target; used to blend regional qhat."""
    d = _target_distance_km(case, target_key)
    x = np.clip((outer_km - d) / max(outer_km - inner_km, 1.0), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32)


def _elevation_m(case: CaseData) -> np.ndarray:
    return (case.static_cpu["z"].numpy().astype(np.float32) / G)


def make_qhat_zwd(case: CaseData) -> torch.Tensor:
    """(1, 2, H, W) ZWD predicted from IWV via per-timestep spatial OLS."""
    q_cpu = case.atmos_cpu["q"]
    zwd_cpu = case.surf_cpu["zwd"]
    slices = []
    for t in range(2):
        iwv = compute_iwv(q_cpu[0, t].numpy(), case.pressure_levels)
        _, _, qhat = fit_spatial_qhat(zwd_cpu[0, t].numpy(), iwv)
        lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_qhat_sp_zwd(case: CaseData) -> torch.Tensor:
    """(1, 2, H, W) ZWD predicted from surface-masked IWV via spatial OLS.

    Identical to `make_qhat_zwd` except that sub-surface levels are excluded
    from the IWV integral, removing the terrain-dependent bias of the plain
    13-level trapezoid.
    """
    q_cpu = case.atmos_cpu["q"]
    zwd_cpu = case.surf_cpu["zwd"]
    sp = load_surface_pressure(case)
    slices = []
    for t in range(2):
        iwv = compute_iwv(q_cpu[0, t].numpy(), case.pressure_levels, sp_hw=sp[t])
        _, _, qhat = fit_spatial_qhat(zwd_cpu[0, t].numpy(), iwv)
        lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_q850_aboveground_target(case: CaseData, target_key: str, level_idx: int):
    """q850 box mean restricted to ABOVE-GROUND cells, plus the above-ground fraction.

    Over high terrain a large part of the target box sits below the 850 hPa
    surface (Rockies 68 %, Himalayas 67 %, Andes 47 %), where ERA5 and Aurora
    both carry extrapolated values. The unmasked box mean is therefore not a
    physically meaningful low-level humidity there. Returns (fn, frac); fn is
    None when the box is entirely below ground, in which case no above-ground
    score exists for that target.

    Matches the unmasked convention (plain mean, g/kg) so the only difference
    is which cells are included.
    """
    tr = TARGETS[target_key]
    lat, lon = case.lat_vals, case.lon_vals
    la = (lat >= tr.box_lat[0]) & (lat <= tr.box_lat[1])
    l3 = np.mod(lon, 360.0)
    w, e = np.mod(tr.box_lon[0], 360.0), np.mod(tr.box_lon[1], 360.0)
    lo = (l3 >= w) & (l3 <= e) if w <= e else (l3 >= w) | (l3 <= e)
    box = la[:, None] & lo[None, :]

    sp = load_surface_pressure(case)[1]           # t1; sp varies little over 6 h
    above = box & (sp >= 850.0 * 100.0)
    n_box = max(int(box.sum()), 1)
    frac = float(above.sum()) / n_box
    if above.sum() == 0:
        return None, frac

    mask_t = torch.tensor(above.astype(np.float32))

    def target_fn(pred):
        q = pred.atmos_vars["q"].float()[0, 0, level_idx]
        m = mask_t.to(q.device)
        # Aurora's Batch.crop drops the LAST latitude row when H % patch_size == 1
        # (721 -> 720, i.e. the -90 deg pole), so predictions are one row short of
        # the input grid. Row indices 0..719 stay aligned; just trim the mask.
        if m.shape[0] != q.shape[0]:
            m = m[: q.shape[0]]
        return (q * m).sum() / m.sum() * 1e3

    return target_fn, frac


def make_qhat_tcwv_zwd(case: CaseData) -> torch.Tensor:
    """(1, 2, H, W) ZWD predicted from ERA5 native total_column_water_vapour.

    The 137-level operational product, so it does not inherit any of the
    13-level quadrature or sub-surface issues. Used to check that `qhat_sp` is
    a faithful stand-in rather than assuming it.
    """
    zwd_cpu = case.surf_cpu["zwd"]
    tcwv = load_wb2_surface_field(case, "total_column_water_vapour")
    lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
    slices = []
    for t in range(2):
        _, _, qhat = fit_spatial_qhat(zwd_cpu[0, t].numpy(), tcwv[t])
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_meanmatched_zwd(
    case: CaseData,
    base_qhat: torch.Tensor,
    target_key: str,
    *,
    disk_km: float = 1500.0,
) -> torch.Tensor:
    """Local-intercept correction of `base_qhat` — the mean-matched counterfactual.

    For each input timestep, mu is the cos(lat)-weighted mean of the residual
    r = zwd_true - base_qhat over a disk of radius `disk_km` around the target,
    and the correction is applied through the existing 1500-3000 km cosine taper:

        qhat_meanmatched(x) = base_qhat(x) + w(x) * mu

    The taper matters: adding mu globally would shift ZWD at every gridpoint
    using a statistic derived from one target region. w = 1 inside `disk_km`
    (where mu is measured) and falls smoothly to 0 by 3000 km.

    This makes the counterfactual match the local mean ZWD, so it differs from
    the truth only in spatial pattern. Together with `base_qhat` it gives the
    exact decomposition

        S(true) - S(base) = [S(true) - S(meanmatched)]  (pattern effect)
                          + [S(meanmatched) - S(base)]  (local-offset effect)
    """
    zwd_cpu = case.surf_cpu["zwd"]
    d_km = _target_distance_km(case, target_key)
    disk = d_km <= disk_km
    blend = _target_disk_weight(case, target_key)  # 1 inside 1500 km -> 0 at 3000 km
    coslat = np.cos(np.deg2rad(case.lat_vals))[:, None] * np.ones((1, len(case.lon_vals)))

    lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
    slices = []
    for t in range(2):
        r = zwd_cpu[0, t].numpy() - base_qhat[0, t].numpy()
        w = coslat[disk]
        mu = float(np.sum(r[disk] * w) / np.sum(w))
        out = base_qhat[0, t].numpy() + blend * mu
        slices.append(torch.tensor(np.clip(out, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_regional_qhat_sp_zwd(case: CaseData, target_key: str) -> torch.Tensor:
    """Local slope AND intercept: OLS of ZWD on surface-masked IWV near the target."""
    q_cpu = case.atmos_cpu["q"]
    zwd_cpu = case.surf_cpu["zwd"]
    sp = load_surface_pressure(case)
    global_qhat = make_qhat_sp_zwd(case)
    fit_mask = _target_distance_km(case, target_key) <= 2500.0
    blend = _target_disk_weight(case, target_key)

    lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
    slices = []
    for t in range(2):
        iwv = compute_iwv(q_cpu[0, t].numpy(), case.pressure_levels, sp_hw=sp[t])
        qhat_reg = _linear_feature_qhat(zwd_cpu[0, t].numpy(), [iwv], sample_mask=fit_mask)
        qhat = blend * qhat_reg + (1.0 - blend) * global_qhat[0, t].numpy()
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_regional_qhat_zwd(case: CaseData, target_key: str) -> torch.Tensor:
    """IWV→ZWD qhat with coefficients fit near the target and blended locally."""
    q_cpu = case.atmos_cpu["q"]
    zwd_cpu = case.surf_cpu["zwd"]
    global_qhat = make_qhat_zwd(case)
    fit_mask = _target_distance_km(case, target_key) <= 2500.0
    blend = _target_disk_weight(case, target_key)

    slices = []
    for t in range(2):
        iwv = compute_iwv(q_cpu[0, t].numpy(), case.pressure_levels)
        zwd_hw = zwd_cpu[0, t].numpy()
        qhat_reg = _linear_feature_qhat(zwd_hw, [iwv], sample_mask=fit_mask)
        qhat = blend * qhat_reg + (1.0 - blend) * global_qhat[0, t].numpy()
        lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


def make_temp_elev_qhat_zwd(case: CaseData) -> torch.Tensor:
    """ZWD predicted from IWV plus 2m temperature and elevation."""
    q_cpu = case.atmos_cpu["q"]
    zwd_cpu = case.surf_cpu["zwd"]
    elev = _elevation_m(case)
    slices = []
    for t in range(2):
        iwv = compute_iwv(q_cpu[0, t].numpy(), case.pressure_levels)
        t2m = case.surf_cpu["2t"][0, t].numpy()
        qhat = _linear_feature_qhat(zwd_cpu[0, t].numpy(), [iwv, t2m, elev])
        lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


_REF_QHAT_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}


def _fit_ref_linear_models(
    ref_times: list[pd.Timestamp],
    n_ref_fit: int,
    rng: np.random.Generator,
    n_sample_per_ref: int = 20000,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit monthly reference IWV→ZWD models for t0/t1 and cache by timestamp set."""
    if not ref_times:
        raise ValueError("No reference times available for qhat_ref_month.")

    n_candidates = min(len(ref_times), max(n_ref_fit * 4, n_ref_fit))
    if len(ref_times) > n_candidates:
        idx = np.linspace(0, len(ref_times) - 1, n_candidates, dtype=int)
        selected = [ref_times[int(i)] for i in idx]
    else:
        selected = list(ref_times)

    key = tuple(pd.Timestamp(t).isoformat() for t in selected), int(n_sample_per_ref)
    if key in _REF_QHAT_CACHE:
        return _REF_QHAT_CACHE[key]

    from searchlight_data import load_case

    xs: list[list[np.ndarray]] = [[], []]
    ys: list[list[np.ndarray]] = [[], []]
    n_loaded = 0
    for ts in selected:
        try:
            ref_case = load_case(pd.Timestamp(ts).to_pydatetime())
        except Exception as exc:
            print(f"  WARNING: skipping qhat_ref_month timestamp {ts}: {exc}")
            continue
        for timestep in range(2):
            iwv = compute_iwv(ref_case.atmos_cpu["q"][0, timestep].numpy(),
                              ref_case.pressure_levels)
            zwd = ref_case.surf_cpu["zwd"][0, timestep].numpy()
            valid = np.isfinite(iwv.ravel()) & np.isfinite(zwd.ravel()) & (iwv.ravel() >= 0.0)
            valid_idx = np.where(valid)[0]
            if valid_idx.size > n_sample_per_ref:
                valid_idx = rng.choice(valid_idx, size=n_sample_per_ref, replace=False)
            xs[timestep].append(iwv.ravel()[valid_idx].astype(np.float64))
            ys[timestep].append(zwd.ravel()[valid_idx].astype(np.float64))
        n_loaded += 1
        if n_loaded >= n_ref_fit:
            break

    if n_loaded == 0:
        raise RuntimeError("No usable reference timestamps for qhat_ref_month.")

    models: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for timestep in range(2):
        x = np.concatenate(xs[timestep])
        y = np.concatenate(ys[timestep])
        mu = np.array([x.mean()], dtype=np.float64)
        sig = np.array([x.std()], dtype=np.float64)
        sig[sig < 1e-12] = 1.0
        X = np.stack([(x - mu[0]) / sig[0], np.ones_like(x)], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        models.append((coeffs.astype(np.float64), mu, sig))

    _REF_QHAT_CACHE[key] = models
    return models


def make_reference_qhat_zwd(
    case: CaseData,
    ref_times: list[pd.Timestamp],
    rng: np.random.Generator,
    n_ref_fit: int = 4,
) -> torch.Tensor:
    """IWV→ZWD qhat whose coefficients are fit from other years in the same month/hour."""
    if not ref_times or n_ref_fit <= 0:
        return make_qhat_zwd(case)

    try:
        models = _fit_ref_linear_models(ref_times, n_ref_fit, rng)
    except Exception as exc:
        print(f"  WARNING: qhat_ref_month fallback to per-case qhat: {exc}")
        return make_qhat_zwd(case)
    slices = []
    for t in range(2):
        iwv = compute_iwv(case.atmos_cpu["q"][0, t].numpy(), case.pressure_levels)
        coeffs, mu, sig = models[t]
        x = ((iwv.astype(np.float64).ravel() - mu[0]) / sig[0])
        X = np.stack([x, np.ones_like(x)], axis=1)
        qhat = (X @ coeffs).reshape(iwv.shape).astype(np.float32)
        lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
        slices.append(torch.tensor(np.clip(qhat, lo, hi), dtype=torch.float32))
    return torch.stack(slices, dim=0).unsqueeze(0)


# ─── Reference frame loading ──────────────────────────────────────────────────

def load_zwd_reference_timestamps(
    month: int,
    hour: int,
    exclude_year: int | None = None,
    n_max: int = N_REF_MAX,
) -> tuple[list[np.ndarray], list[pd.Timestamp]]:
    """Load up to n_max ZWD fields from the zarr for (month, hour).

    Leaves out exclude_year to avoid in-sample leakage.
    """
    ds = xr.open_zarr(ZWD_ZARR_PATH)
    all_times = pd.DatetimeIndex(ds.time.values)
    mask = (all_times.month == month) & (all_times.hour == hour)
    if exclude_year is not None:
        mask &= (all_times.year != exclude_year)
    ref_times = all_times[mask]

    if len(ref_times) == 0:
        print(f"  WARNING: no reference ZWD for month={month} hour={hour}.")
        return [], []

    if len(ref_times) > n_max:
        idx = np.linspace(0, len(ref_times) - 1, n_max, dtype=int)
        ref_times = ref_times[idx]

    zwds, timestamps = [], []
    for t in ref_times:
        try:
            zwds.append(ds["zenith_wet_delay"].sel(time=t).values.astype(np.float32))
            timestamps.append(t)
        except Exception as e:
            print(f"  WARNING: could not load ZWD at {t}: {e}")
    return zwds, timestamps


# ─── ZWD replacement modes ────────────────────────────────────────────────────

def make_climatology_zwd(case: CaseData, ref_zwds: list[np.ndarray]) -> torch.Tensor:
    """(1, 2, H, W) monthly mean ZWD from reference frames."""
    if not ref_zwds:
        return case.surf_cpu["zwd"].clone()
    clim = torch.tensor(np.mean(np.stack(ref_zwds), axis=0).astype(np.float32))
    return clim.unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()


def make_residual_only_zwd(
    case: CaseData,
    qhat_zwd: torch.Tensor,
    climatology_zwd: torch.Tensor,
) -> torch.Tensor:
    """climatology + (true − qhat), clipped to 4σ."""
    out = climatology_zwd + (case.surf_cpu["zwd"] - qhat_zwd)
    lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
    return out.clamp_(lo, hi)


def make_matched_swap_zwd(
    case: CaseData,
    ref_zwds: list[np.ndarray],
    climatology_zwd: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Donor ZWD whose residual is most spatially correlated with the true residual."""
    if not ref_zwds:
        return case.surf_cpu["zwd"].clone()

    clim_hw = climatology_zwd[0, 0].numpy()
    true_resid = (case.surf_cpu["zwd"][0, 1].numpy() - clim_hw).ravel()
    true_norm = true_resid / (true_resid.std() + 1e-8)

    best_corr, best_idx = -np.inf, 0
    for i, ref_hw in enumerate(ref_zwds):
        ref_resid = (ref_hw - clim_hw).ravel()
        corr = float(np.dot(true_norm, ref_resid / (ref_resid.std() + 1e-8)) / len(true_norm))
        if corr > best_corr:
            best_corr, best_idx = corr, i

    donor = torch.tensor(ref_zwds[best_idx].astype(np.float32))
    return donor.unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()


def make_random_same_month_zwd(
    case: CaseData,
    ref_zwds: list[np.ndarray],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Random donor ZWD from same month (negative control)."""
    if not ref_zwds:
        return case.surf_cpu["zwd"].clone()
    donor = torch.tensor(ref_zwds[int(rng.integers(0, len(ref_zwds)))].astype(np.float32))
    return donor.unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()


def build_all_zwd_replacements(
    case: CaseData,
    ref_zwds: list[np.ndarray],
    rng: np.random.Generator,
    modes: list[str],
    *,
    case_obj: Case | None = None,
    ref_times: list[pd.Timestamp] | None = None,
    n_ref_fit: int = 4,
) -> dict[str, torch.Tensor | None]:
    """Build every requested ZWD replacement tensor (CPU float32, (1,2,H,W)).

    'true' → None so make_batch uses the original unmodified ZWD.
    """
    out: dict[str, torch.Tensor | None] = {}
    qhat: torch.Tensor | None = None
    qhat_sp: torch.Tensor | None = None
    clim: torch.Tensor | None = None

    for mode in modes:
        if mode == "true":
            out["true"] = None
        elif mode == "qhat":
            if qhat is None:
                qhat = make_qhat_zwd(case)
            out["qhat"] = qhat
        elif mode == "qhat_sp":
            if qhat_sp is None:
                qhat_sp = make_qhat_sp_zwd(case)
            out["qhat_sp"] = qhat_sp
        elif mode == "qhat_sp_meanmatched":
            if case_obj is None:
                raise ValueError("qhat_sp_meanmatched needs case_obj for the target disk")
            if qhat_sp is None:
                qhat_sp = make_qhat_sp_zwd(case)
            out["qhat_sp_meanmatched"] = make_meanmatched_zwd(case, qhat_sp, case_obj.target)
        elif mode == "qhat_sp_regional":
            if case_obj is None:
                raise ValueError("qhat_sp_regional needs case_obj for the target disk")
            out["qhat_sp_regional"] = make_regional_qhat_sp_zwd(case, case_obj.target)
        elif mode == "qhat_tcwv":
            out["qhat_tcwv"] = make_qhat_tcwv_zwd(case)
        elif mode == "qhat_regional":
            if case_obj is None:
                if qhat is None:
                    qhat = make_qhat_zwd(case)
                out["qhat_regional"] = qhat
            else:
                out["qhat_regional"] = make_regional_qhat_zwd(case, case_obj.target)
        elif mode == "qhat_temp_elev":
            out["qhat_temp_elev"] = make_temp_elev_qhat_zwd(case)
        elif mode == "qhat_ref_month":
            out["qhat_ref_month"] = make_reference_qhat_zwd(
                case, ref_times or [], rng, n_ref_fit=n_ref_fit,
            )
        elif mode == "climatology":
            if clim is None:
                clim = make_climatology_zwd(case, ref_zwds)
            out["climatology"] = clim
        elif mode == "residual_only":
            if qhat is None:
                qhat = make_qhat_zwd(case)
            if clim is None:
                clim = make_climatology_zwd(case, ref_zwds)
            out["residual_only"] = make_residual_only_zwd(case, qhat, clim)
        elif mode == "matched_swap":
            if clim is None:
                clim = make_climatology_zwd(case, ref_zwds)
            out["matched_swap"] = make_matched_swap_zwd(case, ref_zwds, clim, rng)
        elif mode == "random_same_month":
            out["random_same_month"] = make_random_same_month_zwd(case, ref_zwds, rng)
        else:
            raise ValueError(f"Unknown ZWD mode: {mode!r}")
    return out


# ─── Diagnostics ──────────────────────────────────────────────────────────────

def log_qhat_fit_quality(case: CaseData, case_id: str) -> None:
    q_t1 = case.atmos_cpu["q"][0, 1].numpy()
    zwd_t1 = case.surf_cpu["zwd"][0, 1].numpy()
    iwv = compute_iwv(q_t1, case.pressure_levels)
    a, b, qhat = fit_spatial_qhat(zwd_t1, iwv)
    valid = np.isfinite(zwd_t1.ravel()) & np.isfinite(qhat.ravel())
    if valid.sum() > 100:
        r = float(np.corrcoef(zwd_t1.ravel()[valid], qhat.ravel()[valid])[0, 1])
        resid_rms = float(np.sqrt(np.nanmean((zwd_t1 - qhat) ** 2)))
        print(f"  [{case_id}] qhat fit (t1): a={a:.2f} b={b:.1f} r={r:.3f} "
              f"resid_rms={resid_rms:.1f} mm")


def replacement_diagnostics(
    case: CaseData,
    replacements: dict[str, torch.Tensor | None],
    ref_zwds: list[np.ndarray],
    case_obj: Case,
) -> list[dict[str, Any]]:
    """Per-mode quality summary (correlation with true ZWD, residual stats)."""
    true_t1 = case.surf_cpu["zwd"][0, 1].numpy()
    n_ref = len(ref_zwds)
    rows = []

    for mode, zwd_ovr in replacements.items():
        if zwd_ovr is None:
            rows.append({
                "case_id": case_obj.case_id,
                "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(),
                "role": case_obj.role,
                "mode": mode,
                "n_ref_frames": n_ref,
                "corr_with_true": 1.0,
                "residual_rms_mm": 0.0,
                "mean_zwd_mm": float(np.nanmean(true_t1)),
                "std_zwd_mm": float(np.nanstd(true_t1)),
                "residual_mean_mm": 0.0,
                "residual_std_mm": 0.0,
            })
            continue

        rep_t1 = zwd_ovr[0, 1].numpy()
        diff = (true_t1 - rep_t1).ravel()
        valid = np.isfinite(true_t1.ravel()) & np.isfinite(rep_t1.ravel())
        corr = (
            float(np.corrcoef(true_t1.ravel()[valid], rep_t1.ravel()[valid])[0, 1])
            if valid.sum() > 10 else float("nan")
        )
        rows.append({
            "case_id": case_obj.case_id,
            "target": case_obj.target,
            "init_time": case_obj.init_time.isoformat(),
            "role": case_obj.role,
            "mode": mode,
            "n_ref_frames": n_ref,
            "corr_with_true": corr,
            "residual_rms_mm": float(np.sqrt(np.nanmean(diff ** 2))),
            "mean_zwd_mm": float(np.nanmean(rep_t1)),
            "std_zwd_mm": float(np.nanstd(rep_t1)),
            "residual_mean_mm": float(np.nanmean(diff)),
            "residual_std_mm": float(np.nanstd(diff)),
        })
    return rows


# ─── Localized residual masks ────────────────────────────────────────────────

def _target_box_mask(case: CaseData, target_key: str) -> np.ndarray:
    target = TARGETS[target_key]
    lat_grid, lon_grid = _latlon_mesh(case)
    lon = np.mod(lon_grid, 360.0)
    west, east = target.box_lon[0] % 360.0, target.box_lon[1] % 360.0
    if west <= east:
        lon_mask = (lon >= west) & (lon <= east)
    else:
        lon_mask = (lon >= west) | (lon <= east)
    return (
        (lat_grid >= target.box_lat[0]) & (lat_grid <= target.box_lat[1]) & lon_mask
    )


def _upstream_mask(case: CaseData, target_key: str, near_mask: np.ndarray) -> np.ndarray:
    """Approximate upstream sector from target-box mean 850 hPa wind at t1."""
    target = TARGETS[target_key]
    box = _target_box_mask(case, target_key)
    levels = np.asarray(case.pressure_levels)
    idx = int(np.argmin(np.abs(levels - 850)))
    u = case.atmos_cpu["u"][0, 1, idx].numpy()
    v = case.atmos_cpu["v"][0, 1, idx].numpy()
    if np.any(box):
        u0 = float(np.nanmean(u[box]))
        v0 = float(np.nanmean(v[box]))
    else:
        u0 = float(np.nanmean(u))
        v0 = float(np.nanmean(v))
    speed = (u0 ** 2 + v0 ** 2) ** 0.5
    if not np.isfinite(speed) or speed < 1e-6:
        return np.zeros_like(near_mask, dtype=bool)

    lat_grid, lon_grid = _latlon_mesh(case)
    dy = target.center_lat - lat_grid
    dx = ((target.center_lon - lon_grid + 180.0) % 360.0 - 180.0) * np.cos(
        np.deg2rad(target.center_lat)
    )
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    # Dot product between wind direction and vector from grid cell to target.
    align = (u0 * (dx / norm) + v0 * (dy / norm)) / speed
    return near_mask & (align >= np.cos(np.deg2rad(60.0)))


def build_localized_residual_masks(
    case: CaseData,
    case_obj: Case,
) -> dict[str, torch.Tensor]:
    """Spatial masks for qhat + M*(true-qhat) residual localization at t1."""
    target = TARGETS[case_obj.target]
    lat_grid, lon_grid = _latlon_mesh(case)
    d_target = great_circle_km(target.center_lat, target.center_lon, lat_grid, lon_grid)

    target_box = _target_box_mask(case, case_obj.target)
    near = d_target <= 2500.0
    elev = _elevation_m(case)
    lsm = case.static_cpu["lsm"].numpy()
    mountain = near & (lsm > 0.2) & (elev > 750.0)
    if int(mountain.sum()) < 500:
        mountain = near & (lsm > 0.2) & (elev > 400.0)

    upstream = _upstream_mask(case, case_obj.target, near)

    remote = np.zeros_like(near, dtype=bool)
    for spec in generate_mask_centers(target, SCALES["synoptic"], n_remote=4):
        if spec.role != "remote":
            continue
        d_remote = great_circle_km(spec.center_lat, spec.center_lon, lat_grid, lon_grid)
        remote |= d_remote <= 1000.0

    masks_np = {
        "target_box": target_box,
        "near_disk": near,
        "mountain_near": mountain,
        "upstream": upstream,
        "remote_control": remote,
        "full_t1": np.ones_like(near, dtype=bool),
    }
    return {
        name: torch.tensor(mask.astype(np.float32), dtype=torch.float32)
        for name, mask in masks_np.items()
    }


def make_localized_residual_zwd(
    case: CaseData,
    qhat_zwd: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    timestep: int = 1,
) -> torch.Tensor:
    """Return qhat with true-qhat residual restored only inside mask_hw."""
    out = qhat_zwd.clone()
    residual = case.surf_cpu["zwd"] - qhat_zwd
    mask = mask_hw.to(dtype=torch.float32)
    out[0, timestep] = qhat_zwd[0, timestep] + mask * residual[0, timestep]
    lo, hi = case.zwd_loc - 4 * case.zwd_scale, case.zwd_loc + 4 * case.zwd_scale
    return out.clamp_(lo, hi)
