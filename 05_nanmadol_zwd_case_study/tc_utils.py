"""
Utilities for the ZWD tropical-cyclone case study.

This module reuses the existing ZWD Aurora setup from the searchlight and
humidity studies, but adds cyclone-specific helpers:

- case configuration
- 6-hour cadence validation from actual model outputs
- simple moving-window MSL tracker
- differentiable soft storm targets for saliency-through-rollout
- storm-relative pooling and Gaussian-region selection on the initial ZWD field
"""

from __future__ import annotations

import json
import math
import os
import sys
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

sys.path.insert(0, _ROOT)
sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_benchmark import _forward, _gpu_sync_and_gc, rollout_mean, setup_model  # noqa: E402
from searchlight_data import load_case, make_batch  # noqa: E402
from searchlight_ground_truth import perturb_zwd  # noqa: E402
from searchlight_tasks import MaskSpec, gaussian_mask, great_circle_km  # noqa: E402


DEFAULT_OUTPUT_ROOT = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "zwd_tc_case_study",
)
DEFAULT_EXPECTED_STEP_HOURS = 6.0
DEFAULT_STAMP_LEADS_HOURS = (24, 48, 72)


@dataclass(frozen=True)
class TCCaseConfig:
    slug: str
    name: str
    init_time: datetime
    init_lat: float
    init_lon: float
    map_extent: tuple[float, float, float, float]
    analysis_extent: tuple[float, float, float, float]
    track_window_radius_deg: float = 7.5
    intensity_radius_deg: float = 4.0
    selection_sigma_deg: float = 3.0
    selection_stride_deg: float = 2.0
    near_radius_km: float = 1800.0
    remote_min_km: float = 3200.0
    remote_max_count: int = 6
    polar_radius_km: float = 2500.0
    softmin_temp_hpa: float = 2.0


TC_CASES: dict[str, TCCaseConfig] = {
    "nanmadol": TCCaseConfig(
        slug="nanmadol",
        name="Typhoon Nanmadol",
        init_time=datetime(2022, 9, 17, 12, 0),
        init_lat=27.5,
        init_lon=132.0,
        map_extent=(118.0, 146.0, 18.0, 42.0),
        # Wide enough to include remote masks at ≥3200 km from the storm:
        # corners reach ~4000-5500 km from (27.5N, 132E).
        analysis_extent=(70.0, 180.0, -15.0, 65.0),
    ),
}


def get_case_config(case_name: str) -> TCCaseConfig:
    try:
        return TC_CASES[case_name]
    except KeyError as exc:
        valid = ", ".join(sorted(TC_CASES))
        raise KeyError(f"Unknown TC case {case_name!r}. Valid options: {valid}") from exc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)


def to_plot_lon(lon_vals: np.ndarray) -> np.ndarray:
    return ((lon_vals + 180.0) % 360.0) - 180.0


def lon_to_plot_scalar(lon_deg: float) -> float:
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


def plot_lon_order(lon_vals: np.ndarray) -> np.ndarray:
    return np.argsort(to_plot_lon(lon_vals))


def reorder_lon(field: np.ndarray, order: np.ndarray) -> np.ndarray:
    return field[..., order]


def lon_diff_deg(lon_a: np.ndarray | float, lon_b: float) -> np.ndarray | float:
    return ((lon_a - lon_b + 180.0) % 360.0) - 180.0


def format_case_study_id(case_cfg: TCCaseConfig) -> str:
    return f"{case_cfg.slug}_{case_cfg.init_time.strftime('%Y%m%d%H')}_zwd_tc_case_study"


def load_tc_case(case_name: str):
    case_cfg = get_case_config(case_name)
    case_data = load_case(case_cfg.init_time)
    return case_cfg, case_data


BEST_TRACK_COLOR = "#2a7f62"


def best_track_path(case_cfg: TCCaseConfig) -> str:
    return os.path.join(_HERE, "data", f"{case_cfg.slug}_ibtracs.csv")


def load_best_track(case_cfg: TCCaseConfig) -> pd.DataFrame | None:
    """IBTrACS best track for the case, or None if no curated CSV exists.

    The CSV is the raw IBTrACS column subset (see data/); central pressure and
    wind prefer the RSMC Tokyo agency values (authoritative for the WP basin)
    with the WMO merged columns as fallback.

    Columns: valid_time (datetime64), lat, lon (0..360), lon_plot (-180..180),
    pres_hpa, wind_kt.
    """
    path = best_track_path(case_cfg)
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, parse_dates=["ISO_TIME"])
    pres = raw["TOKYO_PRES"].where(raw["TOKYO_PRES"].notna(), raw["WMO_PRES"])
    wind = raw["TOKYO_WIND"].where(raw["TOKYO_WIND"].notna(), raw["WMO_WIND"])
    lon = raw["LON"].astype(float) % 360.0
    out = pd.DataFrame({
        "valid_time": raw["ISO_TIME"],
        "lat": raw["LAT"].astype(float),
        "lon": lon,
        "lon_plot": to_plot_lon(lon.to_numpy()),
        "pres_hpa": pres.astype(float),
        "wind_kt": wind.astype(float),
    })
    return out.sort_values("valid_time").reset_index(drop=True)


def best_track_at_times(best_df: pd.DataFrame, valid_times) -> pd.DataFrame:
    """Linearly time-interpolate the best track to `valid_times`.

    Times outside the best-track record yield NaN. Longitude is interpolated
    in unwrapped space so tracks crossing the 0/360 seam stay continuous.
    """
    t_bt = best_df["valid_time"].astype("int64").to_numpy(dtype=np.float64)
    t_q = pd.to_datetime(pd.Series(list(valid_times))).astype("int64").to_numpy(dtype=np.float64)

    def _interp(vals: np.ndarray) -> np.ndarray:
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            return np.full(t_q.shape, np.nan)
        out = np.interp(t_q, t_bt[ok], vals[ok])
        out[(t_q < t_bt[ok][0]) | (t_q > t_bt[ok][-1])] = np.nan
        return out

    lat = _interp(best_df["lat"].to_numpy(dtype=float))
    lon_unwrapped = np.degrees(np.unwrap(np.radians(best_df["lon"].to_numpy(dtype=float))))
    lon = _interp(lon_unwrapped) % 360.0
    return pd.DataFrame({
        "valid_time": pd.to_datetime(pd.Series(list(valid_times))).to_numpy(),
        "lat": lat,
        "lon": lon,
        "lon_plot": to_plot_lon(lon),
        "pres_hpa": _interp(best_df["pres_hpa"].to_numpy(dtype=float)),
        "wind_kt": _interp(best_df["wind_kt"].to_numpy(dtype=float)),
    })


def attach_best_track(track_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    """Add bt_lat/bt_lon/bt_lon_plot/bt_pres_hpa/bt_wind_kt columns aligned to
    track_df['valid_time'] (ISO strings or datetimes)."""
    bt = best_track_at_times(best_df, track_df["valid_time"])
    out = track_df.copy()
    for col in ("lat", "lon", "lon_plot", "pres_hpa", "wind_kt"):
        out[f"bt_{col}"] = bt[col].to_numpy()
    return out


def plot_best_track(
    ax,
    best_df: pd.DataFrame,
    *,
    start=None,
    end=None,
    label: str = "IBTrACS best track",
    full_context: bool = True,
    linewidth: float = 2.0,
    markersize: float = 2.6,
) -> None:
    """Draw the best track on a cartopy GeoAxes: the [start, end] window as a
    solid marked line, optionally the full record as a faint context line."""
    import cartopy.crs as ccrs

    sub = best_df
    mask = np.ones(len(best_df), dtype=bool)
    if start is not None:
        mask &= (best_df["valid_time"] >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        mask &= (best_df["valid_time"] <= pd.Timestamp(end)).to_numpy()
    sub = best_df[mask]

    if full_context:
        ax.plot(
            best_df["lon_plot"], best_df["lat"],
            transform=ccrs.PlateCarree(),
            color=BEST_TRACK_COLOR, linewidth=0.9, alpha=0.4, zorder=4,
        )
    if not sub.empty:
        ax.plot(
            sub["lon_plot"], sub["lat"],
            transform=ccrs.PlateCarree(),
            color=BEST_TRACK_COLOR, linewidth=linewidth,
            marker="D", markersize=markersize,
            label=label, zorder=4,
        )


def build_model(device: torch.device):
    model = setup_model(device)
    # Checkpoint only the DECODER level_decoder (PerceiverResampler) which
    # allocates a ~6.4 GiB GELU intermediate during forward. Checkpointing
    # frees that tensor and stores only the much smaller input (~400 MiB),
    # recovering ~6 GiB of headroom.  The encoder level_agg is intentionally
    # NOT checkpointed: its inputs are larger than its outputs, so encoder
    # checkpointing would consume MORE memory, not less.
    orig_ld_fwd = model.decoder.level_decoder.forward

    def _cp_ld(*a, **kw):
        return torch_checkpoint(orig_ld_fwd, *a, use_reentrant=False, **kw)

    model.decoder.level_decoder.forward = _cp_ld
    print("Checkpointed decoder.level_decoder only.", flush=True)
    return model


def make_case_batch(case_data, device, *, requires_grad: bool = False, zwd_override=None):
    return make_batch(
        case_data,
        device,
        requires_grad_surf=("zwd",) if requires_grad else (),
        zwd_override=zwd_override,
    )


def single_step_forward(model, batch):
    out = model.forward(batch)
    if isinstance(out, tuple):
        out = out[0]
    return out


def rollout_last(model, batch, steps: int):
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    last = None
    for pred in rollout_mean(model, batch, steps=steps):
        last = pred
    if last is None:
        raise RuntimeError("rollout_last did not produce any predictions")
    return last


def validate_rollout_cadence(
    model,
    case_data,
    device,
    *,
    expected_step_hours: float = DEFAULT_EXPECTED_STEP_HOURS,
    check_steps: int = 3,
) -> dict[str, Any]:
    batch = make_case_batch(case_data, device, requires_grad=False)
    observed_times = []
    with torch.no_grad():
        for idx, pred in enumerate(rollout_mean(model, batch, steps=check_steps), start=1):
            observed_times.append(pred.metadata.time[0])
            if idx >= check_steps:
                break
    if len(observed_times) < 2:
        raise RuntimeError("Could not validate cadence from fewer than two predicted timestamps")
    step_hours = [
        float((observed_times[i] - observed_times[i - 1]).total_seconds()) / 3600.0
        for i in range(1, len(observed_times))
    ]
    init_delta_hours = float((observed_times[0] - case_data.init_time).total_seconds()) / 3600.0
    all_hours = [init_delta_hours, *step_hours]
    observed_step_hours = float(np.median(all_hours))
    passes = bool(all(abs(h - expected_step_hours) < 1e-6 for h in all_hours))
    return {
        "init_time": case_data.init_time.isoformat(),
        "predicted_times": [t.isoformat() for t in observed_times],
        "step_hours": all_hours,
        "observed_step_hours": observed_step_hours,
        "expected_step_hours": expected_step_hours,
        "passes": passes,
    }


def lead_hours_to_steps(lead_hours: int, step_hours: float) -> int:
    steps = lead_hours / float(step_hours)
    rounded = int(round(steps))
    if abs(steps - rounded) > 1e-6:
        raise ValueError(
            f"Lead {lead_hours}h is not an integer multiple of cadence {step_hours}h"
        )
    if rounded <= 0:
        raise ValueError(f"Lead {lead_hours}h must map to at least one rollout step")
    return rounded


def _rectangular_window_mask(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_deg: float,
) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    dlat = lat_grid - center_lat
    dlon = lon_diff_deg(lon_grid, center_lon) * np.cos(np.radians(center_lat))
    return (np.abs(dlat) <= radius_deg) & (np.abs(dlon) <= radius_deg)


def track_storm_center(
    pred,
    *,
    prev_lat: float,
    prev_lon: float,
    window_radius_deg: float,
    intensity_radius_deg: float,
) -> dict[str, Any]:
    lat_vals = pred.metadata.lat.detach().cpu().numpy()
    lon_vals = pred.metadata.lon.detach().cpu().numpy()
    msl_hpa = pred.surf_vars["msl"][0, 0].detach().float().cpu().numpy() / 100.0
    u10 = pred.surf_vars["10u"][0, 0].detach().float().cpu().numpy()
    v10 = pred.surf_vars["10v"][0, 0].detach().float().cpu().numpy()
    wind10 = np.sqrt(np.square(u10) + np.square(v10))

    search_mask = _rectangular_window_mask(lat_vals, lon_vals, prev_lat, prev_lon, window_radius_deg)
    if not search_mask.any():
        search_mask = np.ones_like(msl_hpa, dtype=bool)

    search_indices = np.flatnonzero(search_mask.ravel())
    local_argmin = int(np.argmin(msl_hpa.ravel()[search_indices]))
    flat_idx = int(search_indices[local_argmin])
    lat_idx, lon_idx = np.unravel_index(flat_idx, msl_hpa.shape)

    center_lat = float(lat_vals[lat_idx])
    center_lon = float(lon_vals[lon_idx])
    min_msl_hpa = float(msl_hpa[lat_idx, lon_idx])

    intensity_mask = _rectangular_window_mask(
        lat_vals,
        lon_vals,
        center_lat,
        center_lon,
        intensity_radius_deg,
    )
    if not intensity_mask.any():
        intensity_mask = search_mask
    max_wind_ms = float(np.nanmax(wind10[intensity_mask]))

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_lon_plot": lon_to_plot_scalar(center_lon),
        "min_msl_hpa": min_msl_hpa,
        "max_wind10_ms": max_wind_ms,
        "msl_center_lat_idx": int(lat_idx),
        "msl_center_lon_idx": int(lon_idx),
    }


def _snapshot_from_pred(pred) -> dict[str, np.ndarray]:
    return {
        "lat_vals": pred.metadata.lat.detach().cpu().numpy(),
        "lon_vals": pred.metadata.lon.detach().cpu().numpy(),
        "msl_hpa": pred.surf_vars["msl"][0, 0].detach().float().cpu().numpy() / 100.0,
    }


def compute_baseline_rollout(
    model,
    case_cfg: TCCaseConfig,
    case_data,
    device,
    *,
    steps: int,
    snapshot_steps: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    batch = make_case_batch(case_data, device, requires_grad=False)
    rows: list[dict[str, Any]] = []
    snapshots_by_step: dict[int, dict[str, np.ndarray]] = {}
    prev_lat = case_cfg.init_lat
    prev_lon = case_cfg.init_lon

    with torch.no_grad():
        for step_idx, pred in enumerate(rollout_mean(model, batch, steps=steps), start=1):
            storm = track_storm_center(
                pred,
                prev_lat=prev_lat,
                prev_lon=prev_lon,
                window_radius_deg=case_cfg.track_window_radius_deg,
                intensity_radius_deg=case_cfg.intensity_radius_deg,
            )
            prev_lat = storm["center_lat"]
            prev_lon = storm["center_lon"]
            lead_hours = float((pred.metadata.time[0] - case_cfg.init_time).total_seconds()) / 3600.0
            rows.append({
                "step": step_idx,
                "lead_hours": lead_hours,
                "valid_time": pred.metadata.time[0].isoformat(),
                **storm,
            })
            if snapshot_steps and step_idx in snapshot_steps:
                snapshots_by_step[step_idx] = _snapshot_from_pred(pred)
    return pd.DataFrame(rows), snapshots_by_step


def make_soft_storm_target(
    case_data,
    *,
    center_lat: float,
    center_lon: float,
    window_radius_deg: float,
    target_kind: str,
    softmin_temp_hpa: float,
):
    mask_np = _rectangular_window_mask(
        case_data.lat_vals,
        case_data.lon_vals,
        center_lat,
        center_lon,
        window_radius_deg,
    )
    if not mask_np.any():
        raise ValueError("Storm target window is empty")

    lat_grid, lon_grid = np.meshgrid(case_data.lat_vals, case_data.lon_vals, indexing="ij")
    lat_masked_np = lat_grid[mask_np].astype(np.float32)
    lon_delta_np = lon_diff_deg(lon_grid[mask_np], center_lon).astype(np.float32)
    # Precompute full-grid tensors; target_fn crops to Aurora's output shape at
    # runtime since model.forward trims one latitude row (721→720).
    mask_t_full = torch.from_numpy(mask_np)
    lat_grid_full = torch.from_numpy(lat_grid.astype(np.float32))
    lon_delta_grid_full = torch.from_numpy(lon_diff_deg(lon_grid, center_lon).astype(np.float32))

    def target_fn(pred):
        msl = pred.surf_vars["msl"].float()[0, 0]
        device = msl.device
        H, W = msl.shape
        # Crop mask and coordinate grids to match Aurora's output spatial dims.
        m = mask_t_full[:H, :W].to(device)
        window_vals = msl[m] / 100.0
        weights = torch.softmax(-window_vals / softmin_temp_hpa, dim=0)
        lat_masked = lat_grid_full[:H, :W].to(device)[m]
        lon_delta_masked = lon_delta_grid_full[:H, :W].to(device)[m]

        if target_kind == "center_lat":
            return (weights * lat_masked).sum()
        if target_kind == "center_lon":
            return torch.tensor(center_lon, device=device, dtype=torch.float32) + (
                weights * lon_delta_masked
            ).sum()
        if target_kind == "intensity":
            return -(weights * window_vals).sum()
        raise ValueError(f"Unknown target_kind: {target_kind}")

    return target_fn


def compute_rollout_saliency(
    model,
    case_data,
    device,
    *,
    lead_steps: int,
    target_fn,
) -> tuple[np.ndarray, float, datetime]:
    # Mirror the working code path of _xia_saliency in searchlight_benchmark:
    #   - create a float32 batch with ZWD as a requires_grad leaf
    #   - let model.forward handle crop internally (no external pre-conversion)
    #   - call _forward (= model.forward) directly without rollout_mean overhead
    # Pre-converting to bfloat16 was tried and still OOMed at exactly the same
    # point as the typed approach; keeping float32 batch exactly matches the
    # searchlight saliency code path that provably fits in 95 GiB for 1 step.
    batch = make_case_batch(case_data, device, requires_grad=True)

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        print(f"[MEM before saliency fwd] allocated={alloc:.2f} GiB  reserved={reserv:.2f} GiB", flush=True)

    orig_autocast = getattr(model, "autocast", False)
    model.autocast = True

    def _ckpt_step(c):
        # Rollout-level checkpoint: saves only the step's input Batch, frees all
        # intermediate activations within Aurora's forward. Nested with the
        # Swin-block-level checkpoints applied by setup_model, so backward peak
        # memory ≈ 1 forward pass regardless of the number of rollout steps.
        return _forward(model, c)

    with torch.enable_grad():
        cur = batch
        pred = None
        for step in range(lead_steps):
            print(f"[SALIENCY] step={step}/{lead_steps}", flush=True)
            pred = torch_checkpoint(_ckpt_step, cur, use_reentrant=False)
            if step < lead_steps - 1:
                # model.forward crops the batch internally (721→720); use pred's
                # spatial shape to slice cur consistently so torch.cat doesn't fail.
                H, W = pred.surf_vars[next(iter(pred.surf_vars))].shape[-2:]
                cur = dataclasses.replace(
                    pred,
                    surf_vars={
                        k: torch.cat([cur.surf_vars[k][:, 1:, :H, :W], v], dim=1)
                        for k, v in pred.surf_vars.items()
                    },
                    atmos_vars={
                        k: torch.cat([cur.atmos_vars[k][:, 1:, :, :H, :W], v], dim=1)
                        for k, v in pred.atmos_vars.items()
                    },
                )
        score = target_fn(pred)
    score.float().backward()
    model.autocast = orig_autocast

    zwd_grad = batch.surf_vars["zwd"].grad
    if zwd_grad is None:
        raise RuntimeError("No ZWD gradient available for rollout saliency")
    saliency_map = zwd_grad.detach().float().cpu().numpy()[0, 1]
    score_val = float(score.detach().float().item())
    valid_time = pred.metadata.time[0]
    del batch, cur, pred, score
    _gpu_sync_and_gc()
    return saliency_map.astype(np.float32), score_val, valid_time


def compute_rollout_smoothgrad(
    model,
    case_data,
    device,
    *,
    lead_steps: int,
    target_fn,
    n_samples: int = 16,
    noise_sigma_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, float, object]:
    """SmoothGrad for ZWD through an autoregressive rollout.

    Averages signed rollout-saliency across n_samples copies of the input,
    each with independent Gaussian noise added to the ZWD channel (sigma =
    noise_sigma_frac * std(ZWD) on the case domain). Uses the same rollout-
    level checkpoint as compute_rollout_saliency so peak memory ~ 1 forward
    pass regardless of lead_steps or n_samples.

    Returns:
        smoothgrad_map -- signed mean gradient [H, W] for the t1 ZWD slice
        score_val      -- mean target score across the N samples
        valid_time     -- valid time of the prediction (last sample)
    """
    base_batch = make_case_batch(case_data, device, requires_grad=False)
    actual_zwd = base_batch.surf_vars["zwd"].detach()   # (1, 2, H, W) on device
    zwd_std = float(actual_zwd.std().item())
    sigma = noise_sigma_frac * zwd_std

    grad_accum = np.zeros(actual_zwd.shape, dtype=np.float64)
    scores: list[float] = []
    valid_time = None

    generator = torch.Generator(device=actual_zwd.device).manual_seed(seed)

    orig_autocast = getattr(model, "autocast", False)
    model.autocast = True

    def _ckpt_step(c):
        return _forward(model, c)

    for sample_i in range(n_samples):
        noise = torch.randn(
            actual_zwd.shape,
            generator=generator,
            dtype=actual_zwd.dtype,
            device=actual_zwd.device,
        ) * sigma
        zwd_noisy = (actual_zwd + noise).detach().clone().requires_grad_(True)

        batch = dataclasses.replace(
            base_batch,
            surf_vars={**base_batch.surf_vars, "zwd": zwd_noisy},
        )

        with torch.enable_grad():
            cur = batch
            pred = None
            for step in range(lead_steps):
                pred = torch_checkpoint(_ckpt_step, cur, use_reentrant=False)
                if step < lead_steps - 1:
                    H, W = pred.surf_vars[next(iter(pred.surf_vars))].shape[-2:]
                    cur = dataclasses.replace(
                        pred,
                        surf_vars={
                            k: torch.cat([cur.surf_vars[k][:, 1:, :H, :W], v], dim=1)
                            for k, v in pred.surf_vars.items()
                        },
                        atmos_vars={
                            k: torch.cat([cur.atmos_vars[k][:, 1:, :, :H, :W], v], dim=1)
                            for k, v in pred.atmos_vars.items()
                        },
                    )
            score = target_fn(pred)
        score.float().backward()

        g = zwd_noisy.grad
        if g is not None:
            grad_accum += g.detach().float().cpu().numpy()
        scores.append(float(score.detach().float().item()))
        valid_time = pred.metadata.time[0]

        del batch, cur, pred, score, zwd_noisy
        _gpu_sync_and_gc()
        print(f"  [SMOOTHGRAD] sample {sample_i + 1}/{n_samples} done", flush=True)

    model.autocast = orig_autocast

    mean_grad = (grad_accum / n_samples).astype(np.float32)
    smoothgrad_map = mean_grad[0, 1]   # signed, t1 slice
    mean_score = float(np.mean(scores))
    return smoothgrad_map, mean_score, valid_time


def compute_rollout_ig(
    model,
    case_data,
    device,
    *,
    lead_steps: int,
    target_fn,
    n_steps: int = 10,
    baseline_zwd: float = 0.0,
) -> tuple[np.ndarray, float, object]:
    """Integrated Gradients for ZWD through an autoregressive rollout.

    Integrates from baseline_zwd (uniform constant) to the actual ZWD field.
    Uses the same rollout-level checkpoint as compute_rollout_saliency so
    peak memory ≈ 1 forward pass regardless of lead_steps or n_steps.

    Returns:
        ig_map     -- signed IG attribution [H, W] for the t1 ZWD slice
        score_val  -- target score at the last alpha step
        valid_time -- valid time of the prediction
    """
    base_batch = make_case_batch(case_data, device, requires_grad=False)
    actual_zwd = base_batch.surf_vars["zwd"].detach()          # (1, 2, H, W) on device
    baseline = torch.full_like(actual_zwd, baseline_zwd)
    delta = actual_zwd - baseline                               # (x - x_bl)

    grad_accum = np.zeros(actual_zwd.shape, dtype=np.float64)
    score_val = None
    valid_time = None

    orig_autocast = getattr(model, "autocast", False)
    model.autocast = True

    def _ckpt_step(c):
        return _forward(model, c)

    for step_i in range(n_steps):
        alpha = (step_i + 0.5) / n_steps
        zwd_interp = (baseline + alpha * delta).detach().clone().requires_grad_(True)

        # Build batch with interpolated ZWD, keeping all other vars from actual.
        batch = dataclasses.replace(
            base_batch,
            surf_vars={**base_batch.surf_vars, "zwd": zwd_interp},
        )

        with torch.enable_grad():
            cur = batch
            pred = None
            for step in range(lead_steps):
                pred = torch_checkpoint(_ckpt_step, cur, use_reentrant=False)
                if step < lead_steps - 1:
                    H, W = pred.surf_vars[next(iter(pred.surf_vars))].shape[-2:]
                    cur = dataclasses.replace(
                        pred,
                        surf_vars={
                            k: torch.cat([cur.surf_vars[k][:, 1:, :H, :W], v], dim=1)
                            for k, v in pred.surf_vars.items()
                        },
                        atmos_vars={
                            k: torch.cat([cur.atmos_vars[k][:, 1:, :, :H, :W], v], dim=1)
                            for k, v in pred.atmos_vars.items()
                        },
                    )
            score = target_fn(pred)
        score.float().backward()

        g = zwd_interp.grad
        if g is not None:
            grad_accum += g.detach().float().cpu().numpy()

        if step_i == n_steps - 1:
            score_val = float(score.detach().float().item())
            valid_time = pred.metadata.time[0]

        del batch, cur, pred, score, zwd_interp
        _gpu_sync_and_gc()
        print(f"  [IG] step {step_i + 1}/{n_steps} done", flush=True)

    model.autocast = orig_autocast

    mean_grad = grad_accum / n_steps
    ig_map = (delta.cpu().numpy() * mean_grad)[0, 1]  # t1 slice
    return ig_map.astype(np.float32), score_val, valid_time


def generate_selection_masks(case_cfg: TCCaseConfig) -> list[MaskSpec]:
    west, east, south, north = case_cfg.analysis_extent
    lats = np.arange(south, north + 1e-6, case_cfg.selection_stride_deg)
    lons = np.arange(west, east + 1e-6, case_cfg.selection_stride_deg)
    specs: list[MaskSpec] = []
    mask_id = 0
    remote_count = 0

    for lat in lats:
        for lon_plot in lons:
            lon = lon_plot % 360.0
            distance_km = float(great_circle_km(case_cfg.init_lat, case_cfg.init_lon, lat, lon))
            if distance_km <= case_cfg.near_radius_km:
                role = "near"
            elif distance_km >= case_cfg.remote_min_km:
                if remote_count >= case_cfg.remote_max_count:
                    continue
                role = "remote"
                remote_count += 1
            else:
                continue
            specs.append(
                MaskSpec(
                    scale="storm",
                    role=role,
                    center_lat=float(lat),
                    center_lon=float(lon),
                    mask_id=mask_id,
                )
            )
            mask_id += 1
    return specs


def select_regions_from_saliency(
    *,
    case_cfg: TCCaseConfig,
    case_data,
    saliency_map: np.ndarray,
    masks: list[MaskSpec],
    low_quantile: float = 0.25,
) -> dict[str, dict[str, Any]]:
    weights = np.cos(np.radians(case_data.lat_vals)).astype(np.float32)
    weights = np.clip(weights, 0.0, None)
    weights_2d = np.broadcast_to(weights[:, None], saliency_map.shape)

    pooled_rows = []
    for spec in masks:
        mask = gaussian_mask(spec, case_cfg.selection_sigma_deg, case_data.lat_vals, case_data.lon_vals)
        w = mask * weights_2d
        pooled_mag = float((np.abs(saliency_map) * w).sum() / np.maximum(w.sum(), 1e-12))
        pooled_signed = float((saliency_map * w).sum() / np.maximum(w.sum(), 1e-12))
        distance_km = float(great_circle_km(case_cfg.init_lat, case_cfg.init_lon, spec.center_lat, spec.center_lon))
        pooled_rows.append({
            "spec": spec,
            "pooled_saliency_mag": pooled_mag,
            "pooled_saliency_signed": pooled_signed,
            "distance_km": distance_km,
        })

    near_rows = [row for row in pooled_rows if row["spec"].role == "near"]
    remote_rows = [row for row in pooled_rows if row["spec"].role == "remote"]
    if not near_rows or not remote_rows:
        raise ValueError("Need both near and remote masks for region selection")

    hotspot_row = max(near_rows, key=lambda row: row["pooled_saliency_mag"])
    hotspot_distance = hotspot_row["distance_km"]

    low_pool = [
        row for row in near_rows
        if row["spec"].key != hotspot_row["spec"].key
    ]
    if not low_pool:
        raise ValueError("No low-saliency near candidates available")
    mag_threshold = float(np.quantile([row["pooled_saliency_mag"] for row in low_pool], low_quantile))
    low_pool = [row for row in low_pool if row["pooled_saliency_mag"] <= mag_threshold] or low_pool
    low_row = min(
        low_pool,
        key=lambda row: (
            abs(row["distance_km"] - hotspot_distance),
            row["pooled_saliency_mag"],
        ),
    )
    remote_row = min(remote_rows, key=lambda row: row["pooled_saliency_mag"])

    def pack(label: str, row: dict[str, Any]) -> dict[str, Any]:
        spec = row["spec"]
        return {
            "label": label,
            "mask_key": spec.key,
            "role": spec.role,
            "center_lat": float(spec.center_lat),
            "center_lon": float(spec.center_lon),
            "center_lon_plot": lon_to_plot_scalar(spec.center_lon),
            "pooled_saliency_mag": float(row["pooled_saliency_mag"]),
            "pooled_saliency_signed": float(row["pooled_saliency_signed"]),
            "distance_km": float(row["distance_km"]),
            "sigma_deg": float(case_cfg.selection_sigma_deg),
        }

    return {
        "hotspot": pack("Hotspot", hotspot_row),
        "low_near": pack("Low-saliency near control", low_row),
        "remote": pack("Remote control", remote_row),
    }


def find_mask_by_key(masks: list[MaskSpec], key: str) -> MaskSpec:
    for spec in masks:
        if spec.key == key:
            return spec
    raise KeyError(f"Mask key {key!r} not found")


def build_perturbed_zwd(case_data, region_payload: dict[str, Any], amplitude_sigma: float) -> torch.Tensor:
    spec = MaskSpec(
        scale="storm",
        role=region_payload["role"],
        center_lat=float(region_payload["center_lat"]),
        center_lon=float(region_payload["center_lon"]),
        mask_id=0,
    )
    mask = gaussian_mask(spec, float(region_payload["sigma_deg"]), case_data.lat_vals, case_data.lon_vals)
    return perturb_zwd(
        case_data.surf_cpu["zwd"],
        mask,
        sign=1.0 if amplitude_sigma >= 0.0 else -1.0,
        magnitude=abs(float(amplitude_sigma)),
        zwd_loc=case_data.zwd_loc,
        zwd_scale=case_data.zwd_scale,
        timestep_idx=1,
    )


def run_perturbed_rollout(
    model,
    case_cfg: TCCaseConfig,
    case_data,
    device,
    *,
    steps: int,
    amplitude_sigma: float,
    region_payload: dict[str, Any],
    snapshot_steps: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    zwd_override = build_perturbed_zwd(case_data, region_payload, amplitude_sigma)
    batch = make_case_batch(case_data, device, requires_grad=False, zwd_override=zwd_override)

    rows: list[dict[str, Any]] = []
    snapshots_by_step: dict[int, dict[str, np.ndarray]] = {}
    prev_lat = case_cfg.init_lat
    prev_lon = case_cfg.init_lon

    with torch.no_grad():
        for step_idx, pred in enumerate(rollout_mean(model, batch, steps=steps), start=1):
            storm = track_storm_center(
                pred,
                prev_lat=prev_lat,
                prev_lon=prev_lon,
                window_radius_deg=case_cfg.track_window_radius_deg,
                intensity_radius_deg=case_cfg.intensity_radius_deg,
            )
            prev_lat = storm["center_lat"]
            prev_lon = storm["center_lon"]
            lead_hours = float((pred.metadata.time[0] - case_cfg.init_time).total_seconds()) / 3600.0
            rows.append({
                "step": step_idx,
                "lead_hours": lead_hours,
                "valid_time": pred.metadata.time[0].isoformat(),
                "region_kind": region_payload["label"],
                "amplitude_sigma": float(amplitude_sigma),
                **storm,
            })
            if snapshot_steps and step_idx in snapshot_steps:
                snapshots_by_step[step_idx] = _snapshot_from_pred(pred)
    return pd.DataFrame(rows), snapshots_by_step


def storm_relative_bin(
    field: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    *,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    n_radial_bins: int = 8,
    n_azimuth_bins: int = 16,
) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    dlon = lon_diff_deg(lon_grid, center_lon)
    x_km = dlon * np.cos(np.radians(center_lat)) * 111.0
    y_km = (lat_grid - center_lat) * 111.0
    r_km = np.sqrt(np.square(x_km) + np.square(y_km))
    az_deg = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0

    radial_edges = np.linspace(0.0, radius_km, n_radial_bins + 1)
    az_edges = np.linspace(0.0, 360.0, n_azimuth_bins + 1)
    out = np.full((n_radial_bins, n_azimuth_bins), np.nan, dtype=np.float32)

    for ridx in range(n_radial_bins):
        radial_mask = (r_km >= radial_edges[ridx]) & (r_km < radial_edges[ridx + 1])
        for aidx in range(n_azimuth_bins):
            az_mask = (az_deg >= az_edges[aidx]) & (az_deg < az_edges[aidx + 1])
            mask = radial_mask & az_mask
            if np.any(mask):
                out[ridx, aidx] = float(np.nanmean(field[mask]))
    return out


def choose_stamp_steps(
    lead_hours: list[int] | tuple[int, ...],
    *,
    step_hours: float,
) -> list[int]:
    return [lead_hours_to_steps(int(lead_h), step_hours) for lead_h in lead_hours]
