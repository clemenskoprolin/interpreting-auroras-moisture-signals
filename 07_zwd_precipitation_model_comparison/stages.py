"""
Stage implementations shared by the checkpoint comparison and routing code.

Stage A1: Model trajectory comparison (precip_zwd vs precip_only)
Stage A2: Intervention effects per model
Stage A3: Conditional differences (A-with - A-without)
Stage B is the final thesis saliency comparison. The remaining stages are
ancillary analysis primitives retained for compatibility with dependent
routing code; they are not part of the default publication run.

All stages save CSV/NPZ files to output_dir.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

for _p in (_SEARCHLIGHT_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparison_config import ModelSpec, TargetSpec, RolloutSpec, PRECIP_VAR  # noqa: E402
from comparison_data import (  # noqa: E402
    CaseData, _ensure_dir, _append_csv, _write_json,
    build_precip_batch,
)
from comparison_models import (  # noqa: E402
    rollout_independent, extract_scalar, _forward, _gpu_sync_and_gc,
)
from interventions import (  # noqa: E402
    build_all_interventions, make_removal_mask, apply_precip_removal,
    PRECIP_DISK_KM, PRECIP_TAPER_KM,
)
from metrics import spearman_corr, top_k_overlap, center_of_mass_displacement  # noqa: E402
from xia_methods.saliency import saliency as _xia_saliency  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clock() -> float:
    return time.perf_counter()


def _run_rollout_scalars(
    model,
    batch,
    model_spec: ModelSpec,
    rollout_spec: RolloutSpec,
    targets: list[TargetSpec],
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    target_region,
) -> dict[str, list[float]]:
    """Run rollout and extract scalar for each target at each lead time.

    Returns:
        Dict mapping target.name -> list of scalar values (one per rollout step,
        in order of rollout_spec.steps_hours).
    """
    from searchlight_tasks import TARGETS as _TARGETS

    max_steps = rollout_spec.max_steps
    with torch.no_grad():
        preds = rollout_independent(model, batch, max_steps)

    # steps_hours sorted
    step_hours = sorted(rollout_spec.steps_hours)
    scalars: dict[str, list[float]] = {t.name: [] for t in targets}

    for lead_h in step_hours:
        step_idx = lead_h // 6 - 1   # 0-based index into preds
        if step_idx < 0 or step_idx >= len(preds):
            for t in targets:
                scalars[t.name].append(float("nan"))
            continue
        pred = preds[step_idx]
        for t in targets:
            try:
                val = extract_scalar(pred, t, lat_vals, lon_vals, target_region)
            except Exception as e:
                print(f"    WARNING: extract_scalar failed for {t.name} at lead={lead_h}h: {e}")
                val = float("nan")
            scalars[t.name].append(val)

    return scalars


# ---------------------------------------------------------------------------
# Stage A1: trajectory comparison
# ---------------------------------------------------------------------------

def run_stage_a1_trajectories(
    model_w,
    model_wo,
    model_spec_w: ModelSpec,
    model_spec_wo: ModelSpec,
    cases: list[dict],
    rollout_spec: RolloutSpec,
    targets: list[TargetSpec],
    case_data_map: dict[str, "CaseData"],
    output_dir: str,
) -> list[dict]:
    """Run Stage A1: compute model trajectories for both models over all cases.

    For each case, runs both models' rollouts and records scalar values at each
    lead time. Computes M(h, y) = precip_zwd(h) - precip_only(h).

    Args:
        model_w: Loaded precip_zwd Aurora model.
        model_wo: Loaded precip_only Aurora model.
        model_spec_w: ModelSpec for precip_zwd.
        model_spec_wo: ModelSpec for precip_only.
        cases: List of case dicts (from cases_precipitation.json).
        rollout_spec: RolloutSpec controlling lead times.
        targets: List of TargetSpec to extract.
        case_data_map: Dict mapping init_time ISO string to CaseData.
        output_dir: Directory to write CSV files.

    Returns:
        List of row dicts (also written to stage_a_model_trajectories.csv).
    """
    from searchlight_tasks import TARGETS as _TARGETS

    _ensure_dir(output_dir)
    spatial_diff_dir = os.path.join(output_dir, "stage_a_spatial_differences")
    _ensure_dir(spatial_diff_dir)

    device = next(model_w.parameters()).device
    rows: list[dict] = []
    step_hours = sorted(rollout_spec.steps_hours)

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        grid_path = os.path.join(output_dir, "grid.npz")
        if not os.path.exists(grid_path):
            np.savez(
                grid_path,
                lat_vals=case_data.lat_vals,
                lon_vals=case_data.lon_vals,
            )
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING: unknown target {target_key!r}, skipping")
            continue

        precip_override = case_data.surf_cpu.get(PRECIP_VAR)

        print(f"  [A1] {case_id}")
        t0 = _clock()

        # Build batches
        batch_w = build_precip_batch(
            case_data, model_spec_w, device,
            precip_override=precip_override,
        )
        batch_wo = build_precip_batch(
            case_data, model_spec_wo, device,
            precip_override=precip_override,
        )

        # Run rollouts
        scalars_w = _run_rollout_scalars(
            model_w, batch_w, model_spec_w, rollout_spec, targets,
            case_data.lat_vals, case_data.lon_vals, target_region,
        )
        _gpu_sync_and_gc()

        scalars_wo = _run_rollout_scalars(
            model_wo, batch_wo, model_spec_wo, rollout_spec, targets,
            case_data.lat_vals, case_data.lon_vals, target_region,
        )
        _gpu_sync_and_gc()

        elapsed = _clock() - t0

        for t_spec in targets:
            for i, lead_h in enumerate(step_hours):
                val_w = scalars_w[t_spec.name][i]
                val_wo = scalars_wo[t_spec.name][i]
                M = val_w - val_wo if not (np.isnan(val_w) or np.isnan(val_wo)) else float("nan")
                rows.append({
                    "case_id": case_id,
                    "target": target_key,
                    "init_time": init_time_str,
                    "role": case_entry.get("role", "strong"),
                    "target_var": t_spec.name,
                    "lead_h": lead_h,
                    "score_with_zwd": val_w,
                    "score_without_zwd": val_wo,
                    "M_diff": M,
                    "elapsed_s": elapsed,
                })

        print(f"    done in {elapsed:.1f}s")

    _append_csv(os.path.join(output_dir, "stage_a_model_trajectories.csv"), rows)
    print(f"  [A1] wrote {len(rows)} rows to stage_a_model_trajectories.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage A2: intervention effects
# ---------------------------------------------------------------------------

def run_stage_a2_interventions(
    model,
    model_spec: ModelSpec,
    cases: list[dict],
    rollout_spec: RolloutSpec,
    targets: list[TargetSpec],
    case_data_map: dict[str, "CaseData"],
    doses_mm: tuple[float, ...],
    output_dir: str,
) -> list[dict]:
    """Run Stage A2: for each case, apply all precipitation interventions and record effects.

    For each case and intervention, runs a rollout and records the scalar output
    at each lead time. Results are keyed by (case_id, model_name, intervention, lead_h, target_var).

    Args:
        model: Loaded Aurora model.
        model_spec: ModelSpec for the model.
        cases: List of case dicts.
        rollout_spec: RolloutSpec controlling lead times.
        targets: List of TargetSpec to extract.
        case_data_map: Dict mapping init_time ISO string to CaseData.
        doses_mm: Dose magnitudes for intervention variants.
        output_dir: Directory to write CSV files.

    Returns:
        List of row dicts (also written to stage_a_intervention_effects.csv).
    """
    from searchlight_tasks import TARGETS as _TARGETS

    _ensure_dir(output_dir)
    device = next(model.parameters()).device
    rows: list[dict] = []
    step_hours = sorted(rollout_spec.steps_hours)

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING: unknown target {target_key!r}, skipping")
            continue

        precip_override = case_data.surf_cpu.get(PRECIP_VAR)
        if precip_override is None:
            print(f"  WARNING: no precipitation in case_data for {case_id}, "
                  "using zeros — provide MSWEP data via inject_precip_into_case()")
            H, W = case_data.lat_vals.shape[0], case_data.lon_vals.shape[0]
            precip_override = torch.zeros(1, 2, H, W, dtype=torch.float32)

        # Build interventions
        interventions = build_all_interventions(
            precip_override,
            case_data.lat_vals,
            case_data.lon_vals,
            center_lat=target_region.center_lat,
            center_lon=target_region.center_lon,
            doses_mm=doses_mm,
        )

        print(f"  [A2] {case_id} ({model_spec.name}) — {len(interventions)} interventions")

        for intv_name, precip_intv in interventions.items():
            t0 = _clock()
            batch = build_precip_batch(
                case_data, model_spec, device,
                precip_override=precip_intv,
            )
            scalars = _run_rollout_scalars(
                model, batch, model_spec, rollout_spec, targets,
                case_data.lat_vals, case_data.lon_vals, target_region,
            )
            _gpu_sync_and_gc()
            elapsed = _clock() - t0

            for t_spec in targets:
                for i, lead_h in enumerate(step_hours):
                    rows.append({
                        "case_id": case_id,
                        "model": model_spec.name,
                        "target": target_key,
                        "init_time": init_time_str,
                        "role": case_entry.get("role", "strong"),
                        "target_var": t_spec.name,
                        "intervention": intv_name,
                        "lead_h": lead_h,
                        "score": scalars[t_spec.name][i],
                        "elapsed_s": elapsed,
                    })

            print(f"    [{intv_name}] {elapsed:.1f}s")

    _append_csv(os.path.join(output_dir, "stage_a_intervention_effects.csv"), rows)
    print(f"  [A2] wrote {len(rows)} rows to stage_a_intervention_effects.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage A3: conditional difference
# ---------------------------------------------------------------------------

def run_stage_a3_conditional_diff(
    a2_rows_w: list[dict],
    a2_rows_wo: list[dict],
    output_dir: str,
) -> list[dict]:
    """Run Stage A3: compute conditional difference D(I, h, y) = E_w - E_wo.

    For each (case_id, intervention, lead_h, target_var) that appears in both
    A2 result sets, compute the difference D = score_with_zwd - score_without_zwd.

    Args:
        a2_rows_w: A2 rows from model with ZWD (precip_zwd).
        a2_rows_wo: A2 rows from model without ZWD (precip_only).
        output_dir: Directory to write CSV files.

    Returns:
        List of row dicts (also written to stage_a_conditional_differences.csv).
    """
    _ensure_dir(output_dir)

    # Index a2_rows_wo by (case_id, intervention, lead_h, target_var)
    wo_index: dict[tuple, float] = {}
    for row in a2_rows_wo:
        key = (row["case_id"], row["intervention"], row["lead_h"], row["target_var"])
        wo_index[key] = row["score"]

    rows: list[dict] = []
    for row_w in a2_rows_w:
        key = (row_w["case_id"], row_w["intervention"], row_w["lead_h"], row_w["target_var"])
        score_wo = wo_index.get(key, float("nan"))
        score_w = row_w["score"]
        D = score_w - score_wo if not (np.isnan(score_w) or np.isnan(score_wo)) else float("nan")

        # Baseline scores for "actual" intervention
        actual_key_w = (row_w["case_id"], "actual", row_w["lead_h"], row_w["target_var"])
        actual_key_wo = (row_w["case_id"], "actual", row_w["lead_h"], row_w["target_var"])
        score_actual_w = wo_index.get(actual_key_w, float("nan"))  # will update below
        score_actual_wo = wo_index.get(actual_key_wo, float("nan"))

        rows.append({
            "case_id": row_w["case_id"],
            "target": row_w["target"],
            "init_time": row_w["init_time"],
            "role": row_w.get("role", "strong"),
            "target_var": row_w["target_var"],
            "intervention": row_w["intervention"],
            "lead_h": row_w["lead_h"],
            "score_with_zwd": score_w,
            "score_without_zwd": score_wo,
            "D_conditional_diff": D,
        })

    _append_csv(os.path.join(output_dir, "stage_a_conditional_differences.csv"), rows)
    print(f"  [A3] wrote {len(rows)} rows to stage_a_conditional_differences.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage B: attribution/saliency reliance maps
# ---------------------------------------------------------------------------

def run_stage_b(
    model,
    model_spec: ModelSpec,
    cases: list[dict],
    case_data_map: dict[str, "CaseData"],
    targets: list[TargetSpec],
    diagnostic_case_ids: list[str],
    output_dir: str,
    lead_h: int = 6,
) -> list[dict]:
    """Stage B: Attribution/saliency reliance maps.

    For each diagnostic case, model, and target variable, compute saliency maps
    w.r.t. the precipitation input channel (and ZWD if available), then compute
    rerouting metrics between the two saliency maps.

    Outputs:
      - stage_b_reliance_maps/{case_id}/{model_name}_{var}_{target_var}_{lead_h}h.npy
      - stage_b_reliance_summary.csv
    """
    from searchlight_tasks import TARGETS as _TARGETS, box_indices

    _ensure_dir(os.path.join(output_dir, "stage_b_reliance_maps"))
    rows: list[dict] = []
    device = next(model.parameters()).device
    lead_steps = lead_h // 6

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        if case_id not in diagnostic_case_ids:
            continue

        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING [B]: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING [B]: unknown target {target_key!r}, skipping")
            continue

        lat_imin, lat_imax, lon_imin, lon_imax = box_indices(
            target_region, case_data.lat_vals, case_data.lon_vals
        )

        map_dir = os.path.join(output_dir, "stage_b_reliance_maps", case_id)
        _ensure_dir(map_dir)

        for t_spec in targets:
            print(f"  [B] {case_id} model={model_spec.name} target={t_spec.name}")
            t0 = _clock()

            # Build target_fn: scalar box mean of predicted variable
            if t_spec.level_hpa is not None:
                # Atmospheric variable: extract q (or generic) at level
                _level_hpa = t_spec.level_hpa
                _out_var = t_spec.output_var

                def _target_fn(pred, _lh=_level_hpa, _ov=_out_var,
                               _lamin=lat_imin, _lamax=lat_imax,
                               _lnmin=lon_imin, _lnmax=lon_imax,
                               _case=case_data):
                    levels_arr = np.asarray(pred.metadata.atmos_levels)
                    matches = np.where(levels_arr == _lh)[0]
                    if matches.size == 0:
                        raise ValueError(f"Level {_lh} hPa not found")
                    lidx = int(matches[0])
                    arr = pred.atmos_vars[_ov].float()
                    return arr[0, 0, lidx, _lamin:_lamax + 1, _lnmin:_lnmax + 1].mean()
            else:
                # Surface variable
                _out_var = t_spec.output_var

                def _target_fn(pred, _ov=_out_var,
                               _lamin=lat_imin, _lamax=lat_imax,
                               _lnmin=lon_imin, _lnmax=lon_imax):
                    if _ov not in pred.surf_vars:
                        return pred.surf_vars[list(pred.surf_vars.keys())[0]].float().mean() * 0.0
                    arr = pred.surf_vars[_ov].float()
                    return arr[0, 0, _lamin:_lamax + 1, _lnmin:_lnmax + 1].mean()

            # If lead > 6h: wrap model in a rollout wrapper so saliency goes through
            # the full rollout. For now we do single-step saliency (lead_h == 6).
            # For multi-step: build batch, do grad through rollout (expensive).
            # We keep it at single step for Stage B.

            # --- Precip saliency ---
            precip_override = case_data.surf_cpu.get(PRECIP_VAR)
            precip_saliency = None

            def _batch_fn_precip(requires_grad, _cd=case_data, _ms=model_spec,
                                  _dev=device, _po=precip_override):
                return build_precip_batch(
                    _cd, _ms, _dev,
                    precip_override=_po,
                    requires_grad_surf=(PRECIP_VAR,) if requires_grad else (),
                )

            try:
                result_p = _xia_saliency(
                    model, _batch_fn_precip, _target_fn, str(device),
                    surf_var_names=(PRECIP_VAR,),
                )
                grad_p = result_p["grads"].get(PRECIP_VAR)
                if grad_p is not None:
                    # grad_p shape: (1, 2, H, W) — take t1 slice
                    precip_saliency = np.abs(grad_p[0, 1])  # (H, W)
                    npy_path = os.path.join(
                        map_dir,
                        f"{model_spec.name}_{PRECIP_VAR}_{t_spec.name}_{lead_h}h.npy",
                    )
                    np.save(npy_path, precip_saliency)
            except Exception as e:
                print(f"    WARNING [B]: precip saliency failed: {e}")
            _gpu_sync_and_gc()

            # --- ZWD saliency (only if model has ZWD) ---
            zwd_saliency = None
            if model_spec.has_zwd:
                def _batch_fn_zwd(requires_grad, _cd=case_data, _ms=model_spec,
                                  _dev=device, _po=precip_override):
                    return build_precip_batch(
                        _cd, _ms, _dev,
                        precip_override=_po,
                        requires_grad_surf=("zwd",) if requires_grad else (),
                    )

                try:
                    result_z = _xia_saliency(
                        model, _batch_fn_zwd, _target_fn, str(device),
                        surf_var_names=("zwd",),
                    )
                    grad_z = result_z["grads"].get("zwd")
                    if grad_z is not None:
                        zwd_saliency = np.abs(grad_z[0, 1])  # (H, W), t1 slice
                        npy_path = os.path.join(
                            map_dir,
                            f"{model_spec.name}_zwd_{t_spec.name}_{lead_h}h.npy",
                        )
                        np.save(npy_path, zwd_saliency)
                except Exception as e:
                    print(f"    WARNING [B]: zwd saliency failed: {e}")
                _gpu_sync_and_gc()

            elapsed = _clock() - t0

            # --- Metrics ---
            row: dict = {
                "case_id": case_id,
                "model": model_spec.name,
                "target": target_key,
                "init_time": init_time_str,
                "role": case_entry.get("role", "strong"),
                "target_var": t_spec.name,
                "lead_h": lead_h,
                "elapsed_s": elapsed,
                "spearman_r_precip_zwd": float("nan"),
                "top1pct_overlap": float("nan"),
                "com_displacement_km": float("nan"),
                "target_box_mass_precip": float("nan"),
                "target_box_mass_zwd": float("nan"),
            }

            # Box mass fraction: fraction of saliency inside target box
            if precip_saliency is not None:
                total_p = precip_saliency.sum()
                if total_p > 1e-30:
                    box_mass = float(
                        precip_saliency[lat_imin:lat_imax + 1, lon_imin:lon_imax + 1].sum()
                        / total_p
                    )
                    row["target_box_mass_precip"] = box_mass

            if zwd_saliency is not None:
                total_z = zwd_saliency.sum()
                if total_z > 1e-30:
                    box_mass_z = float(
                        zwd_saliency[lat_imin:lat_imax + 1, lon_imin:lon_imax + 1].sum()
                        / total_z
                    )
                    row["target_box_mass_zwd"] = box_mass_z

            # Cross-map metrics (when both available)
            if precip_saliency is not None and zwd_saliency is not None:
                try:
                    rho, _ = spearman_corr(precip_saliency.ravel(), zwd_saliency.ravel())
                    row["spearman_r_precip_zwd"] = rho
                except Exception:
                    pass
                try:
                    row["top1pct_overlap"] = top_k_overlap(
                        precip_saliency.ravel(), zwd_saliency.ravel(), k_frac=0.01
                    )
                except Exception:
                    pass
                try:
                    row["com_displacement_km"] = center_of_mass_displacement(
                        precip_saliency, zwd_saliency,
                        case_data.lat_vals, case_data.lon_vals,
                    )
                except Exception:
                    pass

            rows.append(row)
            print(f"    done in {elapsed:.1f}s  spearman_r={row['spearman_r_precip_zwd']:.3f}")

    if rows:
        _append_csv(os.path.join(output_dir, "stage_b_reliance_summary.csv"), rows)
    print(f"  [B] wrote {len(rows)} rows to stage_b_reliance_summary.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage C: timing analysis
# ---------------------------------------------------------------------------

def run_stage_c(
    model,
    model_spec: ModelSpec,
    cases: list[dict],
    case_data_map: dict[str, "CaseData"],
    targets: list[TargetSpec],
    diagnostic_case_ids: list[str],
    output_dir: str,
    leads_hours: list[int] = (6, 12, 18, 24),
) -> list[dict]:
    """Stage C: Timing analysis — t0 vs t1 split for precipitation.

    For each diagnostic case, target, and lead time, runs 4 forward passes:
      1. "actual":      unmodified precipitation
      2. "remove_both": remove precipitation at both t0 and t1
      3. "remove_t0":   remove at t0 only
      4. "remove_t1":   remove at t1 only

    Records delta_both, delta_t0, delta_t1, and t1_fraction = delta_t1 / delta_both.

    Output: stage_c_timing_analysis.csv
    """
    from searchlight_tasks import TARGETS as _TARGETS

    _ensure_dir(output_dir)
    device = next(model.parameters()).device
    rows: list[dict] = []

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        if case_id not in diagnostic_case_ids:
            continue

        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING [C]: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING [C]: unknown target {target_key!r}, skipping")
            continue

        precip_override = case_data.surf_cpu.get(PRECIP_VAR)
        if precip_override is None:
            H, W = case_data.lat_vals.shape[0], case_data.lon_vals.shape[0]
            precip_override = torch.zeros(1, 2, H, W, dtype=torch.float32)

        removal_mask = make_removal_mask(
            case_data.lat_vals, case_data.lon_vals,
            center_lat=target_region.center_lat,
            center_lon=target_region.center_lon,
        )

        precip_remove_both = apply_precip_removal(precip_override, removal_mask, timestep=-1)
        precip_remove_t0 = apply_precip_removal(precip_override, removal_mask, timestep=0)
        precip_remove_t1 = apply_precip_removal(precip_override, removal_mask, timestep=1)

        print(f"  [C] {case_id} model={model_spec.name}")

        max_lead = max(leads_hours)
        max_steps = max_lead // 6

        rollout_spec_local = type("_RS", (), {
            "steps_hours": tuple(leads_hours),
            "max_steps": max_steps,
        })()

        t0 = _clock()
        batch_actual = build_precip_batch(case_data, model_spec, device, precip_override=precip_override)
        scalars_actual = _run_rollout_scalars(model, batch_actual, model_spec, rollout_spec_local,
                                              targets, case_data.lat_vals, case_data.lon_vals, target_region)
        _gpu_sync_and_gc()

        batch_both = build_precip_batch(case_data, model_spec, device, precip_override=precip_remove_both)
        scalars_both = _run_rollout_scalars(model, batch_both, model_spec, rollout_spec_local,
                                            targets, case_data.lat_vals, case_data.lon_vals, target_region)
        _gpu_sync_and_gc()

        batch_t0 = build_precip_batch(case_data, model_spec, device, precip_override=precip_remove_t0)
        scalars_t0 = _run_rollout_scalars(model, batch_t0, model_spec, rollout_spec_local,
                                          targets, case_data.lat_vals, case_data.lon_vals, target_region)
        _gpu_sync_and_gc()

        batch_t1 = build_precip_batch(case_data, model_spec, device, precip_override=precip_remove_t1)
        scalars_t1 = _run_rollout_scalars(model, batch_t1, model_spec, rollout_spec_local,
                                          targets, case_data.lat_vals, case_data.lon_vals, target_region)
        _gpu_sync_and_gc()

        elapsed = _clock() - t0

        step_hours = sorted(leads_hours)
        for t_spec in targets:
            for i, lead_h in enumerate(step_hours):
                s_actual = scalars_actual[t_spec.name][i]
                s_both   = scalars_both[t_spec.name][i]
                s_t0     = scalars_t0[t_spec.name][i]
                s_t1     = scalars_t1[t_spec.name][i]

                delta_both = s_actual - s_both if not (np.isnan(s_actual) or np.isnan(s_both)) else float("nan")
                delta_t0   = s_actual - s_t0   if not (np.isnan(s_actual) or np.isnan(s_t0))   else float("nan")
                delta_t1   = s_actual - s_t1   if not (np.isnan(s_actual) or np.isnan(s_t1))   else float("nan")

                if not np.isnan(delta_both) and abs(delta_both) > 1e-12:
                    t1_frac = delta_t1 / delta_both
                else:
                    t1_frac = float("nan")

                rows.append({
                    "case_id": case_id,
                    "model": model_spec.name,
                    "target": target_key,
                    "init_time": init_time_str,
                    "role": case_entry.get("role", "strong"),
                    "target_var": t_spec.name,
                    "lead_h": lead_h,
                    "score_actual": s_actual,
                    "score_remove_both": s_both,
                    "score_remove_t0only": s_t0,
                    "score_remove_t1only": s_t1,
                    "delta_both": delta_both,
                    "delta_t0_only": delta_t0,
                    "delta_t1_only": delta_t1,
                    "t1_fraction_of_both": t1_frac,
                    "elapsed_s": elapsed,
                })

        print(f"    done in {elapsed:.1f}s")

    if rows:
        _append_csv(os.path.join(output_dir, "stage_c_timing_analysis.csv"), rows)
    print(f"  [C] wrote {len(rows)} rows to stage_c_timing_analysis.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage D: routing analysis
# ---------------------------------------------------------------------------

def _build_block_specs(model):
    """Build list of (module, key, family) tuples for all encoder/decoder blocks."""
    specs = []
    for layer_idx, layer in enumerate(model.backbone.encoder_layers):
        for block_idx, blk in enumerate(layer.blocks):
            key = f"enc_s{layer_idx}_b{block_idx:02d}"
            specs.append((blk, key, "encoder"))
    for layer_idx, layer in enumerate(model.backbone.decoder_layers):
        for block_idx, blk in enumerate(layer.blocks):
            key = f"dec_s{layer_idx}_b{block_idx:02d}"
            specs.append((blk, key, "decoder"))
    return specs


def _collect_hidden_states_precip(model, batch, block_specs) -> dict[str, torch.Tensor]:
    """Run one forward pass and collect block output tensors via hooks."""
    from comparison_models import _forward as _fwd

    captures: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(key):
        def _hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[key] = t.detach().float().cpu()
        return _hook

    for mod, key, _ in block_specs:
        handles.append(mod.register_forward_hook(_make_hook(key)))
    try:
        with torch.no_grad():
            _fwd(model, batch)
    finally:
        for h in handles:
            h.remove()
    return captures


def run_stage_d(
    model,
    model_spec: ModelSpec,
    cases: list[dict],
    case_data_map: dict[str, "CaseData"],
    diagnostic_case_ids: list[str],
    output_dir: str,
) -> list[dict]:
    """Stage D: Routing analysis — hidden-state RMS + cosine similarity.

    For each diagnostic case, runs two forward passes (actual vs remove_both)
    and at each Swin3D block computes:
      - baseline_rms: RMS of h_actual
      - delta_rms: RMS of (h_removed - h_actual)
      - relative_rms: delta_rms / (baseline_rms + 1e-8)
      - cosine_sim: cosine similarity between h_actual and h_removed

    Output: stage_d_routing_analysis.csv
    """
    from searchlight_tasks import TARGETS as _TARGETS

    _ensure_dir(output_dir)
    device = next(model.parameters()).device
    rows: list[dict] = []
    block_specs = _build_block_specs(model)

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        if case_id not in diagnostic_case_ids:
            continue

        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING [D]: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING [D]: unknown target {target_key!r}, skipping")
            continue

        precip_override = case_data.surf_cpu.get(PRECIP_VAR)
        if precip_override is None:
            H, W = case_data.lat_vals.shape[0], case_data.lon_vals.shape[0]
            precip_override = torch.zeros(1, 2, H, W, dtype=torch.float32)

        removal_mask = make_removal_mask(
            case_data.lat_vals, case_data.lon_vals,
            center_lat=target_region.center_lat,
            center_lon=target_region.center_lon,
        )
        precip_removed = apply_precip_removal(precip_override, removal_mask, timestep=-1)

        print(f"  [D] {case_id} model={model_spec.name} — collecting hidden states...")
        t0 = _clock()

        batch_actual = build_precip_batch(case_data, model_spec, device, precip_override=precip_override)
        states_actual = _collect_hidden_states_precip(model, batch_actual, block_specs)
        del batch_actual
        _gpu_sync_and_gc()

        batch_removed = build_precip_batch(case_data, model_spec, device, precip_override=precip_removed)
        states_removed = _collect_hidden_states_precip(model, batch_removed, block_specs)
        del batch_removed
        _gpu_sync_and_gc()

        elapsed = _clock() - t0

        for _, key, family in block_specs:
            if key not in states_actual or key not in states_removed:
                continue
            h_actual = states_actual[key]
            h_removed = states_removed[key]
            delta = h_removed - h_actual

            baseline_rms = float(torch.sqrt((h_actual ** 2).mean()).item())
            delta_rms = float(torch.sqrt((delta ** 2).mean()).item())
            relative_rms = delta_rms / (baseline_rms + 1e-8)

            h_f = h_actual.reshape(-1).double()
            r_f = h_removed.reshape(-1).double()
            norm_h = torch.norm(h_f).item()
            norm_r = torch.norm(r_f).item()
            if norm_h > 1e-12 and norm_r > 1e-12:
                cosine_sim = float((torch.dot(h_f, r_f) / (norm_h * norm_r)).item())
            else:
                cosine_sim = float("nan")

            rows.append({
                "case_id": case_id,
                "model": model_spec.name,
                "target": target_key,
                "init_time": init_time_str,
                "role": case_entry.get("role", "strong"),
                "block_key": key,
                "family": family,
                "baseline_rms": baseline_rms,
                "delta_rms": delta_rms,
                "relative_rms": relative_rms,
                "cosine_sim": cosine_sim,
                "elapsed_s": elapsed,
            })

        print(f"    done in {elapsed:.1f}s  ({len(block_specs)} blocks)")

    if rows:
        _append_csv(os.path.join(output_dir, "stage_d_routing_analysis.csv"), rows)
    print(f"  [D] wrote {len(rows)} rows to stage_d_routing_analysis.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage E: spatial restoration
# ---------------------------------------------------------------------------

def _make_restoration_masks(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    lat_imin: int,
    lat_imax: int,
    lon_imin: int,
    lon_imax: int,
    removal_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build binary restoration masks (1 = restore, 0 = leave removed).

    Each mask M is applied as:
      precip_restored = precip_removed + M * (precip_actual - precip_removed)

    Returns dict of mask_name -> (H, W) float32 binary mask.
    """
    H, W = lat_vals.shape[0], lon_vals.shape[0]

    # The removal region is where removal_mask < 0.5 (inside the disk)
    in_disk = (removal_mask < 0.5).astype(np.float32)  # 1 = inside disk

    # target_box: restore only inside the target box
    target_box = np.zeros((H, W), dtype=np.float32)
    target_box[lat_imin:lat_imax + 1, lon_imin:lon_imax + 1] = 1.0
    target_box = target_box * in_disk  # only within the removal region

    # disk_500km: restore within 500 km of target center
    from interventions import _great_circle_km
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    dlon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    lon_adj = center_lon + dlon
    dist = _great_circle_km(center_lat, center_lon, lat_grid, lon_adj)

    disk_500 = (dist <= 500.0).astype(np.float32) * in_disk
    disk_1000 = (dist <= 1000.0).astype(np.float32) * in_disk

    # full_t1_only: restore entire removal disk (for t1 only — handled in rollout)
    full_disk = in_disk.copy()

    return {
        "target_box": target_box,
        "disk_500km": disk_500,
        "disk_1000km": disk_1000,
        "full_t1_only": full_disk,  # applied only to t1 timestep
    }


def run_stage_e(
    model,
    model_spec: ModelSpec,
    cases: list[dict],
    case_data_map: dict[str, "CaseData"],
    targets: list[TargetSpec],
    diagnostic_case_ids: list[str],
    output_dir: str,
    leads_hours: list[int] = (6, 12, 18, 24),
) -> list[dict]:
    """Stage E: Spatial restoration — localized precipitation restoration.

    For each diagnostic case, model, target, and lead:
      1. Compute baseline scores: score_actual and score_removed (remove_both)
      2. For each restoration mask, restore precip in that sub-region and re-run rollout
      3. Compute fraction_restored = (score_restored - score_removed) / (score_actual - score_removed)

    Restoration masks: target_box, disk_500km, disk_1000km, full_t1_only.

    Output: stage_e_spatial_restoration.csv
    """
    from searchlight_tasks import TARGETS as _TARGETS, box_indices

    _ensure_dir(output_dir)
    device = next(model.parameters()).device
    rows: list[dict] = []

    for case_entry in cases:
        case_id = f"{case_entry['target']}_{case_entry['init_time']}"
        if case_id not in diagnostic_case_ids:
            continue

        target_key = case_entry["target"]
        init_time_str = case_entry["init_time"]

        if init_time_str not in case_data_map:
            print(f"  WARNING [E]: no CaseData for {init_time_str}, skipping {case_id}")
            continue

        case_data = case_data_map[init_time_str]
        target_region = _TARGETS.get(target_key)
        if target_region is None:
            print(f"  WARNING [E]: unknown target {target_key!r}, skipping")
            continue

        precip_actual = case_data.surf_cpu.get(PRECIP_VAR)
        if precip_actual is None:
            H, W = case_data.lat_vals.shape[0], case_data.lon_vals.shape[0]
            precip_actual = torch.zeros(1, 2, H, W, dtype=torch.float32)

        removal_mask = make_removal_mask(
            case_data.lat_vals, case_data.lon_vals,
            center_lat=target_region.center_lat,
            center_lon=target_region.center_lon,
        )
        precip_removed = apply_precip_removal(precip_actual, removal_mask, timestep=-1)

        lat_imin, lat_imax, lon_imin, lon_imax = box_indices(
            target_region, case_data.lat_vals, case_data.lon_vals
        )

        restoration_masks = _make_restoration_masks(
            case_data.lat_vals, case_data.lon_vals,
            center_lat=target_region.center_lat,
            center_lon=target_region.center_lon,
            lat_imin=lat_imin, lat_imax=lat_imax,
            lon_imin=lon_imin, lon_imax=lon_imax,
            removal_mask=removal_mask,
        )

        print(f"  [E] {case_id} model={model_spec.name}")
        t0 = _clock()

        max_lead = max(leads_hours)
        max_steps = max_lead // 6

        rollout_spec_local = type("_RS", (), {
            "steps_hours": tuple(leads_hours),
            "max_steps": max_steps,
        })()

        # Baseline: actual
        batch_actual = build_precip_batch(case_data, model_spec, device, precip_override=precip_actual)
        scalars_actual = _run_rollout_scalars(
            model, batch_actual, model_spec, rollout_spec_local,
            targets, case_data.lat_vals, case_data.lon_vals, target_region,
        )
        _gpu_sync_and_gc()

        # Baseline: removed
        batch_removed = build_precip_batch(case_data, model_spec, device, precip_override=precip_removed)
        scalars_removed = _run_rollout_scalars(
            model, batch_removed, model_spec, rollout_spec_local,
            targets, case_data.lat_vals, case_data.lon_vals, target_region,
        )
        _gpu_sync_and_gc()

        # For each restoration mask
        restoration_scalars: dict[str, dict] = {}
        for mask_name, mask_hw in restoration_masks.items():
            mask_t = torch.from_numpy(mask_hw)  # (H, W)

            if mask_name == "full_t1_only":
                # Restore only t1
                precip_r = precip_removed.clone()
                diff = precip_actual - precip_removed  # (1, 2, H, W)
                precip_r[0, 1] = precip_r[0, 1] + mask_t * diff[0, 1]
            else:
                # Restore both timesteps within the mask
                precip_r = precip_removed.clone()
                diff = precip_actual - precip_removed
                precip_r[0, 0] = precip_r[0, 0] + mask_t * diff[0, 0]
                precip_r[0, 1] = precip_r[0, 1] + mask_t * diff[0, 1]

            batch_r = build_precip_batch(case_data, model_spec, device, precip_override=precip_r)
            scalars_r = _run_rollout_scalars(
                model, batch_r, model_spec, rollout_spec_local,
                targets, case_data.lat_vals, case_data.lon_vals, target_region,
            )
            restoration_scalars[mask_name] = scalars_r
            _gpu_sync_and_gc()

        elapsed = _clock() - t0

        step_hours = sorted(leads_hours)
        for t_spec in targets:
            for i, lead_h in enumerate(step_hours):
                s_actual  = scalars_actual[t_spec.name][i]
                s_removed = scalars_removed[t_spec.name][i]
                denom = s_actual - s_removed

                for mask_name in restoration_masks:
                    s_restored = restoration_scalars[mask_name][t_spec.name][i]

                    if not np.isnan(s_restored) and not np.isnan(s_removed) and not np.isnan(s_actual):
                        frac = (s_restored - s_removed) / (denom + 1e-10)
                    else:
                        frac = float("nan")

                    rows.append({
                        "case_id": case_id,
                        "model": model_spec.name,
                        "target": target_key,
                        "init_time": init_time_str,
                        "role": case_entry.get("role", "strong"),
                        "target_var": t_spec.name,
                        "lead_h": lead_h,
                        "restoration_name": mask_name,
                        "score_actual": s_actual,
                        "score_removed": s_removed,
                        "score_restored": s_restored,
                        "fraction_restored": frac,
                        "elapsed_s": elapsed,
                    })

        print(f"    done in {elapsed:.1f}s")

    if rows:
        _append_csv(os.path.join(output_dir, "stage_e_spatial_restoration.csv"), rows)
    print(f"  [E] wrote {len(rows)} rows to stage_e_spatial_restoration.csv")
    return rows
