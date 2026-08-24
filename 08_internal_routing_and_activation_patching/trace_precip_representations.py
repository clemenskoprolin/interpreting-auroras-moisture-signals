"""
Trace precipitation-model representations with paired perturbations.

Modes
-----
zwd_trace           precip_zwd only; paired ZWD +/- 1 sigma Gaussian perturbations
precip_trace        both models; paired tp_mswep +/- dose via hotspot Gaussian mask
q850_trace          both models; paired q(850 hPa) +/- 1 sigma Gaussian perturbations.
                    The reverse direction of the traces above: q850 is the source and
                    precip / zwd are the targets (use --targets precip zwd).
precip_remove_trace both models; remove_t1 deletion (non-symmetric routing trace)
factorial           precip_zwd only; 4-way ZWD x precip factorial interaction
cka                 linear CKA between model representations at each block
probes              ridge probes on stage-terminal pooled hidden vectors

Cross-model hidden-state subtraction is NEVER performed.  cka and probes compare
models through derived metrics only (CKA similarity scores and probe R² values).

Region selection uses Stage B precipitation saliency maps from
07_zwd_precipitation_model_comparison.  If those maps are missing the script fails clearly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_MCC_DIR = os.path.join(_ROOT, "07_zwd_precipitation_model_comparison")
_P_CORR_DIR = os.path.join(_ROOT, "06_precipitation_moisture_relationships")
_COND_DIR = os.path.join(_ROOT, "04_zwd_counterfactual_interventions")

for _p in (_SEARCHLIGHT_DIR, _MCC_DIR, _P_CORR_DIR, _COND_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from searchlight_data import CaseData, load_case  # noqa: E402
from searchlight_tasks import (  # noqa: E402
    SCALES, TARGETS,
    MaskSpec, ScaleConfig,
    generate_mask_centers, gaussian_mask, great_circle_km, cos_lat_weights,
    box_indices,
)
from comparison_config import (  # noqa: E402
    DEFAULT_MODELS, DEFAULT_TARGETS, PRECIP_VAR, MSWEP_STORE_PATH,
    ModelSpec, TargetSpec,
)
from comparison_data import (  # noqa: E402
    build_precip_batch, inject_precip_into_case, load_precip_for_case,
)
from comparison_models import (  # noqa: E402
    load_model, extract_scalar, _forward, _gpu_sync_and_gc,
)
from interventions import (  # noqa: E402
    make_removal_mask, apply_precip_removal, apply_precip_dose,
)
from searchlight_ground_truth import perturb_zwd, smoothed_zwd_baseline  # noqa: E402
from conditional_data import make_qhat_zwd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULTS_ROOT = os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results"))
DEFAULT_MODEL_CONDITIONAL_DIR = os.path.join(
    _RESULTS_ROOT, "model_conditional_comparison",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    _RESULTS_ROOT, "internal_routing",
)
DEFAULT_MODES = ("zwd_trace", "precip_trace", "q850_trace", "factorial", "cka")
DEFAULT_TARGETS_CLI = ("q850", "precip", "zwd")
DEFAULT_SCALES = ("local", "synoptic")
ALL_MODES = ("zwd_trace", "precip_trace", "q850_trace", "precip_remove_trace", "factorial",
             "cka", "probes", "zwd_smooth_removal", "zwd_ivw_removal")

ZWD_SMOOTH_SIGMA_DEG = 10.0

STAGE_B_SALIENCY_SUBDIR = "stage_b_reliance_maps"
STAGE_B_LEAD_H = 6
STAGE_B_MODEL_NAME = "precip_large_zwd"

# Paired perturbation defaults.
# ZWD uses 1σ from the global climatological normalization stats (zwd_scale = 98.54 mm).
# Precipitation uses the equivalent global climatological σ from the same stats file
# (tp_mswep scale = 2.11 mm/6h), making the two perturbations directly comparable as
# "one climatological standard deviation" of their respective variable.
ZWD_MAGNITUDE = 1.0     # sigma_zwd units (dimensionless, applied to zwd_scale)
PRECIP_DOSE_MM = 2.11   # mm/6h  = 1 sigma_tp_mswep from normalization_stats_1979_2021.json
Q850_MAGNITUDE = 1.0    # sigma_q850 units (applied to the q_850 scale from the same stats)
Q850_LEVEL_HPA = 850

# Mask selection defaults
NEAR_RADIUS_KM = 2500.0
LOW_QUANTILE = 0.25

# CKA token subsampling: tokens per block per case
CKA_TOKENS_PER_CASE = 256


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Trace precip model representations.")
    p.add_argument("--model-conditional-dir", default=DEFAULT_MODEL_CONDITIONAL_DIR,
                   help="Root output dir from 07_zwd_precipitation_model_comparison full run.")
    p.add_argument("--diagnostic-selection", default=os.path.join(
                       _ROOT, "07_zwd_precipitation_model_comparison", "cases_diagnostic_22.json"),
                   help="'auto' = select from stage_a CSV, or path to a JSON/CSV of case ids.")
    p.add_argument("--n-per-target", type=int, default=1,
                   help="Cases per target when using auto selection.")
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=ALL_MODES)
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS_CLI))
    p.add_argument("--models", nargs="+", default=["precip_zwd", "precip_only"],
                   choices=["precip_zwd", "precip_only"],
                   help="Checkpoint(s) to load for modes that support either model.")
    p.add_argument("--timesteps", nargs="+", default=["t1"],
                   help="Which input timestep(s) to perturb (t0, t1).")
    p.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--zwd-magnitude", type=float, default=ZWD_MAGNITUDE)
    p.add_argument("--precip-dose-mm", type=float, default=PRECIP_DOSE_MM)
    p.add_argument("--q850-magnitude", type=float, default=Q850_MAGNITUDE,
                   help="q850 perturbation amplitude in units of sigma_q850.")
    p.add_argument("--low-quantile", type=float, default=LOW_QUANTILE)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--zwd-smooth-sigma", type=float, default=ZWD_SMOOTH_SIGMA_DEG,
                   help="Gaussian smoothing sigma in degrees for zwd_smooth_removal mode.")
    p.add_argument("--cka-tokens-per-case", type=int, default=CKA_TOKENS_PER_CASE)
    p.add_argument("--region-source-target", default="q850",
                   help="Which target's Stage B saliency to use for region selection. "
                        "Reused for all output targets so zwd saliency maps are not required.")
    p.add_argument("--stage-b-model-name", default=STAGE_B_MODEL_NAME,
                   help="Model name used in Stage B saliency map paths "
                        "(e.g. 'precip_zwd' or 'precip_large_zwd').")
    p.add_argument("--region-selections", default=None,
                   help="Optional selections.csv from an earlier trace. Reuses its exact "
                        "hotspot/low_near centers instead of loading Stage B maps.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        open(path, "w").close()
        return
    fieldnames = _csv_fieldnames(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in rows])


def _append_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.isfile(path)
    fieldnames = _csv_fieldnames(rows)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in rows])


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-30 else float("nan")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Diagnostic case selection
# ---------------------------------------------------------------------------

@dataclass
class DiagCase:
    case_id: str       # "{target}_{init_time_str}"
    target: str        # searchlight target short name
    init_time: datetime
    init_time_str: str
    m_diff_q850: float


def _case_has_required_stage_b_maps(
    model_conditional_dir: str,
    case_id: str,
    targets: tuple[str, ...] = ("q850", "precip"),
    model_name: str = STAGE_B_MODEL_NAME,
) -> bool:
    return all(
        os.path.isfile(_stage_b_saliency_path(model_conditional_dir, case_id, target,
                                               model_name=model_name))
        for target in targets
    )


def _select_diagnostic_cases(
    model_conditional_dir: str,
    n_per_target: int = 1,
    stage_b_model_name: str = STAGE_B_MODEL_NAME,
) -> list[DiagCase]:
    """Select n_per_target cases per target by max |M_diff(q850, 6h)| with Stage B coverage."""
    csv_path = os.path.join(model_conditional_dir, "stage_a_model_trajectories.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Stage A trajectories not found: {csv_path}\n"
            "Run the checkpoint comparison's Stage A and B workflows first, "
            "or pass the fixed --diagnostic-selection manifest."
        )

    by_target: dict[str, list[dict]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("target_var") != "q850":
                continue
            try:
                lead_h = int(row["lead_h"])
            except (KeyError, ValueError):
                continue
            if lead_h != STAGE_B_LEAD_H:
                continue
            tgt = row.get("target", "")
            if tgt not in TARGETS:
                continue
            try:
                m = abs(float(row["M_diff"]))
            except (KeyError, ValueError):
                m = 0.0
            case_id = row.get("case_id", "")
            by_target.setdefault(tgt, []).append({
                "case_id": case_id,
                "M_diff": m,
                "init_time": row.get("init_time", ""),
                "has_stage_b": _case_has_required_stage_b_maps(
                    model_conditional_dir, case_id,
                    model_name=stage_b_model_name),
            })

    selected: list[DiagCase] = []
    skipped_top_cases: list[tuple[str, str, float]] = []
    for tgt, rows in sorted(by_target.items()):
        rows.sort(key=lambda r: r["M_diff"], reverse=True)
        top_row = rows[0] if rows else None
        eligible_rows = [row for row in rows if row["has_stage_b"]]
        if not eligible_rows:
            raise ValueError(
                f"No Stage B-complete diagnostic cases available for target {tgt!r}."
            )
        if top_row is not None and not top_row["has_stage_b"]:
            skipped_top_cases.append((tgt, top_row["case_id"], top_row["M_diff"]))
        for row in eligible_rows[:n_per_target]:
            try:
                init_time = datetime.fromisoformat(row["init_time"])
            except (ValueError, KeyError):
                continue
            selected.append(DiagCase(
                case_id=row["case_id"],
                target=tgt,
                init_time=init_time,
                init_time_str=row["init_time"],
                m_diff_q850=row["M_diff"],
            ))

    if not selected:
        raise ValueError("No diagnostic cases could be selected from stage_a CSV.")

    print(f"  Selected {len(selected)} diagnostic cases ({n_per_target} per target) with Stage B coverage.")
    if skipped_top_cases:
        print("  Replaced Stage-A-only top cases lacking Stage B maps:")
        for tgt, case_id, m_diff in skipped_top_cases:
            print(f"    - {tgt}: skipped {case_id} (|M_diff|={m_diff:.6f})")
    return selected


def _load_diagnostic_cases_from_file(path: str) -> list[DiagCase]:
    """Load an explicit list of diagnostic cases from a JSON or CSV file.

    JSON format: list of objects with keys case_id, target, init_time,
    and optionally m_diff_q850.

    CSV format: rows with the same columns (header required).

    This allows true one-case smoke tests without the 'auto' selection
    expanding to one case per target region (22+ cases).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Diagnostic selection file not found: {path}")

    records: list[dict] = []
    if path.endswith(".json"):
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(records).__name__}")
    elif path.endswith(".csv"):
        with open(path, newline="") as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError(
            f"Unrecognised extension for --diagnostic-selection: {path!r}. "
            "Use .json or .csv."
        )

    result: list[DiagCase] = []
    for i, rec in enumerate(records):
        try:
            case_id    = str(rec["case_id"])
            target     = str(rec["target"])
            init_time_str = str(rec["init_time"])
            init_time  = datetime.fromisoformat(init_time_str)
            m_diff_q850 = float(rec.get("m_diff_q850", 0.0))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Record {i} in {path} is missing or has invalid fields: {exc}"
            ) from exc
        if target not in TARGETS:
            raise ValueError(
                f"Record {i}: target={target!r} is not a known searchlight target. "
                f"Known: {sorted(TARGETS.keys())}"
            )
        result.append(DiagCase(
            case_id=case_id,
            target=target,
            init_time=init_time,
            init_time_str=init_time_str,
            m_diff_q850=m_diff_q850,
        ))

    if not result:
        raise ValueError(f"No cases found in {path}")

    print(f"  Loaded {len(result)} diagnostic case(s) from {path}")
    return result


# ---------------------------------------------------------------------------
# MSWEP precipitation injection
# ---------------------------------------------------------------------------

def _make_mswep_reader(store_path: str = MSWEP_STORE_PATH):
    from common import MSWEPReader  # noqa: E402 (P_CORR dir on path)
    return MSWEPReader(store_path)


def _load_and_inject_precip(case_data: CaseData, init_time: datetime, mswep_reader) -> CaseData:
    """Load MSWEP precipitation and inject into case_data.surf_cpu."""
    precip = load_precip_for_case(init_time, mswep_reader)
    t0_arr = precip.get("t0")
    t1_arr = precip.get("t1")
    if t0_arr is None or t1_arr is None:
        raise RuntimeError(f"MSWEP data missing for {init_time.isoformat()}")

    H_era5 = case_data.lat_vals.shape[0]

    def _pad(arr, target_h):
        if arr.shape[0] == target_h:
            return arr
        if arr.shape[0] + 1 == target_h:
            return np.concatenate([arr, arr[-1:]], axis=0)
        raise ValueError(f"Cannot pad MSWEP {arr.shape[0]} to ERA5 {target_h}")

    t0_arr = _pad(t0_arr, H_era5)
    t1_arr = _pad(t1_arr, H_era5)
    return inject_precip_into_case(case_data, t0_arr, t1_arr)


# ---------------------------------------------------------------------------
# Stage B saliency loading and mask selection
# ---------------------------------------------------------------------------

def _stage_b_saliency_path(
    model_conditional_dir: str,
    case_id: str,
    target: str,
    lead_h: int = STAGE_B_LEAD_H,
    model_name: str = STAGE_B_MODEL_NAME,
) -> str:
    return os.path.join(
        model_conditional_dir,
        model_name,
        STAGE_B_SALIENCY_SUBDIR,
        case_id,
        f"{model_name}_{PRECIP_VAR}_{target}_{lead_h}h.npy",
    )


def _load_stage_b_saliency(
    model_conditional_dir: str,
    case_id: str,
    target: str,
    model_name: str = STAGE_B_MODEL_NAME,
) -> np.ndarray:
    path = _stage_b_saliency_path(model_conditional_dir, case_id, target,
                                   model_name=model_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Stage B saliency map missing: {path}\n"
            f"Run 07_zwd_precipitation_model_comparison with Stage B first."
        )
    return np.load(path).astype(np.float32)


def _pool_saliency_under_mask(
    saliency: np.ndarray,
    mask_np: np.ndarray,
    lat_vals: np.ndarray,
) -> float:
    """Cosine-lat-weighted sum of saliency under Gaussian mask."""
    W = saliency.shape[1]
    weights = cos_lat_weights(lat_vals, W)
    return float((saliency * mask_np * weights).sum())


@dataclass
class RegionSelection:
    region_kind: str     # "hotspot" or "low_near"
    mask_spec: MaskSpec
    pooled_saliency: float
    distance_km: float


def _load_region_selections(
    path: str,
) -> dict[tuple[str, str], list[RegionSelection]]:
    """Load exact hotspot/low-near centers saved by an earlier trace run."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Region-selection CSV not found: {path}")

    by_case_scale: dict[tuple[str, str], list[RegionSelection]] = {}
    with open(path, newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            try:
                case_id = row["case_id"]
                scale_name = row["scale"]
                region_kind = row["region_kind"]
                center_lat = float(row["center_lat"])
                center_lon = float(row["center_lon"])
                pooled_saliency = float(row["pooled_saliency"])
                distance_km = float(row["distance_km"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid region-selection row {row_index + 2} in {path}: {row}"
                ) from exc
            if scale_name not in SCALES:
                raise ValueError(f"Unknown scale {scale_name!r} in {path}")
            if region_kind not in ("hotspot", "low_near"):
                raise ValueError(f"Unknown region kind {region_kind!r} in {path}")

            selection = RegionSelection(
                region_kind=region_kind,
                mask_spec=MaskSpec(
                    scale=scale_name,
                    role="near",
                    center_lat=center_lat,
                    center_lon=center_lon,
                    mask_id=row_index,
                ),
                pooled_saliency=pooled_saliency,
                distance_km=distance_km,
            )
            by_case_scale.setdefault((case_id, scale_name), []).append(selection)

    for key, selections in by_case_scale.items():
        kinds = [selection.region_kind for selection in selections]
        if sorted(kinds) != ["hotspot", "low_near"]:
            raise ValueError(
                f"Expected one hotspot and one low_near for {key} in {path}; got {kinds}"
            )
        selections.sort(key=lambda selection: ("hotspot", "low_near").index(
            selection.region_kind
        ))
    return by_case_scale


def _select_regions_from_saliency(
    saliency: np.ndarray,
    target_short: str,
    scale_name: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    low_quantile: float = LOW_QUANTILE,
) -> list[RegionSelection]:
    """Select hotspot and low_near mask centers from Stage B saliency."""
    target_region = TARGETS[target_short]
    scale = SCALES[scale_name]
    candidate_specs = generate_mask_centers(target_region, scale)
    near_specs = [s for s in candidate_specs if s.role == "near"]
    if not near_specs:
        raise ValueError(f"No near mask centers for {target_short}/{scale_name}")

    # Score each near mask center
    scores = np.array([
        _pool_saliency_under_mask(
            saliency,
            gaussian_mask(s, scale.sigma_deg, lat_vals, lon_vals),
            lat_vals,
        )
        for s in near_specs
    ], dtype=np.float64)

    hotspot_idx = int(np.argmax(scores))
    hotspot_spec = near_specs[hotspot_idx]
    hotspot_score = float(scores[hotspot_idx])
    hotspot_dist = float(great_circle_km(
        target_region.center_lat, target_region.center_lon,
        hotspot_spec.center_lat, hotspot_spec.center_lon,
    ))

    # Distance from target center to each near mask
    target_dists = np.array([
        float(great_circle_km(
            target_region.center_lat, target_region.center_lon,
            near_specs[i].center_lat, near_specs[i].center_lon,
        ))
        for i in range(len(near_specs))
    ], dtype=np.float64)

    # Low near: lowest saliency quartile, distance-matched to hotspot
    other_idx = np.array([i for i in range(len(near_specs)) if i != hotspot_idx])
    if other_idx.size == 0:
        # Only one near spec; use it as both (degenerate but safe)
        return [
            RegionSelection("hotspot", hotspot_spec, hotspot_score, hotspot_dist),
            RegionSelection("low_near", hotspot_spec, hotspot_score, hotspot_dist),
        ]

    other_scores = scores[other_idx]
    low_cut = float(np.quantile(other_scores, low_quantile))
    low_pool_idx = other_idx[other_scores <= low_cut]
    if low_pool_idx.size == 0:
        low_pool_idx = other_idx

    # Among low-saliency candidates, pick the one with distance-to-target closest to hotspot's
    local_sel = int(np.argmin(
        np.abs(target_dists[low_pool_idx] - hotspot_dist) + 1e-6 * scores[low_pool_idx]
    ))
    low_idx = int(low_pool_idx[local_sel])
    low_spec = near_specs[low_idx]
    low_score = float(scores[low_idx])
    low_dist = float(great_circle_km(
        target_region.center_lat, target_region.center_lon,
        low_spec.center_lat, low_spec.center_lon,
    ))

    return [
        RegionSelection("hotspot", hotspot_spec, hotspot_score, hotspot_dist),
        RegionSelection("low_near", low_spec, low_score, low_dist),
    ]


# ---------------------------------------------------------------------------
# Block spec construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockTraceSpec:
    key: str
    family: str
    stage_name: str
    stage_index: int
    block_index: int
    traversal_index: int
    resolution: tuple[int, int, int]
    is_stage_terminal: bool


def _build_block_specs(model, patch_res: tuple[int, int, int]) -> list[tuple[Any, BlockTraceSpec]]:
    backbone = model.backbone
    enc_res, _ = backbone.get_encoder_specs(patch_res)
    out: list[tuple[Any, BlockTraceSpec]] = []
    traversal = 0
    for stage_idx, layer in enumerate(backbone.encoder_layers):
        for block_idx, block in enumerate(layer.blocks):
            spec = BlockTraceSpec(
                key=f"enc_s{stage_idx}_b{block_idx:02d}",
                family="encoder",
                stage_name=f"enc_s{stage_idx}",
                stage_index=stage_idx,
                block_index=block_idx,
                traversal_index=traversal,
                resolution=tuple(enc_res[stage_idx]),
                is_stage_terminal=(block_idx == len(layer.blocks) - 1),
            )
            out.append((block, spec))
            traversal += 1
    for layer_idx, layer in enumerate(backbone.decoder_layers):
        res_idx = backbone.num_decoder_layers - layer_idx - 1
        for block_idx, block in enumerate(layer.blocks):
            spec = BlockTraceSpec(
                key=f"dec_s{layer_idx}_b{block_idx:02d}",
                family="decoder",
                stage_name=f"dec_s{layer_idx}",
                stage_index=layer_idx,
                block_index=block_idx,
                traversal_index=traversal,
                resolution=tuple(enc_res[res_idx]),
                is_stage_terminal=(block_idx == len(layer.blocks) - 1),
            )
            out.append((block, spec))
            traversal += 1
    return out


def _summarize_stage_map(delta_cpu: torch.Tensor, resolution: tuple[int, int, int]) -> np.ndarray:
    arr = delta_cpu[0].numpy()
    C, H, W = resolution
    if arr.shape[0] != C * H * W:
        raise ValueError(f"Token count mismatch: got {arr.shape[0]}, expected {C*H*W}")
    arr = arr.reshape(C, H, W, arr.shape[-1])
    return np.sqrt(np.mean(arr * arr, axis=-1)).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Hook-based activation capture
# ---------------------------------------------------------------------------

def _capture_actual_activations(
    model, batch, target_fn, block_specs
) -> tuple[dict[str, torch.Tensor], float]:
    actual: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(spec):
        def _hook(_m, _i, output):
            actual[spec.key] = output.detach().cpu().to(torch.float16).contiguous()
        return _hook

    for module, spec in block_specs:
        handles.append(module.register_forward_hook(_make_hook(spec)))
    try:
        with torch.no_grad():
            pred = _forward(model, batch)
            score = float(target_fn(pred).item())
    finally:
        for h in handles:
            h.remove()
    _gpu_sync()
    return actual, score


def _capture_delta_activations(
    model, batch, target_fn, actual_activations, block_specs
) -> tuple[dict[str, torch.Tensor], float]:
    deltas: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(spec):
        def _hook(_m, _i, output):
            cur = output.detach().float().cpu()
            ref = actual_activations[spec.key].float()
            deltas[spec.key] = (cur - ref).to(torch.float16).contiguous()
        return _hook

    for module, spec in block_specs:
        handles.append(module.register_forward_hook(_make_hook(spec)))
    try:
        with torch.no_grad():
            pred = _forward(model, batch)
            score = float(target_fn(pred).item())
    finally:
        for h in handles:
            h.remove()
    _gpu_sync()
    return deltas, score


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.reshape(-1), b.reshape(-1)
    na = float(torch.linalg.vector_norm(a_f).item())
    nb = float(torch.linalg.vector_norm(b_f).item())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a_f, b_f).item() / (na * nb))


def _trace_paired_minus_pass(
    model, batch, target_fn,
    actual_activations: dict[str, torch.Tensor],
    plus_deltas: dict[str, torch.Tensor],
    block_specs,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    block_rows: list[dict] = []
    contrast_stage_maps: dict[str, np.ndarray] = {}
    common_stage_maps: dict[str, np.ndarray] = {}
    handles = []

    def _make_hook(spec):
        def _hook(_m, _i, output):
            ref = actual_activations[spec.key].float()
            minus_delta = output.detach().float().cpu() - ref
            plus_delta = plus_deltas[spec.key].float()
            contrast = 0.5 * (plus_delta - minus_delta)
            common   = 0.5 * (plus_delta + minus_delta)
            pd_rms  = float(torch.sqrt((plus_delta  ** 2).mean()).item())
            md_rms  = float(torch.sqrt((minus_delta ** 2).mean()).item())
            c_rms   = float(torch.sqrt((contrast ** 2).mean()).item())
            k_rms   = float(torch.sqrt((common   ** 2).mean()).item())
            b_rms   = float(torch.sqrt((ref      ** 2).mean()).item())
            block_rows.append({
                "block_key": spec.key,
                "family": spec.family,
                "stage_name": spec.stage_name,
                "stage_index": spec.stage_index,
                "block_index": spec.block_index,
                "traversal_index": spec.traversal_index,
                "plus_delta_rms": pd_rms,
                "minus_delta_rms": md_rms,
                "contrast_rms": c_rms,
                "common_rms": k_rms,
                "baseline_rms": b_rms,
                "plus_relative_rms": _safe_div(pd_rms, b_rms),
                "minus_relative_rms": _safe_div(md_rms, b_rms),
                "contrast_relative_rms": _safe_div(c_rms, b_rms),
                "common_relative_rms": _safe_div(k_rms, b_rms),
                "signed_cosine": _cosine_similarity(plus_delta, -minus_delta),
                "same_sign_cosine": _cosine_similarity(plus_delta, minus_delta),
                "contrast_share": _safe_div(c_rms, c_rms + k_rms),
                "contrast_common_ratio": _safe_div(c_rms, k_rms),
            })
            if spec.is_stage_terminal:
                contrast_stage_maps[spec.stage_name] = _summarize_stage_map(contrast, spec.resolution)
                common_stage_maps[spec.stage_name]   = _summarize_stage_map(common,   spec.resolution)
        return _hook

    for module, spec in block_specs:
        handles.append(module.register_forward_hook(_make_hook(spec)))
    try:
        with torch.no_grad():
            pred = _forward(model, batch)
            score = float(target_fn(pred).item())
    finally:
        for h in handles:
            h.remove()
    _gpu_sync()
    block_rows.sort(key=lambda r: r["traversal_index"])
    return block_rows, contrast_stage_maps, common_stage_maps, score


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_stage_curve(block_rows: list[dict], metadata: dict) -> list[dict]:
    by_stage: dict[str, list[dict]] = {}
    for row in block_rows:
        by_stage.setdefault(row["stage_name"], []).append(row)
    out = []
    for stage_name, rows in sorted(by_stage.items(), key=lambda kv: kv[1][0]["traversal_index"]):
        out.append({
            **metadata,
            "stage_name": stage_name,
            "family": rows[0]["family"],
            "stage_index": rows[0]["stage_index"],
            "n_blocks": len(rows),
            "plus_delta_rms_mean": float(np.mean([r["plus_delta_rms"] for r in rows])),
            "minus_delta_rms_mean": float(np.mean([r["minus_delta_rms"] for r in rows])),
            "contrast_rms_mean": float(np.mean([r["contrast_rms"] for r in rows])),
            "common_rms_mean": float(np.mean([r["common_rms"] for r in rows])),
            "signed_cosine_mean": float(np.mean([r["signed_cosine"] for r in rows])),
            "contrast_share_mean": float(np.mean([r["contrast_share"] for r in rows])),
        })
    return out


def _make_pair_metadata(base: dict, actual_score: float, plus_score: float, minus_score: float) -> dict:
    pd = plus_score  - actual_score
    md = minus_score - actual_score
    return {
        **base,
        "actual_target_score": actual_score,
        "plus_target_score":   plus_score,
        "minus_target_score":  minus_score,
        "plus_target_delta":   pd,
        "minus_target_delta":  md,
        "signed_target_response": 0.5 * (pd - md),
        "common_target_response": 0.5 * (pd + md),
        "mean_abs_target_delta": 0.5 * (abs(pd) + abs(md)),
        "output_opposite_sign": bool(np.sign(pd) == -np.sign(md)),
    }


def _save_pair_outputs(
    out_dir: str,
    metadata: dict,
    block_rows: list[dict],
    stage_rows: list[dict],
    contrast_maps: dict[str, np.ndarray],
    common_maps: dict[str, np.ndarray],
    skip_plots: bool,
) -> None:
    _ensure_dir(out_dir)
    _write_json(os.path.join(out_dir, "pair_metadata.json"), metadata)
    _write_csv(os.path.join(out_dir, "block_pair_metrics.csv"), block_rows)
    _write_csv(os.path.join(out_dir, "stage_pair_metrics.csv"), stage_rows)
    np.savez_compressed(os.path.join(out_dir, "contrast_stage_maps.npz"), **contrast_maps)
    np.savez_compressed(os.path.join(out_dir, "common_stage_maps.npz"), **common_maps)
    if not skip_plots:
        _plot_stage_curve(os.path.join(out_dir, "stage_pair_curve.png"), stage_rows)
        _plot_block_curve(os.path.join(out_dir, "block_pair_curve.png"), block_rows)
        _plot_stage_maps(os.path.join(out_dir, "contrast_stage_maps.png"), contrast_maps)


# ---------------------------------------------------------------------------
# Target function builder
# ---------------------------------------------------------------------------

def _make_target_fn(target_spec: TargetSpec, case_data: CaseData, target_short: str):
    target_region = TARGETS[target_short]
    lat_vals = case_data.lat_vals
    lon_vals = case_data.lon_vals

    def _fn(pred):
        val = extract_scalar(pred, target_spec, lat_vals, lon_vals, target_region)
        return torch.tensor(float(val), dtype=torch.float32)

    return _fn


# ---------------------------------------------------------------------------
# Perturbation builders
# ---------------------------------------------------------------------------

def _perturb_precip_gaussian(
    precip_1_2_H_W: torch.Tensor,
    gaussian_mask_np: np.ndarray,
    dose_mm: float,
    sign: float,
    timestep_idx: int,
) -> torch.Tensor:
    """Add sign*dose_mm*gaussian_mask to precip at timestep_idx, clamp >= 0."""
    out = precip_1_2_H_W.clone()
    delta = torch.from_numpy((sign * dose_mm * gaussian_mask_np).astype(np.float32))
    out[0, timestep_idx] = (out[0, timestep_idx] + delta).clamp(min=0.0)
    return out


def _perturb_q_level_gaussian(
    q_1_2_L_H_W: torch.Tensor,
    gaussian_mask_np: np.ndarray,
    level_idx: int,
    sign: float,
    magnitude: float,
    q_loc: float,
    q_scale: float,
    timestep_idx: int,
) -> tuple[torch.Tensor, dict]:
    """Return (perturbed q, clip diagnostics) for a +/- Gaussian bump at one level.

    Mirrors perturb_zwd (delta = sign * magnitude * sigma * mask, then a
    +/-4 sigma envelope), with two differences forced by q being a 3D field:
    the clamp is applied only to the perturbed (timestep, level) slice, since
    q varies by orders of magnitude across levels and a global clamp would
    destroy the unperturbed ones; and the lower bound is floored at 0 because
    specific humidity cannot be negative.

    That floor breaks the symmetry of the +/- pair: sigma_q850 (~4.1e-3 kg/kg)
    exceeds the background q850 of dry/cold regions, so a -1 sigma bump would
    push q below zero and gets clamped, making the minus arm weaker than the
    plus arm.  The paired contrast 0.5*(d+ - d-) assumes symmetric arms, so the
    realized-vs-intended perturbation mass is returned alongside the field and
    recorded in the run metadata: clip_frac == 0 means the pair is clean, and a
    large clip_frac means that case's contrast term is attenuated on one side
    and should not be read as a signed routing direction.
    """
    out = q_1_2_L_H_W.clone()
    delta = torch.from_numpy(
        (sign * magnitude * q_scale * gaussian_mask_np).astype(np.float32)
    ).to(out.dtype)

    lo = max(0.0, q_loc - 4.0 * q_scale)
    hi = q_loc + 4.0 * q_scale
    unclamped = out[0, timestep_idx, level_idx] + delta
    clamped = unclamped.clamp(min=lo, max=hi)
    out[0, timestep_idx, level_idx] = clamped

    in_mask = torch.from_numpy(gaussian_mask_np).to(out.dtype) > 1e-3
    n_mask = int(in_mask.sum())
    n_clipped = int((in_mask & (unclamped != clamped)).sum())
    intended_l1 = float(delta.abs().sum())
    realized_l1 = float((clamped - q_1_2_L_H_W[0, timestep_idx, level_idx]).abs().sum())

    stats = {
        "clip_frac": (n_clipped / n_mask) if n_mask else 0.0,
        "intended_l1": intended_l1,
        "realized_l1": realized_l1,
        "realized_frac": _safe_div(realized_l1, intended_l1),
    }
    return out, stats


# ---------------------------------------------------------------------------
# Mode: zwd_trace
# ---------------------------------------------------------------------------

def run_zwd_trace(
    *,
    model,
    case_data: CaseData,
    diag_case: DiagCase,
    region: RegionSelection,
    scale_name: str,
    timestep_name: str,
    target_spec: TargetSpec,
    model_spec: ModelSpec,
    device,
    magnitude: float,
    output_dir: str,
    skip_plots: bool,
) -> dict:
    """Paired ZWD +/- trace on precip_zwd model."""
    timestep_idx = {"t0": 0, "t1": 1}[timestep_name]
    mask_np = gaussian_mask(
        region.mask_spec, SCALES[scale_name].sigma_deg,
        case_data.lat_vals, case_data.lon_vals,
    )
    precip_override = case_data.surf_cpu.get(PRECIP_VAR)

    actual_batch = build_precip_batch(case_data, model_spec, device,
                                      precip_override=precip_override)
    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)

    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    plus_zwd = perturb_zwd(
        zwd_actual_1_2_H_W=case_data.surf_cpu["zwd"],
        mask_H_W=mask_np, sign=+1.0, magnitude=magnitude,
        zwd_loc=case_data.zwd_loc, zwd_scale=case_data.zwd_scale,
        timestep_idx=timestep_idx,
    )
    plus_batch = build_precip_batch(case_data, model_spec, device,
                                    precip_override=precip_override,
                                    zwd_override=plus_zwd)
    plus_deltas, plus_score = _capture_delta_activations(
        model, plus_batch, target_fn, actual_acts, block_specs
    )

    minus_zwd = perturb_zwd(
        zwd_actual_1_2_H_W=case_data.surf_cpu["zwd"],
        mask_H_W=mask_np, sign=-1.0, magnitude=magnitude,
        zwd_loc=case_data.zwd_loc, zwd_scale=case_data.zwd_scale,
        timestep_idx=timestep_idx,
    )
    minus_batch = build_precip_batch(case_data, model_spec, device,
                                     precip_override=precip_override,
                                     zwd_override=minus_zwd)
    block_rows, contrast_maps, common_maps, minus_score = _trace_paired_minus_pass(
        model, minus_batch, target_fn, actual_acts, plus_deltas, block_specs
    )
    del plus_deltas
    _gpu_sync()

    base_meta = {
        "case_id": diag_case.case_id, "target": diag_case.target,
        "scale": scale_name, "region_kind": region.region_kind,
        "perturb_timestep": timestep_name, "mode": "zwd_trace",
        "model": model_spec.name,
        "target_var": target_spec.name,
        "mask_center_lat": region.mask_spec.center_lat,
        "mask_center_lon": region.mask_spec.center_lon,
        "pooled_saliency": region.pooled_saliency,
        "magnitude": magnitude,
    }
    pair_meta = _make_pair_metadata(base_meta, actual_score, plus_score, minus_score)
    stage_rows = _aggregate_stage_curve(block_rows, pair_meta)

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           scale_name, region.region_kind, timestep_name, "zwd_trace",
                           target_spec.name)
    _save_pair_outputs(
        out_dir=out_dir, metadata=pair_meta,
        block_rows=[{**pair_meta, **r} for r in block_rows],
        stage_rows=stage_rows,
        contrast_maps=contrast_maps, common_maps=common_maps,
        skip_plots=skip_plots,
    )
    del actual_acts
    _gpu_sync()
    return {"pair_meta": pair_meta, "stage_rows": stage_rows}


# ---------------------------------------------------------------------------
# Mode: precip_trace
# ---------------------------------------------------------------------------

def run_precip_trace(
    *,
    model,
    model_spec: ModelSpec,
    case_data: CaseData,
    diag_case: DiagCase,
    region: RegionSelection,
    scale_name: str,
    timestep_name: str,
    target_spec: TargetSpec,
    dose_mm: float,
    device,
    output_dir: str,
    skip_plots: bool,
) -> dict:
    """Paired precip +/- trace on a single model."""
    timestep_idx = {"t0": 0, "t1": 1}[timestep_name]
    mask_np = gaussian_mask(
        region.mask_spec, SCALES[scale_name].sigma_deg,
        case_data.lat_vals, case_data.lon_vals,
    )
    precip_actual = case_data.surf_cpu.get(PRECIP_VAR)
    if precip_actual is None:
        raise RuntimeError("Precipitation not in case_data; load MSWEP first.")

    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)
    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    actual_batch = build_precip_batch(case_data, model_spec, device,
                                      precip_override=precip_actual)
    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    plus_precip = _perturb_precip_gaussian(precip_actual, mask_np, dose_mm, +1.0, timestep_idx)
    plus_batch = build_precip_batch(case_data, model_spec, device,
                                    precip_override=plus_precip)
    plus_deltas, plus_score = _capture_delta_activations(
        model, plus_batch, target_fn, actual_acts, block_specs
    )

    minus_precip = _perturb_precip_gaussian(precip_actual, mask_np, dose_mm, -1.0, timestep_idx)
    minus_batch = build_precip_batch(case_data, model_spec, device,
                                     precip_override=minus_precip)
    block_rows, contrast_maps, common_maps, minus_score = _trace_paired_minus_pass(
        model, minus_batch, target_fn, actual_acts, plus_deltas, block_specs
    )
    del plus_deltas
    _gpu_sync()

    base_meta = {
        "case_id": diag_case.case_id, "target": diag_case.target,
        "scale": scale_name, "region_kind": region.region_kind,
        "perturb_timestep": timestep_name, "mode": "precip_trace",
        "model": model_spec.name,
        "target_var": target_spec.name,
        "mask_center_lat": region.mask_spec.center_lat,
        "mask_center_lon": region.mask_spec.center_lon,
        "pooled_saliency": region.pooled_saliency,
        "dose_mm": dose_mm,
    }
    pair_meta = _make_pair_metadata(base_meta, actual_score, plus_score, minus_score)
    stage_rows = _aggregate_stage_curve(block_rows, pair_meta)

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           scale_name, region.region_kind, timestep_name, "precip_trace",
                           model_spec.name, target_spec.name)
    _save_pair_outputs(
        out_dir=out_dir, metadata=pair_meta,
        block_rows=[{**pair_meta, **r} for r in block_rows],
        stage_rows=stage_rows,
        contrast_maps=contrast_maps, common_maps=common_maps,
        skip_plots=skip_plots,
    )
    del actual_acts
    _gpu_sync()
    return {"pair_meta": pair_meta, "stage_rows": stage_rows}


# ---------------------------------------------------------------------------
# Mode: q850_trace
# ---------------------------------------------------------------------------

def run_q850_trace(
    *,
    model,
    model_spec: ModelSpec,
    case_data: CaseData,
    diag_case: DiagCase,
    region: RegionSelection,
    scale_name: str,
    timestep_name: str,
    target_spec: TargetSpec,
    device,
    magnitude: float,
    output_dir: str,
    skip_plots: bool,
) -> dict:
    """Paired q850 +/- trace on a single model.

    This is the reverse direction of the existing traces: q850 is the
    perturbation *source* here (it is an input channel of both model
    variants), and precip / zwd are the response targets.
    """
    timestep_idx = {"t0": 0, "t1": 1}[timestep_name]
    mask_np = gaussian_mask(
        region.mask_spec, SCALES[scale_name].sigma_deg,
        case_data.lat_vals, case_data.lon_vals,
    )

    try:
        level_idx = list(case_data.pressure_levels).index(Q850_LEVEL_HPA)
    except ValueError as exc:
        raise RuntimeError(
            f"{Q850_LEVEL_HPA} hPa not in pressure levels {case_data.pressure_levels}"
        ) from exc

    q_key = f"q_{Q850_LEVEL_HPA}"
    q_loc = float(case_data.locations[q_key])
    q_scale = float(case_data.scales[q_key])
    q_actual = case_data.atmos_cpu["q"]

    precip_override = case_data.surf_cpu.get(PRECIP_VAR)
    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)

    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    actual_batch = build_precip_batch(case_data, model_spec, device,
                                      precip_override=precip_override)
    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    plus_q, plus_clip = _perturb_q_level_gaussian(
        q_actual, mask_np, level_idx, +1.0, magnitude, q_loc, q_scale, timestep_idx,
    )
    plus_batch = build_precip_batch(case_data, model_spec, device,
                                    precip_override=precip_override,
                                    atmos_override={"q": plus_q})
    plus_deltas, plus_score = _capture_delta_activations(
        model, plus_batch, target_fn, actual_acts, block_specs
    )

    minus_q, minus_clip = _perturb_q_level_gaussian(
        q_actual, mask_np, level_idx, -1.0, magnitude, q_loc, q_scale, timestep_idx,
    )
    minus_batch = build_precip_batch(case_data, model_spec, device,
                                     precip_override=precip_override,
                                     atmos_override={"q": minus_q})
    block_rows, contrast_maps, common_maps, minus_score = _trace_paired_minus_pass(
        model, minus_batch, target_fn, actual_acts, plus_deltas, block_specs
    )
    del plus_deltas
    _gpu_sync()

    base_meta = {
        "case_id": diag_case.case_id, "target": diag_case.target,
        "scale": scale_name, "region_kind": region.region_kind,
        "perturb_timestep": timestep_name, "mode": "q850_trace",
        "model": model_spec.name,
        "target_var": target_spec.name,
        "mask_center_lat": region.mask_spec.center_lat,
        "mask_center_lon": region.mask_spec.center_lon,
        "pooled_saliency": region.pooled_saliency,
        "magnitude": magnitude,
        "perturb_level_hpa": Q850_LEVEL_HPA,
        # Non-negativity clipping breaks +/- symmetry in dry regions; a run with
        # plus_clip_frac/minus_clip_frac > 0 has an attenuated arm on that side.
        "plus_clip_frac": plus_clip["clip_frac"],
        "minus_clip_frac": minus_clip["clip_frac"],
        "plus_realized_frac": plus_clip["realized_frac"],
        "minus_realized_frac": minus_clip["realized_frac"],
        "pair_asymmetry": abs(plus_clip["realized_l1"] - minus_clip["realized_l1"]) /
                          max(plus_clip["realized_l1"], minus_clip["realized_l1"], 1e-12),
    }
    pair_meta = _make_pair_metadata(base_meta, actual_score, plus_score, minus_score)
    stage_rows = _aggregate_stage_curve(block_rows, pair_meta)

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           scale_name, region.region_kind, timestep_name, "q850_trace",
                           model_spec.name, target_spec.name)
    _save_pair_outputs(
        out_dir=out_dir, metadata=pair_meta,
        block_rows=[{**pair_meta, **r} for r in block_rows],
        stage_rows=stage_rows,
        contrast_maps=contrast_maps, common_maps=common_maps,
        skip_plots=skip_plots,
    )
    del actual_acts
    _gpu_sync()
    return {"pair_meta": pair_meta, "stage_rows": stage_rows}


# ---------------------------------------------------------------------------
# Mode: precip_remove_trace
# ---------------------------------------------------------------------------

def run_precip_remove_trace(
    *,
    model,
    model_spec: ModelSpec,
    case_data: CaseData,
    diag_case: DiagCase,
    target_spec: TargetSpec,
    device,
    output_dir: str,
) -> dict:
    """Remove t1 precipitation from target-region disk; compare to actual."""
    target_region = TARGETS[diag_case.target]
    removal_mask = make_removal_mask(
        case_data.lat_vals, case_data.lon_vals,
        center_lat=target_region.center_lat,
        center_lon=target_region.center_lon,
    )
    precip_actual = case_data.surf_cpu.get(PRECIP_VAR)
    if precip_actual is None:
        raise RuntimeError("Precipitation not in case_data.")

    precip_removed = apply_precip_removal(precip_actual, removal_mask, timestep=1)
    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)

    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    actual_batch = build_precip_batch(case_data, model_spec, device,
                                      precip_override=precip_actual)
    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    removed_batch = build_precip_batch(case_data, model_spec, device,
                                       precip_override=precip_removed)
    block_rows: list[dict] = []
    handles = []

    def _make_hook(spec):
        def _hook(_m, _i, output):
            ref = actual_acts[spec.key].float()
            delta = output.detach().float().cpu() - ref
            d_rms = float(torch.sqrt((delta ** 2).mean()).item())
            b_rms = float(torch.sqrt((ref ** 2).mean()).item())
            block_rows.append({
                "block_key": spec.key, "family": spec.family,
                "stage_name": spec.stage_name, "stage_index": spec.stage_index,
                "block_index": spec.block_index, "traversal_index": spec.traversal_index,
                "delta_rms": d_rms, "baseline_rms": b_rms,
                "relative_rms": _safe_div(d_rms, b_rms),
            })
        return _hook

    for module, spec in block_specs:
        handles.append(module.register_forward_hook(_make_hook(spec)))
    try:
        with torch.no_grad():
            pred_removed = _forward(model, removed_batch)
            removed_score = float(target_fn(pred_removed).item())
    finally:
        for h in handles:
            h.remove()
    _gpu_sync()

    block_rows.sort(key=lambda r: r["traversal_index"])
    base = {
        "case_id": diag_case.case_id, "target": diag_case.target,
        "mode": "precip_remove_trace", "model": model_spec.name,
        "target_var": target_spec.name,
        "actual_score": actual_score, "removed_score": removed_score,
        "delta_score": removed_score - actual_score,
    }
    rows_with_meta = [{**base, **r} for r in block_rows]

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           "precip_remove_trace", model_spec.name, target_spec.name)
    _ensure_dir(out_dir)
    _write_json(os.path.join(out_dir, "summary.json"), base)
    _write_csv(os.path.join(out_dir, "block_metrics.csv"), rows_with_meta)

    del actual_acts
    _gpu_sync()
    return {"summary": base, "block_rows": rows_with_meta}


# ---------------------------------------------------------------------------
# Mode: zwd_smooth_removal / zwd_ivw_removal  (ZWD baseline removal trace)
# ---------------------------------------------------------------------------

def _compute_baseline_zwd(
    baseline_type: str,
    case_data: "CaseData",
    sigma_deg: float,
) -> torch.Tensor:
    """Return a CPU float32 tensor (1, 2, H, W) for the ZWD baseline."""
    zwd = case_data.surf_cpu.get("zwd")
    if zwd is None:
        raise RuntimeError("ZWD not present in case_data for baseline computation.")
    if baseline_type == "smooth":
        return smoothed_zwd_baseline(zwd, sigma_deg)
    elif baseline_type == "ivw":
        return make_qhat_zwd(case_data)
    else:
        raise ValueError(f"Unknown baseline_type={baseline_type!r}")


def run_zwd_removal_trace(
    *,
    model,
    case_data: "CaseData",
    diag_case: "DiagCase",
    region: "RegionSelection",
    scale_name: str,
    timestep_name: str,
    target_spec: "TargetSpec",
    model_spec: "ModelSpec",
    device,
    baseline_zwd_cpu: torch.Tensor,
    baseline_label: str,
    output_dir: str,
    skip_plots: bool,
) -> dict:
    """Measure how replacing actual ZWD with a baseline alters block representations.

    Pattern mirrors precip_remove_trace: runs actual forward (captures activations),
    then runs baseline forward and records per-block (actual - baseline) RMS delta.
    """
    precip_override = case_data.surf_cpu.get(PRECIP_VAR)
    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)

    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    actual_batch = build_precip_batch(case_data, model_spec, device,
                                      precip_override=precip_override)
    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    baseline_batch = build_precip_batch(case_data, model_spec, device,
                                        precip_override=precip_override,
                                        zwd_override=baseline_zwd_cpu)
    block_rows: list[dict] = []
    handles = []

    def _make_hook(spec):
        def _hook(_m, _i, output):
            ref = actual_acts[spec.key].float()
            delta = output.detach().float().cpu() - ref
            d_rms = float(torch.sqrt((delta ** 2).mean()).item())
            b_rms = float(torch.sqrt((ref ** 2).mean()).item())
            block_rows.append({
                "block_key": spec.key, "family": spec.family,
                "stage_name": spec.stage_name, "stage_index": spec.stage_index,
                "block_index": spec.block_index, "traversal_index": spec.traversal_index,
                "delta_rms": d_rms, "baseline_rms": b_rms,
                "relative_rms": _safe_div(d_rms, b_rms),
            })
        return _hook

    for module, spec in block_specs:
        handles.append(module.register_forward_hook(_make_hook(spec)))
    try:
        with torch.no_grad():
            pred_baseline = _forward(model, baseline_batch)
            baseline_score = float(target_fn(pred_baseline).item())
    finally:
        for h in handles:
            h.remove()
    _gpu_sync()

    block_rows.sort(key=lambda r: r["traversal_index"])
    mode_name = f"zwd_{baseline_label}_removal"
    base = {
        "case_id": diag_case.case_id, "target": diag_case.target,
        "scale": scale_name, "region_kind": region.region_kind,
        "perturb_timestep": timestep_name,
        "mode": mode_name, "model": model_spec.name,
        "target_var": target_spec.name,
        "mask_center_lat": region.mask_spec.center_lat,
        "mask_center_lon": region.mask_spec.center_lon,
        "pooled_saliency": region.pooled_saliency,
        "baseline_label": baseline_label,
        "actual_score": actual_score, "baseline_score": baseline_score,
        "delta_score": baseline_score - actual_score,
    }
    rows_with_meta = [{**base, **r} for r in block_rows]

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           scale_name, region.region_kind, timestep_name, mode_name,
                           target_spec.name)
    _ensure_dir(out_dir)
    _write_json(os.path.join(out_dir, "summary.json"), base)
    _write_csv(os.path.join(out_dir, "block_metrics.csv"), rows_with_meta)

    del actual_acts
    _gpu_sync()
    return {"summary": base, "block_rows": rows_with_meta}


# ---------------------------------------------------------------------------
# Mode: factorial
# ---------------------------------------------------------------------------

def run_factorial(
    *,
    model,
    model_spec: ModelSpec,
    case_data: CaseData,
    diag_case: DiagCase,
    region: RegionSelection,
    scale_name: str,
    timestep_name: str,
    target_spec: TargetSpec,
    dose_mm: float,
    zwd_magnitude: float,
    device,
    output_dir: str,
) -> list[dict]:
    """4-way factorial ZWD×precip interaction on precip_zwd model."""
    timestep_idx = {"t0": 0, "t1": 1}[timestep_name]
    mask_np = gaussian_mask(
        region.mask_spec, SCALES[scale_name].sigma_deg,
        case_data.lat_vals, case_data.lon_vals,
    )
    precip_actual = case_data.surf_cpu.get(PRECIP_VAR)
    if precip_actual is None:
        raise RuntimeError("Precipitation not in case_data.")

    target_fn = _make_target_fn(target_spec, case_data, diag_case.target)
    H, W = len(case_data.lat_vals), len(case_data.lon_vals)
    patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
    block_specs = _build_block_specs(model, patch_res)

    # Build 4 batches: actual, Z+, P+, Z+P+ (and analogous minus sign)
    def _build(zwd_sign, precip_sign):
        p = precip_actual
        if precip_sign != 0:
            p = _perturb_precip_gaussian(precip_actual, mask_np, dose_mm, precip_sign, timestep_idx)
        z = case_data.surf_cpu.get("zwd")
        if zwd_sign != 0 and z is not None:
            z = perturb_zwd(
                zwd_actual_1_2_H_W=case_data.surf_cpu["zwd"],
                mask_H_W=mask_np, sign=float(zwd_sign), magnitude=zwd_magnitude,
                zwd_loc=case_data.zwd_loc, zwd_scale=case_data.zwd_scale,
                timestep_idx=timestep_idx,
            )
        return build_precip_batch(case_data, model_spec, device,
                                  precip_override=p,
                                  zwd_override=z if z is not None else None)

    def _collect_deltas(batch, ref_acts):
        deltas: dict[str, torch.Tensor] = {}
        score_val = [0.0]
        handles = []

        def _make_hook(spec):
            def _hook(_m, _i, output):
                cur = output.detach().float().cpu()
                deltas[spec.key] = (cur - ref_acts[spec.key].float()).to(torch.float16)
            return _hook

        for module, spec in block_specs:
            handles.append(module.register_forward_hook(_make_hook(spec)))
        try:
            with torch.no_grad():
                pred = _forward(model, batch)
                score_val[0] = float(target_fn(pred).item())
        finally:
            for h in handles:
                h.remove()
        _gpu_sync()
        return deltas, score_val[0]

    actual_batch = _build(0, 0)
    actual_acts, actual_score = _capture_actual_activations(
        model, actual_batch, target_fn, block_specs
    )

    rows: list[dict] = []
    for sign in (+1, -1):
        z_batch   = _build(sign, 0)
        p_batch   = _build(0, sign)
        zp_batch  = _build(sign, sign)

        delta_Z,  score_Z  = _collect_deltas(z_batch,  actual_acts)
        delta_P,  score_P  = _collect_deltas(p_batch,  actual_acts)
        delta_ZP, score_ZP = _collect_deltas(zp_batch, actual_acts)

        for _, spec in block_specs:
            k = spec.key
            d_Z  = delta_Z[k].float()
            d_P  = delta_P[k].float()
            d_ZP = delta_ZP[k].float()
            interaction = d_ZP - d_Z - d_P  # should be ~0 for additive systems

            rows.append({
                "case_id": diag_case.case_id, "target": diag_case.target,
                "scale": scale_name, "region_kind": region.region_kind,
                "perturb_timestep": timestep_name, "sign": sign,
                "model": model_spec.name, "target_var": target_spec.name,
                "block_key": k, "family": spec.family,
                "stage_name": spec.stage_name, "stage_index": spec.stage_index,
                "traversal_index": spec.traversal_index,
                "score_actual": actual_score,
                "score_Z": score_Z, "score_P": score_P, "score_ZP": score_ZP,
                "delta_Z_rms":  float(torch.sqrt((d_Z  ** 2).mean()).item()),
                "delta_P_rms":  float(torch.sqrt((d_P  ** 2).mean()).item()),
                "delta_ZP_rms": float(torch.sqrt((d_ZP ** 2).mean()).item()),
                "interaction_rms": float(torch.sqrt((interaction ** 2).mean()).item()),
                "interaction_share": _safe_div(
                    float(torch.sqrt((interaction ** 2).mean()).item()),
                    float(torch.sqrt((d_ZP ** 2).mean()).item()) + 1e-30,
                ),
            })

        del delta_Z, delta_P, delta_ZP
        _gpu_sync()

    out_dir = os.path.join(output_dir, "per_case", diag_case.case_id,
                           scale_name, region.region_kind, timestep_name,
                           "factorial", target_spec.name)
    _ensure_dir(out_dir)
    _write_csv(os.path.join(out_dir, "block_factorial.csv"), rows)

    del actual_acts
    _gpu_sync()
    return rows


# ---------------------------------------------------------------------------
# Mode: cka
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Feature-space linear CKA (row-centered). X, Y: (N, D)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    hsic_xy = float(np.linalg.norm(XtY) ** 2)
    hsic_xx = float(np.linalg.norm(XtX))
    hsic_yy = float(np.linalg.norm(YtY))
    denom = hsic_xx * hsic_yy
    if denom < 1e-30:
        return float("nan")
    return float(hsic_xy / denom)


def _subsample_tokens(act: torch.Tensor, n_tokens: int) -> np.ndarray:
    """Subsample n_tokens from a (1, T, D) or (T, D) activation tensor."""
    t = act.squeeze(0).float().numpy() if act.dim() == 3 else act.float().numpy()
    total = t.shape[0]
    if total <= n_tokens:
        return t
    step = max(1, total // n_tokens)
    return t[::step][:n_tokens]


def run_cka(
    *,
    model_w,
    model_wo,
    model_spec_w: ModelSpec,
    model_spec_wo: ModelSpec,
    cases: list[tuple[DiagCase, CaseData]],
    target_spec: TargetSpec,
    device,
    n_tokens_per_case: int = CKA_TOKENS_PER_CASE,
    output_dir: str,
) -> tuple[list[dict], list[dict]]:
    """Compute linear CKA between precip_zwd and precip_only at each block."""
    # Collect activations from both models over all cases
    def _collect_all(model, model_spec):
        all_acts: dict[str, list[np.ndarray]] = {}
        for diag_case, case_data in cases:
            precip = case_data.surf_cpu.get(PRECIP_VAR)
            batch = build_precip_batch(case_data, model_spec, device,
                                       precip_override=precip)
            captures: dict[str, torch.Tensor] = {}
            handles = []

            def _make_hook(key):
                def _hook(_m, _i, output):
                    t = output[0] if isinstance(output, tuple) else output
                    captures[key] = t.detach().cpu()
                return _hook

            # Only need block specs once (same architecture)
            H, W = len(case_data.lat_vals), len(case_data.lon_vals)
            patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
            block_specs = _build_block_specs(model, patch_res)

            for module, spec in block_specs:
                handles.append(module.register_forward_hook(_make_hook(spec.key)))
            try:
                with torch.no_grad():
                    _forward(model, batch)
            finally:
                for h in handles:
                    h.remove()
            _gpu_sync()

            for _, spec in block_specs:
                key = spec.key
                sub = _subsample_tokens(captures[key], n_tokens_per_case)
                all_acts.setdefault(key, []).append(sub)

        return block_specs, {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}

    print("    CKA: collecting activations from precip_zwd ...")
    block_specs, acts_w = _collect_all(model_w, model_spec_w)
    print("    CKA: collecting activations from precip_only ...")
    _, acts_wo = _collect_all(model_wo, model_spec_wo)

    block_rows: list[dict] = []
    for _, spec in block_specs:
        k = spec.key
        if k not in acts_w or k not in acts_wo:
            continue
        X = acts_w[k].astype(np.float32)
        Y = acts_wo[k].astype(np.float32)
        # Ensure same N (should be identical)
        N = min(X.shape[0], Y.shape[0])
        cka_val = _linear_cka(X[:N], Y[:N])
        block_rows.append({
            "block_key": k,
            "family": spec.family,
            "stage_name": spec.stage_name,
            "stage_index": spec.stage_index,
            "traversal_index": spec.traversal_index,
            "n_samples": N,
            "cka": cka_val,
        })

    block_rows.sort(key=lambda r: r["traversal_index"])

    # Aggregate by stage
    by_stage: dict[str, list[dict]] = {}
    for r in block_rows:
        by_stage.setdefault(r["stage_name"], []).append(r)
    stage_rows: list[dict] = []
    for stage_name, srows in sorted(by_stage.items(), key=lambda kv: kv[1][0]["traversal_index"]):
        stage_rows.append({
            "stage_name": stage_name,
            "family": srows[0]["family"],
            "stage_index": srows[0]["stage_index"],
            "n_blocks": len(srows),
            "cka_mean": float(np.nanmean([r["cka"] for r in srows])),
            "cka_min":  float(np.nanmin ([r["cka"] for r in srows])),
            "cka_max":  float(np.nanmax ([r["cka"] for r in srows])),
        })

    _write_csv(os.path.join(output_dir, "cka_by_block.csv"), block_rows)
    _write_csv(os.path.join(output_dir, "cka_by_stage.csv"), stage_rows)
    return block_rows, stage_rows


# ---------------------------------------------------------------------------
# Mode: probes
# ---------------------------------------------------------------------------

def run_probes(
    *,
    model_w,
    model_wo,
    model_spec_w: ModelSpec,
    model_spec_wo: ModelSpec,
    cases: list[tuple[DiagCase, CaseData]],
    target_spec: TargetSpec,
    device,
    output_dir: str,
) -> tuple[list[dict], list[dict]]:
    """Ridge probes on stage-terminal pooled hidden vectors."""
    # Collect stage-terminal pooled features for both models
    def _collect_stage_terminal(model, model_spec):
        features: dict[str, list[np.ndarray]] = {}  # stage_name -> list of (D,) vectors
        labels_q850: list[float] = []
        labels_precip: list[float] = []
        labels_zwd: list[float] = []

        for diag_case, case_data in cases:
            target_region = TARGETS[diag_case.target]
            lat_vals, lon_vals = case_data.lat_vals, case_data.lon_vals
            lat_imin, lat_imax, lon_imin, lon_imax = box_indices(target_region, lat_vals, lon_vals)

            precip = case_data.surf_cpu.get(PRECIP_VAR)
            batch = build_precip_batch(case_data, model_spec, device, precip_override=precip)
            captures: dict[str, torch.Tensor] = {}
            handles = []

            H, W = len(lat_vals), len(lon_vals)
            patch_res = (model.encoder.latent_levels, H // model.patch_size, W // model.patch_size)
            block_specs = _build_block_specs(model, patch_res)
            terminal_specs = {spec.stage_name: (block, spec)
                              for block, spec in block_specs if spec.is_stage_terminal}

            def _make_hook(key):
                def _hook(_m, _i, output):
                    t = output[0] if isinstance(output, tuple) else output
                    captures[key] = t.detach().float().cpu()
                return _hook

            for stage_name, (block, spec) in terminal_specs.items():
                handles.append(block.register_forward_hook(_make_hook(stage_name)))

            with torch.no_grad():
                pred = _forward(model, batch)
            for h in handles:
                h.remove()
            _gpu_sync()

            for stage_name, (_, spec) in terminal_specs.items():
                act = captures.get(stage_name)
                if act is None:
                    continue
                # Pool over all tokens -> (D,)
                pooled = act.squeeze(0).mean(dim=0).numpy()
                features.setdefault(stage_name, []).append(pooled.astype(np.float32))

            # Labels: q850 from ERA5 input (atmos_cpu["q"], t1 slice, 850 hPa level)
            from comparison_models import extract_scalar as _es
            from comparison_config import TargetSpec as _TS
            q850_spec = _TS("q850", "q", level_hpa=850)
            try:
                q850_val = _es(pred, q850_spec, lat_vals, lon_vals, target_region)
            except Exception:
                q850_val = float("nan")
            labels_q850.append(q850_val)

            if precip is not None:
                p_box = float(precip[0, 1, lat_imin:lat_imax+1, lon_imin:lon_imax+1].mean().item())
            else:
                p_box = float("nan")
            labels_precip.append(p_box)

            zwd = case_data.surf_cpu.get("zwd")
            if zwd is not None:
                z_box = float(zwd[0, 1, lat_imin:lat_imax+1, lon_imin:lon_imax+1].mean().item())
            else:
                z_box = float("nan")
            labels_zwd.append(z_box)

        return features, np.array(labels_q850), np.array(labels_precip), np.array(labels_zwd)

    print("    Probes: collecting stage-terminal features from precip_zwd ...")
    feats_w, y_q850, y_precip, y_zwd = _collect_stage_terminal(model_w, model_spec_w)
    print("    Probes: collecting stage-terminal features from precip_only ...")
    feats_wo, _, _, _ = _collect_stage_terminal(model_wo, model_spec_wo)

    def _fit_probe(X: np.ndarray, y: np.ndarray, label: str) -> dict:
        """Fit ridge regression, return LOO R² (or R² if too few samples)."""
        mask = ~np.isnan(y)
        X_, y_ = X[mask], y[mask]
        if len(y_) < 3:
            return {"r2": float("nan"), "r2_loo": float("nan"), "n": len(y_), "label": label}
        try:
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import cross_val_score
            clf = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), cv=min(5, len(y_)))
            clf.fit(X_, y_)
            r2_train = float(clf.score(X_, y_))
            scores = cross_val_score(RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0)),
                                     X_, y_, cv=min(5, len(y_)), scoring="r2")
            r2_cv = float(np.mean(scores))
        except Exception:
            # Fallback: numpy lstsq R²
            coef, _, _, _ = np.linalg.lstsq(
                np.c_[X_, np.ones(len(X_))], y_, rcond=None
            )
            pred_y = X_ @ coef[:-1] + coef[-1]
            ss_res = float(np.sum((y_ - pred_y) ** 2))
            ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
            r2_train = 1.0 - _safe_div(ss_res, ss_tot)
            r2_cv = float("nan")
        return {"r2": r2_train, "r2_cv": r2_cv, "n": int(len(y_)), "label": label}

    # Build probe rows
    all_probe_rows: list[dict] = []
    for model_name, feats in [("precip_zwd", feats_w), ("precip_only", feats_wo)]:
        for stage_name, feat_list in feats.items():
            if not feat_list:
                continue
            X = np.stack(feat_list, axis=0)  # (N, D)
            for label, y in [("q850", y_q850), ("precip_t1_box", y_precip), ("zwd_t1_box", y_zwd)]:
                res = _fit_probe(X, y, label)
                all_probe_rows.append({
                    "model": model_name,
                    "stage_name": stage_name,
                    "probe_target": label,
                    **res,
                })

    _write_csv(os.path.join(output_dir, "probe_scores.csv"), all_probe_rows)
    return all_probe_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_stage_curve(path: str, stage_rows: list[dict]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})"); return
    xs = np.arange(len(stage_rows))
    labels = [r["stage_name"] for r in stage_rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, [r.get("plus_delta_rms_mean", float("nan")) for r in stage_rows], marker="o", label="+")
    ax.plot(xs, [r.get("minus_delta_rms_mean", float("nan")) for r in stage_rows], marker="o", label="−")
    ax.plot(xs, [r.get("contrast_rms_mean", float("nan")) for r in stage_rows], marker="o", label="contrast")
    ax.plot(xs, [r.get("common_rms_mean", float("nan")) for r in stage_rows], marker="o", label="common")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean block RMS"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title("Per-stage paired hidden-state response")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def _plot_block_curve(path: str, block_rows: list[dict]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})"); return
    xs = np.arange(len(block_rows))
    labels = [r.get("block_key", str(i)) for i, r in enumerate(block_rows)]
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(xs, [r.get("contrast_rms", float("nan")) for r in block_rows], label="contrast", lw=1.7)
    ax.plot(xs, [r.get("common_rms",   float("nan")) for r in block_rows], label="common",   lw=1.7)
    ax.plot(xs, [r.get("plus_delta_rms",  float("nan")) for r in block_rows], label="+", alpha=0.6, lw=1.1)
    ax.plot(xs, [r.get("minus_delta_rms", float("nan")) for r in block_rows], label="−", alpha=0.6, lw=1.1)
    step = max(1, len(xs) // 30)
    ax.set_xticks(xs[::step]); ax.set_xticklabels(labels[::step], rotation=90, fontsize=7)
    ax.set_ylabel("Block RMS"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)
    ax.set_title("Per-block paired hidden-state response")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def _plot_stage_maps(path: str, stage_maps: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})"); return
    stage_names = sorted(stage_maps.keys(),
                         key=lambda n: (0 if n.startswith("enc_") else 1, int(n.split("_s")[1])))
    if not stage_names:
        return
    ncols = 3; nrows = math.ceil(len(stage_names) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 3.3*nrows))
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(stage_maps[n].max()) for n in stage_names) or None
    for ax, name in zip(axes, stage_names):
        im = ax.imshow(stage_maps[name], origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(stage_names):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    modes = list(dict.fromkeys(args.modes))
    requested_models = set(args.models)
    scales = args.scales
    timesteps = args.timesteps

    two_model_modes = {"cka", "probes"}
    if two_model_modes.intersection(modes) and requested_models != {
        "precip_zwd", "precip_only"
    }:
        raise ValueError("cka and probes require --models precip_zwd precip_only")
    precip_zwd_only_modes = {
        "zwd_trace", "factorial", "zwd_smooth_removal", "zwd_ivw_removal"
    }
    if precip_zwd_only_modes.intersection(modes) and "precip_zwd" not in requested_models:
        raise ValueError(
            f"{sorted(precip_zwd_only_modes.intersection(modes))} require "
            "--models precip_zwd"
        )

    # Resolve target specs
    target_specs: list[TargetSpec] = []
    for tname in args.targets:
        matches = [t for t in DEFAULT_TARGETS if t.name == tname]
        if not matches:
            raise ValueError(f"Unknown target: {tname!r}. Available: {[t.name for t in DEFAULT_TARGETS]}")
        target_specs.append(matches[0])

    # Select diagnostic cases
    print("Selecting diagnostic cases ...")
    if args.diagnostic_selection == "auto":
        diag_cases = _select_diagnostic_cases(
            args.model_conditional_dir, args.n_per_target,
            stage_b_model_name=args.stage_b_model_name,
        )
    else:
        diag_cases = _load_diagnostic_cases_from_file(args.diagnostic_selection)

    reused_region_selections = None
    if args.region_selections:
        reused_region_selections = _load_region_selections(args.region_selections)
        missing = [
            (dc.case_id, scale_name)
            for dc in diag_cases
            for scale_name in scales
            if (dc.case_id, scale_name) not in reused_region_selections
        ]
        if missing:
            raise ValueError(
                f"Region-selection CSV lacks {len(missing)} requested case/scale pairs; "
                f"first missing entries: {missing[:5]}"
            )
        print(
            f"Reusing exact region selections for "
            f"{len(diag_cases) * len(scales)} case/scale pairs from "
            f"{args.region_selections}"
        )

    # Load MSWEP reader
    mswep_reader = None
    try:
        mswep_reader = _make_mswep_reader(MSWEP_STORE_PATH)
        print(f"MSWEP store opened: {MSWEP_STORE_PATH}")
    except Exception as e:
        print(f"WARNING: MSWEP reader failed ({e}). Precipitation modes may not work.")

    # Determine which models are needed
    needs_precip_zwd = "precip_zwd" in requested_models and any(m in modes for m in (
        "zwd_trace", "precip_trace", "q850_trace", "precip_remove_trace",
        "factorial", "cka", "probes",
        "zwd_smooth_removal", "zwd_ivw_removal",
    ))
    needs_precip_only = "precip_only" in requested_models and any(m in modes for m in (
        "precip_trace", "q850_trace", "precip_remove_trace", "cka", "probes",
    ))

    model_w = model_wo = None
    model_spec_w  = DEFAULT_MODELS["precip_zwd"]
    model_spec_wo = DEFAULT_MODELS["precip_only"]

    if needs_precip_zwd:
        print("Loading precip_zwd model ...")
        model_w = load_model(model_spec_w, device)

    if needs_precip_only:
        print("Loading precip_only model ...")
        model_wo = load_model(model_spec_wo, device)

    # Load case data (once per unique init_time)
    print("Loading ERA5 case data ...")
    case_data_map: dict[str, CaseData] = {}
    for dc in diag_cases:
        key = dc.init_time_str
        if key in case_data_map:
            continue
        cd = load_case(dc.init_time)
        if mswep_reader is not None:
            try:
                cd = _load_and_inject_precip(cd, dc.init_time, mswep_reader)
            except Exception as e:
                print(f"  WARNING: MSWEP injection failed for {key}: {e}")
        case_data_map[key] = cd
    print(f"  Loaded {len(case_data_map)} unique case init times.")

    # Dump config
    _write_json(os.path.join(args.output_dir, "config.json"), {
        "modes": modes, "scales": scales, "timesteps": timesteps,
        "targets": [t.name for t in target_specs],
        "models": args.models,
        "region_source_target": args.region_source_target,
        "region_selections": args.region_selections,
        "n_per_target": args.n_per_target,
        "zwd_magnitude": args.zwd_magnitude,
        "precip_dose_mm": args.precip_dose_mm,
        "model_conditional_dir": args.model_conditional_dir,
        "n_cases": len(diag_cases),
    })

    diag_case_rows = [
        {"case_id": dc.case_id, "target": dc.target,
         "init_time": dc.init_time_str, "m_diff_q850": dc.m_diff_q850}
        for dc in diag_cases
    ]
    _write_csv(os.path.join(args.output_dir, "diagnostic_cases.csv"), diag_case_rows)

    # Aggregate containers
    all_pair_rows:     list[dict] = []
    all_block_rows:    list[dict] = []
    all_stage_rows:    list[dict] = []
    all_model_cmp:     list[dict] = []
    all_factorial:     list[dict] = []
    all_selection:     list[dict] = []

    # -----------------------------------------------------------------------
    # CKA and probes: aggregate over all cases (done once, not per scale)
    # -----------------------------------------------------------------------
    if "cka" in modes and model_w is not None and model_wo is not None:
        print("\n=== Mode: cka ===")
        valid_cases = [
            (dc, case_data_map[dc.init_time_str])
            for dc in diag_cases
            if dc.init_time_str in case_data_map
        ]
        for t_spec in target_specs:
            print(f"  target={t_spec.name}")
            run_cka(
                model_w=model_w, model_wo=model_wo,
                model_spec_w=model_spec_w, model_spec_wo=model_spec_wo,
                cases=valid_cases, target_spec=t_spec,
                device=device,
                n_tokens_per_case=args.cka_tokens_per_case,
                output_dir=args.output_dir,
            )

    if "probes" in modes and model_w is not None and model_wo is not None:
        print("\n=== Mode: probes ===")
        valid_cases = [
            (dc, case_data_map[dc.init_time_str])
            for dc in diag_cases
            if dc.init_time_str in case_data_map
        ]
        for t_spec in target_specs:
            print(f"  target={t_spec.name}")
            run_probes(
                model_w=model_w, model_wo=model_wo,
                model_spec_w=model_spec_w, model_spec_wo=model_spec_wo,
                cases=valid_cases, target_spec=t_spec,
                device=device,
                output_dir=args.output_dir,
            )

    # -----------------------------------------------------------------------
    # Per-case, per-scale modes
    # -----------------------------------------------------------------------
    per_case_modes = [m for m in modes if m not in ("cka", "probes")]

    for dc in diag_cases:
        print(f"\n=== Case {dc.case_id} (target={dc.target}) ===")
        case_data = case_data_map.get(dc.init_time_str)
        if case_data is None:
            print(f"  SKIP: no case data")
            continue

        # precip_remove_trace: no scale dependence
        if "precip_remove_trace" in modes:
            for t_spec in target_specs:
                for mspec, mod in [
                    (model_spec_w, model_w), (model_spec_wo, model_wo)
                ]:
                    if mod is None:
                        continue
                    print(f"  [precip_remove_trace] model={mspec.name} target={t_spec.name}")
                    try:
                        result = run_precip_remove_trace(
                            model=mod, model_spec=mspec,
                            case_data=case_data, diag_case=dc,
                            target_spec=t_spec, device=device,
                            output_dir=args.output_dir,
                        )
                        all_model_cmp.extend(result["block_rows"])
                    except Exception as e:
                        print(f"    ERROR: {e}")

        # Scale-dependent modes
        for scale_name in scales:
            print(f"  -- Scale {scale_name}")

            # Load Stage B saliency once from region_source_target (always q850 by default).
            # This avoids requiring zwd saliency maps and keeps spatial perturbation sites
            # identical across all output targets for a fair cross-target comparison.
            regions: list[RegionSelection] = []
            if reused_region_selections is not None:
                regions = reused_region_selections[(dc.case_id, scale_name)]
            else:
                try:
                    saliency = _load_stage_b_saliency(
                        args.model_conditional_dir, dc.case_id, args.region_source_target,
                        model_name=args.stage_b_model_name,
                    )
                    regions = _select_regions_from_saliency(
                        saliency=saliency,
                        target_short=dc.target,
                        scale_name=scale_name,
                        lat_vals=case_data.lat_vals,
                        lon_vals=case_data.lon_vals,
                        low_quantile=args.low_quantile,
                    )
                except FileNotFoundError as e:
                    print(
                        f"    WARNING (Stage B missing for "
                        f"{args.region_source_target}): {e}"
                    )

            for reg in regions:
                all_selection.append({
                    "case_id": dc.case_id, "target": dc.target,
                    "scale": scale_name,
                    "region_source": args.region_source_target,
                    "region_kind": reg.region_kind,
                    "center_lat": reg.mask_spec.center_lat,
                    "center_lon": reg.mask_spec.center_lon,
                    "pooled_saliency": reg.pooled_saliency,
                    "distance_km": reg.distance_km,
                })

            if not regions:
                print(f"    SKIP {scale_name}: no regions (Stage B saliency for "
                      f"'{args.region_source_target}' missing)")
                continue

            for t_spec in target_specs:
                for timestep_name in timesteps:
                    for region in regions:
                        print(f"     region={region.region_kind} ts={timestep_name} target={t_spec.name}")

                        # zwd_trace
                        if "zwd_trace" in per_case_modes and model_w is not None:
                            try:
                                res = run_zwd_trace(
                                    model=model_w,
                                    case_data=case_data, diag_case=dc,
                                    region=region, scale_name=scale_name,
                                    timestep_name=timestep_name, target_spec=t_spec,
                                    model_spec=model_spec_w,
                                    device=device, magnitude=args.zwd_magnitude,
                                    output_dir=args.output_dir, skip_plots=args.skip_plots,
                                )
                                all_pair_rows.append(res["pair_meta"])
                                all_stage_rows.extend(res["stage_rows"])
                            except Exception as e:
                                print(f"      zwd_trace ERROR: {e}")

                        # precip_trace: both models
                        if "precip_trace" in per_case_modes:
                            for mspec, mod in [
                                (model_spec_w, model_w), (model_spec_wo, model_wo)
                            ]:
                                if mod is None:
                                    continue
                                if t_spec.name == "zwd" and not mspec.has_zwd:
                                    # precip_only has no ZWD output head; record as skipped
                                    print(f"      precip_trace ({mspec.name}) SKIP: "
                                          f"no ZWD head, target_var=zwd")
                                    all_pair_rows.append({
                                        "case_id": dc.case_id, "target": dc.target,
                                        "scale": scale_name, "region_kind": region.region_kind,
                                        "perturb_timestep": timestep_name,
                                        "mode": "precip_trace",
                                        "model": mspec.name,
                                        "target_var": t_spec.name,
                                        "skipped": True,
                                        "skip_reason": "no_zwd_output_head",
                                        "mean_abs_target_delta": float("nan"),
                                        "signed_target_response": float("nan"),
                                        "common_target_response": float("nan"),
                                        "plus_target_delta": float("nan"),
                                        "minus_target_delta": float("nan"),
                                    })
                                    continue
                                try:
                                    res = run_precip_trace(
                                        model=mod, model_spec=mspec,
                                        case_data=case_data, diag_case=dc,
                                        region=region, scale_name=scale_name,
                                        timestep_name=timestep_name, target_spec=t_spec,
                                        dose_mm=args.precip_dose_mm,
                                        device=device, output_dir=args.output_dir,
                                        skip_plots=args.skip_plots,
                                    )
                                    all_pair_rows.append(res["pair_meta"])
                                    all_stage_rows.extend(res["stage_rows"])
                                except Exception as e:
                                    print(f"      precip_trace ({mspec.name}) ERROR: {e}")

                        # q850_trace: both models (q is an input channel of each)
                        if "q850_trace" in per_case_modes:
                            for mspec, mod in [
                                (model_spec_w, model_w), (model_spec_wo, model_wo)
                            ]:
                                if mod is None:
                                    continue
                                if t_spec.name == "zwd" and not mspec.has_zwd:
                                    print(f"      q850_trace ({mspec.name}) SKIP: "
                                          f"no ZWD head, target_var=zwd")
                                    all_pair_rows.append({
                                        "case_id": dc.case_id, "target": dc.target,
                                        "scale": scale_name, "region_kind": region.region_kind,
                                        "perturb_timestep": timestep_name,
                                        "mode": "q850_trace",
                                        "model": mspec.name,
                                        "target_var": t_spec.name,
                                        "skipped": True,
                                        "skip_reason": "no_zwd_output_head",
                                        "mean_abs_target_delta": float("nan"),
                                        "signed_target_response": float("nan"),
                                        "common_target_response": float("nan"),
                                        "plus_target_delta": float("nan"),
                                        "minus_target_delta": float("nan"),
                                    })
                                    continue
                                try:
                                    res = run_q850_trace(
                                        model=mod, model_spec=mspec,
                                        case_data=case_data, diag_case=dc,
                                        region=region, scale_name=scale_name,
                                        timestep_name=timestep_name, target_spec=t_spec,
                                        magnitude=args.q850_magnitude,
                                        device=device, output_dir=args.output_dir,
                                        skip_plots=args.skip_plots,
                                    )
                                    all_pair_rows.append(res["pair_meta"])
                                    all_stage_rows.extend(res["stage_rows"])
                                except Exception as e:
                                    print(f"      q850_trace ({mspec.name}) ERROR: {e}")

                        # factorial: precip_zwd only
                        if "factorial" in per_case_modes and model_w is not None:
                            try:
                                rows = run_factorial(
                                    model=model_w, model_spec=model_spec_w,
                                    case_data=case_data, diag_case=dc,
                                    region=region, scale_name=scale_name,
                                    timestep_name=timestep_name, target_spec=t_spec,
                                    dose_mm=args.precip_dose_mm,
                                    zwd_magnitude=args.zwd_magnitude,
                                    device=device, output_dir=args.output_dir,
                                )
                                all_factorial.extend(rows)
                            except Exception as e:
                                print(f"      factorial ERROR: {e}")

                        # zwd_smooth_removal: precip_zwd model only (has_zwd required)
                        if "zwd_smooth_removal" in per_case_modes \
                                and model_w is not None and model_spec_w.has_zwd:
                            try:
                                baseline = _compute_baseline_zwd(
                                    "smooth", case_data, args.zwd_smooth_sigma
                                )
                                res = run_zwd_removal_trace(
                                    model=model_w, case_data=case_data, diag_case=dc,
                                    region=region, scale_name=scale_name,
                                    timestep_name=timestep_name, target_spec=t_spec,
                                    model_spec=model_spec_w, device=device,
                                    baseline_zwd_cpu=baseline, baseline_label="smooth",
                                    output_dir=args.output_dir, skip_plots=args.skip_plots,
                                )
                                all_model_cmp.extend(res["block_rows"])
                            except Exception as e:
                                print(f"      zwd_smooth_removal ERROR: {e}")

                        # zwd_ivw_removal: precip_zwd model only (has_zwd required)
                        if "zwd_ivw_removal" in per_case_modes \
                                and model_w is not None and model_spec_w.has_zwd:
                            try:
                                baseline = _compute_baseline_zwd(
                                    "ivw", case_data, args.zwd_smooth_sigma
                                )
                                res = run_zwd_removal_trace(
                                    model=model_w, case_data=case_data, diag_case=dc,
                                    region=region, scale_name=scale_name,
                                    timestep_name=timestep_name, target_spec=t_spec,
                                    model_spec=model_spec_w, device=device,
                                    baseline_zwd_cpu=baseline, baseline_label="ivw",
                                    output_dir=args.output_dir, skip_plots=args.skip_plots,
                                )
                                all_model_cmp.extend(res["block_rows"])
                            except Exception as e:
                                print(f"      zwd_ivw_removal ERROR: {e}")

    # Write global aggregates
    _write_csv(os.path.join(args.output_dir, "selections.csv"), all_selection)
    _write_csv(os.path.join(args.output_dir, "all_pair_runs.csv"), all_pair_rows)
    _write_csv(os.path.join(args.output_dir, "all_pair_stage_metrics.csv"), all_stage_rows)
    _write_csv(os.path.join(args.output_dir, "all_model_metric_comparisons.csv"), all_model_cmp)
    _write_csv(os.path.join(args.output_dir, "all_factorial_interactions.csv"), all_factorial)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
