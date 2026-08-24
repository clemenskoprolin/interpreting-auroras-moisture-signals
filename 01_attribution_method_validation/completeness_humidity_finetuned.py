"""
Matched IG-completeness checks for humidity in the fine-tuned checkpoints.

This diagnostic asks whether the larger completeness residuals observed when
attributing ZWD also occur for an original Aurora input, specific humidity
(`q`), in the same fine-tuned models and numerical regime.

Three deliberately small tests are assigned by SLURM rank:

  rank 0: ZWD searchlight checkpoint, Ticino q850 point target
  rank 1: precipitation checkpoint, Ticino precipitation box target
  rank 2: precipitation checkpoint, Japan precipitation box target

For each test, all pressure levels and both history steps of q are interpolated
between the actual field and a 2.5-degree spatially smoothed baseline. IG uses
the same midpoint implementation and backbone-only bfloat16 autocast as the
corresponding ZWD completeness diagnostics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_PRECIP_DIR = os.path.join(_ROOT, "07_zwd_precipitation_model_comparison")
for _path in (_HERE, _ROOT, _SEARCHLIGHT_DIR, _PRECIP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SIGMA_DEG = 2.5
STEP_COUNTS = (8, 32, 64)
OUTPUT_DIR = os.path.join(
    os.environ.get(
        "ATTRIBUTION_VALIDATION_RESULTS_DIR",
        os.path.join(
            os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
            "attribution_method_validation",
        ),
    ),
    "humidity_completeness",
)


def _smooth_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Smooth every q level independently in latitude and longitude."""
    from scipy.ndimage import gaussian_filter

    arr = tensor.detach().cpu().numpy().copy()
    sigma_pixels = SIGMA_DEG / 0.25
    for batch_idx in range(arr.shape[0]):
        for time_idx in range(arr.shape[1]):
            for level_idx in range(arr.shape[2]):
                arr[batch_idx, time_idx, level_idx] = gaussian_filter(
                    arr[batch_idx, time_idx, level_idx],
                    sigma=sigma_pixels,
                    mode=("reflect", "wrap"),
                )
    return torch.from_numpy(arr).float()


def _run_sweep(
    *,
    model,
    batch_fn,
    target_fn,
    q_actual: torch.Tensor,
    q_baseline: torch.Tensor,
    f_actual: float,
    f_baseline: float,
    saved_tensors_context=None,
) -> list[dict]:
    from xia_methods.ig import integrated_gradients

    target_delta = f_actual - f_baseline
    rows = []
    for n_steps in STEP_COUNTS:
        started = time.time()
        context = (
            saved_tensors_context()
            if saved_tensors_context is not None
            else nullcontext()
        )
        with context:
            result = integrated_gradients(
                model=model,
                batch_fn=batch_fn,
                target_fn=target_fn,
                atmos_actual={"q": q_actual},
                atmos_baseline={"q": q_baseline},
                atmos_var_names=("q",),
                device="cuda",
                n_steps=n_steps,
            )

        ig = np.asarray(result["ig"]["q"], dtype=np.float64)
        ig_sum = float(ig.sum())
        ig_abs = float(np.abs(ig).sum())
        gap = ig_sum - target_delta
        row = {
            "n_steps": n_steps,
            "ig_sum": ig_sum,
            "ig_abs": ig_abs,
            "target_delta": target_delta,
            "gap": gap,
            "rel_gap": abs(gap) / (abs(target_delta) + 1e-30),
            "rel_energy": abs(gap) / (ig_abs + 1e-30),
            "runtime_s": time.time() - started,
        }
        rows.append(row)
        print(
            f"n={n_steps:>2} sumIG={ig_sum:.8e} delta={target_delta:.8e} "
            f"gap={gap:.3e} rel_gap={row['rel_gap']:.3%} "
            f"rel_E={row['rel_energy']:.3%}",
            flush=True,
        )
        torch.cuda.empty_cache()
    return rows


def _run_zwd_checkpoint() -> dict:
    from common import REP_CASE_INIT, REP_CASE_TARGET, import_benchmark

    (
        setup_model,
        _run_saliency,
        _run_ig,
        make_q850_target,
        forward,
        saved_tensors_cpu_context,
        _rollout_wrapper,
        load_case,
        make_batch,
        _targets,
        _scales,
        _generate_mask_centers,
        _gaussian_mask,
        _cos_lat_weights,
        _smoothed_zwd_baseline,
        _pool_attribution,
        _spearman,
        _xia_ig,
    ) = import_benchmark()

    case_data = load_case(REP_CASE_INIT)
    target_fn, _ = make_q850_target(case_data, REP_CASE_TARGET, "point")
    q_actual = case_data.atmos_cpu["q"]
    q_baseline = _smooth_spatial(q_actual)
    q_delta = q_actual - q_baseline

    def batch_fn(alpha=0.0, requires_grad=False):
        q_interp = (q_baseline + alpha * q_delta).clone()
        return make_batch(
            case_data,
            "cuda",
            requires_grad_atmos=("q",) if requires_grad else (),
            q_override=q_interp,
        )

    model = setup_model("cuda")
    with torch.no_grad():
        f_actual = float(target_fn(forward(model, batch_fn(alpha=1.0))).item())
        f_baseline = float(target_fn(forward(model, batch_fn(alpha=0.0))).item())
    print(
        f"zwd checkpoint / Ticino q850 point: f(x)={f_actual:.8e} "
        f"f(x')={f_baseline:.8e} delta={f_actual - f_baseline:.8e}",
        flush=True,
    )

    return {
        "test_id": "zwd_checkpoint_ticino_q850_point",
        "checkpoint": "zwd_searchlight",
        "case": REP_CASE_INIT.isoformat(),
        "target": "q850_point_ticino",
        "attributed_input": "q_all_levels_both_times",
        "baseline_sigma_deg": SIGMA_DEG,
        "f_actual": f_actual,
        "f_baseline": f_baseline,
        "rows": _run_sweep(
            model=model,
            batch_fn=batch_fn,
            target_fn=target_fn,
            q_actual=q_actual,
            q_baseline=q_baseline,
            f_actual=f_actual,
            f_baseline=f_baseline,
            saved_tensors_context=lambda: saved_tensors_cpu_context(False),
        ),
    }


def _run_precip_checkpoint(target_name: str) -> dict:
    from comparison_config import DEFAULT_MODELS, PRECIP_VAR
    from comparison_data import build_precip_batch
    from comparison_models import load_model
    import model_conditional_comparison as comparison
    from searchlight_tasks import TARGETS, box_indices

    cases_path = os.path.join(_PRECIP_DIR, "cases_precipitation.json")
    with open(cases_path) as handle:
        candidates = [
            case for case in json.load(handle) if case["target"] == target_name
        ]
    if not candidates:
        raise RuntimeError(f"No precipitation case found for {target_name!r}")
    entry = max(candidates, key=lambda case: case["score"])

    case_map = comparison._load_case_data([entry], None)
    case_data = case_map[entry["init_time"]]
    region = TARGETS[target_name]
    lat_min, lat_max, lon_min, lon_max = box_indices(
        region, case_data.lat_vals, case_data.lon_vals
    )

    model_spec = DEFAULT_MODELS["precip_large_zwd"]
    precip_override = case_data.surf_cpu.get(PRECIP_VAR)
    q_actual = case_data.atmos_cpu["q"]
    q_baseline = _smooth_spatial(q_actual)
    q_delta = q_actual - q_baseline

    def batch_fn(alpha=0.0, requires_grad=False):
        q_interp = (q_baseline + alpha * q_delta).clone()
        return build_precip_batch(
            case_data,
            model_spec,
            "cuda",
            precip_override=precip_override,
            atmos_override={"q": q_interp},
            requires_grad_atmos=("q",) if requires_grad else (),
        )

    def target_fn(pred):
        arr = pred.surf_vars[PRECIP_VAR].float()
        return arr[
            0,
            0,
            lat_min : lat_max + 1,
            lon_min : lon_max + 1,
        ].mean()

    model = load_model(model_spec, "cuda")

    def forward(batch):
        pred = model.forward(batch)
        return pred[0] if isinstance(pred, tuple) else pred

    with torch.no_grad():
        f_actual = float(target_fn(forward(batch_fn(alpha=1.0))).item())
        f_baseline = float(target_fn(forward(batch_fn(alpha=0.0))).item())
    print(
        f"precip checkpoint / {target_name} precip box: f(x)={f_actual:.8e} "
        f"f(x')={f_baseline:.8e} delta={f_actual - f_baseline:.8e}",
        flush=True,
    )

    return {
        "test_id": f"precip_checkpoint_{target_name}_precip_box",
        "checkpoint": "precip_large_zwd",
        "case": entry["init_time"],
        "target": f"precip_box_{target_name}",
        "attributed_input": "q_all_levels_both_times",
        "baseline_sigma_deg": SIGMA_DEG,
        "f_actual": f_actual,
        "f_baseline": f_baseline,
        "rows": _run_sweep(
            model=model,
            batch_fn=batch_fn,
            target_fn=target_fn,
            q_actual=q_actual,
            q_baseline=q_baseline,
            f_actual=f_actual,
            f_baseline=f_baseline,
        ),
    }


def main() -> None:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] humidity completeness "
        f"rank={rank} device=cuda:0",
        flush=True,
    )
    if rank == 0:
        result = _run_zwd_checkpoint()
    elif rank == 1:
        result = _run_precip_checkpoint("ticino")
    elif rank == 2:
        result = _run_precip_checkpoint("japan")
    else:
        print(f"No test assigned to rank {rank}; exiting.", flush=True)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{result['test_id']}.json")
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
