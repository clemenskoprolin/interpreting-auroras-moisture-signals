"""
Small IG-completeness check for the native 2 m temperature surface input.

Runs on the large precipitation+ZWD checkpoint. Targets are assigned to ranks
from the comma-separated ``IG_2T_TARGETS`` environment variable, which defaults
to ``ticino,japan``. Only 8 and 16 midpoint steps are used because the purpose
is to distinguish a general surface-input issue from one specific to the newly
introduced ZWD pathway.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
for _path in (_ROOT, _SEARCHLIGHT_DIR, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SIGMA_DEG = 2.5
STEP_COUNTS = (8, 16)
OUTPUT_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "surface_2t_ig_completeness",
)


def _smooth_surface(tensor: torch.Tensor) -> torch.Tensor:
    from scipy.ndimage import gaussian_filter

    arr = tensor.detach().cpu().numpy().copy()
    sigma_pixels = SIGMA_DEG / 0.25
    for batch_idx in range(arr.shape[0]):
        for time_idx in range(arr.shape[1]):
            arr[batch_idx, time_idx] = gaussian_filter(
                arr[batch_idx, time_idx],
                sigma=sigma_pixels,
                mode=("reflect", "wrap"),
            )
    return torch.from_numpy(arr).float()


def main() -> None:
    from comparison_config import DEFAULT_MODELS, PRECIP_VAR
    from comparison_data import build_precip_batch
    from comparison_models import load_model
    import model_conditional_comparison as comparison
    from searchlight_tasks import TARGETS, box_indices
    from xia_methods.ig import integrated_gradients

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    target_spec = os.environ.get("IG_2T_TARGETS", "ticino,japan")
    target_names = [
        name.strip()
        for name in target_spec.replace(":", ",").split(",")
        if name.strip()
    ]
    if rank >= len(target_names):
        print(f"No 2t completeness test assigned to rank {rank}; exiting.")
        return
    target_name = target_names[rank]
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 2t completeness "
        f"rank={rank} target={target_name} device=cuda:0",
        flush=True,
    )

    cases_path = os.path.join(_HERE, "cases_precipitation.json")
    with open(cases_path) as handle:
        candidates = [
            case for case in json.load(handle) if case["target"] == target_name
        ]
    entry = max(candidates, key=lambda case: case["score"])
    case_data = comparison._load_case_data([entry], None)[entry["init_time"]]

    region = TARGETS[target_name]
    lat_min, lat_max, lon_min, lon_max = box_indices(
        region, case_data.lat_vals, case_data.lon_vals
    )
    model_spec = DEFAULT_MODELS["precip_large_zwd"]
    precip_override = case_data.surf_cpu.get(PRECIP_VAR)

    actual = case_data.surf_cpu["2t"]
    baseline = _smooth_surface(actual)
    input_delta = actual - baseline

    def batch_fn(alpha=0.0, requires_grad=False):
        interpolated = (baseline + alpha * input_delta).clone()
        surf_cpu = dict(case_data.surf_cpu)
        surf_cpu["2t"] = interpolated
        interpolated_case = dataclasses.replace(case_data, surf_cpu=surf_cpu)
        return build_precip_batch(
            interpolated_case,
            model_spec,
            "cuda",
            precip_override=precip_override,
            requires_grad_surf=("2t",) if requires_grad else (),
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
        prediction = model.forward(batch)
        return prediction[0] if isinstance(prediction, tuple) else prediction

    with torch.no_grad():
        f_actual = float(target_fn(forward(batch_fn(alpha=1.0))).item())
        f_baseline = float(target_fn(forward(batch_fn(alpha=0.0))).item())
    target_delta = f_actual - f_baseline
    print(
        f"f(x)={f_actual:.8e} f(x')={f_baseline:.8e} "
        f"delta={target_delta:.8e}",
        flush=True,
    )

    rows = []
    for n_steps in STEP_COUNTS:
        started = time.time()
        result = integrated_gradients(
            model=model,
            batch_fn=batch_fn,
            target_fn=target_fn,
            surf_actual={"2t": actual},
            surf_baseline={"2t": baseline},
            surf_var_names=("2t",),
            device="cuda",
            n_steps=n_steps,
        )
        ig = np.asarray(result["ig"]["2t"], dtype=np.float64)
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

    output = {
        "test_id": f"precip_large_zwd_{target_name}_precip_box_2t",
        "checkpoint": "precip_large_zwd",
        "case": entry["init_time"],
        "target": f"precip_box_{target_name}",
        "attributed_input": "2t_both_times",
        "baseline_sigma_deg": SIGMA_DEG,
        "f_actual": f_actual,
        "f_baseline": f_baseline,
        "rows": rows,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{output['test_id']}.json")
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
