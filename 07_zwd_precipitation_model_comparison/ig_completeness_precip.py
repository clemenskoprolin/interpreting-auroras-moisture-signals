"""
IG completeness diagnostic for the fine-tuned precipitation model
=================================================================
Precip-model mirror of zwd_diagnostics/completeness.py, testing the
supervisor hypothesis that the completeness gap on the ZWD searchlight
model (~13% at 6h) is caused by fine-tuning only on the new inputs:
the precip models were likewise fine-tuned with new surface inputs
(tp_mswep, zwd), so if fine-tuning is the cause, the gap should
reproduce here too.

Verifies the IG completeness axiom:  sum(IG) ~= f(x) - f(x')
  x  = actual ZWD field, x' = spatially smoothed ZWD baseline
  f  = 6h box-mean target (tp_mswep or q850) of precip_large_zwd

Float regime matches the fixed searchlight setup on both sides of the
axiom: load_model() constructs the model with autocast=True
(backbone-only bf16, decoder fp32) and xia_methods/ig.py enforces the
same regime internally.

Usage
-----
  python 07_zwd_precipitation_model_comparison/ig_completeness_precip.py \
      [--steps 8 16 32 64] [--case-targets ticino japan california alps_east] \
      [--target-vars precip q850]

Under SLURM with ntasks>1, cases are round-robined across ranks
(SLURM_PROCID), one model instance per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "02_zwd_attribution_benchmark"), _HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASELINE_SIGMA_DEG = 2.5  # matches skill_run_salgain_ig.py / geoxplain adapter
CASES_PATH = os.path.join(_HERE, "cases_precipitation.json")
OUTPUT_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "precip_ig_completeness")


def _parse_args():
    p = argparse.ArgumentParser(description="Precip-model IG completeness")
    p.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--case-targets", nargs="+",
                   default=["ticino", "japan", "california", "alps_east"],
                   help="One strong case (highest score) is used per target")
    p.add_argument("--target-vars", nargs="+", default=["precip", "q850"],
                   choices=["precip", "q850"])
    p.add_argument("--model", default="precip_large_zwd")
    return p.parse_args()


def main():
    args = _parse_args()

    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    from comparison_config import DEFAULT_MODELS, TargetSpec, PRECIP_VAR  # noqa: E402
    from comparison_models import load_model  # noqa: E402
    from comparison_data import build_precip_batch, _ensure_dir  # noqa: E402
    from searchlight_tasks import TARGETS, box_indices  # noqa: E402
    from searchlight_ground_truth import smoothed_zwd_baseline  # noqa: E402
    from xia_methods.ig import integrated_gradients  # noqa: E402
    import model_conditional_comparison as _mcc  # noqa: E402

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START precip IG completeness "
          f"model={args.model} steps={args.steps} device={device}")
    _ensure_dir(OUTPUT_DIR)

    with open(CASES_PATH) as f:
        all_cases = json.load(f)
    cases = []
    for tgt in args.case_targets:
        cands = [c for c in all_cases if c["target"] == tgt]
        if not cands:
            print(f"  WARNING: no case for target {tgt}")
            continue
        cases.append(max(cands, key=lambda c: c["score"]))

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    cases = cases[rank::world]
    print(f"rank {rank}/{world}: {len(cases)} cases: "
          + ", ".join(f"{c['target']}@{c['init_time']}" for c in cases))

    target_specs = {
        "precip": TargetSpec("precip", PRECIP_VAR),
        "q850": TargetSpec("q850", "q", level_hpa=850),
    }

    model_spec = DEFAULT_MODELS[args.model]
    model = load_model(model_spec, device)

    for entry in cases:
        case_id = f"{entry['target']}_{entry['init_time']}"
        print(f"\n=== {case_id} ===")
        cd_map = _mcc._load_case_data([entry], None)
        case_data = cd_map.get(entry["init_time"])
        if case_data is None:
            print("  ERROR: case data failed to load")
            continue

        region = TARGETS[entry["target"]]
        lat_imin, lat_imax, lon_imin, lon_imax = box_indices(
            region, case_data.lat_vals, case_data.lon_vals)
        precip_override = case_data.surf_cpu.get(PRECIP_VAR)

        zwd_actual = case_data.surf_cpu["zwd"]                     # (1,2,H,W)
        zwd_baseline = smoothed_zwd_baseline(zwd_actual, BASELINE_SIGMA_DEG)
        zwd_delta = zwd_actual - zwd_baseline

        def batch_fn(alpha=0.0, requires_grad=False,
                     _cd=case_data, _po=precip_override):
            zwd_interp = (zwd_baseline + alpha * zwd_delta).clone()
            return build_precip_batch(
                _cd, model_spec, device,
                precip_override=_po,
                zwd_override=zwd_interp,
                requires_grad_surf=("zwd",) if requires_grad else (),
            )

        for tv in args.target_vars:
            t_spec = target_specs[tv]
            if t_spec.level_hpa is not None:
                def target_fn(pred, _lh=t_spec.level_hpa, _ov=t_spec.output_var):
                    levels_arr = np.asarray(pred.metadata.atmos_levels)
                    lidx = int(np.where(levels_arr == _lh)[0][0])
                    arr = pred.atmos_vars[_ov].float()
                    return arr[0, 0, lidx, lat_imin:lat_imax + 1,
                               lon_imin:lon_imax + 1].mean()
            else:
                def target_fn(pred, _ov=t_spec.output_var):
                    arr = pred.surf_vars[_ov].float()
                    return arr[0, 0, lat_imin:lat_imax + 1,
                               lon_imin:lon_imax + 1].mean()

            print(f"\n  [{tv}] computing f_actual / f_baseline ...")
            with torch.no_grad():
                pred_a = model.forward(batch_fn(alpha=1.0))
                if isinstance(pred_a, tuple):
                    pred_a = pred_a[0]
                f_actual = float(target_fn(pred_a).item())
                del pred_a
                pred_b = model.forward(batch_fn(alpha=0.0))
                if isinstance(pred_b, tuple):
                    pred_b = pred_b[0]
                f_baseline = float(target_fn(pred_b).item())
                del pred_b
            torch.cuda.empty_cache()
            target_delta = f_actual - f_baseline
            print(f"  f_actual={f_actual:.6f} f_baseline={f_baseline:.6f} "
                  f"delta={target_delta:.6e}")

            rows = []
            for n_steps in args.steps:
                print(f"\n  [{tv}] n_steps={n_steps} ...")
                t0 = time.time()
                result = integrated_gradients(
                    model=model, batch_fn=batch_fn, target_fn=target_fn,
                    surf_actual={"zwd": zwd_actual},
                    surf_baseline={"zwd": zwd_baseline},
                    surf_var_names=("zwd",), device=str(device),
                    n_steps=n_steps,
                )
                ig = result["ig"]["zwd"].astype(np.float64)  # (1,2,H,W)
                abs_t0 = float(np.abs(ig[0, 0]).sum())
                abs_t1 = float(np.abs(ig[0, 1]).sum())
                ig_sum_all = float(ig.sum())
                gap = ig_sum_all - target_delta
                elapsed = time.time() - t0
                row = {
                    "n_steps": n_steps,
                    "ig_sum_all": ig_sum_all,
                    "ig_sum_t0": float(ig[0, 0].sum()),
                    "ig_sum_t1": float(ig[0, 1].sum()),
                    "t1_fraction": abs_t1 / (abs_t0 + abs_t1 + 1e-30),
                    "target_delta": target_delta,
                    "gap": gap,
                    "rel_gap": abs(gap) / (abs(target_delta) + 1e-30),
                    "rel_energy": abs(gap) / (abs_t0 + abs_t1 + 1e-30),
                    "runtime_s": elapsed,
                }
                rows.append(row)
                print(f"  ig_sum={ig_sum_all:.6e} target={target_delta:.6e} "
                      f"gap={gap:.6e} rel_gap={row['rel_gap']:.3%} "
                      f"rel_energy={row['rel_energy']:.3%} "
                      f"t1_frac={row['t1_fraction']:.3f}  ({elapsed:.1f}s)")

            out = {
                "model": args.model, "case": case_id,
                "target_var": tv, "lead_time_hours": 6,
                "baseline_sigma_deg": BASELINE_SIGMA_DEG,
                "f_actual": f_actual, "f_baseline": f_baseline,
                "target_delta": target_delta, "rows": rows,
            }
            out_path = os.path.join(
                OUTPUT_DIR, f"completeness_{case_id}_{tv}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Saved: {out_path}")

            print(f"\n  [{case_id} / {tv}] Completeness:")
            print(f"  {'steps':>6} {'ig_sum':>12} {'target':>12} {'gap':>12} "
                  f"{'rel_gap':>8} {'rel_E':>8} {'t1_frac':>8}")
            for r in rows:
                print(f"  {r['n_steps']:>6} {r['ig_sum_all']:>12.5e} "
                      f"{r['target_delta']:>12.5e} {r['gap']:>12.5e} "
                      f"{r['rel_gap']:>8.3%} {r['rel_energy']:>8.3%} "
                      f"{r['t1_fraction']:>8.3f}")

        del cd_map, case_data
        torch.cuda.empty_cache()

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
