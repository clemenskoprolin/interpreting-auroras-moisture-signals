"""
Completeness diagnostic — 6 h ZWD forecast
===========================================
Verifies the IG completeness axiom:  sum(IG) ≈ f(x) - f(x')

Runs IG at multiple step counts on the representative 6 h Ticino point target.
Reports:
  ig_sum_all  — sum of the full (1,2,H,W) IG attribution over both ZWD timesteps
  target_delta — f(actual_zwd) - f(smoothed_baseline_zwd)
  gap          — ig_sum_all - target_delta  (should converge to 0)
  rel_gap      — |gap| / |target_delta|
  t1_fraction  — |attr_t1| / (|attr_t0| + |attr_t1|)
               — informative because the benchmark only evaluates the t1 slice
               — while IG attributes both timesteps

Usage
-----
  python 01_attribution_method_validation/completeness.py
  # or via entry gate:
  python 01_attribution_method_validation/diagnostics.py \
      completeness [--steps 8 16 32 64]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import (
    import_benchmark, gpu_sync, save_json,
    OUTPUT_DIR,
    REP_CASE_TARGET, REP_CASE_INIT, REP_CASE_ID_POINT, REP_SCALE,
)


def run(args) -> None:
    (
        setup_model, run_saliency, run_ig, make_q850_target, _forward,
        _saved_tensors_cpu_context, _RolloutForwardWrapper,
        load_case, make_batch,
        TARGETS, SCALES, generate_mask_centers, gaussian_mask,
        cos_lat_weights, smoothed_zwd_baseline,
        pool_attribution, spearman,
        xia_ig,
    ) = import_benchmark()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[completeness] device={device}")

    out_dir = os.path.join(OUTPUT_DIR, "completeness")
    os.makedirs(out_dir, exist_ok=True)

    case_data = load_case(REP_CASE_INIT)
    scale = SCALES[REP_SCALE]
    zwd_actual = case_data.surf_cpu["zwd"]
    zwd_bl_cpu = smoothed_zwd_baseline(zwd_actual, scale.sigma_deg)
    zwd_delta   = zwd_actual - zwd_bl_cpu

    model = setup_model(device)

    # --- 6h ---
    target_fn_pt, _ = make_q850_target(case_data, REP_CASE_TARGET, "point")
    print("  6h: computing f_actual and f_baseline...")
    with torch.no_grad():
        f_actual   = float(target_fn_pt(_forward(model, make_batch(case_data, device))).item())
        f_baseline = float(target_fn_pt(
            _forward(model, make_batch(case_data, device, zwd_override=zwd_bl_cpu))
        ).item())
    target_delta_6h = f_actual - f_baseline
    print(f"  f_actual={f_actual:.6f}  f_baseline={f_baseline:.6f}  "
          f"delta={target_delta_6h:.6f}")

    rows_6h = _run_sweep(
        model=model, case_data=case_data,
        target_fn=target_fn_pt, device=device,
        zwd_actual=zwd_actual, zwd_bl_cpu=zwd_bl_cpu, zwd_delta=zwd_delta,
        make_batch=make_batch, xia_ig=xia_ig,
        _saved_tensors_cpu_context=_saved_tensors_cpu_context,
        f_actual=f_actual, f_baseline=f_baseline,
        step_counts=args.steps, rollout_steps=1, label="6h",
    )
    save_json(
        {"case": REP_CASE_ID_POINT, "scale": REP_SCALE, "lead_time_hours": 6,
         "f_actual": f_actual, "f_baseline": f_baseline,
         "target_delta": target_delta_6h, "rows": rows_6h},
        os.path.join(out_dir, "completeness_6h.json"),
    )
    _print_table(rows_6h, "6h")
    gpu_sync()

    print("\n[completeness] Done.")


# ------------------------------------------------------------------
# Core sweep
# ------------------------------------------------------------------

def _run_sweep(
    *, model, case_data, target_fn, device,
    zwd_actual, zwd_bl_cpu, zwd_delta,
    make_batch, xia_ig, _saved_tensors_cpu_context,
    f_actual, f_baseline,
    step_counts: list[int], rollout_steps: int, label: str,
) -> list[dict]:
    rows = []
    for n_steps in step_counts:
        print(f"\n  [{label}] n_steps={n_steps}...")
        t0 = time.time()

        def batch_fn(alpha, requires_grad=False):
            zwd_interp = (zwd_bl_cpu + alpha * zwd_delta).clone()
            return make_batch(
                case_data, device,
                requires_grad_surf=("zwd",) if requires_grad else (),
                zwd_override=zwd_interp,
            )

        with _saved_tensors_cpu_context(rollout_steps > 1):
            result = xia_ig(
                model=model, batch_fn=batch_fn, target_fn=target_fn,
                surf_actual={"zwd": zwd_actual},
                surf_baseline={"zwd": zwd_bl_cpu},
                surf_var_names=("zwd",), device=device, n_steps=n_steps,
            )

        ig = result["ig"]["zwd"].astype(np.float64)  # (1, 2, H, W)
        abs_t0 = float(np.abs(ig[0, 0]).sum())
        abs_t1 = float(np.abs(ig[0, 1]).sum())
        ig_sum_all = float(ig.sum())
        target_delta = f_actual - f_baseline
        gap = ig_sum_all - target_delta
        elapsed = time.time() - t0

        row = {
            "n_steps": n_steps,
            "rollout_steps": rollout_steps,
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
        print(
            f"  ig_sum={ig_sum_all:.6f} target={target_delta:.6f} "
            f"gap={gap:.6f} rel_gap={row['rel_gap']:.3%} "
            f"rel_energy={row['rel_energy']:.3%} "
            f"t1_frac={row['t1_fraction']:.3f}  ({elapsed:.1f}s)"
        )
        gpu_sync()
    return rows


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _print_table(rows: list[dict], label: str) -> None:
    print(f"\n  [{label}] Completeness:")
    print(f"  {'steps':>6} {'ig_sum':>10} {'target':>10} {'gap':>10} "
          f"{'rel_gap':>8} {'rel_E':>8} {'t1_frac':>8}")
    for r in rows:
        print(
            f"  {r['n_steps']:>6} {r['ig_sum_all']:>10.5f} {r['target_delta']:>10.5f} "
            f"  {r['gap']:>10.5f} {r['rel_gap']:>8.3%} {r['rel_energy']:>8.3%} {r['t1_fraction']:>8.3f}"
        )


# ------------------------------------------------------------------
# Standalone entry
# ------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ZWD IG completeness diagnostic")
    p.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    return p.parse_args()


if __name__ == "__main__":
    import time as _time
    from datetime import datetime as _dt
    _args = _parse_args()
    _t0 = _time.time()
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] completeness (standalone)")
    run(_args)
    _elapsed = _time.time() - _t0
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] Done in {_elapsed:.0f}s ({_elapsed/60:.1f}m)")
