"""IG completeness across the case grid: Σ_i IG(x_i) ≈ f(x) − f(x').

For every case in ``_suite_common.CASES`` and every step count, report the
absolute and relative residual of the completeness axiom. The baseline matches
the shipped IG path: a Gaussian-smoothed actual field (σ = 2.5°, reflect in
latitude, wrap in longitude), midpoint Riemann rule over n_steps ∈ {4,8,16,32,64}.
f(x) and f(x') use the same backbone-only bf16 autocast as the IG gradient.

Residuals emitted per (case, n_steps):
    abs_gap    = Σ IG − (f(x) − f(x'))
    rel_gap    = |abs_gap| / |f(x) − f(x')|     completeness denominator
    rel_energy = |abs_gap| / Σ|IG|              robust when Δ ≈ 0

Sharded across SLURM ranks; aggregated by aggregate.py.
"""

import time
import traceback

import numpy as np
import torch

from geoxplain_aurora_adapter.engine.model import load_model
from geoxplain_aurora_adapter.engine._common import (
    _parse_init_time, _smooth_tensor, IG_BASELINE_SIGMA_DEG,
)
from geoxplain_aurora_adapter.engine.rollout import _forward_unwrapped
from geoxplain_aurora_adapter.engine.data import load_case, make_batch
from geoxplain_aurora_adapter.schema.targets import build_target_fn
from geoxplain_aurora_adapter.engine.xia_methods.ig import integrated_gradients

from _suite_common import CASES, JsonlWriter, rank_world, shard

DEV = "cuda"
STEP_COUNTS = [4, 8, 16, 32, 64]


def completeness_for_case(model, case, writer, rank):
    atmos = list(case.atmos_vars)
    case_data = load_case(_parse_init_time(case.timestamp))
    target_fn = build_target_fn(case.target, case_data)

    actual = {v: case_data.atmos_cpu[v] for v in atmos}
    baseline = {v: _smooth_tensor(case_data.atmos_cpu[v], IG_BASELINE_SIGMA_DEG) for v in atmos}

    def batch_fn(alpha=0.0, requires_grad=False):
        over = {v: (baseline[v] + alpha * (actual[v] - baseline[v])).clone() for v in atmos}
        return make_batch(
            case_data, DEV,
            requires_grad_atmos=tuple(atmos) if requires_grad else (),
            atmos_overrides=over,
        )

    # f(x), f(x') under the IG path's backbone-only autocast.
    orig_ac = getattr(model, "autocast", False)
    model.autocast = True
    with torch.no_grad():
        f_actual = float(target_fn(_forward_unwrapped(model, batch_fn(alpha=1.0))).item())
        f_baseline = float(target_fn(_forward_unwrapped(model, batch_fn(alpha=0.0))).item())
    model.autocast = orig_ac
    delta = f_actual - f_baseline

    print(f"[r{rank}] {case.cid}: f(x)={f_actual:.6g} f(x')={f_baseline:.6g} "
          f"delta={delta:.6g} vars={atmos}", flush=True)

    for n_steps in STEP_COUNTS:
        t0 = time.time()
        res = integrated_gradients(
            model=model, batch_fn=batch_fn, target_fn=target_fn,
            atmos_actual=actual, atmos_baseline=baseline,
            atmos_var_names=tuple(atmos), device=DEV, n_steps=n_steps,
        )
        ig_sum = sum(float(np.asarray(res["ig"][v], dtype=np.float64).sum()) for v in atmos)
        ig_abs = sum(float(np.abs(np.asarray(res["ig"][v], dtype=np.float64)).sum()) for v in atmos)
        abs_gap = ig_sum - delta
        rel_gap = abs(abs_gap) / (abs(delta) + 1e-30)
        rel_energy = abs(abs_gap) / (ig_abs + 1e-30)
        dt = time.time() - t0
        rec = {
            "cid": case.cid, "timestamp": case.timestamp, "note": case.note,
            "target_var": case.target.var, "target_level": case.target.level,
            "target_mode": case.target.mode, "atmos_vars": atmos,
            "n_steps": n_steps, "f_actual": f_actual, "f_baseline": f_baseline,
            "delta": delta, "ig_sum": ig_sum, "ig_abs": ig_abs,
            "abs_gap": abs_gap, "rel_gap": rel_gap, "rel_energy": rel_energy,
            "runtime_s": dt,
        }
        writer.write(rec)
        print(f"[r{rank}] {case.cid} n={n_steps:>3} ig_sum={ig_sum:.6g} "
              f"abs_gap={abs_gap:.3g} rel_gap={rel_gap:.3%} rel_E={rel_energy:.3%} "
              f"({dt:.0f}s)", flush=True)
        torch.cuda.empty_cache()


def main():
    rank, world = rank_world()
    my_cases = shard(CASES, rank, world)
    print(f"[r{rank}/{world}] cases: {[c.cid for c in my_cases]}", flush=True)

    model = load_model(DEV)
    writer = JsonlWriter("completeness", rank)
    for case in my_cases:
        try:
            completeness_for_case(model, case, writer, rank)
        except Exception as e:
            print(f"[r{rank}] {case.cid} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            writer.write({"cid": case.cid, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()
    writer.close()
    print(f"[r{rank}] ########## COMPLETENESS SUITE DONE ##########", flush=True)


if __name__ == "__main__":
    main()
