"""
RISE mask-count convergence — 6 h ZWD forecasts
================================================
RISE is a Monte-Carlo estimator, so its maps are noisy at low mask counts.
Mirrors the adapter verification suite (geoxplain-aurora-adapter/diagnostics/
rise_convergence.py) on the ZWD-augmented model: one long RISE run per seed
(up to 1024 masks) is snapshotted at checkpoints {32,64,128,256,512,1024} via
running covariance sums, for two runs per case:
    cleanA   (seed 42)
    cleanB   (seed 1234)
and two curves vs mask count k:
    self   = pearson(cleanA_k, cleanA_1024)   does the estimate settle?
    repro  = pearson(cleanA_k, cleanB_k)      seed-to-seed agreement

Masking matches the shipped benchmark RISE path exactly: masks blend the t1
ZWD timestep toward the sigma-smoothed baseline (_make_zwd_masked_batch),
18x36 mask grid, p=0.5, covariance-centered normalization.

Cases: the three "strong" event cases (ticino/california/japan, point target,
6h lead), sharded across SLURM ranks (one case per rank, both seeds).

Usage
-----
  python 01_attribution_method_validation/rise_convergence.py
  # or via entry gate (single rank):
  python 01_attribution_method_validation/diagnostics.py \
      rise_convergence
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import (
    import_benchmark, gpu_sync, save_json,
    OUTPUT_DIR, EVENT_CASES, REP_SCALE,
)

RISE_P = 0.5
RISE_CELLS_H, RISE_CELLS_W = 18, 36
RISE_SEED_A, RISE_SEED_B = 42, 1234
DEFAULT_CHECKPOINTS = [32, 64, 128, 256, 512, 1024]

# The three strong event cases (target, init_time, rating="strong").
CONV_CASES = [(t, it, r) for t, it, r in EVENT_CASES if r == "strong"]


def _rise_partial_maps(
    *, model, case_data, target_fn, zwd_bl_cpu, seed,
    checkpoints, make_masked_batch, forward, gen_masks, normalize_cov,
    device, rank, label,
):
    """One RISE run to max(checkpoints) masks, returning {k: (H,W) map}."""
    H, W = case_data.surf_cpu["zwd"].shape[-2:]
    n_max = max(checkpoints)
    cps = set(checkpoints)

    sal = np.zeros((H, W)); msum = np.zeros((H, W))
    msq = np.zeros((H, W)); ssum = 0.0
    out = {}
    t0 = time.time()
    for i, mask in enumerate(gen_masks(
        n=n_max, cells_h=RISE_CELLS_H, cells_w=RISE_CELLS_W,
        H=H, W=W, p=RISE_P, seed=seed,
    )):
        batch = make_masked_batch(case_data, device, mask, zwd_bl_cpu)
        with torch.no_grad():
            val = float(target_fn(forward(model, batch)).item())
        del batch
        sal += val * mask; msum += mask; msq += mask * mask; ssum += val
        k = i + 1
        if k in cps:
            out[k] = normalize_cov(sal.copy(), msum.copy(), msq.copy(), ssum, k)
        if k % 128 == 0:
            torch.cuda.empty_cache()
            print(f"  [r{rank}] {label} seed={seed} {k}/{n_max} masks "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    if a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r)


def run(args) -> None:
    (
        setup_model, _run_saliency, _run_ig, make_q850_target, _forward,
        _saved_tensors_cpu_context, _RolloutForwardWrapper,
        load_case, make_batch,
        TARGETS, SCALES, generate_mask_centers, gaussian_mask,
        cos_lat_weights, smoothed_zwd_baseline,
        pool_attribution, spearman,
        xia_ig,
    ) = import_benchmark()
    from searchlight_benchmark import _make_zwd_masked_batch  # noqa: E402
    from xia_methods.rise import (  # noqa: E402
        generate_rise_masks, normalize_rise_covariance,
    )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = max(1, int(os.environ.get("SLURM_NTASKS", "1")))
    checkpoints = sorted(args.checkpoints)
    my_cases = [c for i, c in enumerate(CONV_CASES) if i % world == rank]
    print(f"[rise_convergence] r{rank}/{world} device={device} "
          f"checkpoints={checkpoints} cases={[c[0] for c in my_cases]}", flush=True)
    if not my_cases:
        print(f"[r{rank}] no cases for this rank, exiting.", flush=True)
        return

    out_dir = os.path.join(OUTPUT_DIR, "rise_convergence")
    os.makedirs(out_dir, exist_ok=True)

    model = setup_model(device)
    sigma_deg = SCALES[REP_SCALE].sigma_deg

    for target_name, init_time, rating in my_cases:
        case_id = f"{target_name}_{init_time.strftime('%Y%m%d%H')}_{rating}__point"
        case_data = load_case(init_time)
        target_fn, _ = make_q850_target(case_data, target_name, "point")
        zwd_bl_cpu = smoothed_zwd_baseline(case_data.surf_cpu["zwd"], sigma_deg)

        common_kw = dict(
            model=model, case_data=case_data, target_fn=target_fn,
            zwd_bl_cpu=zwd_bl_cpu, checkpoints=checkpoints,
            make_masked_batch=_make_zwd_masked_batch, forward=_forward,
            gen_masks=generate_rise_masks, normalize_cov=normalize_rise_covariance,
            device=device, rank=rank, label=case_id,
        )
        t0 = time.time()
        clean_a = _rise_partial_maps(seed=RISE_SEED_A, **common_kw)
        print(f"[r{rank}] {case_id} cleanA done ({time.time() - t0:.0f}s)", flush=True)
        clean_b = _rise_partial_maps(seed=RISE_SEED_B, **common_kw)
        print(f"[r{rank}] {case_id} cleanB done ({time.time() - t0:.0f}s)", flush=True)

        ref = clean_a[max(checkpoints)]
        rows = []
        for k in checkpoints:
            row = {
                "n_masks": k,
                "self_pearson": _pearson(clean_a[k], ref),
                "repro_pearson": _pearson(clean_a[k], clean_b[k]),
                "repro_spearman": _spearman(clean_a[k], clean_b[k]),
            }
            rows.append(row)
            print(f"[r{rank}] {case_id} k={k:>4} self={row['self_pearson']:.3f} "
                  f"repro={row['repro_pearson']:.3f}", flush=True)

        save_json(
            {"case": case_id, "scale": REP_SCALE, "lead_time_hours": 6,
             "cells": [RISE_CELLS_H, RISE_CELLS_W], "p": RISE_P,
             "seed_a": RISE_SEED_A, "seed_b": RISE_SEED_B,
             "runtime_s": time.time() - t0, "rows": rows},
            os.path.join(out_dir, f"rise_convergence_{case_id}.json"),
        )
        gpu_sync()

    print(f"\n[rise_convergence] r{rank} Done.", flush=True)


# ------------------------------------------------------------------
# Standalone entry
# ------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ZWD RISE mask-count convergence")
    p.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    return p.parse_args()


if __name__ == "__main__":
    import time as _time
    from datetime import datetime as _dt
    _args = _parse_args()
    _t0 = _time.time()
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] rise_convergence (standalone)")
    run(_args)
    _elapsed = _time.time() - _t0
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] Done in {_elapsed:.0f}s ({_elapsed/60:.1f}m)")
