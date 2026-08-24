"""
Stability diagnostic — 6 h ZWD forecast
========================================
Re-runs saliency and IG three times on the representative Ticino case with
5% Gaussian noise added to ZWD t1, then compares each noised map against the
original saved attribution from searchlight_allmethods_eventdates_point.

Metrics reported per (seed, method):
  full_pearson / full_spearman   — full-map correlation with original
  pool_spearman_mag              — pooled-mask (near only) Spearman of |attr|
  top10_mask_overlap             — fraction of top-10 pooled masks that match
  delta_rho_mag / delta_rho_sign — change in benchmark score vs GT

Usage
-----
  python 01_attribution_method_validation/stability.py
  # or via entry gate:
  python 01_attribution_method_validation/diagnostics.py \
      stability [--seeds 11 22 33] [--ig-steps 32]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import (
    import_benchmark, gpu_sync, pearson, spearman_flat,
    load_attr, load_gt_json, save_json,
    OUTPUT_DIR, EVENT_SUITE_DIR,
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
    print(f"[stability] device={device}")

    out_dir = os.path.join(OUTPUT_DIR, "stability")
    os.makedirs(out_dir, exist_ok=True)

    # --- Reference data ---
    case_data = load_case(REP_CASE_INIT)
    target_fn, _ = make_q850_target(case_data, REP_CASE_TARGET, "point")
    lat_vals = case_data.lat_vals
    cos_w = cos_lat_weights(lat_vals, 1440)

    scale = SCALES[REP_SCALE]
    target = TARGETS[REP_CASE_TARGET]
    masks = generate_mask_centers(target, scale)
    mask_arrays = [gaussian_mask(m, scale.sigma_deg, lat_vals, case_data.lon_vals)
                   for m in masks]
    near_idx = np.array([i for i, m in enumerate(masks) if m.role == "near"])

    # Ground truth is optional; when ground_truth.json is unavailable, the
    # delta-rho-versus-ground-truth metrics are reported as non-applicable.
    try:
        gt = load_gt_json(EVENT_SUITE_DIR, REP_CASE_ID_POINT, REP_SCALE)
        G = np.array(gt["G"], dtype=np.float64)
        S = np.array(gt["S"], dtype=np.float64)
    except (FileNotFoundError, OSError):
        G = S = None
        print("  WARNING: ground_truth.json missing — Δρ-vs-GT metrics skipped")

    zwd_actual = case_data.surf_cpu["zwd"]   # (1, 2, H, W)
    noise_sigma = 0.05 * float(zwd_actual[0, 1].std().item())
    print(f"  ZWD t1 std={zwd_actual[0,1].std():.4f}  noise_sigma={noise_sigma:.4f}")

    zwd_baseline_cpu = smoothed_zwd_baseline(zwd_actual, scale.sigma_deg)
    model = setup_model(device)

    # Reference attributions: prefer the saved event-suite maps; if the scratch
    # cleanup removed them, recompute a clean (unnoised) reference in-run with
    # the current runners, so noised maps are compared against the same code
    # path that produced them.
    methods = list(args.methods)
    refs: dict[str, np.ndarray] = {}
    ref_source: dict[str, str] = {}
    for method in methods:
        try:
            refs[method] = load_attr(EVENT_SUITE_DIR, REP_CASE_ID_POINT, REP_SCALE, method)
            ref_source[method] = "saved"
            print(f"  reference for {method}: saved event-suite map")
        except (FileNotFoundError, OSError):
            print(f"  reference for {method}: saved map missing — recomputing clean")
            t0 = time.time()
            refs[method] = _run_method(
                method, model=model, case_data=case_data, target_fn=target_fn,
                device=device, zwd_cpu=zwd_actual, ig_steps=args.ig_steps,
                make_batch=make_batch, xia_ig=xia_ig,
                smoothed_zwd_baseline=smoothed_zwd_baseline,
                sigma_deg=scale.sigma_deg,
            )
            ref_source[method] = "recomputed_clean"
            print(f"      {time.time() - t0:.1f}s")
            gpu_sync()

    ref_pools = {m: pool_attribution(refs[m], mask_arrays, cos_w) for m in methods}
    rows = []

    for seed in args.seeds:
        print(f"\n  Seed {seed}")
        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(zwd_actual.shape, generator=rng, dtype=zwd_actual.dtype) * noise_sigma
        zwd_noised = zwd_actual.clone()
        zwd_noised[0, 1] = zwd_noised[0, 1] + noise[0, 1]

        for method in methods:
            print(f"    method={method}")
            t0 = time.time()
            attr_noised = _run_method(
                method, model=model, case_data=case_data, target_fn=target_fn,
                device=device, zwd_cpu=zwd_noised, ig_steps=args.ig_steps,
                make_batch=make_batch, xia_ig=xia_ig,
                smoothed_zwd_baseline=smoothed_zwd_baseline,
                sigma_deg=scale.sigma_deg,
            )
            elapsed = time.time() - t0
            print(f"      {elapsed:.1f}s")

            orig = refs[method]
            orig_A_mag, orig_A_sign = ref_pools[method]

            noised_A_mag, noised_A_sign = pool_attribution(attr_noised, mask_arrays, cos_w)

            full_pearson  = pearson(orig, attr_noised)
            full_spearman = spearman_flat(orig, attr_noised)

            o_pm = orig_A_mag[near_idx];  n_pm = noised_A_mag[near_idx]
            o_ps = orig_A_sign[near_idx]; n_ps = noised_A_sign[near_idx]

            pool_pearson_mag  = pearson(o_pm, n_pm)
            pool_pearson_sign = pearson(o_ps, n_ps)
            pool_spearman_mag  = float(spearman(o_pm, n_pm))
            pool_spearman_sign = float(spearman(o_ps, n_ps))
            top10_masks = _top_mask_overlap(o_pm, n_pm, k=10)

            if G is not None:
                G_near = G[near_idx]; S_near = S[near_idx]
                rho_mag_orig   = float(spearman(orig_A_mag[near_idx],   G_near))
                rho_mag_noised = float(spearman(noised_A_mag[near_idx], G_near))
                rho_sign_orig  = float(spearman(orig_A_sign[near_idx],  S_near))
                rho_sign_noised = float(spearman(noised_A_sign[near_idx], S_near))
            else:
                rho_mag_orig = rho_mag_noised = float("nan")
                rho_sign_orig = rho_sign_noised = float("nan")

            rows.append({
                "seed": seed,
                "method": method,
                "ref_source": ref_source[method],
                "full_pearson": full_pearson,
                "full_spearman": full_spearman,
                "pool_pearson_mag": pool_pearson_mag,
                "pool_spearman_mag": pool_spearman_mag,
                "pool_pearson_sign": pool_pearson_sign,
                "pool_spearman_sign": pool_spearman_sign,
                "top10_mask_overlap": top10_masks,
                "rho_mag_orig": rho_mag_orig,
                "rho_mag_noised": rho_mag_noised,
                "delta_rho_mag": rho_mag_noised - rho_mag_orig,
                "rho_sign_orig": rho_sign_orig,
                "rho_sign_noised": rho_sign_noised,
                "delta_rho_sign": rho_sign_noised - rho_sign_orig,
                "noise_sigma": noise_sigma,
                "noise_frac": 0.05,
                "runtime_s": elapsed,
            })
            print(
                f"      full_P={full_pearson:.4f} full_S={full_spearman:.4f} "
                f"pool_S_mag={pool_spearman_mag:.4f} top10={top10_masks:.3f} "
                f"Δrho_mag={rho_mag_noised - rho_mag_orig:+.4f}"
            )
            gpu_sync()

    save_json(
        {"case": REP_CASE_ID_POINT, "scale": REP_SCALE,
         "ref_sources": ref_source, "gt_available": G is not None, "rows": rows},
        os.path.join(out_dir, "stability_results.json"),
    )
    _print_table(rows)
    print("\n[stability] Done.")


# ------------------------------------------------------------------
# Attribution runners for (possibly noised) ZWD
# ------------------------------------------------------------------

def _run_method(
    method, *, model, case_data, target_fn, device, zwd_cpu,
    ig_steps, make_batch, xia_ig, smoothed_zwd_baseline, sigma_deg,
):
    """Run one attribution method on the given ZWD field (actual or noised)."""
    if method == "saliency":
        return _run_saliency_noised(
            model=model, case_data=case_data, target_fn=target_fn,
            device=device, zwd_noised_cpu=zwd_cpu, make_batch=make_batch,
        )
    if method == "ig":
        return _run_ig_noised(
            model=model, case_data=case_data, target_fn=target_fn,
            device=device, zwd_noised_cpu=zwd_cpu, ig_steps=ig_steps,
            make_batch=make_batch, xia_ig=xia_ig,
            smoothed_zwd_baseline=smoothed_zwd_baseline, sigma_deg=sigma_deg,
        )
    raise ValueError(f"unsupported stability method: {method}")


def _run_saliency_noised(*, model, case_data, target_fn, device, zwd_noised_cpu, make_batch):
    from xia_methods.saliency import saliency as _xia_sal

    def batch_fn(requires_grad=False):
        return make_batch(
            case_data, device,
            requires_grad_surf=("zwd",) if requires_grad else (),
            zwd_override=zwd_noised_cpu,
        )

    result = _xia_sal(
        model=model, batch_fn=batch_fn, target_fn=target_fn,
        atmos_var_names=(), surf_var_names=("zwd",), device=device,
    )
    return result["grads"]["zwd"][0, 1].astype(np.float32)  # t1 slice


def _run_ig_noised(
    *, model, case_data, target_fn, device,
    zwd_noised_cpu, ig_steps, make_batch, xia_ig, smoothed_zwd_baseline, sigma_deg,
):
    zwd_baseline_cpu = smoothed_zwd_baseline(zwd_noised_cpu, sigma_deg)
    zwd_delta_cpu    = zwd_noised_cpu - zwd_baseline_cpu

    def batch_fn(alpha, requires_grad=False):
        zwd_interp = (zwd_baseline_cpu + alpha * zwd_delta_cpu).clone()
        return make_batch(
            case_data, device,
            requires_grad_surf=("zwd",) if requires_grad else (),
            zwd_override=zwd_interp,
        )

    result = xia_ig(
        model=model, batch_fn=batch_fn, target_fn=target_fn,
        surf_actual={"zwd": zwd_noised_cpu},
        surf_baseline={"zwd": zwd_baseline_cpu},
        surf_var_names=("zwd",), device=device, n_steps=ig_steps,
    )
    return result["ig"]["zwd"][0, 1].astype(np.float32)  # t1 slice


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _top_mask_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    a = np.abs(np.asarray(a, dtype=np.float64))
    b = np.abs(np.asarray(b, dtype=np.float64))
    k = min(k, len(a))
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return len(top_a & top_b) / k


def _print_table(rows: list[dict]) -> None:
    print(
        f"\n{'seed':>4} {'method':>8} {'full_P':>7} {'full_S':>7} "
        f"{'pool_S_mag':>10} {'top10':>6} {'Δrho_mag':>9} {'Δrho_sgn':>9}"
    )
    for r in rows:
        print(
            f"{r['seed']:>4} {r['method']:>8} {r['full_pearson']:>7.4f} "
            f"{r['full_spearman']:>7.4f} {r['pool_spearman_mag']:>10.4f} "
            f"{r['top10_mask_overlap']:>6.3f} {r['delta_rho_mag']:>+9.4f} "
            f"{r['delta_rho_sign']:>+9.4f}"
        )


# ------------------------------------------------------------------
# Standalone entry
# ------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ZWD stability diagnostic")
    p.add_argument("--seeds",    type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--methods",  type=str, nargs="+", default=["saliency", "ig"],
                   choices=["saliency", "ig"])
    return p.parse_args()


if __name__ == "__main__":
    import time as _time
    from datetime import datetime as _dt
    _args = _parse_args()
    _t0 = _time.time()
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] stability (standalone)")
    run(_args)
    _elapsed = _time.time() - _t0
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] Done in {_elapsed:.0f}s ({_elapsed/60:.1f}m)")
