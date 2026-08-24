"""
Randomization diagnostic — 6 h ZWD forecast
=============================================
Adebayo-style cascading parameter randomization for saliency and IG.

Cascade order (decoder first, then encoder, outer to inner):
  decoder_layers[2] → decoder_layers[1] → decoder_layers[0]
  encoder_layers[2] → encoder_layers[1] → encoder_layers[0]

After each stage the model weights are re-randomized cumulatively and both
methods are re-run on the representative 6h Ticino case. Attribution maps are
compared to the original via full-map Pearson and Spearman.

Usage
-----
  python 01_attribution_method_validation/randomization.py
  # or via entry gate:
  python 01_attribution_method_validation/diagnostics.py randomization [--ig-steps 32]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import (
    import_benchmark, gpu_sync, pearson, spearman_flat,
    load_attr, save_json,
    OUTPUT_DIR, EVENT_SUITE_DIR,
    REP_CASE_TARGET, REP_CASE_INIT, REP_CASE_ID_POINT, REP_SCALE,
)

# Cascade order: decoder outer-to-inner, then encoder outer-to-inner
_CASCADE_STAGES = [
    ("decoder", 2), ("decoder", 1), ("decoder", 0),
    ("encoder", 2), ("encoder", 1), ("encoder", 0),
]


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
    print(f"[randomization] device={device}")

    out_dir = os.path.join(OUTPUT_DIR, "randomization")
    os.makedirs(out_dir, exist_ok=True)

    case_data = load_case(REP_CASE_INIT)
    target_fn_pt, _ = make_q850_target(case_data, REP_CASE_TARGET, "point")
    scale = SCALES[REP_SCALE]
    zwd_baseline_cpu = smoothed_zwd_baseline(case_data.surf_cpu["zwd"], scale.sigma_deg)

    orig_sal_6h = load_attr(EVENT_SUITE_DIR, REP_CASE_ID_POINT, REP_SCALE, "saliency")
    orig_ig_6h  = load_attr(EVENT_SUITE_DIR, REP_CASE_ID_POINT, REP_SCALE, "ig")

    model = setup_model(device)
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # --- 6h cascade ---
    cascade_rows = []
    randomized_so_far: list[tuple] = []

    for stage_kind, stage_idx in _CASCADE_STAGES:
        randomized_so_far.append((stage_kind, stage_idx))
        stage_label = f"{stage_kind}_{stage_idx}"
        print(f"\n  Stage: {stage_label}  "
              f"(cumulative: {[f'{k}_{i}' for k, i in randomized_so_far]})")

        model.load_state_dict(orig_state)
        for kind, idx in randomized_so_far:
            layer = (model.backbone.decoder_layers[idx]
                     if kind == "decoder"
                     else model.backbone.encoder_layers[idx])
            _randomize_layer(layer, seed=_stage_seed(kind, idx))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for method in ["saliency", "ig"]:
            t0 = time.time()
            attr = (
                run_saliency(model, case_data, target_fn_pt, device)
                if method == "saliency"
                else run_ig(model, case_data, target_fn_pt, device,
                            zwd_baseline_cpu=zwd_baseline_cpu,
                            n_steps=args.ig_steps)
            )
            elapsed = time.time() - t0

            orig = orig_sal_6h if method == "saliency" else orig_ig_6h
            row = {
                "stage": stage_label,
                "cumulative_stages": [f"{k}_{i}" for k, i in randomized_so_far],
                "method": method,
                "full_pearson": pearson(orig, attr),
                "full_spearman": spearman_flat(orig, attr),
                "runtime_s": elapsed,
            }
            cascade_rows.append(row)
            print(f"    {method}: Pearson={row['full_pearson']:.4f} "
                  f"Spearman={row['full_spearman']:.4f}  ({elapsed:.1f}s)")
            gpu_sync()

    # Restore before saving
    model.load_state_dict(orig_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    save_json(
        {"case": REP_CASE_ID_POINT, "scale": REP_SCALE,
         "lead_time_hours": 6, "cascade": cascade_rows},
        os.path.join(out_dir, "cascade_6h.json"),
    )
    _print_cascade_table(cascade_rows)

    print("\n[randomization] Done.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _stage_seed(kind: str, idx: int) -> int:
    base = 2000 if kind == "decoder" else 3000
    return base + idx


def _randomize_layer(layer, seed: int) -> None:
    generator = torch.Generator(device=next(layer.parameters()).device).manual_seed(seed)
    for p in layer.parameters():
        torch.nn.init.normal_(p.data, mean=0.0, std=0.02, generator=generator)


def _print_cascade_table(rows: list[dict]) -> None:
    print(f"\n{'stage':>15} {'method':>8} {'full_Pearson':>12} {'full_Spearman':>13}")
    for r in rows:
        print(f"{r['stage']:>15} {r['method']:>8} "
              f"{r['full_pearson']:>12.4f} {r['full_spearman']:>13.4f}")


# ------------------------------------------------------------------
# Standalone entry
# ------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ZWD randomization diagnostic")
    p.add_argument("--ig-steps",  type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    import time as _time
    from datetime import datetime as _dt
    _args = _parse_args()
    _t0 = _time.time()
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] randomization (standalone)")
    run(_args)
    _elapsed = _time.time() - _t0
    print(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] Done in {_elapsed:.0f}s ({_elapsed/60:.1f}m)")
