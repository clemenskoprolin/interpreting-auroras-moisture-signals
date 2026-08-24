"""RISE mask-count convergence.

RISE is a Monte-Carlo estimator, so its maps are noisy at low mask counts. This
runner shows the variance reduction with more masks by snapshotting a single long
RISE run (up to 1024 masks) at checkpoints {32,64,128,256,512,1024} via running
covariance sums, for two runs per case:
    cleanA   (seed 42)
    cleanB   (seed 1234)
and tracking two curves vs mask count k:
    self   = sim(cleanA_k, cleanA_1024)   does the estimate settle?
    repro  = sim(cleanA_k, cleanB_k)      seed-to-seed agreement

Sharded across SLURM ranks; aggregated by aggregate.py.
"""

import time
import traceback

import numpy as np
import torch

from geoxplain_aurora_adapter.engine.model import load_model, forward as model_forward
from geoxplain_aurora_adapter.engine._common import (
    _smooth_tensor, _make_masked_batch, IG_BASELINE_SIGMA_DEG, INPUT_H, INPUT_W,
)
from geoxplain_aurora_adapter.engine.xia_methods.rise import (
    generate_rise_masks, normalize_rise_covariance,
)
from geoxplain_aurora_adapter.schema.targets import build_target_fn

from _suite_common import (
    CASES_BY_ID, JsonlWriter, rank_world, shard_balanced,
    load_case_cached, pearson, spearman,
    env_csv,
    RISE_CELLS_H, RISE_CELLS_W,
)

DEV = "cuda"
RISE_CASE_IDS = env_csv("RISE_CASES", ["alps_w", "trop_s", "atl_s"])
RISE_P = 0.5
RISE_CHECKPOINTS = [int(x) for x in env_csv(
    "RISE_CHECKPOINTS", ["32", "64", "128", "256", "512", "1024"])]
RISE_MAX_MASKS = max(RISE_CHECKPOINTS)
RISE_SEED_A, RISE_SEED_B = 42, 1234
PER_RUN_S = {"rise": 1200}


def rise_partial_maps(case, model, *, case_data, seed):
    """One RISE run to RISE_MAX_MASKS, returning {k: (H,W) map} at checkpoints."""
    var = case.target.var
    tfn = build_target_fn(case.target, case_data)
    baseline = _smooth_tensor(case_data.atmos_cpu[var], IG_BASELINE_SIGMA_DEG)

    sal = np.zeros((INPUT_H, INPUT_W)); msum = np.zeros((INPUT_H, INPUT_W))
    msq = np.zeros((INPUT_H, INPUT_W)); ssum = 0.0
    cps = set(RISE_CHECKPOINTS)
    out = {}
    for i, mask in enumerate(generate_rise_masks(
        n=RISE_MAX_MASKS, cells_h=RISE_CELLS_H, cells_w=RISE_CELLS_W,
        H=INPUT_H, W=INPUT_W, p=RISE_P, seed=seed,
    )):
        batch = _make_masked_batch(case_data, DEV, mask, var, baseline, "atmos")
        with torch.no_grad():
            val = float(tfn(model_forward(model, batch)).item())
        del batch
        sal += val * mask; msum += mask; msq += mask * mask; ssum += val
        k = i + 1
        if k in cps:
            out[k] = normalize_rise_covariance(sal.copy(), msum.copy(), msq.copy(), ssum, k)
        if k % 128 == 0:
            torch.cuda.empty_cache()
    return out


def do_rise_conv(case, model, writer, rank):
    cd = load_case_cached(case.timestamp)
    t0 = time.time()
    cleanA = rise_partial_maps(case, model, case_data=cd, seed=RISE_SEED_A)
    print(f"[r{rank}] rise_conv {case.cid} cleanA done ({time.time()-t0:.0f}s)", flush=True)
    cleanB = rise_partial_maps(case, model, case_data=cd, seed=RISE_SEED_B)
    print(f"[r{rank}] rise_conv {case.cid} cleanB done ({time.time()-t0:.0f}s)", flush=True)

    ref = cleanA[RISE_MAX_MASKS]
    for k in RISE_CHECKPOINTS:
        rec = {
            "kind": "rise_conv", "cid": case.cid, "n_masks": k,
            "target": f"{case.target.var}@{case.target.level}",
            "self_pearson": pearson(cleanA[k], ref),
            "repro_pearson": pearson(cleanA[k], cleanB[k]),
            "repro_spearman": spearman(cleanA[k], cleanB[k]),
        }
        writer.write(rec)
        print(f"[r{rank}] rise_conv {case.cid} k={k:>4} self={rec['self_pearson']:.3f} "
              f"repro={rec['repro_pearson']:.3f}", flush=True)


def main():
    rank, world = rank_world()
    items = [("rise_conv", cid) for cid in RISE_CASE_IDS]
    costs = [2 * PER_RUN_S["rise"]] * len(items)
    mine = shard_balanced(items, costs, rank, world)
    print(f"[r{rank}/{world}] {len(mine)} items: {mine}", flush=True)

    model = load_model(DEV)
    writer = JsonlWriter("rise_convergence", rank)
    for kind, cid in mine:
        case = CASES_BY_ID[cid]
        try:
            do_rise_conv(case, model, writer, rank)
        except Exception as e:
            print(f"[r{rank}] {kind}/{cid} FAILED: {e}", flush=True)
            traceback.print_exc()
            writer.write({"kind": kind, "cid": cid, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()
    writer.close()
    print(f"[r{rank}] ########## RISE_CONVERGENCE DONE ##########", flush=True)


if __name__ == "__main__":
    main()
