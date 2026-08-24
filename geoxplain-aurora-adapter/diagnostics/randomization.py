"""Randomization sanity suite (Adebayo et al.): does attribution depend on the
trained weights?

Three checks per method (saliency, IG, RISE, ViT-CX) on the same cases:
  1. Cascading layer randomization, output head → input tokeniser, cumulative.
  2. Full-model randomization over 5 seeds (the appendix randomization table).
  3. Spatial-baseline calibration (i.i.d. noise / Gaussian blob / value-shuffle),
     which fixes the ~0 floor the randomized maps are compared against.

Method settings are held fixed across every run; defaults come from
``_suite_common`` and can be overridden per run with the RISE_MASKS / VIT_CLUSTERS
/ VIT_HOOK_STAGE env vars. Work is greedily load-balanced (``shard_balanced``)
so the chunky ViT-CX runs spread across GPUs.

Sharded across SLURM ranks; aggregated by aggregate.py.
"""

import time
import traceback

import numpy as np
import torch

from geoxplain_aurora_adapter.engine.model import load_model

from _suite_common import (
    CASES_BY_ID, JsonlWriter, rank_world, shard_balanced,
    run_attribution, similarity_record,
    snapshot_state, restore_state, build_cascade, randomize_params, gaussian_blob,
    env_csv, env_int, env_smooth_sigma,
    RISE_MASKS as RISE_MASKS_DEFAULT,
    VIT_CLUSTERS as VIT_CLUSTERS_DEFAULT,
    VIT_HOOK_STAGE as VIT_HOOK_STAGE_DEFAULT,
)

DEV = "cuda"
SANITY_CASE_IDS = env_csv("SF_SANITY_CASES", ["alps_w", "atl_s", "trop_s"])
ALL_METHODS = env_csv(
    "SF_METHODS",
    env_csv("DIAG_METHODS", ["saliency", "ig", "rise", "vit_cx"]),
)
CASCADE_SEEDS = [11, 23]
FULLRAND_SEEDS = [11, 23, 37, 51, 67]

IG_STEPS = 16
RISE_MASKS = env_int("RISE_MASKS", RISE_MASKS_DEFAULT)
VIT_CLUSTERS = env_int("VIT_CLUSTERS", VIT_CLUSTERS_DEFAULT)
VIT_HOOK_STAGE = env_int("VIT_HOOK_STAGE", VIT_HOOK_STAGE_DEFAULT)
VIT_SMOOTH_SIGMA = env_smooth_sigma("VIT_SMOOTH_SIGMA", 0)
PER_RUN_S = {"saliency": 7, "ig": 50, "rise": 1200, "vit_cx": 1800}


def _attr(method, case, model, var, case_data=None):
    return run_attribution(
        method, case, model, DEV, var,
        ig_steps=IG_STEPS, rise_masks=RISE_MASKS, vit_clusters=VIT_CLUSTERS,
        vit_hook_stage=VIT_HOOK_STAGE, vit_smooth_sigma=VIT_SMOOTH_SIGMA,
        case_data=case_data,
    )


def _safe(method, case, model, var, rank, tag, case_data=None):
    try:
        return _attr(method, case, model, var, case_data=case_data), None
    except Exception as e:
        print(f"[r{rank}] {tag} raised {type(e).__name__}: {e}", flush=True)
        return None, f"{type(e).__name__}: {e}"


def _emit(writer, base, sim, **extra):
    rec = dict(base); rec.update(sim); rec.update(extra); writer.write(rec)


def do_cascade(model, snap, case, method, stages, total, writer, rank):
    var = case.target.var
    restore_state(model, snap)
    ref, err = _safe(method, case, model, var, rank, f"{case.cid}/{method}/ref")
    if ref is None:
        writer.write({"kind": "cascade", "cid": case.cid, "method": method, "error": err}); return
    base = {"kind": "cascade", "cid": case.cid, "method": method, "target_var": var,
            "n_clusters": VIT_CLUSTERS if method == "vit_cx" else None,
            "hook_stage": VIT_HOOK_STAGE if method == "vit_cx" else None,
            "smooth_sigma": VIT_SMOOTH_SIGMA if method == "vit_cx" else None}
    for seed in CASCADE_SEEDS:
        for k, (stage_name, _) in enumerate(stages):
            preds = [p for _, p in stages[: k + 1]]
            cumul = lambda n, preds=preds: any(p(n) for p in preds)
            restore_state(model, snap)
            n_rand = randomize_params(model, cumul, seed=seed)
            t0 = time.time()
            cmp, err = _safe(method, case, model, var, rank, f"{case.cid}/{method}/{stage_name}")
            if cmp is None:
                _emit(writer, base, {"collapsed": True, "energy_ratio": float("nan")},
                      seed=seed, stage_idx=k, stage=stage_name,
                      frac_params=n_rand / total, error=err); continue
            sim = similarity_record(ref, cmp)
            _emit(writer, base, sim, seed=seed, stage_idx=k, stage=stage_name,
                  frac_params=n_rand / total, runtime_s=time.time() - t0)
            print(f"[r{rank}] cascade {case.cid}/{method} s={seed} st={k}:{stage_name} "
                  f"pearson={sim['pearson']:.3f} E={sim['energy_ratio']:.3f} "
                  f"coll={sim['collapsed']}", flush=True)
    restore_state(model, snap)


def do_fullrand(model, snap, case, method, total, writer, rank):
    var = case.target.var
    restore_state(model, snap)
    ref, err = _safe(method, case, model, var, rank, f"{case.cid}/{method}/ref")
    if ref is None:
        writer.write({"kind": "fullrand", "cid": case.cid, "method": method, "error": err}); return
    base = {"kind": "fullrand", "cid": case.cid, "method": method, "target_var": var,
            "n_clusters": VIT_CLUSTERS if method == "vit_cx" else None,
            "hook_stage": VIT_HOOK_STAGE if method == "vit_cx" else None,
            "smooth_sigma": VIT_SMOOTH_SIGMA if method == "vit_cx" else None}
    for seed in FULLRAND_SEEDS:
        restore_state(model, snap)
        n_rand = randomize_params(model, lambda n: True, seed=seed)
        t0 = time.time()
        cmp, err = _safe(method, case, model, var, rank, f"{case.cid}/{method}/full")
        if cmp is None:
            _emit(writer, base, {"collapsed": True, "energy_ratio": float("nan")},
                  seed=seed, frac_params=1.0, error=err); continue
        sim = similarity_record(ref, cmp)
        _emit(writer, base, sim, seed=seed, frac_params=n_rand / total, runtime_s=time.time() - t0)
        print(f"[r{rank}] fullrand {case.cid}/{method} s={seed} pearson={sim['pearson']:.3f} "
              f"E={sim['energy_ratio']:.3f} coll={sim['collapsed']}", flush=True)
    restore_state(model, snap)


def do_baselines(model, snap, case, writer, rank):
    var = case.target.var
    rng = np.random.default_rng(2024)
    for method in ["saliency", "ig"]:
        restore_state(model, snap)
        ref, err = _safe(method, case, model, var, rank, f"{case.cid}/{method}/ref")
        if ref is None:
            writer.write({"kind": "baseline", "cid": case.cid, "method": method, "error": err}); continue
        L = ref.shape[0]
        bmaps = {"iid_noise": rng.standard_normal(ref.shape),
                 "gaussian_blob": gaussian_blob(case, L),
                 "value_shuffled": rng.permutation(ref.ravel()).reshape(ref.shape)}
        base = {"kind": "baseline", "cid": case.cid, "method": method, "target_var": var}
        for name, m in bmaps.items():
            _emit(writer, base, similarity_record(ref, m), baseline=name)
    restore_state(model, snap)


def build_work():
    kinds = set(env_csv("SF_KINDS", ["baselines", "cascade", "fullrand"]))
    items, costs = [], []
    for cid in SANITY_CASE_IDS:
        if "baselines" in kinds:
            items.append(("baselines", cid, None)); costs.append(2 * (7 + 50))
        for m in ALL_METHODS:
            if "cascade" in kinds:
                items.append(("cascade", cid, m))
                costs.append((1 + len(CASCADE_SEEDS) * 5) * PER_RUN_S[m])
            if "fullrand" in kinds:
                items.append(("fullrand", cid, m))
                costs.append((1 + len(FULLRAND_SEEDS)) * PER_RUN_S[m])
    return items, costs


def main():
    rank, world = rank_world()
    items, costs = build_work()
    mine = shard_balanced(items, costs, rank, world)
    est = sum(c for it, c in zip(items, costs) if it in mine)
    print(f"[r{rank}/{world}] {len(mine)} items, est {est/60:.0f} min: {mine}", flush=True)

    model = load_model(DEV)
    snap = snapshot_state(model)
    total = sum(1 for _ in model.parameters())
    stages = build_cascade(model)

    writer = JsonlWriter("randomization", rank)
    for kind, cid, method in mine:
        case = CASES_BY_ID[cid]
        try:
            if kind == "baselines":
                do_baselines(model, snap, case, writer, rank)
            elif kind == "cascade":
                do_cascade(model, snap, case, method, stages, total, writer, rank)
            elif kind == "fullrand":
                do_fullrand(model, snap, case, method, total, writer, rank)
        except Exception as e:
            print(f"[r{rank}] {kind}/{cid}/{method} FAILED: {e}", flush=True)
            traceback.print_exc()
            writer.write({"kind": kind, "cid": cid, "method": method, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()
    writer.close()
    print(f"[r{rank}] ########## RANDOMIZATION DONE ##########", flush=True)


if __name__ == "__main__":
    main()
