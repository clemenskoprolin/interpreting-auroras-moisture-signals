"""Aggregate per-rank JSONL from the verification suites into Markdown tables.

Standalone: imports only the standard library + numpy, so it runs on the login
node without touching Aurora / torch.

Takes a suite name as its single argument and writes out/<suite>_summary.md
(also echoed to stdout).
"""

import glob
import json
import os
import re
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def load(suite):
    rows = []
    rank_file = re.compile(rf"^{re.escape(suite)}_rank[0-9]+\.jsonl$")
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{suite}_rank*.jsonl"))):
        if not rank_file.match(os.path.basename(path)):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _ms(vals):
    """nan-aware mean±std as a compact string; '—' if nothing finite."""
    a = np.array([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "—"
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f}±{a.std():.3f}"


# ── completeness ──────────────────────────────────────────────────────────────

def agg_completeness(rows):
    rows = [r for r in rows if "error" not in r]
    cids = sorted({r["cid"] for r in rows})
    out = ["# IG completeness — multi-case suite\n"]
    out.append("Baseline: Gaussian-smoothed actual field, σ=2.5° (reflect in lat, "
               "wrap in lon); midpoint Riemann rule. "
               "`rel_gap = |ΣIG−Δ| / |Δ|` with Δ=f(x)−f(x'); "
               "`rel_E = |ΣIG−Δ| / Σ|IG|` (robust when Δ≈0).\n")
    worst64 = []
    for cid in cids:
        cr = [r for r in rows if r["cid"] == cid]
        meta = cr[0]
        out.append(f"\n## {cid} — {meta.get('note','')}  "
                   f"(`{meta['target_var']}@{meta['target_level']}` {meta['target_mode']}, "
                   f"vars={meta['atmos_vars']}, {meta['timestamp']})")
        out.append(f"Δ = f(x)−f(x') = {meta['delta']:.6g}\n")
        out.append("| n_steps | ΣIG | abs_gap | rel_gap | rel_E | t(s) |")
        out.append("|--:|--:|--:|--:|--:|--:|")
        for r in sorted(cr, key=lambda r: r["n_steps"]):
            out.append(f"| {r['n_steps']} | {r['ig_sum']:.6g} | {r['abs_gap']:.3g} | "
                       f"{r['rel_gap']:.3%} | {r['rel_energy']:.3%} | {r.get('runtime_s',0):.0f} |")
            if r["n_steps"] == 64:
                worst64.append((cid, r["rel_gap"], r["rel_energy"], abs(r["abs_gap"])))
    out.append("\n## Summary (n_steps = 64)\n")
    out.append("| case | rel_gap | rel_E | |abs_gap| |")
    out.append("|--|--:|--:|--:|")
    for cid, rg, re_, ag in sorted(worst64, key=lambda x: -x[1]):
        out.append(f"| {cid} | {rg:.3%} | {re_:.3%} | {ag:.3g} |")
    if worst64:
        mx = max(worst64, key=lambda x: x[1])
        out.append(f"\nWorst-case relative residual at 64 steps: **{mx[1]:.2%}** "
                   f"(case `{mx[0]}`). Cases with tiny Δ are read via `rel_E` instead.")
    return "\n".join(out)


# ── sanity ────────────────────────────────────────────────────────────────────

_METRICS = ["pearson", "spearman", "cosine", "ssim", "top1", "top5", "energy_ratio"]


def _collapse_frac(group):
    flags = [bool(r.get("collapsed", False)) for r in group]
    return sum(flags) / len(flags) if flags else float("nan")


def agg_sanity(rows):
    out = ["# Sanity / randomization — multi-case, multi-seed suite\n"]
    out.append("Similarity of the randomized-model attribution to the trained-model "
               "attribution. A method passes when similarity DECAYS to the spatial-baseline "
               "level as the network is randomized. `collapse` = fraction of runs whose "
               "map went (near-)constant — correlations are NaN there by construction and "
               "`energy_ratio`=std_rand/std_ref→0 is the collapse evidence.\n")

    # --- cascading randomization
    casc = [r for r in rows if r.get("kind") == "cascade" and "stage" in r]
    methods = sorted({r["method"] for r in casc})
    if casc:
        out.append("\n## 1. Cascading layer randomization (output→input, cumulative)\n")
        out.append("Aggregated over seeds × cases. Stage 0 = only the Perceiver decoder "
                   "randomized; last stage = whole network randomized.\n")
        for m in methods:
            mr = [r for r in casc if r["method"] == m]
            stages = sorted({(r["stage_idx"], r["stage"]) for r in mr})
            out.append(f"\n### {m}")
            out.append("| stage | params | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |")
            out.append("|--|--:|--:|--:|--:|--:|--:|--:|--:|")
            for idx, name in stages:
                g = [r for r in mr if r["stage_idx"] == idx]
                fp = _ms([r.get("frac_params") for r in g])
                cells = " | ".join(_ms([r.get(k) for r in g]) for k in
                                   ["pearson", "spearman", "cosine", "ssim", "top1", "energy_ratio"])
                out.append(f"| {idx}:{name} | {fp} | {cells} | {_collapse_frac(g):.0%} |")

    # --- full randomization
    full = [r for r in rows if r.get("kind") == "fullrand" and "energy_ratio" in r]
    if full:
        out.append("\n## 2. Full randomization — all params, multiple seeds (mean±std over seeds×cases)\n")
        out.append("| method | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |")
        out.append("|--|--:|--:|--:|--:|--:|--:|--:|")
        for m in sorted({r["method"] for r in full}):
            g = [r for r in full if r["method"] == m]
            cells = " | ".join(_ms([r.get(k) for r in g]) for k in
                               ["pearson", "spearman", "cosine", "ssim", "top1", "energy_ratio"])
            out.append(f"| {m} | {cells} | {_collapse_frac(g):.0%} |")

    # --- spatial baselines
    bl = [r for r in rows if r.get("kind") == "baseline" and "energy_ratio" in r]
    if bl:
        out.append("\n## 3. Spatial-baseline calibration (trained attribution vs naive maps)\n")
        out.append("What a similarity value *means*: a randomized-model score at or below "
                   "the `iid_noise` row is genuine collapse.\n")
        out.append("| case | method | baseline | pearson | spearman | cosine | ssim | top1% |")
        out.append("|--|--|--|--:|--:|--:|--:|--:|")
        for r in sorted(bl, key=lambda r: (r["cid"], r["method"], r.get("baseline", ""))):
            vals = " | ".join(f"{r.get(k):.3f}" if r.get(k) is not None and np.isfinite(r.get(k, np.nan))
                              else "—" for k in ["pearson", "spearman", "cosine", "ssim", "top1"])
            out.append(f"| {r['cid']} | {r['method']} | {r.get('baseline')} | {vals} |")

    errs = [r for r in rows if "error" in r]
    if errs:
        out.append("\n## Errors\n")
        for r in errs:
            out.append(f"- {r.get('kind')}/{r.get('cid')}/{r.get('method')}: {r['error']}")
    return "\n".join(out)


def agg_rise_convergence(rows):
    out = ["# RISE mask-count convergence\n"]
    rc = [r for r in rows if r.get("kind") == "rise_conv"]
    if rc:
        max_masks = max(r.get("n_masks", 0) for r in rc)
        out.append(f"`self` = sim(cleanA_k, cleanA_{max_masks}) (MC convergence); "
                   "`repro` = sim(cleanA_k, cleanB_k) (seed agreement). Both rising "
                   "toward 1.0 with more masks is the expected variance reduction.\n")
        for cid in sorted({r["cid"] for r in rc}):
            cr = sorted([r for r in rc if r["cid"] == cid], key=lambda r: r["n_masks"])
            tgt = cr[0].get("target", "")
            out.append(f"\n### {cid} ({tgt})")
            out.append("| n_masks | self_pearson | repro_pearson |")
            out.append("|--:|--:|--:|")
            for r in cr:
                out.append(f"| {r['n_masks']} | {r['self_pearson']:.3f} | "
                           f"{r['repro_pearson']:.3f} |")
    errs = [r for r in rows if "error" in r]
    if errs:
        out.append("\n## Errors\n")
        for r in errs:
            out.append(f"- {r.get('kind')}/{r.get('cid')}: {r['error']}")
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate.py "
              "{completeness|randomization|rise_convergence}")
        sys.exit(1)
    suite = sys.argv[1]
    rows = load(suite)
    if not rows:
        print(f"No records found for suite '{suite}' in {OUT_DIR}")
        sys.exit(1)
    if "completeness" in suite:
        md = agg_completeness(rows)
    elif "rise_convergence" in suite:
        md = agg_rise_convergence(rows)
    else:
        md = agg_sanity(rows)
    summary_path = os.path.join(OUT_DIR, f"{suite}_summary.md")
    with open(summary_path, "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\n[aggregate] wrote {summary_path}")


if __name__ == "__main__":
    main()
