#!/usr/bin/env python3
"""
Regenerate all derived outputs from per_case/ files on disk:
  - leaderboard.csv / leaderboard.json
  - all_results.csv / all_results.json
  - per-case scatter plots (*_scatter.png)

Usage:
    python generate_leaderboard.py <output_dir>

Safe to run at any time: mid-run, after a cancelled job, or after adding new
methods/cases. Plots are only regenerated for entries where the .npy files exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Leaderboard aggregation
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def collect_rows(output_dir: str) -> list[dict]:
    """Walk per_case/ and load every *_metrics.json into a flat row list."""
    rows = []
    per_case_root = os.path.join(output_dir, "per_case")
    if not os.path.isdir(per_case_root):
        raise FileNotFoundError(f"No per_case/ directory found in {output_dir}")

    for case_id in sorted(os.listdir(per_case_root)):
        case_dir = os.path.join(per_case_root, case_id)
        if not os.path.isdir(case_dir):
            continue
        for scale in sorted(os.listdir(case_dir)):
            scale_dir = os.path.join(case_dir, scale)
            if not os.path.isdir(scale_dir):
                continue
            for fname in sorted(os.listdir(scale_dir)):
                if not fname.endswith("_metrics.json"):
                    continue
                fpath = os.path.join(scale_dir, fname)
                with open(fpath) as f:
                    row = json.load(f)
                rows.append(row)

    return rows


def write_leaderboard(*, output_dir: str, rows: list[dict]) -> None:
    """Write all_results.{csv,json} and leaderboard.{csv,json} from row list."""
    if not rows:
        return
    os.makedirs(output_dir, exist_ok=True)

    norm_rows = []
    for row in rows:
        norm = dict(row)
        norm["target_mode"] = norm.get("target_mode") or "box"
        norm["base_case_id"] = norm.get("base_case_id") or norm.get("case_id", "")
        norm_rows.append(norm)
    rows = norm_rows

    fieldnames = [
        "method", "target_mode", "case_id", "base_case_id", "scale",
        "rho_mag", "rho_signed", "ndcg_at_10", "top10_recall",
        "remote_gap", "n_masks", "n_remote", "runtime_s",
    ]
    with open(os.path.join(output_dir, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(_to_jsonable(rows), f, indent=2)

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["method"], r["target_mode"])].append(r)

    leaderboard = []
    for (method, target_mode), items in agg.items():
        def _mean(key, _items=items):
            vals = [x[key] for x in _items
                    if isinstance(x.get(key), (int, float)) and np.isfinite(x[key])]
            return float(np.mean(vals)) if vals else float("nan")
        leaderboard.append({
            "method": method,
            "target_mode": target_mode,
            "mean_rho_mag":      _mean("rho_mag"),
            "mean_rho_signed":   _mean("rho_signed"),
            "mean_ndcg_at_10":   _mean("ndcg_at_10"),
            "mean_top10_recall": _mean("top10_recall"),
            "mean_remote_gap":   _mean("remote_gap"),
            "mean_runtime_s":    _mean("runtime_s"),
            "n_rows": len(items),
        })

    leaderboard.sort(key=lambda x: (
        x["target_mode"],
        -(x["mean_rho_mag"]     if np.isfinite(x["mean_rho_mag"])     else -1e9),
        -(x["mean_ndcg_at_10"]  if np.isfinite(x["mean_ndcg_at_10"])  else -1e9),
        -(x["mean_top10_recall"] if np.isfinite(x["mean_top10_recall"]) else -1e9),
        x["mean_runtime_s"] if np.isfinite(x["mean_runtime_s"]) else 1e18,
    ))

    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump(_to_jsonable(leaderboard), f, indent=2)

    lb_fields = [
        "method", "target_mode", "mean_rho_mag", "mean_rho_signed",
        "mean_ndcg_at_10", "mean_top10_recall",
        "mean_remote_gap", "mean_runtime_s", "n_rows",
    ]
    with open(os.path.join(output_dir, "leaderboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lb_fields)
        w.writeheader()
        for r in leaderboard:
            w.writerow({k: r.get(k, "") for k in lb_fields})


def regenerate_plots(output_dir: str) -> int:
    """Regenerate *_scatter.png for every (case, scale, method) that has .npy files.

    Returns the number of plots written.
    """
    from searchlight_report import plot_case_scatter

    per_case_root = os.path.join(output_dir, "per_case")
    n_plots = 0

    for case_id in sorted(os.listdir(per_case_root)):
        case_dir = os.path.join(per_case_root, case_id)
        if not os.path.isdir(case_dir):
            continue
        for scale in sorted(os.listdir(case_dir)):
            scale_dir = os.path.join(case_dir, scale)
            gt_path = os.path.join(scale_dir, "ground_truth.json")
            if not os.path.isfile(gt_path):
                continue

            with open(gt_path) as f:
                gt = json.load(f)

            G = np.array(gt["G"])
            # Reconstruct mask objects with just the .role attribute needed by the plot.
            masks = [types.SimpleNamespace(**m) for m in gt["masks"]]

            for fname in sorted(os.listdir(scale_dir)):
                if not fname.endswith("_pooled_Amag.npy"):
                    continue
                method = fname[: -len("_pooled_Amag.npy")]
                pooled_A_mag = np.load(os.path.join(scale_dir, fname))
                plot_case_scatter(
                    output_dir=output_dir,
                    case_id=case_id,
                    scale=scale,
                    method=method,
                    pooled_A_mag=pooled_A_mag,
                    G=G,
                    masks=masks,
                )
                n_plots += 1

    return n_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Benchmark output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip scatter plot regeneration")
    args = parser.parse_args()

    rows = collect_rows(args.output_dir)
    if not rows:
        print("No *_metrics.json files found — nothing to write.", file=sys.stderr)
        sys.exit(1)

    write_leaderboard(output_dir=args.output_dir, rows=rows)

    methods = sorted({r["method"] for r in rows})
    cases  = sorted({r["case_id"] for r in rows})
    scales = sorted({r["scale"]   for r in rows})
    print(f"Leaderboard generated from {len(rows)} rows.")
    print(f"  Methods : {methods}")
    print(f"  Cases   : {cases}")
    print(f"  Scales  : {scales}")
    print(f"Written to: {args.output_dir}/leaderboard.csv")

    if not args.no_plots:
        n = regenerate_plots(args.output_dir)
        print(f"Scatter plots regenerated: {n}")


if __name__ == "__main__":
    main()
