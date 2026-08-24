"""
Conditional ZWD Mechanism Ablations — main entry point.

Answers in sequence:
  1. Does ZWD add information beyond humidity?
  2. If so, what does it add (attribution, timing, routing)?

See README.md for full documentation and usage examples.
Module layout:
  conditional_data.py   — ZWD replacement builders, I/O helpers
  conditional_stages.py — Stage A/B/C/D implementations
  conditional_zwd_mechanism.py (this file) — CLI + orchestration
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from typing import Any

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_benchmark import make_q850_target, setup_model, TARGET_LEVEL_HPA  # noqa: E402
from searchlight_data import load_case, CaseData  # noqa: E402
from searchlight_tasks import Case, default_cases, load_cases_from_json, TARGETS  # noqa: E402

from conditional_data import (  # noqa: E402
    _ensure_dir, _write_json, _append_csv,
    build_all_zwd_replacements,
    load_zwd_reference_timestamps,
    log_qhat_fit_quality,
    make_q850_aboveground_target,
    make_qhat_zwd,
    replacement_diagnostics,
)
from conditional_stages import (  # noqa: E402
    run_random_null_stage,
    run_stage_a, run_stage_b, run_stage_c, run_stage_d,
    run_stage_e_localized_residual,
)

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_LEADS = [6, 12, 24]
DEFAULT_MODES = ["true", "qhat_tcwv"]
DEFAULT_TARGETS = ["ticino", "california", "japan"]
# Relative threshold: |delta_beyond_humidity| / |score_true| > this → run Stages B/C/D.
# 0.01 = 1 % of the q850 target score.
BEYOND_HUMIDITY_THRESHOLD_FRAC = 0.01
N_REF_MAX = 50

DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "zwd_counterfactual_interventions",
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Conditional ZWD Mechanism Ablations")
    p.add_argument("--cases", default="auto",
                   help="'auto' for built-in cases or path to cases.json")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--leads", type=int, nargs="+", default=DEFAULT_LEADS,
                   help="Forecast lead times in hours (multiples of 6).")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--target-mode", choices=("box", "point"), default="box")
    p.add_argument("--threshold-frac", type=float, default=BEYOND_HUMIDITY_THRESHOLD_FRAC,
                   help="Relative threshold: |delta_beyond_humidity|/|score_true| > this "
                        "to trigger Stages B/C/D. Default 0.01 = 1%% of q850 score.")
    p.add_argument("--no-stage-b", action="store_true")
    p.add_argument("--no-stage-c", action="store_true")
    p.add_argument("--no-stage-d", action="store_true")
    p.add_argument("--stage-b", action="store_false", dest="no_stage_b",
                   help="Enable the ancillary attribution-overlap stage.")
    p.add_argument("--stage-c", action="store_false", dest="no_stage_c",
                   help="Enable the ancillary timing stage.")
    p.add_argument("--stage-d", action="store_false", dest="no_stage_d",
                   help="Enable the ancillary hidden-state stage.")
    p.set_defaults(no_stage_b=True, no_stage_c=True, no_stage_d=True)
    p.add_argument("--stage-e-localized", action="store_true",
                   help="Run qhat + M*(true-qhat) localized residual masks.")
    p.add_argument("--localized-all", action="store_true",
                   help="Run Stage E for all cases, not only threshold-triggered cases.")
    p.add_argument("--n-random-donors", type=int, default=0,
                   help="Number of repeated random_same_month donors for the null distribution.")
    p.add_argument("--n-ref-fit", type=int, default=4,
                   help="Reference timestamps used for qhat_ref_month coefficient fitting.")
    p.add_argument("--n-ref-max", type=int, default=N_REF_MAX)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true",
                   help="Smoke test: 1 case, lead=6h, modes true/qhat/climatology.")
    return p.parse_args()


# ─── Summary report ───────────────────────────────────────────────────────────

def write_summary(out_dir: str, threshold: float) -> None:
    ablation_path = os.path.join(out_dir, "ablation_scores.csv")
    if not os.path.exists(ablation_path):
        return

    import csv as _csv
    with open(ablation_path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return

    lines = ["# Conditional ZWD Mechanism — Auto-Summary\n"]
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["case_id"], r["lead_h"])].append(r)

    def _get(group, field):
        for r in group:
            v = r.get(field)
            if v not in ("", None, "nan"):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return float("nan")

    for (case_id, lead_h), group in sorted(groups.items()):
        d_bh = _get(group, "delta_beyond_humidity")
        d_total = _get(group, "delta_total_zwd")
        d_resid = _get(group, "delta_residual")
        s_true = _get(group, "score_true")
        rel = abs(d_bh) / (abs(s_true) + 1e-10) if not math.isnan(d_bh) else float("nan")
        if math.isnan(d_bh):
            interp = "Incomplete data."
        elif rel > threshold:
            interp = (
                f"ZWD ADDS BEYOND HUMIDITY (|Δ|/score={rel*100:.1f}% > {threshold*100:.1f}%). "
                f"Total ZWD effect: {d_total:.4f}. Residual-only effect: {d_resid:.4f}."
            )
        else:
            interp = (
                f"ZWD likely redundant (|Δ|/score={rel*100:.1f}% ≤ {threshold*100:.1f}%). "
                "True and qhat ZWD produce similar forecasts."
            )
        lines.append(f"## {case_id}  lead={lead_h}h\n{interp}\n")

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary written to {summary_path}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _q850_level_index(case: CaseData) -> int:
    matches = np.where(np.asarray(case.pressure_levels) == TARGET_LEVEL_HPA)[0]
    if matches.size != 1:
        raise ValueError(f"No unique {TARGET_LEVEL_HPA} hPa level.")
    return int(matches[0])


def _any_beyond_humidity(ablation_rows: list[dict[str, Any]], threshold_frac: float) -> bool:
    """True if any row has |delta_beyond_humidity| / |score_true| > threshold_frac."""
    for r in ablation_rows:
        dbh = r.get("delta_beyond_humidity")
        st = r.get("score_true")
        if dbh in (None, "", "nan") or st in (None, "", "nan"):
            continue
        try:
            rel = abs(float(dbh)) / (abs(float(st)) + 1e-10)
            if rel > threshold_frac:
                return True
        except (ValueError, TypeError):
            pass
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Conditional ZWD Mechanism — rank {rank}/{world_size}")
        print(f"Output: {args.output_dir}")
        _ensure_dir(args.output_dir)

    cases = default_cases() if args.cases == "auto" else load_cases_from_json(args.cases)
    if args.targets:
        cases = [c for c in cases if c.target in args.targets]

    if args.debug:
        cases = cases[:1]
        args.leads = [6]
        args.modes = ["true", "qhat", "climatology"]
        print("DEBUG: 1 case, lead=6h, 3 modes.")

    my_cases = [c for i, c in enumerate(cases) if i % world_size == rank]
    if not my_cases:
        print(f"Rank {rank}: no cases assigned.")
        return
    print(f"Rank {rank}: {len(my_cases)} cases.")

    for lead_h in args.leads:
        if lead_h % 6 != 0 or lead_h < 6:
            raise ValueError(f"lead_h must be a positive multiple of 6, got {lead_h}")

    model = setup_model(device)
    all_diag: list[dict[str, Any]] = []

    for case_obj in my_cases:
        case_id = case_obj.case_id
        print(f"\n{'='*60}\nCase: {case_id}\n{'='*60}")

        case = load_case(case_obj.init_time)
        log_qhat_fit_quality(case, case_id)

        init = case_obj.init_time
        print(f"  Loading reference ZWD frames (month={init.month} hour={init.hour})…")
        ref_zwds, ref_times = load_zwd_reference_timestamps(
            month=init.month, hour=init.hour,
            exclude_year=init.year, n_max=args.n_ref_max,
        )
        print(f"  {len(ref_zwds)} reference frames loaded.")

        replacements = build_all_zwd_replacements(
            case, ref_zwds, rng, args.modes,
            case_obj=case_obj, ref_times=ref_times, n_ref_fit=args.n_ref_fit,
        )

        diag = replacement_diagnostics(case, replacements, ref_zwds, case_obj)
        all_diag.extend(diag)
        _append_csv(os.path.join(args.output_dir, "zwd_replacement_diagnostics.csv"), diag)

        target_fn, _ = make_q850_target(case, case_obj.target, args.target_mode)
        level_idx_850 = _q850_level_index(case)

        # Above-ground-only q850 as a second diagnostic off the same forward
        # pass (free). Over high terrain much of the box is below 850 hPa.
        target_fns = {"q850": target_fn}
        if args.target_mode == "box":
            ag_fn, ag_frac = make_q850_aboveground_target(
                case, case_obj.target, level_idx_850,
            )
            print(f"  [{case_id}] q850 box above-ground fraction: {ag_frac*100:.1f}%")
            if ag_fn is not None and ag_frac < 0.999:
                target_fns["q850_ag"] = ag_fn
            elif ag_fn is None:
                print(f"  [{case_id}] WARNING: box entirely below 850 hPa; "
                      f"the unmasked q850 score is not physically meaningful here.")

        # Stage A
        print(f"  Stage A: {len(args.leads)} leads × {len(args.modes)} modes…")
        ablation_rows = run_stage_a(
            model, case, case_obj, target_fns, device,
            replacements, args.leads, args.output_dir,
        )

        strong = _any_beyond_humidity(ablation_rows, args.threshold_frac)
        if not strong:
            print(f"  [{case_id}] Beyond-humidity delta below threshold "
                  f"({args.threshold_frac*100:.1f}% of score); skipping Stages B/C/D.")

        qhat_zwd = replacements.get("qhat")
        if qhat_zwd is None:
            qhat_zwd = make_qhat_zwd(case)

        if args.n_random_donors > 0:
            print(f"  Random null: {args.n_random_donors} same-month donors…")
            run_random_null_stage(
                model, case, case_obj, target_fn, device,
                ref_zwds, rng, args.leads, args.output_dir,
                n_random=args.n_random_donors, qhat_zwd=qhat_zwd,
            )

        if args.stage_e_localized and (strong or args.localized_all or args.debug):
            print("  Stage E: localized residual masks…")
            run_stage_e_localized_residual(
                model, case, case_obj, target_fn, device,
                qhat_zwd, args.leads, args.output_dir,
            )

        # Stage B
        if not args.no_stage_b and (strong or args.debug):
            print(f"  Stage B: attribution…")
            run_stage_b(
                model, case, case_obj, target_fn, device,
                replacements, level_idx_850, args.output_dir,
            )

        # Stage C
        if not args.no_stage_c and (strong or args.debug):
            print(f"  Stage C: timing…")
            run_stage_c(
                model, case, case_obj, target_fn, device,
                qhat_zwd, args.leads, args.output_dir,
            )

        # Stage D
        if not args.no_stage_d and strong:
            print(f"  Stage D: routing…")
            run_stage_d(model, case, case_obj, device, qhat_zwd, args.output_dir)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank == 0:
        _write_json(
            os.path.join(args.output_dir, "run_config.json"),
            {
                "cases": [c.case_id for c in cases],
                "targets": args.targets or DEFAULT_TARGETS,
                "leads": args.leads,
                "modes": args.modes,
                "target_mode": args.target_mode,
                "threshold_frac": args.threshold_frac,
                "stage_e_localized": args.stage_e_localized,
                "localized_all": args.localized_all,
                "n_random_donors": args.n_random_donors,
                "n_ref_fit": args.n_ref_fit,
                "n_ref_max": args.n_ref_max,
                "seed": args.seed,
                "world_size": world_size,
            },
        )
        write_summary(args.output_dir, args.threshold_frac)
        print("\nAll done.")


if __name__ == "__main__":
    main()
