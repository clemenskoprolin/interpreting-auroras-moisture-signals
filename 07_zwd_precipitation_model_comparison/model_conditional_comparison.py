"""
model_conditional_comparison.py — Main CLI entry point for 07_zwd_precipitation_model_comparison.

The publication default runs the final 22-case Stage-B saliency comparison.
Ancillary stage implementations remain callable because the routing analysis
reuses their shared model, target, and intervention machinery.

IMPORTANT: This script imports Aurora and must run inside a GPU allocation.

Usage:
    python 07_zwd_precipitation_model_comparison/model_conditional_comparison.py \\
        --models precip_large_zwd \\
        --cases 07_zwd_precipitation_model_comparison/cases_diagnostic_22.json \\
        --outputs q850 --leads 6 --stages B \\
        --output-dir results/model_conditional_comparison

    # Debug mode (first final-cohort case, Stage B):
    python model_conditional_comparison.py --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_P_CORR_DIR = os.path.join(_ROOT, "06_precipitation_moisture_relationships")

for _p in (_HERE, _SEARCHLIGHT_DIR, _P_CORR_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from comparison_config import (  # noqa: E402
    DEFAULT_MODELS, DEFAULT_ROLLOUT, DEFAULT_TARGETS, DEFAULT_DOSES_MM,
    ModelSpec, TargetSpec, RolloutSpec,
)
from comparison_data import (  # noqa: E402
    _ensure_dir, _write_json, _append_csv,
    load_case, inject_precip_into_case,
)
from comparison_models import load_model, _gpu_sync_and_gc  # noqa: E402
from stages import (  # noqa: E402
    run_stage_a1_trajectories,
    run_stage_a2_interventions,
    run_stage_a3_conditional_diff,
    run_stage_b, run_stage_c, run_stage_d, run_stage_e,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="07_zwd_precipitation_model_comparison: Aurora precip_zwd vs precip_only"
    )
    p.add_argument(
        "--models", nargs="+", default=["precip_large_zwd"],
        choices=["precip_zwd", "precip_only",
                 "precip_large_zwd", "precip_large_only", "all"],
        help="Which models to load and compare",
    )
    p.add_argument(
        "--cases", type=str, default=os.path.join(_HERE, "cases_diagnostic_22.json"),
        help="'auto' = built-in default (ticino 2020-01-01), or path to cases JSON",
    )
    p.add_argument(
        "--targets", nargs="+", default=None,
        help="Subset of target region names (e.g. ticino california japan). "
             "Default: all targets found in cases file",
    )
    p.add_argument(
        "--outputs", nargs="+", default=["q850"],
        help="Output variables to extract. Each must match a TargetSpec name. "
             "Available: precip, q850",
    )
    p.add_argument(
        "--leads", nargs="+", type=int, default=[6],
        help="Lead times in hours (multiples of 6)",
    )
    p.add_argument(
        "--rollout-mode", choices=["independent"], default="independent",
        help="Rollout mode (currently only 'independent' is supported)",
    )
    p.add_argument(
        "--stages", nargs="+", default=["B"],
        choices=["A", "A1", "A2", "A3", "B", "C", "D", "E"],
        help="Stages to run. 'A' expands to A1/A2/A3; the thesis default is B.",
    )
    p.add_argument(
        "--doses-mm", nargs="+", type=float, default=list(DEFAULT_DOSES_MM),
        help="Precipitation dose magnitudes for intervention stage",
    )
    p.add_argument(
        "--diagnostic-cases-per-target", type=int, default=1,
        help="Number of cases per target for detailed stages B–E (gated by |M|)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: $AURORA_XAI_RESULTS_DIR/"
             "model_conditional_comparison).",
    )
    p.add_argument(
        "--mswep-store", type=str, default=None,
        help="Path to MSWEP zarr store (default from comparison_config.py)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Debug mode: first final-cohort case, lead=6h, Stage B only",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cases(args) -> list[dict]:
    """Load case list from CLI args."""
    from datetime import datetime

    if args.cases == "auto":
        # Built-in default: ticino 2020-01-01
        return [{
            "target": "ticino",
            "init_time": "2020-01-01T22:00:00",
            "role": "strong",
            "score": float("nan"),
        }]
    elif os.path.exists(args.cases):
        with open(args.cases) as f:
            cases = json.load(f)
        return cases
    else:
        raise FileNotFoundError(f"Cases file not found: {args.cases}")


def _load_case_data(cases: list[dict], mswep_store_path: str | None) -> dict[str, object]:
    """Load CaseData (ERA5 + ZWD) for each unique init_time.

    Also attempts to load MSWEP precipitation and inject it into each CaseData.
    """
    from datetime import datetime

    init_times = sorted(set(c["init_time"] for c in cases))
    case_data_map: dict[str, object] = {}

    from common import MSWEPReader, DEFAULT_PRECIP_PATH
    store_path = mswep_store_path or DEFAULT_PRECIP_PATH
    if not os.path.isdir(store_path):
        raise FileNotFoundError(
            f"MSWEP store not found: {store_path}. Set AURORA_MSWEP_DATA or "
            "pass --mswep-store; zero precipitation is not a valid fallback."
        )
    mswep_reader = MSWEPReader(store_path)
    print(f"MSWEP store opened: {store_path}")

    for init_time_str in init_times:
        print(f"  Loading case data for {init_time_str} ...")
        init_time = datetime.fromisoformat(init_time_str)
        case_data = load_case(init_time)

        # Inject MSWEP precipitation
        from comparison_data import load_precip_for_case
        precip = load_precip_for_case(init_time, mswep_reader)
        t0_arr = precip["t0"]
        t1_arr = precip["t1"]
        if t0_arr is None or t1_arr is None:
            raise RuntimeError(f"MSWEP t0/t1 missing for {init_time_str}")
        H_era5 = case_data.lat_vals.shape[0]

        # MSWEP has 720 rows and ERA5 721; repeat the south-pole row.
        def _pad_to_era5(arr, target_h):
            if arr.shape[0] == target_h:
                return arr
            rows_needed = target_h - arr.shape[0]
            if rows_needed != 1:
                raise ValueError(f"Cannot align MSWEP height {arr.shape[0]} to {target_h}")
            return np.concatenate([arr, np.repeat(arr[-1:], rows_needed, axis=0)], axis=0)

        t0_arr = _pad_to_era5(t0_arr, H_era5)
        t1_arr = _pad_to_era5(t1_arr, H_era5)
        case_data = inject_precip_into_case(case_data, t0_arr, t1_arr)
        print(f"    Injected MSWEP precipitation for {init_time_str}")

        case_data_map[init_time_str] = case_data

    return case_data_map


# ---------------------------------------------------------------------------
# Diagnostic case selection (gate for stages B–E)
# ---------------------------------------------------------------------------

def _select_diagnostic_cases(
    a1_rows: list[dict],
    targets: list[TargetSpec],
    n_per_target: int,
) -> list[str]:
    """Select top-n case IDs per target by max |M_diff| at lead=6h.

    Returns list of case_id strings to use for detailed analysis.
    """
    import collections

    # Group by (target, case_id) and find max |M_diff| at lead=6h
    scores: dict[str, float] = {}
    for row in a1_rows:
        if row["lead_h"] != 6:
            continue
        case_id = row["case_id"]
        target = row["target"]
        M = row.get("M_diff", float("nan"))
        if not np.isnan(M):
            key = f"{target}::{case_id}"
            scores[key] = max(scores.get(key, 0.0), abs(M))

    # Group by target, sort by score, select top-n
    by_target: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for key, score in scores.items():
        target, case_id = key.split("::", 1)
        by_target[target].append((score, case_id))

    selected: list[str] = []
    for target, items in by_target.items():
        items.sort(reverse=True)
        for _, case_id in items[:n_per_target]:
            selected.append(case_id)
            print(f"  Diagnostic case: {case_id} (|M|={items[0][0]:.4f})")

    return selected


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    t_start = time.time()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"START rank={rank}/{world_size} device={device}")

    # --- Apply debug overrides ---
    if args.debug:
        print("DEBUG MODE: first final-cohort case, lead=6h, Stage B only")
        args.leads = [6]
        args.stages = ["B"]
        args.diagnostic_cases_per_target = 1

    # --- Output directory ---
    output_dir = args.output_dir or os.path.join(
        os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
        "model_conditional_comparison",
    )
    _ensure_dir(output_dir)

    # --- Determine stages to run ---
    stages_requested = set(args.stages)
    if "A" in stages_requested:
        stages_requested.update(["A1", "A2", "A3"])
        stages_requested.discard("A")

    # --- Cases ---
    cases = _load_cases(args)
    if args.debug:
        cases = cases[:1]
    if args.targets:
        target_set = set(args.targets)
        cases = [c for c in cases if c["target"] in target_set]
    if not cases:
        print("ERROR: No cases to process. Check --cases and --targets arguments.")
        sys.exit(1)

    print(f"Cases: {len(cases)}")
    for c in cases[:5]:
        print(f"  {c['target']} @ {c['init_time']}")
    if len(cases) > 5:
        print(f"  ... and {len(cases) - 5} more")

    # --- Build targets list ---
    target_map: dict[str, TargetSpec] = {t.name: t for t in DEFAULT_TARGETS}
    targets = [target_map[name] for name in args.outputs if name in target_map]
    if not targets:
        print(f"ERROR: None of --outputs {args.outputs} match known targets: "
              f"{list(target_map.keys())}")
        sys.exit(1)

    # --- Rollout spec ---
    rollout_spec = RolloutSpec(
        steps_hours=tuple(args.leads),
        mode=args.rollout_mode,
        record_each_step=True,
    )

    # --- Resolve model names ---
    model_names = args.models
    if "all" in model_names:
        model_names = list(DEFAULT_MODELS.keys())

    # --- Save run config ---
    if rank == 0:
        run_config = {
            "models": model_names,
            "cases_source": args.cases,
            "n_cases": len(cases),
            "targets": [t.name for t in targets],
            "leads_hours": args.leads,
            "stages": sorted(stages_requested),
            "doses_mm": args.doses_mm,
            "diagnostic_cases_per_target": args.diagnostic_cases_per_target,
            "output_dir": output_dir,
            "debug": args.debug,
            "started_at": datetime.now().isoformat(),
        }
        _write_json(os.path.join(output_dir, "run_config.json"), run_config)

    # --- Load case data (ERA5 + MSWEP) ---
    print("\nLoading case data ...")
    case_data_map = _load_case_data(cases, args.mswep_store)
    print(f"Loaded {len(case_data_map)} case data objects")

    # --- Load models (only on rank 0 or all ranks if needed) ---
    # For stages A1/A2, we run both models — only one model per process if multi-GPU
    # Simple approach: rank 0 loads and runs both models sequentially
    if rank != 0:
        print(f"Rank {rank}: waiting (non-rank-0 ranks have no work in this configuration)")
        elapsed = time.time() - t_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"END rank={rank} elapsed={elapsed:.0f}s")
        return

    loaded_models: dict[str, object] = {}
    model_specs: dict[str, ModelSpec] = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            print(f"WARNING: unknown model {name!r}, skipping")
            continue
        spec = DEFAULT_MODELS[name]
        model_specs[name] = spec
        print(f"\nLoading {name} ...")
        try:
            m = load_model(spec, device)
            loaded_models[name] = m
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            continue

    # --- Stage A1: trajectory comparison ---
    # Dynamically detect which loaded model has ZWD vs not, rather than hardcoding names.
    model_w_name = next((n for n, s in model_specs.items() if s.has_zwd), None)
    model_wo_name = next((n for n, s in model_specs.items() if not s.has_zwd), None)

    a1_rows: list[dict] = []
    if "A1" in stages_requested:
        if model_w_name and model_wo_name:
            print("\n--- Stage A1: Trajectory comparison ---")
            print(f"    model_w={model_w_name}  model_wo={model_wo_name}")
            a1_rows = run_stage_a1_trajectories(
                model_w=loaded_models[model_w_name],
                model_wo=loaded_models[model_wo_name],
                model_spec_w=model_specs[model_w_name],
                model_spec_wo=model_specs[model_wo_name],
                cases=cases,
                rollout_spec=rollout_spec,
                targets=targets,
                case_data_map=case_data_map,
                output_dir=output_dir,
            )
        else:
            print("WARNING: Stage A1 requires one ZWD model and one non-ZWD model. "
                  f"has_zwd={model_w_name!r}  no_zwd={model_wo_name!r}. Skipping A1.")

    # --- Select diagnostic cases ---
    diagnostic_case_ids: list[str] = []
    if args.diagnostic_cases_per_target > 0 and a1_rows:
        print("\nSelecting diagnostic cases ...")
        diagnostic_case_ids = _select_diagnostic_cases(
            a1_rows, targets, args.diagnostic_cases_per_target
        )
        print(f"Selected {len(diagnostic_case_ids)} diagnostic cases")
    elif args.diagnostic_cases_per_target > 0:
        # No A1 rows but still want diagnostic cases — use all cases
        diagnostic_case_ids = [
            f"{c['target']}_{c['init_time']}" for c in cases
        ][:args.diagnostic_cases_per_target * len(set(c["target"] for c in cases))]

    # --- Stage A2: intervention effects ---
    a2_rows_all: dict[str, list[dict]] = {}
    if "A2" in stages_requested:
        print("\n--- Stage A2: Intervention effects ---")
        for model_name, model in loaded_models.items():
            spec = model_specs[model_name]
            print(f"\n  Model: {model_name}")
            rows = run_stage_a2_interventions(
                model=model,
                model_spec=spec,
                cases=cases,
                rollout_spec=rollout_spec,
                targets=targets,
                case_data_map=case_data_map,
                doses_mm=tuple(args.doses_mm),
                output_dir=os.path.join(output_dir, model_name),
            )
            a2_rows_all[model_name] = rows

    # --- Stage A3: conditional differences ---
    if "A3" in stages_requested:
        print("\n--- Stage A3: Conditional differences ---")
        if model_w_name and model_wo_name and \
                model_w_name in a2_rows_all and model_wo_name in a2_rows_all:
            run_stage_a3_conditional_diff(
                a2_rows_w=a2_rows_all[model_w_name],
                a2_rows_wo=a2_rows_all[model_wo_name],
                output_dir=output_dir,
            )
        else:
            print(f"WARNING: A3 needs ZWD and non-ZWD A2 results. "
                  f"Available: {list(a2_rows_all.keys())}. Skipping.")

    # --- Stages B–E: detailed diagnostic stages ---
    # Each stage writes per-model CSVs to output_dir/{model_name}/ and then
    # merges them into a combined top-level CSV (with a 'model' column).
    import pandas as _pd

    def _merge_stage_csv(stage_csv_name: str) -> None:
        """Concatenate per-model CSVs into a single top-level file."""
        parts = []
        for mn in loaded_models:
            p = os.path.join(output_dir, mn, stage_csv_name)
            if os.path.exists(p):
                parts.append(_pd.read_csv(p))
        if parts:
            combined = _pd.concat(parts, ignore_index=True)
            combined.to_csv(os.path.join(output_dir, stage_csv_name), index=False)
            print(f"  Merged {stage_csv_name} ({len(combined)} rows)")

    for model_name, model in loaded_models.items():
        spec = model_specs[model_name]
        stage_out = os.path.join(output_dir, model_name)

        if "B" in stages_requested and diagnostic_case_ids:
            print(f"\n--- Stage B ({model_name}) ---")
            run_stage_b(
                model=model, model_spec=spec,
                cases=cases, case_data_map=case_data_map,
                targets=targets,
                diagnostic_case_ids=diagnostic_case_ids,
                output_dir=stage_out,
            )

        if "C" in stages_requested and diagnostic_case_ids:
            print(f"\n--- Stage C ({model_name}) ---")
            run_stage_c(
                model=model, model_spec=spec,
                cases=cases, case_data_map=case_data_map,
                targets=targets,
                diagnostic_case_ids=diagnostic_case_ids,
                output_dir=stage_out,
                leads_hours=args.leads,
            )

        if "D" in stages_requested and diagnostic_case_ids:
            print(f"\n--- Stage D ({model_name}) ---")
            run_stage_d(
                model=model, model_spec=spec,
                cases=cases, case_data_map=case_data_map,
                diagnostic_case_ids=diagnostic_case_ids,
                output_dir=stage_out,
            )

        if "E" in stages_requested and diagnostic_case_ids:
            print(f"\n--- Stage E ({model_name}) ---")
            run_stage_e(
                model=model, model_spec=spec,
                cases=cases, case_data_map=case_data_map,
                targets=targets,
                diagnostic_case_ids=diagnostic_case_ids,
                output_dir=stage_out,
                leads_hours=args.leads,
            )

    # Merge per-model B–E CSVs into top-level combined files
    for _csv in [
        "stage_b_reliance_summary.csv",
        "stage_c_timing_analysis.csv",
        "stage_d_routing_analysis.csv",
        "stage_e_spatial_restoration.csv",
    ]:
        _merge_stage_csv(_csv)

    # --- Write summary ---
    elapsed = time.time() - t_start
    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "n_cases": len(cases),
        "n_case_data_loaded": len(case_data_map),
        "models_loaded": list(loaded_models.keys()),
        "stages_run": sorted(stages_requested),
        "diagnostic_case_ids": diagnostic_case_ids,
        "output_dir": output_dir,
    }
    _write_json(os.path.join(output_dir, "run_summary.json"), summary)

    # Write a brief summary.md
    summary_md_path = os.path.join(output_dir, "summary.md")
    with open(summary_md_path, "w") as f:
        f.write(f"# 07_zwd_precipitation_model_comparison Run Summary\n\n")
        f.write(f"Completed: {summary['completed_at']}\n")
        f.write(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Models: {', '.join(loaded_models.keys())}\n")
        f.write(f"- Cases: {len(cases)} ({len(case_data_map)} loaded)\n")
        f.write(f"- Leads: {args.leads}h\n")
        f.write(f"- Stages: {sorted(stages_requested)}\n\n")
        f.write(f"## Output files\n")
        base_fnames = [
            "run_config.json",
            "stage_a_model_trajectories.csv",
            "stage_a_conditional_differences.csv",
        ]
        model_fnames = [f"{mn}/stage_a_intervention_effects.csv" for mn in loaded_models]
        for fname in base_fnames + model_fnames:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                f.write(f"- {fname} ({size} bytes)\n")

    print(f"\nOutput: {output_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"END elapsed={elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
