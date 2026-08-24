"""Controlled activation patching for ZWD, humidity, and precipitation.

Every variable is tested in the same frozen ``precip_large_zwd`` checkpoint,
on the same cases, with the same q850 target and an analogous baseline:

* zwd:       actual field vs per-timestep spatial mean
* q:         actual field vs per-timestep/per-level spatial mean
* tp_mswep:  actual field vs per-timestep spatial mean

The source/base inputs differ only in the selected variable.  With
``--shard-by-rank``, four Slurm ranks split the shared eight-case file and
write independent shard directories that can safely be concatenated later.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ZWD_TRACE_DIR = os.path.join(_ROOT, "08_internal_routing_and_activation_patching/activation_patching")
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
_COND_DIR = os.path.join(_ROOT, "04_zwd_counterfactual_interventions")
_MCC_DIR = os.path.join(_ROOT, "07_zwd_precipitation_model_comparison")
_P_CORR_DIR = os.path.join(_ROOT, "06_precipitation_moisture_relationships")

for _p in (
    _HERE,
    _ZWD_TRACE_DIR,
    _ROOT,
    _SEARCHLIGHT_DIR,
    _COND_DIR,
    _MCC_DIR,
    _P_CORR_DIR,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from activation_patch_unet import (  # noqa: E402
    LEADS_HOURS,
    PATCH_REGIONS,
    PATCH_SITES,
    PATCH_SITE_SPATIAL,
    _RegionSpec,
    _SALIENCY_SIGMA_DEG,
    _build_region_masks,
    _cache_all_sites,
    _ensure_dir,
    _gpu_sync_and_gc,
    _mask_to_token_tensor,
    _run_baseline_rollout,
    _run_patched_rollout,
    _safe_div,
    _write_csv,
    _write_json,
    _write_plots,
    _write_summary_csvs,
)
MODEL_KEY = "precip_large_zwd"
VARIABLES = ("zwd", "q", "precip")
INPUT_NAMES = {"zwd": "zwd", "q": "q", "precip": "tp_mswep"}
CONTRASTS = {
    "zwd": "actual_vs_uniform_zwd",
    "q": "actual_vs_uniform_q",
    "precip": "actual_vs_uniform_tp",
}
BASELINE_DESCRIPTIONS = {
    "zwd": "per-timestep spatial mean",
    "q": "per-timestep/per-level spatial mean",
    "precip": "per-timestep spatial mean",
}

DEFAULT_CASES_JSON = os.path.join(
    _HERE, "cases_activation_patch_8.json"
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "activation_patch_all_inputs",
)


def _uniform_spatial_mean(value: torch.Tensor) -> torch.Tensor:
    """Remove horizontal structure while preserving every leading slice mean."""
    return value.mean(dim=(-2, -1), keepdim=True).expand_as(value).clone()


def _setup_model(device: torch.device):
    from comparison_config import DEFAULT_MODELS
    from comparison_models import load_model

    model_spec = DEFAULT_MODELS[MODEL_KEY]
    return load_model(model_spec, device), model_spec


_MSWEP_READER: Any | None = None


def _load_mswep_strict(case: Any, init_time: Any) -> Any:
    """Inject real MSWEP t0/t1 fields; never substitute synthetic zeros."""
    global _MSWEP_READER

    from common import MSWEPReader
    from comparison_config import MSWEP_STORE_PATH
    from comparison_data import inject_precip_into_case, load_precip_for_case

    if _MSWEP_READER is None:
        _MSWEP_READER = MSWEPReader(MSWEP_STORE_PATH)
    precip = load_precip_for_case(init_time, _MSWEP_READER)
    t0, t1 = precip.get("t0"), precip.get("t1")
    if t0 is None or t1 is None:
        raise RuntimeError(
            f"MSWEP t0/t1 unavailable for {init_time}; refusing a zero fallback"
        )

    expected_h = len(case.lat_vals)

    def pad_lat(array: np.ndarray) -> np.ndarray:
        if array.shape[0] == expected_h:
            return array
        if array.shape[0] + 1 == expected_h:
            return np.concatenate([array, array[-1:]], axis=0)
        raise ValueError(f"Cannot align MSWEP latitude size {array.shape[0]} to {expected_h}")

    return inject_precip_into_case(case, pad_lat(t0), pad_lat(t1))


def _make_batch(
    case: Any,
    model_spec: Any,
    device: torch.device,
    *,
    variable: str | None = None,
    override: torch.Tensor | None = None,
    requires_grad: bool = False,
):
    from comparison_data import build_precip_batch

    kwargs: dict[str, Any] = {}
    if variable == "zwd":
        kwargs["zwd_override"] = override
        if requires_grad:
            kwargs["requires_grad_surf"] = ("zwd",)
    elif variable == "precip":
        kwargs["precip_override"] = override
        if requires_grad:
            kwargs["requires_grad_surf"] = ("tp_mswep",)
    elif variable == "q":
        kwargs["atmos_override"] = {"q": override} if override is not None else None
        if requires_grad:
            kwargs["requires_grad_atmos"] = ("q",)
    elif variable is not None:
        raise ValueError(f"Unsupported variable: {variable}")

    return build_precip_batch(case, model_spec, device, **kwargs)


def _make_uniform_baseline(case: Any, variable: str) -> torch.Tensor:
    if variable == "q":
        value = case.atmos_cpu["q"]
    else:
        value = case.surf_cpu[INPUT_NAMES[variable]]
    return _uniform_spatial_mean(value)


def _saliency_map(
    model: Any,
    model_spec: Any,
    case: Any,
    variable: str,
    target_fn: Any,
    device: torch.device,
) -> np.ndarray:
    """Return t1 absolute saliency, reducing levels for atmospheric q."""
    from xia_methods.saliency import saliency

    input_name = INPUT_NAMES[variable]

    def batch_fn(requires_grad: bool):
        return _make_batch(
            case,
            model_spec,
            device,
            variable=variable,
            requires_grad=requires_grad,
        )

    kwargs = (
        {"atmos_var_names": (input_name,)}
        if variable == "q"
        else {"surf_var_names": (input_name,)}
    )
    result = saliency(model, batch_fn, target_fn, device, **kwargs)
    grad = result["grads"].get(input_name)
    if grad is None:
        raise ValueError(f"No {input_name} gradient returned by saliency")
    if variable == "q":
        return np.abs(grad[0, 1]).sum(axis=0)
    return np.abs(grad[0, 1])


def _select_gaussian_regions(
    model: Any,
    model_spec: Any,
    case: Any,
    case_obj: Any,
    variable: str,
    target_fn: Any,
    device: torch.device,
) -> tuple[_RegionSpec, _RegionSpec, _RegionSpec]:
    """Select variable-specific hotspot and matched low/remote controls."""
    from searchlight_tasks import (
        SCALES,
        TARGETS,
        cos_lat_weights,
        gaussian_mask,
        generate_mask_centers,
        great_circle_km,
    )

    target = TARGETS[case_obj.target]
    specs = generate_mask_centers(target, SCALES["synoptic"])
    lat_vals, lon_vals = case.lat_vals, case.lon_vals

    try:
        saliency_hw = _saliency_map(
            model, model_spec, case, variable, target_fn, device
        )
    except Exception as exc:
        print(
            f"  WARNING: {variable} saliency failed ({exc}); using uniform saliency.",
            flush=True,
        )
        saliency_hw = np.ones((len(lat_vals), len(lon_vals)), dtype=np.float32)
    _gpu_sync_and_gc()

    cos_w = cos_lat_weights(lat_vals, len(lon_vals))
    near_specs = [spec for spec in specs if spec.role == "near"]
    remote_specs = [spec for spec in specs if spec.role == "remote"]

    def pool(spec: Any) -> float:
        mask = gaussian_mask(spec, _SALIENCY_SIGMA_DEG, lat_vals, lon_vals)
        weights = mask * cos_w
        denom = float(weights.sum())
        return float((saliency_hw * weights).sum() / denom) if denom > 1e-10 else 0.0

    near_values = np.asarray([pool(spec) for spec in near_specs])
    remote_values = np.asarray([pool(spec) for spec in remote_specs])
    if near_values.size == 0:
        hotspot = _RegionSpec("hotspot", target.center_lat, target.center_lon, math.nan)
        low = _RegionSpec("low_near", target.center_lat + 5.0, target.center_lon, math.nan)
        remote = _RegionSpec(
            "remote", target.center_lat, (target.center_lon + 180.0) % 360.0, math.nan
        )
        return hotspot, low, remote

    hotspot_i = int(np.argmax(near_values))
    hotspot_spec = near_specs[hotspot_i]
    hotspot_dist = float(
        great_circle_km(
            target.center_lat,
            target.center_lon,
            hotspot_spec.center_lat,
            hotspot_spec.center_lon,
        )
    )

    candidates = [i for i in range(len(near_specs)) if i != hotspot_i]
    if candidates:
        low_cut = float(np.quantile(near_values[candidates], 0.25))
        low_candidates = [i for i in candidates if near_values[i] <= low_cut] or candidates
        distances = np.asarray(
            [
                float(
                    great_circle_km(
                        target.center_lat,
                        target.center_lon,
                        near_specs[i].center_lat,
                        near_specs[i].center_lon,
                    )
                )
                for i in low_candidates
            ]
        )
        low_i = low_candidates[int(np.argmin(np.abs(distances - hotspot_dist)))]
    else:
        low_i = hotspot_i

    remote_i = int(np.argmin(remote_values)) if remote_values.size else 0
    remote_spec = remote_specs[remote_i] if remote_specs else hotspot_spec
    return (
        _RegionSpec(
            "hotspot",
            hotspot_spec.center_lat,
            hotspot_spec.center_lon,
            float(near_values[hotspot_i]),
        ),
        _RegionSpec(
            "low_near",
            near_specs[low_i].center_lat,
            near_specs[low_i].center_lon,
            float(near_values[low_i]),
        ),
        _RegionSpec(
            "remote",
            remote_spec.center_lat,
            remote_spec.center_lon,
            float(remote_values[remote_i]) if remote_values.size else math.nan,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled all-input activation patching in precip_large_zwd."
    )
    parser.add_argument("--variable", required=True, choices=VARIABLES)
    parser.add_argument("--cases", default=DEFAULT_CASES_JSON)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sites", nargs="+", default=list(PATCH_SITES), choices=PATCH_SITES)
    parser.add_argument(
        "--regions", nargs="+", default=list(PATCH_REGIONS), choices=PATCH_REGIONS
    )
    parser.add_argument(
        "--leads", nargs="+", type=int, default=list(LEADS_HOURS), choices=LEADS_HOURS
    )
    parser.add_argument(
        "--shard-by-rank",
        action="store_true",
        help="Split cases by SLURM_PROCID/SLURM_NTASKS and write shard directories.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from comparison_config import DEFAULT_MODELS
    from searchlight_benchmark import make_q850_box_target
    from searchlight_data import load_case
    from searchlight_tasks import load_cases_from_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_cases_from_json(args.cases)
    if not cases:
        raise ValueError("No cases found")

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if args.shard_by_rank:
        cases = [case for i, case in enumerate(cases) if i % world_size == rank]
        output_dir = os.path.join(args.output_root, args.variable, f"shard{rank + 1}")
    else:
        output_dir = os.path.join(args.output_root, args.variable)

    sites, regions, leads = args.sites, args.regions, args.leads
    if args.smoke:
        cases = cases[:1]
        sites = ["enc_s0_skip", "enc_s2_bottleneck", "dec_s2_post_concat"]
        regions = ["whole", "hotspot_gaussian", "remote_control"]
        leads = [6]

    if not cases:
        print(f"Rank {rank}: no assigned cases; exiting.", flush=True)
        return

    model_spec_for_config = DEFAULT_MODELS[MODEL_KEY]
    _ensure_dir(output_dir)
    _write_json(
        os.path.join(output_dir, "config.json"),
        {
            "experiment": "controlled_all_inputs_activation_patch",
            "variable": args.variable,
            "input_name": INPUT_NAMES[args.variable],
            "contrast": CONTRASTS[args.variable],
            "baseline": BASELINE_DESCRIPTIONS[args.variable],
            "target_metric": "q850_target_box_mean_g_per_kg",
            "model_key": MODEL_KEY,
            "checkpoint_path": model_spec_for_config.checkpoint_path,
            "surf_vars": model_spec_for_config.surf_vars,
            "atmos_vars": model_spec_for_config.atmos_vars,
            "cases_file": os.path.abspath(args.cases),
            "rank": rank,
            "world_size": world_size,
            "sites": sites,
            "regions": regions,
            "leads": leads,
            "patch_site_spatial": PATCH_SITE_SPATIAL,
            "cases": [
                {
                    "target": case.target,
                    "init_time": case.init_time.isoformat(),
                    "role": case.role,
                }
                for case in cases
            ],
        },
    )

    print(
        f"Controlled patching: variable={args.variable}, rank={rank}/{world_size}, "
        f"cases={len(cases)}, output={output_dir}",
        flush=True,
    )
    model, model_spec = _setup_model(device)
    backbone = model.backbone
    rows: list[dict[str, Any]] = []

    for case_obj in cases:
        print(f"\n=== Case {case_obj.case_id} ===", flush=True)
        case = load_case(case_obj.init_time)
        case = _load_mswep_strict(case, case_obj.init_time)
        target_fn, _ = make_q850_box_target(case, case_obj.target)
        actual_batch = _make_batch(case, model_spec, device)
        uniform = _make_uniform_baseline(case, args.variable)
        base_batch = _make_batch(
            case,
            model_spec,
            device,
            variable=args.variable,
            override=uniform,
        )

        print(f"  selecting {args.variable} saliency regions ...", flush=True)
        hotspot, low_near, remote = _select_gaussian_regions(
            model,
            model_spec,
            case,
            case_obj,
            args.variable,
            target_fn,
            device,
        )
        print(
            f"  hotspot=({hotspot.center_lat:.2f}, {hotspot.center_lon:.2f}) "
            f"low=({low_near.center_lat:.2f}, {low_near.center_lon:.2f})",
            flush=True,
        )

        masks = _build_region_masks(case, case_obj, hotspot, low_near, remote)
        masks = {name: masks[name] for name in regions}
        full_h, full_w = len(case.lat_vals), len(case.lon_vals)

        print("  caching source activations ...", flush=True)
        source_acts, _ = _cache_all_sites(model, actual_batch, target_fn, backbone)
        print("  caching baseline activations ...", flush=True)
        base_acts, _ = _cache_all_sites(model, base_batch, target_fn, backbone)

        for lead_h in leads:
            lead_steps = lead_h // 6
            score_base = _run_baseline_rollout(model, base_batch, target_fn, lead_steps)
            score_source = _run_baseline_rollout(model, actual_batch, target_fn, lead_steps)
            denominator = score_source - score_base
            print(
                f"  lead={lead_h}h base={score_base:.6f} source={score_source:.6f} "
                f"delta={denominator:.6f}",
                flush=True,
            )

            for site in sites:
                for region_name, mask_hw in masks.items():
                    mask_tok = _mask_to_token_tensor(
                        mask_hw,
                        site,
                        full_h,
                        full_w,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                    score_patched = _run_patched_rollout(
                        model=model,
                        base_batch=base_batch,
                        backbone=backbone,
                        site=site,
                        base_act=base_acts[site],
                        src_act=source_acts[site],
                        mask_Ntok_1=mask_tok,
                        target_fn=target_fn,
                        lead_steps=lead_steps,
                    )
                    patched_delta = score_patched - score_base
                    recovery = _safe_div(patched_delta, denominator)
                    rows.append(
                        {
                            "case_id": case_obj.case_id,
                            "target": case_obj.target,
                            "role": case_obj.role,
                            "model_key": MODEL_KEY,
                            "variable": args.variable,
                            "input_name": INPUT_NAMES[args.variable],
                            "target_metric": "q850_box_mean_g_per_kg",
                            "baseline": BASELINE_DESCRIPTIONS[args.variable],
                            "contrast": CONTRASTS[args.variable],
                            "lead_h": lead_h,
                            "patch_site": site,
                            "patch_region": region_name,
                            "score_base": score_base,
                            "score_source": score_source,
                            "score_patched": score_patched,
                            "delta_source_minus_base": denominator,
                            "delta_patched_minus_base": patched_delta,
                            "recovery": recovery,
                            "hotspot_lat": hotspot.center_lat,
                            "hotspot_lon": hotspot.center_lon,
                            "low_near_lat": low_near.center_lat,
                            "low_near_lon": low_near.center_lon,
                        }
                    )
                    print(
                        f"    [{site}][{region_name}] recovery={recovery:.3f}",
                        flush=True,
                    )
        _gpu_sync_and_gc()

    _write_csv(os.path.join(output_dir, "activation_patch_scores.csv"), rows)
    _write_summary_csvs(output_dir, rows)
    if not args.skip_plots:
        _write_plots(
            output_dir,
            rows,
            [CONTRASTS[args.variable]],
            leads,
            sites,
            regions,
        )
    print(f"Done. Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
