"""
Storm-centered XAI for the ZWD tropical-cyclone case study.

This script answers:
    Which parts of the initial ZWD field most affect the future cyclone's
    tracked position and intensity at selected lead times?

The outputs are designed to be visually compact and thesis-friendly:

- geographic saliency overview (targets x leads)
- storm-relative saliency overview
- selection-basis map with hotspot / control regions
- baseline MSL stamp panels with the forecast track
- CSV / JSON summaries for later perturbation experiments
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch


def _gauss_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(4 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_saliency(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2-D Gaussian filter (pure numpy). sigma in grid cells."""
    if sigma <= 0:
        return arr
    a = arr.astype(np.float64)
    k = _gauss_kernel_1d(sigma)
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), axis=1, arr=a)
    a = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), axis=0, arr=a)
    return a.astype(np.float32)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tc_utils import (  # noqa: E402
    DEFAULT_EXPECTED_STEP_HOURS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAMP_LEADS_HOURS,
    build_model,
    compute_baseline_rollout,
    compute_rollout_ig,
    compute_rollout_saliency,
    ensure_dir,
    format_case_study_id,
    generate_selection_masks,
    lead_hours_to_steps,
    load_best_track,
    load_tc_case,
    make_soft_storm_target,
    plot_best_track,
    plot_lon_order,
    reorder_lon,
    select_regions_from_saliency,
    storm_relative_bin,
    to_plot_lon,
    validate_rollout_cadence,
    write_json,
)
from searchlight_benchmark import _gpu_sync_and_gc  # noqa: E402


TARGET_LABELS = {
    "intensity": "Future Intensity (-soft MSL)",
    "center_lat": "Future Center Latitude",
    "center_lon": "Future Center Longitude",
}


def parse_args():
    p = argparse.ArgumentParser(description="Storm-centered ZWD saliency for a tropical cyclone case.")
    p.add_argument("--case", type=str, default="nanmadol", choices=("nanmadol",))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--lead-hours", type=int, nargs="+", default=[24, 48, 72])
    p.add_argument(
        "--target-kinds",
        type=str,
        nargs="+",
        default=["intensity", "center_lat", "center_lon"],
        choices=sorted(TARGET_LABELS),
    )
    p.add_argument("--selection-target", type=str, default="intensity", choices=sorted(TARGET_LABELS))
    p.add_argument("--selection-lead-hours", type=int, default=48)
    p.add_argument("--selection-low-quantile", type=float, default=0.25)
    p.add_argument("--expected-step-hours", type=float, default=DEFAULT_EXPECTED_STEP_HOURS)
    p.add_argument("--allow-step-mismatch", action="store_true")
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma (grid cells, 0.25 deg each) applied to attribution maps before plotting. 0 = no smoothing.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="saliency",
        choices=("saliency", "ig"),
        help="Attribution method: vanilla gradient saliency or Integrated Gradients.",
    )
    p.add_argument(
        "--ig-steps",
        type=int,
        default=10,
        help="Number of IG interpolation steps (midpoint rule). Only used with --method ig.",
    )
    return p.parse_args()


def _plot_base_map(ax, case_cfg):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(case_cfg.analysis_extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, linestyle="--", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False


_BEST_TRACK_CACHE: dict[str, pd.DataFrame | None] = {}


def _best_track_for(case_cfg) -> pd.DataFrame | None:
    if case_cfg.slug not in _BEST_TRACK_CACHE:
        _BEST_TRACK_CACHE[case_cfg.slug] = load_best_track(case_cfg)
    return _BEST_TRACK_CACHE[case_cfg.slug]


def _plot_track(ax, baseline_df: pd.DataFrame, lead_hours: int, case_cfg=None):
    import cartopy.crs as ccrs

    sub = baseline_df[baseline_df["lead_hours"] <= float(lead_hours)]
    if sub.empty:
        return
    if case_cfg is not None:
        best_df = _best_track_for(case_cfg)
        if best_df is not None:
            plot_best_track(
                ax, best_df,
                start=case_cfg.init_time,
                end=pd.to_datetime(sub["valid_time"]).max(),
                full_context=False,
                linewidth=1.4,
                markersize=2.2,
            )
    ax.plot(
        sub["center_lon_plot"],
        sub["center_lat"],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=1.2,
        marker="o",
        markersize=3.5,
        zorder=5,
    )
    ax.scatter(
        sub["center_lon_plot"].iloc[-1],
        sub["center_lat"].iloc[-1],
        transform=ccrs.PlateCarree(),
        color="white",
        edgecolors="black",
        linewidths=0.7,
        s=42,
        zorder=6,
    )


def _plot_selected_regions(ax, selected_regions: dict[str, dict[str, Any]]):
    import cartopy.crs as ccrs

    styles = {
        "hotspot": ("Hotspot", "red", "*"),
        "low_near": ("Low-saliency near", "gold", "o"),
        "remote": ("Remote control", "cyan", "^"),
    }
    for key, payload in selected_regions.items():
        label, color, marker = styles[key]
        ax.scatter(
            payload["center_lon_plot"],
            payload["center_lat"],
            transform=ccrs.PlateCarree(),
            color=color,
            marker=marker,
            s=90,
            edgecolors="black",
            linewidths=0.7,
            label=label,
            zorder=7,
        )


def _plot_geographic_overview(
    out_path: str,
    *,
    case_cfg,
    case_data,
    baseline_df: pd.DataFrame,
    saliency_results: dict[str, dict[int, dict[str, Any]]],
    selected_regions: dict[str, dict[str, Any]],
    selection_target: str,
    selection_lead_hours: int,
    smooth_sigma: float = 3.0,
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_kinds = list(saliency_results.keys())
    lead_hours = sorted(next(iter(saliency_results.values())).keys())
    lon_order = plot_lon_order(case_data.lon_vals)
    lon_plot = to_plot_lon(case_data.lon_vals)[lon_order]

    fig, axes = plt.subplots(
        len(target_kinds),
        len(lead_hours),
        figsize=(4.6 * len(lead_hours), 3.8 * len(target_kinds)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_2d(axes)

    for row_idx, target_kind in enumerate(target_kinds):
        vmax = float(
            np.nanpercentile(
                [
                    np.abs(smooth_saliency(saliency_results[target_kind][lead]["saliency_map"], smooth_sigma))
                    for lead in lead_hours
                ],
                99.0,
            )
        )
        vmax = max(vmax, 1e-8)
        mappable = None
        for col_idx, lead_h in enumerate(lead_hours):
            ax = axes[row_idx, col_idx]
            _plot_base_map(ax, case_cfg)
            sal_map = saliency_results[target_kind][lead_h]["saliency_map"]
            sal_map = smooth_saliency(sal_map, smooth_sigma)
            sal_plot = reorder_lon(sal_map, lon_order)
            mesh = ax.pcolormesh(
                lon_plot,
                case_data.lat_vals,
                sal_plot,
                cmap="coolwarm",
                shading="auto",
                vmin=-vmax,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            mappable = mesh
            _plot_track(ax, baseline_df, lead_h, case_cfg=case_cfg)
            if target_kind == selection_target and lead_h == selection_lead_hours:
                _plot_selected_regions(ax, selected_regions)
            if row_idx == 0:
                ax.set_title(f"+{lead_h}h", fontsize=11)
            if col_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    TARGET_LABELS[target_kind],
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                )
        cb = fig.colorbar(mappable, ax=axes[row_idx, :], shrink=0.88, pad=0.02)
        cb.set_label("Signed saliency")

    handles = []
    labels = []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        dedup = {}
        for h, l in zip(handles, labels):
            dedup[l] = h
        fig.legend(
            list(dedup.values()),
            list(dedup.keys()),
            loc="lower center",
            ncol=min(3, len(dedup)),
            frameon=True,
        )
    fig.suptitle(f"{case_cfg.name}: storm-centered ZWD saliency", fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_storm_relative_overview(
    out_path: str,
    *,
    case_cfg,
    saliency_results: dict[str, dict[int, dict[str, Any]]],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_kinds = list(saliency_results.keys())
    lead_hours = sorted(next(iter(saliency_results.values())).keys())

    fig, axes = plt.subplots(
        len(target_kinds),
        len(lead_hours),
        figsize=(4.2 * len(lead_hours), 3.2 * len(target_kinds)),
        squeeze=False,
    )

    for row_idx, target_kind in enumerate(target_kinds):
        vmax = float(
            np.nanpercentile(
                [
                    np.abs(saliency_results[target_kind][lead]["storm_relative"])
                    for lead in lead_hours
                ],
                99.0,
            )
        )
        vmax = max(vmax, 1e-8)
        mappable = None
        for col_idx, lead_h in enumerate(lead_hours):
            ax = axes[row_idx, col_idx]
            polar_map = saliency_results[target_kind][lead_h]["storm_relative"]
            mappable = ax.imshow(
                polar_map,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            ax.set_xticks([0, 4, 8, 12, 15], labels=["N", "E", "S", "W", "N"])
            ax.set_yticks(range(polar_map.shape[0]))
            ax.set_yticklabels([f"{int(v)}" for v in np.linspace(0, case_cfg.polar_radius_km, polar_map.shape[0])])
            if row_idx == 0:
                ax.set_title(f"+{lead_h}h", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{TARGET_LABELS[target_kind]}\nRadius (km)")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Azimuth")
        cb = fig.colorbar(mappable, ax=axes[row_idx, :], shrink=0.88, pad=0.02)
        cb.set_label("Mean saliency")

    fig.suptitle(f"{case_cfg.name}: storm-relative saliency structure", fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_basis(
    out_path: str,
    *,
    case_cfg,
    case_data,
    baseline_df: pd.DataFrame,
    selected_regions: dict[str, dict[str, Any]],
    selection_result: dict[str, Any],
    smooth_sigma: float = 3.0,
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sal_map = smooth_saliency(selection_result["saliency_map"], smooth_sigma)
    lon_order = plot_lon_order(case_data.lon_vals)
    lon_plot = to_plot_lon(case_data.lon_vals)[lon_order]
    sal_plot = reorder_lon(sal_map, lon_order)
    vmax = float(np.nanpercentile(np.abs(sal_plot), 99.0))
    vmax = max(vmax, 1e-8)

    fig = plt.figure(figsize=(10.5, 6.4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _plot_base_map(ax, case_cfg)
    mesh = ax.pcolormesh(
        lon_plot,
        case_data.lat_vals,
        sal_plot,
        cmap="coolwarm",
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    _plot_track(ax, baseline_df, int(selection_result["lead_hours"]), case_cfg=case_cfg)
    _plot_selected_regions(ax, selected_regions)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_title(
        f"{case_cfg.name}: selection basis = {TARGET_LABELS[selection_result['target_kind']]} at +{selection_result['lead_hours']}h"
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.03)
    cb.set_label("Signed saliency")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_msl_stamps(
    out_path: str,
    *,
    case_cfg,
    baseline_df: pd.DataFrame,
    baseline_snapshots: dict[int, dict[str, np.ndarray]],
    stamp_steps: list[int],
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(stamp_steps),
        figsize=(4.8 * len(stamp_steps), 4.4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    axes = axes[0]

    for ax, step_idx in zip(axes, stamp_steps):
        snap = baseline_snapshots[step_idx]
        lat_vals = snap["lat_vals"]
        lon_vals = snap["lon_vals"]
        msl = snap["msl_hpa"]

        lon_order = plot_lon_order(lon_vals)
        lon_plot = to_plot_lon(lon_vals)[lon_order]
        msl_plot = reorder_lon(msl, lon_order)

        _plot_base_map(ax, case_cfg)
        mesh = ax.pcolormesh(
            lon_plot,
            lat_vals,
            msl_plot,
            cmap="Spectral_r",
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        step_lead = int(round(float(baseline_df.loc[baseline_df["step"] == step_idx, "lead_hours"].iloc[0])))
        _plot_track(ax, baseline_df, step_lead, case_cfg=case_cfg)
        ax.set_title(f"+{step_lead}h")
        cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.03)
        cb.set_label("MSL (hPa)")

    fig.suptitle(f"{case_cfg.name}: baseline MSL evolution and tracked path", fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _quadrant_records(
    saliency_results: dict[str, dict[int, dict[str, Any]]],
    *,
    case_cfg,
) -> list[dict[str, Any]]:
    rows = []
    quadrants = [
        ("north", 315.0, 45.0),
        ("east", 45.0, 135.0),
        ("south", 135.0, 225.0),
        ("west", 225.0, 315.0),
    ]
    radius_bins_km = np.linspace(0.0, case_cfg.polar_radius_km, 9)

    for target_kind, by_lead in saliency_results.items():
        for lead_h, payload in by_lead.items():
            field = payload["saliency_map"]
            center_lat = float(payload["center_lat"])
            center_lon = float(payload["center_lon"])
            lat_vals = payload["lat_vals"]
            lon_vals = payload["lon_vals"]

            lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
            dlon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
            x_km = dlon * np.cos(np.radians(center_lat)) * 111.0
            y_km = (lat_grid - center_lat) * 111.0
            r_km = np.sqrt(np.square(x_km) + np.square(y_km))
            az_deg = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0

            for quadrant, az0, az1 in quadrants:
                if az0 < az1:
                    az_mask = (az_deg >= az0) & (az_deg < az1)
                else:
                    az_mask = (az_deg >= az0) | (az_deg < az1)
                ring_mask = r_km <= case_cfg.polar_radius_km
                mask = az_mask & ring_mask
                if not np.any(mask):
                    continue
                rows.append({
                    "target_kind": target_kind,
                    "lead_hours": int(lead_h),
                    "quadrant": quadrant,
                    "mean_signed_saliency": float(np.nanmean(field[mask])),
                    "mean_abs_saliency": float(np.nanmean(np.abs(field[mask]))),
                })

            for ridx in range(len(radius_bins_km) - 1):
                mask = (r_km >= radius_bins_km[ridx]) & (r_km < radius_bins_km[ridx + 1])
                if not np.any(mask):
                    continue
                rows.append({
                    "target_kind": target_kind,
                    "lead_hours": int(lead_h),
                    "quadrant": f"ring_{int(radius_bins_km[ridx])}_{int(radius_bins_km[ridx + 1])}km",
                    "mean_signed_saliency": float(np.nanmean(field[mask])),
                    "mean_abs_saliency": float(np.nanmean(np.abs(field[mask]))),
                })
    return rows


def main():
    args = parse_args()

    case_cfg, case_data = load_tc_case(args.case)
    study_id = format_case_study_id(case_cfg)
    out_dir = os.path.join(args.output_dir, study_id, "storm_centered_xai")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    timing = validate_rollout_cadence(
        model,
        case_data,
        device,
        expected_step_hours=args.expected_step_hours,
        check_steps=2,
    )
    write_json(os.path.join(out_dir, "timing_check.json"), timing)
    if not timing["passes"] and not args.allow_step_mismatch:
        raise RuntimeError(
            "Model rollout cadence does not match the requested 6h setup. "
            "See timing_check.json."
        )
    step_hours = float(timing["observed_step_hours"])

    max_steps = max(lead_hours_to_steps(int(lead_h), step_hours) for lead_h in args.lead_hours)
    stamp_steps = {
        lead_hours_to_steps(lead_h, step_hours)
        for lead_h in DEFAULT_STAMP_LEADS_HOURS
        if lead_h in [int(x) for x in args.lead_hours]
    }
    baseline_df, baseline_snapshots = compute_baseline_rollout(
        model,
        case_cfg,
        case_data,
        device,
        steps=max_steps,
        snapshot_steps=stamp_steps,
    )
    baseline_df.to_csv(os.path.join(out_dir, "baseline_track.csv"), index=False)
    _gpu_sync_and_gc()

    saliency_results: dict[str, dict[int, dict[str, Any]]] = {}
    score_rows = []

    for target_kind in args.target_kinds:
        saliency_results[target_kind] = {}
        for lead_h in args.lead_hours:
            step_idx = lead_hours_to_steps(int(lead_h), step_hours)
            baseline_row = baseline_df.loc[baseline_df["step"] == step_idx].iloc[0]
            target_fn = make_soft_storm_target(
                case_data,
                center_lat=float(baseline_row["center_lat"]),
                center_lon=float(baseline_row["center_lon"]),
                window_radius_deg=case_cfg.track_window_radius_deg,
                target_kind=target_kind,
                softmin_temp_hpa=case_cfg.softmin_temp_hpa,
            )
            if args.method == "ig":
                saliency_map, score_val, valid_time = compute_rollout_ig(
                    model,
                    case_data,
                    device,
                    lead_steps=step_idx,
                    target_fn=target_fn,
                    n_steps=args.ig_steps,
                )
            else:
                saliency_map, score_val, valid_time = compute_rollout_saliency(
                    model,
                    case_data,
                    device,
                    lead_steps=step_idx,
                    target_fn=target_fn,
                )
            storm_relative = storm_relative_bin(
                saliency_map,
                case_data.lat_vals,
                case_data.lon_vals,
                center_lat=float(baseline_row["center_lat"]),
                center_lon=float(baseline_row["center_lon"]),
                radius_km=case_cfg.polar_radius_km,
            )
            saliency_results[target_kind][int(lead_h)] = {
                "target_kind": target_kind,
                "lead_hours": int(lead_h),
                "valid_time": valid_time.isoformat(),
                "center_lat": float(baseline_row["center_lat"]),
                "center_lon": float(baseline_row["center_lon"]),
                "score": score_val,
                "saliency_map": saliency_map,
                "storm_relative": storm_relative,
                "lat_vals": case_data.lat_vals,
                "lon_vals": case_data.lon_vals,
            }
            np.save(
                os.path.join(out_dir, f"{args.method}_{target_kind}_{int(lead_h):03d}h.npy"),
                saliency_map,
            )
            np.save(
                os.path.join(out_dir, f"storm_relative_{target_kind}_{int(lead_h):03d}h.npy"),
                storm_relative,
            )
            score_rows.append({
                "target_kind": target_kind,
                "lead_hours": int(lead_h),
                "valid_time": valid_time.isoformat(),
                "score": score_val,
                "anchor_center_lat": float(baseline_row["center_lat"]),
                "anchor_center_lon": float(baseline_row["center_lon"]),
            })

    pd.DataFrame(score_rows).to_csv(os.path.join(out_dir, f"{args.method}_scores.csv"), index=False)

    selection_result = saliency_results[args.selection_target][int(args.selection_lead_hours)]
    masks = generate_selection_masks(case_cfg)
    selected_regions = select_regions_from_saliency(
        case_cfg=case_cfg,
        case_data=case_data,
        saliency_map=selection_result["saliency_map"],
        masks=masks,
        low_quantile=args.selection_low_quantile,
    )
    write_json(os.path.join(out_dir, "selected_regions.json"), selected_regions)

    quadrant_rows = _quadrant_records(saliency_results, case_cfg=case_cfg)
    pd.DataFrame(quadrant_rows).to_csv(os.path.join(out_dir, "quadrant_summary.csv"), index=False)

    _plot_geographic_overview(
        os.path.join(out_dir, f"{args.method}_geographic_overview.png"),
        case_cfg=case_cfg,
        case_data=case_data,
        baseline_df=baseline_df,
        saliency_results=saliency_results,
        selected_regions=selected_regions,
        selection_target=args.selection_target,
        selection_lead_hours=int(args.selection_lead_hours),
        smooth_sigma=args.smooth_sigma,
    )
    _plot_storm_relative_overview(
        os.path.join(out_dir, "saliency_storm_relative_overview.png"),
        case_cfg=case_cfg,
        saliency_results=saliency_results,
    )
    _plot_selection_basis(
        os.path.join(out_dir, "selection_basis_map.png"),
        case_cfg=case_cfg,
        case_data=case_data,
        baseline_df=baseline_df,
        selected_regions=selected_regions,
        selection_result=selection_result,
        smooth_sigma=args.smooth_sigma,
    )
    if stamp_steps:
        _plot_baseline_msl_stamps(
            os.path.join(out_dir, "baseline_msl_stamps.png"),
            case_cfg=case_cfg,
            baseline_df=baseline_df,
            baseline_snapshots=baseline_snapshots,
            stamp_steps=sorted(stamp_steps),
        )

    write_json(
        os.path.join(out_dir, "summary.json"),
        {
            "case": case_cfg.slug,
            "study_id": study_id,
            "step_hours": step_hours,
            "lead_hours": list(args.lead_hours),
            "target_kinds": list(args.target_kinds),
            "selection_target": args.selection_target,
            "selection_lead_hours": int(args.selection_lead_hours),
            "selected_regions": selected_regions,
        },
    )
    print(f"Storm-centered XAI outputs written to {out_dir}")


if __name__ == "__main__":
    main()
