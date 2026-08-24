"""
Direct ZWD perturbation tests for the tropical-cyclone case study.

This script reuses the region selection from `storm_centered_xai.py` and asks:
    If we perturb the initial ZWD field inside the selected hotspot / controls,
    how does the forecast track and storm intensity respond over the next days?

Outputs emphasize visual comparison:

- map overlay of baseline and perturbed tracks
- lead-time track displacement curves
- lead-time intensity-response curves
- storm MSL stamp panels for baseline and +1σ perturbations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tc_utils import (  # noqa: E402
    DEFAULT_EXPECTED_STEP_HOURS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAMP_LEADS_HOURS,
    build_model,
    compute_baseline_rollout,
    ensure_dir,
    format_case_study_id,
    great_circle_km,
    lead_hours_to_steps,
    load_best_track,
    load_tc_case,
    plot_best_track,
    plot_lon_order,
    reorder_lon,
    run_perturbed_rollout,
    to_plot_lon,
    validate_rollout_cadence,
    write_json,
)


REGION_STYLES = {
    "hotspot": {"label": "Hotspot", "color": "#d1495b"},
    "low_near": {"label": "Low-saliency near", "color": "#edae49"},
    "remote": {"label": "Remote control", "color": "#00798c"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Track perturbation study for a ZWD cyclone case.")
    p.add_argument("--case", type=str, default="nanmadol", choices=("nanmadol",))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--amplitudes", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
    p.add_argument("--selection-json", type=str, default=None)
    p.add_argument("--expected-step-hours", type=float, default=DEFAULT_EXPECTED_STEP_HOURS)
    p.add_argument("--allow-step-mismatch", action="store_true")
    return p.parse_args()


def _load_selected_regions(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    for key in ("hotspot", "low_near", "remote"):
        if key not in payload:
            raise KeyError(f"Missing {key!r} in {path}")
    return payload


def _plot_base_map_no_extent(ax):
    import cartopy.feature as cfeature

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, linestyle="--", alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False


def _plot_base_map(ax, case_cfg):
    import cartopy.crs as ccrs

    ax.set_extent(case_cfg.map_extent, crs=ccrs.PlateCarree())
    _plot_base_map_no_extent(ax)


def _plot_track(ax, df: pd.DataFrame, *, color: str, linewidth: float, linestyle: str, label: str):
    import cartopy.crs as ccrs

    ax.plot(
        df["center_lon_plot"],
        df["center_lat"],
        transform=ccrs.PlateCarree(),
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker="o",
        markersize=2.8,
        label=label,
        zorder=5,
    )


def _extent_including_point(base_extent, lon_plot: float, lat: float, pad_deg: float = 4.0):
    west, east, south, north = base_extent
    west = min(west, lon_plot - pad_deg)
    east = max(east, lon_plot + pad_deg)
    south = max(-89.5, min(south, lat - pad_deg))
    north = min(89.5, max(north, lat + pad_deg))
    return (west, east, south, north)


def _plot_track_overlay(
    out_path: str,
    *,
    case_cfg,
    baseline_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
    selected_regions: dict | None = None,
    best_df: pd.DataFrame | None = None,
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    region_keys = ("hotspot", "low_near", "remote")
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18.0, 6.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    show_df = perturb_df[perturb_df["amplitude_sigma"].isin([-1.0, 1.0])]

    for ax, region_key in zip(axes, region_keys):
        style = REGION_STYLES[region_key]
        payload = (selected_regions or {}).get(region_key)

        if payload is not None:
            extent = _extent_including_point(
                case_cfg.map_extent,
                payload["center_lon_plot"],
                payload["center_lat"],
            )
        else:
            extent = case_cfg.map_extent
        # Set extent BEFORE adding cartopy features so they render correctly.
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        _plot_base_map_no_extent(ax)

        if best_df is not None:
            plot_best_track(
                ax, best_df,
                start=case_cfg.init_time,
                end=pd.to_datetime(baseline_df["valid_time"]).max(),
                full_context=False,
                linewidth=1.6,
            )
        _plot_track(
            ax, baseline_df,
            color="black", linewidth=2.0, linestyle="-", label="Baseline",
        )

        region_df = show_df[show_df["region_key"] == region_key]
        for amplitude in (-1.0, 1.0):
            sub = region_df[region_df["amplitude_sigma"] == amplitude]
            if sub.empty:
                continue
            _plot_track(
                ax,
                sub.sort_values("step"),
                color=style["color"],
                linewidth=1.6,
                linestyle="--" if amplitude < 0 else "-",
                label=f"{style['label']} {amplitude:+.0f}σ",
            )

        if payload is not None:
            ax.scatter(
                [payload["center_lon_plot"]],
                [payload["center_lat"]],
                transform=ccrs.PlateCarree(),
                marker="x",
                color=style["color"],
                s=140,
                linewidths=2.6,
                zorder=6,
                label=f"{style['label']} perturbation center",
            )

        ax.set_title(style["label"], fontsize=11)
        ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    fig.suptitle(f"{case_cfg.name}: baseline and perturbed tracks", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_displacement_timeseries(out_path: str, *, response_df: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for region_key in ("hotspot", "low_near", "remote"):
        style = REGION_STYLES[region_key]
        for amplitude in sorted(response_df["amplitude_sigma"].unique()):
            sub = response_df[
                (response_df["region_key"] == region_key)
                & (response_df["amplitude_sigma"] == amplitude)
            ].sort_values("lead_hours")
            if sub.empty:
                continue
            ax.plot(
                sub["lead_hours"],
                sub["track_shift_km"],
                color=style["color"],
                linestyle="--" if amplitude < 0 else "-",
                linewidth=1.8,
                marker="o",
                markersize=3.2,
                label=f"{style['label']} {amplitude:+.0f}σ",
            )
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Track displacement from baseline (km)")
    ax.set_title("Track response to localized initial ZWD perturbations")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_response_timeseries(out_path: str, *, response_df: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    metrics = [
        ("delta_min_msl_hpa", "Δ minimum MSL (hPa)"),
        ("delta_max_wind10_ms", "Δ max 10 m wind (m s$^{-1}$)"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        for region_key in ("hotspot", "low_near", "remote"):
            style = REGION_STYLES[region_key]
            for amplitude in sorted(response_df["amplitude_sigma"].unique()):
                sub = response_df[
                    (response_df["region_key"] == region_key)
                    & (response_df["amplitude_sigma"] == amplitude)
                ].sort_values("lead_hours")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lead_hours"],
                    sub[metric],
                    color=style["color"],
                    linestyle="--" if amplitude < 0 else "-",
                    linewidth=1.8,
                    marker="o",
                    markersize=3.2,
                    label=f"{style['label']} {amplitude:+.0f}σ",
                )
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_title("Intensity response relative to the baseline forecast")
    axes[-1].set_xlabel("Lead time (h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_response_summary_heatmap(out_path: str, *, summary_df: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat_cols = [
        ("max_track_shift_km", "Max track shift (km)"),
        ("max_abs_delta_min_msl_hpa", "Max |Δ min MSL| (hPa)"),
        ("max_abs_delta_max_wind10_ms", "Max |Δ max wind| (m s$^{-1}$)"),
    ]
    row_labels = [f"{row.region_label} {row.amplitude_sigma:+.0f}σ" for row in summary_df.itertuples()]
    data = np.stack([summary_df[col].to_numpy(dtype=float) for col, _ in heat_cols], axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(row_labels) + 2.0))
    mesh = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(heat_cols)), labels=[label for _, label in heat_cols], rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Peak perturbation response summary")
    fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.03)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_stamp_grid(
    out_path: str,
    *,
    case_cfg,
    baseline_df: pd.DataFrame,
    baseline_snapshots: dict[int, dict[str, np.ndarray]],
    comparison_runs: dict[str, tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]],
    stamp_steps: list[int],
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_items = [("Baseline", baseline_df, baseline_snapshots), *[(label, df, snaps) for label, (df, snaps) in comparison_runs.items()]]
    fig, axes = plt.subplots(
        len(row_items),
        len(stamp_steps),
        figsize=(4.4 * len(stamp_steps), 3.7 * len(row_items)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    for row_idx, (row_label, df, snapshots_by_step) in enumerate(row_items):
        for col_idx, step_idx in enumerate(stamp_steps):
            ax = axes[row_idx, col_idx]
            snap = snapshots_by_step[step_idx]
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
                vmin=920.0,
                vmax=1015.0,
            )
            lead_hours = int(round(float(df.loc[df["step"] == step_idx, "lead_hours"].iloc[0])))
            _plot_track(
                ax,
                df[df["step"] <= step_idx].sort_values("step"),
                color="black",
                linewidth=1.5,
                linestyle="-",
                label=row_label,
            )
            if row_idx == 0:
                ax.set_title(f"+{lead_hours}h")
            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    row_label,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )
    cb = fig.colorbar(mesh, ax=axes, shrink=0.92, pad=0.02)
    cb.set_label("MSL (hPa)")
    fig.suptitle(f"{case_cfg.name}: baseline vs +1σ perturbation stamp panels", fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    case_cfg, case_data = load_tc_case(args.case)
    study_id = format_case_study_id(case_cfg)
    study_root = os.path.join(args.output_dir, study_id)
    out_dir = os.path.join(study_root, "track_perturbation")
    ensure_dir(out_dir)

    if args.selection_json is None:
        args.selection_json = os.path.join(study_root, "storm_centered_xai", "selected_regions.json")
    if not os.path.exists(args.selection_json):
        raise FileNotFoundError(
            f"Selection file not found: {args.selection_json}. "
            "Run storm_centered_xai.py first or pass --selection-json."
        )

    selected_regions = _load_selected_regions(args.selection_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    timing = validate_rollout_cadence(
        model,
        case_data,
        device,
        expected_step_hours=args.expected_step_hours,
    )
    write_json(os.path.join(out_dir, "timing_check.json"), timing)
    if not timing["passes"] and not args.allow_step_mismatch:
        raise RuntimeError(
            "Model rollout cadence does not match the requested 6h setup. "
            "See timing_check.json."
        )
    step_hours = float(timing["observed_step_hours"])

    positive_amp = min((amp for amp in args.amplitudes if amp > 0), default=None)
    max_available_lead = int(round(args.rollout_steps * step_hours))
    stamp_steps = {
        lead_hours_to_steps(lead_h, step_hours)
        for lead_h in DEFAULT_STAMP_LEADS_HOURS
        if lead_h <= max_available_lead
    }

    baseline_df, baseline_snapshots = compute_baseline_rollout(
        model,
        case_cfg,
        case_data,
        device,
        steps=args.rollout_steps,
        snapshot_steps=stamp_steps if positive_amp is not None else None,
    )
    baseline_df.to_csv(os.path.join(out_dir, "baseline_track.csv"), index=False)

    perturb_rows = []
    for region_key in ("hotspot", "low_near", "remote"):
        region_payload = selected_regions[region_key]
        for amplitude in args.amplitudes:
            df, _ = run_perturbed_rollout(
                model,
                case_cfg,
                case_data,
                device,
                steps=args.rollout_steps,
                amplitude_sigma=float(amplitude),
                region_payload=region_payload,
            )
            merged = df.merge(
                baseline_df[["step", "lead_hours", "center_lat", "center_lon", "min_msl_hpa", "max_wind10_ms"]],
                on=["step", "lead_hours"],
                suffixes=("", "_baseline"),
            ).sort_values("step")
            merged["region_key"] = region_key
            merged["region_label"] = REGION_STYLES[region_key]["label"]
            merged["track_shift_km"] = great_circle_km(
                merged["center_lat"].to_numpy(dtype=float),
                merged["center_lon"].to_numpy(dtype=float),
                merged["center_lat_baseline"].to_numpy(dtype=float),
                merged["center_lon_baseline"].to_numpy(dtype=float),
            )
            merged["delta_min_msl_hpa"] = merged["min_msl_hpa"] - merged["min_msl_hpa_baseline"]
            merged["delta_max_wind10_ms"] = merged["max_wind10_ms"] - merged["max_wind10_ms_baseline"]
            perturb_rows.append(merged)

    perturb_df = pd.concat(perturb_rows, ignore_index=True)
    perturb_df.to_csv(os.path.join(out_dir, "perturbation_responses.csv"), index=False)

    summary_df = (
        perturb_df.groupby(["region_key", "region_label", "amplitude_sigma"], as_index=False)
        .agg(
            max_track_shift_km=("track_shift_km", "max"),
            max_abs_delta_min_msl_hpa=("delta_min_msl_hpa", lambda s: float(np.max(np.abs(s)))),
            max_abs_delta_max_wind10_ms=("delta_max_wind10_ms", lambda s: float(np.max(np.abs(s)))),
        )
        .sort_values(["region_key", "amplitude_sigma"])
    )
    summary_df.to_csv(os.path.join(out_dir, "response_summary.csv"), index=False)

    _plot_track_overlay(
        os.path.join(out_dir, "track_overlay.png"),
        case_cfg=case_cfg,
        baseline_df=baseline_df,
        perturb_df=perturb_df,
        selected_regions=selected_regions,
        best_df=load_best_track(case_cfg),
    )
    _plot_displacement_timeseries(
        os.path.join(out_dir, "displacement_timeseries.png"),
        response_df=perturb_df,
    )
    _plot_response_timeseries(
        os.path.join(out_dir, "response_timeseries.png"),
        response_df=perturb_df,
    )
    _plot_response_summary_heatmap(
        os.path.join(out_dir, "response_summary_heatmap.png"),
        summary_df=summary_df,
    )

    if positive_amp is not None:
        comparison_runs = {}
        for region_key in ("hotspot", "low_near", "remote"):
            df, snapshots = run_perturbed_rollout(
                model,
                case_cfg,
                case_data,
                device,
                steps=args.rollout_steps,
                amplitude_sigma=float(positive_amp),
                region_payload=selected_regions[region_key],
                snapshot_steps=stamp_steps,
            )
            comparison_runs[f"{REGION_STYLES[region_key]['label']} +{positive_amp:.0f}σ"] = (df, snapshots)

        if stamp_steps:
            _plot_stamp_grid(
                os.path.join(out_dir, "storm_stamp_panels.png"),
                case_cfg=case_cfg,
                baseline_df=baseline_df,
                baseline_snapshots=baseline_snapshots,
                comparison_runs=comparison_runs,
                stamp_steps=sorted(stamp_steps),
            )

    write_json(
        os.path.join(out_dir, "summary.json"),
        {
            "case": case_cfg.slug,
            "study_id": study_id,
            "selection_json": args.selection_json,
            "step_hours": step_hours,
            "rollout_steps": int(args.rollout_steps),
            "amplitudes": [float(x) for x in args.amplitudes],
            "selected_regions": selected_regions,
        },
    )
    print(f"Track perturbation outputs written to {out_dir}")


if __name__ == "__main__":
    main()
