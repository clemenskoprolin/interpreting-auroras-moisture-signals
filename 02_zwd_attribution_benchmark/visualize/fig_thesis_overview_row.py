"""Thesis figure: causal ground truth, saliency and IG side by side in one row.

Same content as `visualize_masking.py --overview --overview-exclude-contrastive`,
but laid out as a single 1x3 row with compact colorbars, so it fits a thesis
page without the tall two-row block.

Run with the project's xarray venv:
  source .venv/bin/activate
  python 02_zwd_attribution_benchmark/visualize/fig_thesis_overview_row.py \
      --results-dir results/zwd_attribution_benchmark/36h_point \
      --case ticino_2020042012_strong__point --scale local \
      --out results/img/36h_visual_results_thesis.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from searchlight_tasks import TARGETS, SCALES
from visualize_masking import (
    ZWD_ZARR, _GT_LABELS, _METHOD_LABELS, _add_basemap, _draw_target_marker,
    _imshow_on, regional_slice, smooth_global,
)

# Colorbar geometry — the whole point of this variant.
CB_SHRINK = 0.62      # fraction of the axes height the colorbar spans
CB_ASPECT = 26        # length / width; higher = thinner
CB_PAD = 0.02
CB_TICKLABEL_SIZE = 7
CB_LABEL_SIZE = 8


def _colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, shrink=CB_SHRINK, aspect=CB_ASPECT,
                      pad=CB_PAD, extend="both")
    cb.set_label(label, fontsize=CB_LABEL_SIZE)
    cb.ax.tick_params(labelsize=CB_TICKLABEL_SIZE)
    cb.ax.yaxis.get_offset_text().set_fontsize(CB_TICKLABEL_SIZE)
    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--case", required=True,
                    help="per_case subdirectory name, e.g. ticino_2020042012_strong__point")
    ap.add_argument("--scale", default="local", choices=("local", "synoptic"))
    ap.add_argument("--methods", nargs="+", default=["saliency", "ig"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-suptitle", action="store_true",
                    help="Omit the title (thesis captions usually carry it).")
    args = ap.parse_args()

    scale_dir = os.path.join(args.results_dir, "per_case", args.case, args.scale)
    if not os.path.isdir(scale_dir):
        sys.exit(f"No such case/scale directory: {scale_dir}")

    is_point = args.case.endswith("__point")
    base_case = args.case[:-7] if is_point else args.case
    target_mode = "point" if is_point else "box"
    target_obj = next((TARGETS[loc] for loc in TARGETS
                       if base_case.startswith(loc + "_")), None)
    if target_obj is None:
        sys.exit(f"Cannot infer target location from {base_case}")

    ds = xr.open_zarr(ZWD_ZARR)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    extent = target_obj.map_extent
    _, lat_z, lon_z = regional_slice(
        np.zeros((lat.shape[0], lon.shape[0])), lat, lon, extent)
    lon_plot_z = np.where(lon_z > 180, lon_z - 360, lon_z)
    grid_lon, grid_lat = np.meshgrid(lon_plot_z, lat_z)

    n_cols = 1 + len(args.methods)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4.6 * n_cols, 3.4),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        constrained_layout=True,
    )

    # ---- panel 1: causal ground truth ---------------------------------------
    gt = json.load(open(os.path.join(scale_dir, "ground_truth.json")))
    near = np.array([m["role"] == "near" for m in gt["masks"]])
    S = np.asarray(gt["S"], dtype=np.float64)[near]
    clats = np.array([m["center_lat"] for m in gt["masks"]])[near]
    clons = np.array([m["center_lon"] for m in gt["masks"]])[near]
    clons_plot = np.where(clons > 180, clons - 360, clons)

    S_grid = griddata(np.column_stack([clons_plot, clats]), S,
                      (grid_lon, grid_lat), method="linear")
    valid = S_grid[~np.isnan(S_grid)]
    S_vmax = float(np.nanpercentile(np.abs(valid), 97)) if valid.size else 1.0

    ax = axes[0]
    _add_basemap(ax, extent)
    _draw_target_marker(ax, target_obj, target_mode, gt_data=gt)
    im = _imshow_on(ax, S_grid, lat_z, lon_plot_z, cmap="RdBu_r",
                    vmin=-S_vmax, vmax=S_vmax, interpolation="nearest")
    ax.scatter(clons_plot, clats,
               s=14 if SCALES[args.scale].sigma_deg >= 6.0 else 8,
               facecolors="none", edgecolors="black", linewidths=0.3,
               alpha=0.5, transform=ccrs.PlateCarree(), zorder=6)
    ax.set_title(_GT_LABELS.get("ground_truth", "Plain GT"), fontsize=9)
    _colorbar(fig, im, ax, "S_r [g kg⁻¹]")

    # ---- panels 2..n: attribution maps ---------------------------------------
    for ax, method in zip(axes[1:], args.methods):
        path = os.path.join(scale_dir, f"{method}_attr.npy")
        if not os.path.isfile(path):
            sys.exit(f"Missing attribution map: {path}")
        # Smooth globally (wrapping in longitude) *before* cropping, otherwise a
        # prime-meridian-crossing extent picks up a seam at 0 degrees.
        attr, _, _ = regional_slice(
            smooth_global(np.load(path), sigma=3.0), lat, lon, extent)
        vmax = float(np.nanpercentile(np.abs(attr), 97)) or 1.0

        _add_basemap(ax, extent)
        _draw_target_marker(ax, target_obj, target_mode, gt_data=gt)
        im = _imshow_on(ax, attr, lat_z, lon_plot_z, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_title(_METHOD_LABELS.get(method, method), fontsize=9)
        _colorbar(fig, im, ax, "attribution")

    if not args.no_suptitle:
        m = re.search(r"_(\d{10})_", base_case)
        time_str = ""
        if m:
            s = m.group(1)
            t = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), 0)
            time_str = f"  ·  {t:%Y-%m-%d %H:%M UTC}"
        fig.suptitle(
            f"case: {base_case}  ·  scale: {args.scale}  ·  "
            f"target: {target_mode}{time_str}", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
