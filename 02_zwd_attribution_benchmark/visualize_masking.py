"""
Educational figure for the ZWD Searchlight benchmark: show what "masking" means
and how saliency relates to the causal ground truth.

Panels (3 rows x 3 cols):

  Row 1 — perturbation operator (regional zoom over the target):
    (a) unperturbed ZWD at t1
    (b) ZWD + 1·sigma_zwd·M_local   (sigma_deg = 2.5)
    (c) ZWD + 1·sigma_zwd·M_synoptic (sigma_deg = 6.0)

  Row 2 — the mask itself and the resulting delta:
    (a) Gaussian mask M_local  (peak = 1)
    (b) Gaussian mask M_synoptic
    (c) delta fields (scale bar in mm of ZWD)

  Row 3 — what saliency computes:
    (a) ZWD saliency map at t1 (global → regional crop), target box overlaid
    (b) pooled |A| vs G for local near-masks
    (c) pooled |A| vs G for synoptic near-masks

Case: ticino_2020042012_strong. Target = Ticino box (45.5-47 N, 7.5-10 E).
Illustrative mask center = (46.23 N, 8.68 E) — the top-G local mask.

--ground-truth mode: generates two separate figures (box and point targets)
  showing the signed causal ground-truth response S_r for all three locations
  (ticino / california / japan) at both scales (local / synoptic).
  Output files: zwd_searchlight_ground_truth_box.png
                zwd_searchlight_ground_truth_point.png

Run from the repository root in the project environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from searchlight_tasks import (
    TARGETS, SCALES, MaskSpec, gaussian_mask, cos_lat_weights
)

CASE_ID = "ticino_2020042012_strong"
INIT_TIME = datetime(2020, 4, 20, 12, 0)
TARGET = TARGETS["ticino"]

# Illustrative mask center (top-G local near mask; same (lat, lon) used for
# both scales so the two bumps are visually comparable).
ILLUSTRATION_CLAT = 46.23
ILLUSTRATION_CLON = 8.68
MAGNITUDE = 1.0  # sigma_zwd

# Corrected-precision 6 h box results used by the paper figures.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "zwd_attribution_benchmark", "6h_box",
)
OUT_PATH = os.path.join(RESULTS_ROOT, "viz", "zwd_searchlight_masking_explained.png")

# Strong case for each location used in the default ground-truth overview figures.
LOCATION_CASES: dict[str, tuple[str, datetime]] = {
    "ticino":     ("ticino_2020042012_strong",      datetime(2020,  4, 20, 12, 0)),
    "california": ("california_2021102412_strong",  datetime(2021, 10, 24, 12, 0)),
    "japan":      ("japan_2020021612_strong",        datetime(2020,  2, 16, 12, 0)),
}

GROUND_TRUTH_CASE_PRESETS: dict[str, dict[str, tuple[str, datetime]]] = {
    "eventdates": LOCATION_CASES,
    "baseline_20200101": {
        "ticino":     ("ticino_2020010122_strong",     datetime(2020, 1, 1, 22, 0)),
        "california": ("california_2020010122_strong", datetime(2020, 1, 1, 22, 0)),
        "japan":      ("japan_2020010122_strong",      datetime(2020, 1, 1, 22, 0)),
    },
    "baseline_20210101": {
        "ticino":     ("ticino_2021010122_secondary",     datetime(2021, 1, 1, 22, 0)),
        "california": ("california_2021010122_secondary", datetime(2021, 1, 1, 22, 0)),
        "japan":      ("japan_2021010122_secondary",      datetime(2021, 1, 1, 22, 0)),
    },
}

ZWD_ZARR = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)
NORM_STATS = os.environ.get(
    "AURORA_NORMALIZATION_STATS",
    os.path.join(_ROOT, "config", "normalization_stats_1979_2021.json"),
)


def load_zwd_t1(init_time):
    ds = xr.open_zarr(ZWD_ZARR)
    zwd_t1 = ds["zenith_wet_delay"].sel(time=pd.Timestamp(init_time)).values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    return zwd_t1.astype(np.float32), lat, lon


def load_zwd_norm():
    with open(NORM_STATS) as f:
        s = json.load(f)
    return float(s["locations"]["zwd"]), float(s["scales"]["zwd"])


def make_mask(scale_name, lat, lon, clat=ILLUSTRATION_CLAT, clon=ILLUSTRATION_CLON):
    sigma = SCALES[scale_name].sigma_deg
    spec = MaskSpec(
        scale=scale_name, role="near",
        center_lat=clat, center_lon=clon % 360.0, mask_id=-1,
    )
    return gaussian_mask(spec, sigma, lat, lon), sigma


def perturb(zwd_hw, mask_hw, zwd_loc, zwd_scale, magnitude=MAGNITUDE, sign=+1.0):
    delta = sign * magnitude * zwd_scale * mask_hw.astype(np.float32)
    out = zwd_hw + delta
    lo = zwd_loc - 4.0 * zwd_scale
    hi = zwd_loc + 4.0 * zwd_scale
    return np.clip(out, lo, hi), delta


def regional_slice(arr_hw, lat, lon, extent):
    """Crop (H,W) array to a lon/lat extent (lon_w, lon_e, lat_s, lat_n).
    Handles ERA5 descending lat and 0..360 lon. Assumes extent doesn't wrap."""
    lon_w, lon_e, lat_s, lat_n = extent
    lon_w_wrap = lon_w % 360.0
    lon_e_wrap = lon_e % 360.0

    if lat[0] > lat[-1]:
        lat_mask = (lat <= lat_n) & (lat >= lat_s)
    else:
        lat_mask = (lat >= lat_s) & (lat <= lat_n)
    lat_idx = np.where(lat_mask)[0]

    if lon_w_wrap <= lon_e_wrap:
        lon_idx = np.where((lon >= lon_w_wrap) & (lon <= lon_e_wrap))[0]
    else:
        lon_idx = np.where((lon >= lon_w_wrap) | (lon <= lon_e_wrap))[0]

    sub = arr_hw[np.ix_(lat_idx, lon_idx)]
    return sub, lat[lat_idx], lon[lon_idx]


def smooth_global(arr_hw, sigma=3.0):
    """Gaussian-smooth a global (H,W) field. Must be called *before* cropping.

    `regional_slice` returns columns in raw index order, so for an extent that
    crosses the prime meridian (e.g. Ticino, -15..45) the two longitude blocks
    end up at opposite ends of the cropped array. Smoothing the crop therefore
    applies a `reflect` boundary at lon=0 instead of joining the two halves,
    which leaves a visible vertical seam at 0° in the plotted map (and bleeds
    data across the false internal junction at the other edge). Smoothing the
    global field with `wrap` in longitude first avoids both.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr_hw, sigma=sigma, mode=("nearest", "wrap"))


def _add_basemap(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeat.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeat.COASTLINE, edgecolor="gray", linewidth=0.5)
    ax.add_feature(cfeat.BORDERS, edgecolor="gray", linewidth=0.3, linestyle=":")


def _add_target_box(ax, color="black"):
    w = TARGET.box_lon[0]; e = TARGET.box_lon[1]
    if w > 180: w -= 360
    if e > 180: e -= 360
    s = TARGET.box_lat[0]; n = TARGET.box_lat[1]
    ax.add_patch(Rectangle(
        (w, s), e - w, n - s,
        fill=False, edgecolor=color, linewidth=1.8,
        transform=ccrs.PlateCarree(), zorder=5,
    ))


def _imshow_on(ax, arr, lat, lon, **kw):
    lon_plot = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_plot)
    arr_p = arr[:, order]
    lon_p = lon_plot[order]
    # A per-pixel alpha map must follow the same column permutation as the data.
    alpha = kw.get("alpha")
    if isinstance(alpha, np.ndarray) and alpha.ndim == 2:
        kw["alpha"] = alpha[:, order]
    extent = (lon_p.min(), lon_p.max(), lat.min(), lat.max())
    origin = "upper" if lat[0] > lat[-1] else "lower"
    return ax.imshow(
        arr_p, extent=extent, origin=origin,
        transform=ccrs.PlateCarree(), **kw,
    )


def load_pooled_vs_gt(scale):
    per = os.path.join(RESULTS_ROOT, "per_case", CASE_ID, scale)
    amag = np.load(os.path.join(per, "saliency_pooled_Amag.npy"))
    gt = json.load(open(os.path.join(per, "ground_truth.json")))
    G = np.asarray(gt["G"], dtype=np.float64)
    S = np.asarray(gt["S"], dtype=np.float64)
    masks = gt["masks"]
    near = np.array([m["role"] == "near" for m in masks])
    clats = np.array([m["center_lat"] for m in masks])[near]
    clons = np.array([m["center_lon"] for m in masks])[near]
    return amag[near], G[near], S[near], clats, clons


_SCATTER_GRADIENT_ORDER = [
    "saliency", "smoothgrad",
    "ig",
    "contrastive_saliency_global", "contrastive_saliency_remote",
    "contrastive_ig_global",       "contrastive_ig_remote",
]
_SCATTER_OCCLUSION_ORDER = ["rise_200masks", "rise_1500masks", "rise", "vit_cx"]

_SCATTER_STYLE: dict[str, dict] = {
    "saliency":                    dict(color="tab:blue",   marker="o",  label="saliency"),
    "smoothgrad":                  dict(color="steelblue",  marker="s",  label="SmoothGrad"),
    "ig":                          dict(color="tab:orange", marker="^",  label="IG"),
    "contrastive_saliency_global": dict(color="tab:green",  marker="D",  label="contr. saliency (global)"),
    "contrastive_saliency_remote": dict(color="limegreen",  marker="*",  label="contr. saliency (remote)"),
    "contrastive_ig_global":       dict(color="tab:red",    marker="D",  label="contr. IG (global)"),
    "contrastive_ig_remote":       dict(color="tomato",     marker="*",  label="contr. IG (remote)"),
    "rise_200masks":               dict(color="tab:purple", marker="o",  label="RISE (200 masks)"),
    "rise_1500masks":              dict(color="indigo",     marker="s",  label="RISE (1500 masks)"),
    "rise":                        dict(color="tab:purple", marker="o",  label="RISE"),
    "vit_cx":                      dict(color="tab:pink",   marker="^",  label="ViT-CX"),
}


def plot_scatter_all(
    results_root: str,
    case_id: str,
    scale: str,
    output_dir: str | None = None,
) -> None:
    """Two-panel scatter of pooled attribution magnitude vs causal GT for all methods.

    Left panel: gradient-based methods.
    Right panel: occlusion-based methods (RISE, ViT-CX).
    Each method's scores are normalised independently to [0, 1] so rankings
    are directly comparable across methods with different absolute scales.
    All methods are scored against the plain ground-truth (ground_truth.json).
    """
    import re

    if output_dir is None:
        output_dir = os.path.join(results_root, "viz")
    os.makedirs(output_dir, exist_ok=True)

    per_dir = os.path.join(results_root, "per_case", case_id, scale)
    gt = json.load(open(os.path.join(per_dir, "ground_truth.json")))
    near = np.array([m["role"] == "near" for m in gt["masks"]])
    G_near = np.asarray(gt["G"], dtype=np.float64)[near]

    all_amag: dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(per_dir)):
        if fname.endswith("_pooled_Amag.npy"):
            method = fname[: -len("_pooled_Amag.npy")]
            all_amag[method] = np.load(os.path.join(per_dir, fname))[near]

    grad_methods = [m for m in _SCATTER_GRADIENT_ORDER if m in all_amag]
    occ_methods  = [m for m in _SCATTER_OCCLUSION_ORDER  if m in all_amag]
    known = set(grad_methods + occ_methods)
    for m in sorted(all_amag):
        if m not in known:
            if any(tag in m for tag in ("rise", "vit")):
                occ_methods.append(m)
            else:
                grad_methods.append(m)

    panels = []
    if grad_methods:
        panels.append(("Gradient-based methods", grad_methods))
    if occ_methods:
        panels.append(("Occlusion-based methods", occ_methods))
    if not panels:
        print(f"No Amag files found in {per_dir}, skipping.")
        return

    def _n01(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.5),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    sigma_deg = SCALES[scale].sigma_deg
    for ax, (panel_title, methods) in zip(axes, panels):
        for method in methods:
            amag = all_amag[method]
            rho, _ = spearmanr(amag, G_near)
            style = _SCATTER_STYLE.get(method, dict(color="gray", marker="o", label=method))
            ax.scatter(
                G_near, _n01(amag),
                s=30, alpha=0.82, edgecolor="black", linewidth=0.3,
                color=style["color"], marker=style["marker"],
                label=f"{style['label']}  ρ={rho:.2f}",
            )
        ax.set_xlabel("|G_r|  ground-truth causal effect  [g kg⁻¹]", fontsize=10)
        ax.set_ylabel("pooled |A_r|  (normalised 0–1 per method)", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(alpha=0.3)

    m = re.search(r'_(\d{10})_', case_id)
    time_str = ""
    if m:
        s = m.group(1)
        t = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), 0)
        time_str = f"  ·  {t:%Y-%m-%d %H:%M UTC}"

    is_point = case_id.endswith("__point")
    target_mode_str = "point" if is_point else "box"
    fig.suptitle(
        f"Attribution magnitude vs. causal ground truth — {scale} masks "
        f"(σ={sigma_deg}°)\n"
        f"case: {case_id}{time_str}  ·  target: {target_mode_str}",
        fontsize=10,
    )

    out = os.path.join(output_dir, f"zwd_scatter_all_{case_id}_{scale}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _draw_target_marker(ax, target, target_mode: str, gt_data: dict | None = None, color: str = "black") -> None:
    """Draw target box or star on a cartopy axes depending on target_mode."""
    if target_mode == "box":
        w, e = target.box_lon[0], target.box_lon[1]
        s, n = target.box_lat[0], target.box_lat[1]
        if w > 180: w -= 360
        if e > 180: e -= 360
        ax.add_patch(Rectangle(
            (w, s), e - w, n - s,
            fill=False, edgecolor=color, linewidth=1.8,
            transform=ccrs.PlateCarree(), zorder=5,
        ))
    else:
        if gt_data is not None and "target_meta" in gt_data:
            clon = gt_data["target_meta"]["lon"]
            clat = gt_data["target_meta"]["lat"]
        else:
            clon = 0.5 * (target.box_lon[0] + target.box_lon[1])
            clat = 0.5 * (target.box_lat[0] + target.box_lat[1])
        if clon > 180:
            clon -= 360
        ax.plot(clon, clat, "*", color=color, markersize=10,
                transform=ccrs.PlateCarree(), zorder=6)


_GT_LABELS: dict[str, str] = {
    "ground_truth":        "Plain GT\n(f_tgt vs perturbed)",
    "ground_truth_global": "Contrastive GT\n(global reference)",
    "ground_truth_remote": "Contrastive GT\n(remote reference)",
}
_METHOD_LABELS: dict[str, str] = {
    "saliency":                    "Saliency\n∂f/∂ZWD",
    "ig":                          "Integrated Gradients",
    "contrastive_saliency_global": "Contr. Saliency\n(global)",
    "contrastive_saliency_remote": "Contr. Saliency\n(remote)",
    "contrastive_ig_global":       "Contr. IG\n(global)",
    "contrastive_ig_remote":       "Contr. IG\n(remote)",
}
_METHOD_ORDER = list(_METHOD_LABELS.keys())
_GT_ORDER = list(_GT_LABELS.keys())


def plot_gt_xai_overview(
    results_dir: str,
    lat: np.ndarray,
    lon: np.ndarray,
    output_dir: str | None = None,
    exclude_contrastive: bool = False,
) -> None:
    """Combined GT + XAI attribution overview figure for every case/scale in results_dir.

    Generates one PNG per (case_id, scale), with GT maps in the top row and XAI
    attribution maps in the bottom row, aligned over the same regional extent.
    """
    import re
    from scipy.interpolate import griddata
    from matplotlib.gridspec import GridSpec

    if output_dir is None:
        output_dir = os.path.join(results_dir, "viz")
    os.makedirs(output_dir, exist_ok=True)

    per_case_dir = os.path.join(results_dir, "per_case")
    if not os.path.isdir(per_case_dir):
        print(f"No per_case directory found in {results_dir}")
        return

    for case_id in sorted(os.listdir(per_case_dir)):
        case_path = os.path.join(per_case_dir, case_id)
        if not os.path.isdir(case_path):
            continue

        is_point = case_id.endswith("__point")
        base_case_id = case_id[:-7] if is_point else case_id
        target_mode = "point" if is_point else "box"

        target_loc, target_obj = None, None
        for loc in TARGETS:
            if base_case_id.startswith(loc + "_") or base_case_id == loc:
                target_loc, target_obj = loc, TARGETS[loc]
                break
        if target_obj is None:
            print(f"Cannot infer target for {case_id}, skipping.")
            continue

        m = re.search(r'_(\d{10})_', base_case_id)
        case_time = None
        if m:
            s = m.group(1)
            case_time = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), 0)

        for scale in sorted(os.listdir(case_path)):
            scale_path = os.path.join(case_path, scale)
            if not os.path.isdir(scale_path) or scale not in SCALES:
                continue

            gt_names = _GT_ORDER if not exclude_contrastive else ["ground_truth"]
            available_gt = [n for n in gt_names
                            if os.path.isfile(os.path.join(scale_path, f"{n}.json"))]
            all_attr = {
                fname[:-9]: os.path.join(scale_path, fname)
                for fname in os.listdir(scale_path)
                if fname.endswith("_attr.npy")
            }
            if exclude_contrastive:
                all_attr = {k: v for k, v in all_attr.items() if not k.startswith("contrastive_")}
            available_methods = [m for m in _METHOD_ORDER if m in all_attr]
            available_methods += sorted(k for k in all_attr if k not in _METHOD_ORDER)

            if not available_gt and not available_methods:
                print(f"Nothing to plot for {case_id}/{scale}, skipping.")
                continue

            extent = target_obj.map_extent
            _, lat_z, lon_z = regional_slice(
                np.zeros((lat.shape[0], lon.shape[0])), lat, lon, extent
            )
            lon_plot_z = np.where(lon_z > 180, lon_z - 360, lon_z)
            grid_lon, grid_lat = np.meshgrid(lon_plot_z, lat_z)
            sigma_deg = SCALES[scale].sigma_deg

            n_gt  = len(available_gt)
            n_xia = len(available_methods)
            n_rows = (1 if n_gt else 0) + (1 if n_xia else 0)
            n_cols = max(n_gt, n_xia, 1)

            fig = plt.figure(figsize=(5.2 * n_cols, 5.5 * n_rows), constrained_layout=True)
            gs = GridSpec(n_rows, n_cols, figure=fig)

            row_idx = 0

            # ---- GT row ----
            if available_gt:
                gt_data_cache: dict[str, dict] = {}
                for col, gt_name in enumerate(available_gt):
                    gt_data = json.load(open(os.path.join(scale_path, f"{gt_name}.json")))
                    gt_data_cache[gt_name] = gt_data
                    near = np.array([mm["role"] == "near" for mm in gt_data["masks"]])
                    S = np.asarray(gt_data["S"], dtype=np.float64)[near]
                    clats = np.array([mm["center_lat"] for mm in gt_data["masks"]])[near]
                    clons = np.array([mm["center_lon"] for mm in gt_data["masks"]])[near]
                    clons_plot = np.where(clons > 180, clons - 360, clons)

                    S_grid = griddata(
                        np.column_stack([clons_plot, clats]),
                        S, (grid_lon, grid_lat), method="linear",
                    )
                    valid = S_grid[~np.isnan(S_grid)]
                    S_vmax = float(np.nanpercentile(np.abs(valid), 97)) if valid.size else 1.0

                    ax = fig.add_subplot(gs[row_idx, col], projection=ccrs.PlateCarree())
                    _add_basemap(ax, extent)
                    _draw_target_marker(ax, target_obj, target_mode, gt_data=gt_data)
                    im = _imshow_on(ax, S_grid, lat_z, lon_plot_z,
                                    cmap="RdBu_r", vmin=-S_vmax, vmax=S_vmax,
                                    interpolation="nearest")
                    ax.scatter(clons_plot, clats,
                               s=14 if sigma_deg >= 6.0 else 8,
                               facecolors="none", edgecolors="black",
                               linewidths=0.3, alpha=0.5,
                               transform=ccrs.PlateCarree(), zorder=6)
                    ax.set_title(_GT_LABELS.get(gt_name, gt_name), fontsize=9)
                    fig.colorbar(im, ax=ax, shrink=0.75, label="S_r [g kg⁻¹]", extend="both")
                    if col == 0:
                        ax.set_ylabel("Ground Truth", fontsize=9, labelpad=4)

                for col in range(n_gt, n_cols):
                    fig.add_subplot(gs[row_idx, col]).set_axis_off()
                row_idx += 1

            # ---- XAI row ----
            if available_methods:
                first_gt_data = next(iter(gt_data_cache.values())) if available_gt else None
                for col, method in enumerate(available_methods):
                    attr = np.load(all_attr[method])
                    attr_smooth, _, _ = regional_slice(
                        smooth_global(attr, sigma=3.0), lat, lon, extent
                    )
                    vmax = float(np.nanpercentile(np.abs(attr_smooth), 97)) or 1.0

                    ax = fig.add_subplot(gs[row_idx, col], projection=ccrs.PlateCarree())
                    _add_basemap(ax, extent)
                    _draw_target_marker(ax, target_obj, target_mode, gt_data=first_gt_data)
                    im = _imshow_on(ax, attr_smooth, lat_z, lon_plot_z,
                                    cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    ax.set_title(_METHOD_LABELS.get(method, method), fontsize=9)
                    fig.colorbar(im, ax=ax, shrink=0.75, label="attribution", extend="both")
                    if col == 0:
                        ax.set_ylabel("XAI Attribution", fontsize=9, labelpad=4)

                for col in range(n_xia, n_cols):
                    fig.add_subplot(gs[row_idx, col]).set_axis_off()

            time_str = f"  ·  {case_time:%Y-%m-%d %H:%M UTC}" if case_time else ""
            fig.suptitle(
                f"ZWD Searchlight — Ground Truth & XAI Attribution Overview\n"
                f"case: {base_case_id}  ·  scale: {scale}  ·  target: {target_mode}{time_str}",
                fontsize=11,
            )

            safe = case_id.replace("/", "_")
            out = os.path.join(output_dir, f"zwd_overview_{safe}_{scale}.png")
            plt.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")


def _load_gt_near(case_id: str, scale: str, results_root: str = RESULTS_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (G_near, S_near, clats_near, clons_near) for near masks, plain ground truth."""
    per = os.path.join(results_root, "per_case", case_id, scale)
    gt = json.load(open(os.path.join(per, "ground_truth.json")))
    near = np.array([m["role"] == "near" for m in gt["masks"]])
    G = np.asarray(gt["G"], dtype=np.float64)[near]
    S = np.asarray(gt["S"], dtype=np.float64)[near]
    clats = np.array([m["center_lat"] for m in gt["masks"]])[near]
    clons = np.array([m["center_lon"] for m in gt["masks"]])[near]
    return G, S, clats, clons


def _load_gt_near_point(case_id: str, scale: str, results_root: str = RESULTS_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Same as _load_gt_near but also returns target_meta for the point location."""
    per = os.path.join(results_root, "per_case", case_id + "__point", scale)
    gt = json.load(open(os.path.join(per, "ground_truth.json")))
    near = np.array([m["role"] == "near" for m in gt["masks"]])
    G = np.asarray(gt["G"], dtype=np.float64)[near]
    S = np.asarray(gt["S"], dtype=np.float64)[near]
    clats = np.array([m["center_lat"] for m in gt["masks"]])[near]
    clons = np.array([m["center_lon"] for m in gt["masks"]])[near]
    return G, S, clats, clons, gt["target_meta"]


def _gt_panel(fig, ax, S, clats, clons, lat_z, lon_plot_z, extent, sigma_deg: float,
              title: str, target_box=None, point_lonlat=None):
    """Draw one ground-truth signed panel with sharp interpolation between centers."""
    from scipy.interpolate import griddata

    clons_plot = np.where(clons > 180, clons - 360, clons)
    grid_lon, grid_lat = np.meshgrid(lon_plot_z, lat_z)
    S_grid = griddata(
        np.column_stack([clons_plot, clats]),
        S,
        (grid_lon, grid_lat),
        method="linear",
    )
    S_vmax = float(np.nanpercentile(np.abs(S_grid[~np.isnan(S_grid)]), 97))

    _add_basemap(ax, extent)

    if target_box is not None:
        w, e = target_box.box_lon[0], target_box.box_lon[1]
        if w > 180:
            w -= 360
        if e > 180:
            e -= 360
        s, n = target_box.box_lat[0], target_box.box_lat[1]
        ax.add_patch(Rectangle(
            (w, s), e - w, n - s,
            fill=False, edgecolor="black", linewidth=1.8,
            transform=ccrs.PlateCarree(), zorder=5,
        ))

    if point_lonlat is not None:
        px = point_lonlat[0] if point_lonlat[0] <= 180 else point_lonlat[0] - 360
        ax.plot(
            px, point_lonlat[1], "*",
            color="black", markersize=10,
            transform=ccrs.PlateCarree(), zorder=6
        )

    im = _imshow_on(
        ax, S_grid, lat_z, lon_plot_z,
        cmap="RdBu_r", vmin=-S_vmax, vmax=S_vmax,
        interpolation="nearest",
    )

    ax.scatter(
        clons_plot, clats,
        s=18 if sigma_deg >= 6.0 else 10,
        facecolors="none",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.75, label="S_r  [g kg⁻¹]", extend="both")


def plot_ground_truth(
    lat: np.ndarray,
    lon: np.ndarray,
    results_root: str = RESULTS_ROOT,
    location_cases: dict[str, tuple[str, datetime]] = LOCATION_CASES,
    output_dir: str | None = None,
    target_modes: tuple[str, ...] = ("box", "point"),
    locations: list[str] | None = None,
) -> None:
    """Generate ground-truth overview figures for all locations × scales × target modes."""
    if output_dir is None:
        output_dir = os.path.join(results_root, "viz")
    os.makedirs(output_dir, exist_ok=True)

    scales = ["local", "synoptic"]
    if locations is None:
        locations = ["ticino", "california", "japan"]
    scale_labels = {
        "local":    f"local masks   (σ={SCALES['local'].sigma_deg}°)",
        "synoptic": f"synoptic masks  (σ={SCALES['synoptic'].sigma_deg}°)",
    }
    loc_labels = {
        "ticino":     "Ticino / Southern Alps",
        "california": "N. California AR",
        "japan":      "Central Honshu",
    }

    for target_mode in target_modes:
        suffix = "__point" if target_mode == "point" else ""
        present_locs = [
            loc for loc in locations
            if loc in location_cases and os.path.isdir(
                os.path.join(
                    results_root, "per_case", location_cases[loc][0] + suffix
                )
            )
        ]
        if not present_locs:
            print(f"Skipping {target_mode} GT figure for {results_root}: "
                  f"no case directories found for any of {locations}.")
            continue

        present_scales = [
            s for s in scales
            if any(
                os.path.isfile(os.path.join(
                    results_root, "per_case",
                    location_cases[loc][0] + suffix, s, "ground_truth.json",
                )) for loc in present_locs
            )
        ]
        if not present_scales:
            print(f"Skipping {target_mode} GT figure for {results_root}: "
                  f"no ground_truth.json on disk for any requested scale.")
            continue

        n_rows = len(present_scales)
        n_cols = len(present_locs)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
            squeeze=False,
        )

        for row, scale in enumerate(present_scales):
            for col, loc in enumerate(present_locs):
                case_id, case_time = location_cases[loc]
                gt_path = os.path.join(
                    results_root, "per_case", case_id + suffix, scale,
                    "ground_truth.json",
                )
                if not os.path.isfile(gt_path):
                    axes[row, col].set_axis_off()
                    axes[row, col].set_title(
                        f"{loc} / {scale}: no GT on disk", fontsize=9,
                    )
                    continue
                target = TARGETS[loc]
                extent = target.map_extent
                _, lat_z, lon_z = regional_slice(
                    np.zeros((lat.shape[0], lon.shape[0])), lat, lon, extent
                )
                lon_plot_z = np.where(lon_z > 180, lon_z - 360, lon_z)

                panel_letter = chr(ord("a") + row * n_cols + col)
                sig = SCALES[scale].sigma_deg
                title = (
                    f"({panel_letter}) {loc_labels[loc]}\n"
                    f"{scale_labels[scale]}\n"
                    f"{case_time:%Y-%m-%d %H:%M UTC}"
                )

                ax = axes[row, col]

                if target_mode == "box":
                    _, S, clats, clons = _load_gt_near(case_id, scale, results_root=results_root)
                    _gt_panel(fig, ax, S, clats, clons, lat_z, lon_plot_z, extent,
                              sig, title, target_box=target)
                else:
                    _, S, clats, clons, meta = _load_gt_near_point(case_id, scale, results_root=results_root)
                    pt_lon = meta["lon"]
                    pt_lat = meta["lat"]
                    _gt_panel(fig, ax, S, clats, clons, lat_z, lon_plot_z, extent,
                              sig, title, point_lonlat=(pt_lon, pt_lat))

        row_labels = [scale_labels[s] for s in present_scales]
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=9)

        mode_title = "box-region target  (q@850 hPa regional mean)" if target_mode == "box" \
            else "point target  (q@850 hPa nearest grid point to box centre)"
        fig.suptitle(
            f"ZWD Searchlight — signed causal ground truth S_r\n{mode_title}",
            fontsize=12,
        )

        out = os.path.join(output_dir, f"zwd_searchlight_ground_truth_{target_mode}.png")
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=None,
        help="Override the results root for every mode (default: %(default)s). "
             "Mode-specific flags (--overview-dir, --scatter-dir, "
             "--gt-results-root) still take precedence when given.",
    )
    parser.add_argument("--ground-truth", action="store_true",
                        help="Generate ground-truth overview figures for all locations and scales")
    parser.add_argument(
        "--gt-results-root",
        default=None,
        help="Results root containing per_case/<case_id>/<scale>/ground_truth.json for GT figures",
    )
    parser.add_argument(
        "--gt-case-preset",
        choices=tuple(GROUND_TRUTH_CASE_PRESETS.keys()),
        default="eventdates",
        help="Which location→case mapping to use for GT figures",
    )
    parser.add_argument(
        "--gt-output-dir",
        default=None,
        help="Output directory for GT figures; defaults to <gt-results-root>/viz",
    )
    parser.add_argument(
        "--gt-target-modes",
        nargs="+",
        choices=("box", "point"),
        default=("box", "point"),
        help="Which GT target modes to render",
    )
    parser.add_argument(
        "--gt-locations",
        nargs="+",
        choices=("ticino", "california", "japan"),
        default=None,
        help="Subset of locations to render in the GT figure "
             "(default: all three). Locations with no on-disk GT are skipped "
             "automatically.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Generate a combined GT + XAI attribution overview figure for every "
             "case/scale found in --overview-dir. Top row shows ground-truth S_r "
             "maps (plain/global/remote), bottom row shows all computed attribution "
             "maps — both aligned over the same regional extent.",
    )
    parser.add_argument(
        "--overview-dir",
        default=None,
        help="Results root directory to scan for the overview figure "
             "(default: --results-root).",
    )
    parser.add_argument(
        "--overview-output-dir",
        default=None,
        help="Output directory for overview figures; defaults to <overview-dir>/viz.",
    )
    parser.add_argument(
        "--overview-exclude-contrastive",
        action="store_true",
        help="Hide contrastive methods from the XAI row and limit GT to plain ground_truth only.",
    )
    parser.add_argument(
        "--scatter-all",
        action="store_true",
        help="Generate attribution-magnitude-vs-GT scatter plots for all available "
             "methods in a results directory, split into gradient (left) and "
             "occlusion-based (right) panels.",
    )
    parser.add_argument(
        "--scatter-dir",
        default=None,
        help="Results root to load Amag files from (default: --results-root).",
    )
    parser.add_argument(
        "--scatter-case",
        default=CASE_ID,
        help="Case ID (per_case subdirectory) to use (default: %(default)s).",
    )
    parser.add_argument(
        "--scatter-scale",
        default="local",
        choices=("local", "synoptic"),
        help="Scale to use (default: %(default)s).",
    )
    parser.add_argument(
        "--scatter-output-dir",
        default=None,
        help="Output directory; defaults to <scatter-dir>/viz.",
    )
    args = parser.parse_args()

    global RESULTS_ROOT, OUT_PATH
    if args.results_root:
        RESULTS_ROOT = args.results_root
        OUT_PATH = os.path.join(RESULTS_ROOT, "viz",
                                "zwd_searchlight_masking_explained.png")
    args.scatter_dir = args.scatter_dir or RESULTS_ROOT
    args.overview_dir = args.overview_dir or RESULTS_ROOT
    args.gt_results_root = args.gt_results_root or RESULTS_ROOT

    if args.scatter_all:
        plot_scatter_all(
            results_root=args.scatter_dir,
            case_id=args.scatter_case,
            scale=args.scatter_scale,
            output_dir=args.scatter_output_dir,
        )
        return

    if args.overview:
        print("Loading lat/lon from zarr for overview figures…")
        ds = xr.open_zarr(ZWD_ZARR)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        plot_gt_xai_overview(
            results_dir=args.overview_dir,
            lat=lat,
            lon=lon,
            output_dir=args.overview_output_dir,
            exclude_contrastive=args.overview_exclude_contrastive,
        )
        return

    if args.ground_truth:
        print("Loading lat/lon from zarr for ground-truth figures…")
        ds = xr.open_zarr(ZWD_ZARR)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        plot_ground_truth(
            lat,
            lon,
            results_root=args.gt_results_root,
            location_cases=GROUND_TRUTH_CASE_PRESETS[args.gt_case_preset],
            output_dir=args.gt_output_dir,
            target_modes=tuple(args.gt_target_modes),
            locations=list(args.gt_locations) if args.gt_locations else None,
        )
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    zwd_loc, zwd_scale = load_zwd_norm()
    print(f"ZWD norm: loc={zwd_loc:.2f} mm, scale={zwd_scale:.2f} mm")

    print("Loading ZWD at t1 from zarr…")
    zwd_t1, lat, lon = load_zwd_t1(INIT_TIME)
    print(f"  zwd_t1: shape={zwd_t1.shape}, range={zwd_t1.min():.1f}..{zwd_t1.max():.1f} mm")

    m_local, sig_local = make_mask("local", lat, lon)
    m_syno,  sig_syno  = make_mask("synoptic", lat, lon)

    zwd_local, delta_local = perturb(zwd_t1, m_local, zwd_loc, zwd_scale)
    zwd_syno,  delta_syno  = perturb(zwd_t1, m_syno,  zwd_loc, zwd_scale)

    saliency = np.load(os.path.join(
        RESULTS_ROOT, "per_case", CASE_ID, "local", "saliency_attr.npy"
    ))
    ig = np.load(os.path.join(
        RESULTS_ROOT, "per_case", CASE_ID, "local", "ig_attr.npy"
    ))

    ext_zoom = TARGET.map_extent
    zwd_z, lat_z, lon_z = regional_slice(zwd_t1, lat, lon, ext_zoom)
    zwd_lo_z, _, _      = regional_slice(zwd_local, lat, lon, ext_zoom)
    zwd_sy_z, _, _      = regional_slice(zwd_syno,  lat, lon, ext_zoom)
    dl_z, _, _          = regional_slice(delta_local, lat, lon, ext_zoom)
    ds_z, _, _          = regional_slice(delta_syno,  lat, lon, ext_zoom)
    ml_z, _, _          = regional_slice(m_local, lat, lon, ext_zoom)
    ms_z, _, _          = regional_slice(m_syno,  lat, lon, ext_zoom)

    sal_z, _, _    = regional_slice(saliency, lat, lon, ext_zoom)

    vmax_zwd = float(np.percentile(np.concatenate([
        zwd_z.ravel(), zwd_lo_z.ravel(), zwd_sy_z.ravel()
    ]), 99))
    vmin_zwd = float(np.percentile(zwd_z.ravel(), 1))

    delta_max = float(max(np.abs(dl_z).max(), np.abs(ds_z).max()))

    # Saliency has a ~10-30× spike at the target itself; show surrounding
    # structure by saturating the central peak (percentile 97 of |sal| in the
    # regional crop) rather than using its absolute max.
    sal_abs = np.abs(sal_z)
    sal_vmax = float(np.percentile(sal_abs, 97))

    amag_l, G_l, S_l, clats_l, clons_l = load_pooled_vs_gt("local")
    amag_s, G_s, S_s, clats_s, clons_s = load_pooled_vs_gt("synoptic")
    clons_l_plot = np.where(clons_l > 180, clons_l - 360, clons_l)
    rho_l, _ = spearmanr(amag_l, G_l)
    rho_s, _ = spearmanr(amag_s, G_s)
    print(f"rho_mag local  = {rho_l:.3f}  (n={len(G_l)})")
    print(f"rho_mag synop  = {rho_s:.3f}  (n={len(G_s)})")

    import matplotlib.colors as mcolors

    sal_z_smooth, _, _ = regional_slice(
        smooth_global(saliency, sigma=3.0), lat, lon, ext_zoom)
    ig_z_smooth, _, _ = regional_slice(
        smooth_global(ig, sigma=3.0), lat, lon, ext_zoom)

    # Contrastive attribution maps (f_tgt - f_global reference)
    per_dir = os.path.join(RESULTS_ROOT, "per_case", CASE_ID, "local")
    csal_g = np.load(os.path.join(per_dir, "contrastive_saliency_global_attr.npy"))
    cig_g  = np.load(os.path.join(per_dir, "contrastive_ig_global_attr.npy"))
    csal_z_smooth, _, _ = regional_slice(
        smooth_global(csal_g, sigma=3.0), lat, lon, ext_zoom)
    cig_z_smooth, _, _ = regional_slice(
        smooth_global(cig_g, sigma=3.0), lat, lon, ext_zoom)

    # Signed pooled attributions + signed GT for sign-recovery scatter
    sal_sign_all  = np.load(os.path.join(per_dir, "saliency_pooled_Asign.npy"))
    ig_sign_all   = np.load(os.path.join(per_dir, "ig_pooled_Asign.npy"))
    csal_sign_all = np.load(os.path.join(per_dir, "contrastive_saliency_global_pooled_Asign.npy"))
    cig_sign_all  = np.load(os.path.join(per_dir, "contrastive_ig_global_pooled_Asign.npy"))

    gt_plain  = json.load(open(os.path.join(per_dir, "ground_truth.json")))
    gt_global = json.load(open(os.path.join(per_dir, "ground_truth_global.json")))
    near_sign_mask = np.array([m["role"] == "near" for m in gt_plain["masks"]])
    S_plain  = np.array(gt_plain["S"])[near_sign_mask]
    S_global = np.array(gt_global["S"])[near_sign_mask]
    sal_sign_near  = sal_sign_all[near_sign_mask]
    ig_sign_near   = ig_sign_all[near_sign_mask]
    csal_sign_near = csal_sign_all[near_sign_mask]
    cig_sign_near  = cig_sign_all[near_sign_mask]

    # ------------------------------------------------------------------ layout
    # Row 0: (a)(b)(c)          — ZWD perturbations        [cols 0-1, 2:5]
    # Row 1: (d)(e)(f)          — masks + cross-section    [cols 0-1, 2:5]
    # Row 2: (g1)(g2)(g3)(g4)(h) — attribution + GT + scatter [5 cols]
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(25, 19.5), constrained_layout=True)
    gs = GridSpec(4, 5, figure=fig)

    # --- Row 0: ZWD perturbations ---
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    _add_basemap(ax1, ext_zoom); _add_target_box(ax1)
    im = _imshow_on(ax1, zwd_z, lat_z, lon_z,
                    cmap="viridis", vmin=vmin_zwd, vmax=vmax_zwd)
    ax1.set_title("(a) unperturbed ZWD at t1")
    fig.colorbar(im, ax=ax1, shrink=0.75, label="ZWD [mm]")

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    _add_basemap(ax2, ext_zoom); _add_target_box(ax2)
    im = _imshow_on(ax2, zwd_lo_z, lat_z, lon_z,
                    cmap="viridis", vmin=vmin_zwd, vmax=vmax_zwd)
    ax2.set_title(f"(b) ZWD + 1·σ_zwd·M_local   (σ={sig_local}°)")
    fig.colorbar(im, ax=ax2, shrink=0.75, label="ZWD [mm]")
    ax2.plot(ILLUSTRATION_CLON if ILLUSTRATION_CLON < 180 else ILLUSTRATION_CLON - 360,
             ILLUSTRATION_CLAT, "x", color="red", markersize=10,
             transform=ccrs.PlateCarree(), zorder=6)
    # Mark all 376 near-mask centers so the viewer sees the full evaluation grid
    ax2.plot(clons_l_plot, clats_l, "x", color="yellow",
             markersize=4, markeredgewidth=0.9, linestyle="none",
             transform=ccrs.PlateCarree(), zorder=6)

    # (c) spans cols 2-4 so it has the same width as (a)/(b)
    ax3 = fig.add_subplot(gs[0, 2:5], projection=ccrs.PlateCarree())
    _add_basemap(ax3, ext_zoom); _add_target_box(ax3)
    im = _imshow_on(ax3, zwd_sy_z, lat_z, lon_z,
                    cmap="viridis", vmin=vmin_zwd, vmax=vmax_zwd)
    ax3.set_title(f"(c) ZWD + 1·σ_zwd·M_synoptic   (σ={sig_syno}°)")
    fig.colorbar(im, ax=ax3, shrink=0.75, label="ZWD [mm]")
    ax3.plot(ILLUSTRATION_CLON if ILLUSTRATION_CLON < 180 else ILLUSTRATION_CLON - 360,
             ILLUSTRATION_CLAT, "x", color="red", markersize=10,
             transform=ccrs.PlateCarree(), zorder=6)

    # --- Row 1: masks + cross-section ---
    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    _add_basemap(ax4, ext_zoom); _add_target_box(ax4)
    im = _imshow_on(ax4, ml_z, lat_z, lon_z,
                    cmap="Reds", vmin=0, vmax=1)
    ax4.set_title(f"(d) Gaussian M_local   (σ={sig_local}°, peak=1)")
    fig.colorbar(im, ax=ax4, shrink=0.75, label="mask value")

    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    _add_basemap(ax5, ext_zoom); _add_target_box(ax5)
    im = _imshow_on(ax5, ms_z, lat_z, lon_z,
                    cmap="Reds", vmin=0, vmax=1)
    ax5.set_title(f"(e) Gaussian M_synoptic   (σ={sig_syno}°, peak=1)")
    fig.colorbar(im, ax=ax5, shrink=0.75, label="mask value")

    # (f) cross-section spans cols 2-4
    ax6 = fig.add_subplot(gs[1, 2:5])
    lat_idx = int(np.argmin(np.abs(lat - ILLUSTRATION_CLAT)))
    lon_plot_full = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_plot_full)
    lon_ordered = lon_plot_full[order]
    cl_plot = ILLUSTRATION_CLON if ILLUSTRATION_CLON < 180 else ILLUSTRATION_CLON - 360
    band = (lon_ordered >= cl_plot - 15) & (lon_ordered <= cl_plot + 15)
    dl_line = delta_local[lat_idx][order][band]
    ds_line = delta_syno[lat_idx][order][band]
    zo_line = zwd_t1[lat_idx][order][band]
    ax6.plot(lon_ordered[band], zo_line, color="gray", lw=1.0,
             label="unperturbed ZWD", alpha=0.7)
    ax6.plot(lon_ordered[band], zo_line + dl_line, color="tab:red", lw=1.5,
             label=f"+ local bump (σ={sig_local}°)")
    ax6.plot(lon_ordered[band], zo_line + ds_line, color="tab:blue", lw=1.5,
             label=f"+ synoptic bump (σ={sig_syno}°)")
    ax6.axvline(cl_plot, color="black", lw=0.8, linestyle=":", alpha=0.5)
    ax6.set_xlabel("longitude [°E]")
    ax6.set_ylabel("ZWD [mm]")
    ax6.set_title(f"(f) ZWD cross-section at {ILLUSTRATION_CLAT}° N\n"
                  f"+1·σ_zwd peak = +{zwd_scale:.1f} mm")
    ax6.legend(fontsize=9, loc="upper left")
    ax6.grid(alpha=0.3)

    # --- Row 2: comparison maps + scatter plots ---
    from scipy.interpolate import griddata

    # Build a regular lon/lat grid covering the zoom extent for interpolation
    lon_plot_z = np.where(lon_z > 180, lon_z - 360, lon_z)
    grid_lon, grid_lat = np.meshgrid(lon_plot_z, lat_z)

    S_grid = griddata(
        np.column_stack([clons_l_plot, clats_l]),
        S_l,
        (grid_lon, grid_lat),
        method="linear",
    )

    sal_vmax_g = float(np.nanpercentile(np.abs(sal_z_smooth), 97))
    ig_vmax_g  = float(np.nanpercentile(np.abs(ig_z_smooth),  97))
    S_vmax     = float(np.nanpercentile(np.abs(S_grid[~np.isnan(S_grid)]), 97))

    # Synoptic GT field (92 points, σ=6°)
    clons_s_plot = np.where(clons_s > 180, clons_s - 360, clons_s)
    S_grid_s = griddata(
        np.column_stack([clons_s_plot, clats_s]),
        S_s,
        (grid_lon, grid_lat),
        method="linear",
    )
    S_vmax_s = float(np.nanpercentile(np.abs(S_grid_s[~np.isnan(S_grid_s)]), 97))

    # Contrastive GT fields (global and remote), same mask centers as local
    gt_remote = json.load(open(os.path.join(per_dir, "ground_truth_remote.json")))
    S_global_near = np.array(gt_global["S"])[near_sign_mask]
    S_remote_near = np.array(gt_remote["S"])[near_sign_mask]
    S_grid_global_l = griddata(
        np.column_stack([clons_l_plot, clats_l]),
        S_global_near,
        (grid_lon, grid_lat),
        method="linear",
    )
    S_grid_remote_l = griddata(
        np.column_stack([clons_l_plot, clats_l]),
        S_remote_near,
        (grid_lon, grid_lat),
        method="linear",
    )
    S_vmax_global_l = float(np.nanpercentile(np.abs(S_grid_global_l[~np.isnan(S_grid_global_l)]), 97))
    S_vmax_remote_l = float(np.nanpercentile(np.abs(S_grid_remote_l[~np.isnan(S_grid_remote_l)]), 97))

    ax7a = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ax7b = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
    ax7c = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
    ax7d = fig.add_subplot(gs[2, 3], projection=ccrs.PlateCarree())

    for ax7, field, vmin, vmax, cmap, cb_label, title in (
        (ax7a, sal_z_smooth, -sal_vmax_g, sal_vmax_g, "RdBu_r",
         "∂(q@850)/∂ZWD_t1",
         "(g1) saliency\n∂f/∂ZWD at unperturbed input"),
        (ax7b, ig_z_smooth,  -ig_vmax_g,  ig_vmax_g,  "RdBu_r",
         "IG attribution",
         "(g2) integrated gradients\n∫ ∂f/∂ZWD along smoothed→actual ZWD"),
        (ax7c, S_grid,       -S_vmax,      S_vmax,     "RdBu_r",
         "S_r  [g kg⁻¹]",
         f"(g3) signed GT causal response — local masks\n"
         f"376 bump locations, σ={sig_local}°"),
        (ax7d, S_grid_s,     -S_vmax_s,    S_vmax_s,   "RdBu_r",
         "S_r  [g kg⁻¹]",
         f"(g4) signed GT causal response — synoptic masks\n"
         f"92 bump locations, σ={sig_syno}°"),
    ):
        _add_basemap(ax7, ext_zoom)
        _add_target_box(ax7, color="red")
        im = _imshow_on(ax7, field, lat_z, lon_plot_z,
                        cmap=cmap, vmin=vmin, vmax=vmax)
        ax7.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax7, shrink=0.75, label=cb_label, extend="both")

    ax8 = fig.add_subplot(gs[2, 4])
    amag_vit_l = np.load(os.path.join(
        RESULTS_ROOT, "per_case", CASE_ID, "local", "ig_pooled_Amag.npy"
    ))
    gt_local = json.load(open(os.path.join(
        RESULTS_ROOT, "per_case", CASE_ID, "local", "ground_truth.json"
    )))
    near_mask = np.array([m["role"] == "near" for m in gt_local["masks"]])
    amag_vit_l_near = amag_vit_l[near_mask]

    def _n01(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)

    rho_vit_l = spearmanr(amag_vit_l_near, G_l)[0]
    ax8.scatter(G_l, _n01(amag_l), s=22, color="tab:blue", edgecolor="black",
                linewidth=0.3, alpha=0.85, label=f"saliency  ρ={rho_l:.2f}")
    ax8.scatter(G_l, _n01(amag_vit_l_near), s=22, color="tab:orange", edgecolor="black",
                linewidth=0.3, alpha=0.85, label=f"IG            ρ={rho_vit_l:.2f}",
                marker="^")
    ax8.set_xlabel("|G_r|   ground-truth effect  [g kg⁻¹]")
    ax8.set_ylabel("pooled |A|_r  (normalised 0–1 each)")
    ax8.set_title("(h) attribution vs GT — local masks\n(each method normalised independently)")
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)

    # --- Row 3: contrastive references + maps + both contrastive GTs + sign scatter ---
    # Use a 6-column sub-GridSpec so this row can exceed the 5-col outer layout.
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs3 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[3, :], wspace=0.35)

    csal_vmax = float(np.nanpercentile(np.abs(csal_z_smooth), 97))
    cig_vmax  = float(np.nanpercentile(np.abs(cig_z_smooth),  97))

    # (i) World map: f_tgt (blue box), f_remote (orange antipodal box), f_global label
    ax9a = fig.add_subplot(gs3[0, 0], projection=ccrs.PlateCarree())
    ax9a.set_global()
    ax9a.add_feature(cfeat.LAND,      facecolor="whitesmoke", zorder=0)
    ax9a.add_feature(cfeat.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax9a.add_feature(cfeat.COASTLINE, edgecolor="gray", linewidth=0.3)
    ax9a.add_feature(cfeat.BORDERS,   edgecolor="gray", linewidth=0.2, linestyle=":")

    tb_w, tb_e = TARGET.box_lon[0], TARGET.box_lon[1]
    tb_s, tb_n = TARGET.box_lat[0], TARGET.box_lat[1]
    ax9a.add_patch(Rectangle(
        (tb_w, tb_s), tb_e - tb_w, tb_n - tb_s,
        fill=True, facecolor="steelblue", alpha=0.4,
        edgecolor="steelblue", linewidth=1.5,
        transform=ccrs.PlateCarree(), zorder=5,
    ))
    ax9a.text(0.5 * (tb_w + tb_e), tb_s - 9, "f_tgt",
              ha="center", fontsize=7, color="steelblue",
              transform=ccrs.PlateCarree())

    center_lon_tgt = 0.5 * (TARGET.box_lon[0] + TARGET.box_lon[1])
    remote_clon = (center_lon_tgt + 180.0) % 360.0
    half_w = 0.5 * (TARGET.box_lon[1] - TARGET.box_lon[0])
    rem_w_p = (remote_clon - half_w) - 360 if (remote_clon - half_w) > 180 else (remote_clon - half_w)
    rem_e_p = (remote_clon + half_w) - 360 if (remote_clon + half_w) > 180 else (remote_clon + half_w)
    ax9a.add_patch(Rectangle(
        (rem_w_p, tb_s), rem_e_p - rem_w_p, tb_n - tb_s,
        fill=True, facecolor="darkorange", alpha=0.4,
        edgecolor="darkorange", linewidth=1.5,
        transform=ccrs.PlateCarree(), zorder=5,
    ))
    ax9a.text(0.5 * (rem_w_p + rem_e_p), tb_n + 7, "f_remote",
              ha="center", fontsize=7, color="darkorange",
              transform=ccrs.PlateCarree())
    ax9a.text(0, -72, "f_global = cos-lat\nweighted global mean",
              ha="center", fontsize=6.5, color="dimgray",
              transform=ccrs.PlateCarree(), style="italic")
    ax9a.set_title("(i) contrastive references\nf_tgt · f_remote · f_global", fontsize=8)

    # (j) contrastive saliency (global)
    ax9b = fig.add_subplot(gs3[0, 1], projection=ccrs.PlateCarree())
    _add_basemap(ax9b, ext_zoom)
    _add_target_box(ax9b, color="red")
    im9b = _imshow_on(ax9b, csal_z_smooth, lat_z, lon_plot_z,
                      cmap="RdBu_r", vmin=-csal_vmax, vmax=csal_vmax)
    ax9b.set_title("(j) contr. saliency (global)\n∂(f_tgt−f_global)/∂ZWD", fontsize=9)
    fig.colorbar(im9b, ax=ax9b, shrink=0.75, label="attribution", extend="both")

    # (k) contrastive IG (global)
    ax9c = fig.add_subplot(gs3[0, 2], projection=ccrs.PlateCarree())
    _add_basemap(ax9c, ext_zoom)
    _add_target_box(ax9c, color="red")
    im9c = _imshow_on(ax9c, cig_z_smooth, lat_z, lon_plot_z,
                      cmap="RdBu_r", vmin=-cig_vmax, vmax=cig_vmax)
    ax9c.set_title("(k) contr. IG (global)\n∫∂(f_tgt−f_global)/∂ZWD", fontsize=9)
    fig.colorbar(im9c, ax=ax9c, shrink=0.75, label="attribution", extend="both")

    # (l) contrastive GT global: signed Δ(f_tgt − f_global)
    ax9d = fig.add_subplot(gs3[0, 3], projection=ccrs.PlateCarree())
    _add_basemap(ax9d, ext_zoom)
    _add_target_box(ax9d, color="red")
    im9d = _imshow_on(ax9d, S_grid_global_l, lat_z, lon_plot_z,
                      cmap="RdBu_r", vmin=-S_vmax_global_l, vmax=S_vmax_global_l)
    ax9d.set_title(f"(l) contrastive GT (global)\nΔ(f_tgt−f_global) / 2  σ={sig_local}°", fontsize=9)
    fig.colorbar(im9d, ax=ax9d, shrink=0.75, label="S_r  [g kg⁻¹]", extend="both")

    # (m) contrastive GT remote: signed Δ(f_tgt − f_remote)
    ax9e_map = fig.add_subplot(gs3[0, 4], projection=ccrs.PlateCarree())
    _add_basemap(ax9e_map, ext_zoom)
    _add_target_box(ax9e_map, color="red")
    im9e = _imshow_on(ax9e_map, S_grid_remote_l, lat_z, lon_plot_z,
                      cmap="RdBu_r", vmin=-S_vmax_remote_l, vmax=S_vmax_remote_l)
    ax9e_map.set_title(f"(m) contrastive GT (remote)\nΔ(f_tgt−f_remote) / 2  σ={sig_local}°", fontsize=9)
    fig.colorbar(im9e, ax=ax9e_map, shrink=0.75, label="S_r  [g kg⁻¹]", extend="both")

    # (n) Sign-recovery scatter
    def _norm_sign(x):
        m = np.abs(x).max()
        return x / (m + 1e-30)

    rho_sal_sign  = spearmanr(sal_sign_near,  S_plain)[0]
    rho_csal_sign = spearmanr(csal_sign_near, S_global)[0]
    rho_ig_sign   = spearmanr(ig_sign_near,   S_plain)[0]
    rho_cig_sign  = spearmanr(cig_sign_near,  S_global)[0]

    ax9f = fig.add_subplot(gs3[0, 5])
    s_norm = _norm_sign(S_plain)
    ax9f.scatter(s_norm, _norm_sign(sal_sign_near), s=20, color="tab:blue",
                 edgecolor="black", linewidth=0.3, alpha=0.85,
                 label=f"saliency           ρ={rho_sal_sign:.2f}")
    ax9f.scatter(s_norm, _norm_sign(csal_sign_near), s=20, color="tab:cyan",
                 edgecolor="black", linewidth=0.3, alpha=0.85, marker="D",
                 label=f"contr. saliency  ρ={rho_csal_sign:.2f}")
    ax9f.scatter(s_norm, _norm_sign(ig_sign_near), s=20, color="tab:orange",
                 edgecolor="black", linewidth=0.3, alpha=0.85, marker="^",
                 label=f"IG                   ρ={rho_ig_sign:.2f}")
    ax9f.scatter(s_norm, _norm_sign(cig_sign_near), s=20, color="tab:red",
                 edgecolor="black", linewidth=0.3, alpha=0.85, marker="s",
                 label=f"contr. IG         ρ={rho_cig_sign:.2f}")
    ax9f.axhline(0, color="gray", lw=0.6, linestyle="--")
    ax9f.axvline(0, color="gray", lw=0.6, linestyle="--")
    ax9f.set_xlabel("S_r (signed GT, normalised)")
    ax9f.set_ylabel("A_sign_r (normalised ±1 each)")
    ax9f.set_title("(n) sign recovery — near masks\n(saliency tracks GT sign; IG does not)")
    ax9f.legend(fontsize=7.5)
    ax9f.grid(alpha=0.3)

    fig.suptitle(
        "ZWD Searchlight — how masking works and what saliency recovers",
        fontsize=13, y=1.005,
    )
    fig.text(
        0.5, -0.005,
        f"case: {CASE_ID}  ·  t1 = {INIT_TIME:%Y-%m-%d %H:%M UTC}  ·  "
        f"magnitude = {MAGNITUDE}·σ_zwd  (σ_zwd = {zwd_scale:.1f} mm)",
        ha="center", va="top", fontsize=10, color="dimgray",
    )

    plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
