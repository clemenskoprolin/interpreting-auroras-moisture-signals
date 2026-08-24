#!/usr/bin/env python3
"""Thesis figure: precipitation and ZWD saliency and their spatial difference.

The two gradients have different physical units and differ by roughly a
factor of 20, so a raw pixel-wise subtraction is not meaningful.  Even after
amplitude normalisation, an absolute difference is dominated by the central
peak: modest relative differences become large residuals wherever both input
maps are large.

All three panels therefore treat each smoothed field as a spatial saliency
distribution by dividing it by its sum over the canonical regional extent.
The first two use one shared square-root colour scale so weaker peripheral
structures remain visible without clipping.  The signed difference
``P/sum(P) - ZWD/sum(ZWD)`` remains linear.  Keeping the sign is important:
the two sides of a displaced or differently shaped peak appear as opposing
red and blue lobes instead of being folded into one misleading central blob.
Red means relatively more precipitation saliency and blue relatively more ZWD
saliency.

The figure reads the 22-case q850 maps produced for
``08_internal_routing_and_activation_patching/obs_vs_saliency_hotspots.py``.
This is a plotting-only utility: it does not import or run Aurora.

The layout deliberately follows the compact, white-background searchlight
figure in ``02_zwd_attribution_benchmark/visualize/fig_thesis_overview_row.py``.

Example
-------
source .venv/bin/activate
python 07_zwd_precipitation_model_comparison/visualize/fig_stage_b_saliency_maps.py \
    --out results/img/fig_large_stage_b_saliency_maps.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")
if _SEARCHLIGHT_DIR not in sys.path:
    sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_tasks import TARGETS  # noqa: E402
from visualize_masking import (  # noqa: E402
    _add_basemap,
    _draw_target_marker,
    _imshow_on,
    regional_slice,
    smooth_global,
)


_EXTERNAL_RESULTS_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "model_conditional_comparison", "precip_large_zwd",
)
_LOCAL_RESULTS_DIR = os.path.join(
    _ROOT, "local", "results", "obs_vs_saliency_hotspots_v1"
)
_REPO_RESULTS_DIR = os.path.join(_ROOT, "results", "obs_vs_saliency_hotspots_v1")


def _is_complete(results_dir: str) -> bool:
    """A usable copy needs both the grid and the per-case reliance maps."""
    return os.path.isfile(os.path.join(results_dir, "grid.npz")) and os.path.isdir(
        os.path.join(results_dir, "stage_b_reliance_maps")
    )


# The scratch copy loses files to the 30-day cleanup.  The repo tracks only the
# aggregate CSVs -- the per-case .npy maps are too large to version, so they
# live in the untracked local/ mirror, which is on /users and not subject to
# the cleanup.  Prefer whichever copy is actually complete.
DEFAULT_RESULTS_DIR = next(
    (
        d
        for d in (_EXTERNAL_RESULTS_DIR, _LOCAL_RESULTS_DIR, _REPO_RESULTS_DIR)
        if _is_complete(d)
    ),
    _REPO_RESULTS_DIR,
)

# Representative rather than cherry-picked: rho=0.777 is close to the
# 22-case q850 median (0.772), while its two centers of mass are only 84 km
# apart and its target-region signal is visually clear.
DEFAULT_CASE = "andes_2020-03-25T12:00:00"
DEFAULT_OUT = os.path.join(
    _ROOT, "results", "img", "fig_large_stage_b_saliency_maps.png"
)

CB_SHRINK = 0.66
CB_ASPECT = 26
CB_PAD = 0.02
CB_TICKLABEL_SIZE = 7
CB_LABEL_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--target-var", default="q850", choices=("q850",))
    parser.add_argument("--lead-h", type=int, default=6)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument(
        "--saliency-gamma",
        type=float,
        default=0.5,
        help=(
            "Power-law colour normalization for the first two panels "
            "(default: 0.5, i.e. square root; use 1 for linear)."
        ),
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=0.65,
        help=(
            "Fraction of the canonical map width and height to display, "
            "centred on the target box (default: 0.65; use 1 for full extent)."
        ),
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Omit the case/metric title; useful when the thesis caption carries it.",
    )
    return parser.parse_args()


def infer_target(case_id: str) -> str:
    matches = [name for name in TARGETS if case_id.startswith(name + "_")]
    if not matches:
        raise ValueError(f"Cannot infer target region from case ID {case_id!r}")
    return max(matches, key=len)


def zoomed_extent(target, factor: float) -> tuple[float, float, float, float]:
    """Scale the canonical extent around the target-box centre."""
    lon_w, lon_e, lat_s, lat_n = target.map_extent
    center_lon = (target.center_lon + 180.0) % 360.0 - 180.0
    half_width = 0.5 * (lon_e - lon_w) * factor
    half_height = 0.5 * (lat_n - lat_s) * factor
    return (
        center_lon - half_width,
        center_lon + half_width,
        target.center_lat - half_height,
        target.center_lat + half_height,
    )


def map_path(
    results_dir: str,
    case_id: str,
    channel: str,
    target_var: str,
    lead_h: int,
) -> str:
    return os.path.join(
        results_dir,
        "stage_b_reliance_maps",
        case_id,
        f"precip_large_zwd_{channel}_{target_var}_{lead_h}h.npy",
    )


def load_case_metrics(results_dir: str, case_id: str) -> dict[str, float]:
    path = os.path.join(results_dir, "stage_b_reliance_summary.csv")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return {
                    "rho": float(row["spearman_r_precip_zwd"]),
                    "top1_overlap": float(row["top1pct_overlap"]),
                    "com_km": float(row["com_displacement_km"]),
                }
    raise ValueError(f"No Stage-B metrics found for {case_id!r} in {path}")


def add_colorbar(fig, image, ax, label: str, extend: str = "max"):
    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=CB_SHRINK,
        aspect=CB_ASPECT,
        pad=CB_PAD,
        extend=extend,
    )
    colorbar.set_label(label, fontsize=CB_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=CB_TICKLABEL_SIZE)
    colorbar.ax.yaxis.get_offset_text().set_fontsize(CB_TICKLABEL_SIZE)
    return colorbar


def main() -> None:
    args = parse_args()
    if not 0.0 < args.saliency_gamma <= 1.0:
        raise ValueError("--saliency-gamma must be in (0, 1]")
    if not 0.0 < args.zoom_factor <= 1.0:
        raise ValueError("--zoom-factor must be in (0, 1]")

    target_name = infer_target(args.case)
    target = TARGETS[target_name]
    plot_extent = zoomed_extent(target, args.zoom_factor)
    metrics = load_case_metrics(args.results_dir, args.case)

    grid = np.load(os.path.join(args.results_dir, "grid.npz"))
    lat = grid["lat_vals"]
    lon = grid["lon_vals"]

    channels = (
        (
            "tp_mswep",
            "Precipitation saliency\n"
            r"normalized $\left|\partial q_{850}(+6\,\mathrm{h})/"
            r"\partial P(t_1)\right|$",
        ),
        (
            "zwd",
            "ZWD saliency\n"
            r"normalized $\left|\partial q_{850}(+6\,\mathrm{h})/"
            r"\partial\mathrm{ZWD}(t_1)\right|$",
        ),
    )

    saliency_distributions: dict[str, np.ndarray] = {}
    saliency_display_vmaxes: dict[str, float] = {}
    lat_crop = lon_crop = None

    for channel, _ in channels:
        path = map_path(
            args.results_dir,
            args.case,
            channel,
            args.target_var,
            args.lead_h,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        field = np.load(path)
        if field.shape != (lat.size, lon.size):
            raise ValueError(
                f"Unexpected shape for {path}: {field.shape}; "
                f"expected {(lat.size, lon.size)}"
            )
        if not np.isfinite(field).all() or np.nanmin(field) < 0:
            raise ValueError(f"Expected finite non-negative saliency in {path}")

        # Match the searchlight thesis map: smooth globally before cropping so
        # wrapped longitudes cannot produce a seam at the crop boundary.
        field = smooth_global(field, sigma=args.smooth_sigma)
        regional_field, _, _ = regional_slice(
            field, lat, lon, target.map_extent
        )
        field, lat_crop, lon_crop = regional_slice(
            field, lat, lon, plot_extent
        )
        # Unit-sum normalization makes the fields comparable as spatial
        # distributions while preserving every value and their relative
        # strength within each channel.  Keep the denominator tied to the
        # canonical region so changing only the display zoom cannot change
        # the values plotted at a given location.
        regional_total = float(np.nansum(regional_field))
        saliency_distributions[channel] = field / regional_total
        saliency_display_vmaxes[channel] = float(
            np.nanmax(regional_field / regional_total)
        )

    shared_saliency_vmax = max(saliency_display_vmaxes.values())
    if not np.isfinite(shared_saliency_vmax) or shared_saliency_vmax <= 0:
        shared_saliency_vmax = 1.0
    shared_saliency_norm = mcolors.PowerNorm(
        gamma=args.saliency_gamma,
        vmin=0.0,
        vmax=shared_saliency_vmax,
    )
    difference = (
        saliency_distributions["tp_mswep"]
        - saliency_distributions["zwd"]
    )
    diff_vmax = float(np.nanmax(np.abs(difference)))
    if not np.isfinite(diff_vmax) or diff_vmax <= 0:
        diff_vmax = 1.0
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 3.4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    source_image = None
    for ax, (channel, title) in zip(axes, channels):
        _add_basemap(ax, plot_extent)
        _draw_target_marker(ax, target, "box")
        source_image = _imshow_on(
            ax,
            saliency_distributions[channel],
            lat_crop,
            lon_crop,
            cmap="Reds",
            norm=shared_saliency_norm,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
    add_colorbar(
        fig,
        source_image,
        axes[:2],
        "fraction of regional saliency",
        extend="neither",
    )

    diff_ax = axes[2]
    _add_basemap(diff_ax, plot_extent)
    _draw_target_marker(diff_ax, target, "box")
    diff_image = _imshow_on(
        diff_ax,
        difference,
        lat_crop,
        lon_crop,
        cmap="RdBu_r",
        vmin=-diff_vmax,
        vmax=diff_vmax,
        interpolation="nearest",
    )
    diff_ax.set_title(
        "Relative saliency difference\n"
        r"$+$ precipitation stronger (red)  ·  $-$ ZWD stronger (blue)",
        fontsize=9,
    )
    add_colorbar(
        fig,
        diff_image,
        diff_ax,
        "relative saliency-mass difference",
        extend="neither",
    )

    if not args.no_suptitle:
        init_time = datetime.fromisoformat(args.case[len(target_name) + 1 :])
        fig.suptitle(
            f"{target.name}  ·  {init_time:%Y-%m-%d %H:%M UTC}  ·  "
            rf"Spearman $\rho={metrics['rho']:.3f}$",
            fontsize=9,
        )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {out}\n"
        f"case={args.case} rho={metrics['rho']:.6f} "
        f"top1_overlap={metrics['top1_overlap']:.6f} "
        f"com_displacement_km={metrics['com_km']:.1f}\n"
        f"plot_extent={plot_extent} zoom_factor={args.zoom_factor:.3f}\n"
        f"shared saliency-distribution vmax={shared_saliency_vmax:.6g} "
        f"gamma={args.saliency_gamma:.3f}\n"
        f"signed-distribution-difference panel: vmax={diff_vmax:.6g} "
        f"min={float(np.nanmin(difference)):.6g} "
        f"max={float(np.nanmax(difference)):.6g} "
        f"mean_abs={float(np.nanmean(np.abs(difference))):.6g}"
    )


if __name__ == "__main__":
    main()
