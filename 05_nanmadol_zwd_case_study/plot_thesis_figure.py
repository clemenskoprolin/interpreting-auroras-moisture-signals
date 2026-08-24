"""Create the thesis two-panel figure for the Nanmadol ZWD case study.

This is a plotting-only script.  It reads the saved storm-centred saliency and
perturbation outputs and does not import or run Aurora.

The panels are:

* smoothed signed saliency of the +48 h intensity target, with the
  saliency-selected hotspot and the low-near control; and
* a zoomed overlay of the baseline and +/-1 sigma ZWD-perturbed forecast
  tracks, together with the IBTrACS best track.

A high-resolution PNG is written for direct inclusion in the thesis.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "nanmadol_2022091712_zwd_tc_case_study"
RESULTS_ROOT = Path(os.environ.get("AURORA_XAI_RESULTS_DIR", REPO_ROOT / "results"))
EXTERNAL_STUDY_ROOT = RESULTS_ROOT / "zwd_tc_case_study" / STUDY_ID
REPO_STUDY_ROOT = REPO_ROOT / "results" / "zwd_tc_case_study" / STUDY_ID
BEST_TRACK_PATH = Path(__file__).resolve().parent / "data" / "nanmadol_ibtracs.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "results" / "img" / "fig_nanmadol_case_study_thesis"

INIT_TIME = datetime(2022, 9, 17, 12, 0)
INIT_LAT = 27.5
INIT_LON = 132.0

# Colourblind-safe track palette (Okabe--Ito inspired).
COLORS = {
    "baseline": "#202020",
    "best_track": "#009E73",
    "hotspot": "#D55E00",
    "low_near": "#CC79A7",
    "remote": "#0072B2",
}
REGION_LABELS = {
    "hotspot": "Hotspot",
    "low_near": "Low-saliency near control",
    "remote": "Remote control",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot the two-panel Nanmadol saliency and track-response figure."
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help=(
            "Directory containing storm_centered_xai/ and track_perturbation/. "
            "Defaults to the scratch result if available, then the repo copy."
        ),
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Output path without suffix; .png is added.",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in 0.25-degree grid cells (default: 3).",
    )
    p.add_argument(
        "--max-track-lead",
        type=float,
        default=96.0,
        help="Largest forecast lead shown in the track panel (default: 96 h).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300 dpi).",
    )
    return p.parse_args()


def resolve_study_root(requested: Path | None) -> Path:
    candidates = [requested] if requested is not None else [EXTERNAL_STUDY_ROOT, REPO_STUDY_ROOT]
    required = (
        Path("storm_centered_xai/saliency_intensity_048h.npy"),
        Path("storm_centered_xai/selected_regions.json"),
        Path("track_perturbation/perturbation_responses.csv"),
    )
    for candidate in candidates:
        if candidate is not None and all((candidate / item).exists() for item in required):
            return candidate
    checked = "\n  ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not find a complete saved Nanmadol result. Checked:\n  " + checked
    )


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable two-dimensional Gaussian smoothing without scipy."""
    if sigma <= 0:
        return np.asarray(arr, dtype=np.float32)
    radius = max(1, int(4 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    work = np.asarray(arr, dtype=np.float64)
    work = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=work
    )
    work = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=work
    )
    return work.astype(np.float32)


def add_basemap(ax, extent: tuple[float, float, float, float]) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#666666", linewidth=0.65, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor="#888888", linewidth=0.35, zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#8a8a8a",
        alpha=0.45,
        linestyle=":",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8.5, "color": "#333333"}
    gl.ylabel_style = {"size": 8.5, "color": "#333333"}
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(5)


def load_best_track(max_lead: float) -> pd.DataFrame:
    best = pd.read_csv(BEST_TRACK_PATH, parse_dates=["ISO_TIME"])
    end_time = INIT_TIME + timedelta(hours=float(max_lead))
    best = best[(best["ISO_TIME"] >= INIT_TIME) & (best["ISO_TIME"] <= end_time)].copy()
    best["lead_hours"] = (best["ISO_TIME"] - INIT_TIME).dt.total_seconds() / 3600.0
    best["lon_plot"] = ((best["LON"].astype(float) + 180.0) % 360.0) - 180.0
    best["lat"] = best["LAT"].astype(float)
    return best.sort_values("ISO_TIME")


def prepend_initial_point(df: pd.DataFrame) -> pd.DataFrame:
    """Add the common initialized storm centre to a forecast-track frame."""
    if df.empty or float(df["lead_hours"].min()) <= 0:
        return df
    initial = {column: np.nan for column in df.columns}
    initial.update(
        {
            "lead_hours": 0.0,
            "center_lat": INIT_LAT,
            "center_lon": INIT_LON,
            "center_lon_plot": INIT_LON,
        }
    )
    return pd.concat([pd.DataFrame([initial]), df], ignore_index=True)


def region_ellipse(ax, payload: dict, *, color: str, linestyle: str, linewidth: float) -> None:
    import cartopy.crs as ccrs
    from matplotlib.patches import Ellipse

    lat = float(payload["center_lat"])
    lon = float(payload["center_lon_plot"])
    sigma = float(payload.get("sigma_deg", 3.0))
    lon_radius = sigma / max(np.cos(np.radians(lat)), 0.2)
    ellipse = Ellipse(
        (lon, lat),
        width=2.0 * lon_radius,
        height=2.0 * sigma,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )
    ax.add_patch(ellipse)


def plot_saliency_panel(
    ax,
    *,
    saliency: np.ndarray,
    baseline: pd.DataFrame,
    selected_regions: dict,
    smooth_sigma: float,
) -> object:
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D

    extent = (118.0, 147.0, 12.0, 43.0)
    add_basemap(ax, extent)

    signed_smooth = gaussian_smooth(saliency, smooth_sigma)
    magnitude_smooth = gaussian_smooth(np.abs(saliency), smooth_sigma)
    nlat, nlon = saliency.shape
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon, dtype=np.float64) * (360.0 / nlon)

    inside = (
        (lat[:, None] >= extent[2])
        & (lat[:, None] <= extent[3])
        & (lon[None, :] >= extent[0])
        & (lon[None, :] <= extent[1])
    )
    vmax = float(np.nanpercentile(np.abs(signed_smooth[inside]), 99.5))
    vmax = max(vmax, 1e-12)
    high_magnitude = float(np.nanpercentile(magnitude_smooth[inside], 97.5))

    lat_idx = np.where((lat >= extent[2] - 1.0) & (lat <= extent[3] + 1.0))[0]
    lon_idx = np.where((lon >= extent[0] - 1.0) & (lon <= extent[1] + 1.0))[0]
    field = signed_smooth[np.ix_(lat_idx, lon_idx)]
    magnitude = magnitude_smooth[np.ix_(lat_idx, lon_idx)]
    mesh = ax.pcolormesh(
        lon[lon_idx],
        lat[lat_idx],
        field * 1.0e5,
        cmap="RdBu_r",
        shading="auto",
        vmin=-vmax * 1.0e5,
        vmax=vmax * 1.0e5,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=1,
    )
    ax.contour(
        lon[lon_idx],
        lat[lat_idx],
        magnitude,
        levels=[high_magnitude],
        colors="#303030",
        linewidths=0.7,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    track = prepend_initial_point(
        baseline[baseline["lead_hours"] <= 48.0].sort_values("lead_hours").copy()
    )
    ax.plot(
        track["center_lon_plot"],
        track["center_lat"],
        color="#242424",
        linewidth=1.55,
        marker="o",
        markersize=2.6,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    endpoint = track.iloc[-1]
    ax.scatter(
        endpoint["center_lon_plot"],
        endpoint["center_lat"],
        s=46,
        marker="s",
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )
    ax.annotate(
        "+48 h target",
        xy=(endpoint["center_lon_plot"], endpoint["center_lat"]),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
        color="#202020",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    hotspot = selected_regions["hotspot"]
    low_near = selected_regions["low_near"]
    region_ellipse(ax, hotspot, color="#202020", linestyle="-", linewidth=1.25)
    region_ellipse(ax, low_near, color="#505050", linestyle="--", linewidth=1.0)
    ax.scatter(
        hotspot["center_lon_plot"],
        hotspot["center_lat"],
        marker="*",
        s=155,
        facecolor="#202020",
        edgecolor="white",
        linewidth=0.65,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.annotate(
        "selected hotspot",
        xy=(hotspot["center_lon_plot"], hotspot["center_lat"]),
        xytext=(-55, -22),
        textcoords="offset points",
        fontsize=8.3,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": "#202020", "lw": 0.7},
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.scatter(
        low_near["center_lon_plot"],
        low_near["center_lat"],
        marker="o",
        s=54,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.annotate(
        "low-saliency control",
        xy=(low_near["center_lon_plot"], low_near["center_lat"]),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=8,
        color="#202020",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    handles = [
        Line2D([], [], color="#242424", marker="o", markersize=2.7, lw=1.4, label="Aurora baseline (+48 h)"),
        Line2D([], [], color="#202020", marker="*", markerfacecolor="#202020", lw=0, markersize=8.5, label="Selected hotspot"),
        Line2D([], [], color="#202020", marker="o", markerfacecolor="white", lw=0, markersize=5.2, label="Low-saliency control"),
        Line2D([], [], color="#303030", lw=0.75, label="Upper 2.5% |saliency|"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.012, 1.0),
        fontsize=6.35,
        frameon=True,
        framealpha=0.92,
        borderpad=0.45,
        labelspacing=0.4,
        handlelength=2.1,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    ax.text(
        0.018,
        0.018,
        rf"Gaussian smoothing: $\sigma={0.25 * smooth_sigma:.2g}^\circ$",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#333333",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.86, "pad": 2.2},
        zorder=12,
    )
    ax.set_title(
        r"(a) $+48$ h intensity saliency  $\partial I_{48}/\partial \mathrm{ZWD}_{t_1}$",
        fontsize=11.5,
        loc="left",
        pad=8,
    )
    return mesh


def track_line(
    ax,
    df: pd.DataFrame,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    marker: str | None = None,
    markersize: float = 2.5,
    zorder: float = 5,
) -> None:
    import cartopy.crs as ccrs

    ax.plot(
        df["center_lon_plot"],
        df["center_lat"],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        marker=marker,
        markersize=markersize,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
    )


def plot_track_panel(
    ax,
    *,
    baseline: pd.DataFrame,
    perturbations: pd.DataFrame,
    max_lead: float,
) -> None:
    import cartopy.crs as ccrs
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D

    extent = (127.0, 160.0, 27.0, 49.0)
    add_basemap(ax, extent)

    show = perturbations[
        (perturbations["lead_hours"] <= max_lead)
        & (perturbations["amplitude_sigma"].isin([-1.0, 1.0]))
    ].copy()

    # Plot controls first, then the causal hotspot response and baseline on top.
    for region_key in ("remote", "low_near", "hotspot"):
        for amplitude in (-1.0, 1.0):
            sub = show[
                (show["region_key"] == region_key)
                & (show["amplitude_sigma"] == amplitude)
            ].sort_values("lead_hours")
            sub = prepend_initial_point(sub)
            if sub.empty:
                continue
            is_hotspot = region_key == "hotspot"
            track_line(
                ax,
                sub,
                color=COLORS[region_key],
                linestyle="--" if amplitude < 0 else "-",
                linewidth=2.05 if is_hotspot else 1.25,
                alpha=0.95 if is_hotspot else 0.68,
                marker="o" if is_hotspot else None,
                markersize=2.6,
                zorder=6 if is_hotspot else 4,
            )

    base = prepend_initial_point(
        baseline[baseline["lead_hours"] <= max_lead].sort_values("lead_hours").copy()
    )
    baseline_line = ax.plot(
        base["center_lon_plot"],
        base["center_lat"],
        color=COLORS["baseline"],
        linewidth=2.25,
        marker="o",
        markersize=2.8,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )[0]
    baseline_line.set_path_effects(
        [path_effects.Stroke(linewidth=3.5, foreground="white", alpha=0.8), path_effects.Normal()]
    )

    best = load_best_track(max_lead)
    ax.plot(
        best["lon_plot"],
        best["lat"],
        color=COLORS["best_track"],
        linewidth=2.25,
        marker="D",
        markevery=2,
        markersize=3.0,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    for lead in (24.0, 48.0, 72.0, 96.0):
        if lead > max_lead:
            continue
        row = base[np.isclose(base["lead_hours"], lead)]
        if row.empty:
            continue
        point = row.iloc[0]
        offset = (5, -11) if lead in (48.0, 96.0) else (5, 5)
        ax.annotate(
            f"+{int(lead)} h",
            xy=(point["center_lon_plot"], point["center_lat"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.4,
            color="#202020",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

    ax.scatter(
        [INIT_LON],
        [INIT_LAT],
        marker="*",
        s=86,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.annotate(
        "initial centre",
        xy=(INIT_LON, INIT_LAT),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=7.8,
        color="#202020",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    handles = [
        Line2D([], [], color=COLORS["best_track"], marker="D", markersize=4, lw=2.2, label="IBTrACS best track"),
        Line2D([], [], color=COLORS["baseline"], marker="o", markersize=3, lw=2.2, label="Aurora baseline"),
        Line2D([], [], color=COLORS["hotspot"], lw=2.1, label="Hotspot"),
        Line2D([], [], color=COLORS["low_near"], lw=1.5, label="Low-saliency near"),
        Line2D([], [], color=COLORS["remote"], lw=1.5, label="Remote control"),
        Line2D([], [], color="#555555", lw=1.6, linestyle="-", label=r"$+1\sigma_{\mathrm{ZWD}}$"),
        Line2D([], [], color="#555555", lw=1.6, linestyle="--", label=r"$-1\sigma_{\mathrm{ZWD}}$"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        fontsize=7.3,
        frameon=True,
        framealpha=0.93,
        borderpad=0.55,
        columnspacing=0.9,
        handlelength=2.5,
    )
    ax.set_title(
        rf"(b) Track response to localized ZWD perturbations (to $+{int(max_lead)}$ h)",
        fontsize=11.5,
        loc="left",
        pad=8,
    )


def make_figure(
    *,
    study_root: Path,
    output_base: Path,
    smooth_sigma: float,
    max_track_lead: float,
    dpi: int,
) -> Path:
    # Set the non-interactive backend before pyplot is imported.
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    xai_dir = study_root / "storm_centered_xai"
    perturb_dir = study_root / "track_perturbation"
    saliency = np.load(xai_dir / "saliency_intensity_048h.npy")
    with open(xai_dir / "selected_regions.json", "r", encoding="utf-8") as handle:
        selected_regions = json.load(handle)
    baseline = pd.read_csv(perturb_dir / "baseline_track.csv")
    perturbations = pd.read_csv(perturb_dir / "perturbation_responses.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(15.2, 6.55), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.12),
        left=0.045,
        right=0.985,
        bottom=0.095,
        top=0.865,
        wspace=0.15,
    )
    ax_saliency = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
    ax_tracks = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    # GeoAxes otherwise centre themselves vertically when their map aspects
    # differ, which makes the two panel titles look misaligned.
    ax_saliency.set_anchor("N")
    ax_tracks.set_anchor("N")

    mesh = plot_saliency_panel(
        ax_saliency,
        saliency=saliency,
        baseline=baseline,
        selected_regions=selected_regions,
        smooth_sigma=smooth_sigma,
    )
    plot_track_panel(
        ax_tracks,
        baseline=baseline,
        perturbations=perturbations,
        max_lead=max_track_lead,
    )

    # A compact vertical scale in the inter-panel margin leaves the map height
    # available for data and avoids a wide bar underneath the saliency panel.
    colorbar_ax = ax_saliency.inset_axes([1.025, 0.10, 0.032, 0.52])
    cbar = fig.colorbar(
        mesh,
        cax=colorbar_ax,
        orientation="vertical",
        extend="both",
    )
    cbar.set_label(
        r"Signed saliency  [$10^{-5}$ hPa mm$^{-1}$]",
        fontsize=7.7,
        labelpad=4.8,
    )
    cbar.ax.tick_params(labelsize=7.2, length=2.5, pad=2.2)

    fig.suptitle(
        "Typhoon Nanmadol: ZWD saliency and causal forecast response",
        fontsize=15.0,
        fontweight="semibold",
        y=0.968,
    )
    fig.text(
        0.5,
        0.918,
        "Aurora initialized 17 September 2022, 12 UTC",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    study_root = resolve_study_root(args.study_root)
    png_path = make_figure(
        study_root=study_root,
        output_base=args.output_base,
        smooth_sigma=args.smooth_sigma,
        max_track_lead=args.max_track_lead,
        dpi=args.dpi,
    )
    print(f"Loaded saved outputs from {study_root}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
