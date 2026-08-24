"""Plot the lead-time evolution of Nanmadol intensity saliency.

This is a plotting-only script.  It reads the saved storm-centred saliency
products and never imports or runs Aurora.

The figure shows maps of the initialized-ZWD saliency magnitude for the +24,
+48, and +72 h intensity targets on a shared colour scale.  The maps are
smoothed for display; the peak annotations use the unsmoothed saliency.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "nanmadol_2022091712_zwd_tc_case_study"
RESULTS_ROOT = Path(os.environ.get("AURORA_XAI_RESULTS_DIR", REPO_ROOT / "results"))
EXTERNAL_STUDY_ROOT = RESULTS_ROOT / "zwd_tc_case_study" / STUDY_ID
REPO_STUDY_ROOT = REPO_ROOT / "results" / "zwd_tc_case_study" / STUDY_ID
DEFAULT_OUTPUT_BASE = REPO_ROOT / "results" / "img" / "fig_nanmadol_saliency_evolution"

INIT_LAT = 27.5
INIT_LON = 132.0
LEADS = (24, 48, 72)
DISPLAY_SCALE = 1.0e5
WINDOW_RADIUS_KM = 1550.0

# Retained for the optional storm-relative summary helper functions below.
LEAD_COLORS = {
    24: "#0072B2",
    48: "#D55E00",
    72: "#009E73",
}
QUADRANT_ORDER = ("north", "east", "south", "west")
QUADRANT_LABELS = ("N", "E", "S", "W")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Nanmadol intensity-saliency maps across forecast leads."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help=(
            "Directory containing storm_centered_xai/. Defaults to the scratch "
            "result if complete, then the repository copy."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Output path without suffix; .png is added.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in 0.25-degree grid cells (default: 3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300 dpi).",
    )
    return parser.parse_args()


def resolve_study_root(requested: Path | None) -> Path:
    candidates = (
        [requested]
        if requested is not None
        else [EXTERNAL_STUDY_ROOT, REPO_STUDY_ROOT]
    )
    required = [
        Path("storm_centered_xai/baseline_track.csv"),
        *[
            Path(f"storm_centered_xai/saliency_intensity_{lead:03d}h.npy")
            for lead in LEADS
        ],
    ]
    for candidate in candidates:
        if candidate is not None and all((candidate / item).exists() for item in required):
            return candidate
    checked = "\n  ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not find a complete saved Nanmadol saliency result. Checked:\n  "
        + checked
    )


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable two-dimensional Gaussian smoothing without scipy."""
    if sigma <= 0:
        return np.asarray(arr, dtype=np.float32)
    radius = max(1, int(4 * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    work = np.asarray(arr, dtype=np.float64)
    work = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=work
    )
    work = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=work
    )
    return work.astype(np.float32)


def destination_points(
    center_lon: float,
    center_lat: float,
    radius_km: float,
    *,
    samples: int = 361,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a spherical small circle as longitude/latitude coordinates."""
    earth_radius_km = 6371.0
    bearing = np.linspace(0.0, 2.0 * np.pi, samples)
    angular_distance = radius_km / earth_radius_km
    lat1 = np.radians(center_lat)
    lon1 = np.radians(center_lon)
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
    )
    lon = ((np.degrees(lon2) + 180.0) % 360.0) - 180.0
    return lon, np.degrees(lat2)


def add_basemap(
    ax,
    *,
    center_lat: float,
    center_lon: float,
    show_left_labels: bool,
) -> tuple[float, float, float, float]:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    lat_half = WINDOW_RADIUS_KM / 111.0
    lon_half = WINDOW_RADIUS_KM / (111.0 * max(np.cos(np.radians(center_lat)), 0.2))
    extent = (
        center_lon - lon_half,
        center_lon + lon_half,
        center_lat - lat_half,
        center_lat + lat_half,
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#666666", linewidth=0.62, zorder=4)
    ax.add_feature(cfeature.BORDERS, edgecolor="#888888", linewidth=0.32, zorder=4)
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.32,
        color="#888888",
        alpha=0.45,
        linestyle=":",
        zorder=3,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = show_left_labels
    gridlines.xlabel_style = {"size": 7.2, "color": "#333333"}
    gridlines.ylabel_style = {"size": 7.2, "color": "#333333"}
    gridlines.xlocator = mticker.MaxNLocator(4)
    gridlines.ylocator = mticker.MaxNLocator(4)
    return extent


def subset_for_extent(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin_deg = 0.5
    lat_idx = np.where(
        (lat >= extent[2] - margin_deg) & (lat <= extent[3] + margin_deg)
    )[0]
    lon_idx = np.where(
        (lon >= extent[0] - margin_deg) & (lon <= extent[1] + margin_deg)
    )[0]
    return lon[lon_idx], lat[lat_idx], field[np.ix_(lat_idx, lon_idx)]


def prepend_initial_point(track: pd.DataFrame) -> pd.DataFrame:
    if track.empty or float(track["lead_hours"].min()) <= 0:
        return track
    initial = {column: np.nan for column in track.columns}
    initial.update(
        {
            "lead_hours": 0.0,
            "center_lat": INIT_LAT,
            "center_lon": INIT_LON,
            "center_lon_plot": INIT_LON,
        }
    )
    return pd.concat([pd.DataFrame([initial]), track], ignore_index=True)


def plot_range_rings(
    ax,
    *,
    center_lat: float,
    center_lon: float,
    annotate: bool,
) -> None:
    import cartopy.crs as ccrs

    styles = (
        (312.0, "-", 0.9, "#202020"),
        (937.0, (0, (4, 3)), 0.75, "#555555"),
        (1250.0, (0, (4, 3)), 0.75, "#555555"),
    )
    for radius_km, linestyle, linewidth, color in styles:
        ring_lon, ring_lat = destination_points(center_lon, center_lat, radius_km)
        ax.plot(
            ring_lon,
            ring_lat,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

    if annotate:
        inner_lon, inner_lat = destination_points(center_lon, center_lat, 312.0)
        outer_lon, outer_lat = destination_points(center_lon, center_lat, 1090.0)
        ax.text(
            inner_lon[45],
            inner_lat[45],
            "312 km",
            fontsize=6.6,
            color="#202020",
            ha="left",
            va="bottom",
            transform=ccrs.PlateCarree(),
            zorder=9,
        )
        ax.text(
            outer_lon[205],
            outer_lat[205],
            "937–1250 km",
            fontsize=6.6,
            color="#444444",
            ha="center",
            va="top",
            transform=ccrs.PlateCarree(),
            zorder=9,
        )


def plot_map_panel(
    ax,
    *,
    lead: int,
    smoothed_magnitude: np.ndarray,
    raw_peak: float,
    lat: np.ndarray,
    lon: np.ndarray,
    track: pd.DataFrame,
    vmax: float,
    panel_label: str,
    show_left_labels: bool,
) -> object:
    import cartopy.crs as ccrs

    target = track.loc[np.isclose(track["lead_hours"], float(lead))]
    if len(target) != 1:
        raise ValueError(f"Expected one baseline-track row at +{lead} h, found {len(target)}")
    target = target.iloc[0]
    center_lat = float(target["center_lat"])
    center_lon = float(target["center_lon_plot"])
    extent = add_basemap(
        ax,
        center_lat=center_lat,
        center_lon=center_lon,
        show_left_labels=show_left_labels,
    )
    lon_sub, lat_sub, field_sub = subset_for_extent(
        smoothed_magnitude, lat, lon, extent
    )
    mesh = ax.pcolormesh(
        lon_sub,
        lat_sub,
        field_sub * DISPLAY_SCALE,
        cmap="Reds",
        shading="auto",
        vmin=0.0,
        vmax=vmax * DISPLAY_SCALE,
        alpha=0.9,
        rasterized=True,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    partial_track = prepend_initial_point(
        track[track["lead_hours"] <= float(lead)].sort_values("lead_hours").copy()
    )
    ax.plot(
        partial_track["center_lon_plot"],
        partial_track["center_lat"],
        color="#202020",
        linewidth=1.25,
        marker="o",
        markersize=2.2,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )
    ax.scatter(
        INIT_LON,
        INIT_LAT,
        marker="*",
        s=58,
        facecolor="white",
        edgecolor="#202020",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.scatter(
        center_lon,
        center_lat,
        marker="s",
        s=39,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    plot_range_rings(
        ax,
        center_lat=center_lat,
        center_lon=center_lon,
        annotate=(lead == 24),
    )
    ax.text(
        0.03,
        0.965,
        rf"peak $|S_{{{lead}}}|$: {raw_peak * DISPLAY_SCALE:.1f} $\times 10^{{-5}}$",
        transform=ax.transAxes,
        fontsize=7.0,
        ha="left",
        va="top",
        color="#202020",
        bbox={
            "facecolor": "white",
            "edgecolor": "#aaaaaa",
            "alpha": 0.86,
            "pad": 2.0,
        },
        zorder=12,
    )
    ax.set_title(
        rf"({panel_label}) Intensity target at $+{lead}$ h",
        fontsize=10.6,
        loc="left",
        pad=7,
    )
    ax.set_anchor("N")
    return mesh


def intensity_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[
        (summary["target_kind"] == "intensity")
        & (summary["lead_hours"].isin(LEADS))
    ].copy()
    if rows.empty:
        raise ValueError("quadrant_summary.csv has no intensity rows for the requested leads")
    return rows


def radial_profiles(summary: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    rows = intensity_rows(summary)
    rows = rows[rows["quadrant"].str.startswith("ring_")].copy()
    bounds = rows["quadrant"].str.extract(r"ring_(\d+)_(\d+)km").astype(float)
    rows["r0_km"] = bounds[0]
    rows["r1_km"] = bounds[1]
    rows["radius_km"] = 0.5 * (rows["r0_km"] + rows["r1_km"])
    radii = np.sort(rows["radius_km"].unique())
    profiles: dict[int, np.ndarray] = {}
    for lead in LEADS:
        lead_rows = rows[rows["lead_hours"] == lead].set_index("radius_km")
        profiles[lead] = lead_rows.loc[radii, "mean_abs_saliency"].to_numpy()
    return radii, profiles


def plot_radial_panel(ax, summary: pd.DataFrame, *, panel_label: str) -> float:
    radii, profiles = radial_profiles(summary)
    for lead in LEADS:
        ax.plot(
            radii,
            profiles[lead] * DISPLAY_SCALE,
            color=LEAD_COLORS[lead],
            linewidth=1.75,
            marker="o",
            markersize=3.8,
            label=rf"$+{lead}$ h",
        )
    ax.axvspan(0.0, 312.0, color="#777777", alpha=0.08, linewidth=0)
    ax.axvspan(937.0, 1250.0, color="#777777", alpha=0.08, linewidth=0)
    ax.set_xlim(0.0, 2500.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Distance from forecast target centre (km)")
    ax.set_ylabel(r"Mean $|$saliency$|$  [$10^{-5}$ hPa mm$^{-1}$]")
    ax.grid(True, color="#b0b0b0", linewidth=0.45, alpha=0.5, linestyle=":")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.1)

    rows = intensity_rows(summary).set_index(["lead_hours", "quadrant"])
    inner = float(rows.loc[(24, "ring_0_312km"), "mean_abs_saliency"])
    outer = float(rows.loc[(24, "ring_937_1250km"), "mean_abs_saliency"])
    ratio = inner / outer
    ax.text(
        0.97,
        0.69,
        rf"$+24$ h: inner / outer = {ratio:.1f}$\times$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9, "pad": 2.5},
    )
    ax.set_title(
        rf"({panel_label}) Radial concentration",
        fontsize=10.6,
        loc="left",
        pad=7,
    )
    return ratio


def quadrant_values(summary: pd.DataFrame) -> dict[int, np.ndarray]:
    rows = intensity_rows(summary)
    rows = rows[rows["quadrant"].isin(QUADRANT_ORDER)].copy()
    values: dict[int, np.ndarray] = {}
    for lead in LEADS:
        lead_rows = rows[rows["lead_hours"] == lead].set_index("quadrant")
        values[lead] = (
            lead_rows.loc[list(QUADRANT_ORDER), "mean_abs_saliency"].to_numpy()
        )
    return values


def plot_quadrant_panel(ax, summary: pd.DataFrame, *, panel_label: str) -> dict[int, str]:
    values = quadrant_values(summary)
    x = np.arange(len(QUADRANT_ORDER), dtype=float)
    width = 0.23
    for index, lead in enumerate(LEADS):
        offset = (index - 1) * width
        ax.bar(
            x + offset,
            values[lead] * DISPLAY_SCALE,
            width=width,
            color=LEAD_COLORS[lead],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            label=rf"$+{lead}$ h",
        )
    ax.set_xticks(x, QUADRANT_LABELS)
    ax.set_xlabel("Quadrant relative to forecast target centre")
    ax.set_ylabel(r"Mean $|$saliency$|$  [$10^{-5}$ hPa mm$^{-1}$]")
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.45, alpha=0.5, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.1)
    dominant = {
        lead: QUADRANT_LABELS[int(np.nanargmax(values[lead]))]
        for lead in LEADS
    }
    ax.text(
        0.97,
        0.95,
        "South is largest at every lead",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9, "pad": 2.5},
    )
    ax.set_title(
        rf"({panel_label}) Quadrant asymmetry (within 2500 km)",
        fontsize=10.6,
        loc="left",
        pad=7,
    )
    return dominant


def make_figure(
    *,
    study_root: Path,
    output_base: Path,
    smooth_sigma: float,
    dpi: int,
) -> tuple[Path, dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    xai_dir = study_root / "storm_centered_xai"
    track = pd.read_csv(xai_dir / "baseline_track.csv")
    raw = {
        lead: np.load(xai_dir / f"saliency_intensity_{lead:03d}h.npy")
        for lead in LEADS
    }
    shapes = {array.shape for array in raw.values()}
    if len(shapes) != 1:
        raise ValueError(f"Saliency arrays do not share one grid shape: {sorted(shapes)}")
    nlat, nlon = next(iter(shapes))
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon, dtype=np.float64) * (360.0 / nlon)
    smoothed = {
        lead: gaussian_smooth(np.abs(array), smooth_sigma)
        for lead, array in raw.items()
    }
    raw_peaks = {lead: float(np.nanmax(np.abs(array))) for lead, array in raw.items()}
    vmax = max(float(np.nanmax(array)) for array in smoothed.values())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.8,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.6, 4.85), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.075),
        left=0.055,
        right=0.955,
        bottom=0.16,
        top=0.79,
        wspace=0.36,
    )
    map_axes = [
        fig.add_subplot(grid[0:2], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[2:4], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[4:6], projection=ccrs.PlateCarree()),
    ]
    map_labels = ("a", "b", "c")
    mesh = None
    for index, (ax, lead, label) in enumerate(zip(map_axes, LEADS, map_labels)):
        mesh = plot_map_panel(
            ax,
            lead=lead,
            smoothed_magnitude=smoothed[lead],
            raw_peak=raw_peaks[lead],
            lat=lat,
            lon=lon,
            track=track,
            vmax=vmax,
            panel_label=label,
            show_left_labels=(index == 0),
        )

    colorbar_ax = fig.add_subplot(grid[6])
    colorbar = fig.colorbar(mesh, cax=colorbar_ax, orientation="vertical")
    colorbar.set_label(
        r"Smoothed $|$saliency$|$  [$10^{-5}$ hPa mm$^{-1}$]",
        fontsize=8.0,
        labelpad=5,
    )
    colorbar.ax.tick_params(labelsize=7.3, length=2.5, pad=2.2)

    fig.suptitle(
        "Typhoon Nanmadol: forecast-intensity sensitivity to initialized ZWD",
        fontsize=14.4,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.885,
        (
            r"$S_L(\mathbf{x}) = |\partial I_L/\partial \mathrm{ZWD}_{t_1}(\mathbf{x})|$; "
            r"each lead $L$ defines a different moving storm-centred intensity target"
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#444444",
    )
    fig.text(
        0.055,
        0.065,
        (
            rf"Maps show Gaussian-smoothed $|S_L|$ ($\sigma={0.25 * smooth_sigma:.2g}^\circ$) "
            "on a shared scale; labels show the unsmoothed maxima. "
            "Star: initialized centre; square: lead-specific forecast target centre."
        ),
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#444444",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    peak_ratios = {
        "24_to_48": raw_peaks[24] / raw_peaks[48],
        "48_to_72": raw_peaks[48] / raw_peaks[72],
    }
    diagnostics: dict[str, object] = {
        "raw_peaks": raw_peaks,
        "peak_ratios": peak_ratios,
    }
    return png_path, diagnostics


def main() -> None:
    args = parse_args()
    study_root = resolve_study_root(args.study_root)
    png_path, diagnostics = make_figure(
        study_root=study_root,
        output_base=args.output_base,
        smooth_sigma=args.smooth_sigma,
        dpi=args.dpi,
    )
    raw_peaks = diagnostics["raw_peaks"]
    peak_ratios = diagnostics["peak_ratios"]
    print(f"Loaded saved outputs from {study_root}")
    print(
        "Raw peak |saliency|: "
        + ", ".join(f"+{lead} h={raw_peaks[lead]:.6g}" for lead in LEADS)
    )
    print(
        "Successive peak ratios: "
        f"24/48={peak_ratios['24_to_48']:.3f}, "
        f"48/72={peak_ratios['48_to_72']:.3f}"
    )
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
