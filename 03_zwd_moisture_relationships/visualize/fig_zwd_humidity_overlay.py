#!/usr/bin/env python3
"""
ZWD vs column-humidity overlay — showing they are virtually the same signal.

Layout:
  Row 1 (a, b, c): One panel per target region, 12-month z-scored time-series
    overlay. ZWD and q_column, both z-scored over the full record per target,
    are plotted on the same axis. The two lines nearly coincide.

  Row 2:
    (d) Scatter of all ~10 k samples: ZWD_z vs q_column_z, coloured by target.
        The 1:1 line and pooled Spearman annotated.
    (e) Residual (ZWD_z − q_column_z) over the full 5-year record. The top-2
        largest-residual dates per target are annotated — these are the moments
        when ZWD carries information beyond column humidity alone.

  Row 3 (spatial context — requires zarr access):
    (f) ZWD field over Europe at the Ticino strong-event date (2020-04-20 12 UTC).
    (g) q@850 hPa at the same timestamp and extent.
    Both use viridis so spatial patterns are directly comparable. The Ticino
    target box is marked in black.

Usage:
  source .venv/bin/activate
  python 03_zwd_moisture_relationships/visualize/fig_zwd_humidity_overlay.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "zwd_correlation_diagnostics",
)

ZWD_ZARR = os.environ.get(
    "AURORA_ZWD_DATA",
    "/capstor/store/cscs/swissai/a01/ZWDX/era5/zwd_data_1h_lead_time.zarr",
)
_FALLBACK_WB_PATHS = (
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
    "/capstor/store/cscs/swissai/a01/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
WB_PATHS = tuple(
    path for path in os.environ.get(
        "AURORA_WB2_STORES", os.pathsep.join(_FALLBACK_WB_PATHS)
    ).split(os.pathsep) if path
)

# Ticino strong-event date used for the map panels
MAP_TIMESTAMP = pd.Timestamp("2020-04-20 12:00")
# Europe extent for maps: (lon_w, lon_e, lat_s, lat_n)
MAP_EXTENT = (-10.0, 35.0, 35.0, 60.0)
# Ticino target box (hardcoded to keep this script standalone)
TICINO_BOX = dict(lon_w=7.5, lon_e=10.0, lat_s=45.5, lat_n=47.0)

TARGETS = ["ticino", "california", "japan"]
TARGET_COLORS = {"ticino": "#4477AA", "california": "#EE6677", "japan": "#228833"}
TARGET_LABELS = {"ticino": "Ticino", "california": "N. California", "japan": "Central Honshu"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default=None, help="Defaults to <results-dir>/viz/")
    p.add_argument(
        "--window-year",
        type=int,
        default=2021,
        help="Calendar year shown in the time-series overlay (default: 2021)",
    )
    p.add_argument("--zwd-col", default="zwd_disk_mean")
    p.add_argument("--q-col", default="q_column_disk_mean")
    p.add_argument(
        "--no-maps", action="store_true",
        help="Skip the spatial map row (useful if zarr is unavailable)",
    )
    return p.parse_args()


def _zscore_per_target(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score each target's column independently over the full record."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for tgt, sub in df.groupby("target"):
        v = sub[col].replace([np.inf, -np.inf], np.nan)
        m, s = float(v.mean()), float(v.std(ddof=0))
        out.loc[sub.index] = (v - m) / (s + 1e-30)
    return out


def _crop_to_extent(data, lat, lon, extent):
    """Crop a (H, W) array and lat/lon vectors to a lon/lat extent box."""
    lon_w, lon_e, lat_s, lat_n = extent
    # ERA5/WB2: lat descending (90→−90), lon ascending 0→360
    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    # Extent given in −180…180; convert to 0…360 for the data
    lon_w360 = lon_w % 360.0
    lon_e360 = lon_e % 360.0
    if lon_w360 <= lon_e360:
        lon_mask = (lon >= lon_w360) & (lon <= lon_e360)
    else:
        lon_mask = (lon >= lon_w360) | (lon <= lon_e360)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    return data[np.ix_(lat_idx, lon_idx)], lat[lat_idx], lon[lon_idx]


def _load_map_fields(timestamp: pd.Timestamp) -> tuple | None:
    """Load ZWD and q@850 for the given timestamp. Returns None on failure."""
    try:
        import xarray as xr
    except ImportError:
        print("[warn] xarray not available; skipping map panels.")
        return None
    try:
        print(f"Loading ZWD from zarr at {timestamp} …")
        ds_zwd = xr.open_zarr(ZWD_ZARR)
        zwd = ds_zwd["zenith_wet_delay"].sel(time=timestamp).values.astype(np.float32)
        lat = ds_zwd["latitude"].values
        lon = ds_zwd["longitude"].values
    except Exception as e:
        print(f"[warn] Could not load ZWD from zarr: {e}")
        return None
    q850 = None
    for wb_path in WB_PATHS:
        if not os.path.exists(wb_path):
            continue
        try:
            ds_wb = xr.open_zarr(wb_path)
            times = pd.DatetimeIndex(ds_wb.time.values)
            if timestamp not in times:
                continue
            print(f"Loading q@850 from {wb_path} …")
            q850 = (
                ds_wb["specific_humidity"]
                .sel(time=timestamp, level=850)
                .values.astype(np.float32)
            ) * 1000.0  # kg/kg → g/kg
            break
        except Exception as e:
            print(f"[warn] Could not load WB2 from {wb_path}: {e}")
    if q850 is None:
        print("[warn] q@850 not found in any WB2 store; skipping map panels.")
        return None
    return zwd, q850, lat, lon


def _add_basemap(ax):
    import cartopy.feature as cfeat
    ax.add_feature(cfeat.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeat.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeat.COASTLINE, edgecolor="gray", linewidth=0.6, zorder=2)
    ax.add_feature(cfeat.BORDERS, edgecolor="gray", linewidth=0.35, linestyle=":", zorder=2)


def _add_ticino_box(ax):
    import cartopy.crs as ccrs
    b = TICINO_BOX
    ax.add_patch(Rectangle(
        (b["lon_w"], b["lat_s"]), b["lon_e"] - b["lon_w"], b["lat_n"] - b["lat_s"],
        fill=False, edgecolor="black", linewidth=1.8,
        transform=ccrs.PlateCarree(), zorder=5,
    ))


def _plot_map_field(fig, ax, data, lat, lon, extent, cmap, vmin, vmax, label, title):
    """Plot a cropped field on a Cartopy axes."""
    import cartopy.crs as ccrs
    data_c, lat_c, lon_c = _crop_to_extent(data, lat, lon, extent)
    # Convert lon to −180…180 for display
    lon_plot = np.where(lon_c > 180, lon_c - 360, lon_c)
    sort_idx = np.argsort(lon_plot)
    data_c = data_c[:, sort_idx]
    lon_plot = lon_plot[sort_idx]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    _add_basemap(ax)
    _add_ticino_box(ax)

    # pcolormesh needs cell-corner coords → compute midpoints as edges
    lon_edges = np.concatenate([
        [lon_plot[0] - 0.125],
        0.5 * (lon_plot[:-1] + lon_plot[1:]),
        [lon_plot[-1] + 0.125],
    ])
    lat_edges = np.concatenate([
        [lat_c[0] + 0.125],
        0.5 * (lat_c[:-1] + lat_c[1:]),
        [lat_c[-1] - 0.125],
    ])
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, data_c,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), zorder=1,
    )
    fig.colorbar(mesh, ax=ax, shrink=0.75, label=label, extend="both")
    ax.set_title(title, fontsize=10)
    return mesh


def main() -> None:
    args = parse_args()
    ts_path = os.path.join(args.results_dir, "timeseries", "all_targets_timeseries.csv")
    df = (
        pd.read_csv(ts_path, parse_dates=["init_time"])
        .set_index("init_time")
        .sort_index()
    )
    out_dir = args.output_dir or os.path.join(args.results_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    zwd_col = args.zwd_col
    q_col = args.q_col

    # Compute z-scores as separate series then rebuild to avoid fragmentation
    zwd_z = _zscore_per_target(df, zwd_col).rename("zwd_z")
    q_col_z = _zscore_per_target(df, q_col).rename("q_col_z")
    extras = pd.concat([zwd_z, q_col_z], axis=1)
    extras["resid"] = extras["zwd_z"] - extras["q_col_z"]
    df = pd.concat([df, extras], axis=1).copy()

    # Try loading map data (row 3)
    map_data = None if args.no_maps else _load_map_fields(MAP_TIMESTAMP)
    n_rows = 3 if map_data is not None else 2

    fig = plt.figure(
        figsize=(18, 15 if n_rows == 3 else 10),
        constrained_layout=True,
    )
    height_ratios = [1.1, 1.0, 1.1] if n_rows == 3 else [1.1, 1.0]
    gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios)
    ax_ts = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_scatter = fig.add_subplot(gs[1, 0:2])
    ax_resid = fig.add_subplot(gs[1, 2])

    window_start = pd.Timestamp(f"{args.window_year}-01-01")
    window_end = pd.Timestamp(f"{args.window_year + 1}-01-01")

    # ---- Row 1: per-target time-series overlay ----
    for ax, tgt in zip(ax_ts, TARGETS):
        col = TARGET_COLORS[tgt]
        sub = df[df["target"] == tgt]
        win = sub.loc[window_start:window_end]

        ax.plot(win.index, win["q_col_z"], color=col, lw=1.6, alpha=0.9,
                label="q_column  (z-score)")
        ax.plot(win.index, win["zwd_z"], color="black", lw=1.0, alpha=0.65,
                linestyle="--", label="ZWD  (z-score)")

        rho, _ = spearmanr(
            sub["zwd_z"].dropna(),
            sub["q_col_z"].reindex(sub["zwd_z"].dropna().index).dropna(),
        )
        ax.text(
            0.03, 0.97, f"ρ = {rho:.3f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        label = chr(ord("a") + TARGETS.index(tgt))
        ax.set_title(f"({label}) {TARGET_LABELS[tgt]}  ·  {args.window_year}", fontsize=10)
        ax.set_ylabel("z-score", fontsize=9)
        ax.grid(alpha=0.22)
        ax.axhline(0, color="gray", lw=0.5)
        if tgt == "ticino":
            ax.legend(fontsize=8.5, loc="upper right")

    # ---- Bottom-left: scatter (all years, all targets) ----
    for tgt in TARGETS:
        sub = df[df["target"] == tgt]
        ax_scatter.scatter(
            sub["q_col_z"], sub["zwd_z"],
            s=5, alpha=0.22, color=TARGET_COLORS[tgt],
            label=TARGET_LABELS[tgt], rasterized=True,
        )

    lim = 4.5
    ax_scatter.plot([-lim, lim], [-lim, lim], color="black", lw=0.9,
                    linestyle="--", alpha=0.55, zorder=1, label="1:1")
    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_xlabel("q_column_disk_mean  (z-scored per target)", fontsize=9)
    ax_scatter.set_ylabel("ZWD_disk_mean  (z-scored per target)", fontsize=9)
    ax_scatter.set_title("(d) ZWD vs column humidity — all years, all targets", fontsize=10)
    ax_scatter.grid(alpha=0.2)
    ax_scatter.legend(fontsize=8.5, markerscale=3, loc="upper left")

    valid = df[["zwd_z", "q_col_z"]].dropna()
    rho_all, _ = spearmanr(valid["zwd_z"], valid["q_col_z"])
    ax_scatter.text(
        0.97, 0.04, f"pooled ρ = {rho_all:.3f}",
        transform=ax_scatter.transAxes, fontsize=9.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
    )

    # ---- Bottom-right: residual over full record ----
    for tgt in TARGETS:
        sub = df[df["target"] == tgt].copy()
        col = TARGET_COLORS[tgt]
        ax_resid.plot(sub.index, sub["resid"], color=col, lw=0.7, alpha=0.75,
                      label=TARGET_LABELS[tgt])
        top = sub["resid"].abs().nlargest(2).index
        for ts in top:
            val = float(sub.loc[ts, "resid"])
            ax_resid.annotate(
                ts.strftime("%Y-%m-%d"),
                xy=(ts, val),
                xytext=(8, 10 if val > 0 else -14),
                textcoords="offset points",
                fontsize=6.5, color=col,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.5),
            )

    ax_resid.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax_resid.set_title("(e) Residual: ZWD_z − q_column_z", fontsize=10)
    ax_resid.set_ylabel("residual z-score", fontsize=9)
    ax_resid.legend(fontsize=8.5, loc="upper right")
    ax_resid.grid(alpha=0.22)
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    ax_resid.xaxis.set_major_locator(mdates.YearLocator())

    # ---- Row 3: spatial maps (if data available) ----
    if map_data is not None:
        import cartopy.crs as ccrs
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        zwd_field, q850_field, lat, lon = map_data

        # Use a nested 2-column sub-grid so both maps are exactly equal width
        gs_maps = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.08)
        ax_zwd = fig.add_subplot(gs_maps[0], projection=ccrs.PlateCarree())
        ax_q = fig.add_subplot(gs_maps[1], projection=ccrs.PlateCarree())

        zwd_c, lat_c, lon_c = _crop_to_extent(zwd_field, lat, lon, MAP_EXTENT)
        zwd_vmax = float(np.nanpercentile(zwd_c, 99))
        zwd_vmin = float(np.nanpercentile(zwd_c, 1))

        _plot_map_field(
            fig, ax_zwd, zwd_field, lat, lon,
            extent=MAP_EXTENT,
            cmap="viridis", vmin=zwd_vmin, vmax=zwd_vmax,
            label="ZWD [mm]",
            title=f"(f) Zenith Wet Delay — Europe  ·  {MAP_TIMESTAMP:%Y-%m-%d %H:%M UTC}",
        )

        q850_c, _, _ = _crop_to_extent(q850_field, lat, lon, MAP_EXTENT)
        q_vmax = float(np.nanpercentile(q850_c, 99))
        q_vmin = float(np.nanpercentile(q850_c, 1))

        _plot_map_field(
            fig, ax_q, q850_field, lat, lon,
            extent=MAP_EXTENT,
            cmap="viridis", vmin=q_vmin, vmax=q_vmax,
            label="q@850 [g kg⁻¹]",
            title=f"(g) Specific humidity @850 hPa  ·  {MAP_TIMESTAMP:%Y-%m-%d %H:%M UTC}",
        )

        for ax in (ax_zwd, ax_q):
            ax.text(
                0.02, 0.04,
                "■ Ticino target",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

    fig.suptitle(
        f"ZWD ≈ column specific humidity — {args.window_year} overlay, full-record scatter, and spatial snapshot\n"
        "Both z-scored independently per target over the full 2020–2024 record",
        fontsize=12,
    )

    out_path = os.path.join(out_dir, "fig_zwd_humidity_overlay.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
