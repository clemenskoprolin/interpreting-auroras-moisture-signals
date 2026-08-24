#!/usr/bin/env python3
"""
Vertical profile of ZWD–humidity correlation by pressure level.

Shows Spearman(ZWD_disk_mean, q_<level>_disk_mean) plotted against pressure
level (hPa, log-scale, inverted so the surface is at the bottom). One line
per target region, plus a thick pooled "ALL" line. Solid = raw correlations,
dashed = monthly_z anomaly correlations.

The figure shows the ~0.97 Spearman peak near 850–700 hPa and the rapid
drop-off above 500 hPa, confirming that ZWD represents lower-to-mid-
tropospheric moisture rather than upper-level structure.

Usage:
  source .venv/bin/activate
  python 03_zwd_moisture_relationships/visualize/fig_zwd_vertical_profile.py
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(
    os.environ.get("AURORA_XAI_RESULTS_DIR", os.path.join(_ROOT, "results")),
    "zwd_correlation_diagnostics",
)

AURORA_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

TARGETS = ["ticino", "california", "japan", "ALL"]
TARGET_COLORS = {
    "ticino": "#4477AA",
    "california": "#EE6677",
    "japan": "#228833",
    "ALL": "#333333",
}
TARGET_LABELS = {
    "ticino": "Ticino",
    "california": "N. California",
    "japan": "Central Honshu",
    "ALL": "Pooled (ALL)",
}
TARGET_LW = {"ALL": 2.2, "ticino": 1.3, "california": 1.3, "japan": 1.3}
TARGET_ALPHA = {"ALL": 1.0, "ticino": 0.82, "california": 0.82, "japan": 0.82}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", default=None, help="Defaults to <results-dir>/viz/")
    p.add_argument("--zwd-col", default="zwd_disk_mean",
                   help="ZWD column to use (default: zwd_disk_mean)")
    p.add_argument("--region", default="disk", choices=["disk", "box"],
                   help="Region suffix for humidity columns (default: disk)")
    return p.parse_args()


def col_to_level(col: str, region: str) -> int | None:
    m = re.match(rf"q_(\d+)_{region}_mean$", col)
    return int(m.group(1)) if m else None


def main() -> None:
    args = parse_args()
    corr_path = os.path.join(args.results_dir, "correlations", "correlations.csv")
    df = pd.read_csv(corr_path)
    out_dir = args.output_dir or os.path.join(args.results_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    # Keep only lag-0, selected ZWD column, humidity q_<level> proxies
    df = df[
        (df["lag_hours"] == 0)
        & (df["zwd_column"] == args.zwd_col)
        & (df["proxy_family"] == "humidity")
    ].copy()

    df["level"] = df["proxy_column"].apply(lambda c: col_to_level(c, args.region))
    df = df.dropna(subset=["level"])
    df["level"] = df["level"].astype(int)

    # Keep only the standard Aurora pressure levels
    df = df[df["level"].isin(AURORA_LEVELS)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True,
                             sharey=True)
    transforms = [("raw", "Raw correlations"), ("monthly_z", "Monthly anomaly correlations")]

    for ax, (transform, title) in zip(axes, transforms):
        sub = df[df["transform"] == transform]
        # Draw per-target lines (excluding ALL first, then ALL on top)
        for tgt in TARGETS:
            t_sub = sub[sub["target"] == tgt].sort_values("level")
            if t_sub.empty:
                continue
            lw = TARGET_LW.get(tgt, 1.3)
            alpha = TARGET_ALPHA.get(tgt, 0.8)
            ls = "-" if tgt == "ALL" else "--"
            ax.plot(
                t_sub["spearman"], t_sub["level"],
                color=TARGET_COLORS[tgt],
                lw=lw, alpha=alpha, linestyle=ls,
                marker="o" if tgt == "ALL" else None,
                markersize=5,
                label=TARGET_LABELS[tgt],
                zorder=3 if tgt == "ALL" else 2,
            )

        ax.axvline(0.9, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.axvline(0.7, color="gray", lw=0.8, linestyle=":", alpha=0.4)
        ax.text(0.905, 200, "ρ=0.9", fontsize=8, color="gray", va="center")
        ax.text(0.705, 200, "ρ=0.7", fontsize=8, color="gray", va="center")

        ax.set_xlabel("Spearman correlation", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xbound(lower=0.0, upper=1.02)
        ax.grid(alpha=0.22)
        if ax is axes[0]:
            ax.set_ylabel("Pressure level (hPa)", fontsize=10)
            ax.legend(fontsize=9, loc="upper left")

    # Pressure axis: log-scale, inverted (surface at bottom)
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_ylim(1050, 40)
    ax.set_yticks(list(AURORA_LEVELS))
    ax.set_yticklabels([str(lv) for lv in AURORA_LEVELS], fontsize=8.5)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    # Highlight 850 hPa reference
    for ax in axes:
        ax.axhline(850, color="#4477AA", lw=0.8, linestyle=":", alpha=0.5)
        ax.text(0.01, 870, "850 hPa", fontsize=7.5, color="#4477AA", va="top",
                transform=ax.get_yaxis_transform())

    fig.suptitle(
        f"Vertical profile of ZWD–humidity coupling\n"
        f"Spearman({args.zwd_col}, q_<level>_{args.region}_mean)  ·  lag=0",
        fontsize=12,
    )

    out_path = os.path.join(out_dir, "fig_zwd_vertical_profile.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
