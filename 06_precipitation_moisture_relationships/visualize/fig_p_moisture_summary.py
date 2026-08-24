#!/usr/bin/env python3
"""Thesis summary of precipitation coupling to ZWD and humidity proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"zwd": "#D55E62", "humidity": "#3B75AF"}
LABELS = {"zwd": "ZWD", "humidity": "Humidity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correlations",
        type=Path,
        default=Path("results/p_correlation_diagnostics/correlations/correlations.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/img/fig_p_moisture_coupling.png"),
    )
    return parser.parse_args()


def family_means(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[(df["season"] == "ALL") & (df["transform"] == "monthly_z")].copy()
    selected["abs_spearman"] = selected["spearman"].abs()
    return (
        selected.groupby(["target", "lag_hours", "proxy_family"], as_index=False)
        ["abs_spearman"]
        .mean()
    )


def main() -> None:
    args = parse_args()
    means = family_means(pd.read_csv(args.correlations))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    fig, (ax_region, ax_lead) = plt.subplots(
        1, 2, figsize=(11.4, 4.35), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )

    # Panel (a): simultaneous associations by region plus the pooled estimate.
    targets = ["ticino", "california", "japan", "ALL"]
    display = ["Ticino", "California", "Japan", "Pooled"]
    lag_zero = means[(means["lag_hours"] == 0) & means["target"].isin(targets)]
    x = np.arange(len(targets))
    width = 0.34
    ax_region.axvspan(2.53, 3.47, color="#F2F2F2", zorder=0)
    for offset, family in zip((-width / 2, width / 2), ("zwd", "humidity")):
        vals = [
            lag_zero[(lag_zero["target"] == target) &
                     (lag_zero["proxy_family"] == family)]["abs_spearman"].iloc[0]
            for target in targets
        ]
        bars = ax_region.bar(x + offset, vals, width, color=COLORS[family],
                             label=LABELS[family])
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("#555555")
        bars[-1].set_linewidth(0.7)
        ax_region.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax_region.set_title("(a) Simultaneous reference by region", loc="left")
    ax_region.set_xticks(x, display)
    ax_region.set_ylim(0, 0.6)
    ax_region.set_ylabel(r"Mean absolute Spearman correlation $|\rho|$")
    ax_region.legend(frameon=False)
    ax_region.get_xticklabels()[-1].set_fontweight("bold")

    # Panel (b): the forecasting-oriented direction, with moisture at t and
    # precipitation at the later time t + tau.
    pooled = means[means["target"] == "ALL"]
    for family in ("zwd", "humidity"):
        values = pooled[pooled["proxy_family"] == family].sort_values("lag_hours")
        ax_lead.plot(
            values["lag_hours"], values["abs_spearman"], marker="o", markersize=6,
            linewidth=2.2, color=COLORS[family], label=LABELS[family],
        )
        for lead, value in zip(values["lag_hours"], values["abs_spearman"]):
            label_offset = 9 if family == "zwd" else -15
            ax_lead.annotate(
                f"{value:.3f}", (lead, value), xytext=(0, label_offset),
                textcoords="offset points", ha="center", fontsize=8,
                color=COLORS[family],
            )
    ax_lead.set_title("(b) Moisture leading precipitation", loc="left")
    ax_lead.set_xlabel(r"Moisture lead $\tau$ [h]: $M(t)$ vs. $P(t+\tau)$")
    ax_lead.set_ylabel(r"Mean absolute Spearman correlation $|\rho|$")
    ax_lead.set_xticks([0, 6, 12, 18, 24])
    ax_lead.set_ylim(0, 0.4)
    ax_lead.legend(frameon=False)

    for ax in (ax_region, ax_lead):
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Precipitation association with atmospheric moisture", fontsize=13)
    fig.text(
        0.5, 0.005,
        "Family means across monthly-standardised feature pairs; pooled values use all target-time samples",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), w_pad=2.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
