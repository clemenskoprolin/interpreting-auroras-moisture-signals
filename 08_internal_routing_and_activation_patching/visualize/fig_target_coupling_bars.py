#!/usr/bin/env python3
"""Compact thesis bar chart of raw target-level hotspot responses.

Each panel contains one output head and therefore one physical unit.  Bars show
the local median signed hotspot response H for perturbations of ZWD, q850, and
precipitation.  Independent linear axes are required because the three output
heads use different units.

Both checkpoints are shown.  The precipitation-only model has no ZWD input
channel and no ZWD output head, so its ZWD cells are empty by construction.

Provenance of the values:
  * ZWD/precipitation perturbation rows (``precip_zwd``):
    retained aggregate table in
    ``results/representation_trace/target_coupling_provenance.md``.
  * q850 -> precip/zwd, both checkpoints:
    ``results/representation_trace/q850_trace_all_pair_runs.csv``.
  * q850 -> q850, ``precip_zwd``:
    ``results/representation_trace/q850_self_trace_all_pair_runs.csv``.
  * q850 -> q850 and precip -> q850, ``precip_only``: final 22-case aggregate.

Run on the login node (this script does not import Aurora):

    source .venv/bin/activate
    python 08_internal_routing_and_activation_patching/visualize/fig_target_coupling_bars.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "img" / "fig_target_coupling_bars.png"
SHARED_XLIM = (-6.0, 55.0)
SHARED_TICKS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
STANDARDIZED_XLIM = (-0.08, 0.58)
STANDARDIZED_TICKS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Global climatological scales used to define the 1-sigma interventions.
# q850 is converted from kg/kg in the stats file to the g/kg plotted here.
OUTPUT_SCALES = {
    "q850": 4.075226373970509,
    "precip": 2.11,
    "zwd": 98.5413,
}
STANDARDIZED_UNITS = {
    "q850": r"$\sigma_{q_{850}}$",
    "precip": r"$\sigma_{\mathrm{precip}}$",
    "zwd": r"$\sigma_{\mathrm{ZWD}}$",
}

POSITIVE = "#B2182B"
NEGATIVE = "#2166AC"
MISSING = "#E7E9EB"
# Both checkpoints are drawn in every panel.  Hue keeps encoding the sign of the
# response; the checkpoint is encoded by fill style so that the sign remains
# readable in the grouped bars.
MODELS = ("precip_zwd", "precip_only")
MODEL_LABELS = {
    "precip_zwd": "precipitation + ZWD model",
    "precip_only": "precipitation-only model",
}
MODEL_STYLE = {
    "precip_zwd": {"alpha": 0.94, "hatch": None},
    "precip_only": {"alpha": 0.42, "hatch": "////"},
}
GROUP_OFFSET = 0.17
GROUP_HEIGHT = 0.30
SINGLE_HEIGHT = 0.56
GRID = "#D9DDE1"
TEXT = "#202326"
SECONDARY_TEXT = "#62676D"

# Match the vertical input order to the horizontal output-panel order so that
# self-responses occupy the visual diagonal from top left to bottom right.
INPUTS = ("q850", "precip", "zwd")
INPUT_LABELS = {
    "zwd": "ZWD",
    "q850": r"$q_{850}$",
    "precip": "precipitation",
}


@dataclass(frozen=True)
class OutputPanel:
    key: str
    title: str
    unit: str
    # input name -> checkpoint -> value (None = measured slot with no number yet)
    values: dict[str, dict[str, float | None]]
    value_labels: dict[str, dict[str, str]]
    xlim: tuple[float, float]
    ticks: tuple[float, ...]


# The precipitation-only checkpoint has neither a ZWD input channel nor a ZWD
# output head, so every cell touching ZWD is empty by construction rather than
# unmeasured.  ``_structurally_absent`` distinguishes the two cases.
def _structurally_absent(panel_key: str, input_name: str, model: str) -> bool:
    return model == "precip_only" and "zwd" in (panel_key, input_name)


PANELS = (
    OutputPanel(
        key="q850",
        title=r"$q_{850}$ output",
        unit=r"g kg$^{-1}$",
        values={
            "zwd": {"precip_zwd": +0.143, "precip_only": None},
            "q850": {"precip_zwd": +0.918, "precip_only": +0.967},
            "precip": {"precip_zwd": +0.0002, "precip_only": +0.0041},
        },
        value_labels={
            "zwd": {"precip_zwd": "+0.143", "precip_only": ""},
            "q850": {"precip_zwd": "+0.918", "precip_only": "+0.967"},
            "precip": {"precip_zwd": "+0.0002", "precip_only": "+0.0041"},
        },
        xlim=(-0.052, 1.060),
        ticks=(0.00, 0.25, 0.50, 0.75, 1.00),
    ),
    OutputPanel(
        key="precip",
        title="precipitation output",
        unit="mm / 6 h",
        values={
            "zwd": {"precip_zwd": +0.218, "precip_only": None},
            "q850": {"precip_zwd": +0.265, "precip_only": +0.313},
            "precip": {"precip_zwd": +0.316, "precip_only": +0.346},
        },
        value_labels={
            "zwd": {"precip_zwd": "+0.218", "precip_only": ""},
            "q850": {"precip_zwd": "+0.265", "precip_only": "+0.313"},
            "precip": {"precip_zwd": "+0.316", "precip_only": "+0.346"},
        },
        xlim=(-0.018, 0.390),
        ticks=(0.0, 0.1, 0.2, 0.3),
    ),
    OutputPanel(
        key="zwd",
        title="ZWD output",
        unit="mm",
        values={
            "zwd": {"precip_zwd": +52.0, "precip_only": None},
            "q850": {"precip_zwd": +0.659, "precip_only": None},
            "precip": {"precip_zwd": -0.198, "precip_only": None},
        },
        value_labels={
            "zwd": {"precip_zwd": "+52.0", "precip_only": ""},
            "q850": {"precip_zwd": "+0.659", "precip_only": ""},
            "precip": {"precip_zwd": "−0.198", "precip_only": ""},
        },
        xlim=(-6.0, 55.0),
        ticks=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    scale_mode = parser.add_mutually_exclusive_group()
    scale_mode.add_argument(
        "--shared-scale",
        action="store_true",
        help="Use the ZWD panel's numeric axis range for all three outputs.",
    )
    scale_mode.add_argument(
        "--standardized",
        action="store_true",
        help="Divide each response by the climatological scale of its output.",
    )
    return parser.parse_args()


def _tick_formatter(value: float, _position: float) -> str:
    if abs(value) >= 1:
        return f"{value:.0f}"
    if value == 0:
        return "0"
    return f"{value:.2f}".rstrip("0")


def _standardized_label(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    magnitude = abs(value)
    if magnitude < 0.001:
        return f"{sign}{magnitude:.5f}"
    if magnitude < 0.01:
        return f"{sign}{magnitude:.4f}"
    return f"{sign}{magnitude:.3f}"


def _plot(
    output: Path,
    dpi: int,
    shared_scale: bool = False,
    standardized: bool = False,
) -> None:
    if output.suffix.lower() != ".png":
        raise ValueError("This repository uses PNG figure output; --output must end in .png")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(6.25, 2.85),
        sharey=True,
        facecolor="white",
    )
    y_positions = np.arange(len(INPUTS), dtype=float)

    for panel_index, (ax, panel) in enumerate(zip(axes, PANELS)):
        if standardized:
            xlim = STANDARDIZED_XLIM
            ticks = STANDARDIZED_TICKS
        elif shared_scale:
            xlim = SHARED_XLIM
            ticks = SHARED_TICKS
        else:
            xlim = panel.xlim
            ticks = panel.ticks
        ax.set_facecolor("white")
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(_tick_formatter))
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.axvline(0.0, color="#8A8F94", lw=0.9, zorder=2)

        span = xlim[1] - xlim[0]
        for y, input_name in zip(y_positions, INPUTS):
            slots = [
                m for m in MODELS
                if not _structurally_absent(panel.key, input_name, m)
            ]
            grouped = len(slots) > 1

            for model in slots:
                if grouped:
                    y_bar = y + (GROUP_OFFSET if model == "precip_only" else -GROUP_OFFSET)
                    height = GROUP_HEIGHT
                else:
                    y_bar = y
                    height = SINGLE_HEIGHT

                raw_value = panel.values[input_name][model]
                if raw_value is None:
                    ax.barh(
                        y_bar,
                        0.035 * span,
                        left=0.0,
                        height=height,
                        color=MISSING,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=3,
                    )
                    ax.text(
                        0.045 * span,
                        y_bar,
                        "not measured",
                        ha="left",
                        va="center",
                        color="#8A8F94",
                        fontsize=6.4 if grouped else 7.0,
                        zorder=4,
                    )
                    continue

                value = raw_value / OUTPUT_SCALES[panel.key] if standardized else raw_value
                colour = POSITIVE if value >= 0 else NEGATIVE
                style = MODEL_STYLE[model]
                ax.barh(
                    y_bar,
                    value,
                    left=0.0,
                    height=height,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    zorder=3,
                )

                label = (
                    _standardized_label(value)
                    if standardized
                    else panel.value_labels[input_name][model]
                )
                inside_label_threshold = (0.25 if standardized else 0.22) * span
                if abs(value) >= inside_label_threshold:
                    label_x = value - np.sign(value) * 0.025 * span
                    label_ha = "right" if value > 0 else "left"
                    label_colour = "white" if model == "precip_zwd" else TEXT
                else:
                    offset = (0.018 if standardized else 0.035) * span
                    if standardized:
                        label_x = value + offset if value >= 0 else value - offset
                    else:
                        label_x = offset if value >= 0 else -offset
                    label_ha = "left" if value >= 0 else "right"
                    label_colour = TEXT
                ax.text(
                    label_x,
                    y_bar,
                    label,
                    ha=label_ha,
                    va="center",
                    color=label_colour,
                    fontsize=6.8 if grouped else 7.5,
                    fontweight="semibold",
                    zorder=4,
                )

        unit = STANDARDIZED_UNITS[panel.key] if standardized else panel.unit
        ax.set_title(f"{panel.title}\n[{unit}]", pad=8)
        ax.set_ylim(len(INPUTS) - 0.5, -0.5)
        ax.tick_params(axis="x", length=3, color="#8A8F94")
        ax.tick_params(axis="y", length=0, pad=7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#8A8F94")
        ax.spines["bottom"].set_linewidth(0.8)

        if panel_index == 0:
            ax.set_yticks(y_positions, [INPUT_LABELS[x] for x in INPUTS])
            ax.set_ylabel("Perturbed input", labelpad=10, fontweight="semibold")

    xlabel = (
        "Local median signed hotspot response (output standard deviations)"
        if standardized
        else "Local median signed hotspot response"
    )
    fig.supxlabel(xlabel, y=0.085, fontsize=8.7)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.72, bottom=0.25, wspace=0.30)

    legend_handles = [
        mpl.patches.Patch(
            facecolor="#6E7378",
            edgecolor="white",
            linewidth=0.8,
            alpha=MODEL_STYLE[model]["alpha"],
            hatch=MODEL_STYLE[model]["hatch"],
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handlelength=1.6,
        handleheight=0.9,
        columnspacing=1.4,
    )
    fig.text(
        0.20,
        0.028,
        (
            "Each response divided by its output's climatological σ; red/blue = positive/negative response"
            if standardized
            else (
                "Shared numeric axis range despite different output units; red/blue = positive/negative response"
                if shared_scale
                else "Independent linear axes in each output unit; red/blue = positive/negative response"
            )
        )
        + "\nThe precipitation-only checkpoint has no ZWD input channel and no ZWD output head, so its ZWD cells are empty by construction.",
        ha="left",
        va="center",
        fontsize=7.0,
        color=SECONDARY_TEXT,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = _parse_args()
    _plot(
        args.output,
        args.dpi,
        shared_scale=args.shared_scale,
        standardized=args.standardized,
    )


if __name__ == "__main__":
    main()
