#!/usr/bin/env python3
"""Plot the controlled all-input activation-patching comparison.

The underlying experiment holds the model, cases, target, patch sites, regions,
and lead times fixed.  For each selected variable, the source run uses the
observed field and the baseline run removes only its horizontal structure:

* ZWD and MSWEP precipitation use a per-timestep spatial mean.
* Specific humidity uses a per-timestep/per-level spatial mean.

Run on the login node (this script does not import Aurora):

    source .venv/bin/activate
    python 08_internal_routing_and_activation_patching/visualize/fig_large_activation_patch_unified.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[2]
SECTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "AURORA_ACTIVATION_PATCH_RESULTS_DIR",
        SECTION_ROOT / "results" / "activation_patch_all_inputs",
    )
)
DEFAULT_OUTPUT = ROOT / "results" / "img" / "fig_large_activation_patch_unified_lead6.png"

VARIABLES = ("zwd", "q", "precip")
VARIABLE_TITLES = {
    "zwd": "ZWD\nactual vs spatial mean",
    "q": "specific humidity q\nactual vs levelwise spatial mean",
    "precip": "MSWEP precipitation\nactual vs spatial mean",
}
EXPECTED_BASELINES = {
    "zwd": "per-timestep spatial mean",
    "q": "per-timestep/per-level spatial mean",
    "precip": "per-timestep spatial mean",
}

SITE_ORDER = [
    "enc_s0_skip",
    "enc_s1_skip",
    "enc_s2_bottleneck",
    "dec_s0_pre_skip",
    "dec_s1_pre_skip",
    "dec_s1_post_skip",
    "dec_s2_pre_concat",
    "dec_s2_post_concat",
]
SITE_LABELS = {
    "enc_s0_skip": "enc s0 skip\nhi-res",
    "enc_s1_skip": "enc s1 skip\nmid-res",
    "enc_s2_bottleneck": "enc s2\nbottleneck",
    "dec_s0_pre_skip": "dec s0\npre-skip",
    "dec_s1_pre_skip": "dec s1\npre-skip",
    "dec_s1_post_skip": "dec s1\npost-skip",
    "dec_s2_pre_concat": "dec s2\npre-concat",
    "dec_s2_post_concat": "dec s2\npost-concat",
}
REGION_ORDER = [
    "whole",
    "hotspot_gaussian",
    "target_box",
    "upstream",
    "mountain_near",
    "low_near_gaussian",
    "remote_control",
]
REGION_LABELS = {
    "whole": "whole\nfield",
    "hotspot_gaussian": "hotspot",
    "target_box": "target\nbox",
    "upstream": "upstream",
    "mountain_near": "mountain\nnear",
    "low_near_gaussian": "low-near\ncontrol",
    "remote_control": "remote\ncontrol",
}

KEY_COLUMNS = ["case_id", "lead_h", "patch_site", "patch_region"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lead", type=int, default=6)
    return parser.parse_args()


def _load_variable(data_root: Path, variable: str) -> tuple[pd.DataFrame, dict]:
    variable_root = data_root / variable
    score_paths = sorted(variable_root.glob("shard*/activation_patch_scores.csv"))
    config_paths = sorted(variable_root.glob("shard*/config.json"))
    direct_score = variable_root / "activation_patch_scores.csv"
    direct_config = variable_root / "config.json"
    if direct_score.is_file() and direct_config.is_file():
        score_paths = [direct_score]
        config_paths = [direct_config]
    if not score_paths or len(score_paths) != len(config_paths):
        raise FileNotFoundError(
            f"Expected matching score/config files or shards under {variable_root}; "
            f"found {len(score_paths)} score files and {len(config_paths)} configs"
        )

    configs = [json.loads(path.read_text()) for path in config_paths]
    reference = configs[0]
    controlled_fields = (
        "experiment",
        "variable",
        "baseline",
        "target_metric",
        "model_key",
        "checkpoint_path",
        "cases_file",
        "sites",
        "regions",
        "leads",
    )
    for path, config in zip(config_paths[1:], configs[1:]):
        for field in controlled_fields:
            if config[field] != reference[field]:
                raise ValueError(
                    f"Inconsistent {field!r} in {path}: "
                    f"{config[field]!r} != {reference[field]!r}"
                )

    if reference["experiment"] != "controlled_all_inputs_activation_patch":
        raise ValueError(f"Unexpected experiment in {config_paths[0]}")
    if reference["variable"] != variable:
        raise ValueError(f"Variable mismatch in {config_paths[0]}")
    if reference["model_key"] != "precip_large_zwd":
        raise ValueError(f"Unexpected model in {config_paths[0]}")
    if reference["target_metric"] != "q850_target_box_mean_g_per_kg":
        raise ValueError(f"Unexpected target metric in {config_paths[0]}")
    if reference["baseline"] != EXPECTED_BASELINES[variable]:
        raise ValueError(f"Unexpected baseline in {config_paths[0]}")

    data = pd.concat(
        [pd.read_csv(path) for path in score_paths],
        ignore_index=True,
    )
    duplicated = data.duplicated(KEY_COLUMNS)
    if duplicated.any():
        duplicate_keys = data.loc[duplicated, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"Duplicate activation-patching rows: {duplicate_keys}")
    return data, reference


def _load_controlled_data(data_root: Path, lead_h: int) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    configs: dict[str, dict] = {}
    for variable in VARIABLES:
        data, config = _load_variable(data_root, variable)
        lead_data = data.loc[data["lead_h"].eq(lead_h)].copy()
        if lead_data.empty:
            raise ValueError(f"No lead-{lead_h} rows for {variable}")
        frames[variable] = lead_data
        configs[variable] = config

    shared_fields = (
        "experiment",
        "target_metric",
        "model_key",
        "checkpoint_path",
        "cases_file",
        "sites",
        "regions",
        "leads",
    )
    reference = configs[VARIABLES[0]]
    for variable in VARIABLES[1:]:
        for field in shared_fields:
            if configs[variable][field] != reference[field]:
                raise ValueError(
                    f"{variable} does not share controlled field {field!r}"
                )

    reference_cases = set(frames[VARIABLES[0]]["case_id"])
    for variable in VARIABLES:
        data = frames[variable]
        if set(data["case_id"]) != reference_cases:
            raise ValueError(f"{variable} uses a different case set")
        expected_rows = len(reference_cases) * len(SITE_ORDER) * len(REGION_ORDER)
        if len(data) != expected_rows:
            raise ValueError(
                f"{variable}: expected {expected_rows} lead-{lead_h} rows, "
                f"found {len(data)}"
            )
        if set(data["patch_site"]) != set(SITE_ORDER):
            raise ValueError(f"{variable} has an unexpected patch-site set")
        if set(data["patch_region"]) != set(REGION_ORDER):
            raise ValueError(f"{variable} has an unexpected patch-region set")
    return frames


def _median_matrix(data: pd.DataFrame) -> np.ndarray:
    return (
        data.pivot_table(
            index="patch_site",
            columns="patch_region",
            values="recovery",
            aggfunc="median",
        )
        .reindex(index=SITE_ORDER, columns=REGION_ORDER)
        .to_numpy()
    )


def _plot(frames: dict[str, pd.DataFrame], lead_h: int, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
        }
    )

    fig = plt.figure(figsize=(16.8, 5.7))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.105,
        right=0.955,
        bottom=0.17,
        top=0.85,
        wspace=0.045,
    )
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(
        fig.add_subplot(grid[0, panel], sharey=axes[0])
        for panel in (1, 2)
    )
    colorbar_axis = fig.add_subplot(grid[0, 3])
    fig.patch.set_facecolor("white")
    cmap = mpl.colormaps["RdBu_r"]
    norm = Normalize(vmin=-1.0, vmax=1.0)

    for panel_index, (ax, variable) in enumerate(zip(axes, VARIABLES)):
        matrix = _median_matrix(frames[variable])
        image = ax.imshow(
            np.clip(matrix, -1.0, 1.0),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_facecolor("white")
        ax.set_title(VARIABLE_TITLES[variable], pad=6)
        ax.set_xticks(np.arange(len(REGION_ORDER)))
        ax.set_xticklabels(
            [REGION_LABELS[region] for region in REGION_ORDER],
            rotation=0,
        )
        ax.set_yticks(np.arange(len(SITE_ORDER)))
        ax.set_yticklabels([SITE_LABELS[site] for site in SITE_ORDER])

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isnan(value):
                    continue
                clipped = float(np.clip(value, -1.0, 1.0))
                text_color = "white" if abs(clipped) >= 0.56 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.4,
                    fontweight="bold",
                )

        ax.axhline(2.5, color="0.25", linestyle="--", linewidth=1.1)
        ax.axvline(4.5, color="0.35", linestyle=":", linewidth=1.1)
        ax.tick_params(axis="x", length=3)
        ax.tick_params(axis="y", labelleft=panel_index == 0)

    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        ticks=np.linspace(-1, 1, 9),
    )
    colorbar.set_label("median recovered fraction (colour clipped to [-1, 1])")

    case_count = frames[VARIABLES[0]]["case_id"].nunique()
    fig.suptitle(
        "Controlled activation patching of the q850 target\n"
        f"same large precipitation+ZWD model and {case_count} cases; "
        f"median recovery at {lead_h} h",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "Observed-field activations are patched into a matched spatial-mean "
        "baseline; the dotted line separates the low-near/remote controls. "
        "Numbers show unclipped medians.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = _parse_args()
    frames = _load_controlled_data(args.data_root, args.lead)
    _plot(frames, args.lead, args.output)


if __name__ == "__main__":
    main()
