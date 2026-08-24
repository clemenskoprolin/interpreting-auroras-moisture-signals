"""
comparison_config.py — dataclasses and default configurations for the
07_zwd_precipitation_model_comparison package.

The precipitation variable name used inside Aurora surf_vars can be adjusted
by changing PRECIP_VAR once the checkpoint audit confirms the exact key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Variable name constants (update after checkpoint audit if needed)
# ---------------------------------------------------------------------------
PRECIP_VAR = "tp_mswep"

# Checkpoint paths — large ESFM models (audit job 2026-05-29, embed_dim=512)
# precip_large: surf_heads keys include zwd → has_zwd=True
# precip_large_without_zwd: no zwd surf_head → has_zwd=False
PRECIP_ZWD_CHECKPOINT = os.environ.get("AURORA_PRECIP_ZWD_CHECKPOINT", (
    "/capstor/store/cscs/swissai/a122/ltrentini/checkpoints_xAI/"
    "precip_large/model_ckpt-step=7400-loss_train=0.07.ckpt"
))
PRECIP_ONLY_CHECKPOINT = os.environ.get("AURORA_PRECIP_ONLY_CHECKPOINT", (
    "/capstor/store/cscs/swissai/a122/ltrentini/checkpoints_xAI/"
    "precip_large_without_zwd/model_ckpt-step=7400-loss_train=0.07.ckpt"
))
# Aliases used by Plan1 (07_zwd_precipitation_model_comparison large-model run)
PRECIP_LARGE_ZWD_CHECKPOINT = PRECIP_ZWD_CHECKPOINT
PRECIP_LARGE_ONLY_CHECKPOINT = PRECIP_ONLY_CHECKPOINT

# MSWEP store path (from 06_precipitation_moisture_relationships/common.py)
MSWEP_STORE_PATH = os.environ.get("AURORA_MSWEP_DATA", (
    "/capstor/store/cscs/swissai/a122/hydrological_data/"
    "MSWEP-v280-720x1440-6h_acc_3h_sampling.zarr"
))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Full specification for a single Aurora checkpoint."""
    name: str
    checkpoint_path: str
    surf_vars: Tuple[str, ...]
    atmos_vars: Tuple[str, ...] = ("z", "u", "v", "t", "q")
    static_vars: Tuple[str, ...] = ("lsm", "z", "slt")
    embed_dim: int = 256
    encoder_depths: Tuple[int, ...] = (2, 6, 2)
    encoder_num_heads: Tuple[int, ...] = (4, 8, 16)
    decoder_depths: Tuple[int, ...] = (2, 6, 2)
    decoder_num_heads: Tuple[int, ...] = (16, 8, 4)
    num_heads: int = 8
    has_zwd: bool = False
    has_precip: bool = True

    @property
    def dataset_name(self) -> str:
        """Dataset name string used in Aurora Metadata."""
        parts = ["era5"]
        if self.has_precip:
            parts.append("precip")
        if self.has_zwd:
            parts.append("zwd")
        return "_".join(parts)


@dataclass
class InputVariableSpec:
    """Specification of a single input variable for attribution analysis."""
    name: str
    kind: str         # "surf" or "atmos"
    level_hpa: Optional[int] = None   # only for atmos


@dataclass
class TargetSpec:
    """Output variable specification for trajectory recording and metrics."""
    name: str
    output_var: str           # key in Aurora pred.surf_vars or pred.atmos_vars
    level_hpa: Optional[int] = None   # None for surface vars


@dataclass
class InterventionSpec:
    """A single precipitation intervention: name and dose in mm."""
    name: str
    dose_mm: float   # 0 = removal; positive/negative for dose interventions


@dataclass
class AttributionChannelSpec:
    """One attribution channel: which variable, timestep, and applicable models."""
    var: str
    kind: str                  # "surf" or "atmos"
    level_hpa: Optional[int]
    timestep: int              # 0 = t0, 1 = t1
    applicable_models: Tuple[str, ...] = ("precip_zwd", "precip_only")


@dataclass
class ContrastSpec:
    """Specification for a contrastive pair of model runs."""
    model_a: str       # name of first model (e.g. "precip_zwd")
    model_b: str       # name of second model (e.g. "precip_only")
    intervention: str  # intervention key (e.g. "actual", "remove_t1")
    output: str        # target name (from TargetSpec.name)


@dataclass
class RolloutSpec:
    """Autoregressive rollout configuration."""
    steps_hours: Tuple[int, ...] = (6, 12, 18, 24)
    mode: str = "independent"   # "independent" or "chained"
    record_each_step: bool = True

    @property
    def max_steps(self) -> int:
        return max(self.steps_hours) // 6

    def steps_for(self, lead_h: int) -> int:
        if lead_h % 6 != 0 or lead_h <= 0:
            raise ValueError(f"lead_h must be a positive multiple of 6, got {lead_h}")
        return lead_h // 6


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

# Architecture confirmed by checkpoint audit (2026-05-29, large ESFM models):
#   embed_dim=512, encoder/decoder depths=(6,10,8)/(8,10,6), num_heads=16.
#   precip_large surf_heads: 2t, 10u, 10v, msl, tp_mswep, zwd → has_zwd=True
#   precip_large_without_zwd surf_heads: 2t, 10u, 10v, msl, tp_mswep → has_zwd=False
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "precip_zwd": ModelSpec(
        name="precip_zwd",
        checkpoint_path=PRECIP_ZWD_CHECKPOINT,
        surf_vars=("2t", "10u", "10v", "msl", "tp_mswep", "zwd"),
        atmos_vars=("z", "u", "v", "t", "q"),
        static_vars=("lsm", "z", "slt"),
        embed_dim=512,
        encoder_depths=(6, 10, 8),
        encoder_num_heads=(8, 16, 32),
        decoder_depths=(8, 10, 6),
        decoder_num_heads=(32, 16, 8),
        num_heads=16,
        has_zwd=True,
        has_precip=True,
    ),
    "precip_only": ModelSpec(
        name="precip_only",
        checkpoint_path=PRECIP_ONLY_CHECKPOINT,
        surf_vars=("2t", "10u", "10v", "msl", "tp_mswep"),
        atmos_vars=("z", "u", "v", "t", "q"),
        static_vars=("lsm", "z", "slt"),
        embed_dim=512,
        encoder_depths=(6, 10, 8),
        encoder_num_heads=(8, 16, 32),
        decoder_depths=(8, 10, 6),
        decoder_num_heads=(32, 16, 8),
        num_heads=16,
        has_zwd=False,
        has_precip=True,
    ),
    # Named large-model entries for explicit --models precip_large_zwd precip_large_only
    "precip_large_zwd": ModelSpec(
        name="precip_large_zwd",
        checkpoint_path=PRECIP_LARGE_ZWD_CHECKPOINT,
        surf_vars=("2t", "10u", "10v", "msl", "tp_mswep", "zwd"),
        atmos_vars=("z", "u", "v", "t", "q"),
        static_vars=("lsm", "z", "slt"),
        embed_dim=512,
        encoder_depths=(6, 10, 8),
        encoder_num_heads=(8, 16, 32),
        decoder_depths=(8, 10, 6),
        decoder_num_heads=(32, 16, 8),
        num_heads=16,
        has_zwd=True,
        has_precip=True,
    ),
    "precip_large_only": ModelSpec(
        name="precip_large_only",
        checkpoint_path=PRECIP_LARGE_ONLY_CHECKPOINT,
        surf_vars=("2t", "10u", "10v", "msl", "tp_mswep"),
        atmos_vars=("z", "u", "v", "t", "q"),
        static_vars=("lsm", "z", "slt"),
        embed_dim=512,
        encoder_depths=(6, 10, 8),
        encoder_num_heads=(8, 16, 32),
        decoder_depths=(8, 10, 6),
        decoder_num_heads=(32, 16, 8),
        num_heads=16,
        has_zwd=False,
        has_precip=True,
    ),
}

DEFAULT_ROLLOUT = RolloutSpec()

DEFAULT_TARGETS: list[TargetSpec] = [
    TargetSpec("precip", PRECIP_VAR),
    TargetSpec("q850", "q", level_hpa=850),
    TargetSpec("zwd", "zwd"),
]

DEFAULT_DOSES_MM: Tuple[float, ...] = (1.0, 5.0, 10.0)

# Spatial parameters for precipitation disk interventions
PRECIP_DISK_KM: float = 1000.0
PRECIP_TAPER_KM: float = 2500.0
