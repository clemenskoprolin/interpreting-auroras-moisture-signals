"""Per-method XIA runners — the Aurora-specific glue around each algorithm.

Each module here adapts one explanation method to Aurora: it builds the input
batch, wires the target, fans perturbation passes across GPUs, drives progress,
and packs the result into per-variable / per-level attribution maps.  The pure,
model-agnostic algorithms live one level up in ``xia_methods``.

``compute._run_local`` dispatches to these by method id.
"""

from .ig import _run_ig
from .rise import _run_rise
from .saliency import _run_saliency
from .vit_cx import _run_vit_cx

__all__ = ["_run_saliency", "_run_ig", "_run_rise", "_run_vit_cx"]
