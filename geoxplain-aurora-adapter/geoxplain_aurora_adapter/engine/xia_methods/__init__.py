"""
xia_methods — General XAI method implementations for Aurora.

Each method takes a model, a batch_fn, and a target_fn as its core
interface, making them reusable across scripts (original_copy, SCLIB,
custom analyses) without duplicating Aurora-specific logic.

Methods:
    saliency  — vanilla gradient |dy/dx|
    ig        — Integrated Gradients (x - x_bl) * mean_grad
"""

from .saliency import saliency
from .smoothgrad import smoothgrad
from .ig import integrated_gradients
from .vit_cx import (
    extract_feature_map,
    cluster_features,
    score_clusters,
    aggregate_and_upsample,
)
from .rise import (
    generate_rise_masks,
    accumulate_rise,
    normalize_rise,
    accumulate_rise_with_stats,
    normalize_rise_covariance,
)

__all__ = [
    "saliency",
    "smoothgrad",
    "integrated_gradients",
    "extract_feature_map",
    "cluster_features",
    "score_clusters",
    "aggregate_and_upsample",
    "generate_rise_masks",
    "accumulate_rise",
    "normalize_rise",
    "accumulate_rise_with_stats",
    "normalize_rise_covariance",
]
