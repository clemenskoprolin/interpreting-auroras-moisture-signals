"""Reusable XAI method implementations for Aurora.

Submodules are imported lazily so that using a gradient method does not require
optional ViT-CX dependencies such as scikit-learn and scikit-image.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "saliency": (".saliency", "saliency"),
    "smoothgrad": (".smoothgrad", "smoothgrad"),
    "integrated_gradients": (".ig", "integrated_gradients"),
    "extract_feature_map": (".vit_cx", "extract_feature_map"),
    "cluster_features": (".vit_cx", "cluster_features"),
    "score_clusters": (".vit_cx", "score_clusters"),
    "aggregate_and_upsample": (".vit_cx", "aggregate_and_upsample"),
    "generate_rise_masks": (".rise", "generate_rise_masks"),
    "accumulate_rise": (".rise", "accumulate_rise"),
    "normalize_rise": (".rise", "normalize_rise"),
    "accumulate_rise_with_stats": (".rise", "accumulate_rise_with_stats"),
    "normalize_rise_covariance": (".rise", "normalize_rise_covariance"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
