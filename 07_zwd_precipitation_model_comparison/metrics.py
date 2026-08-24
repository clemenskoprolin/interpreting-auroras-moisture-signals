"""
metrics.py — Evaluation metrics for 07_zwd_precipitation_model_comparison.

All functions are pure-numpy, no Aurora imports — safe to run locally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Spatial box mean (cosine-lat weighted)
# ---------------------------------------------------------------------------

def box_mean_metric(
    tensor: np.ndarray,
    lat_imin: int,
    lat_imax: int,
    lon_imin: int,
    lon_imax: int,
    lat_vals: np.ndarray,
) -> float:
    """Cosine-latitude-weighted box mean.

    Args:
        tensor: (..., H, W) array.
        lat_imin, lat_imax, lon_imin, lon_imax: inclusive box indices.
        lat_vals: (H,) latitude array in degrees.

    Returns:
        Scalar weighted mean over the box.
    """
    sub = tensor[..., lat_imin:lat_imax + 1, lon_imin:lon_imax + 1]
    sub_lat = lat_vals[lat_imin:lat_imax + 1]
    cos_w = np.cos(np.radians(sub_lat)).astype(np.float64)
    cos_w = np.clip(cos_w, 0.0, None)
    # Collapse any leading batch dims, then compute cos-lat weighted mean over (lat, lon)
    sub_2d = sub.reshape(-1, sub.shape[-2], sub.shape[-1]).mean(0)  # (lat_box, lon_box)
    n_lon = sub_2d.shape[-1]
    # cos_w[:, None] broadcast to (lat_box, lon_box); total weight = sum over lat * n_lon
    total_w = cos_w.sum() * n_lon
    if total_w == 0.0:
        return float("nan")
    return float((sub_2d * cos_w[:, None]).sum() / total_w)


# ---------------------------------------------------------------------------
# Spatial RMS difference
# ---------------------------------------------------------------------------

def spatial_rms_diff(
    a: np.ndarray,
    b: np.ndarray,
    lat_vals: np.ndarray,
) -> float:
    """Cosine-latitude-weighted RMS difference between two (H, W) arrays.

    Args:
        a, b: Arrays of the same shape (..., H, W).
        lat_vals: (H,) latitude array in degrees.

    Returns:
        Scalar weighted RMS.
    """
    diff = (a - b).astype(np.float64)
    H = lat_vals.shape[0]
    cos_w = np.cos(np.radians(lat_vals)).astype(np.float64)
    cos_w = np.clip(cos_w, 0.0, None)[:, None]  # (H, 1)
    # Handle batch dims: flatten everything except H, W
    diff_2d = diff.reshape(-1, diff.shape[-2], diff.shape[-1]).mean(0)  # (H, W)
    total_w = cos_w.sum() * diff_2d.shape[-1]
    if total_w == 0.0:
        return float("nan")
    return float(np.sqrt((diff_2d ** 2 * cos_w).sum() / total_w))


# ---------------------------------------------------------------------------
# Trajectory difference metrics
# ---------------------------------------------------------------------------

def trajectory_diff_metrics(
    baseline_w: list[float],
    baseline_wo: list[float],
    lat_vals: Optional[np.ndarray] = None,
) -> dict:
    """Compute trajectory-level difference metrics.

    Args:
        baseline_w: List of scalars from model_with_zwd at each lead time.
        baseline_wo: List of scalars from model_without_zwd at each lead time.
        lat_vals: Not used for scalars, kept for API consistency.

    Returns:
        Dict with keys: M (list), abs_M (list), rel_M (list), rms_diff, box_diff.
    """
    if len(baseline_w) != len(baseline_wo):
        raise ValueError(
            f"baseline_w (len={len(baseline_w)}) and baseline_wo (len={len(baseline_wo)}) "
            "must have the same length"
        )
    w = np.asarray(baseline_w, dtype=np.float64)
    wo = np.asarray(baseline_wo, dtype=np.float64)
    M = w - wo
    abs_M = np.abs(M)
    denom = np.abs(wo)
    rel_M = np.where(denom > 1e-10, M / denom, np.nan)
    rms_diff = float(np.sqrt(np.mean(M ** 2)))
    box_diff = float(np.mean(M))

    return {
        "M": M.tolist(),
        "abs_M": abs_M.tolist(),
        "rel_M": rel_M.tolist(),
        "rms_diff": rms_diff,
        "box_diff": box_diff,
    }


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation between x and y.

    Returns:
        (rho, p_value) using scipy if available, else (rho, nan).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except ImportError:
        # Manual rank correlation
        def _rank(arr):
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(arr), dtype=np.float64)
            return ranks
        rx, ry = _rank(x), _rank(y)
        mx, my = rx.mean(), ry.mean()
        num = ((rx - mx) * (ry - my)).sum()
        den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
        r = num / den if den > 1e-12 else 0.0
        return float(r), float("nan")


# ---------------------------------------------------------------------------
# Top-k overlap
# ---------------------------------------------------------------------------

def top_k_overlap(
    x: np.ndarray,
    y: np.ndarray,
    k_frac: float = 0.01,
) -> float:
    """Fraction of top-k elements in x that are also in top-k of y.

    Args:
        x, y: Arrays of the same shape.
        k_frac: Fraction of elements to consider (default 0.01 = 1%).

    Returns:
        Overlap fraction in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    k = max(1, int(k_frac * x.size))
    top_x = set(np.argpartition(x, -k)[-k:])
    top_y = set(np.argpartition(y, -k)[-k:])
    return float(len(top_x & top_y) / k)


# ---------------------------------------------------------------------------
# Center of mass displacement
# ---------------------------------------------------------------------------

def center_of_mass_displacement(
    x: np.ndarray,
    y: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> float:
    """Great-circle distance between centers of mass of two (H, W) maps (km).

    The center of mass is defined as the weighted average lat/lon where weights
    are the absolute values of the map (so signed maps are supported).
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    y = np.abs(np.asarray(y, dtype=np.float64))
    H, W = lat_vals.shape[0], lon_vals.shape[0]
    x = x.reshape(H, W)
    y = y.reshape(H, W)

    def _com(arr):
        s = arr.sum()
        if s < 1e-30:
            return float("nan"), float("nan")
        w = arr / s
        lat_c = float((w * lat_vals[:, None]).sum())
        lon_c = float((w * lon_vals[None, :]).sum())
        return lat_c, lon_c

    lat_x, lon_x = _com(x)
    lat_y, lon_y = _com(y)
    if np.isnan(lat_x) or np.isnan(lat_y):
        return float("nan")

    # Great-circle distance
    import math
    lat1, lon1 = math.radians(lat_x), math.radians(lon_x)
    lat2, lon2 = math.radians(lat_y), math.radians(lon_y)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * 6371.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ---------------------------------------------------------------------------
# NDCG and top-k recall (for ranked attribution evaluation)
# ---------------------------------------------------------------------------

def ndcg_at_k(
    scores: np.ndarray,
    relevance: np.ndarray,
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        scores: Predicted ranking scores (higher = more relevant).
        relevance: Ground-truth relevance values (non-negative).
        k: Number of top items to consider.

    Returns:
        NDCG@k in [0, 1], or nan if ideal DCG is 0.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    relevance = np.asarray(relevance, dtype=np.float64).ravel()
    k = min(k, len(scores))

    order = np.argsort(-scores)[:k]
    dcg = float(np.sum(relevance[order] / np.log2(np.arange(2, k + 2))))

    ideal_order = np.argsort(-relevance)[:k]
    idcg = float(np.sum(relevance[ideal_order] / np.log2(np.arange(2, k + 2))))

    if idcg < 1e-12:
        return float("nan")
    return float(dcg / idcg)


def top_k_recall(
    scores: np.ndarray,
    relevance: np.ndarray,
    k: int,
) -> float:
    """Top-k recall: fraction of top-k-relevant items that appear in top-k predicted.

    Args:
        scores: Predicted ranking scores.
        relevance: Ground-truth relevance (used to define "top-k relevant").
        k: Number of top items to consider in both predicted and relevant sets.

    Returns:
        Recall in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    relevance = np.asarray(relevance, dtype=np.float64).ravel()
    k = min(k, len(scores))

    pred_top = set(np.argpartition(-scores, k - 1)[:k])
    rel_top = set(np.argpartition(-relevance, k - 1)[:k])

    if not rel_top:
        return float("nan")
    return float(len(pred_top & rel_top) / len(rel_top))
