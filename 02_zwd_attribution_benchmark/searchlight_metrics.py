"""
Pooled attribution metrics for the searchlight benchmark.

Given
    attr_zwd:       (H, W) full-resolution ZWD attribution map from one method
    masks:          list of MaskSpec
    mask_arrays:    list of (H, W) Gaussian masks aligned with `masks`
    G, S:           ground-truth magnitude / signed arrays, same length

we compute:

    A_mag_r  = weighted_mean(|attr_zwd| inside mask_r, cos-lat weights)
    A_sign_r = weighted_mean( attr_zwd  inside mask_r, cos-lat weights)

and return a dict with rho_mag (Spearman), rho_signed (if signed=True),
NDCG@10, top-10 recall, and the remote-control gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
# Pooling
# ------------------------------------------------------------------
def pool_attribution(
    attr_map: np.ndarray,
    mask_arrays: list[np.ndarray],
    cos_lat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_mag, A_sign) arrays of length n_masks.

    Each pooled score is a cos-lat-weighted mean of attr_map inside the
    Gaussian mask.  A_mag uses |attr_map|, A_sign uses attr_map as-is.
    """
    abs_attr = np.abs(attr_map)
    A_mag = np.empty(len(mask_arrays), dtype=np.float64)
    A_sign = np.empty(len(mask_arrays), dtype=np.float64)
    for i, m in enumerate(mask_arrays):
        w = (m * cos_lat_w).astype(np.float64)
        wsum = w.sum()
        if wsum < 1e-12:
            A_mag[i] = 0.0
            A_sign[i] = 0.0
            continue
        A_mag[i] = float((w * abs_attr).sum() / wsum)
        A_sign[i] = float((w * attr_map).sum() / wsum)
    return A_mag, A_sign


# ------------------------------------------------------------------
# Rank-based metrics
# ------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank implementation; avoids the scipy.stats dependency."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1  # 1-based
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """NDCG@k where `scores` is the model's predicted ranking signal and
    `relevance` is the ground-truth non-negative relevance.
    """
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    relevance = np.clip(relevance, 0.0, None)
    n = len(scores)
    if n == 0:
        return float("nan")
    k = min(k, n)

    order = np.argsort(-scores)[:k]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())

    ideal_order = np.argsort(-relevance)[:k]
    ideal_gains = relevance[ideal_order]
    idcg = float((ideal_gains * discounts).sum())
    if idcg < 1e-12:
        return float("nan")
    return dcg / idcg


def top_k_recall(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """Fraction of the top-k relevant masks recovered by ranking `scores`."""
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("nan")
    k = min(k, n)
    top_pred = set(np.argsort(-scores)[:k].tolist())
    top_true = set(np.argsort(-relevance)[:k].tolist())
    return len(top_pred & top_true) / k


# ------------------------------------------------------------------
# Main metric bundle
# ------------------------------------------------------------------
@dataclass
class MetricBundle:
    method: str
    case_id: str
    scale: str
    rho_mag: float
    rho_signed: float
    ndcg_at_10: float
    top10_recall: float
    remote_gap: float
    n_masks: int
    n_remote: int
    pooled_A_mag: np.ndarray    # (n_masks,)
    pooled_A_sign: np.ndarray   # (n_masks,)


def evaluate(
    method: str,
    case_id: str,
    scale: str,
    attr_map: np.ndarray,
    masks,
    mask_arrays: list[np.ndarray],
    G: np.ndarray,
    S: np.ndarray,
    cos_lat_w: np.ndarray,
    signed: bool = True,
) -> MetricBundle:
    A_mag, A_sign = pool_attribution(attr_map, mask_arrays, cos_lat_w)

    near_idx = np.array([i for i, m in enumerate(masks) if m.role == "near"])
    remote_idx = np.array([i for i, m in enumerate(masks) if m.role == "remote"])

    # Primary effectiveness metrics are computed on the NEAR set only;
    # remote control masks are a separate sanity check.
    G_near = G[near_idx] if near_idx.size else G
    A_mag_near = A_mag[near_idx] if near_idx.size else A_mag
    A_sign_near = A_sign[near_idx] if near_idx.size else A_sign
    S_near = S[near_idx] if near_idx.size else S

    rho_mag = spearman(A_mag_near, G_near)
    rho_signed = spearman(A_sign_near, S_near) if signed else float("nan")
    ndcg = ndcg_at_k(A_mag_near, G_near, k=10)
    recall = top_k_recall(A_mag_near, G_near, k=10)

    if remote_idx.size:
        remote_gap = float(G_near.mean() - G[remote_idx].mean())
    else:
        remote_gap = float("nan")

    return MetricBundle(
        method=method, case_id=case_id, scale=scale,
        rho_mag=rho_mag, rho_signed=rho_signed,
        ndcg_at_10=ndcg, top10_recall=recall,
        remote_gap=remote_gap,
        n_masks=int(G_near.shape[0]),
        n_remote=int(remote_idx.size),
        pooled_A_mag=A_mag,
        pooled_A_sign=A_sign,
    )
