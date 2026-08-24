"""
ViT-CX — general implementation.

Causal cluster attribution by latent-feature similarity (Xie et al., 2022).

Core functions (GPU-free after the initial forward pass):
    extract_feature_map    — reshape hook output to (feat_H, feat_W, D)
    cluster_features       — agglomerative clustering via cosine distance
    score_clusters         — call scorer_fn per cluster, accumulate
    aggregate_and_upsample — normalize + upsample to full input resolution

Designed to be called from any script that can supply:
  - encoder features (from a forward-hook on a Swin3D encoder block)
  - scorer_fn(mask_hw: np.ndarray) -> float
      mask (H, W) float32 with 1.0 = occlude (replace with baseline),
      0.0 = keep original.
      Should return: orig_val - model_pred  (higher = more important)
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.transform import resize as sk_resize
from scipy.ndimage import gaussian_filter
import torch


def extract_feature_map(feat_tensor, num_levels, feat_H, feat_W):
    """Reshape hook output (1, L, D) → (feat_H, feat_W, D) by averaging over levels.

    Args:
        feat_tensor: torch.Tensor or np.ndarray of shape (1, L, D)
        num_levels:  number of vertical/channel levels
        feat_H, feat_W: spatial grid size at this encoder stage

    Returns:
        np.ndarray of shape (feat_H, feat_W, D)
    """
    if isinstance(feat_tensor, torch.Tensor):
        feat_tensor = feat_tensor.detach().cpu().float().numpy()
    _, L, D = feat_tensor.shape
    expected = num_levels * feat_H * feat_W
    if L != expected:
        raise ValueError(
            f"Token count mismatch: got L={L}, expected "
            f"{num_levels}×{feat_H}×{feat_W}={expected}."
        )
    spatial = feat_tensor[0].reshape(num_levels, feat_H, feat_W, D)
    return spatial.mean(axis=0)  # (feat_H, feat_W, D)


def cluster_features(feature_map, n_clusters=256, distance_threshold=None):
    """Cluster spatial tokens by cosine similarity.

    Provide exactly one of ``n_clusters`` or ``distance_threshold``:

    - ``n_clusters`` (fixed budget): merge tokens into exactly this many clusters.
      The cluster count *is* the number of occlusion forward passes, so a fixed
      budget bounds the run cost regardless of variable or case.  This is the
      shipped default.
    - ``distance_threshold`` (cosine-distance cutoff on ``1 - cosine_sim``):
      merge until no within-cluster distance exceeds the cutoff.  The resulting
      cluster count is data-dependent (and can be the full token count if no
      tokens are similar enough to merge), so cost is unbounded.

    Args:
        feature_map:        np.ndarray (feat_H, feat_W, D)
        n_clusters:         fixed cluster budget, or None to use distance_threshold.
                            Clamped to [1, feat_H*feat_W].
        distance_threshold: AgglomerativeClustering threshold on [1 - cosine_sim],
                            or None to use n_clusters.

    Returns:
        labels:      np.ndarray (feat_H * feat_W,)  cluster label per token
        patch_masks: np.ndarray (n_clusters, feat_H, feat_W)  binary per-cluster masks
        n_clusters:  int
    """
    if (n_clusters is None) == (distance_threshold is None):
        raise ValueError(
            "cluster_features: provide exactly one of n_clusters or "
            f"distance_threshold (got n_clusters={n_clusters!r}, "
            f"distance_threshold={distance_threshold!r})."
        )

    feat_H, feat_W, D = feature_map.shape
    feat_flat = feature_map.reshape(feat_H * feat_W, D)

    if n_clusters is not None:
        n_clusters = max(1, min(int(n_clusters), feat_H * feat_W))

    norms  = np.linalg.norm(feat_flat, axis=1, keepdims=True)
    norms  = np.maximum(norms, 1e-8)
    feat_n = feat_flat / norms
    sim    = feat_n @ feat_n.T
    dist   = np.clip(1.0 - sim, 0.0, None).astype(np.float64)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="complete",
    )
    clustering.fit(dist)
    labels     = clustering.labels_
    n_clusters = int(labels.max()) + 1

    patch_masks = np.zeros((n_clusters, feat_H, feat_W), dtype=np.float32)
    for idx, lbl in enumerate(labels):
        r, c = divmod(int(idx), feat_W)
        patch_masks[lbl, r, c] = 1.0

    return labels, patch_masks, n_clusters


def score_clusters(
    scorer_fn,
    patch_masks,
    feat_H, feat_W,
    H, W,
    cluster_indices=None,
    verbose=True,
    rank=0,
    progress_callback=None,
):
    """Score each cluster by calling scorer_fn(mask_hw: np.ndarray) -> float.

    scorer_fn receives a (H, W) float32 mask (1.0 = occlude, 0.0 = keep) and
    should return orig_val - masked_pred (positive = warming influence).

    Args:
        scorer_fn:       callable(mask_hw: np.ndarray (H,W) float32) -> float
        patch_masks:     np.ndarray (n_clusters, feat_H, feat_W)
        feat_H, feat_W:  spatial dimensions at encoder stage
        H, W:            full input resolution
        cluster_indices: list of cluster indices to process (None = all)
        verbose:         print progress
        rank:            rank id for progress messages

    Returns:
        partial_saliency: np.ndarray (feat_H, feat_W) float64
        partial_weight:   np.ndarray (feat_H, feat_W) float64
    """
    if cluster_indices is None:
        cluster_indices = list(range(patch_masks.shape[0]))

    partial_saliency = np.zeros((feat_H, feat_W), dtype=np.float64)
    partial_weight   = np.zeros((feat_H, feat_W), dtype=np.float64)
    total = len(cluster_indices)

    for count, i in enumerate(cluster_indices):
        mask_full = sk_resize(
            patch_masks[i], (H, W), order=0,
            preserve_range=True, anti_aliasing=False,
        )
        score = scorer_fn(mask_full.astype(np.float32))

        partial_saliency += score * patch_masks[i]
        partial_weight   += patch_masks[i]

        if verbose and ((count + 1) % 50 == 0 or count == total - 1):
            print(f"[Rank {rank}] Processed {count + 1}/{total} clusters ...")
        if progress_callback is not None:
            progress_callback(count + 1, total)

    return partial_saliency, partial_weight


def aggregate_and_upsample(
    partial_saliency,
    partial_weight,
    H, W,
    smooth_sigma=0,
):
    """Normalize partial results and upsample to full input resolution.

    Args:
        partial_saliency: np.ndarray (feat_H, feat_W) — sum of score * mask
        partial_weight:   np.ndarray (feat_H, feat_W) — sum of mask
        H, W:             target output resolution
        smooth_sigma:     Gaussian smoothing sigma (tuple or scalar); 0/None = no smoothing

    Returns:
        np.ndarray (H, W) float32
    """
    patch_saliency = np.where(
        partial_weight > 1e-8,
        partial_saliency / partial_weight,
        0.0,
    ).astype(np.float32)

    saliency = sk_resize(
        patch_saliency, (H, W), order=3,
        preserve_range=True, anti_aliasing=False,
    )

    if smooth_sigma:
        sigma = smooth_sigma
        mode = ["reflect", "wrap"] if isinstance(sigma, (tuple, list)) else "reflect"
        saliency = gaussian_filter(saliency, sigma=sigma, mode=mode)

    return saliency.astype(np.float32)
