"""
RISE — general implementation.

Randomized Input Sampling for Explanation (Petsiuk et al., 2018).

Mask semantics: mask=1 means KEEP original, mask=0 means replace with baseline.
Canonical estimator: S(x) = sum(f(x ⊙ M_i) * M_i) / sum(M_i)

For signed or regression-valued targets, a large target intercept can dominate
the canonical estimator.  Use accumulate_rise_with_stats +
normalize_rise_covariance when the desired quantity is the conditional effect
of keeping a pixel, not the raw conditional target value.

Core functions:
    generate_rise_masks — yield upsampled binary masks from a deterministic RNG
    accumulate_rise     — run scorer_fn per mask and accumulate partial sums
    normalize_rise      — apply canonical normalization to partial sums
    accumulate_rise_with_stats / normalize_rise_covariance
                          — centered estimator for regression targets

Multi-GPU: split N masks across ranks by passing (n=my_n, start_idx=my_start).
Each rank calls an accumulate_* function independently; rank 0 sums the partial
results and calls the matching normalize_* function once on the aggregate.
"""

import numpy as np
import torch
import torch.nn.functional as F


def generate_rise_masks(n, cells_h, cells_w, H, W, p=0.5, seed=42, start_idx=0):
    """Yield n RISE masks starting from start_idx in the RNG sequence.

    Each mask is bilinear-upsampled from (cells_h, cells_w) to (H, W) with
    3× longitude tiling to avoid boundary seam artifacts.

    Args:
        n:          number of masks to generate
        cells_h:    low-res grid rows
        cells_w:    low-res grid cols
        H, W:       output mask resolution
        p:          keep probability per low-res cell (canonical RISE: 0.5)
        seed:       base random seed
        start_idx:  number of masks to skip (for multi-GPU splitting)

    Yields:
        mask_np: np.ndarray (H, W) float32 with values in [0, 1]
    """
    cell_h = int(np.ceil(H / cells_h))
    cell_w = int(np.ceil(W / cells_w))
    up_h   = (cells_h + 1) * cell_h
    up_w   = cells_w * cell_w   # = W when W % cells_w == 0

    rng = np.random.RandomState(seed)
    # Advance RNG to the starting position
    for _ in range(start_idx):
        rng.rand(1, 1, cells_h, cells_w)
        rng.randint(0, max(cell_h, 1))
        rng.randint(0, W)

    for _ in range(n):
        grid    = (rng.rand(1, 1, cells_h, cells_w) < p).astype(np.float32)
        grid_t  = torch.from_numpy(grid)
        grid_t3 = grid_t.repeat(1, 1, 1, 3)  # 3× longitude for seam-free upsample
        up      = F.interpolate(grid_t3, size=(up_h, 3 * up_w), mode="bilinear", align_corners=False)
        up_np   = up[0, 0].numpy()
        sh = rng.randint(0, max(cell_h, 1))
        sw = rng.randint(0, W)
        center = up_np[sh:sh + H, up_w:2 * up_w]
        idx    = np.arange(sw, sw + W) % up_w
        yield center[:, idx]


def accumulate_rise(
    scorer_fn,
    n,
    cells_h, cells_w,
    H, W,
    p=0.5,
    seed=42,
    start_idx=0,
    verbose=True,
    rank=0,
):
    """Generate RISE masks, score each one, and accumulate partial sums.

    scorer_fn receives a (H, W) float32 mask (1=keep original, 0=replace with
    baseline) and returns the raw model prediction at the target location.

    Args:
        scorer_fn:  callable(mask_hw: np.ndarray (H,W) float32) -> float
        n:          number of masks to process
        cells_h/w:  low-res grid dimensions
        H, W:       full input resolution
        p:          keep probability
        seed:       base random seed
        start_idx:  number of masks to skip from start
        verbose:    print progress
        rank:       rank id for progress messages

    Returns:
        saliency_accum: np.ndarray (H, W) float64  sum(f(x⊙M) * M)
        mask_sum:       np.ndarray (H, W) float64  sum(M)
    """
    saliency_accum = np.zeros((H, W), dtype=np.float64)
    mask_sum       = np.zeros((H, W), dtype=np.float64)

    for i, mask_np in enumerate(generate_rise_masks(
        n=n, cells_h=cells_h, cells_w=cells_w, H=H, W=W,
        p=p, seed=seed, start_idx=start_idx,
    )):
        pred_val = scorer_fn(mask_np)
        saliency_accum += pred_val * mask_np
        mask_sum       += mask_np

        if verbose and ((i + 1) % 10 == 0 or i == n - 1):
            print(f"[Rank {rank}] Processed {i + 1}/{n} masks ...")

    return saliency_accum, mask_sum


def normalize_rise(saliency_accum, mask_sum):
    """Canonical RISE normalization: S(x) = sum(f*M) / sum(M).

    Args:
        saliency_accum: np.ndarray (H, W) float64
        mask_sum:       np.ndarray (H, W) float64

    Returns:
        raw: np.ndarray (H, W) float32
    """
    raw = np.where(mask_sum > 1e-8, saliency_accum / mask_sum, 0.0)
    return raw.astype(np.float32)


def accumulate_rise_with_stats(
    scorer_fn,
    n,
    cells_h, cells_w,
    H, W,
    p=0.5,
    seed=42,
    start_idx=0,
    verbose=True,
    rank=0,
    progress_callback=None,
):
    """Like accumulate_rise, but keep enough moments for a centered estimator.

    The extra statistics let callers compute Cov(f(M), M_i) / Var(M_i), which
    removes any constant score offset and is therefore much better behaved for
    regression targets than the canonical positive-class RISE heatmap.

    Returns:
        saliency_accum: np.ndarray (H, W) float64  sum(f(M) * M)
        mask_sum:       np.ndarray (H, W) float64  sum(M)
        mask_sq_sum:    np.ndarray (H, W) float64  sum(M^2)
        score_sum:      float                       sum(f(M))
        n_seen:         int                         number of masks processed
    """
    saliency_accum = np.zeros((H, W), dtype=np.float64)
    mask_sum = np.zeros((H, W), dtype=np.float64)
    mask_sq_sum = np.zeros((H, W), dtype=np.float64)
    score_sum = 0.0

    for i, mask_np in enumerate(generate_rise_masks(
        n=n, cells_h=cells_h, cells_w=cells_w, H=H, W=W,
        p=p, seed=seed, start_idx=start_idx,
    )):
        pred_val = float(scorer_fn(mask_np))
        saliency_accum += pred_val * mask_np
        mask_sum += mask_np
        mask_sq_sum += mask_np * mask_np
        score_sum += pred_val

        if verbose and ((i + 1) % 10 == 0 or i == n - 1):
            print(f"[Rank {rank}] Processed {i + 1}/{n} masks ...")
        if progress_callback is not None:
            progress_callback(i + 1, n)

    return saliency_accum, mask_sum, mask_sq_sum, score_sum, int(n)


def normalize_rise_covariance(
    saliency_accum,
    mask_sum,
    mask_sq_sum,
    score_sum,
    n_seen,
    eps=1e-8,
):
    """Centered RISE normalization for regression targets.

    Computes a per-pixel randomized-mask regression coefficient:

        Cov(f(M), M_i) / Var(M_i)

    For binary masks this is equivalent to E[f | M_i=1] - E[f | M_i=0].
    The implementation also works for bilinearly upsampled RISE masks whose
    values are continuous in [0, 1].
    """
    if n_seen <= 0:
        return np.zeros_like(saliency_accum, dtype=np.float32)

    n = float(n_seen)
    score_mean = float(score_sum) / n
    mask_mean = mask_sum / n
    mask_var = mask_sq_sum / n - mask_mean * mask_mean
    cov = saliency_accum / n - score_mean * mask_mean
    raw = np.where(mask_var > eps, cov / mask_var, 0.0)
    return raw.astype(np.float32)
