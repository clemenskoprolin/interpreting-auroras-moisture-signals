"""ViT-CX runner — cluster-based causal attribution for Aurora.

A forward hook captures encoder features; tokens are clustered, each cluster is
occluded with the baseline and scored, and the per-cluster scores are
aggregated and upsampled to the full grid.  ``smooth_sigma`` controls the
Gaussian post-smoothing of that upsampled map (set it to ``0``/``None`` for the
raw, unsmoothed result).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from .._common import (
    COLLAPSED_LEVEL_KEY,
    IG_BASELINE_SIGMA_DEG,
    INPUT_H,
    INPUT_W,
    STAGE_GRID,
    VIT_CX_N_CLUSTERS,
    VIT_CX_STAGE_DEFAULT,
    _gpu_sync_and_gc,
    _make_masked_batch,
    _smooth_tensor,
)
from ..data import CaseData, make_batch
from ..progress import ProgressReporter

# Default Gaussian post-smoothing for the upsampled ViT-CX map.  A falsy value
# disables post-smoothing and returns the raw cluster map.
VIT_CX_SMOOTH_SIGMA_DEFAULT = 0

# A sigma the caller can pass as a scalar, a per-axis (lat, lon) pair, or a
# falsy value (0 / 0.0 / None) to disable post-smoothing entirely.
SmoothSigma = Union[float, tuple, list, None]


def _run_vit_cx_one_var(
    case: CaseData,
    target_fn,
    var_name: str,
    var_type: str,
    baseline_cpu: torch.Tensor,
    model,
    device: str,
    hook_stage: int = VIT_CX_STAGE_DEFAULT,
    n_clusters: int = VIT_CX_N_CLUSTERS,
    smooth_sigma: SmoothSigma = VIT_CX_SMOOTH_SIGMA_DEFAULT,
    progress_reporter: Optional[ProgressReporter] = None,
) -> np.ndarray:
    from ..xia_methods.vit_cx import (
        extract_feature_map, cluster_features, score_clusters, aggregate_and_upsample,
    )
    from ..model import forward as _forward
    from ..parallel import attribution_devices, get_replicas, parallel_map_scores
    from skimage.transform import resize as sk_resize

    feat_store: dict = {}

    def _feat_hook(module, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        feat_store["feat"] = t.detach().cpu().float()

    handle = model.backbone.encoder_layers[hook_stage].blocks[-1].register_forward_hook(
        _feat_hook
    )

    if progress_reporter is not None:
        progress_reporter.set_phase("vit_cx", f"{var_name} feature pass")
    batch_orig = make_batch(case, device)
    with torch.no_grad():
        pred_orig = _forward(model, batch_orig)
        orig_val = float(target_fn(pred_orig).item())
    handle.remove()
    del batch_orig, pred_orig
    _gpu_sync_and_gc()

    num_levels, feat_H, feat_W = STAGE_GRID[hook_stage]
    feat_tensor = feat_store["feat"]
    _, L, _ = feat_tensor.shape
    assert L == num_levels * feat_H * feat_W, (
        f"Stage {hook_stage} token count mismatch: L={L}, "
        f"expected {num_levels}×{feat_H}×{feat_W}"
    )
    if progress_reporter is not None:
        progress_reporter.set_phase("vit_cx", f"{var_name} clustering")
    feature_map = extract_feature_map(feat_tensor, num_levels, feat_H, feat_W)
    labels, patch_masks, n_clusters = cluster_features(
        feature_map, n_clusters=n_clusters
    )
    if progress_reporter is not None:
        progress_reporter.add_total_units(n_clusters, force=True)

    def _score(model_d, device_d, occlusion_mask_hw: np.ndarray) -> float:
        keep_mask = 1.0 - occlusion_mask_hw
        batch = _make_masked_batch(case, device_d, keep_mask, var_name, baseline_cpu, var_type)
        with torch.no_grad():
            pred = _forward(model_d, batch)
            masked_val = float(target_fn(pred).item())
        del batch, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return orig_val - masked_val

    last_cluster = 0

    def _progress(done: int, total: int) -> None:
        nonlocal last_cluster
        delta = max(0, int(done) - last_cluster)
        last_cluster = int(done)
        if progress_reporter is not None and delta:
            progress_reporter.advance(
                delta,
                phase="vit_cx",
                detail=f"{var_name} cluster {done}/{total}",
            )

    devices = attribution_devices(device)

    if devices:
        # Multi-GPU: score every cluster's occlusion in parallel (one replica per
        # GPU), then run the reduction below over the scores in cluster order.
        # Each task reproduces score_clusters' exact upsampled mask, so the
        # result is identical to the single-GPU path.
        replicas = get_replicas(model, device, devices)

        def _task(i: int, model_d, device_d) -> float:
            mask_full = sk_resize(
                patch_masks[i], (INPUT_H, INPUT_W), order=0,
                preserve_range=True, anti_aliasing=False,
            )
            return _score(model_d, device_d, mask_full.astype(np.float32))

        scores = parallel_map_scores(
            _task, n_clusters, replicas,
            progress_callback=_progress if progress_reporter is not None else None,
        )
        score_iter = iter(scores)

        def scorer_fn(occlusion_mask_hw: np.ndarray) -> float:
            return next(score_iter)

        accum_progress = None
    else:
        def scorer_fn(occlusion_mask_hw: np.ndarray) -> float:
            return _score(model, device, occlusion_mask_hw)

        accum_progress = _progress if progress_reporter is not None else None

    partial_sal, partial_wgt = score_clusters(
        scorer_fn=scorer_fn,
        patch_masks=patch_masks,
        feat_H=feat_H, feat_W=feat_W,
        H=INPUT_H, W=INPUT_W,
        cluster_indices=list(range(n_clusters)),
        verbose=True, rank=0,
        progress_callback=accum_progress,
    )

    # Falsy sigma (0 / 0.0 / None) disables post-smoothing, giving the raw
    # upsampled cluster map.
    sigma = smooth_sigma if smooth_sigma else None
    return aggregate_and_upsample(
        partial_sal, partial_wgt, H=INPUT_H, W=INPUT_W, smooth_sigma=sigma
    ).astype(np.float32)


def _run_vit_cx(
    case: CaseData,
    target_fn,
    atmos_vars: list[str],
    surf_vars: list[str],
    model,
    device: str,
    baseline_sigma_deg: float = IG_BASELINE_SIGMA_DEG,
    progress_reporter: Optional[ProgressReporter] = None,
    **vit_kwargs,
) -> tuple[dict[str, dict[str, np.ndarray]], None]:
    attributions: dict[str, dict[str, np.ndarray]] = {}

    for v in surf_vars:
        if progress_reporter is not None:
            progress_reporter.set_phase("vit_cx", f"{v} baseline")
        baseline = _smooth_tensor(case.surf_cpu[v], baseline_sigma_deg)
        attr_hw = _run_vit_cx_one_var(
            case, target_fn, v, "surf", baseline, model, device,
            progress_reporter=progress_reporter, **vit_kwargs
        )
        attributions[v] = {"sfc": attr_hw.astype(np.float32)}
        _gpu_sync_and_gc()

    for v in atmos_vars:
        if progress_reporter is not None:
            progress_reporter.set_phase("vit_cx", f"{v} baseline")
        baseline = _smooth_tensor(case.atmos_cpu[v], baseline_sigma_deg)
        attr_hw = _run_vit_cx_one_var(
            case, target_fn, v, "atmos", baseline, model, device,
            progress_reporter=progress_reporter, **vit_kwargs
        )
        # ViT-CX, like RISE, produces one (H, W) map shared by every level, so
        # store it once under the single column-uniform layer.
        attributions[v] = {COLLAPSED_LEVEL_KEY: attr_hw.astype(np.float32)}
        _gpu_sync_and_gc()

    return attributions, None
