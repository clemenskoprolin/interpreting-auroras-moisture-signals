"""RISE runner — randomized-mask perturbation attribution for Aurora.

N random masks per variable, each scored by a forward pass; the covariance of
score against mask gives the saliency.  Mask scoring fans out across all
visible GPUs and the reduction runs in mask order, so the result is identical
to the single-GPU path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .._common import (
    COLLAPSED_LEVEL_KEY,
    IG_BASELINE_SIGMA_DEG,
    INPUT_H,
    INPUT_W,
    RISE_CELLS_H,
    RISE_CELLS_W,
    RISE_N_MASKS_DEFAULT,
    RISE_P,
    _gpu_sync_and_gc,
    _make_masked_batch,
    _smooth_tensor,
)
from ..data import CaseData
from ..progress import ProgressReporter


def _run_rise_one_var(
    case: CaseData,
    target_fn,
    var_name: str,
    var_type: str,
    baseline_cpu: torch.Tensor,
    model,
    device: str,
    n_masks: int = RISE_N_MASKS_DEFAULT,
    cells_h: int = RISE_CELLS_H,
    cells_w: int = RISE_CELLS_W,
    seed: int = 42,
    progress_reporter: Optional[ProgressReporter] = None,
) -> np.ndarray:
    from ..xia_methods.rise import (
        accumulate_rise_with_stats,
        generate_rise_masks,
        normalize_rise_covariance,
    )
    from ..model import forward as _forward
    from ..parallel import attribution_devices, get_replicas, parallel_map_scores

    def _score(model_d, device_d, mask_np: np.ndarray) -> float:
        batch = _make_masked_batch(case, device_d, mask_np, var_name, baseline_cpu, var_type)
        with torch.no_grad():
            pred = _forward(model_d, batch)
            val = float(target_fn(pred).item())
        del batch, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val

    last_mask = 0

    def _progress(done: int, total: int) -> None:
        nonlocal last_mask
        delta = max(0, int(done) - last_mask)
        last_mask = int(done)
        if progress_reporter is not None and delta:
            progress_reporter.advance(
                delta,
                phase="rise",
                detail=f"{var_name} mask {done}/{total}",
            )

    mask_kwargs = dict(
        cells_h=cells_h, cells_w=cells_w, H=INPUT_H, W=INPUT_W, p=RISE_P, seed=seed,
    )
    devices = attribution_devices(device)

    if devices:
        # Multi-GPU: score every mask in parallel (one replica per GPU), then run
        # the canonical reduction below over the scores in mask order.  Masks are
        # regenerated deterministically per index, so they match the masks the
        # reduction regenerates — keeping the result identical to single-GPU.
        replicas = get_replicas(model, device, devices)

        def _task(i: int, model_d, device_d) -> float:
            mask_np = next(generate_rise_masks(n=1, start_idx=i, **mask_kwargs))
            return _score(model_d, device_d, mask_np)

        scores = parallel_map_scores(
            _task, n_masks, replicas,
            progress_callback=_progress if progress_reporter is not None else None,
        )
        score_iter = iter(scores)

        def scorer_fn(mask_np: np.ndarray) -> float:
            return next(score_iter)

        accum_progress = None
    else:
        def scorer_fn(mask_np: np.ndarray) -> float:
            return _score(model, device, mask_np)

        accum_progress = _progress if progress_reporter is not None else None

    sal, msk, msk2, score_sum, n_seen = accumulate_rise_with_stats(
        scorer_fn=scorer_fn,
        n=n_masks,
        cells_h=cells_h, cells_w=cells_w,
        H=INPUT_H, W=INPUT_W,
        p=RISE_P, seed=seed, start_idx=0,
        verbose=True, rank=0,
        progress_callback=accum_progress,
    )
    return normalize_rise_covariance(sal, msk, msk2, score_sum, n_seen)


def _run_rise(
    case: CaseData,
    target_fn,
    atmos_vars: list[str],
    surf_vars: list[str],
    model,
    device: str,
    baseline_sigma_deg: float = IG_BASELINE_SIGMA_DEG,
    progress_reporter: Optional[ProgressReporter] = None,
    **rise_kwargs,
) -> tuple[dict[str, dict[str, np.ndarray]], None]:
    attributions: dict[str, dict[str, np.ndarray]] = {}

    for v in surf_vars:
        if progress_reporter is not None:
            progress_reporter.set_phase("rise", f"{v} baseline")
        baseline = _smooth_tensor(case.surf_cpu[v], baseline_sigma_deg)
        attr_hw = _run_rise_one_var(
            case, target_fn, v, "surf", baseline, model, device,
            progress_reporter=progress_reporter, **rise_kwargs
        )
        attributions[v] = {"sfc": attr_hw.astype(np.float32)}
        _gpu_sync_and_gc()

    for v in atmos_vars:
        if progress_reporter is not None:
            progress_reporter.set_phase("rise", f"{v} baseline")
        baseline = _smooth_tensor(case.atmos_cpu[v], baseline_sigma_deg)
        attr_hw = _run_rise_one_var(
            case, target_fn, v, "atmos", baseline, model, device,
            progress_reporter=progress_reporter, **rise_kwargs
        )
        # RISE spatial mask is at full (H, W) but applies across all levels
        # simultaneously, so every level would carry an identical 2-D map.
        # Store it once under a single column-uniform layer (COLLAPSED_LEVEL_KEY).
        attributions[v] = {COLLAPSED_LEVEL_KEY: attr_hw.astype(np.float32)}
        _gpu_sync_and_gc()

    return attributions, None
