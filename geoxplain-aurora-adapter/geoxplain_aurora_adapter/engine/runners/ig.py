"""Integrated Gradients runner — gradient attribution along a baseline path.

The baseline is a spatially smoothed copy of the actual field
(``baseline_sigma_deg`` controls the smoothing); attribution integrates the
gradient over the straight-line path from baseline to actual.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._common import IG_BASELINE_SIGMA_DEG, _smooth_tensor, _split_atmos_levels
from ..data import CaseData, make_batch
from ..progress import ProgressReporter
from ..rollout import _RolloutForwardWrapper, _saved_tensors_cpu_context


def _run_ig(
    case: CaseData,
    target_fn,
    atmos_vars: list[str],
    surf_vars: list[str],
    model,
    device: str,
    n_steps: int = 32,
    baseline_sigma_deg: float = IG_BASELINE_SIGMA_DEG,
    rollout_steps: int = 1,
    levels: Optional[list[int]] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> tuple[dict[str, dict[str, np.ndarray]], Optional[float]]:
    from ..xia_methods.ig import integrated_gradients

    keep_levels = set(levels) if levels else None

    # Build per-variable baselines (spatially smoothed actual values)
    atmos_actual = {v: case.atmos_cpu[v] for v in atmos_vars}
    atmos_baseline = {v: _smooth_tensor(case.atmos_cpu[v], baseline_sigma_deg) for v in atmos_vars}
    surf_actual = {v: case.surf_cpu[v] for v in surf_vars}
    surf_baseline = {v: _smooth_tensor(case.surf_cpu[v], baseline_sigma_deg) for v in surf_vars}

    def batch_fn(alpha: float = 0.0, requires_grad: bool = False):
        a_overrides = {
            v: (atmos_baseline[v] + alpha * (atmos_actual[v] - atmos_baseline[v])).clone()
            for v in atmos_vars
        }
        s_overrides = {
            v: (surf_baseline[v] + alpha * (surf_actual[v] - surf_baseline[v])).clone()
            for v in surf_vars
        }
        return make_batch(
            case, device,
            requires_grad_atmos=tuple(atmos_vars) if requires_grad else (),
            requires_grad_surf=tuple(surf_vars) if requires_grad else (),
            atmos_overrides=a_overrides if atmos_vars else None,
            surf_overrides=s_overrides if surf_vars else None,
        )

    last_step = 0

    def _progress(done: int, total: int) -> None:
        nonlocal last_step
        delta = max(0, int(done) - last_step)
        last_step = int(done)
        if progress_reporter is not None and delta:
            progress_reporter.advance(
                delta,
                phase="ig",
                detail=f"step {done}/{total}",
            )

    if progress_reporter is not None:
        progress_reporter.set_phase("ig", f"preparing {n_steps} steps")

    model_fwd = _RolloutForwardWrapper(model, rollout_steps) if rollout_steps > 1 else model
    with _saved_tensors_cpu_context(rollout_steps > 1):
        result = integrated_gradients(
            model=model_fwd,
            batch_fn=batch_fn,
            target_fn=target_fn,
            atmos_actual=atmos_actual or None,
            atmos_baseline=atmos_baseline or None,
            atmos_var_names=tuple(atmos_vars),
            surf_actual=surf_actual or None,
            surf_baseline=surf_baseline or None,
            surf_var_names=tuple(surf_vars),
            device=device,
            n_steps=n_steps,
            progress_callback=_progress if progress_reporter is not None else None,
        )

    attributions: dict[str, dict[str, np.ndarray]] = {}
    for v in atmos_vars:
        ig_arr = result["ig"].get(v)
        if ig_arr is not None:
            attributions[v] = _split_atmos_levels(ig_arr, case.pressure_levels, keep_levels)
    for v in surf_vars:
        ig_arr = result["ig"].get(v)
        if ig_arr is not None:
            attributions[v] = {"sfc": ig_arr[0, 1].astype(np.float32)}

    return attributions, result.get("score")
