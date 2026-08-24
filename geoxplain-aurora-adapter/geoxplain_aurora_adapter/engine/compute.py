"""Local compute orchestrator for geoxplain_aurora_adapter.

``_run_local`` is the single entry point called by the dispatch layer for
in-process computation (local) and by the server's worker for remote
computation (gpu-listener, sbatch-oneshot, sbatch-persistent).  It loads the
case, builds the target, dispatches to the per-method runner, and packs the
result into a ``XiaResult``.  ``_run_local_batch`` runs the same for several
targets and returns one multi-frame bundle.

The actual attribution work lives in sibling modules:

- ``runners/`` — one module per method (saliency, ig, rise, vit_cx), the
  Aurora-specific glue around each algorithm in ``xia_methods``.
- ``rollout.py`` — autoregressive rollout (``_run_local_rollout``).
- ``overlay_compute.py`` — raw ERA5 overlays.
- ``_common.py`` — constants and helpers shared across the above.
"""

from __future__ import annotations

import socket
import time
from typing import Optional

from ._common import (
    _detect_diverging,
    _estimate_total_units,
    _layer_labels_for,
    _parse_init_time,
    _split_input_vars,
)
from .data import ATMOS_VARS, SURF_VARS, load_case
from ..schema.metadata import method_display_name
from .progress import ProgressReporter
from ..schema.result import XiaResult
from .rollout import _run_local_rollout
from .runners import _run_ig, _run_rise, _run_saliency, _run_vit_cx
from ..schema.spec import TargetSpec
from ..schema.targets import build_target_fn


def _run_local(
    method: str,
    target: TargetSpec,
    input_vars: list[str],
    model,
    device: str,
    progress_reporter: Optional[ProgressReporter] = None,
    finish_progress: bool = True,
    **options,
) -> XiaResult:
    """Run an XIA method in-process and return a ``XiaResult``.

    Parameters
    ----------
    method:     ``"saliency"`` | ``"ig"`` | ``"rise"`` | ``"vit_cx"``.
    target:     ``TargetSpec`` describing the scalar to explain.
    input_vars: Input variable names to attribute (subset of
                ``ATMOS_VARS ∪ SURF_VARS``).
    model:      Loaded Aurora model (from ``geoxplain_aurora_adapter.model.load_model``).
    device:     ``"cuda"`` or ``"cpu"``.
    **options:  Method-specific keyword arguments forwarded to the runner:
                - saliency: (none)
                - ig: ``n_steps`` (int, default 32),
                      ``baseline_sigma_deg`` (float, default 2.5)
                - rise: ``n_masks`` (int), ``cells_h``, ``cells_w``, ``seed``,
                        ``baseline_sigma_deg``
                - vit_cx: ``hook_stage`` (int), ``n_clusters`` (int),
                          ``smooth_sigma`` (float | (lat, lon) | 0 to disable),
                          ``baseline_sigma_deg``
    """
    rollout_timeframes = options.pop("_rollout_timeframes", None)
    if rollout_timeframes is not None:
        return _run_local_rollout(
            method,
            target,
            input_vars,
            model,
            device,
            timeframes=rollout_timeframes,
            progress_reporter=progress_reporter,
            finish_progress=finish_progress,
            **options,
        )

    t0 = time.time()
    atmos_vars, surf_vars = _split_input_vars(input_vars)

    if progress_reporter is None:
        progress_reporter = ProgressReporter(
            method,
            _estimate_total_units(
                method,
                n_frames=1,
                n_vars=len(atmos_vars) + len(surf_vars),
                options=options,
            ),
            print_updates=True,
        )
    else:
        estimated_total = _estimate_total_units(
            method,
            n_frames=1,
            n_vars=len(atmos_vars) + len(surf_vars),
            options=options,
        )
        if progress_reporter.total_units is None and estimated_total is not None:
            progress_reporter.set_total(estimated_total, emit=False)

    progress_reporter.set_phase("reading data")
    init_time = _parse_init_time(target.timestamp)
    case = load_case(init_time)
    target_fn = build_target_fn(target, case)

    score: Optional[float] = None

    if method == "saliency":
        attributions, score = _run_saliency(
            case, target_fn, atmos_vars, surf_vars, model, device,
            levels=options.get("levels"),
            progress_reporter=progress_reporter,
        )
    elif method == "ig":
        attributions, score = _run_ig(
            case, target_fn, atmos_vars, surf_vars, model, device,
            progress_reporter=progress_reporter, **options
        )
    elif method == "rise":
        attributions, score = _run_rise(
            case, target_fn, atmos_vars, surf_vars, model, device,
            progress_reporter=progress_reporter, **options
        )
    elif method == "vit_cx":
        attributions, score = _run_vit_cx(
            case, target_fn, atmos_vars, surf_vars, model, device,
            progress_reporter=progress_reporter, **options
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose from: 'saliency', 'ig', 'rise', 'vit_cx'."
        )

    runtime_s = time.time() - t0
    diverging = _detect_diverging(attributions)

    # A single forward pass explains the model's one-step (6 h) prediction.  The
    # frame keeps the *requested* timestamp as its displayed time so the
    # explanation sits on the timeline exactly where it was asked for, and
    # overlays pulled for this frame default to the same clock time.  We do NOT
    # shift the displayed timestamp to the forecast valid time (init + 6 h);
    # ``lead_hours`` records the 6 h prediction horizon as metadata instead.
    lead_hours = 6

    meta: dict = {
        "runtime_s": round(runtime_s, 2),
        "host": socket.gethostname(),
        "method_options": options,
        "input_timestamp": target.timestamp,
        "lead_hours": lead_hours,
    }
    if score is not None:
        meta["target_score"] = float(score)

    # Human-readable names for the vertical layers actually present, so the
    # viewer can show "850 hPa" rather than the bare order index.
    layer_labels = _layer_labels_for(case, attributions, method)

    if finish_progress:
        progress_reporter.finish("done")

    return XiaResult.single(
        method=method,
        method_label=method_display_name(method),
        target=target,
        timestamp=target.timestamp,
        attributions=attributions,
        diverging=diverging,
        meta=meta,
        layer_labels=layer_labels,
    )


def _run_local_batch(
    method: str,
    targets: list[TargetSpec],
    input_vars: list[str],
    model,
    device: str,
    progress_reporter: Optional[ProgressReporter] = None,
    finish_progress: bool = True,
    **options,
) -> XiaResult:
    """Run an XIA method for multiple targets and return one multi-frame bundle."""
    if not targets:
        raise ValueError("targets is empty - batch jobs need at least one target.")

    t0 = time.time()
    frames = []
    layer_labels: dict[str, str] = {}
    per_frame_runtime = 0.0
    known_vars = [v for v in input_vars if v in ATMOS_VARS or v in SURF_VARS]
    if progress_reporter is None:
        progress_reporter = ProgressReporter(
            f"{method} batch",
            _estimate_total_units(
                method,
                n_frames=len(targets),
                n_vars=len(known_vars),
                options=options,
            ),
            total_frames=len(targets),
            print_updates=True,
        )
    else:
        estimated_total = _estimate_total_units(
            method,
            n_frames=len(targets),
            n_vars=len(known_vars),
            options=options,
        )
        if progress_reporter.total_units is None and estimated_total is not None:
            progress_reporter.set_total(estimated_total, emit=False)

    for idx, target in enumerate(targets):
        progress_reporter.set_frame(idx + 1)
        single = _run_local(
            method,
            target,
            input_vars,
            model,
            device,
            progress_reporter=progress_reporter,
            finish_progress=False,
            **options,
        )
        frame = single.frames[0]
        frame.meta = {
            **single.meta,
            **frame.meta,
            "batch_index": idx,
        }
        per_frame_runtime += float(single.meta.get("runtime_s", 0.0))
        frames.append(frame)
        layer_labels.update(single.layer_labels)

    runtime_s = time.time() - t0
    step_hours = None
    if len(targets) > 1:
        first = _parse_init_time(targets[0].timestamp)
        second = _parse_init_time(targets[1].timestamp)
        step_hours = (second - first).total_seconds() / 3600.0

    meta = {
        "batch": True,
        "timeframes": len(targets),
        "start_timestamp": frames[0].timestamp,
        "end_timestamp": frames[-1].timestamp,
        "runtime_s": round(runtime_s, 2),
        "frame_runtime_s": round(per_frame_runtime, 2),
        "host": socket.gethostname(),
        "method_options": options,
    }
    if step_hours is not None:
        meta["step_hours"] = step_hours

    if finish_progress:
        progress_reporter.finish("done")

    return XiaResult(
        method=method,
        method_label=method_display_name(method),
        frames=frames,
        layer_labels=layer_labels,
        meta=meta,
    )
