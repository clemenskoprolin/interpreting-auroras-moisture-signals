"""Public XIA / overlay method wrappers for geoxplain_aurora_adapter.

These are the user-facing ``run_<method>`` functions and ``pull_overlay``.
Mode selection is automatic:

    run_<method>(..., remote=None)    → in-process compute (local)
    run_<method>(..., remote="http://...") → HTTP client (gpu-listener / sbatch-oneshot / sbatch-persistent)

The actual routing helpers live in :mod:`dispatch`. They are referenced
through the ``dispatch`` module (rather than imported by name) so that the
single/batch/remote/local entry points remain a single, well-defined dispatch
surface — and so tests can monkeypatch them on the ``dispatch`` module.
"""

from __future__ import annotations

from typing import Optional

from . import dispatch
from ..schema.metadata import (
    OVERLAY_COLORMAPS,
    default_overlay_colormap,
    default_overlay_label,
    default_overlay_unit,
)
from ..schema.overlay import OverlayResult
from ..schema.result import XiaResult
from ..schema.spec import TargetSpec
from .timeparse import _expand_overlay_timestamps, _shift_timestamp


# How far (in hours) to shift the overlay's *fetched* field relative to the
# *displayed* frame time, per ``overlay_time`` choice.  The frame keeps its
# requested timestamp (it is NOT shifted to the forecast valid time), and that
# requested time is Aurora's most-recent input step t1.  Aurora's window is two
# input steps (t0 = t1-6h, t1) feeding the t2 = t1+6h prediction, so relative to
# the displayed (= t1) frame:
#   "input"     -> t1 (the displayed frame itself)         offset   0  (default)
#   "prior"     -> t0 (the earliest input step)            offset  -6
#   "predicted" -> t2 (the forecast valid time explained)  offset  +6
_OVERLAY_TIME_OFFSETS: dict[str, int] = {
    "input": 0,
    "prior": -6,
    "predicted": 6,
}

# Optional free-text annotation the viewer shows alongside the offset.  Keyed by
# the same ``overlay_time`` choice as the offsets above.  The default ("input",
# the frame's own time) gets no annotation.
_OVERLAY_TIME_LABELS: dict[str, str | None] = {
    "input": None,
    "prior": "Aurora input step t0",
    "predicted": "Forecast valid time t2",
}


def pull_overlay(
    variable: str,
    dates: str | list[str] | tuple[str, ...] | None = None,
    *,
    level: Optional[int] = None,
    remote: Optional[str] = None,
    name: Optional[str] = None,
    unit: Optional[str] = None,
    colormap: Optional[str] = None,
    visible: bool = True,
    overlay_time: str = "input",
    step_hours: int = 6,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
    poll_max_s: float = 30.0,
) -> OverlayResult:
    """Pull ERA5 fields as timestamped viewer overlays.

    ``dates`` accepts ISO timestamps, dates, ranges, or ``None`` to reuse the
    XIA frame timestamps recorded this session. The displayed frame keeps its
    requested timestamp (Aurora's most-recent input step t1), so ``overlay_time``
    chooses the field time relative to it: ``"input"`` (0 h, the default — the
    frame's own time t1), ``"prior"`` (-6 h, the earlier input step t0), or
    ``"predicted"`` (+6 h, the forecast valid time t2). It also sets a
    ``time_label`` annotation on the result for the non-default choices
    (``"prior"`` → ``"Aurora input step t0"``, ``"predicted"`` → ``"Forecast
    valid time t2"``), which the viewer shows next to the offset. Remote calls
    use the listener at ``remote``; local calls read from the configured
    dataset.
    """

    if overlay_time not in _OVERLAY_TIME_OFFSETS:
        raise ValueError(
            f"overlay_time must be one of {sorted(_OVERLAY_TIME_OFFSETS)}. "
            f"Got: {overlay_time!r}"
        )
    offset_hours = _OVERLAY_TIME_OFFSETS[overlay_time]

    if dates is None:
        display_timestamps = dispatch.session_timestamps()
        if not display_timestamps:
            raise ValueError(
                f"pull_overlay({variable!r}) was called without `dates=`, but no "
                "XIA computations have been run this session, so there is no "
                "timestamp to infer from. Run an explanation first (e.g. "
                "ax.run_saliency(...)), or pass dates= explicitly, for example "
                "dates='2024-01-16T12:00:00Z'."
            )
    else:
        display_timestamps = _expand_overlay_timestamps(dates, step_hours=step_hours)

    # Fetch the field from the (possibly shifted) clock time, but keep the frame
    # labelled with the displayed time so it lines up with the explained frame.
    fetch_timestamps = [_shift_timestamp(ts, offset_hours) for ts in display_timestamps]

    if colormap is None:
        colormap = default_overlay_colormap(variable)
    elif colormap not in OVERLAY_COLORMAPS:
        raise ValueError(
            f"colormap must be one of {OVERLAY_COLORMAPS}. Got: {colormap!r}"
        )

    options = {
        "level": level,
        "name": name if name is not None else default_overlay_label(variable, level),
        "unit": unit if unit is not None else default_overlay_unit(variable),
        "colormap": colormap,
        "visible": visible,
    }
    if remote:
        result = dispatch._pull_overlay_remote(
            variable,
            fetch_timestamps,
            remote,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            poll_max_s=poll_max_s,
            **options,
        )
    else:
        from ..engine.overlay_compute import _pull_overlay_local
        result = _pull_overlay_local(variable, fetch_timestamps, **options)

    if offset_hours != 0:
        _apply_overlay_time_shift(result, display_timestamps, offset_hours)
    time_label = _OVERLAY_TIME_LABELS.get(overlay_time)
    if time_label is not None:
        try:
            result.time_label = time_label
        except AttributeError:
            pass
    return result


def _apply_overlay_time_shift(result, display_timestamps, offset_hours: int) -> None:
    """Relabel overlay frames to their displayed time and record the offset.

    The field data was fetched from ``display + offset_hours``; the frame is
    re-labelled with the displayed (requested) time so it co-locates with the
    explained frame on the timeline, and ``overlay_offset_hours`` records the
    shift so the viewer can annotate it (e.g. "Specific Humidity (−6h)").
    """
    for frame, display_ts in zip(getattr(result, "frames", []), display_timestamps):
        frame.timestamp = display_ts
    try:
        result.overlay_offset_hours = offset_hours
    except AttributeError:
        pass

def run_saliency(
    target: TargetSpec,
    input: list[str],
    *,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    timeframes: int = 1,
    step_hours: int = 6,
    levels: Optional[int | list[int]] = None,
    **options,
) -> XiaResult:
    """Compute vanilla gradient saliency attribution.

    Parameters
    ----------
    target:         What scalar to explain.
    input:          Input variable names to attribute (e.g. ``["t", "q", "z"]``).
    remote:         If set, delegates computation to the listener at this URL
                    (e.g. ``"http://gpu01:8765"`` or
                    ``"http://localhost:8765"`` for an SSH-tunnelled cluster).
                    If ``None``, runs in-process on the local GPU.
    checkpoint_path:Override the default checkpoint path (local mode only).
    timeframes:     Number of consecutive timeframes to compute.  Values
                    greater than 1 return one multi-frame XiaResult.
    step_hours:     Hours between consecutive timeframes when timeframes > 1.
    levels:         Pressure levels (hPa) to return attribution for, e.g.
                    ``[925, 850, 700]``.  ``None`` (the default) returns every
                    level in :data:`AURORA_LEVELS`.  Only affects atmospheric
                    input variables; surface variables are unaffected.  The
                    single backward pass computes gradients for all levels
                    regardless, so this filters the output rather than reducing
                    GPU cost.
    **options:      No other method-specific options for saliency.

    Returns
    -------
    XiaResult
        One frame, or a multi-frame bundle when ``timeframes > 1``.
    """
    levels = dispatch._validate_levels(levels)
    if levels is not None:
        options["levels"] = levels
    if timeframes != 1:
        return dispatch._run_batch(
            "saliency",
            target,
            input,
            timeframes=timeframes,
            step_hours=step_hours,
            remote=remote,
            checkpoint_path=checkpoint_path,
            **options,
        )
    if step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")
    if remote:
        return dispatch._run_remote("saliency", target, input, remote, **options)
    return dispatch._run_local_dispatch("saliency", target, input,
                                        checkpoint_path=checkpoint_path, **options)


def run_ig(
    target: TargetSpec,
    input: list[str],
    *,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    timeframes: int = 1,
    step_hours: int = 6,
    n_steps: int = 32,
    baseline_sigma_deg: float = 2.5,
    levels: Optional[int | list[int]] = None,
    **options,
) -> XiaResult:
    """Compute Integrated Gradients attribution.

    Parameters
    ----------
    target:             What scalar to explain.
    input:              Input variable names to attribute.
    remote:             Listener URL, or ``None`` for in-process.
    checkpoint_path:    Override checkpoint path (local mode only).
    timeframes:         Number of consecutive timeframes to compute.
    step_hours:         Hours between consecutive timeframes when timeframes > 1.
    n_steps:            Number of integration steps (midpoint Riemann rule).
    baseline_sigma_deg: Gaussian sigma (degrees latitude) for the smoothed baseline.
    levels:             Pressure levels (hPa) to return attribution for, e.g.
                        ``[925, 850, 700]``.  ``None`` (the default) returns
                        every level in :data:`AURORA_LEVELS`.  Only affects
                        atmospheric input variables; surface variables are
                        unaffected.  Each integration step's backward pass
                        computes gradients for all levels regardless, so this
                        filters the output rather than reducing GPU cost.

    Returns
    -------
    XiaResult
        One frame, or a multi-frame bundle when ``timeframes > 1``.
    """
    levels = dispatch._validate_levels(levels)
    if levels is not None:
        options["levels"] = levels
    if timeframes != 1:
        return dispatch._run_batch(
            "ig",
            target,
            input,
            timeframes=timeframes,
            step_hours=step_hours,
            remote=remote,
            checkpoint_path=checkpoint_path,
            n_steps=n_steps,
            baseline_sigma_deg=baseline_sigma_deg,
            **options,
        )
    if step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")
    if remote:
        return dispatch._run_remote("ig", target, input, remote,
                                    n_steps=n_steps, baseline_sigma_deg=baseline_sigma_deg, **options)
    return dispatch._run_local_dispatch("ig", target, input, checkpoint_path=checkpoint_path,
                                        n_steps=n_steps, baseline_sigma_deg=baseline_sigma_deg,
                                        **options)


def run_rise(
    target: TargetSpec,
    input: list[str],
    *,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    timeframes: int = 1,
    step_hours: int = 6,
    n_masks: int = 1200,
    cells_h: int = 400,
    cells_w: int = 800,
    seed: int = 42,
    baseline_sigma_deg: float = 2.5,
    **options,
) -> XiaResult:
    """Compute RISE (Randomized Input Sampling for Explanation) attribution.

    Parameters
    ----------
    target:             What scalar to explain.
    input:              Input variable names to attribute.
    remote:             Listener URL, or ``None`` for in-process.
    checkpoint_path:    Override checkpoint path (local mode only).
    timeframes:         Number of consecutive timeframes to compute.
    step_hours:         Hours between consecutive timeframes when timeframes > 1.
    n_masks:            Number of random masks.
    cells_h/cells_w:    Low-resolution grid dimensions for mask upsampling.
    seed:               Random seed for mask generation.
    baseline_sigma_deg: Gaussian sigma for the smoothed baseline field.

    Returns
    -------
    XiaResult
        One frame, or a multi-frame bundle when ``timeframes > 1``.
    """
    kw = dict(n_masks=n_masks, cells_h=cells_h, cells_w=cells_w,
              seed=seed, baseline_sigma_deg=baseline_sigma_deg)
    if timeframes != 1:
        return dispatch._run_batch(
            "rise",
            target,
            input,
            timeframes=timeframes,
            step_hours=step_hours,
            remote=remote,
            checkpoint_path=checkpoint_path,
            **kw,
            **options,
        )
    if step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")
    if remote:
        return dispatch._run_remote("rise", target, input, remote, **kw, **options)
    return dispatch._run_local_dispatch("rise", target, input, checkpoint_path=checkpoint_path, **kw)


def run_vit_cx(
    target: TargetSpec,
    input: list[str],
    *,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    timeframes: int = 1,
    step_hours: int = 6,
    hook_stage: int = 1,
    n_clusters: int = 4096,
    smooth_sigma: float | tuple | None = 0,
    baseline_sigma_deg: float = 2.5,
    **options,
) -> XiaResult:
    """Compute ViT-CX (cluster-based causal attribution) attribution.

    Parameters
    ----------
    target:             What scalar to explain.
    input:              Input variable names to attribute.
    remote:             Listener URL, or ``None`` for in-process.
    checkpoint_path:    Override checkpoint path (local mode only).
    timeframes:         Number of consecutive timeframes to compute.
    step_hours:         Hours between consecutive timeframes when timeframes > 1.
    hook_stage:         Aurora encoder stage to hook (0-2; default 1).
    n_clusters:         Fixed cluster budget = number of occlusion forward passes
                        per variable.  Bounds run cost regardless of input.
    smooth_sigma:       Gaussian post-smoothing of the upsampled attribution map.
                        Default ``0`` disables smoothing and keeps the raw
                        cluster map.  A non-zero scalar applies the same sigma
                        to both axes; a ``(lat, lon)`` pair sets them
                        independently.
    baseline_sigma_deg: Gaussian sigma for the smoothed baseline field.

    Returns
    -------
    XiaResult
        One frame, or a multi-frame bundle when ``timeframes > 1``.
    """
    kw = dict(hook_stage=hook_stage, n_clusters=n_clusters,
              smooth_sigma=smooth_sigma, baseline_sigma_deg=baseline_sigma_deg)
    if timeframes != 1:
        return dispatch._run_batch(
            "vit_cx",
            target,
            input,
            timeframes=timeframes,
            step_hours=step_hours,
            remote=remote,
            checkpoint_path=checkpoint_path,
            **kw,
            **options,
        )
    if step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")
    if remote:
        return dispatch._run_remote("vit_cx", target, input, remote, **kw, **options)
    return dispatch._run_local_dispatch("vit_cx", target, input, checkpoint_path=checkpoint_path, **kw)


def run_rollout(
    target: TargetSpec,
    input: list[str],
    *,
    method: str = "saliency",
    timeframes: int,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    n_steps: int = 32,
    baseline_sigma_deg: float = 2.5,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
    poll_max_s: float = 30.0,
) -> XiaResult:
    """Compute autoregressive rollout XIA in fixed six-hour Aurora frames.

    Parameters
    ----------
    target:
        Scalar output target for the first frame.
    input:
        Input variable names to attribute.
    method:
        ``"saliency"`` or ``"ig"``. RISE and ViT-CX are recognized method
        identifiers but are not implemented for rollout.
    timeframes:
        Required positive number of autoregressive frames.
    remote:
        Listener URL, or ``None`` for in-process execution.
    checkpoint_path:
        Override the default checkpoint in local mode only.
    n_steps, baseline_sigma_deg:
        Integrated Gradients options, ignored for Saliency.
    timeout_s, poll_interval_s, poll_max_s:
        Remote-client timeout and polling controls.

    Returns
    -------
    XiaResult
        One bundle containing all rollout frames.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``timeframes`` is not positive.
    NotImplementedError
        If rollout is requested with RISE or ViT-CX.
    """
    if method not in {"saliency", "ig", "rise", "vit_cx"}:
        raise ValueError(
            f"Unknown rollout method {method!r}. "
            "Choose from: 'saliency', 'ig', 'rise', 'vit_cx'."
        )
    if method not in {"saliency", "ig"}:
        raise NotImplementedError(
            f"run_rollout(method={method!r}) is only implemented for 'saliency' and 'ig'."
        )

    # Session timestamps are recorded from the returned frames (valid times)
    # by the dispatch helpers below, so the auto-overlay lines up with the
    # displayed rollout frames rather than the input init times.
    options = {"_rollout_timeframes": timeframes}
    if method == "ig":
        options.update(n_steps=n_steps, baseline_sigma_deg=baseline_sigma_deg)

    if remote:
        return dispatch._run_remote(
            method,
            target,
            input,
            remote,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            poll_max_s=poll_max_s,
            **options,
        )
    return dispatch._run_local_dispatch(
        method,
        target,
        input,
        checkpoint_path=checkpoint_path,
        **options,
    )
