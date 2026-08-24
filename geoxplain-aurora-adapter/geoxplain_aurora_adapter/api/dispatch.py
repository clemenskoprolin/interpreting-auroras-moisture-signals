"""Dispatch core for geoxplain_aurora_adapter.

This module holds the routing layer behind the public ``run_<method>`` /
``pull_overlay`` wrappers (in :mod:`methods`) and ``listen_for_request`` (in
:mod:`listener`).  Mode selection is automatic:

    run_<method>(..., remote=None)    → in-process compute (local)
    run_<method>(..., remote="http://...") → HTTP client (gpu-listener / sbatch-oneshot / sbatch-persistent)

GPU detection uses ``torch.cuda.is_available()`` if torch is installed, else
``False`` (correct for client-only installations: no torch = no compute path).
sbatch detection uses ``shutil.which("sbatch")``.  Both results are cached.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from typing import Optional

from ..schema.metadata import AURORA_LEVELS
from ..schema.overlay import OverlayResult
from ..schema.result import XiaResult
from ..schema.spec import TargetSpec
from .timeparse import _expand_timeframe_targets


# ── Session timestamp registry ────────────────────────────────────────────────
#
# Every XIA computation run this session records its target timestamp(s) here,
# in first-seen order.  ``pull_overlay`` uses them to infer ``dates`` when the
# caller omits it — so an overlay lines up with the frames already explained.

_SESSION_TIMESTAMPS: list[str] = []


def _record_session_timestamps(timestamps) -> None:
    """Remember frame timestamps from an XIA run for later overlay inference."""
    for ts in timestamps:
        if ts and ts not in _SESSION_TIMESTAMPS:
            _SESSION_TIMESTAMPS.append(ts)


def _record_result_timestamps(result) -> None:
    """Record the displayed timestamps of a result's frames.

    Recording from the returned frames — rather than the input targets — keeps
    the session list in lock-step with what the viewer actually shows. A
    single-frame result's ``timestamp`` is the requested time (Aurora's input
    step t1; the explained 6 h prediction is t2 = t1 + 6 h), while a rollout
    frame's is the forecast valid time of that step. Either way, an
    auto-inferred overlay (``pull_overlay`` with ``dates=None``) lines up with
    the explained frames regardless of the single / batch / rollout path.
    """
    frames = getattr(result, "frames", None) or []
    _record_session_timestamps(getattr(frame, "timestamp", None) for frame in frames)
    return result


def session_timestamps() -> list[str]:
    """Return the XIA target timestamps recorded this session (first-seen order)."""
    return list(_SESSION_TIMESTAMPS)


# ── Environment detection ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _has_gpu() -> bool:
    """Return True if a CUDA GPU is available (cached on first call)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _has_sbatch() -> bool:
    """Return True if ``sbatch`` is on PATH (cached on first call)."""
    return shutil.which("sbatch") is not None


# ── Lazy model singleton ──────────────────────────────────────────────────────

_model_cache: dict = {}  # key → (model, device)


def _get_or_load_model(checkpoint_path: Optional[str] = None):
    """Return (model, device), loading the model at most once per process."""
    import torch
    device = "cuda" if _has_gpu() else "cpu"
    key = (device, checkpoint_path or "__default__")
    if key not in _model_cache:
        from ..engine.model import load_model
        _model_cache[key] = (load_model(device, checkpoint_path=checkpoint_path), device)
    return _model_cache[key]


# ── Local dispatch ────────────────────────────────────────────────────────────

def _run_local_dispatch(
    method: str,
    target: TargetSpec,
    input_vars: list[str],
    checkpoint_path: Optional[str] = None,
    **options,
) -> XiaResult:
    if not _has_gpu():
        raise RuntimeError(
            "geoxplain_aurora_adapter: no GPU visible and no `remote=` URL provided.\n"
            "  • On a GPU allocation, re-run without `remote=`.\n"
            "  • From a non-GPU machine, pass `remote='http://<host>:<port>'` to\n"
            "    route the computation to a remote listener.\n"
            "  • To start a listener: geoxplain-aurora-adapter listen --help"
        )
    model, device = _get_or_load_model(checkpoint_path)
    from ..engine.compute import _run_local
    return _record_result_timestamps(
        _run_local(method, target, input_vars, model, device, **options)
    )


def _run_local_batch_dispatch(
    method: str,
    targets: list[TargetSpec],
    input_vars: list[str],
    checkpoint_path: Optional[str] = None,
    **options,
) -> XiaResult:
    if not _has_gpu():
        raise RuntimeError(
            "geoxplain_aurora_adapter: no GPU visible and no `remote=` URL provided.\n"
            "  * On a GPU allocation, re-run without `remote=`.\n"
            "  * From a non-GPU machine, pass `remote='http://<host>:<port>'` to\n"
            "    route the batch computation to a remote listener.\n"
            "  * To start a listener: geoxplain-aurora-adapter listen --help"
        )
    model, device = _get_or_load_model(checkpoint_path)
    from ..engine.compute import _run_local_batch
    return _record_result_timestamps(
        _run_local_batch(method, targets, input_vars, model, device, **options)
    )


# ── Remote dispatch ───────────────────────────────────────────────────────────

def _run_remote(
    method: str,
    target: TargetSpec,
    input_vars: list[str],
    remote: str,
    **options,
) -> XiaResult:
    from ..remote.client import run_remote
    return _record_result_timestamps(
        run_remote(method, target, input_vars, remote, **options)
    )


def _run_remote_batch(
    method: str,
    targets: list[TargetSpec],
    input_vars: list[str],
    remote: str,
    **options,
) -> XiaResult:
    from ..remote.client import run_remote_batch
    return _record_result_timestamps(
        run_remote_batch(method, targets, input_vars, remote, **options)
    )


def _pull_overlay_remote(
    variable: str,
    timestamps: list[str],
    remote: str,
    **options,
) -> OverlayResult:
    from ..remote.overlay_client import pull_remote_overlay
    return pull_remote_overlay(variable, timestamps, remote, **options)


def _run_batch(
    method: str,
    target: TargetSpec,
    input: list[str],
    *,
    timeframes: int,
    step_hours: int = 6,
    remote: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    **options,
) -> XiaResult:
    targets = _expand_timeframe_targets(
        target,
        timeframes=timeframes,
        step_hours=step_hours,
    )
    if remote:
        return _run_remote_batch(method, targets, input, remote, **options)
    return _run_local_batch_dispatch(
        method,
        targets,
        input,
        checkpoint_path=checkpoint_path,
        **options,
    )


def _validate_levels(levels) -> Optional[list[int]]:
    """Normalize and validate a user-supplied ``levels=`` selection.

    Accepts a single int or any iterable of ints (pressure levels in hPa),
    returns a de-duplicated list in the caller's order, or ``None`` when
    ``levels`` is ``None`` (meaning "all levels", the default).  Raises
    ``ValueError`` if any level is not one of Aurora's pressure levels — so
    the caller gets an immediate, GPU-free error rather than a silent empty
    attribution after the computation has run.
    """
    if levels is None:
        return None
    if isinstance(levels, int):
        levels = [levels]
    try:
        levels = [int(lvl) for lvl in levels]
    except (TypeError, ValueError):
        raise ValueError(
            f"levels must be an int or an iterable of ints (hPa). Got: {levels!r}"
        )
    if not levels:
        raise ValueError(
            "levels is empty — pass at least one pressure level, "
            "or omit levels= to attribute all levels."
        )
    unknown = [lvl for lvl in levels if lvl not in AURORA_LEVELS]
    if unknown:
        raise ValueError(
            f"Unsupported level(s) {unknown} in levels=. "
            f"Supported levels (hPa): {list(AURORA_LEVELS)}"
        )
    seen: set[int] = set()
    return [lvl for lvl in levels if not (lvl in seen or seen.add(lvl))]
