"""Progress reporting helpers for long-running XIA computations."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


ProgressCallback = Callable[[dict], None]

PROGRESS_STEP = 0.05        # emit at most once per +5% of progress
PROGRESS_MAX_QUIET_S = 15.0  # ...but always emit if this long has passed


def should_show_progress(
    progress: Optional[float],
    last_shown_progress: Optional[float],
    seconds_since_shown: float,
    *,
    step: float = PROGRESS_STEP,
    max_quiet_s: float = PROGRESS_MAX_QUIET_S,
) -> bool:
    """Decide whether a progress update is worth surfacing.

    Show it when progress has advanced by at least ``step`` since the last shown
    update, or when ``max_quiet_s`` has elapsed with nothing shown.  The first
    update (``last_shown_progress is None``) always shows.  For indeterminate
    progress (``progress is None``) only the time fallback applies.
    """
    if last_shown_progress is None:
        return True
    if seconds_since_shown >= max_quiet_s:
        return True
    if progress is None:
        return False
    # Small epsilon so exact step boundaries (e.g. 0.06 - 0.01) aren't lost to
    # binary floating-point rounding.
    return (progress - last_shown_progress) >= step - 1e-9


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def write_status_file(path: str, snapshot: dict) -> None:
    """Atomically write a progress snapshot for sbatch oneshot polling."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    os.replace(tmp_path, path)


def read_status_file(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class ProgressSnapshot:
    progress: Optional[float]
    eta_s: Optional[float]
    text_output: str

    def to_dict(self) -> dict:
        return {
            "progress": self.progress,
            "eta_s": self.eta_s,
            "text_output": self.text_output,
        }


class ProgressReporter:
    """Render and publish throttled progress for normal and batch XIA jobs."""

    def __init__(
        self,
        label: str,
        total_units: Optional[int] = None,
        *,
        total_frames: int = 1,
        min_interval_s: float = 2.0,
        warmup_units: int = 3,
        warmup_fraction: float = 0.05,
        status_callback: Optional[ProgressCallback] = None,
        print_updates: bool = False,
        heartbeat_s: Optional[float] = None,
        progress_step: float = PROGRESS_STEP,
        max_quiet_s: float = PROGRESS_MAX_QUIET_S,
    ) -> None:
        self.label = label
        self.total_units = max(0, int(total_units)) if total_units is not None else None
        self.total_frames = max(1, int(total_frames))
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.warmup_units = max(1, int(warmup_units))
        self.warmup_fraction = min(1.0, max(0.0, float(warmup_fraction)))
        self.status_callback = status_callback
        self.print_updates = print_updates
        self.heartbeat_s = float(heartbeat_s) if heartbeat_s else None
        self.progress_step = max(0.0, float(progress_step))
        self.max_quiet_s = max(0.0, float(max_quiet_s))

        self.completed_units = 0.0
        self.current_frame = 1
        self.phase = "starting"
        self.detail = ""
        self.started_at = time.monotonic()
        self._first_progress_at: Optional[float] = None
        self._last_emit_at = 0.0
        self._last_emit_progress: Optional[float] = None
        self._last_text = ""
        self._done = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._maybe_start_heartbeat()

    def _maybe_start_heartbeat(self) -> None:
        # A heartbeat re-emits the current snapshot on a timer so that the
        # indeterminate bar (e.g. a single-pass ``saliency`` forward/backward
        # that never calls ``advance``) keeps refreshing its elapsed time on
        # both the client and the server CLI instead of freezing until done.
        if not self.heartbeat_s or (self.status_callback is None and not self.print_updates):
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="progress-heartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_s):
            if self._done:
                break
            self._emit(force=True)

    def set_total(
        self,
        total_units: Optional[int],
        *,
        force: bool = True,
        emit: bool = True,
    ) -> None:
        self.total_units = max(0, int(total_units)) if total_units is not None else None
        if not emit:
            return
        self._emit(force=force)

    def add_total_units(self, units: int, *, force: bool = False) -> None:
        units = int(units)
        if units <= 0:
            return
        self.total_units = (self.total_units or 0) + units
        self._emit(force=force)

    def set_frame(self, frame_index: int, *, force: bool = True) -> None:
        self.current_frame = min(max(1, int(frame_index)), self.total_frames)
        self._emit(force=force)

    def set_phase(
        self,
        phase: str,
        detail: str = "",
        *,
        force: bool = True,
    ) -> None:
        self.phase = phase
        self.detail = detail
        self._emit(force=force)

    def advance(
        self,
        units: float = 1,
        *,
        phase: Optional[str] = None,
        detail: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if phase is not None:
            self.phase = phase
        if detail is not None:
            self.detail = detail
        self.completed_units += max(0.0, float(units))
        if self.total_units is not None:
            self.completed_units = min(self.completed_units, float(self.total_units))
        if self.completed_units > 0 and self._first_progress_at is None:
            # Anchor the ETA clock to when real progress began, so a long
            # idle preamble (model load, data read) doesn't inflate the rate.
            self._first_progress_at = time.monotonic()
        self._emit(force=force)

    def finish(self, detail: str = "done") -> None:
        if self.total_units is not None:
            self.completed_units = self.total_units
        self.phase = detail
        self.detail = ""
        self._done = True
        self._stop_event.set()
        self._emit(force=True)

    def snapshot(self) -> ProgressSnapshot:
        now = time.monotonic()
        elapsed = max(0.0, now - self.started_at)
        progress = self._progress()
        eta_s = self._eta(now, progress)
        return ProgressSnapshot(
            progress=progress,
            eta_s=eta_s,
            text_output=self._render(progress, eta_s, elapsed),
        )

    def _progress(self) -> Optional[float]:
        if self.total_units is None or self.total_units <= 0:
            return 1.0 if self._done else None
        return min(1.0, max(0.0, self.completed_units / self.total_units))

    def _eta(self, now: float, progress: Optional[float]) -> Optional[float]:
        if self._done:
            return 0.0
        if progress is None or progress <= 0.0 or self._first_progress_at is None:
            return None
        # Measure the rate from when progress began, not from job start, so a
        # long idle preamble (model load, data read) doesn't inflate the ETA.
        progress_elapsed = now - self._first_progress_at
        if progress_elapsed <= 0.0:
            progress_elapsed = 1e-9
        # Need enough signal before extrapolating.  Honor the integer warmup
        # when the total is large enough to reach it (multi-step methods like
        # ig/rise); otherwise — e.g. saliency, where the whole job is one
        # frame-unit advanced in sub-block fractions — fall back to a progress
        # fraction so the ETA appears partway through the single pass.
        warmed = (
            self.completed_units >= self.warmup_units
            or progress >= self.warmup_fraction
        )
        if not warmed:
            return None
        remaining_fraction = max(0.0, 1.0 - progress)
        return remaining_fraction * (progress_elapsed / progress)

    def _render(
        self,
        progress: Optional[float],
        eta_s: Optional[float],
        elapsed: float,
    ) -> str:
        width = 24
        if progress is None:
            bar = "[" + "-" * width + "]"
            pct = "  0%"
        else:
            filled = min(width, max(0, int(round(progress * width))))
            bar = "[" + "#" * filled + "-" * (width - filled) + "]"
            pct = f"{progress * 100:3.0f}%"

        parts = [f"{self.label} {bar} {pct}"]
        if self.total_frames > 1:
            parts.append(f"frame {self.current_frame}/{self.total_frames}")
        if self.total_units:
            parts.append(f"{int(round(self.completed_units))}/{self.total_units}")
        parts.append(self.phase)
        if self.detail:
            parts.append(self.detail)
        if eta_s is None:
            parts.append(f"ETA estimating (elapsed {format_duration(elapsed)})")
        else:
            parts.append(f"ETA {format_duration(eta_s)}")
        return " | ".join(parts)

    def _emit(self, *, force: bool = False) -> None:
        with self._lock:
            snapshot = self.snapshot()
            now = time.monotonic()
            text = snapshot.text_output
            should_emit = (
                force
                or text != self._last_text
                and now - self._last_emit_at >= self.min_interval_s
                and should_show_progress(
                    snapshot.progress,
                    self._last_emit_progress,
                    now - self._last_emit_at,
                    step=self.progress_step,
                    max_quiet_s=self.max_quiet_s,
                )
            )
            if not should_emit:
                return
            self._last_emit_at = now
            self._last_emit_progress = snapshot.progress
            self._last_text = text
            data = snapshot.to_dict()
            if self.status_callback is not None:
                self.status_callback(data)
            if self.print_updates:
                print(text, flush=True)
