"""Shared progress mirroring for listener backends."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .progress_log import format_progress_log

# Fetch a fresh ``{progress, eta_s, text_output}`` snapshot for a job, given its
# id and a copy of its record.  Return ``None`` when no snapshot is available
# this pass (source not ready / unreachable) so cached values are left intact.
SnapshotFetcher = Callable[[str, dict], Optional[dict]]


def apply_snapshot(job: dict, snapshot: dict) -> Optional[str]:
    """Cache ``snapshot`` onto ``job``; return bar text to print if it changed.

    The caller must hold the backend's jobs lock.  Returns the progress text
    when it differs from what was last printed for this job (recording it so the
    same bar isn't printed twice), else ``None``.
    """
    text = snapshot.get("text_output")
    job["progress"] = snapshot.get("progress")
    job["eta_s"] = snapshot.get("eta_s")
    job["text_output"] = text
    if text and text != job.get("_server_printed_text_output"):
        job["_server_printed_text_output"] = text
        return text
    return None


def print_snapshot(job_id: str, job: dict, snapshot: dict) -> None:
    """``apply_snapshot`` + print the bar (caller must hold the jobs lock).

    For backends that already hold the lock and have the snapshot in hand
    (e.g. an in-process progress callback).
    """
    to_print = apply_snapshot(job, snapshot)
    if to_print:
        print(format_progress_log(job_id, to_print), flush=True)


def start_progress_mirror(
    jobs: dict,
    jobs_lock,
    fetch: SnapshotFetcher,
    *,
    interval_s: float,
    name: str = "geoxplain-aurora-adapter-progress-poller",
) -> threading.Thread:
    """Spawn a daemon thread mirroring active jobs' progress to the terminal.

    Every ``interval_s`` it fetches a fresh snapshot for each non-terminal job
    and reprints the bar (via ``format_progress_log``) whenever the text
    changes.  ``fetch`` is the only backend-specific part.
    """
    def _loop() -> None:
        while True:
            time.sleep(interval_s)
            with jobs_lock:
                active = [
                    (jid, dict(job))
                    for jid, job in jobs.items()
                    if job.get("status") not in ("done", "error")
                ]
            for job_id, job_copy in active:
                try:
                    snapshot = fetch(job_id, job_copy)
                except Exception:
                    snapshot = None  # transient; try again next pass
                if not snapshot:
                    continue
                with jobs_lock:
                    job = jobs.get(job_id)
                    if job is None:
                        continue
                    to_print = apply_snapshot(job, snapshot)
                if to_print:
                    print(format_progress_log(job_id, to_print), flush=True)

    thread = threading.Thread(target=_loop, daemon=True, name=name)
    thread.start()
    return thread
