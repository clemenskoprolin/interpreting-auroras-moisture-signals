"""Login-node overlay execution for the sbatch-* listener modes.

Overlay pulls (``OverlayRequest``) only read raw ERA5 fields and slice them —
they never touch the Aurora model or a GPU (see ``compute._pull_overlay_local``,
which just calls ``load_case`` and copies a CPU array out).  In the sbatch-*
modes the listener already runs on the login node, so submitting a whole GPU
allocation per overlay is wasteful: the same work can be done in-process here.

``LocalOverlayRunner`` is composed by the sbatch backends.  An overlay
``submit()`` is delegated to it, and ``status`` / ``get_result`` route to it for
the job ids it owns.  Everything else (saliency, ig, ...) still goes to SLURM.

Jobs run on a small thread pool so a burst of overlay requests does not spawn
an unbounded number of threads competing for the login node.
"""

from __future__ import annotations

import threading
import time
import traceback

from .protocol import JobStatus, OverlayRequest, new_job_id
from .server_log import report_job_failure
from . import transfer_guard


class LocalOverlayRunner:
    """Compute overlay pulls in-process on the login node."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        # Bound concurrent loads so overlay bursts don't hammer the login node.
        self._slots = threading.Semaphore(max(1, int(max_concurrent)))

    # ── Ownership ──────────────────────────────────────────────────────────

    def owns(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def active(self) -> int:
        """Number of jobs still queued or running (for health reporting)."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j["status"] in ("queued_locally", "running")
            )

    # ── Backend protocol ────────────────────────────────────────────────────

    def submit(self, req: OverlayRequest) -> str:
        job_id = new_job_id()
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued_locally",
                "result": None,
                "error": None,
                "log": "",
            }
        threading.Thread(
            target=self._run,
            args=(job_id, req),
            daemon=True,
            name="geoxplain-aurora-adapter-overlay",
        ).start()
        return job_id

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        done = job["status"] == "done"
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            eta_s=0.0 if done else None,
            progress=1.0 if done else None,
            text_output=None,
            log_tail=job.get("log", "")[-200:],
            error_message=job.get("error"),
        )

    def get_result(self, job_id: str) -> bytes:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] != "done":
            raise ValueError(f"Job {job_id} is not done yet (status: {job['status']})")
        return job["result"]

    # ── Worker ────────────────────────────────────────────────────────────

    def _run(self, job_id: str, req: OverlayRequest) -> None:
        with self._slots:
            self._set_status(job_id, "running")
            try:
                from ..engine.overlay_compute import _pull_overlay_local
                result = _pull_overlay_local(
                    req.variable,
                    req.timestamps,
                    level=req.level,
                    **req.options,
                )
                packed = result.to_msgpack()
                with self._lock:
                    self._jobs[job_id]["result"] = packed
                self._set_status(job_id, "done")
                print(f"[LocalOverlay] Job {job_id} done — {result.summary()}")
            except Exception as exc:
                err = traceback.format_exc()
                with self._lock:
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["log"] = err[-2000:]
                self._set_status(job_id, "error")
                report_job_failure(
                    job_id,
                    str(exc),
                    log_tail=(
                        f"{err.rstrip()}\n"
                        "If the login node cannot load ERA5 data, restart the "
                        "listener with --no-overlay-on-login to run overlays on a "
                        "GPU job instead."
                    ),
                    source="LocalOverlay",
                )

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = status
            if status in ("done", "error"):
                job["done_at"] = time.time()

    # ── Retention ───────────────────────────────────────────────────────────

    def purge(self, retention_s: float) -> int:
        """Drop completed overlay jobs past retention. Returns count removed."""
        cutoff = time.time() - retention_s
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job.get("done_at") is not None and job["done_at"] < cutoff
                and not transfer_guard.is_pinned(jid)
            ]
            for jid in stale:
                del self._jobs[jid]
        return len(stale)
