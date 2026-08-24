"""In-process backend for geoxplain_aurora_adapter (local and gpu-listener).

A single worker thread processes jobs one at a time.  The Aurora model is
loaded once at startup (or lazily on the first request if ``prewarm=False``).

Single-worker concurrency is by design: Aurora at 0.25° already uses ~92 GiB
VRAM on one GH200, leaving no room for a second concurrent inference.

local — Jupyter on a GPU allocation, direct in-process call:
    ``dispatch.py`` calls ``compute._run_local`` directly, bypassing HTTP.

gpu-listener — GPU allocation with an HTTP listener:
    This module starts a FastAPI server via uvicorn.  The client on the login
    node or Mac connects via SSH tunnel.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Optional

from ..engine.progress import ProgressReporter
from .progress_mirror import print_snapshot
from .protocol import HealthResponse, JobStatus, RunRequest, new_job_id
from .retention import DEFAULT_JANITOR_INTERVAL_S, start_janitor
from .server_log import report_job_failure
from .server import app, set_backend
from . import transfer_guard


class InprocBackend:
    """Single-GPU, single-worker backend."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        memory_retention_s: Optional[float] = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._device: Optional[str] = None
        self._model_warm = False
        self._model_lock = threading.Lock()

        # job store: job_id -> {"status", "result", "error", "log", progress fields}
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        self._memory_retention_s = memory_retention_s
        self.retention_info = {
            "enabled": bool(memory_retention_s),
            "interval_s": DEFAULT_JANITOR_INTERVAL_S,
            "windows": f"memory={memory_retention_s:.0f}s" if memory_retention_s else "",
        }
        start_janitor(
            self._purge,
            enabled=bool(memory_retention_s),
            description=f"memory={memory_retention_s:.0f}s" if memory_retention_s else "",
            announce=False,
        )

        # FIFO work queue
        self._work_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="geoxplain-aurora-adapter-worker"
        )
        self._worker_thread.start()

    # ── Model loading ─────────────────────────────────────────────────────

    def warm(self) -> None:
        with self._model_lock:
            if self._model is None:
                self._load_model()

    def _load_model(self) -> None:
        import torch
        from ..engine.model import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[InprocBackend] Loading model on {device}...")
        self._model = load_model(device, checkpoint_path=self._checkpoint_path)
        self._device = device
        self._model_warm = True
        print("[InprocBackend] Model ready.")

    # ── Worker ────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while True:
            job_id, req = self._work_queue.get()
            self._set_status(job_id, "running")
            try:
                if hasattr(req, "timestamps"):
                    from ..engine.overlay_compute import _pull_overlay_local
                    result = _pull_overlay_local(
                        req.variable,
                        req.timestamps,
                        level=req.level,
                        **req.options,
                    )
                else:
                    with self._model_lock:
                        if self._model is None:
                            self._load_model()
                        model = self._model
                        device = self._device

                    from ..schema.spec import TargetSpec
                    progress = ProgressReporter(
                        f"{req.method} batch" if hasattr(req, "targets") else req.method,
                        None,
                        total_frames=len(req.targets) if hasattr(req, "targets") else 1,
                        min_interval_s=0.1,
                        status_callback=lambda snap, jid=job_id: self._set_progress(jid, snap),
                        # Printing is done in _set_progress with the [job_id]
                        # prefix (shared with the sbatch backends), not raw here.
                        print_updates=False,
                        heartbeat_s=15.0,
                    )
                    if hasattr(req, "targets"):
                        from ..engine.compute import _run_local_batch
                        targets = [TargetSpec.from_dict(t) for t in req.targets]
                        result = _run_local_batch(
                            req.method, targets, req.input_vars, model, device,
                            progress_reporter=progress,
                            **req.options
                        )
                    else:
                        from ..engine.compute import _run_local
                        target = TargetSpec.from_dict(req.target)
                        result = _run_local(
                            req.method, target, req.input_vars, model, device,
                            progress_reporter=progress,
                            **req.options
                        )
                result_bytes = result.to_msgpack()
                with self._jobs_lock:
                    self._jobs[job_id]["result"] = result_bytes
                self._set_status(job_id, "done")
                print(f"[InprocBackend] Job {job_id} done.")
            except Exception as exc:
                import traceback
                err = traceback.format_exc()
                with self._jobs_lock:
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["log"] = err[-2000:]
                self._set_status(job_id, "error")
                report_job_failure(
                    job_id,
                    str(exc),
                    log_tail=err,
                    source="InprocBackend",
                )
            finally:
                self._work_queue.task_done()

    def _set_status(self, job_id: str, status: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = status
            if status in ("done", "error"):
                job["done_at"] = time.time()

    def _purge(self) -> None:
        """Drop completed jobs (and their cached result bytes) past memory retention."""
        if not self._memory_retention_s:
            return
        cutoff = time.time() - self._memory_retention_s
        with self._jobs_lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job.get("done_at") is not None and job["done_at"] < cutoff
                and not transfer_guard.is_pinned(jid)
            ]
            for jid in stale:
                del self._jobs[jid]
    def _set_progress(self, job_id: str, snapshot: dict) -> None:
        # Compute runs in this process, so progress is pushed here via callback
        # rather than polled.  Cache it and mirror the bar to the listener
        # terminal with the same [job_id]-prefixed format as the sbatch modes.
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            print_snapshot(job_id, job, snapshot)

    # ── Backend protocol ──────────────────────────────────────────────────

    def submit(self, req: RunRequest) -> str:
        job_id = new_job_id()
        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": "queued",
                "result": None,
                "error": None,
                "log": "",
                "progress": None,
                "eta_s": None,
                "text_output": None,
            }
        self._work_queue.put((job_id, req))
        queue_pos = self._work_queue.qsize()
        if queue_pos > 1:
            print(
                f"[InprocBackend] Job {job_id} queued (position {queue_pos}). "
                "Note: only one job runs at a time on this node."
            )
        return job_id

    def status(self, job_id: str) -> JobStatus:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        s = job["status"]
        return JobStatus(
            job_id=job_id,
            status=s,
            eta_s=job.get("eta_s"),
            progress=job.get("progress"),
            text_output=job.get("text_output"),
            log_tail=job.get("log", "")[-200:],
            error_message=job.get("error"),
        )

    def get_result(self, job_id: str) -> bytes:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] != "done":
            raise ValueError(f"Job {job_id} is not done yet (status: {job['status']})")
        return job["result"]

    def health(self) -> HealthResponse:
        return HealthResponse(
            mode="inproc",
            model_warm=self._model_warm,
            queue_depth=self._work_queue.qsize(),
        )


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    prewarm: bool = False,
    checkpoint_path: Optional[str] = None,
    memory_retention: Optional[str] = None,
) -> None:
    """Start the inproc FastAPI server.  Blocks until the server is stopped.

    ``memory_retention`` bounds how long completed jobs and cached result bytes
    are kept (default ``DEFAULT_MEMORY_RETENTION`` when not given; ``"never"``
    disables purging).  This is also the backend the persistent GPU worker runs.
    """
    import uvicorn

    from ..serving.config import DEFAULT_MEMORY_RETENTION, parse_retention

    if memory_retention is None:
        memory_retention = DEFAULT_MEMORY_RETENTION
    backend = InprocBackend(
        checkpoint_path=checkpoint_path,
        memory_retention_s=parse_retention(memory_retention),
    )
    if prewarm:
        print("[InprocBackend] Pre-warming model...")
        backend.warm()

    set_backend(backend)

    from ..api.dispatch import _has_gpu, _has_sbatch
    from ..serving.cli_style import render_listener_banner

    hostname = socket.gethostname()
    render_listener_banner(
        mode="gpu-listener",
        hostname=hostname,
        port=port,
        gpu=_has_gpu(),
        sbatch=_has_sbatch(),
        bind_host=host,
        retention=backend.retention_info,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")
