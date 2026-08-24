"""sbatch-oneshot backend: one sbatch per request (default sbatch-backed mode).

For each incoming ``POST /run``, this backend:
    1. Renders a submit script into a temp file.
    2. Calls ``sbatch`` and records the SLURM job ID.
    3. Polls ``squeue --me`` every few seconds to track ``PD → R → CG``.
    4. Tails the job log file for ``GET /jobs/{id}`` ``log_tail``.
    5. Reads the output ``.xia.npz`` written by the sbatch job back into
       a ``XiaResult`` and delivers it via the msgpack result endpoint.

The project is currently limited to 2 queued SLURM jobs.  Additional
requests are serialized in an in-process local queue and reported as
``"queued_locally"`` until a SLURM slot is free.

The sbatch worker script imports ``geoxplain_aurora_adapter.remote.worker_oneshot`` which
runs the method and saves the result to a temp path passed on the command line.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from typing import Optional

from ..engine.progress import read_status_file
from .local_overlay import LocalOverlayRunner
from .progress_mirror import apply_snapshot, start_progress_mirror
from .progress_log import format_progress_log
from .protocol import (
    ERROR_LOG_TAIL_CHARS,
    HealthResponse,
    JobStatus,
    RunRequest,
    log_tail_for_status,
    new_job_id,
)
from .retention import DEFAULT_JANITOR_INTERVAL_S, start_janitor
from .sbatch_config import SbatchConfig, ResolvedSbatchConfig
from .server import app, set_backend
from .server_log import report_job_failure
from .slurm import _sbatch, _squeue_eta, _squeue_states, _ts
from . import transfer_guard


_PROGRESS_POLL_INTERVAL_S = 0.5


class SbatchOneshotBackend:
    """Login-node backend: one sbatch per request."""

    def __init__(
        self,
        resolved_cfg: ResolvedSbatchConfig,
        memory_retention_s: Optional[float] = None,
        result_retention_s: Optional[float] = None,
    ) -> None:
        self._cfg = resolved_cfg
        self._memory_retention_s = memory_retention_s   # in-memory job records + bytes
        self._result_retention_s = result_retention_s   # on-disk result directories
        self._job_limit = resolved_cfg.job_limit
        # Overlay pulls don't need a GPU; compute them in this login-node
        # process instead of submitting a SLURM job (unless disabled).
        self._local_overlay = (
            LocalOverlayRunner() if resolved_cfg.overlay_on_login else None
        )
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        self._local_queue: queue.Queue = queue.Queue()
        self._active_slurm: list[str] = []  # SLURM job IDs currently queued/running
        self._slurm_lock = threading.Lock()

        # Coordinator thread: dequeues local jobs and submits to SLURM
        self._coordinator = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="geoxplain-aurora-adapter-coordinator"
        )
        self._coordinator.start()
        # Poller thread: updates status of submitted SLURM jobs
        self._poller = threading.Thread(
            target=self._poller_loop, daemon=True, name="geoxplain-aurora-adapter-poller"
        )
        self._poller.start()
        # Status files are cheap to read and update much more frequently than
        # SLURM state.  Poll them separately so the listener terminal mirrors
        # worker progress and clients can see fresh cached values even between
        # squeue polls.
        self._progress_poller = start_progress_mirror(
            self._jobs, self._jobs_lock, self._fetch_status_file,
            interval_s=_PROGRESS_POLL_INTERVAL_S,
        )
        windows = []
        if memory_retention_s:
            windows.append(f"memory={memory_retention_s:.0f}s")
        if result_retention_s:
            windows.append(f"disk={result_retention_s:.0f}s")
        enabled = bool(memory_retention_s or result_retention_s)
        self.retention_info = {
            "enabled": enabled,
            "interval_s": DEFAULT_JANITOR_INTERVAL_S,
            "windows": " ".join(windows),
        }
        start_janitor(
            self._purge,
            enabled=enabled,
            description=" ".join(windows),
            announce=False,
        )

    # ── Coordinator: submit from local queue to SLURM when slots free ──────

    def _coordinator_loop(self) -> None:
        while True:
            job_id, req = self._local_queue.get()
            # Wait until a SLURM slot is free
            while True:
                with self._slurm_lock:
                    if len(self._active_slurm) < self._job_limit:
                        break
                time.sleep(5)
            self._submit_to_slurm(job_id, req)
            self._local_queue.task_done()

    def _submit_to_slurm(self, job_id: str, req: RunRequest) -> None:
        try:
            # All per-job files live under one shared-FS directory so the
            # worker (running on a compute node, often inside a container
            # with its own private /tmp) can read what the listener wrote.
            out_dir = os.path.join(self._cfg.output_dir, "xia_results", job_id)
            os.makedirs(out_dir, exist_ok=True)
            result_name = "result.overlay.npz" if hasattr(req, "timestamps") else "result.xia.npz"
            out_path = os.path.join(out_dir, result_name)
            req_path = os.path.join(out_dir, "request.json")
            status_path = os.path.join(out_dir, "status.json")
            script_path = os.path.join(out_dir, "submit.sh")

            with open(req_path, "w") as rf:
                payload = {
                    "job_id": job_id,
                    "options": req.options,
                }
                if hasattr(req, "timestamps"):
                    payload["variable"] = req.variable
                    payload["timestamps"] = req.timestamps
                    payload["level"] = req.level
                elif hasattr(req, "targets"):
                    payload["method"] = req.method
                    payload["input_vars"] = req.input_vars
                    payload["targets"] = req.targets
                else:
                    payload["method"] = req.method
                    payload["input_vars"] = req.input_vars
                    payload["target"] = req.target
                json.dump(payload, rf)

            body = (
                f"python -m geoxplain_aurora_adapter.remote.worker_oneshot "
                f"--request {req_path} --output {out_path} --status {status_path}"
            )
            script_content = self._cfg.render_submit_script(body)
            with open(script_path, "w") as sf:
                sf.write(script_content)
            os.chmod(script_path, 0o755)

            slurm_job_id = _sbatch(script_path)
            print(f"[SbatchOneshot] [{_ts()}] Job {job_id} submitted as SLURM {slurm_job_id}")

            # Resolve the SLURM log path the same way SLURM does
            # (`output` template uses %x=job-name=geoxplain-aurora-adapter, %j=slurm_jid).
            cfg_output = self._cfg.output.replace("%x", "geoxplain-aurora-adapter").replace("%j", slurm_job_id)
            if not os.path.isabs(cfg_output):
                cfg_output = os.path.abspath(os.path.join(os.getcwd(), cfg_output))

            with self._jobs_lock:
                self._jobs[job_id]["slurm_job_id"] = slurm_job_id
                self._jobs[job_id]["out_path"] = out_path
                self._jobs[job_id]["req_path"] = req_path
                self._jobs[job_id]["status_path"] = status_path
                self._jobs[job_id]["script_path"] = script_path
                self._jobs[job_id]["log_file"] = cfg_output
                self._jobs[job_id]["status"] = "queued"

            with self._slurm_lock:
                self._active_slurm.append(slurm_job_id)

        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            with self._jobs_lock:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)
                self._jobs[job_id]["log"] = err[-2000:]
                self._jobs[job_id].setdefault("done_at", time.time())
            report_job_failure(
                job_id,
                str(exc),
                log_tail=err,
                source="SbatchOneshot",
            )

    # ── Poller: update status of active SLURM jobs ──────────────────────────

    def _poller_loop(self) -> None:
        while True:
            time.sleep(5)
            with self._slurm_lock:
                active = list(self._active_slurm)
            if not active:
                continue

            states = _squeue_states(active)
            for slurm_jid, state in states.items():
                self._update_slurm_job(slurm_jid, state)

    def _fetch_status_file(self, job_id: str, job: dict) -> Optional[dict]:
        """Snapshot source for the progress mirror: read the worker's status file."""
        return read_status_file(job.get("status_path")) or None

    def _refresh_progress(self, job_id: str) -> None:
        """Refresh one job's cached progress now (e.g. on a SLURM state change).

        The background mirror polls continuously; this is for the off-cadence
        refresh after a status transition so the cache and terminal don't lag.
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            status_path = job.get("status_path") if job is not None else None
        snapshot = read_status_file(status_path)
        if not snapshot:
            return
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            to_print = apply_snapshot(job, snapshot)
        if to_print:
            print(format_progress_log(job_id, to_print), flush=True)

    def _update_slurm_job(self, slurm_jid: str, state: str) -> None:
        # Find the aurora job_id for this SLURM job
        with self._jobs_lock:
            aurora_id = next(
                (jid for jid, j in self._jobs.items()
                 if j.get("slurm_job_id") == slurm_jid),
                None
            )
        if aurora_id is None:
            return

        with self._jobs_lock:
            job = self._jobs[aurora_id]
            old_status = job.get("status")
            log_file = job.get("log_file", "")
            out_path = job.get("out_path", "")
            status_path = job.get("status_path")

        # Map SLURM states to geoxplain_aurora_adapter statuses.
        #
        # COMPLETING (CG) is a *transition* state, not a terminal one:
        # the controller is still tearing down nodes and flushing the
        # worker's output file.  Treat it as in-flight and keep polling.
        # Only on COMPLETED (CD) is the job truly finished; if the output
        # file still isn't visible there, give the shared FS a couple of
        # seconds to flush before declaring failure.
        if state in ("PENDING", "PD"):
            new_status = "queued"
        elif state in ("RUNNING", "R", "CONFIGURING", "CF"):
            new_status = "running"
        elif state in ("COMPLETING", "CG"):
            # Output file may not be flushed yet; result might still appear.
            new_status = "done" if (out_path and os.path.exists(out_path)) else "completing"
        elif state in ("COMPLETED", "CD"):
            if out_path and os.path.exists(out_path):
                new_status = "done"
            else:
                # Brief grace period for shared-FS flush.
                missed = 0
                for _ in range(3):
                    time.sleep(1.0)
                    if out_path and os.path.exists(out_path):
                        break
                    missed += 1
                if missed < 3:
                    new_status = "done"
                else:
                    new_status = "error"
        elif state in (
            "FAILED", "F", "CANCELLED", "CA", "TIMEOUT", "TO",
            "NODE_FAIL", "NF", "OUT_OF_MEMORY", "OOM",
            "BOOT_FAIL", "BF", "DEADLINE", "DL", "PREEMPTED", "PR",
        ):
            new_status = "error"
        else:
            new_status = old_status

        # Update status and tail log
        log_content = ""
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()[-ERROR_LOG_TAIL_CHARS:]
            except OSError:
                pass
        with self._jobs_lock:
            self._jobs[aurora_id]["status"] = new_status
            if log_content:
                self._jobs[aurora_id]["log"] = log_content
            if new_status == "error" and not self._jobs[aurora_id].get("error"):
                self._jobs[aurora_id]["error"] = f"SLURM job {slurm_jid} ended with state {state}"
            if new_status in ("done", "error"):
                self._jobs[aurora_id].setdefault("done_at", time.time())
            error_message = self._jobs[aurora_id].get("error")

        if new_status == "error" and old_status != "error":
            report_job_failure(
                aurora_id,
                error_message,
                log_tail=log_content,
                source="SbatchOneshot",
            )

        self._refresh_progress(aurora_id)

        if new_status in ("done", "error"):
            with self._slurm_lock:
                if slurm_jid in self._active_slurm:
                    self._active_slurm.remove(slurm_jid)

    # ── Backend protocol ──────────────────────────────────────────────────

    def submit(self, req: RunRequest) -> str:
        # Overlay requests carry ``timestamps``; run them on the login node.
        if self._local_overlay is not None and hasattr(req, "timestamps"):
            return self._local_overlay.submit(req)
        job_id = new_job_id()
        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": "queued_locally",
                "slurm_job_id": None,
                "result": None,
                "error": None,
                "log": "",
                "out_path": None,
                "status_path": None,
                "progress": None,
                "eta_s": None,
                "text_output": None,
            }
        queue_pos = self._local_queue.qsize() + 1
        if queue_pos > 1:
            print(
                f"[SbatchOneshot] [{_ts()}] Job {job_id} queued locally (position {queue_pos}). "
                f"SLURM limit: {self._job_limit} active jobs."
            )
        self._local_queue.put((job_id, req))
        return job_id

    def status(self, job_id: str) -> JobStatus:
        if self._local_overlay is not None and self._local_overlay.owns(job_id):
            return self._local_overlay.status(job_id)
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        slurm_jid = job.get("slurm_job_id")
        eta = None
        if slurm_jid and job["status"] == "queued":
            eta = _squeue_eta(slurm_jid)
        self._refresh_progress(job_id)
        with self._jobs_lock:
            current = self._jobs.get(job_id)
            if current is not None:
                job = dict(current)

        return JobStatus(
            job_id=job_id,
            status=job["status"],
            eta_s=job.get("eta_s") if job.get("eta_s") is not None else eta,
            progress=job.get("progress"),
            text_output=job.get("text_output"),
            log_tail=log_tail_for_status(job.get("log", ""), job["status"]),
            error_message=job.get("error"),
        )

    def get_result(self, job_id: str) -> bytes:
        if self._local_overlay is not None and self._local_overlay.owns(job_id):
            return self._local_overlay.get_result(job_id)
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] != "done":
            raise ValueError(f"Job {job_id} is not done yet (status: {job['status']})")

        # Load on first access and cache
        if job.get("result") is None:
            if str(job["out_path"]).endswith(".overlay.npz"):
                from ..schema.overlay import OverlayResult
                result = OverlayResult.load(job["out_path"])
            else:
                from ..schema.result import XiaResult
                result = XiaResult.load(job["out_path"])
            packed = result.to_msgpack()
            with self._jobs_lock:
                self._jobs[job_id]["result"] = packed
        return job["result"]

    def _purge(self) -> None:
        """Run both retention windows: in-memory job records and on-disk results.

        The two are independent — memory is bounded by default while disk is
        opt-in — so on-disk dirs are cleaned by a filesystem scan rather than via
        the in-memory job records (which may already be gone).
        """
        self._purge_memory()
        self._purge_disk()

    def _purge_memory(self) -> None:
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
        if self._local_overlay:
            self._local_overlay.purge(self._memory_retention_s)

    def _purge_disk(self) -> None:
        """Remove on-disk result dirs older than the result-retention window.

        Scans ``{output_dir}/xia_results`` by completion time (newest result
        file, else dir mtime), skipping dirs of jobs still in flight so a
        long-running job is never deleted out from under itself.
        """
        if not self._result_retention_s:
            return
        base = os.path.join(self._cfg.output_dir, "xia_results")
        if not os.path.isdir(base):
            return
        cutoff = time.time() - self._result_retention_s
        # job_id == result-dir name; never touch dirs of non-terminal jobs.
        with self._jobs_lock:
            active = {
                jid for jid, job in self._jobs.items()
                if job.get("status") not in ("done", "error")
            }
        removed = 0
        try:
            entries = list(os.scandir(base))
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name in active:
                continue
            # A first-time fetch reads the result off disk; don't delete it
            # while that transfer is in flight.
            if transfer_guard.is_pinned(entry.name):
                continue
            try:
                mtime = entry.stat().st_mtime
                for fname in os.listdir(entry.path):
                    if fname.endswith(".npz"):
                        mtime = max(mtime, os.path.getmtime(os.path.join(entry.path, fname)))
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(entry.path)
                    removed += 1
                except OSError as exc:
                    print(f"[SbatchOneshot] Could not remove {entry.path}: {exc}")
        if removed:
            print(f"[SbatchOneshot] Purged {removed} on-disk result dir(s) past retention.")

    def health(self) -> HealthResponse:
        with self._slurm_lock:
            n_slurm = len(self._active_slurm)
        n_local = self._local_queue.qsize()
        n_overlay = self._local_overlay.active() if self._local_overlay else 0
        return HealthResponse(
            mode="sbatch-oneshot",
            model_warm=False,
            queue_depth=n_slurm + n_local + n_overlay,
            sbatch_config=self._cfg.to_dict(),
        )


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    prewarm: bool = False,
    memory_retention: Optional[str] = None,
    result_retention: Optional[str] = None,
    **sbatch_kwargs,
) -> None:
    """Start the sbatch-oneshot FastAPI server.  Blocks until stopped.

    ``memory_retention`` bounds completed jobs/result bytes in the listener's
    memory (default ``DEFAULT_MEMORY_RETENTION``).  ``result_retention`` bounds
    the on-disk result directories (default ``"never"``).
    """
    import uvicorn
    import socket

    from ..serving.config import DEFAULT_MEMORY_RETENTION, parse_retention

    if memory_retention is None:
        memory_retention = DEFAULT_MEMORY_RETENTION
    cfg = SbatchConfig(**{k: v for k, v in sbatch_kwargs.items() if v is not None}).resolve()
    backend = SbatchOneshotBackend(
        cfg,
        memory_retention_s=parse_retention(memory_retention),
        result_retention_s=parse_retention(result_retention),
    )
    set_backend(backend)

    from ..api.dispatch import _has_gpu, _has_sbatch
    from ..serving.cli_style import render_listener_banner

    hostname = socket.gethostname()
    render_listener_banner(
        mode="sbatch-oneshot",
        hostname=hostname,
        port=port,
        gpu=_has_gpu(),
        sbatch=_has_sbatch(),
        sbatch_config=cfg.to_dict(),
        job_limit=cfg.job_limit,
        bind_host=host,
        retention=backend.retention_info,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")
