"""sbatch-persistent backend: persistent GPU worker (opt-in via ``--persistent``).

Behavior:
    1. On first request (or immediately with ``prewarm=True``), the login-node
       listener submits **one** GPU worker job using the resolved sbatch config.
       The worker runs ``geoxplain_aurora_adapter.remote.inproc.run_server`` on the GPU node
       and writes ``{scratch}/geoxplain_aurora_adapter_worker_{id}.json`` containing
       ``{"host": str, "port": int, "job_id": str}``.
    2. The login-node listener reads that file, then forwards all subsequent
       requests to the worker via HTTP.
    3. Health monitor: the listener polls the worker's ``GET /health`` endpoint
       every 30 s (the worker can't push heartbeats — it never learns the
       listener's address).  If the worker is unreachable for 5 min, the
       listener replaces it with a fresh job.
    4. Wall-time replacement: a worker job runs under a SLURM wall-time limit.
       Shortly before it expires, the listener launches a replacement worker,
       waits for it to come up, and switches over — so the persistent worker
       survives across wall-time boundaries with minimal downtime.  New requests
       go to the replacement immediately; the old worker is *drained* first
       (its in-flight jobs are allowed to finish and their results pulled back
       to the listener) before it is cancelled.

The GPU worker port is picked at random (50000–59999) to avoid conflicts on
shared login nodes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from typing import Optional

from .local_overlay import LocalOverlayRunner
from .progress_mirror import start_progress_mirror
from .protocol import HealthResponse, JobStatus, RunRequest, new_job_id
from .retention import DEFAULT_JANITOR_INTERVAL_S, start_janitor
from .sbatch_config import SbatchConfig, ResolvedSbatchConfig
from .server import app, set_backend
from .server_log import report_job_failure
from .slurm import _parse_walltime_s, _pick_worker_port, _scancel
from . import transfer_guard


_WORKER_JSON_TEMPLATE = "geoxplain_aurora_adapter_worker_{id}.json"
_MONITOR_INTERVAL_S = 30      # how often the listener polls the worker /health
_HEARTBEAT_TIMEOUT_S = 300    # declare the worker dead after this long unreachable
_OVERLAP_S = 120              # launch the replacement this long before wall-time
_HEALTH_TIMEOUT_S = 10        # per-poll HTTP timeout
_DRAIN_POLL_INTERVAL_S = 3    # how often to re-poll draining jobs on the old worker
_DRAIN_MARGIN_S = 20          # stop draining this long before the old node's wall-time
_RESULT_TIMEOUT_S = 120       # HTTP timeout when pulling result bytes back
_PROGRESS_POLL_INTERVAL_S = 2 # how often the listener mirrors worker progress to its CLI


class SbatchPersistentBackend:
    """Login-node backend that forwards to a long-lived GPU worker job."""

    def __init__(
        self,
        resolved_cfg: ResolvedSbatchConfig,
        worker_port: Optional[int] = None,
        memory_retention_s: Optional[float] = None,
    ) -> None:
        self._cfg = resolved_cfg
        self._memory_retention_s = memory_retention_s
        self._worker_port = worker_port or _pick_worker_port()
        os.makedirs(resolved_cfg.output_dir, exist_ok=True)
        self._wall_time_s = _parse_walltime_s(resolved_cfg.time)

        # Overlay pulls don't need the GPU worker; compute them in this
        # login-node process instead of forwarding to it (unless disabled).
        self._local_overlay = (
            LocalOverlayRunner() if resolved_cfg.overlay_on_login else None
        )

        # Live worker state (guarded by ``_worker_lock``).
        self._worker_url: Optional[str] = None
        self._worker_slurm_id: Optional[str] = None
        self._worker_json: Optional[str] = None   # the live worker's address file
        self._worker_submit_time: float = 0.0     # for wall-time replacement
        self._worker_lock = threading.Lock()
        self._last_heartbeat: float = 0.0
        self._replacing = False                   # guards against concurrent replaces

        # Jobs forwarded to (or queued for) the worker
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()

        # Background monitor: polls worker /health and handles wall-time rotation.
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="geoxplain-aurora-adapter-monitor"
        )
        self._monitor_thread.start()
        # Progress poller: mirrors each active job's worker-side progress to the
        # listener's own terminal (the worker's bar otherwise only shows in the
        # GPU job log, not here on the login node).
        self._progress_poller = start_progress_mirror(
            self._jobs, self._jobs_lock, self._fetch_worker_status,
            interval_s=_PROGRESS_POLL_INTERVAL_S,
        )
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

    # ── Worker lifecycle ──────────────────────────────────────────────────

    def _launch_worker(self) -> tuple[str, str, str]:
        """Submit a GPU worker job and wait for its worker.json.

        Each launch uses a *fresh* worker.json path so a replacement worker
        never reads a stale (previous worker's) address.  Returns
        ``(url, slurm_id, worker_json_path)`` without mutating live state — the
        caller decides when to switch over.
        """
        worker_id = new_job_id()[:8]
        worker_json = os.path.join(
            self._cfg.output_dir, _WORKER_JSON_TEMPLATE.format(id=worker_id)
        )
        port = self._worker_port
        body = (
            f"python -m geoxplain_aurora_adapter.remote.worker_persistent "
            f"--port {port} "
            f"--worker-json {worker_json}"
        )
        script = self._cfg.render_submit_script(body)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="geoxplain_aurora_adapter_worker_"
        ) as f:
            f.write(script)
            script_path = f.name
        os.chmod(script_path, 0o755)

        submit = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True, text=True,
        )
        if submit.returncode != 0:
            # A bare CalledProcessError only reports the exit status; surface
            # sbatch's stderr so the actual rejection reason is visible.
            detail = (submit.stderr or submit.stdout or "").strip()
            raise RuntimeError(
                f"sbatch rejected the worker submit script (exit "
                f"{submit.returncode}): {detail or 'no output'}"
            )
        slurm_id = submit.stdout.strip().split(";")[0]
        print(f"[SbatchPersistent] Worker job submitted: SLURM {slurm_id}")
        print(f"[SbatchPersistent] Waiting for worker.json at {worker_json} ...")

        import httpx

        deadline = time.time() + 600
        url: Optional[str] = None
        while time.time() < deadline:
            if os.path.exists(worker_json):
                with open(worker_json) as f:
                    info = json.load(f)
                url = f"http://{info['host']}:{info['port']}"
                break
            time.sleep(5)

        # The worker writes worker.json *before* its HTTP server starts
        # accepting connections: model prewarm and the uvicorn bind both happen
        # after the file lands.  File existence therefore means "address known",
        # not "server reachable".  Gate readiness on a successful /health probe
        # so the first forwarded job doesn't race startup and hit
        # "[Errno 111] Connection refused".
        while url is not None and time.time() < deadline:
            try:
                with httpx.Client(timeout=_HEALTH_TIMEOUT_S) as client:
                    client.get(f"{url}/health").raise_for_status()
            except Exception:
                time.sleep(2)
                continue
            print(f"[SbatchPersistent] Worker ready at {url}")
            return url, slurm_id, worker_json

        _scancel(slurm_id)
        try:
            os.unlink(worker_json)
        except OSError:
            pass
        reason = (
            "did not write worker.json"
            if url is None
            else "wrote worker.json but its /health endpoint never became reachable"
        )
        raise RuntimeError(
            f"GPU worker job {slurm_id} {reason} within 600s. "
            f"Check logs in {self._cfg.log_dir}"
        )

    def _adopt_worker(self, url: str, slurm_id: str, worker_json: str) -> Optional[str]:
        """Switch live state to a freshly launched worker; return old slurm id."""
        with self._worker_lock:
            old_slurm_id = self._worker_slurm_id
            old_json = self._worker_json
            self._worker_url = url
            self._worker_slurm_id = slurm_id
            self._worker_json = worker_json
            self._worker_submit_time = time.time()
            self._last_heartbeat = time.time()
        if old_json and old_json != worker_json:
            try:
                os.unlink(old_json)
            except OSError:
                pass
        return old_slurm_id

    def _ensure_worker(self) -> str:
        """Return the worker URL, submitting a new job if none is live."""
        with self._worker_lock:
            url = self._worker_url
        if url:
            return url
        # Serialize first-launch so concurrent requests don't submit duplicates.
        with self._worker_lock:
            busy = self._replacing
            if not busy:
                self._replacing = True
        if busy:
            # Another thread is bringing a worker up; wait for it.
            for _ in range(120):
                time.sleep(5)
                with self._worker_lock:
                    if self._worker_url:
                        return self._worker_url
            raise RuntimeError("Timed out waiting for the GPU worker to come up.")
        try:
            url, slurm_id, worker_json = self._launch_worker()
            self._adopt_worker(url, slurm_id, worker_json)
            return url
        finally:
            with self._worker_lock:
                self._replacing = False

    def _replace_worker(self, reason: str, *, drain: bool) -> None:
        """Launch a replacement worker, switch over, optionally drain, then cancel.

        ``drain`` waits for jobs already dispatched to the *old* worker to finish
        (pulling their results back to the listener so they survive the old
        worker's shutdown) before cancelling it.  New requests go to the new
        worker as soon as it is adopted.  Draining only makes sense for a still
        live worker (wall-time rotation), not one we are replacing because it
        already went unreachable.
        """
        with self._worker_lock:
            if self._replacing:
                return  # a replacement is already in flight
            self._replacing = True
            old_url = self._worker_url
            old_submit_time = self._worker_submit_time
        print(f"[SbatchPersistent] Replacing GPU worker ({reason}).")
        try:
            url, slurm_id, worker_json = self._launch_worker()
            old_slurm_id = self._adopt_worker(url, slurm_id, worker_json)
            # The new worker is now serving; drain the old one if asked.
            if drain and old_url and self._wall_time_s is not None:
                deadline = (old_submit_time + self._wall_time_s) - _DRAIN_MARGIN_S
                if time.time() < deadline:
                    try:
                        self._drain_worker(old_url, deadline)
                    except Exception as e:
                        print(f"[SbatchPersistent] Drain error (continuing): {e}")
            if old_slurm_id and old_slurm_id != slurm_id:
                _scancel(old_slurm_id)
            print(f"[SbatchPersistent] Worker replacement complete ({reason}).")
        except Exception as e:
            print(f"[SbatchPersistent] Worker replacement failed: {e}")
        finally:
            with self._worker_lock:
                self._replacing = False

    def _pending_on(self, worker_url: str) -> list[tuple[str, str]]:
        """``(job_id, worker_job_id)`` for non-terminal jobs sent to ``worker_url``."""
        with self._jobs_lock:
            return [
                (jid, job["worker_job_id"])
                for jid, job in self._jobs.items()
                if job.get("worker_url") == worker_url
                and job.get("worker_job_id")
                and job.get("status") not in ("done", "error")
            ]

    def _drain_worker(self, old_url: str, deadline: float) -> None:
        """Wait for the old worker's in-flight jobs to finish, caching results.

        Polls each non-terminal job dispatched to ``old_url``; when one completes
        its result bytes are pulled back and cached on the listener (so a later
        client fetch is served from cache, not the dead worker).  Jobs still
        unfinished at ``deadline`` are failed with a clear message rather than
        left polling a worker that is about to disappear.
        """
        try:
            import httpx
        except ImportError:
            return

        pending = self._pending_on(old_url)
        if not pending:
            return
        print(f"[SbatchPersistent] Draining {len(pending)} in-flight job(s) from the old worker...")

        with httpx.Client(timeout=_HEALTH_TIMEOUT_S) as client:
            while time.time() < deadline:
                pending = self._pending_on(old_url)
                if not pending:
                    break
                for job_id, worker_job_id in pending:
                    try:
                        resp = client.get(f"{old_url}/jobs/{worker_job_id}")
                        resp.raise_for_status()
                        s = resp.json()
                    except Exception:
                        continue  # transient; retry on the next pass
                    if s.get("status") == "done":
                        self._pull_drained_result(job_id, old_url, worker_job_id)
                    elif s.get("status") == "error":
                        self._finalize_drained(job_id, "error", s.get("error_message") or "worker error")
                if self._pending_on(old_url):
                    time.sleep(_DRAIN_POLL_INTERVAL_S)

        leftover = self._pending_on(old_url)
        for job_id, _ in leftover:
            self._finalize_drained(
                job_id, "error",
                "GPU worker was rotated out at its SLURM wall-time before this job "
                "finished; please resubmit.",
            )
        if leftover:
            print(f"[SbatchPersistent] Drain deadline reached; failed {len(leftover)} unfinished job(s).")

    def _pull_drained_result(self, job_id: str, old_url: str, worker_job_id: str) -> None:
        """Pull a completed job's result bytes off the old worker into the cache."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.get("result") is not None:
                job["status"] = "done"
                job.setdefault("done_at", time.time())
                return
        try:
            import httpx
            with httpx.Client(timeout=_RESULT_TIMEOUT_S) as client:
                resp = client.get(f"{old_url}/jobs/{worker_job_id}/result")
                resp.raise_for_status()
                packed = resp.content
        except Exception as e:
            print(f"[SbatchPersistent] Could not pull drained result for {job_id}: {e}")
            return
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["result"] = packed
                job["status"] = "done"
                job.setdefault("done_at", time.time())

    def _finalize_drained(self, job_id: str, status: str, error: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            old_status = job.get("status")
            job["status"] = status
            if status == "error":
                job.setdefault("error", error)
            job.setdefault("done_at", time.time())
        if status == "error" and old_status != "error":
            report_job_failure(job_id, error, source="SbatchPersistent")

    # ── Health monitor ────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        try:
            import httpx
        except ImportError:
            print("[SbatchPersistent] httpx not available; worker monitoring disabled.")
            return
        while True:
            time.sleep(_MONITOR_INTERVAL_S)
            with self._worker_lock:
                url = self._worker_url
                submit_time = self._worker_submit_time
                replacing = self._replacing
            if url is None or replacing:
                continue

            # Wall-time replacement: rotate before SLURM kills the job.
            if (
                self._wall_time_s is not None
                and submit_time
                and (time.time() - submit_time) >= (self._wall_time_s - _OVERLAP_S)
            ):
                self._replace_worker("approaching wall-time limit", drain=True)
                continue

            # Liveness: poll the worker's own /health endpoint.
            try:
                with httpx.Client(timeout=_HEALTH_TIMEOUT_S) as client:
                    resp = client.get(f"{url}/health")
                    resp.raise_for_status()
                with self._worker_lock:
                    self._last_heartbeat = time.time()
            except Exception:
                with self._worker_lock:
                    age = time.time() - self._last_heartbeat
                if age > _HEARTBEAT_TIMEOUT_S:
                    self._replace_worker(f"unreachable for {age:.0f}s", drain=False)

    # ── Progress mirroring ────────────────────────────────────────────────

    def _fetch_worker_status(self, job_id: str, job: dict) -> Optional[dict]:
        """Snapshot source for the progress mirror: poll the worker over HTTP.

        The GPU worker renders its progress bar to the SLURM job log, invisible
        on the login node; the mirror polls each in-flight job here and reprints
        it.  Returns ``None`` until the job has been forwarded to a worker.
        """
        worker_url = job.get("worker_url")
        worker_job_id = job.get("worker_job_id")
        if not worker_url or not worker_job_id:
            return None
        import httpx
        with httpx.Client(timeout=_HEALTH_TIMEOUT_S) as client:
            resp = client.get(f"{worker_url}/jobs/{worker_job_id}")
            resp.raise_for_status()
            return resp.json()

    # ── Backend protocol ──────────────────────────────────────────────────

    def submit(self, req: RunRequest) -> str:
        """Accept a request and return immediately with a job id.

        Bringing up the GPU worker can take minutes (the SLURM job has to be
        scheduled), so we must **not** block the HTTP request on it — the
        client's submit POST would time out.  Instead the job is recorded as
        ``"starting"`` and a background thread provisions the worker and
        forwards the request; the client polls ``GET /jobs/{id}`` and waits.
        """
        if self._local_overlay is not None and hasattr(req, "timestamps"):
            return self._local_overlay.submit(req)
        try:
            import httpx  # noqa: F401  (fail fast if the server extra is missing)
        except ImportError:
            raise RuntimeError(
                "httpx is required for the persistent backend.  "
                "From the adapter source directory, run "
                "python -m pip install -e '.[server]'"
            )
        job_id = new_job_id()
        with self._jobs_lock:
            self._jobs[job_id] = {"status": "starting", "worker_job_id": None, "result": None}
        threading.Thread(
            target=self._dispatch_to_worker,
            args=(job_id, req),
            daemon=True,
            name=f"geoxplain-aurora-adapter-dispatch-{job_id[:8]}",
        ).start()
        return job_id

    def _dispatch_to_worker(self, job_id: str, req: RunRequest) -> None:
        """Provision the worker (may block on the SLURM queue) and forward."""
        import httpx

        try:
            url = self._ensure_worker()
        except Exception as exc:
            self._fail_job(
                job_id,
                f"GPU worker did not start: {exc}",
            )
            return

        if hasattr(req, "timestamps"):
            path = "overlay"
        elif hasattr(req, "targets"):
            path = "run_batch"
        else:
            path = "run"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{url}/{path}", json=req.to_dict())
                resp.raise_for_status()
                worker_job_id = resp.json()["job_id"]
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["worker_job_id"] = worker_job_id
                    job["worker_url"] = url
                    # Once forwarded, the worker owns the status; "starting" gives
                    # way to "queued" until the worker reports running/done.
                    if job.get("status") == "starting":
                        job["status"] = "queued"
        except Exception as exc:
            self._fail_job(job_id, str(exc))

    def _fail_job(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            old_status = job.get("status")
            job["status"] = "error"
            job["error"] = error
            job["done_at"] = time.time()
        if old_status != "error":
            report_job_failure(job_id, error, source="SbatchPersistent")

    def status(self, job_id: str) -> JobStatus:
        if self._local_overlay is not None and self._local_overlay.owns(job_id):
            return self._local_overlay.status(job_id)
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required")

        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.get("status") == "error":
            return JobStatus(job_id=job_id, status="error", error_message=job.get("error"))

        # A terminal "done" is authoritative (e.g. set by draining, which also
        # cached the result bytes here) — don't re-poll, the worker that ran it
        # may already be gone.
        if job.get("status") == "done":
            return JobStatus(job_id=job_id, status="done", progress=1.0, eta_s=0.0)

        worker_url = job.get("worker_url")
        worker_job_id = job.get("worker_job_id")
        if not worker_url or not worker_job_id:
            # Not yet forwarded: the GPU worker is still being provisioned
            # ("starting") or just queued.  Report the stored state so the
            # client knows to keep waiting instead of treating it as an error.
            return JobStatus(job_id=job_id, status=job.get("status", "queued"))

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{worker_url}/jobs/{worker_job_id}")
                resp.raise_for_status()
                s = resp.json()
        except Exception as exc:
            return JobStatus(job_id=job_id, status="running", log_tail=str(exc)[:200])

        report_error = False
        if s["status"] in ("done", "error"):
            with self._jobs_lock:
                job = self._jobs[job_id]
                old_status = job.get("status")
                job["status"] = s["status"]
                job["worker_url"] = worker_url
                job["worker_job_id"] = worker_job_id
                job.setdefault("done_at", time.time())
                if s["status"] == "error":
                    job["error"] = s.get("error_message") or "worker error"
                    job["log"] = s.get("log_tail", "")
                    report_error = old_status != "error"

        if report_error:
            report_job_failure(
                job_id,
                s.get("error_message"),
                log_tail=s.get("log_tail"),
                source="SbatchPersistent",
            )

        return JobStatus(
            job_id=job_id,
            status=s["status"],
            eta_s=s.get("eta_s"),
            progress=s.get("progress"),
            text_output=s.get("text_output"),
            log_tail=s.get("log_tail", "")[-200:],
            error_message=s.get("error_message"),
        )

    def get_result(self, job_id: str) -> bytes:
        if self._local_overlay is not None and self._local_overlay.owns(job_id):
            return self._local_overlay.get_result(job_id)
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required")

        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("result"):
            return job["result"]

        worker_url = job.get("worker_url")
        worker_job_id = job.get("worker_job_id")
        if not worker_url or not worker_job_id:
            raise ValueError(f"Job {job_id} has no worker reference")

        with httpx.Client(timeout=120.0) as client:
            resp = client.get(f"{worker_url}/jobs/{worker_job_id}/result")
            resp.raise_for_status()
            packed = resp.content

        with self._jobs_lock:
            self._jobs[job_id]["result"] = packed
        return packed

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
        if self._local_overlay:
            self._local_overlay.purge(self._memory_retention_s)

    def health(self) -> HealthResponse:
        with self._worker_lock:
            url = self._worker_url
        model_warm = url is not None
        n_overlay = self._local_overlay.active() if self._local_overlay else 0
        return HealthResponse(
            mode="sbatch-persistent",
            model_warm=model_warm,
            queue_depth=n_overlay,
            sbatch_config=self._cfg.to_dict(),
        )

    def shutdown(self) -> None:
        """Cancel the GPU worker job — called when the listener stops (Ctrl-C).

        A persistent worker outlives individual requests, so without this it
        would keep burning the SLURM allocation after the listener is gone.
        Idempotent and best-effort: clears live state, then ``scancel``s the
        tracked job id(s).
        """
        with self._worker_lock:
            slurm_id = self._worker_slurm_id
            self._worker_url = None
            self._worker_slurm_id = None
        if slurm_id:
            print(f"[SbatchPersistent] Cancelling GPU worker job {slurm_id} ...")
            _scancel(slurm_id)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    prewarm: bool = False,
    worker_port: Optional[int] = None,
    memory_retention: Optional[str] = None,
    **sbatch_kwargs,
) -> None:
    """Start the persistent-worker FastAPI listener.  Blocks until stopped.

    ``memory_retention`` bounds how long completed jobs and cached result bytes
    live in the listener's memory (default ``DEFAULT_MEMORY_RETENTION`` when not
    given; pass ``"never"`` to disable).  This backend keeps nothing on disk, so
    there is no separate result-retention knob.
    """
    import uvicorn

    from ..serving.config import DEFAULT_MEMORY_RETENTION, parse_retention

    if memory_retention is None:
        memory_retention = DEFAULT_MEMORY_RETENTION
    cfg = SbatchConfig(**{k: v for k, v in sbatch_kwargs.items() if v is not None}).resolve()
    backend = SbatchPersistentBackend(
        cfg, worker_port=worker_port, memory_retention_s=parse_retention(memory_retention)
    )
    set_backend(backend)

    if prewarm:
        print("[SbatchPersistent] Pre-warming: submitting GPU worker job...")
        threading.Thread(target=backend._ensure_worker, daemon=True).start()

    from ..api.dispatch import _has_gpu, _has_sbatch
    from ..serving.cli_style import render_listener_banner

    hostname = socket.gethostname()
    render_listener_banner(
        mode="sbatch-persistent",
        hostname=hostname,
        port=port,
        gpu=_has_gpu(),
        sbatch=_has_sbatch(),
        sbatch_config=cfg.to_dict(),
        worker_port=backend._worker_port,
        bind_host=host,
        retention=backend.retention_info,
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        backend.shutdown()
