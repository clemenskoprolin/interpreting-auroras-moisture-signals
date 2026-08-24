"""FastAPI application for geoxplain_aurora_adapter remote execution.

Endpoints:
    POST /run                 → submit a job, returns {job_id}
    POST /run_batch           → submit a multi-timeframe job
    POST /overlay             → submit a raw ERA5 overlay pull
    GET  /jobs/{job_id}       → poll status (JSON)
    GET  /jobs/{job_id}/result→ fetch msgpack-packed XiaResult (when done)
    GET  /health              → backend info

The ``app`` object is imported by backend modules (``inproc``,
``sbatch_oneshot``, ``sbatch_persistent``) which attach a ``Backend``
implementation and then call ``uvicorn.run(app, ...)``.

Backend protocol
----------------
A backend must implement:
    backend.submit(req: RunRequest) → str (job_id)
    backend.status(job_id: str) → JobStatus
    backend.get_result(job_id: str) → bytes (msgpack)
    backend.health() → HealthResponse
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.background import BackgroundTask

from . import transfer_guard
from .protocol import BatchRunRequest, HealthResponse, JobStatus, OverlayRequest, RunRequest
from .server_log import report_job_failure

app = FastAPI(title="geoxplain_aurora_adapter listener")

# Replaced at startup by the specific backend module
_backend: Any = None


def set_backend(backend) -> None:
    global _backend
    _backend = backend


def _require_backend():
    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    return _backend


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/run", status_code=202)
async def submit_job(body: dict) -> dict:
    """Submit an XIA computation request.

    Body: ``RunRequest`` dict (see ``geoxplain_aurora_adapter.remote.protocol``).
    Returns: ``{"job_id": "<uuid>"}``
    """
    try:
        req = RunRequest.from_dict(body)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    backend = _require_backend()
    try:
        job_id = backend.submit(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id}


@app.post("/run_batch", status_code=202)
async def submit_batch_job(body: dict) -> dict:
    """Submit a multi-timeframe XIA computation request."""
    try:
        req = BatchRunRequest.from_dict(body)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    backend = _require_backend()
    try:
        job_id = backend.submit(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id}


@app.post("/overlay", status_code=202)
async def submit_overlay_job(body: dict) -> dict:
    """Submit a raw ERA5 overlay pull request."""
    try:
        req = OverlayRequest.from_dict(body)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    backend = _require_backend()
    try:
        job_id = backend.submit(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request) -> JSONResponse:
    """Poll job status.

    Returns a ``JobStatus`` dict.  When ``status == "done"``, the body also
    includes an *absolute* ``result_url`` (scheme + host + path) so the
    client can fetch it without having to know the listener's base URL.
    """
    backend = _require_backend()
    try:
        status: JobStatus = backend.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    payload = status.to_dict()
    if status.status == "error":
        report_job_failure(
            job_id,
            status.error_message,
            log_tail=status.log_tail,
        )
    if status.status == "done":
        # str(base_url) already ends with '/'.
        base = str(request.base_url).rstrip("/")
        payload.setdefault("result_url", f"{base}/jobs/{job_id}/result")

    return JSONResponse(content=payload)


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> Response:
    """Fetch the msgpack-packed ``XiaResult`` for a completed job."""
    backend = _require_backend()
    # Pin before reading and keep it pinned until the response has finished
    # sending (the body streams after this handler returns), so the retention
    # janitor can't delete the job's record or on-disk result mid-transfer.
    transfer_guard.pin(job_id)
    delivered = False
    try:
        data: bytes = backend.get_result(job_id)
        response = Response(
            content=data,
            media_type="application/octet-stream",
            background=BackgroundTask(transfer_guard.unpin, job_id),
        )
        delivered = True
        return response
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # On any failure the response (and its post-send unpin) never runs, so
        # release the pin here; on success the BackgroundTask owns the unpin.
        if not delivered:
            transfer_guard.unpin(job_id)


@app.get("/health")
async def health() -> JSONResponse:
    """Return backend health information."""
    backend = _require_backend()
    try:
        h: HealthResponse = backend.health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=h.to_dict())
