"""HTTP client for geoxplain_aurora_adapter remote execution.

Mirrors the ``_run_local`` signature: takes a TargetSpec + method + input vars,
posts to the server, polls ``GET /jobs/{id}`` with exponential backoff, and
returns a ``XiaResult`` once the job is done.

Progress is surfaced using ``tqdm`` if available, falling back to plain prints.
"""

from __future__ import annotations

import time
from typing import Optional

from ..engine.progress import should_show_progress
from ..schema.result import XiaResult
from ..schema.spec import TargetSpec
from .protocol import BatchRunRequest, RunRequest
from .result_fetch import _fetch_result_bytes

WORKER_START_TIMEOUT_S = 600.0  # 10 minutes


def _status_label(job_status: str) -> str:
    """Human-friendly one-liner for a job status transition."""
    if job_status == "starting":
        return "starting — provisioning GPU worker on SLURM (can take a few minutes)"
    return job_status


def _progress_value_label(progress: Optional[float]) -> str:
    if progress is None:
        return "unknown"
    return f"{progress * 100:.0f}% ({progress:.3f})"


def _progress_timeout_error(
    *,
    kind: str,
    job_id: str,
    timeout_s: float,
    last_status: str,
    last_progress: Optional[float],
) -> TimeoutError:
    return TimeoutError(
        f"{kind} {job_id} made no progress for {timeout_s:.0f}s.\n"
        f"Last status: {last_status or 'unknown'}\n"
        f"Last progress: {_progress_value_label(last_progress)}\n"
        f"Check the listener logs for details."
    )


def run_remote(
    method: str,
    target: TargetSpec,
    input_vars: list[str],
    remote_url: str,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
    poll_max_s: float = 30.0,
    **options,
) -> XiaResult:
    """Submit a job to a remote geoxplain_aurora_adapter listener and return the result.

    Parameters
    ----------
    method:          XIA method name.
    target:          TargetSpec describing the scalar to explain.
    input_vars:      Input variable names.
    remote_url:      Base URL of the listener, e.g. ``"http://localhost:8765"``.
    timeout_s:       Maximum time to wait without progress activity (seconds).
    poll_interval_s: Starting polling interval (doubles on each no-op, capped by
                     ``poll_max_s``).
    **options:       Method-specific options forwarded to the server.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for remote execution.  "
            "From the adapter source directory, install with: "
            "python -m pip install -e '.[client]'"
        )

    base = remote_url.rstrip("/")
    req = RunRequest(
        method=method,
        target=target.to_dict(),
        input_vars=input_vars,
        options=options,
    )

    # Submit
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base}/run", json=req.to_dict())
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    print(f"[geoxplain_aurora_adapter client] Job {job_id} submitted to {base}")

    # Poll
    interval = poll_interval_s
    last_status = ""
    last_observed_text_output = ""
    last_printed_text_output = ""
    last_observed_progress: Optional[float] = None
    last_progress_activity_at = time.monotonic()
    last_shown_progress: Optional[float] = None
    last_shown_at = 0.0
    starting_since: Optional[float] = None

    try:
        from tqdm.auto import tqdm  # type: ignore[import]
        pbar = tqdm(total=100, unit="%", desc=f"{method}", leave=True)
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    try:
        with httpx.Client(timeout=30.0) as client:
            while True:
                now = time.monotonic()
                if now - last_progress_activity_at > timeout_s:
                    raise _progress_timeout_error(
                        kind="Remote job",
                        job_id=job_id,
                        timeout_s=timeout_s,
                        last_status=last_status,
                        last_progress=last_observed_progress,
                    )
                try:
                    status_resp = client.get(f"{base}/jobs/{job_id}")
                    status_resp.raise_for_status()
                    status = status_resp.json()
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    print(f"[geoxplain_aurora_adapter client] Poll error: {e}; retrying...")
                    time.sleep(interval)
                    continue

                job_status = status.get("status", "unknown")
                eta = status.get("eta_s")
                progress = status.get("progress")
                log_tail = status.get("log_tail", "")
                text_output = status.get("text_output")

                status_msg = job_status
                if eta is not None:
                    status_msg += f" (ETA {eta:.0f}s)"

                if job_status == "starting":
                    if starting_since is None:
                        starting_since = time.monotonic()
                    elif time.monotonic() - starting_since > WORKER_START_TIMEOUT_S:
                        raise TimeoutError(
                            f"GPU worker for job {job_id} did not start within "
                            f"{WORKER_START_TIMEOUT_S / 60:.0f} min — the SLURM job may be "
                            f"stuck in the queue. Check the listener logs."
                        )
                else:
                    starting_since = None

                if job_status != last_status:
                    label = _status_label(job_status)
                    if eta is not None:
                        label += f" (ETA {eta:.0f}s)"
                    print(f"[geoxplain_aurora_adapter client] {method} | {label}")
                    if log_tail and job_status not in ("done", "completing") and log_tail != last_status:
                        print(f"  {log_tail}")
                    last_status = job_status

                if job_status == "done":
                    result_url = status.get("result_url", f"{base}/jobs/{job_id}/result")
                    if not result_url.startswith(("http://", "https://")):
                        result_url = f"{base.rstrip('/')}/{result_url.lstrip('/')}"
                    content = _fetch_result_bytes(
                        client, result_url, method,
                        write=pbar.write if use_tqdm and pbar is not None else print,
                    )
                    result = XiaResult.from_msgpack(content)
                    print(f"[geoxplain_aurora_adapter client] Done — {result.summary()}")
                    return result

                now = time.monotonic()
                progress_changed = (
                    progress is not None and progress != last_observed_progress
                )
                text_output_changed = bool(
                    text_output and text_output != last_observed_text_output
                )
                progress_activity = progress_changed or text_output_changed
                if progress_activity:
                    last_progress_activity_at = now
                if progress_changed:
                    last_observed_progress = progress
                if text_output_changed:
                    last_observed_text_output = text_output

                printable_progress_changed = bool(
                    text_output and text_output != last_printed_text_output
                )
                # Throttle the printed lines (every +5%, or after 15s of quiet)
                # so a fast job doesn't flood the terminal; the tqdm bar below
                # still advances smoothly on every poll.
                if printable_progress_changed and should_show_progress(
                    progress, last_shown_progress, now - last_shown_at
                ):
                    if use_tqdm and pbar is not None:
                        pbar.write(text_output)
                    else:
                        print(text_output)
                    last_printed_text_output = text_output
                    last_shown_progress = progress
                    last_shown_at = now

                if use_tqdm and pbar is not None and progress is not None:
                    pbar.n = int(progress * 100)
                    pbar.set_postfix_str(status_msg)
                    pbar.refresh()

                if job_status == "error":
                    err_msg = status.get("error_message", "unknown error")
                    raise RuntimeError(
                        f"Remote job {job_id} failed: {err_msg}\n"
                        f"Log: {log_tail}"
                    )

                if progress_activity:
                    # Progress is actively streaming — poll tightly so the bar
                    # stays smooth instead of backing off and missing frames.
                    interval = min(poll_interval_s, 0.5)
                elif job_status == "running":
                    # Compute can finish quickly (e.g. saliency is a single
                    # forward/backward); keep polls frequent enough to catch it.
                    interval = min(interval * 1.5, 1.0)
                elif job_status == "completing":
                    # `completing` is a transition state: the SLURM job has
                    # stopped running and the controller is just flushing
                    # output.  Done is imminent — keep the poll interval tight.
                    interval = min(interval, 1.0)
                else:
                    interval = min(interval * 1.5, poll_max_s)
                time.sleep(interval)

    finally:
        if use_tqdm and pbar is not None:
            pbar.close()


def run_remote_batch(
    method: str,
    targets: list[TargetSpec],
    input_vars: list[str],
    remote_url: str,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
    poll_max_s: float = 30.0,
    **options,
) -> XiaResult:
    """Submit a multi-timeframe job to a remote listener and return the result."""
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for remote execution.  "
            "From the adapter source directory, install with: "
            "python -m pip install -e '.[client]'"
        )

    base = remote_url.rstrip("/")
    req = BatchRunRequest(
        method=method,
        targets=[target.to_dict() for target in targets],
        input_vars=input_vars,
        options=options,
    )

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base}/run_batch", json=req.to_dict())
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    print(
        f"[geoxplain_aurora_adapter client] Batch job {job_id} "
        f"({len(targets)} frames) submitted to {base}"
    )

    interval = poll_interval_s
    last_status = ""
    last_observed_text_output = ""
    last_printed_text_output = ""
    last_observed_progress: Optional[float] = None
    last_progress_activity_at = time.monotonic()
    last_shown_progress: Optional[float] = None
    last_shown_at = 0.0
    starting_since: Optional[float] = None

    try:
        from tqdm.auto import tqdm  # type: ignore[import]
        pbar = tqdm(total=100, unit="%", desc=f"{method} batch", leave=True)
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    try:
        with httpx.Client(timeout=30.0) as client:
            while True:
                now = time.monotonic()
                if now - last_progress_activity_at > timeout_s:
                    raise _progress_timeout_error(
                        kind="Remote batch job",
                        job_id=job_id,
                        timeout_s=timeout_s,
                        last_status=last_status,
                        last_progress=last_observed_progress,
                    )
                try:
                    status_resp = client.get(f"{base}/jobs/{job_id}")
                    status_resp.raise_for_status()
                    status = status_resp.json()
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    print(f"[geoxplain_aurora_adapter client] Poll error: {e}; retrying...")
                    time.sleep(interval)
                    continue

                job_status = status.get("status", "unknown")
                eta = status.get("eta_s")
                progress = status.get("progress")
                text_output = status.get("text_output")
                log_tail = status.get("log_tail", "")

                status_msg = job_status
                if eta is not None:
                    status_msg += f" (ETA {eta:.0f}s)"

                # "starting" (persistent mode bringing up the GPU worker) is a
                # normal waiting state — wait, but not forever.
                if job_status == "starting":
                    if starting_since is None:
                        starting_since = time.monotonic()
                    elif time.monotonic() - starting_since > WORKER_START_TIMEOUT_S:
                        raise TimeoutError(
                            f"GPU worker for batch job {job_id} did not start within "
                            f"{WORKER_START_TIMEOUT_S / 60:.0f} min — the SLURM job may be "
                            f"stuck in the queue. Check the listener logs."
                        )
                else:
                    starting_since = None

                if job_status != last_status:
                    label = _status_label(job_status)
                    if eta is not None:
                        label += f" (ETA {eta:.0f}s)"
                    print(f"[geoxplain_aurora_adapter client] {method} batch | {label}")
                    if log_tail and job_status not in ("done", "completing") and log_tail != last_status:
                        print(f"  {log_tail}")
                    last_status = job_status

                if job_status == "done":
                    result_url = status.get("result_url", f"{base}/jobs/{job_id}/result")
                    if not result_url.startswith(("http://", "https://")):
                        result_url = f"{base.rstrip('/')}/{result_url.lstrip('/')}"
                    content = _fetch_result_bytes(
                        client, result_url, f"{method} batch",
                        write=pbar.write if use_tqdm and pbar is not None else print,
                    )
                    result = XiaResult.from_msgpack(content)
                    print(f"[geoxplain_aurora_adapter client] Done - {result.summary()}")
                    return result

                now = time.monotonic()
                progress_changed = (
                    progress is not None and progress != last_observed_progress
                )
                text_output_changed = bool(
                    text_output and text_output != last_observed_text_output
                )
                progress_activity = progress_changed or text_output_changed
                if progress_activity:
                    last_progress_activity_at = now
                if progress_changed:
                    last_observed_progress = progress
                if text_output_changed:
                    last_observed_text_output = text_output

                printable_progress_changed = bool(
                    text_output and text_output != last_printed_text_output
                )
                # Throttle the printed lines (every +5%, or after 15s of quiet)
                # so a fast job doesn't flood the terminal; the tqdm bar below
                # still advances smoothly on every poll.
                if printable_progress_changed and should_show_progress(
                    progress, last_shown_progress, now - last_shown_at
                ):
                    if use_tqdm and pbar is not None:
                        pbar.write(text_output)
                    else:
                        print(text_output)
                    last_printed_text_output = text_output
                    last_shown_progress = progress
                    last_shown_at = now

                if use_tqdm and pbar is not None and progress is not None:
                    pbar.n = int(progress * 100)
                    pbar.set_postfix_str(status_msg)
                    pbar.refresh()

                if job_status == "error":
                    err_msg = status.get("error_message", "unknown error")
                    raise RuntimeError(
                        f"Remote job {job_id} failed: {err_msg}\n"
                        f"Log: {log_tail}"
                    )

                if progress_activity:
                    # Progress is actively streaming — poll tightly so the bar
                    # stays smooth instead of backing off and missing frames.
                    interval = min(poll_interval_s, 0.5)
                elif job_status == "running":
                    interval = min(interval * 1.5, 1.0)
                elif job_status == "completing":
                    interval = min(interval, 1.0)
                else:
                    interval = min(interval * 1.5, min(poll_max_s, 5.0))
                time.sleep(interval)

    finally:
        if use_tqdm and pbar is not None:
            pbar.close()
