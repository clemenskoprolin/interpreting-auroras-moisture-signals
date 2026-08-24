"""HTTP client for remote overlay pulls.

Submits an overlay request to a listener, polls ``GET /jobs/{id}`` until the
overlay is computed, and returns the resulting ``OverlayResult``.
"""

from __future__ import annotations

import time
from typing import Optional

from ..schema.overlay import OverlayResult
from .protocol import OverlayRequest
from .result_fetch import _fetch_result_bytes


def pull_remote_overlay(
    variable: str,
    timestamps: list[str],
    remote_url: str,
    *,
    level: Optional[int] = None,
    name: Optional[str] = None,
    unit: Optional[str] = None,
    colormap: str = "viridis",
    visible: bool = True,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 2.0,
    poll_max_s: float = 30.0,
) -> OverlayResult:
    """Submit an overlay pull to a remote listener and return the result."""

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for remote execution.  "
            "From the adapter source directory, install with: "
            "python -m pip install -e '.[client]'"
        )

    base = remote_url.rstrip("/")
    req = OverlayRequest(
        variable=variable,
        timestamps=timestamps,
        level=level,
        options={
            "name": name,
            "unit": unit,
            "colormap": colormap,
            "visible": visible,
        },
    )

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base}/overlay", json=req.to_dict())
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    print(
        f"[geoxplain_aurora_adapter client] Overlay job {job_id} "
        f"({variable}, {len(timestamps)} frames) submitted to {base}"
    )

    deadline = time.time() + timeout_s
    interval = poll_interval_s
    last_status = ""

    try:
        from tqdm.auto import tqdm  # type: ignore[import]
        pbar = tqdm(total=100, unit="%", desc=f"{variable} overlay", leave=True)
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    try:
        with httpx.Client(timeout=30.0) as client:
            while time.time() < deadline:
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

                status_msg = job_status
                if eta is not None:
                    status_msg += f" (ETA {eta:.0f}s)"

                if job_status != last_status:
                    print(f"[geoxplain_aurora_adapter client] {variable} overlay | {status_msg}")
                    if log_tail and log_tail != last_status:
                        print(f"  {log_tail}")
                    last_status = job_status

                if use_tqdm and pbar is not None and progress is not None:
                    pbar.n = int(progress * 100)
                    pbar.set_postfix_str(status_msg)
                    pbar.refresh()

                if job_status == "done":
                    result_url = status.get("result_url", f"{base}/jobs/{job_id}/result")
                    if not result_url.startswith(("http://", "https://")):
                        result_url = f"{base.rstrip('/')}/{result_url.lstrip('/')}"
                    content = _fetch_result_bytes(
                        client, result_url, f"{variable} overlay",
                        write=pbar.write if use_tqdm and pbar is not None else print,
                    )
                    result = OverlayResult.from_msgpack(content)
                    print(f"[geoxplain_aurora_adapter client] Done - {result.summary()}")
                    return result

                if job_status == "error":
                    err_msg = status.get("error_message", "unknown error")
                    raise RuntimeError(
                        f"Remote overlay job {job_id} failed: {err_msg}\n"
                        f"Log: {log_tail}"
                    )

                time.sleep(interval)
                if job_status == "completing":
                    interval = min(interval, 1.0)
                else:
                    interval = min(interval * 1.5, poll_max_s)

    finally:
        if use_tqdm and pbar is not None:
            pbar.close()

    raise TimeoutError(
        f"Remote overlay job {job_id} did not complete within {timeout_s:.0f}s.\n"
        f"Last status: {last_status}\n"
        f"Check the listener logs for details."
    )
