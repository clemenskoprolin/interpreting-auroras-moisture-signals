"""Server-console reporting for failed remote jobs."""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from typing import Optional


_MAX_REPORTED_JOBS = 4096
_reported_jobs: OrderedDict[str, None] = OrderedDict()
_reported_jobs_lock = threading.Lock()


def report_job_failure(
    job_id: str,
    error_message: Optional[str],
    *,
    log_tail: Optional[str] = None,
    source: str = "Server",
) -> bool:
    """Print one failed job to the server console, at most once per process.

    Returns ``True`` when this call emitted the failure and ``False`` when the
    job had already been reported.  The bounded history avoids repeated output
    from status polling without growing for the full lifetime of a listener.
    """
    with _reported_jobs_lock:
        if job_id in _reported_jobs:
            return False
        _reported_jobs[job_id] = None
        if len(_reported_jobs) > _MAX_REPORTED_JOBS:
            _reported_jobs.popitem(last=False)

    message = str(error_message or "unknown error")
    print(f"[{source}] Job {job_id} FAILED: {message}", file=sys.stderr, flush=True)
    details = str(log_tail or "").rstrip()
    if details:
        print(details, file=sys.stderr, flush=True)
    return True
