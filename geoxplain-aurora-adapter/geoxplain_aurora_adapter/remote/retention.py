"""Background retention janitor for listener backends.

Completed jobs accumulate in every backend's in-memory job store (and, for
sbatch-oneshot, as result directories on shared storage).  Left unbounded, a
long-lived listener eventually exhausts RAM or disk.

Two independent windows govern cleanup:

* **memory retention** — completed job records and cached result bytes.  Bounded
  by default (``DEFAULT_MEMORY_RETENTION``); a completed job only needs to
  outlive the client's poll → fetch round-trip.
* **result (disk) retention** — sbatch-oneshot on-disk result directories.
  ``"never"`` by default; disk is cheap, so purging is opt-in.

``start_janitor`` spawns a daemon thread that calls a backend's ``purge``
callback on a fixed cadence.  The callback decides what to delete using whatever
windows the backend captured; the janitor only schedules it.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


# Completed jobs are checked on this cadence.  Coarse on purpose: purging is
# housekeeping, not latency-sensitive, and we don't want to wake constantly.
DEFAULT_JANITOR_INTERVAL_S = 300


def start_janitor(
    purge_fn: Callable[[], None],
    *,
    enabled: bool,
    interval_s: float = DEFAULT_JANITOR_INTERVAL_S,
    name: str = "geoxplain-aurora-adapter-janitor",
    description: str = "",
    announce: bool = True,
) -> Optional[threading.Thread]:
    """Start a daemon thread that calls ``purge_fn()`` periodically.

    Returns the thread, or ``None`` when ``enabled`` is falsy (every retention
    window is "never" — nothing to purge).  Pass ``announce=False`` to suppress
    the startup line when the caller surfaces retention in its own banner.
    """
    if not enabled:
        return None

    def _loop() -> None:
        while True:
            time.sleep(interval_s)
            try:
                purge_fn()
            except Exception as exc:  # never let housekeeping kill the listener
                print(f"[janitor] purge failed: {exc}")

    thread = threading.Thread(target=_loop, daemon=True, name=name)
    thread.start()
    if announce:
        suffix = f" ({description})" if description else ""
        print(
            f"[janitor] Retention janitor started — purging every "
            f"{interval_s:.0f}s{suffix}."
        )
    return thread
