"""Protect a job's result from retention purges while it is being downloaded.

A result fetch (``GET /jobs/{id}/result``) materializes the packed bytes and
hands them to the HTTP layer, which keeps streaming them to the client *after*
the route handler has returned.  The retention janitor runs on its own thread,
so without coordination it could delete a job's in-memory record — or, for the
oneshot backend, its on-disk result directory — out from under an in-flight
transfer (especially a slow one, now that large results stream in chunks).

The server *pins* a job for the lifetime of a fetch (from before the bytes are
read until the response has finished sending); every backend's purge skips
pinned jobs.  Pins are reference-counted so concurrent fetches of the same job
are handled, and carry a TTL so a client that disconnects mid-transfer — which
can prevent the post-send unpin from running — cannot keep a job pinned (and
thus un-purgeable) forever.
"""

from __future__ import annotations

import threading
import time

# Self-heal a leaked pin after this long. Comfortably longer than any result
# transfer, short enough that a leak doesn't pin a job indefinitely.
DEFAULT_PIN_TTL_S = 1800.0

_lock = threading.Lock()
_pins: dict[str, list] = {}  # job_id -> [refcount, expiry_epoch]


def pin(job_id: str, ttl_s: float = DEFAULT_PIN_TTL_S) -> None:
    """Mark ``job_id`` as having an in-flight transfer (reference-counted)."""
    expiry = time.time() + ttl_s
    with _lock:
        entry = _pins.get(job_id)
        if entry is None:
            _pins[job_id] = [1, expiry]
        else:
            entry[0] += 1
            entry[1] = max(entry[1], expiry)


def unpin(job_id: str) -> None:
    """Release one transfer pin on ``job_id``; drop it when the count hits 0."""
    with _lock:
        entry = _pins.get(job_id)
        if entry is None:
            return
        entry[0] -= 1
        if entry[0] <= 0:
            del _pins[job_id]


def is_pinned(job_id: str) -> bool:
    """True while ``job_id`` has a live transfer pin (expired pins self-clear)."""
    with _lock:
        entry = _pins.get(job_id)
        if entry is None:
            return False
        if entry[1] <= time.time():
            del _pins[job_id]
            return False
        return True
