"""Shared result-download helper for the remote HTTP clients.

``_fetch_result_bytes`` streams a finished job's result body and reports
progress when the transfer is slow.  It is used by both the XIA run clients
(:mod:`client`) and the overlay client (:mod:`overlay_client`).
"""

from __future__ import annotations

import time

RESULT_PROGRESS_INTERVAL_S = 15.0  # report a slow result download this often


def _fetch_result_bytes(client, result_url: str, label: str, write=print) -> bytes:
    """Download a finished job's result, reporting progress when it is slow.

    The body is streamed so a large transfer never looks hung: if it runs
    longer than ``RESULT_PROGRESS_INTERVAL_S``, a one-line update is emitted
    every interval (e.g. ``"… | transferring result — 12.0/40.0 MB"``);
    fast transfers stay silent.  ``write`` lets callers route the line through
    a live ``tqdm`` bar (``pbar.write``) instead of plain ``print``.

    Raises ``RuntimeError`` on an HTTP error, surfacing the server's detail.
    """
    chunks: list[bytes] = []
    downloaded = 0
    next_report = time.monotonic() + RESULT_PROGRESS_INTERVAL_S

    with client.stream("GET", result_url) as resp:
        if resp.status_code >= 400:
            resp.read()  # buffer the body so .json()/.text are available
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"Result fetch failed: HTTP {resp.status_code} "
                f"from {result_url}\n  server detail: {detail}"
            )

        total = resp.headers.get("content-length")
        total_mb = int(total) / 1e6 if total and total.isdigit() else None

        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now >= next_report:
                got_mb = downloaded / 1e6
                size = f"{got_mb:.1f}/{total_mb:.1f} MB" if total_mb else f"{got_mb:.1f} MB"
                write(f"[geoxplain_aurora_adapter client] {label} | transferring result — {size}")
                next_report = now + RESULT_PROGRESS_INTERVAL_S

    return b"".join(chunks)
