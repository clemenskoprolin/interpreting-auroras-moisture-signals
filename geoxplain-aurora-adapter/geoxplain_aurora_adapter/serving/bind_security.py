"""Bind-address security checks for the (unauthenticated) listener HTTP API."""

from __future__ import annotations

import sys

# Bind addresses that keep the (unauthenticated) HTTP API loopback-only.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}


def is_public_bind(host: str) -> bool:
    """True when ``host`` is a non-loopback address (network-reachable)."""
    return str(host).strip().lower() not in _LOOPBACK_HOSTS


def warn_if_public_bind(host: str) -> None:
    """Warn when the listener binds a non-loopback address.

    The HTTP API has no authentication and permits job submission, result
    access, and config inspection.  On a shared HPC node a public bind exposes
    all of that to anyone who can reach the address.  The intended access path
    is an SSH tunnel to ``localhost``.
    """
    if not is_public_bind(host):
        return
    print(
        f"WARNING: binding {host} exposes the unauthenticated geoxplain-aurora-adapter "
        "HTTP API (job submission, results, config) to the network.\n"
        "         Prefer the default 127.0.0.1 and reach it via an SSH tunnel:\n"
        f"           ssh -L <port>:localhost:<port> <this-host>\n"
        "         Only bind a public address on a trusted/firewalled network.",
        file=sys.stderr,
    )
