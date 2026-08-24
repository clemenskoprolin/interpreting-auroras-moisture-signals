"""HTTP listener startup for geoxplain_aurora_adapter.

``listen_for_request`` auto-selects the serving backend from the environment:

    listen_for_request()  → sbatch-oneshot/sbatch-persistent (sbatch) or gpu-listener (GPU only)

GPU/sbatch detection is shared with the dispatch layer (:func:`dispatch._has_gpu`
and :func:`dispatch._has_sbatch`).
"""

from __future__ import annotations

from typing import Optional

from .dispatch import _has_gpu, _has_sbatch


def listen_for_request(
    port: int = 8765,
    host: str = "127.0.0.1",
    persistent: bool = False,
    prewarm: bool = False,
    memory_retention: Optional[str] = None,
    result_retention: Optional[str] = None,
    **sbatch_kwargs,
) -> None:
    """Start the geoxplain_aurora_adapter HTTP listener server.

    Mode is auto-detected:

    +------------------+------------------+---------------------------------+
    | GPU visible?     | sbatch on PATH?  | Resolved mode                   |
    +==================+==================+=================================+
    | yes              | yes              | sbatch-oneshot (or sbatch-persistent if persistent) |
    | no               | yes              | sbatch-oneshot (or sbatch-persistent if persistent) |
    | yes              | no               | gpu-listener                    |
    | no               | no               | Error                           |
    +------------------+------------------+---------------------------------+

    Parameters
    ----------
    port:           TCP port to listen on (default 8765).
    host:           Bind address (default ``"127.0.0.1"`` — loopback only; the
                    HTTP API is unauthenticated, so reach it via an SSH tunnel.
                    Pass ``"0.0.0.0"`` only on a trusted/firewalled network).
    persistent:     Opt into sbatch-persistent (persistent GPU worker) instead of the
                    default sbatch-oneshot (one sbatch per request).
    prewarm:        sbatch-persistent only — submit the GPU worker job immediately
                    rather than waiting for the first request.
    memory_retention: How long completed jobs and cached result bytes live in the
                    listener's memory (default ``"1h"``; ``"never"`` disables).
    result_retention: sbatch-oneshot only — how long on-disk result directories
                    are kept (default ``"never"``).
    **sbatch_kwargs: Forwarded to ``SbatchConfig`` (account, partition, etc.).
                    Also accepted as CLI flags by ``geoxplain-aurora-adapter listen``.
    """
    gpu = _has_gpu()
    sbatch = _has_sbatch()

    if not gpu and not sbatch:
        raise RuntimeError(
            "geoxplain_aurora_adapter.listen_for_request(): cannot serve requests.\n"
            "  Neither a GPU (CUDA) nor sbatch were found.\n"
            "  • On a GPU allocation: run directly and geoxplain_aurora_adapter will use it.\n"
            "  • On a login node with SLURM: ensure sbatch is on PATH."
        )

    if sbatch:
        if persistent:
            from ..remote.sbatch_persistent import run_server
            run_server(
                host=host, port=port, prewarm=prewarm,
                memory_retention=memory_retention, **sbatch_kwargs,
            )
        else:
            from ..remote.sbatch_oneshot import run_server
            run_server(
                host=host, port=port, prewarm=prewarm,
                memory_retention=memory_retention,
                result_retention=result_retention, **sbatch_kwargs,
            )
    else:
        from ..remote.inproc import run_server
        run_server(
            host=host, port=port, prewarm=prewarm,
            memory_retention=memory_retention,
        )
