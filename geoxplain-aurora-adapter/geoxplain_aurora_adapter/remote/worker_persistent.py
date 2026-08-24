"""Persistent worker script — invoked inside a sbatch job by sbatch-persistent.

Usage (called by SbatchPersistentBackend via sbatch)::

    python -m geoxplain_aurora_adapter.remote.worker_persistent \\
        --port 52345 \\
        --worker-json /path/to/worker.json

Starts an inproc FastAPI server on the specified port and writes ``worker.json``
with ``{"host": hostname, "port": port, "job_id": slurm_job_id}``.  The
login-node listener discovers the worker from that file and monitors liveness by
polling the worker's ``GET /health`` endpoint — the worker does not push
heartbeats (it never learns the listener's address).
"""

from __future__ import annotations

import argparse
import json
import os
import socket


def _write_worker_json(path: str, port: int) -> None:
    info = {
        "host": socket.gethostname(),
        "port": port,
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f)
    print(f"[worker_persistent] Wrote worker.json: {info}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True,
                        help="Port to listen on.")
    parser.add_argument("--worker-json", required=True, dest="worker_json",
                        help="Path to write worker info JSON.")
    args = parser.parse_args(argv)

    _write_worker_json(args.worker_json, args.port)

    # Bind 0.0.0.0: the worker runs on a compute node and must be reachable by
    # the login-node listener across the cluster network (not just loopback).
    from .inproc import run_server
    run_server(host="0.0.0.0", port=args.port, prewarm=True)


if __name__ == "__main__":
    main()
