"""Oneshot worker script — invoked inside a sbatch job by sbatch-oneshot.

Usage (called by SbatchOneshotBackend via sbatch)::

    python -m geoxplain_aurora_adapter.remote.worker_oneshot \\
        --request /path/to/request.json \\
        --output  /path/to/result.xia.npz

Loads the model, runs the requested XIA method, and saves the result.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Path to request JSON file.")
    parser.add_argument("--output",  required=True, help="Path to write .xia.npz result.")
    parser.add_argument("--status", help="Path to write progress status JSON.")
    args = parser.parse_args(argv)

    with open(args.request) as f:
        req_dict = json.load(f)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[worker_oneshot] device={device}")

    if "timestamps" in req_dict:
        from ..engine.overlay_compute import _pull_overlay_local
        result = _pull_overlay_local(
            variable=req_dict["variable"],
            timestamps=req_dict["timestamps"],
            level=req_dict.get("level"),
            **req_dict.get("options", {}),
        )
    elif "targets" in req_dict:
        from ..schema.spec import TargetSpec
        from ..engine.model import load_model
        from ..engine.compute import _run_local_batch
        from ..engine.progress import ProgressReporter, write_status_file
        model = load_model(device)
        progress = None
        if args.status:
            progress = ProgressReporter(
                f"{req_dict['method']} batch",
                None,
                total_frames=len(req_dict["targets"]),
                min_interval_s=0.1,
                status_callback=lambda snap: write_status_file(args.status, snap),
                print_updates=True,
                heartbeat_s=15.0,
            )
        targets = [TargetSpec.from_dict(t) for t in req_dict["targets"]]
        result = _run_local_batch(
            method=req_dict["method"],
            targets=targets,
            input_vars=req_dict["input_vars"],
            model=model,
            device=device,
            progress_reporter=progress,
            **req_dict.get("options", {}),
        )
    else:
        from ..schema.spec import TargetSpec
        from ..engine.model import load_model
        from ..engine.compute import _run_local
        from ..engine.progress import ProgressReporter, write_status_file
        model = load_model(device)
        progress = None
        if args.status:
            progress = ProgressReporter(
                req_dict["method"],
                None,
                min_interval_s=0.1,
                status_callback=lambda snap: write_status_file(args.status, snap),
                print_updates=True,
                heartbeat_s=15.0,
            )
        target = TargetSpec.from_dict(req_dict["target"])
        result = _run_local(
            method=req_dict["method"],
            target=target,
            input_vars=req_dict["input_vars"],
            model=model,
            device=device,
            progress_reporter=progress,
            **req_dict.get("options", {}),
        )
    result.save(args.output)
    print(f"[worker_oneshot] Saved result to {args.output}")


if __name__ == "__main__":
    main()
