"""Listener CLI for ``geoxplain-aurora-adapter listen``.

The listener reads ``~/.config/geoxplain-aurora-adapter/listen.toml``. Create
that config with ``geoxplain-aurora-adapter setup`` before starting a listener.
"""

from __future__ import annotations

import argparse
import errno
import re
import socket
import subprocess
import sys

from ..serving.listener_modes import normalize_mode
from ..serving.runtime_config import prepare_listener_settings


def _port_in_use(host: str, port: int) -> bool:
    """Return True only if an active listener already holds ``host:port``.

    The bind test must mirror uvicorn exactly, which sets ``SO_REUSEADDR``.
    Without it, a plain bind also fails on lingering ``TIME_WAIT`` / orphaned
    sockets (e.g. an SSH tunnel's connection left behind when a previous
    listener was killed) — a false positive, since ``ss -ltnp`` shows no
    listener and uvicorn would bind happily.  With ``SO_REUSEADDR`` set, the
    bind succeeds over those leftovers and fails (``EADDRINUSE``) only when a
    real listener is bound — exactly the case we want to catch.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host or "0.0.0.0", int(port)))
        except OSError as e:
            # Only a genuine "already bound" conflict counts; let any other
            # bind error surface later from uvicorn with its full context.
            return e.errno == errno.EADDRINUSE
    return False


def _port_listeners(port: int) -> list[tuple[int, str]]:
    """Best-effort ``[(pid, name), ...]`` of processes listening on ``port``.

    Tries ``ss`` then ``lsof``; returns ``[]`` if neither is available or the
    owner can't be determined (e.g. the socket belongs to another user).
    """
    found: dict[int, str] = {}
    try:
        out = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for name, pid in re.findall(r'\(\("([^"]+)",pid=(\d+)', out):
            found[int(pid)] = name
    except Exception:
        pass
    if not found:
        try:
            out = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    found[int(parts[1])] = parts[0]
        except Exception:
            pass
    return sorted(found.items())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geoxplain-aurora-adapter listen",
        description=(
            "Start a geoxplain-aurora-adapter HTTP listener.\n"
            "If no listener config exists, run `geoxplain-aurora-adapter setup` first.\n"
            "If config exists, it is read and startup proceeds without prompts.\n"
            "If the saved setup profile is not a listener profile, run "
            "`geoxplain-aurora-adapter setup --gpu-listener` or `setup --login-node`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--config", default=None,
                   help="Listener config path (default: ~/.config/geoxplain-aurora-adapter/listen.toml).")
    p.add_argument("--reset", action="store_true",
                   help="Delete the resolved listener config before startup; run setup again before listening.")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Run non-interactively; setup must already have written config.")

    # Network
    p.add_argument("--port", type=int, default=None,
                   help="TCP port to listen on (default: config or 8765).")
    p.add_argument("--host", default=None,
                   help="Bind address (default: config or 127.0.0.1 — loopback only; "
                        "the API is unauthenticated, reach it via an SSH tunnel).")
    p.add_argument("--remote-url", default=None,
                   help="Client-facing URL override.")
    p.add_argument("--result-retention", default=None, dest="result_retention",
                   help="Delete sbatch-oneshot on-disk result directories older "
                        "than this. E.g. 'never' (default), '24h', '7d', '30d'.")
    p.add_argument("--memory-retention", default=None, dest="memory_retention",
                   help="Drop completed jobs and cached result bytes from the "
                        "listener's memory after this long. E.g. '1h' (default), "
                        "'30m', '24h', 'never'.")

    # Mode flags
    p.add_argument("--mode", type=lambda value: normalize_mode(value, listener_only=True),
                   default=None,
                   help="Listener mode: gpu-listener, sbatch-oneshot, or sbatch-persistent.")
    p.add_argument("--persistent", action="store_true",
                   help="Shortcut for --mode sbatch-persistent.")
    p.add_argument("--prewarm", action="store_true",
                   help="Submit / load the model before the first request arrives.")
    p.add_argument("--checkpoint", default=None,
                   help="Override the default Aurora checkpoint path (gpu-listener only).")

    # sbatch attributes
    g = p.add_argument_group("SLURM sbatch attributes (sbatch-oneshot/sbatch-persistent only)")
    g.add_argument("--account",          default=None, help="SLURM account.")
    g.add_argument("--partition",        default=None, help="SLURM partition.")
    g.add_argument("--time",             default=None, help="Wall-clock limit (HH:MM:SS).")
    g.add_argument("--nodes",            default=None, help="Number of nodes.")
    g.add_argument("--ntasks",           default=None, help="Number of tasks.")
    g.add_argument("--gpus-per-task",    default=None, dest="gpus_per_task",
                   help="GPUs per task.")
    g.add_argument("--venv",             default=None,
                   help="Virtualenv to activate inside the job (e.g. ~/venv-aurora-xai).")
    g.add_argument("--log-dir",          default=None, dest="log_dir",
                   help="Directory for SLURM log files.")
    g.add_argument("--output-dir",       default=None, dest="output_dir",
                   help="Base directory for per-job results (default: $SCRATCH/geoxplain-aurora-adapter).")
    g.add_argument("--extra-sbatch",     default=None, dest="extra_sbatch",
                   help="Raw extra #SBATCH options appended verbatim (e.g. '--qos=preempt').")
    g.add_argument("--extra-srun",       default=None, dest="extra_srun",
                   help="Raw extra srun options appended after `srun -ul` "
                        "(e.g. '--container-image=...' or site-specific environment flags).")
    g.add_argument("--job-limit",        default=None, type=int, dest="job_limit",
                   help="sbatch-oneshot only: max concurrent SLURM jobs (default: 2).")
    g.add_argument("--overlay-on-login", default=None, dest="overlay_on_login",
                   action=argparse.BooleanOptionalAction,
                   help="Compute weather overlays in the login-node listener instead "
                        "of on a GPU job (default: on). Use --no-overlay-on-login to "
                        "submit overlays as SLURM jobs like other requests.")
    g.add_argument("--worker-port",      default=None, type=int, dest="worker_port",
                   help="sbatch-persistent only: port for the persistent GPU worker process.")

    d = p.add_argument_group("data paths")
    d.add_argument("--weatherbench2-path", action="append", dest="weatherbench2_paths",
                   help="WeatherBench2/ERA5 zarr fallback path. Repeat for multiple stores. "
                        "Defaults to the hosted public Google Cloud store.")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode and args.persistent and args.mode != "sbatch-persistent":
        parser.error("--persistent conflicts with --mode values other than sbatch-persistent")

    from ..api.dispatch import _has_gpu, _has_sbatch

    gpu = _has_gpu()
    sbatch = _has_sbatch()
    settings = prepare_listener_settings(args, gpu=gpu, sbatch=sbatch)

    if settings.mode == "gpu-listener" and not gpu:
        print(
            "ERROR: gpu-listener requires a CUDA GPU in the current allocation.",
            file=sys.stderr,
        )
        sys.exit(1)
    if settings.mode.startswith("sbatch") and not sbatch:
        print("ERROR: sbatch-oneshot/sbatch-persistent require sbatch on PATH (login node).", file=sys.stderr)
        sys.exit(1)

    # Setup is complete and the port is now known — bail out early (before the
    # backend spins up threads / prewarms the model) if something is already
    # listening there, instead of failing with a late "address already in use".
    if _port_in_use(settings.host, settings.port):
        lines = [
            f"ERROR: port {settings.port} on {settings.host} is already in use — "
            f"another geoxplain-aurora-adapter listener is likely running.",
        ]
        listeners = _port_listeners(settings.port)
        if listeners:
            who = ", ".join(f"PID {pid} ({name})" for pid, name in listeners)
            pids = " ".join(str(pid) for pid, _ in listeners)
            lines.append(f"  Listening now: {who}")
            lines.append(f"  Terminate it with:  kill {pids}    (force: kill -9 {pids})")
        else:
            lines.append(
                f"  Find the owning process with:  ss -ltnp 'sport = :{settings.port}'"
                f"   (or: lsof -ti :{settings.port})   then: kill <pid>"
            )
        lines.append("  Or start this listener elsewhere with --port <other>.")
        print("\n".join(lines), file=sys.stderr)
        sys.exit(1)

    if settings.mode.startswith("sbatch"):
        if settings.persistent:
            from .sbatch_persistent import run_server
            run_server(
                host=settings.host,
                port=settings.port,
                prewarm=args.prewarm,
                worker_port=args.worker_port,
                memory_retention=settings.memory_retention,
                **settings.sbatch_kwargs,
            )
        else:
            from .sbatch_oneshot import run_server
            run_server(
                host=settings.host,
                port=settings.port,
                prewarm=args.prewarm,
                memory_retention=settings.memory_retention,
                result_retention=settings.result_retention,
                **settings.sbatch_kwargs,
            )
    else:
        from .inproc import run_server
        run_server(
            host=settings.host,
            port=settings.port,
            prewarm=args.prewarm,
            checkpoint_path=args.checkpoint,
            memory_retention=settings.memory_retention,
        )


if __name__ == "__main__":
    main()
