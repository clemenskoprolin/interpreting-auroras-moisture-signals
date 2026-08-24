"""Terminal styling, interactive prompts, and the listener startup banner.

The setup / listener CLI use these to print colored headings,
ask for missing settings, and render the startup screen.  Color is suppressed
when stdout is not a TTY or ``NO_COLOR`` is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Sequence

from .bind_security import is_public_bind
from .listener_modes import ModeInfo, normalize_mode


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def headline(text: str) -> None:
    print()
    print(_color(text, "1;36"))
    print(_color("=" * len(text), "36"))


def section(text: str) -> None:
    print()
    print(_color(text, "1"))
    print(_color("-" * len(text), "2"))


def warning_section(title: str, lines: Sequence[str]) -> None:
    """A section whose heading and rule are amber, for attention-worthy notes."""
    print()
    print(_color(title, "1;33"))
    print(_color("-" * len(title), "33"))
    for idx, line in enumerate(lines):
        marker = _color("⚠", "1;33") + "  " if idx == 0 else "   "
        print(f"  {marker}{line}")


def prompt(text: str, default: Any = None, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        answer = input(f"  {text}{suffix}: ").strip()
        if not answer and default is not None:
            return str(default)
        if answer or not required:
            return answer
        print("  Please enter a value.")


def prompt_yes_no(text: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"  {text} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Enter yes or no.")


def prompt_int(text: str, default: int) -> int:
    while True:
        answer = prompt(text, default)
        try:
            return int(answer)
        except ValueError:
            print("  Enter a whole number.")


def prompt_choice(
    title: str,
    choices: Sequence[ModeInfo],
    default_key: str,
    *,
    listener_only: bool = False,
) -> str:
    section(title)
    default_idx = 1 + next(i for i, (key, _, _) in enumerate(choices) if key == default_key)
    for idx, (key, label, description) in enumerate(choices, start=1):
        marker = " (default)" if key == default_key else ""
        print(f"  {idx}. {_color(label, '1')}{marker}")
        print(f"     {description}")
    while True:
        answer = input(f"  Choice [{default_idx}]: ").strip()
        if not answer:
            return default_key
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(choices):
                return choices[idx - 1][0]
        try:
            return normalize_mode(answer, listener_only=listener_only)
        except argparse.ArgumentTypeError as exc:
            print(f"  {exc}")


_MODE_TAGLINES = {
    "gpu-listener": "in-process GPU listener",
    "sbatch-oneshot": "one SLURM job per request",
    "sbatch-persistent": "persistent GPU worker",
}


def _field(label: str, value: Any, *, width: int = 13) -> None:
    """Print an aligned ``label   value`` row (label dimmed)."""
    print(f"  {_color(f'{label:<{width}}', '2')}{value}")


def _format_retention(retention: dict[str, Any] | None) -> str:
    """One-line summary of the retention janitor for the banner."""
    if not retention or not retention.get("enabled"):
        return _color("off — completed jobs kept until shutdown", "2")
    interval = retention.get("interval_s")
    windows = retention.get("windows") or ""
    parts = []
    if interval:
        parts.append(f"purge every {float(interval):.0f}s")
    if windows:
        parts.append(windows)
    return _color("   ·   ".join(parts), "2")


def render_listener_banner(
    *,
    mode: str,
    hostname: str,
    port: int,
    gpu: bool,
    sbatch: bool,
    sbatch_config: dict[str, Any] | None = None,
    job_limit: int | None = None,
    worker_port: int | None = None,
    bind_host: str | None = None,
    retention: dict[str, Any] | None = None,
) -> None:
    """Print the listener startup screen, matching the setup style."""
    headline("GeoXplain Aurora Listener")

    tagline = _MODE_TAGLINES.get(mode, "")
    _field("Mode", f"{_color(mode, '1;36')}  {_color('— ' + tagline, '2') if tagline else ''}")
    _field("Address", f"{hostname}:{port}")

    compute = [f"GPU {'✓' if gpu else '—'}", f"sbatch {'✓' if sbatch else '—'}"]
    if job_limit is not None:
        compute.append(f"job limit {job_limit}")
    if worker_port is not None:
        compute.append(f"worker port {worker_port}")
    _field("Compute", _color("   ·   ".join(compute), "2"))
    _field("Retention", _format_retention(retention))

    if bind_host is not None and is_public_bind(bind_host):
        warning_section(
            "Security",
            (
                f"Bound to {_color(bind_host, '1')} — the unauthenticated HTTP API "
                "(job submission, results,",
                "config) is reachable by anyone on the network.",
                "Prefer the default 127.0.0.1 and reach it via an SSH tunnel:",
                _color(f"ssh -L {port}:localhost:{port} {hostname}", "1"),
                "Only bind a public address on a trusted/firewalled network.",
            ),
        )

    if sbatch_config:
        cfg = sbatch_config
        section("SLURM Submission")
        _field("account", cfg.get("account", ""))
        _field("partition", cfg.get("partition", ""))
        _field("time", cfg.get("time", ""))
        _field(
            "resources",
            f"{cfg.get('nodes', '')} node(s)  ·  {cfg.get('ntasks', '')} task(s)"
            f"  ·  {cfg.get('gpus_per_task', '')} gpu/task",
        )
        _field("venv", cfg.get("venv", ""))
        _field("log dir", cfg.get("log_dir", ""))
        _field("output dir", cfg.get("output_dir", ""))
        _field(
            "overlays",
            "login node (no GPU job)" if cfg.get("overlay_on_login", True)
            else "SLURM GPU job",
        )
        if cfg.get("extra_sbatch"):
            _field("extra sbatch", cfg["extra_sbatch"])
        if cfg.get("extra_srun"):
            _field("extra srun", cfg["extra_srun"])

    section("Connect")
    if mode == "gpu-listener":
        remote = "remote='http://{}:{}'".format(hostname, port)
        tunnel = "ssh -L {0}:{1}:{0} <login-node>".format(port, hostname)
        print("  From a machine that can reach the GPU node:")
        print(f"    {_color(remote, '1')}")
        print("  Or tunnel through a login node first:")
        print(f"    {_color(tunnel, '1')}")
    else:
        tunnel = "ssh -L {0}:localhost:{0} {1}".format(port, hostname)
        remote = "remote='http://localhost:{}'".format(port)
        print("  Use it directly in a notebook / script on the login node:")
        print(f"     {_color(remote, '1')}")
        print("  Or remotely:")
        print("  1. Forward the port from your machine:")
        print(f"     {_color(tunnel, '1')}")
        print("  2. Then in your notebook / script:")
        print(f"     {_color(remote, '1')}")

    print()
    print(f"  {_color('●', '1;32')} Listener ready — waiting for requests.  {_color('Press Ctrl-C to stop.', '2')}")
    print()
