"""Thin wrappers around the SLURM command-line tools.

These shell out to ``sbatch`` / ``squeue`` / ``sacct`` / ``scancel`` and parse
their output.  The sbatch-oneshot backend uses the submit/query helpers; the
sbatch-persistent backend uses the cancel / port / wall-time helpers.  Keeping
them here isolates the (mockable) SLURM interface from the backend logic.
"""

from __future__ import annotations

import random
import subprocess
import time
from typing import Optional


def _ts() -> str:
    """Local wall-clock timestamp for server log lines, e.g. ``2026-06-15 18:05:33``."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sbatch(script_path: str) -> str:
    """Submit ``script_path`` and return the SLURM job ID string.

    On failure, surface ``sbatch``'s stderr -- a bare ``CalledProcessError``
    only reports the exit status, hiding the actual rejection reason.
    """
    result = subprocess.run(
        ["sbatch", "--parsable", script_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"sbatch rejected the submit script (exit {result.returncode}): "
            f"{detail or 'no output'}"
        )
    return result.stdout.strip().split(";")[0]


def _sacct_state(job_id: str) -> str:
    """Look up the terminal state of a SLURM job via ``sacct``.

    Returns the state string (e.g. ``"COMPLETED"``, ``"FAILED"``,
    ``"CANCELLED"``, ``"TIMEOUT"``, ``"OUT_OF_MEMORY"``) or ``"UNKNOWN"``
    if sacct doesn't know about the job either.
    """
    try:
        result = subprocess.run(
            ["sacct", "-X", "-j", job_id, "-P", "-n", "-o", "state"],
            capture_output=True, text=True, check=True, timeout=10
        )
        first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        # sacct may emit "CANCELLED by <uid>" → keep the leading token
        return first.split()[0] if first else "UNKNOWN"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return "UNKNOWN"


def _squeue_states(job_ids: list[str]) -> dict[str, str]:
    """Return {slurm_job_id: state_code} for the listed job IDs.

    Jobs no longer in ``squeue`` are looked up in ``sacct`` so the listener
    sees a true ``FAILED`` / ``TIMEOUT`` instead of misreporting them as
    ``COMPLETED`` (and then failing the file-existence check).
    """
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "--me", "--format=%i %T", "--noheader"],
            capture_output=True, text=True, check=True
        )
        out: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
        return {jid: out[jid] if jid in out else _sacct_state(jid) for jid in job_ids}
    except subprocess.CalledProcessError:
        return {jid: "UNKNOWN" for jid in job_ids}


def _squeue_eta(slurm_job_id: str) -> Optional[float]:
    """Try to get the estimated start time ETA via ``squeue --start``."""
    try:
        result = subprocess.run(
            ["squeue", "--start", "-j", slurm_job_id, "--format=%S", "--noheader"],
            capture_output=True, text=True, timeout=5, check=True
        )
        start_str = result.stdout.strip()
        if not start_str or start_str in ("N/A", "Unknown"):
            return None
        from datetime import datetime
        start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
        eta = (start - datetime.now()).total_seconds()
        return max(0.0, eta)
    except Exception:
        return None


def _scancel(slurm_id: Optional[str]) -> None:
    """Best-effort ``scancel`` of a worker job."""
    if not slurm_id:
        return
    try:
        subprocess.run(["scancel", slurm_id], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _pick_worker_port() -> int:
    return random.randint(50000, 59999)


def _parse_walltime_s(value: str) -> Optional[float]:
    """Parse a SLURM ``--time`` string into seconds.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``DD-HH:MM:SS`` and the
    ``DD-HH``/``DD-HH:MM`` variants.  Returns ``None`` when it can't be parsed
    (wall-time replacement is then disabled rather than guessed).
    """
    text = (value or "").strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = text.split(":") if text else []
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if not nums:
        total = 0
    elif len(nums) == 1:
        # Bare number: minutes when no day part, else hours (DD-HH).
        total = nums[0] * (3600 if days else 60)
    elif len(nums) == 2:
        # HH:MM with a day part, otherwise MM:SS.
        total = nums[0] * 3600 + nums[1] * 60 if days else nums[0] * 60 + nums[1]
    elif len(nums) == 3:
        total = nums[0] * 3600 + nums[1] * 60 + nums[2]
    else:
        return None
    total += days * 86400
    return float(total) if total > 0 else None
