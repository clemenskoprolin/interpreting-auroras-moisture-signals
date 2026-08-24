"""Formatting helpers for remote progress log lines."""

from __future__ import annotations


def short_job_id(job_id: str) -> str:
    return f"{job_id[:4]}..." if len(job_id) > 4 else job_id


def format_progress_log(job_id: str, text: str) -> str:
    return f"[{short_job_id(job_id)}] {text}"
