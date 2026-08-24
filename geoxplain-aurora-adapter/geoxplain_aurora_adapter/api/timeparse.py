"""Timestamp, overlay-date, and timeframe parsing helpers.

Pure functions shared by the dispatch core (``_run_batch`` expands timeframe
targets) and the public wrappers (``pull_overlay`` expands overlay date
selectors). They have no GPU/torch dependency and perform no I/O, so they are
kept separate from the routing logic in :mod:`dispatch`.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import re

from ..schema.spec import TargetSpec


def _parse_timestamp(timestamp: str) -> datetime:
    ts = timestamp.rstrip("Z").replace("+00:00", "")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        raise ValueError(
            f"Cannot parse timestamp {timestamp!r}. "
            "Expected ISO-8601, e.g. '2024-03-20T00:00:00Z'."
        )


def _format_timestamp(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


def _shift_timestamp(timestamp: str, hours: int) -> str:
    """Return ``timestamp`` shifted by ``hours`` (may be negative)."""
    if hours == 0:
        return timestamp
    return _format_timestamp(_parse_timestamp(timestamp) + timedelta(hours=hours))


def _split_overlay_range(value: str) -> tuple[str, str] | None:
    if "..." in value:
        left, right = value.split("...", 1)
        return left.strip(), right.strip()
    if ".." in value:
        left, right = value.split("..", 1)
        return left.strip(), right.strip()
    return None


def _parse_overlay_time_token(value: str) -> tuple[datetime, bool]:
    token = value.strip()
    if not token:
        raise ValueError("Empty timestamp in overlay date selector.")
    is_date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", token))
    if is_date_only:
        return datetime.fromisoformat(token), True
    return _parse_timestamp(token), False


def _expand_overlay_single(value: str, *, step_hours: int) -> list[str]:
    dt, is_date_only = _parse_overlay_time_token(value)
    if not is_date_only:
        return [_format_timestamp(dt)]
    return [_format_timestamp(dt + timedelta(hours=hour)) for hour in range(0, 24, step_hours)]


def _expand_overlay_range(start: str, end: str, *, step_hours: int) -> list[str]:
    start_dt, _ = _parse_overlay_time_token(start)
    end_dt, end_is_date = _parse_overlay_time_token(end)
    if end_is_date:
        end_dt = end_dt + timedelta(hours=24 - step_hours)
    if end_dt < start_dt:
        raise ValueError(f"Overlay date range end {end!r} is before start {start!r}.")
    timestamps = []
    current = start_dt
    while current <= end_dt:
        timestamps.append(_format_timestamp(current))
        current += timedelta(hours=step_hours)
    return timestamps


def _expand_overlay_timestamps(
    dates: str | list[str] | tuple[str, ...],
    *,
    step_hours: int = 6,
) -> list[str]:
    if not isinstance(step_hours, int) or step_hours < 1 or 24 % step_hours != 0:
        raise ValueError(f"step_hours must be a positive divisor of 24, got {step_hours!r}")

    items = [dates] if isinstance(dates, str) else list(dates)
    expanded: list[str] = []
    for item in items:
        range_parts = _split_overlay_range(str(item))
        if range_parts:
            expanded.extend(_expand_overlay_range(range_parts[0], range_parts[1], step_hours=step_hours))
        else:
            expanded.extend(_expand_overlay_single(str(item), step_hours=step_hours))

    seen = set()
    unique = []
    for timestamp in expanded:
        if timestamp not in seen:
            seen.add(timestamp)
            unique.append(timestamp)
    if not unique:
        raise ValueError("Overlay date selector produced no timestamps.")
    return unique


def _expand_timeframe_targets(
    target: TargetSpec,
    *,
    timeframes: int,
    step_hours: int = 6,
) -> list[TargetSpec]:
    if not isinstance(timeframes, int) or timeframes < 1:
        raise ValueError(f"timeframes must be a positive integer, got {timeframes!r}")
    if not isinstance(step_hours, int) or step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")

    start = _parse_timestamp(target.timestamp)
    return [
        replace(target, timestamp=_format_timestamp(start + timedelta(hours=step_hours * i)))
        for i in range(timeframes)
    ]
