"""Helpers for path-like setup and SLURM option values."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterator


REMOTE_URI_PREFIXES = ("gs://", "s3://", "http://", "https://")
SRUN_FILE_OPTIONS = {"--environment", "--edf"}


def is_remote_uri(value: str) -> bool:
    return value.strip().lower().startswith(REMOTE_URI_PREFIXES)


def is_local_path_value(value: str) -> bool:
    text = value.strip()
    if not text or is_remote_uri(text):
        return False
    return (
        text.startswith(("~", ".", "/"))
        or os.sep in text
        or (os.altsep is not None and os.altsep in text)
    )


def normalize_local_path(value: str) -> str:
    text = str(value)
    if text == "~" or text.startswith(("~/", "~\\")):
        home = os.environ.get("HOME")
        if home:
            text = str(Path(home) / text[2:]) if len(text) > 1 else home
    return str(Path(text).expanduser().resolve())


def normalize_srun_file_options(extra_srun: str) -> str:
    """Expand local path values in known srun file options.

    Pyxis does not shell-expand ``~`` in ``--environment=~/...`` because the
    tilde is not at the start of the shell word.  Normalizing the value before
    it reaches ``srun`` prevents Pyxis from treating the path literally.
    """
    if not extra_srun.strip():
        return extra_srun

    tokens = shlex.split(extra_srun)
    normalized: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        handled = False
        for option in SRUN_FILE_OPTIONS:
            prefix = f"{option}="
            if token.startswith(prefix):
                value = token[len(prefix):]
                if is_local_path_value(value):
                    value = normalize_local_path(value)
                normalized.append(f"{option}={value}")
                handled = True
                break
        if handled:
            idx += 1
            continue

        if token in SRUN_FILE_OPTIONS and idx + 1 < len(tokens):
            value = tokens[idx + 1]
            if is_local_path_value(value):
                value = normalize_local_path(value)
            normalized.extend((token, value))
            idx += 2
            continue

        normalized.append(token)
        idx += 1

    if os.name == "nt":
        return " ".join(normalized)
    return shlex.join(normalized)


def iter_srun_file_options(extra_srun: str) -> Iterator[tuple[str, str]]:
    """Yield ``(option, value)`` for known srun file path options."""
    if not extra_srun.strip():
        return

    tokens = shlex.split(extra_srun)
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        for option in SRUN_FILE_OPTIONS:
            prefix = f"{option}="
            if token.startswith(prefix):
                yield option, token[len(prefix):]
                break
        else:
            if token in SRUN_FILE_OPTIONS and idx + 1 < len(tokens):
                yield token, tokens[idx + 1]
                idx += 2
                continue
        idx += 1
