"""Listener / setup mode definitions and normalization.

Pure metadata: the deployment modes the setup and listener CLI offer, the
aliases users may type for each, and the helpers that normalize a raw value to a
canonical mode key.
"""

from __future__ import annotations

import argparse

ModeInfo = tuple[str, str, str]

LISTENER_MODES: tuple[ModeInfo, ...] = (
    (
        "gpu-listener",
        "gpu-listener: remote GPU listener",
        "Run the HTTP listener inside an existing GPU allocation.",
    ),
    (
        "sbatch-oneshot",
        "sbatch-oneshot: one sbatch per request",
        "Run on a login node and submit one SLURM job for each request.",
    ),
    (
        "sbatch-persistent",
        "sbatch-persistent: persistent GPU worker",
        "Run on a login node and keep one long-lived GPU worker warm.",
    ),
)

SETUP_MODES: tuple[ModeInfo, ...] = (
    (
        "client",
        "client: remote-only notebooks/scripts",
        "Submit requests to an already running listener.",
    ),
    (
        "local",
        "local: notebook directly on a GPU node",
        "Run notebooks directly inside a GPU allocation.",
    ),
    (
        "gpu-listener",
        "gpu-listener: remote GPU listener",
        "Run the HTTP listener inside an existing GPU allocation.",
    ),
    (
        "login-node",
        "login-node: SLURM listener plus GPU worker",
        "Run the listener on a login node and submit GPU work through SLURM.",
    ),
)

INSTALL_MODES = SETUP_MODES

SETUP_MODE_KEYS = {key for key, _, _ in SETUP_MODES}
MODE_KEYS = SETUP_MODE_KEYS
LISTENER_MODE_KEYS = {key for key, _, _ in LISTENER_MODES}

INSTALL_MODE_ALIASES = {
    "1": "client",
    "mode1": "client",
    "mode-1": "client",
    "client": "client",
    "client-only": "client",
    "remote": "client",
    "remote-only": "client",
    "2": "local",
    "mode2": "local",
    "mode-2": "local",
    "notebook": "local",
    "gpu-notebook": "local",
    "local-gpu": "local",
    "inprocess": "local",
    "in-process": "local",
    "local": "local",
    "3": "gpu-listener",
    "mode3": "gpu-listener",
    "mode-3": "gpu-listener",
    "gpu": "gpu-listener",
    "listener": "gpu-listener",
    "gpu-listener": "gpu-listener",
    "4": "login-node",
    "mode4": "login-node",
    "mode-4": "login-node",
    "login": "login-node",
    "login-node": "login-node",
    "login-server": "login-node",
    "slurm": "login-node",
    "sbatch": "login-node",
    "oneshot": "login-node",
    "one-shot": "login-node",
    "sbatch-oneshot": "login-node",
    "persistent": "login-node",
    "sbatch-persistent": "login-node",
}

LISTENER_MODE_ALIASES = {
    "2": "gpu-listener",
    "mode2": "gpu-listener",
    "mode-2": "gpu-listener",
    "gpu": "gpu-listener",
    "listener": "gpu-listener",
    "gpu-listener": "gpu-listener",
    "3": "sbatch-oneshot",
    "3b": "sbatch-oneshot",
    "mode3": "sbatch-oneshot",
    "mode-3": "sbatch-oneshot",
    "oneshot": "sbatch-oneshot",
    "one-shot": "sbatch-oneshot",
    "sbatch": "sbatch-oneshot",
    "sbatch-oneshot": "sbatch-oneshot",
    "persistent": "sbatch-persistent",
    "sbatch-persistent": "sbatch-persistent",
}


def normalize_mode(value: str, *, listener_only: bool = False) -> str:
    key = value.strip().lower().replace("_", "-").replace(" ", "")
    aliases = LISTENER_MODE_ALIASES if listener_only else INSTALL_MODE_ALIASES
    mode = aliases.get(key)
    allowed = LISTENER_MODE_KEYS if listener_only else MODE_KEYS
    if mode is None or mode not in allowed:
        choices = ", ".join(sorted(allowed))
        raise argparse.ArgumentTypeError(f"unknown mode {value!r}; choose one of: {choices}")
    return mode


def guess_listener_mode(*, gpu: bool, sbatch: bool) -> str:
    if sbatch:
        return "sbatch-oneshot"
    if gpu:
        return "gpu-listener"
    return "sbatch-oneshot"
