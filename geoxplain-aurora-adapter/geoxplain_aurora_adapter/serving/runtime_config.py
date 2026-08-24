"""Runtime configuration resolution for the listener CLI.

Resolves listener settings from CLI args and an existing setup config file.
Run ``geoxplain-aurora-adapter setup`` to create that file before starting a
listener. Mode metadata lives in :mod:`listener_modes`, terminal styling /
prompts in :mod:`cli_style`, and bind-address checks in :mod:`bind_security`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cli_style import prompt, prompt_int, section
from .config import (
    DEFAULT_MEMORY_RETENTION,
    DEFAULT_RESULT_RETENTION,
    DEFAULT_WEATHERBENCH2_PATHS,
    default_output_dir,
    load_config,
    resolve_config_path,
)
from .listener_modes import (
    LISTENER_MODE_KEYS,
    normalize_mode,
)

SBATCH_DEFAULTS = {
    "account": "",
    "partition": "",
    "time": "01:00:00",
    "nodes": "1",
    "ntasks": "1",
    "gpus_per_task": "1",
    "venv": "",
    "log_dir": "logs/",
    "output_dir": "",
    "extra_sbatch": "",
    "extra_srun": "",
    "job_limit": "2",
    "overlay_on_login": True,
}

SBATCH_FIELDS = (
    ("account", "SLURM account"),
    ("partition", "SLURM partition"),
    ("time", "Wall time"),
    ("nodes", "Nodes"),
    ("ntasks", "Tasks"),
    ("gpus_per_task", "GPUs per task"),
    ("venv", "Python virtualenv"),
    ("log_dir", "SLURM log directory"),
    ("output_dir", "Results output directory"),
    ("extra_sbatch", "Extra #SBATCH options"),
    ("extra_srun", "Extra srun options"),
    ("job_limit", "Max concurrent SLURM jobs"),
)

# Fields that only apply to a subset of listener modes; not prompted otherwise.
SBATCH_FIELD_MODES = {"job_limit": {"sbatch-oneshot"}}


@dataclass
class ListenerSettings:
    config_path: Path
    mode: str
    host: str
    port: int
    remote_url: str
    persistent: bool
    sbatch_kwargs: dict[str, str]
    wrote_config: bool
    result_retention: str
    memory_retention: str


def current_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section_data = config.get(name, {})
    return dict(section_data) if isinstance(section_data, dict) else {}


def config_default(section_data: dict[str, Any], key: str, fallback: Any) -> Any:
    value = section_data.get(key)
    return fallback if value in (None, "") else value


def sbatch_default(key: str) -> Any:
    if key == "output_dir":
        return default_output_dir()
    return SBATCH_DEFAULTS[key]


def split_interactive_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def _section_value(section_data: dict[str, Any], key: str) -> Any:
    return section_data.get(key, section_data.get(key.replace("_", "-")))


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def _setup_deployment(existing: dict[str, Any]) -> str:
    setup = current_section(existing, "setup")
    return str(setup.get("deployment", "")).strip()


def _default_listener_mode_for_deployment(deployment: str) -> str:
    if deployment == "gpu-listener":
        return "gpu-listener"
    if deployment == "login-node":
        return "sbatch-oneshot"
    return ""


def _setup_command_for_listener_mode(mode: str) -> str:
    if mode == "gpu-listener":
        return "geoxplain-aurora-adapter setup --gpu-listener"
    return "geoxplain-aurora-adapter setup --login-node"


def _fail_missing_config(config_path: Path) -> None:
    raise SystemExit(
        f"No listener config found at {config_path}.\n"
        "Run setup first, for example:\n"
        "  geoxplain-aurora-adapter setup --gpu-listener\n"
        "  geoxplain-aurora-adapter setup --login-node"
    )


def _has_data_config(args: argparse.Namespace, existing: dict[str, Any]) -> bool:
    if getattr(args, "weatherbench2_paths", None):
        return True
    data = current_section(existing, "data")
    return _has_value(data.get("weatherbench2_paths") or data.get("wb2_paths"))


def _has_sbatch_config_value(args: argparse.Namespace, existing: dict[str, Any], key: str) -> bool:
    if _has_value(getattr(args, key, None)):
        return True
    return _has_value(_section_value(current_section(existing, "sbatch"), key))


def _fail_non_listener_deployment(deployment: str) -> None:
    label = {
        "client": "client-only",
        "local": "notebook directly on a GPU node",
    }.get(deployment, deployment)
    raise SystemExit(
        f"Not configured to run a listener: current setup profile is {label!r}.\n"
        "For direct GPU notebooks, call the Python API without remote=.\n"
        "To configure a listener, run one of:\n"
        "  geoxplain-aurora-adapter setup --gpu-listener\n"
        "  geoxplain-aurora-adapter setup --login-node"
    )


def _ensure_configured_for_listener(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    mode: str,
) -> None:
    missing: list[str] = []
    if not _has_data_config(args, existing):
        missing.append("[data].weatherbench2_paths")

    if mode.startswith("sbatch"):
        for key in ("account", "partition", "time", "venv"):
            if not _has_sbatch_config_value(args, existing, key):
                missing.append(f"[sbatch].{key}")

    if missing:
        command = _setup_command_for_listener_mode(mode)
        raise SystemExit(
            f"Not configured for listener mode {mode!r}.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Run: {command}\n"
            "You can also override individual values for one run with "
            "geoxplain-aurora-adapter listen --help."
        )


def _collect_sbatch(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    mode: str,
    interactive: bool,
    prompt_missing: bool,
) -> dict[str, str]:
    existing_sbatch = current_section(existing, "sbatch")
    values: dict[str, str] = {}
    if interactive and prompt_missing:
        section("SLURM / srun Parameters")
        print("  Press Enter to accept the selected default for each field.")
    for key, label in SBATCH_FIELDS:
        arg_value = getattr(args, key, None)
        default = config_default(existing_sbatch, key, sbatch_default(key))
        applies_to_mode = mode in SBATCH_FIELD_MODES.get(key, {mode})
        prompting = arg_value is None and interactive and prompt_missing and applies_to_mode
        if key == "gpus_per_task" and prompting:
            print(
                "  Note: RISE / ViT-CX parallelise across this many GPUs "
                "(one model replica each).\n"
                "        With 1, those methods run single-GPU — no parallelism. Recommendation: set to node's GPU count"
            )
        if arg_value is not None:
            values[key] = str(arg_value)
        elif prompting:
            values[key] = prompt(label, default)
        else:
            values[key] = str(default)
        if key == "gpus_per_task" and interactive and prompt_missing and str(values[key]).strip() == "1":
            print(
                "  Warning: gpus_per_task=1 — RISE and ViT-CX will NOT be "
                "parallelised (single-GPU)."
            )

    # overlay_on_login is a boolean toggle, not a free-text field — never
    # prompt for it; take the CLI flag, else the existing config, else default.
    overlay_arg = getattr(args, "overlay_on_login", None)
    if overlay_arg is not None:
        values["overlay_on_login"] = bool(overlay_arg)
    else:
        existing_overlay = existing_sbatch.get("overlay_on_login")
        if existing_overlay is None:
            values["overlay_on_login"] = bool(SBATCH_DEFAULTS["overlay_on_login"])
        elif isinstance(existing_overlay, bool):
            values["overlay_on_login"] = existing_overlay
        else:
            values["overlay_on_login"] = (
                str(existing_overlay).strip().lower() in ("1", "true", "yes", "on")
            )
    return values


def _collect_network(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
    prompt_missing: bool,
) -> dict[str, Any]:
    existing_network = current_section(existing, "network")
    host_default = config_default(existing_network, "host", "127.0.0.1")
    port_default = int(config_default(existing_network, "port", 8765))
    if interactive and prompt_missing:
        section("Listener Network")

    host = getattr(args, "host", None)
    if host is None:
        host = prompt("Bind host", host_default) if interactive and prompt_missing else str(host_default)

    port = getattr(args, "port", None)
    if port is None:
        port = prompt_int("Port", port_default) if interactive and prompt_missing else port_default
    else:
        port = int(port)

    remote_default = config_default(existing_network, "remote_url", f"http://localhost:{port}")
    remote_url = getattr(args, "remote_url", None)
    if remote_url is None:
        remote_url = (
            prompt("Client remote URL", remote_default)
            if interactive and prompt_missing
            else str(remote_default)
        )

    return {"host": str(host), "port": int(port), "remote_url": str(remote_url)}


def _collect_data(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
    prompt_missing: bool,
) -> dict[str, Any]:
    existing_data = current_section(existing, "data")
    existing_wb2 = existing_data.get("weatherbench2_paths") or existing_data.get("wb2_paths")
    if isinstance(existing_wb2, list) and existing_wb2:
        default_wb2 = [str(path) for path in existing_wb2]
    elif isinstance(existing_wb2, str) and existing_wb2:
        default_wb2 = split_interactive_paths(existing_wb2)
    else:
        default_wb2 = list(DEFAULT_WEATHERBENCH2_PATHS)

    cli_paths = getattr(args, "weatherbench2_paths", None)
    if cli_paths:
        wb2_paths = list(cli_paths)
    elif interactive and prompt_missing:
        section("Data Paths")
        print("  WeatherBench2/ERA5 zarr stores are tried in order.")
        answer = prompt(
            "WeatherBench2 paths (; separated; leave empty for hosted Google Cloud public bucket)",
            "; ".join(default_wb2),
        )
        wb2_paths = split_interactive_paths(answer)
    else:
        wb2_paths = default_wb2

    return {"weatherbench2_paths": wb2_paths}


def _collect_retention(args: argparse.Namespace, existing: dict[str, Any]) -> tuple[str, str]:
    """Resolve (result_retention, memory_retention): CLI flag > config > default.

    Not prompted interactively — both live in the ``[retention]`` config section
    so users can edit them, and can be overridden per run with --result-retention
    / --memory-retention.  ``result_retention`` governs on-disk result dirs
    (default ``"never"``); ``memory_retention`` governs the in-memory job store
    and cached result bytes (default ``"1h"``).
    """
    existing_retention = current_section(existing, "retention")

    result = getattr(args, "result_retention", None)
    if result is None:
        result = config_default(existing_retention, "result_retention", DEFAULT_RESULT_RETENTION)

    memory = getattr(args, "memory_retention", None)
    if memory is None:
        memory = config_default(existing_retention, "memory_retention", DEFAULT_MEMORY_RETENTION)

    return str(result), str(memory)


def _mode_from_args(args: argparse.Namespace) -> str | None:
    mode = getattr(args, "mode", None)
    if mode:
        return normalize_mode(mode, listener_only=True)
    if getattr(args, "persistent", False):
        return "sbatch-persistent"
    return None


def prepare_listener_settings(
    args: argparse.Namespace,
    *,
    gpu: bool,
    sbatch: bool,
) -> ListenerSettings:
    """Resolve listener settings from an existing setup config and CLI overrides."""
    config_path = resolve_config_path(getattr(args, "config", None))
    if getattr(args, "reset", False) and config_path.exists():
        config_path.unlink()
    existing = load_config(config_path)
    config_exists = config_path.exists()
    interactive = bool(sys.stdin.isatty() and not getattr(args, "yes", False))
    if not config_exists:
        _fail_missing_config(config_path)

    cli_mode = _mode_from_args(args)
    setup_section = current_section(existing, "setup")
    setup_deployment = _setup_deployment(existing)
    if config_exists and cli_mode is None and setup_deployment in {"client", "local"}:
        _fail_non_listener_deployment(setup_deployment)

    config_mode = str(setup_section.get("mode", "")).strip()
    if config_mode not in LISTENER_MODE_KEYS:
        config_mode = _default_listener_mode_for_deployment(setup_deployment)

    default_mode = cli_mode or config_mode
    if not default_mode:
        _fail_missing_config(config_path)
    mode = default_mode

    _ensure_configured_for_listener(args, existing, mode=mode)

    prompt_missing = False
    sbatch_values = (
        _collect_sbatch(args, existing, mode=mode, interactive=interactive, prompt_missing=prompt_missing)
        if mode.startswith("sbatch")
        else {}
    )
    data = _collect_data(args, existing, interactive=interactive, prompt_missing=prompt_missing)
    network = _collect_network(args, existing, interactive=interactive, prompt_missing=prompt_missing)
    result_retention, memory_retention = _collect_retention(args, existing)

    config = {
        "setup": {"mode": mode},
        "network": network,
        "retention": {
            "result_retention": result_retention,
            "memory_retention": memory_retention,
        },
        "data": data,
    }
    if sbatch_values:
        config["sbatch"] = sbatch_values

    wrote_config = False

    cli_sbatch_kwargs = {
        key: str(getattr(args, key))
        for key, _ in SBATCH_FIELDS
        if getattr(args, key, None) is not None
    }
    if getattr(args, "overlay_on_login", None) is not None:
        cli_sbatch_kwargs["overlay_on_login"] = bool(args.overlay_on_login)

    cli_paths = getattr(args, "weatherbench2_paths", None)
    if cli_paths:
        os.environ["GEOXPLAIN_AURORA_ADAPTER_WB2_PATHS"] = ";".join(cli_paths)
    os.environ["GEOXPLAIN_AURORA_ADAPTER_CONFIG"] = str(config_path)

    return ListenerSettings(
        config_path=config_path,
        mode=mode,
        host=network["host"],
        port=int(network["port"]),
        remote_url=network["remote_url"],
        persistent=mode == "sbatch-persistent",
        sbatch_kwargs=cli_sbatch_kwargs,
        wrote_config=wrote_config,
        result_retention=result_retention,
        memory_retention=memory_retention,
    )
