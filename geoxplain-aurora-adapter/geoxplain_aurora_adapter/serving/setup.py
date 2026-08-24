"""Non-mutating setup guide for geoxplain_aurora_adapter."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .cli_style import headline, prompt, prompt_choice, prompt_int, section
from .config import (
    DEFAULT_MEMORY_RETENTION,
    DEFAULT_RESULT_RETENTION,
    DEFAULT_WEATHERBENCH2_PATHS,
    load_config,
    resolve_config_path,
    write_config,
)
from .listener_modes import LISTENER_MODES, SETUP_MODES, normalize_mode
from .path_options import (
    is_local_path_value,
    is_remote_uri,
    iter_srun_file_options,
    normalize_local_path,
    normalize_srun_file_options,
)
from .runtime_config import (
    SBATCH_FIELD_MODES,
    SBATCH_FIELDS,
    config_default,
    current_section,
    sbatch_default,
    split_interactive_paths,
)


PROFILE_LABELS = {
    "client": "client-only notebooks/scripts",
    "local": "notebooks directly on a GPU node",
    "gpu-listener": "HTTP listener inside a GPU allocation",
    "login-node": "login-node listener with SLURM GPU workers",
}

SBATCH_LISTENER_MODES = tuple(mode for mode in LISTENER_MODES if mode[0].startswith("sbatch"))
SBATCH_EXISTING_DIR_FIELDS = {"venv"}
SBATCH_CREATABLE_DIR_FIELDS = {"log_dir", "output_dir"}

INSTALL_COMMANDS = {
    "client": (
        ("client machine", "python -m pip install -e '.[client]'"),
    ),
    "local": (
        ("GPU notebook environment", "python -m pip install -e '.[gpu]'"),
    ),
    "gpu-listener": (
        ("GPU listener environment", "python -m pip install -e '.[gpu,server]'"),
        ("client machines", "python -m pip install -e '.[client]'"),
    ),
    "login-node": (
        ("login-node listener environment", "python -m pip install -e '.[server]'"),
        ("GPU worker environment", "python -m pip install -e '.[gpu,server,client]'"),
        ("client machines", "python -m pip install -e '.[client]'"),
    ),
}


@dataclass
class SetupPlan:
    profile: str
    listener_mode: str | None
    config_path: Path
    config: dict[str, dict[str, Any]]
    wrote_config: bool
    no_write: bool


def _setup_mode(value: str) -> str:
    return normalize_mode(value, listener_only=False)


def _listener_mode(value: str) -> str:
    return normalize_mode(value, listener_only=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geoxplain-aurora-adapter setup",
        description=(
            "Choose a preferred deployment mode, write only the adapter config "
            "needed for that mode, and print the source extras to install."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help="Config path (default: ~/.config/geoxplain-aurora-adapter/listen.toml).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing config defaults and replace the config that setup writes.",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Accept defaults for any missing setup choices.",
    )
    p.add_argument(
        "--dry-run",
        "--no-write",
        action="store_true",
        dest="no_write",
        help="Print the guide and resolved config without writing listen.toml.",
    )

    profile = p.add_mutually_exclusive_group()
    profile.add_argument(
        "--mode",
        type=_setup_mode,
        dest="profile",
        metavar="{client,local,gpu-listener,login-node}",
        help="Preferred deployment profile.",
    )
    profile.add_argument(
        "--client",
        action="store_const",
        const="client",
        dest="profile",
        help="Configure a client-only environment.",
    )
    profile.add_argument(
        "--local",
        action="store_const",
        const="local",
        dest="profile",
        help="Configure notebooks that run directly on a GPU node.",
    )
    profile.add_argument(
        "--gpu-listener",
        action="store_const",
        const="gpu-listener",
        dest="profile",
        help="Configure an HTTP listener running inside a GPU allocation.",
    )
    profile.add_argument(
        "--login-node",
        "--login-server",
        action="store_const",
        const="login-node",
        dest="profile",
        help="Configure a login-node listener that submits GPU work through SLURM.",
    )

    p.add_argument(
        "--listener-mode",
        type=_listener_mode,
        default=None,
        help="Preferred listener mode: gpu-listener, sbatch-oneshot, or sbatch-persistent.",
    )
    p.add_argument(
        "--persistent",
        action="store_true",
        help="Shortcut for --listener-mode sbatch-persistent with --login-node.",
    )

    network = p.add_argument_group("listener network")
    network.add_argument("--host", default=None, help="Bind address for listener profiles.")
    network.add_argument("--port", type=int, default=None, help="TCP port for listener profiles.")
    network.add_argument("--remote-url", default=None, help="Client-facing listener URL.")

    retention = p.add_argument_group("listener retention")
    retention.add_argument("--result-retention", default=None, dest="result_retention")
    retention.add_argument("--memory-retention", default=None, dest="memory_retention")

    sbatch = p.add_argument_group("SLURM settings for --login-node")
    sbatch.add_argument("--account", default=None, help="SLURM account.")
    sbatch.add_argument("--partition", default=None, help="SLURM partition.")
    sbatch.add_argument("--time", default=None, help="Wall-clock limit (HH:MM:SS).")
    sbatch.add_argument("--nodes", default=None, help="Number of nodes.")
    sbatch.add_argument("--ntasks", default=None, help="Number of tasks.")
    sbatch.add_argument("--gpus-per-task", default=None, dest="gpus_per_task")
    sbatch.add_argument("--venv", default=None, help="Virtualenv activated inside GPU jobs.")
    sbatch.add_argument("--log-dir", default=None, dest="log_dir")
    sbatch.add_argument("--output-dir", default=None, dest="output_dir")
    sbatch.add_argument("--extra-sbatch", default=None, dest="extra_sbatch")
    sbatch.add_argument(
        "--extra-srun",
        default=None,
        dest="extra_srun",
        help="Raw extra srun options (e.g. '--container-image=/path/to/image.sqsh').",
    )
    sbatch.add_argument("--job-limit", default=None, type=int, dest="job_limit")
    sbatch.add_argument(
        "--overlay-on-login",
        default=None,
        dest="overlay_on_login",
        action=argparse.BooleanOptionalAction,
        help="Compute weather overlays on the login-node listener instead of as GPU jobs.",
    )

    data = p.add_argument_group("data paths")
    data.add_argument(
        "--weatherbench2-path",
        action="append",
        dest="weatherbench2_paths",
        help="WeatherBench2/ERA5 zarr path. Repeat for multiple stores. Defaults to the hosted public Google Cloud store.",
    )
    return p


def _select_profile(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
) -> str:
    if args.profile:
        return str(args.profile)

    setup = current_section(existing, "setup")
    existing_profile = str(setup.get("deployment", "")).strip()
    default = existing_profile if existing_profile in PROFILE_LABELS else "client"
    if interactive:
        return prompt_choice("Preferred Deployment Mode", SETUP_MODES, default)
    return default


def _select_listener_mode(
    profile: str,
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
) -> str | None:
    if profile == "gpu-listener":
        if args.listener_mode and args.listener_mode != "gpu-listener":
            raise SystemExit("--gpu-listener only supports --listener-mode gpu-listener.")
        if args.persistent:
            raise SystemExit("--persistent only applies to --login-node.")
        return "gpu-listener"

    if profile != "login-node":
        if args.listener_mode:
            raise SystemExit("--listener-mode only applies to listener profiles.")
        if args.persistent:
            raise SystemExit("--persistent only applies to --login-node.")
        return None

    if args.persistent:
        if args.listener_mode and args.listener_mode != "sbatch-persistent":
            raise SystemExit("--persistent conflicts with --listener-mode values other than sbatch-persistent.")
        return "sbatch-persistent"

    if args.listener_mode:
        if not args.listener_mode.startswith("sbatch"):
            raise SystemExit("--login-node requires sbatch-oneshot or sbatch-persistent.")
        return args.listener_mode

    setup = current_section(existing, "setup")
    existing_mode = str(setup.get("mode", "")).strip()
    default = existing_mode if existing_mode.startswith("sbatch") else "sbatch-oneshot"
    if interactive:
        return prompt_choice(
            "Preferred Listener Mode",
            SBATCH_LISTENER_MODES,
            default,
            listener_only=True,
        )
    return default


def _collect_network(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
) -> dict[str, Any]:
    existing_network = current_section(existing, "network")
    host_default = config_default(existing_network, "host", "127.0.0.1")
    port_default = int(config_default(existing_network, "port", 8765))
    if interactive:
        section("Listener Network")
        print("  CLI flags still override these values for a single run.")

    host = args.host
    if host is None:
        host = prompt("Bind host", host_default) if interactive else str(host_default)

    port = args.port
    if port is None:
        port = prompt_int("Port", port_default) if interactive else port_default
    else:
        port = int(port)

    remote_default = config_default(existing_network, "remote_url", f"http://localhost:{port}")
    remote_url = args.remote_url
    if remote_url is None:
        remote_url = prompt("Client remote URL", remote_default) if interactive else str(remote_default)

    return {"host": str(host), "port": int(port), "remote_url": str(remote_url)}


def _collect_data(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
) -> dict[str, Any]:
    existing_data = current_section(existing, "data")
    existing_paths = existing_data.get("weatherbench2_paths") or existing_data.get("wb2_paths")
    if isinstance(existing_paths, list) and existing_paths:
        default_paths = [str(path) for path in existing_paths]
    elif isinstance(existing_paths, str) and existing_paths:
        default_paths = split_interactive_paths(existing_paths)
    else:
        default_paths = list(DEFAULT_WEATHERBENCH2_PATHS)

    def _validate_paths(raw_paths: list[str]) -> list[str]:
        return [_normalize_existing_data_path(path) for path in raw_paths]

    if args.weatherbench2_paths:
        paths = _validate_or_exit(
            lambda: _validate_paths(list(args.weatherbench2_paths)),
            "[data].weatherbench2_paths",
        )
    elif interactive:
        section("Data Paths")
        print("  WeatherBench2/ERA5 zarr stores are tried in order.")
        while True:
            answer = prompt(
                "WeatherBench2 paths (; separated; leave empty for hosted Google Cloud public bucket)",
                "; ".join(default_paths),
            )
            try:
                paths = _validate_paths(split_interactive_paths(answer))
                break
            except ValueError as exc:
                print(f"  {exc}")
                default_paths = []
    else:
        paths = _validate_or_exit(lambda: _validate_paths(default_paths), "[data].weatherbench2_paths")

    return {"weatherbench2_paths": paths}


def _collect_retention(args: argparse.Namespace, existing: dict[str, Any]) -> dict[str, str]:
    existing_retention = current_section(existing, "retention")
    result = args.result_retention
    if result is None:
        result = config_default(existing_retention, "result_retention", DEFAULT_RESULT_RETENTION)
    memory = args.memory_retention
    if memory is None:
        memory = config_default(existing_retention, "memory_retention", DEFAULT_MEMORY_RETENTION)
    return {"result_retention": str(result), "memory_retention": str(memory)}


def _collect_sbatch(
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    listener_mode: str,
    interactive: bool,
) -> dict[str, Any]:
    existing_sbatch = current_section(existing, "sbatch")
    values: dict[str, Any] = {}
    if interactive:
        section("SLURM / srun Parameters")
        print("  These are only for login-node listener modes.")
    for key, label in SBATCH_FIELDS:
        if listener_mode not in SBATCH_FIELD_MODES.get(key, {listener_mode}):
            continue
        arg_value = getattr(args, key, None)
        default = config_default(existing_sbatch, key, sbatch_default(key))
        if key == "gpus_per_task" and interactive:
            print(
                "  Note: RISE / ViT-CX parallelise across this many GPUs "
                "(one model replica each)."
            )
        if arg_value is not None:
            values[key] = _validate_or_exit(
                lambda key=key, arg_value=arg_value: _normalize_sbatch_setup_value(key, str(arg_value)),
                f"[sbatch].{key}",
            )
        elif interactive:
            while True:
                candidate = prompt(label, default)
                try:
                    values[key] = _normalize_sbatch_setup_value(key, candidate)
                    break
                except ValueError as exc:
                    print(f"  {exc}")
                    default = ""
        else:
            values[key] = _validate_or_exit(
                lambda key=key, default=default: _normalize_sbatch_setup_value(key, str(default)),
                f"[sbatch].{key}",
            )
        if key == "gpus_per_task" and interactive and str(values[key]).strip() == "1":
            print("  Warning: gpus_per_task=1 means RISE and ViT-CX run single-GPU.")

    overlay_arg = args.overlay_on_login
    if overlay_arg is not None:
        values["overlay_on_login"] = bool(overlay_arg)
    else:
        existing_overlay = existing_sbatch.get("overlay_on_login")
        if existing_overlay is None:
            values["overlay_on_login"] = bool(sbatch_default("overlay_on_login"))
        elif isinstance(existing_overlay, bool):
            values["overlay_on_login"] = existing_overlay
        else:
            values["overlay_on_login"] = str(existing_overlay).strip().lower() in {"1", "true", "yes", "on"}
    return values


def _validate_or_exit(callback, label: str):
    try:
        return callback()
    except ValueError as exc:
        raise SystemExit(f"Invalid {label}: {exc}") from exc


def _normalize_existing_data_path(value: str) -> str:
    path = str(value).strip()
    if not path or is_remote_uri(path):
        return path
    resolved = Path(normalize_local_path(path))
    if not resolved.exists():
        raise ValueError(f"local data path does not exist: {resolved}")
    return str(resolved.resolve())


def _normalize_existing_directory(value: str, *, label: str) -> str:
    path = str(value).strip()
    if not path:
        return ""
    resolved = Path(normalize_local_path(path))
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {resolved}")
    return str(resolved.resolve())


def _normalize_creatable_directory(value: str, *, label: str) -> str:
    path = str(value).strip()
    if not path:
        return ""
    resolved = Path(normalize_local_path(path))
    parent = resolved if resolved.exists() else resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"{label} parent directory does not exist: {parent}")
    return str(resolved.resolve())


def _normalize_extra_srun(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        normalized = normalize_srun_file_options(raw)
        file_options = list(iter_srun_file_options(normalized))
    except ValueError as exc:
        raise ValueError(f"could not parse extra srun options: {exc}") from exc
    for option, path in file_options:
        if not is_local_path_value(path):
            continue
        resolved = Path(normalize_local_path(path))
        if not resolved.is_file():
            raise ValueError(f"{option} must reference an existing file: {resolved}")
    return normalized


def _normalize_sbatch_setup_value(key: str, value: str) -> str:
    if key in SBATCH_EXISTING_DIR_FIELDS:
        normalized = _normalize_existing_directory(value, label=key)
        if key == "venv" and normalized:
            activate = Path(normalized) / "bin" / "activate"
            if not activate.is_file():
                raise ValueError(f"venv must contain bin/activate: {activate}")
        return normalized
    if key in SBATCH_CREATABLE_DIR_FIELDS:
        return _normalize_creatable_directory(value, label=key)
    if key == "extra_srun":
        return _normalize_extra_srun(value)
    return str(value)


def _build_config(
    profile: str,
    listener_mode: str | None,
    args: argparse.Namespace,
    existing: dict[str, Any],
    *,
    interactive: bool,
) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {
        "setup": {"deployment": profile},
    }
    if listener_mode:
        config["setup"]["mode"] = listener_mode
        config["network"] = _collect_network(args, existing, interactive=interactive)
        config["retention"] = _collect_retention(args, existing)

    if profile in {"local", "gpu-listener", "login-node"}:
        config["data"] = _collect_data(args, existing, interactive=interactive)

    if profile == "login-node":
        assert listener_mode is not None
        config["sbatch"] = _collect_sbatch(
            args,
            existing,
            listener_mode=listener_mode,
            interactive=interactive,
        )

    return config


def _print_install_guide(plan: SetupPlan) -> None:
    section("Install What Where")
    print("  From the adapter source directory, install the extras below in")
    print("  the environment that will do the work:")
    for location, command in INSTALL_COMMANDS[plan.profile]:
        print(f"  - {location}:")
        print(f"      {command}")


def _print_config_guide(plan: SetupPlan) -> None:
    section("Configuration")
    if plan.wrote_config:
        print(f"  Wrote {plan.config_path}")
    elif plan.no_write:
        print(f"  Dry run only. Would write {plan.config_path}")
    else:
        print("  No config was required for this profile.")

    print(f"  Preferred deployment: {plan.profile} ({PROFILE_LABELS[plan.profile]})")
    if plan.listener_mode:
        print(f"  Preferred listener mode: {plan.listener_mode}")
        print("  Override it for one run with: geoxplain-aurora-adapter listen --mode <mode>")
    else:
        print("  No listener mode is configured for this profile.")

    data = plan.config.get("data")
    if data:
        section("Data")
        paths = data.get("weatherbench2_paths") or []
        print("  weatherbench2_paths:")
        if paths:
            for path in paths:
                print(f"    - {path}")
        else:
            print("    (not configured)")

    network = plan.config.get("network")
    if network:
        section("Listener Network")
        print(f"  host: {network['host']}")
        print(f"  port: {network['port']}")
        print(f"  remote_url: {network['remote_url']}")

    sbatch = plan.config.get("sbatch")
    if sbatch:
        section("SLURM")
        for key, _ in SBATCH_FIELDS:
            if key in sbatch:
                print(f"  {key}: {sbatch[key]}")
        print(f"  overlay_on_login: {sbatch.get('overlay_on_login')}")


def _missing_required_config(plan: SetupPlan) -> list[str]:
    missing: list[str] = []
    if plan.profile in {"local", "gpu-listener", "login-node"}:
        data = plan.config.get("data") or {}
        if not data.get("weatherbench2_paths"):
            missing.append("[data].weatherbench2_paths")
    if plan.profile == "login-node":
        sbatch = plan.config.get("sbatch") or {}
        for key in ("account", "partition", "venv"):
            if not str(sbatch.get(key, "")).strip():
                missing.append(f"[sbatch].{key}")
    return missing


def _print_next_steps(plan: SetupPlan) -> None:
    missing = _missing_required_config(plan)
    if missing:
        section("Required Config Still Missing")
        print("  If not installed yet, please follow the guide above.")
        for item in missing:
            print(f"  - {item}")
        print("  Re-run setup with the missing flags or edit the config before starting a listener.")
        return
    section("Next Steps")
    if plan.profile == "client":
        print("  If not installed yet, please follow the guide above.")
        print("  Use remote='http://...' in notebook calls to connect to an existing listener.")
        return
    if plan.profile == "local":
        print("  If not installed yet, please follow the guide above.")
        print("  Run notebooks inside the GPU environment and call the API without remote=.")
        print("  Required config: WeatherBench2/ERA5 zarr paths in the [data] section.")
        return
    print("  If not installed yet, please follow the guide above.")
    print("  Start the listener with: geoxplain-aurora-adapter listen")
    print("  Re-run this guide any time with: geoxplain-aurora-adapter setup")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)
    existing = {} if args.reset else load_config(config_path)
    interactive = bool(sys.stdin.isatty() and not args.yes)

    if interactive:
        headline("GeoXplain Aurora Adapter Setup")
        print("  This setup writes adapter config and prints install guidance only.")

    profile = _select_profile(args, existing, interactive=interactive)
    listener_mode = _select_listener_mode(profile, args, existing, interactive=interactive)
    config = _build_config(profile, listener_mode, args, existing, interactive=interactive)

    wrote_config = False
    if config and not args.no_write:
        write_config(config_path, config)
        wrote_config = True

    plan = SetupPlan(
        profile=profile,
        listener_mode=listener_mode,
        config_path=config_path,
        config=config,
        wrote_config=wrote_config,
        no_write=bool(args.no_write),
    )

    if not interactive:
        headline("GeoXplain Aurora Adapter Setup")
    _print_config_guide(plan)
    _print_install_guide(plan)
    _print_next_steps(plan)


if __name__ == "__main__":
    main()
