"""Resolved sbatch attribute configuration for geoxplain_aurora_adapter.

Precedence (highest to lowest):
    1. CLI flag (passed to SbatchConfig constructor).
    2. Environment variable ``GEOXPLAIN_AURORA_ADAPTER_SBATCH_<UPPER_ATTR>``.
    3. ``~/.config/geoxplain-aurora-adapter/listen.toml`` [sbatch] section.
    4. Generic built-in defaults.

The resolved config is echoed by ``GET /health`` so clients can sanity-check
what the listener will actually submit.

Extra pass-through
------------------
``extra_sbatch`` is a raw string appended verbatim to the ``#SBATCH``
preamble.  Use it for options that are not modelled here, e.g.::

    --extra-sbatch "--qos=preempt --mem=200G"

``extra_srun`` is appended to the ``srun -ul`` command.  Use it for
site-specific launch options, e.g. container/runtime environment flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..serving.config import DEFAULT_CONFIG_PATH, default_output_dir, get_config_section
from ..serving.path_options import normalize_local_path, normalize_srun_file_options


_ENV_PREFIX = "GEOXPLAIN_AURORA_ADAPTER_SBATCH_"
_TOML_PATH = str(DEFAULT_CONFIG_PATH)

# Generic built-in defaults. Site-specific values should come from setup config,
# CLI flags, or GEOXPLAIN_AURORA_ADAPTER_SBATCH_* environment variables.
_DEFAULTS = {
    "account": "",
    "partition": "",
    "time": "00:30:00",
    "nodes": "1",
    "ntasks": "1",
    "gpus_per_task": "1",
    "output": "logs/slurm-%x-%j.log",
    "venv": "",
    "log_dir": "logs/",
    "extra_sbatch": "",
    "extra_srun": "",
    "job_limit": "2",
    "overlay_on_login": "true",
}

# sbatch-oneshot concurrency limit used when none is configured.
_DEFAULT_JOB_LIMIT = 2


def _as_bool(value, default: bool = True) -> bool:
    """Coerce a config value (bool or string) to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _load_toml_section() -> dict:
    """Load [sbatch] section from ``~/.config/geoxplain-aurora-adapter/listen.toml``."""
    return get_config_section("sbatch")


@dataclass
class SbatchConfig:
    """Resolved sbatch submission attributes.

    All fields default to ``None``, meaning "use the built-in default".
    Pass explicit values from CLI flags or environment variables.
    """
    account: Optional[str] = None
    partition: Optional[str] = None
    time: Optional[str] = None
    nodes: Optional[str] = None
    ntasks: Optional[str] = None
    gpus_per_task: Optional[str] = None
    output: Optional[str] = None
    venv: Optional[str] = None
    log_dir: Optional[str] = None
    output_dir: Optional[str] = None
    extra_sbatch: Optional[str] = None
    extra_srun: Optional[str] = None
    job_limit: Optional[str] = None
    # Compute overlay pulls in the login-node listener instead of on a GPU job.
    overlay_on_login: Optional[bool] = None

    def resolve(self) -> "ResolvedSbatchConfig":
        """Return a ``ResolvedSbatchConfig`` applying the full precedence chain."""
        toml_section = _load_toml_section()

        def _get(attr: str) -> str:
            # 1. Explicit constructor arg
            v = getattr(self, attr)
            if v is not None:
                return str(v)
            # 2. Environment variable (use _ prefix form)
            env_key = _ENV_PREFIX + attr.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return env_val
            # 3. TOML config
            toml_key = attr.replace("_", "-")
            if toml_key in toml_section:
                return str(toml_section[toml_key])
            if attr in toml_section:
                return str(toml_section[attr])
            # 4. Built-in default
            return _DEFAULTS.get(attr, "")

        def _abspath(val: str) -> str:
            if not str(val).strip():
                return ""
            return normalize_local_path(val)

        def _positive_int(val: str) -> int:
            try:
                parsed = int(val)
            except (TypeError, ValueError):
                return _DEFAULT_JOB_LIMIT
            return parsed if parsed >= 1 else _DEFAULT_JOB_LIMIT

        def _get_bool(attr: str, default: bool) -> bool:
            # Same precedence chain as ``_get`` but coerced to bool.
            v = getattr(self, attr)
            if v is not None:
                return _as_bool(v, default)
            env_val = os.environ.get(_ENV_PREFIX + attr.upper())
            if env_val is not None:
                return _as_bool(env_val, default)
            toml_key = attr.replace("_", "-")
            if toml_key in toml_section:
                return _as_bool(toml_section[toml_key], default)
            if attr in toml_section:
                return _as_bool(toml_section[attr], default)
            return _as_bool(_DEFAULTS.get(attr), default)

        return ResolvedSbatchConfig(
            account=_get("account"),
            partition=_get("partition"),
            time=_get("time"),
            nodes=_get("nodes"),
            ntasks=_get("ntasks"),
            gpus_per_task=_get("gpus_per_task"),
            output=_get("output"),
            venv=_abspath(_get("venv")),
            log_dir=_abspath(_get("log_dir")),
            output_dir=_abspath(_get("output_dir") or default_output_dir()),
            extra_sbatch=_get("extra_sbatch"),
            extra_srun=normalize_srun_file_options(_get("extra_srun")),
            job_limit=_positive_int(_get("job_limit")),
            overlay_on_login=_get_bool("overlay_on_login", True),
        )


@dataclass
class ResolvedSbatchConfig:
    """Fully resolved sbatch configuration (no None fields)."""
    account: str
    partition: str
    time: str
    nodes: str
    ntasks: str
    gpus_per_task: str
    output: str
    venv: str
    log_dir: str
    output_dir: str
    extra_sbatch: str
    extra_srun: str
    job_limit: int
    overlay_on_login: bool

    def to_dict(self) -> dict:
        return {
            "account": self.account,
            "partition": self.partition,
            "time": self.time,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "gpus_per_task": self.gpus_per_task,
            "output": self.output,
            "venv": self.venv,
            "log_dir": self.log_dir,
            "output_dir": self.output_dir,
            "extra_sbatch": self.extra_sbatch,
            "extra_srun": self.extra_srun,
            "job_limit": self.job_limit,
            "overlay_on_login": self.overlay_on_login,
        }

    def preamble_lines(self) -> list[str]:
        """Return the ``#SBATCH`` directive lines for the submit script."""
        lines = [
            f"#SBATCH --job-name=geoxplain-aurora-adapter",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.ntasks}",
            f"#SBATCH --gpus-per-task={self.gpus_per_task}",
            f"#SBATCH --output={self.output}",
        ]
        if self.extra_sbatch:
            for token in self.extra_sbatch.split():
                lines.append(f"#SBATCH {token}")
        return lines

    def render_submit_script(self, body: str) -> str:
        """Render a complete sbatch submit script.

        ``body`` is the shell commands to run after ``srun``.
        """
        preamble = "\n".join(self.preamble_lines())
        os.makedirs(os.path.expanduser(self.log_dir), exist_ok=True)
        srun_args = f" {self.extra_srun}" if self.extra_srun else ""
        script = (
            "#!/bin/bash\n"
            f"{preamble}\n\n"
            f"srun -ul{srun_args} bash -c \"\n"
            f"    source {self.venv}/bin/activate\n"
            f"    {body}\n"
            f"\"\n"
        )
        return script
