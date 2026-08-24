"""Shared configuration helpers for geoxplain_aurora_adapter."""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


APP_NAME = "geoxplain-aurora-adapter"
CONFIG_ENV = "GEOXPLAIN_AURORA_ADAPTER_CONFIG"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / APP_NAME
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "listen.toml"


def default_output_dir() -> str:
    """Base directory for per-job result files (default ``$SCRATCH/<app>``).

    sbatch backends write each job's request/result/status here on shared
    storage so a worker on a compute node can read what the listener wrote.
    Falls back to the system temp dir when ``$SCRATCH`` is unset.
    """
    scratch = os.environ.get("SCRATCH", tempfile.gettempdir())
    return os.path.join(scratch, APP_NAME)

PUBLIC_WEATHERBENCH2_BUCKET = "gs://weatherbench2/datasets"
PUBLIC_WEATHERBENCH2_ERA5_PATH = (
    f"{PUBLIC_WEATHERBENCH2_BUCKET}/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
_PUBLIC_WEATHERBENCH2_ALIASES = {
    PUBLIC_WEATHERBENCH2_BUCKET,
    f"{PUBLIC_WEATHERBENCH2_BUCKET}/era5",
}

DEFAULT_WEATHERBENCH2_PATHS: tuple[str, ...] = (PUBLIC_WEATHERBENCH2_ERA5_PATH,)


def normalize_weatherbench2_path(value: str) -> str:
    """Expand public WeatherBench2 bucket aliases to the ERA5 Zarr store."""
    path = str(value).strip()
    return PUBLIC_WEATHERBENCH2_ERA5_PATH if path.rstrip("/") in _PUBLIC_WEATHERBENCH2_ALIASES else path


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the active config path, honoring the test/user override env var."""
    chosen = path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    return Path(chosen).expanduser()


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the adapter TOML config.

    Missing or unreadable config files are treated as empty config so runtime
    defaults still work before setup has been run.
    """
    config_path = resolve_config_path(path)
    if not config_path.exists():
        return {}
    raw = config_path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import,no-redef]
        except ImportError:
            return _parse_simple_toml(raw)
    try:
        from io import BytesIO

        data = tomllib.load(BytesIO(raw.encode("utf-8")))
    except Exception:
        return _parse_simple_toml(raw)
    return data if isinstance(data, dict) else {}


def _parse_simple_toml(raw: str) -> dict[str, Any]:
    """Parse the simple section/key TOML emitted by the setup command.

    This keeps Python 3.10 installs usable even before optional dependency
    resolution has installed tomli. It is intentionally not a full TOML parser.
    """
    data: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    pending_key: str | None = None
    pending_values: list[Any] = []

    for original in raw.splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if pending_key is not None:
            if line == "]":
                if current is not None:
                    current[pending_key] = pending_values
                pending_key = None
                pending_values = []
                continue
            pending_values.append(_parse_simple_value(line.rstrip(",")))
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = data.setdefault(section, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value == "[":
            pending_key = key
            pending_values = []
        else:
            current[key] = _parse_simple_value(value)
    return data


def _parse_simple_value(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if value.startswith('"') and value.endswith('"'):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def get_config_section(
    name: str,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return one top-level config section as a normal dict."""
    section = load_config(path).get(name, {})
    return dict(section) if isinstance(section, Mapping) else {}


# How long sbatch-oneshot on-disk result directories are kept before a janitor
# thread purges them.  ``"never"`` (the default) keeps them forever.
DEFAULT_RESULT_RETENTION = "never"

# How long completed jobs are kept in the listener's memory (job records and
# cached result bytes).
DEFAULT_MEMORY_RETENTION = "1h"

_RETENTION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_retention(value: Any) -> float | None:
    """Parse a retention spec into seconds, or ``None`` to keep forever.

    Accepts ``"never"``/``"none"``/``""``/``"0"`` (→ ``None``), a bare number of
    seconds (``"3600"``, ``3600``), or a ``<number><unit>`` string where unit is
    one of ``s``/``m``/``h``/``d``/``w`` (e.g. ``"24h"``, ``"7d"``, ``"30d"``).
    Unparseable or non-positive values fall back to ``None`` (keep forever).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip().lower()
    if text in ("", "never", "none", "off", "0"):
        return None
    unit = text[-1]
    if unit in _RETENTION_UNITS:
        number = text[:-1].strip()
        mult = _RETENTION_UNITS[unit]
    else:
        number = text
        mult = 1
    try:
        seconds = float(number) * mult
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = ",\n".join(f"    {_toml_value(str(item))}" for item in value)
        return "[\n" + items + ",\n]"
    return _toml_string(str(value))


def render_config(config: Mapping[str, Mapping[str, Any]]) -> str:
    """Render the adapter config as TOML without requiring a writer dependency."""
    lines = [
        "# Generated by geoxplain-aurora-adapter.",
        "# CLI flags and environment variables still override these values.",
        "",
    ]
    for section_name in ("setup", "network", "retention", "sbatch", "data"):
        section = config.get(section_name)
        if not section:
            continue
        lines.append(f"[{section_name}]")
        for key, value in section.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def write_config(path: str | os.PathLike[str], config: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write adapter config TOML and return the resolved path."""
    config_path = resolve_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_config(config), encoding="utf-8")
    return config_path
