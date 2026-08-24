"""Dataclasses for the remote listener's JSON request/status messages.

The final result endpoint returns msgpack bytes produced by ``XiaResult`` or
``OverlayResult``; polling and health checks stay JSON-only.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


ACTIVE_LOG_TAIL_CHARS = 200
ERROR_LOG_TAIL_CHARS = 4000


def log_tail_for_status(log: str, status: str) -> str:
    """Keep failures actionable without flooding normal progress responses."""
    limit = ERROR_LOG_TAIL_CHARS if status == "error" else ACTIVE_LOG_TAIL_CHARS
    return str(log)[-limit:]


def new_job_id() -> str:
    return str(uuid.uuid4())


@dataclass
class RunRequest:
    method: str
    target: dict
    input_vars: list[str]
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRequest":
        return cls(
            method=d["method"],
            target=d["target"],
            input_vars=d["input_vars"],
            options=d.get("options", {}),
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "target": self.target,
            "input_vars": self.input_vars,
            "options": self.options,
        }


@dataclass
class BatchRunRequest:
    method: str
    targets: list[dict]
    input_vars: list[str]
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "BatchRunRequest":
        return cls(
            method=d["method"],
            targets=d["targets"],
            input_vars=d["input_vars"],
            options=d.get("options", {}),
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "targets": self.targets,
            "input_vars": self.input_vars,
            "options": self.options,
        }


@dataclass
class OverlayRequest:
    variable: str
    timestamps: list[str]
    level: Optional[int] = None
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayRequest":
        return cls(
            variable=d["variable"],
            timestamps=d["timestamps"],
            level=d.get("level"),
            options=d.get("options", {}),
        )

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "timestamps": self.timestamps,
            "level": self.level,
            "options": self.options,
        }


@dataclass
class JobStatus:
    job_id: str
    status: str           # "queued" | "queued_locally" | "running" | "completing" | "done" | "error"
                          # "completing" is a SLURM-only transition state (CG):
                          # the job has left RUNNING but the result file may
                          # not be flushed yet.  Treat as still-in-flight.
    eta_s: Optional[float] = None
    progress: Optional[float] = None
    text_output: Optional[str] = None
    log_tail: str = ""
    result_url: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "eta_s": self.eta_s,
            "progress": self.progress,
            "text_output": self.text_output,
            "log_tail": self.log_tail,
        }
        if self.result_url:
            d["result_url"] = self.result_url
        if self.error_message:
            d["error_message"] = self.error_message
        return d


@dataclass
class HealthResponse:
    mode: str
    model_warm: bool
    queue_depth: int
    sbatch_config: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "mode": self.mode,
            "model_warm": self.model_warm,
            "queue_depth": self.queue_depth,
        }
        if self.sbatch_config:
            d["sbatch_config"] = self.sbatch_config
        return d
