"""Autoregressive rollout XIA for geoxplain_aurora_adapter.

Two things live here:

1. The **forward-pass machinery** that turns an Aurora model into one whose
   ``forward(batch)`` returns the final N-step rollout prediction
   (``_RolloutForwardWrapper``), plus the gradient-checkpointing helpers that
   keep the deep rollout graph within GPU memory.  ``compute._run_saliency`` /
   ``compute._run_ig`` wrap the model with these when ``rollout_steps > 1``.

2. ``_run_local_rollout`` — the orchestrator that runs a gradient method
   (saliency or IG) at increasing lead times from one initial condition and
   returns a single multi-frame ``XiaResult``.
"""

from __future__ import annotations

import socket
import time
from dataclasses import replace
from datetime import timedelta
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ._common import (
    _detect_diverging,
    _estimate_total_units,
    _format_timestamp,
    _gpu_sync_and_gc,
    _layer_labels_for,
    _parse_init_time,
    _split_input_vars,
)
from .data import load_case
from ..schema.metadata import method_display_name
from .progress import ProgressReporter
from ..schema.result import XiaFrame, XiaResult
from ..schema.spec import TargetSpec
from ..schema.targets import build_target_fn


# ── Forward-pass machinery ────────────────────────────────────────────────────

def _forward_unwrapped(model, batch):
    out = model.forward(batch)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _rollout_mean(model, batch, steps: int):
    """Yield mean predictions from an autoregressive Aurora rollout."""
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if steps == 0:
        return

    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if getattr(model, "use_resolution_specific_patch_tokenizers", False):
        patch_size = model.patch_tokenizer_identifier.get_patch_size(
            batch.metadata.grid_resolution
        )
    else:
        patch_size = model.patch_size
    batch = batch.crop(patch_size=patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = _forward_unwrapped(model, batch)
        yield pred
        batch = replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )


def _enable_decoder_rollout_checkpointing(model) -> None:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return
    level_decoder = getattr(decoder, "level_decoder", None)
    if level_decoder is None or getattr(level_decoder, "_rollout_checkpoint_wrapped", False):
        return

    orig_forward = level_decoder.forward

    def _checkpointed_forward(latents, x):
        if torch.is_grad_enabled() and (
            getattr(latents, "requires_grad", False)
            or getattr(x, "requires_grad", False)
        ):
            return torch_checkpoint(orig_forward, latents, x, use_reentrant=True)
        return orig_forward(latents, x)

    level_decoder.forward = _checkpointed_forward
    level_decoder._rollout_checkpoint_wrapped = True


def _enable_encoder_rollout_checkpointing(model) -> None:
    for enc in [getattr(model, "encoder", None)] + list(getattr(model, "encoders", [])):
        if enc is None or not hasattr(enc, "level_agg"):
            continue
        level_agg = enc.level_agg
        if getattr(level_agg, "_rollout_checkpoint_wrapped", False):
            continue
        orig_forward = level_agg.forward

        def _make_level_agg_cp(fn):
            def _checkpointed(*a, **kw):
                if not torch.is_grad_enabled():
                    return fn(*a, **kw)
                if kw:
                    return torch_checkpoint(
                        lambda *args: fn(*args, **kw),
                        *a,
                        use_reentrant=True,
                    )
                return torch_checkpoint(fn, *a, use_reentrant=True)
            return _checkpointed

        level_agg.forward = _make_level_agg_cp(orig_forward)
        level_agg._rollout_checkpoint_wrapped = True


def _enable_rollout_checkpointing(model) -> None:
    _enable_decoder_rollout_checkpointing(model)
    _enable_encoder_rollout_checkpointing(model)


def _saved_tensors_cpu_context(enabled: bool):
    from contextlib import nullcontext

    if not enabled:
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    if graph_mod is None or not hasattr(graph_mod, "save_on_cpu"):
        return nullcontext()
    save_on_cpu = graph_mod.save_on_cpu
    try:
        return save_on_cpu(pin_memory=torch.cuda.is_available())
    except TypeError:
        return save_on_cpu()


class _RolloutForwardWrapper:
    """Make ``model.forward(batch)`` return the final N-step rollout prediction."""

    def __init__(self, model, steps: int):
        if steps < 1:
            raise ValueError(f"rollout steps must be >= 1, got {steps}")
        _enable_rollout_checkpointing(model)
        object.__setattr__(self, "_wrapped_model", model)
        object.__setattr__(self, "_rollout_steps", int(steps))

    def forward(self, batch):
        model = object.__getattribute__(self, "_wrapped_model")
        steps = object.__getattribute__(self, "_rollout_steps")
        if steps == 1:
            return _forward_unwrapped(model, batch)
        final = None
        for pred in _rollout_mean(model, batch, steps):
            final = pred
        if final is None:
            raise RuntimeError(f"_rollout_mean yielded no predictions for steps={steps}")
        return final

    def __call__(self, batch):
        return self.forward(batch)

    def __getattr__(self, name):
        model = object.__getattribute__(self, "_wrapped_model")
        return getattr(model, name)

    def __setattr__(self, name, value):
        model = object.__getattribute__(self, "_wrapped_model")
        setattr(model, name, value)


# ── Rollout orchestrator ──────────────────────────────────────────────────────

def _run_local_rollout(
    method: str,
    target: TargetSpec,
    input_vars: list[str],
    model,
    device: str,
    *,
    timeframes: int,
    progress_reporter: Optional[ProgressReporter] = None,
    finish_progress: bool = True,
    **options,
) -> XiaResult:
    """Run autoregressive rollout XIA from one initial condition."""
    # Imported lazily to avoid an import cycle: the runners import the forward
    # machinery (``_RolloutForwardWrapper``) defined above in this module.
    from .runners import _run_saliency, _run_ig

    if method not in {"saliency", "ig"}:
        raise NotImplementedError(
            f"run_rollout(method={method!r}) is only wired for 'saliency' and 'ig'."
        )
    if not isinstance(timeframes, int) or timeframes < 1:
        raise ValueError(f"timeframes must be a positive integer, got {timeframes!r}")

    t0 = time.time()
    atmos_vars, surf_vars = _split_input_vars(input_vars)
    n_vars = len(atmos_vars) + len(surf_vars)

    if progress_reporter is None:
        progress_reporter = ProgressReporter(
            f"{method} rollout",
            _estimate_total_units(
                method,
                n_frames=timeframes,
                n_vars=n_vars,
                options=options,
            ),
            total_frames=timeframes,
            print_updates=True,
        )
    else:
        estimated_total = _estimate_total_units(
            method,
            n_frames=timeframes,
            n_vars=n_vars,
            options=options,
        )
        if progress_reporter.total_units is None and estimated_total is not None:
            progress_reporter.set_total(estimated_total, emit=False)

    progress_reporter.set_phase("reading data")
    init_time = _parse_init_time(target.timestamp)
    case = load_case(init_time)
    target_fn = build_target_fn(target, case)

    frames: list[XiaFrame] = []
    layer_labels: dict[str, str] = {}
    per_frame_runtime = 0.0

    for idx in range(timeframes):
        rollout_steps = idx + 1
        lead_hours = 6 * rollout_steps
        progress_reporter.set_frame(idx + 1)
        frame_t0 = time.time()

        if method == "saliency":
            attributions, score = _run_saliency(
                case,
                target_fn,
                atmos_vars,
                surf_vars,
                model,
                device,
                rollout_steps=rollout_steps,
                progress_reporter=progress_reporter,
            )
        else:
            attributions, score = _run_ig(
                case,
                target_fn,
                atmos_vars,
                surf_vars,
                model,
                device,
                rollout_steps=rollout_steps,
                progress_reporter=progress_reporter,
                **options,
            )

        frame_runtime = time.time() - frame_t0
        per_frame_runtime += frame_runtime
        frame_target = replace(
            target,
            timestamp=_format_timestamp(init_time + timedelta(hours=lead_hours)),
        )
        frame_meta = {
            "rollout_index": idx,
            "rollout_step": rollout_steps,
            "lead_hours": lead_hours,
            "rollout_start_timestamp": target.timestamp,
            "runtime_s": round(frame_runtime, 2),
        }
        if score is not None:
            frame_meta["target_score"] = float(score)

        frames.append(
            XiaFrame(
                target=frame_target,
                timestamp=frame_target.timestamp,
                attributions=attributions,
                diverging=_detect_diverging(attributions),
                meta=frame_meta,
            )
        )
        layer_labels.update(_layer_labels_for(case, attributions, method))
        _gpu_sync_and_gc()

    runtime_s = time.time() - t0
    meta = {
        "rollout": True,
        "timeframes": timeframes,
        "step_hours": 6,
        "start_timestamp": frames[0].timestamp,
        "end_timestamp": frames[-1].timestamp,
        "input_timestamp": target.timestamp,
        "runtime_s": round(runtime_s, 2),
        "frame_runtime_s": round(per_frame_runtime, 2),
        "host": socket.gethostname(),
        "method_options": options,
    }

    if finish_progress:
        progress_reporter.finish("done")

    return XiaResult(
        method=method,
        method_label=method_display_name(method),
        frames=frames,
        layer_labels=layer_labels,
        meta=meta,
    )
