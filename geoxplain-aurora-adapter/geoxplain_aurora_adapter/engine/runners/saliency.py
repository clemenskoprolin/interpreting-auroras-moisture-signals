"""Saliency runner — vanilla gradient attribution for Aurora.

Includes the per-block progress instrumentation that turns the single
forward+backward pass into a real percentage bar.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

import numpy as np
import torch

from .._common import _split_atmos_levels
from ..data import CaseData, make_batch
from ..progress import ProgressReporter
from ..rollout import _RolloutForwardWrapper, _saved_tensors_cpu_context


def _swin_blocks(model) -> list:
    """Return the Swin3D transformer blocks that dominate Aurora's runtime.

    These are the units we count to turn the otherwise-opaque single saliency
    forward+backward pass into a real percentage.  Returns ``[]`` if the model
    is not shaped as expected, so callers can fall back gracefully.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return []
    blocks: list = []
    for layer in list(getattr(backbone, "encoder_layers", [])) + list(
        getattr(backbone, "decoder_layers", [])
    ):
        blocks.extend(list(getattr(layer, "blocks", [])))
    return blocks


@contextlib.contextmanager
def _saliency_block_progress(model, reporter, *, frame_units: float = 1.0, forward_weight: float = 0.4):
    """Drive a real per-block percentage across one saliency forward+backward.

    Each Swin3D block runs once in the forward pass (forward hook) and its
    output tensor receives one gradient in the backward pass (tensor hook), so
    counting those fires gives a smooth ``0 → 100%`` bar over the single pass.
    Forward gets ``forward_weight`` of the budget, backward (heavier, since
    reentrant checkpointing recomputes activations) gets the rest.

    The hooks only ever read and count — they never mutate tensors or
    gradients, and every callback body swallows its own exceptions so a hook
    can never corrupt the computation.  On exit any unfilled remainder is
    topped up so the frame always contributes exactly ``frame_units``,
    regardless of how many hooks actually fired.
    """
    blocks = _swin_blocks(model)
    if reporter is None or not blocks:
        yield
        return

    n = len(blocks)
    fwd_step = frame_units * forward_weight / n
    bwd_step = frame_units * (1.0 - forward_weight) / n
    state = {"fwd": 0, "bwd": 0, "advanced": 0.0}
    lock = threading.Lock()
    handles: list = []

    def _count_backward(_grad):
        try:
            with lock:
                if state["bwd"] >= n:
                    return None
                state["bwd"] += 1
                done = state["bwd"]
                state["advanced"] += bwd_step
            reporter.advance(bwd_step, phase="saliency", detail=f"backward {done}/{n}")
        except Exception:
            pass
        return None

    def _on_forward(_module, _inp, out):
        try:
            tensor = out[0] if isinstance(out, (tuple, list)) and out else out
            with lock:
                if state["fwd"] >= n:
                    return
                state["fwd"] += 1
                done = state["fwd"]
                state["advanced"] += fwd_step
            reporter.advance(fwd_step, phase="saliency", detail=f"forward {done}/{n}")
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(_count_backward)
        except Exception:
            pass

    for blk in blocks:
        try:
            handles.append(blk.register_forward_hook(_on_forward))
        except Exception:
            pass
    try:
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        try:
            with lock:
                remainder = frame_units - state["advanced"]
            if remainder > 1e-9:
                reporter.advance(
                    remainder, phase="saliency", detail="gradients ready", force=True
                )
        except Exception:
            pass


def _run_saliency(
    case: CaseData,
    target_fn,
    atmos_vars: list[str],
    surf_vars: list[str],
    model,
    device: str,
    rollout_steps: int = 1,
    levels: Optional[list[int]] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> tuple[dict[str, dict[str, np.ndarray]], Optional[float]]:
    from ..xia_methods.saliency import saliency

    keep_levels = set(levels) if levels else None

    def batch_fn(requires_grad: bool = False):
        return make_batch(
            case, device,
            requires_grad_atmos=tuple(atmos_vars) if requires_grad else (),
            requires_grad_surf=tuple(surf_vars) if requires_grad else (),
        )

    if progress_reporter is not None:
        progress_reporter.set_phase("saliency", "forward/backward")
    model_fwd = _RolloutForwardWrapper(model, rollout_steps) if rollout_steps > 1 else model

    # Per-block percentage is only well-defined for a single forward+backward;
    # an N-step rollout runs the backbone N times, so fall back to the coarse
    # (heartbeat-refreshed) bar there.
    track_blocks = progress_reporter is not None and rollout_steps == 1
    block_progress = (
        _saliency_block_progress(model_fwd, progress_reporter)
        if track_blocks
        else contextlib.nullcontext()
    )
    with _saved_tensors_cpu_context(rollout_steps > 1), block_progress:
        result = saliency(
            model=model_fwd,
            batch_fn=batch_fn,
            target_fn=target_fn,
            device=device,
            atmos_var_names=tuple(atmos_vars),
            surf_var_names=tuple(surf_vars),
        )
    if progress_reporter is not None and not track_blocks:
        progress_reporter.advance(1, phase="saliency", detail="gradients ready", force=True)

    attributions: dict[str, dict[str, np.ndarray]] = {}
    for v in atmos_vars:
        g = result["grads"].get(v)
        if g is not None:
            attributions[v] = _split_atmos_levels(g, case.pressure_levels, keep_levels)
    for v in surf_vars:
        g = result["grads"].get(v)
        if g is not None:
            attributions[v] = {"sfc": g[0, 1].astype(np.float32)}

    return attributions, result.get("score")
