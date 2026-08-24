"""Aurora model construction and checkpoint loading for geoxplain_aurora_adapter.

This loads the **public Microsoft Aurora** model, using the 6 h-lead
``AuroraPretrained`` variant.  

The model includes:
    - surf_vars  = ("2t", "10u", "10v", "msl")
    - atmos_vars = ("z", "u", "v", "t", "q")
    - static_vars = ("lsm", "z", "slt")
    - built-in normalisation (no ``locations`` / ``scales`` overrides)

Weights are downloaded automatically by ``model.load_checkpoint()`` (the
HuggingFace ``microsoft/aurora`` pretrained checkpoint bundled with the model
class).

Gradient checkpointing wraps every ``Swin3DTransformerBlock`` in both the
encoder and decoder layers, reducing RAM requirements.
Forward-only methods (RISE, ViT-CX) run the wrapped blocks
eagerly under ``torch.no_grad()``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def load_model(
    device: str | torch.device,
    checkpoint_path: Optional[str] = None,
) -> "AuroraPretrained":  # noqa: F821
    """Construct and return the public Microsoft Aurora (6 h pretrained) model.

    Parameters
    ----------
    device:
        Target device, e.g. ``"cuda"`` or ``"cpu"``.
    checkpoint_path:
        Optional path to a *local* checkpoint file.  When ``None`` (the
        default) the public pretrained weights are downloaded via
        ``model.load_checkpoint()``.  When given, the weights are loaded from
        that local file via ``model.load_checkpoint_local(...)``.

    Returns
    -------
    An ``AuroraPretrained`` instance in ``eval`` mode with:
    - All parameters frozen (``requires_grad=False``).
    - ``torch.utils.checkpoint`` wrapping every ``Swin3DTransformerBlock``.
    """
    from aurora import AuroraPretrained  # type: ignore[import]

    model = AuroraPretrained()

    if checkpoint_path:
        print(f"Loading local checkpoint: {checkpoint_path}")
        model.load_checkpoint_local(checkpoint_path)
    else:
        print("Loading public pretrained Aurora checkpoint (microsoft/aurora)")
        model.load_checkpoint()

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    _wrap_swin_grad_checkpoint(model)

    print("Public Aurora (6 h pretrained) loaded, frozen, grad-checkpointed.")
    return model


def _wrap_swin_grad_checkpoint(model) -> None:
    """Wrap every ``Swin3DTransformerBlock`` with activation checkpointing.

    ``use_reentrant=False`` matches the working base-model experiments and is the
    recommended non-reentrant variant, which correctly saves/restores autocast
    state across the backward recomputation.  Forward-only methods run the
    wrapped blocks eagerly under ``torch.no_grad()``, where checkpointing is a
    numerical no-op, so this is safe to apply to inference-only replicas too.
    """
    for layer in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        for blk in layer.blocks:
            orig = blk.forward
            def _make_cp(fn):
                def _cp(*a, **kw):
                    return torch_checkpoint(fn, *a, use_reentrant=False, **kw)
                return _cp
            blk.forward = _make_cp(orig)


def replicate_model(primary, device: str | torch.device) -> "AuroraPretrained":  # noqa: F821
    """Build a weight-identical Aurora replica of ``primary`` on ``device``.

    Used to fan forward-only attribution methods (RISE, ViT-CX) across the GPUs
    of a node, one replica per device.  The replica copies ``primary``'s exact
    weights via ``load_state_dict`` (no checkpoint download, no random init left
    in place), is frozen, set to ``eval``, and grad-checkpoint-wrapped exactly
    like the primary.  On identical GPUs a batch-1 forward through the replica is
    bit-identical to the same forward through ``primary``, so attributions are
    unchanged.
    """
    from aurora import AuroraPretrained  # type: ignore[import]

    replica = AuroraPretrained()
    replica.load_state_dict(primary.state_dict())
    replica.to(device)
    replica.eval()
    for p in replica.parameters():
        p.requires_grad_(False)
    _wrap_swin_grad_checkpoint(replica)
    return replica


def forward(model, batch) -> "Batch":  # noqa: F821
    """Call ``model.forward`` and unwrap the ``(pred, std, preds)`` tuple."""
    out = model.forward(batch)
    if isinstance(out, tuple):
        out = out[0]
    return out
