"""
Vanilla Gradient Saliency — general implementation.

Computes dy/dx for a scalar target y w.r.t. all atmospheric input
variables, via a single forward + backward pass.
"""

import gc
import numpy as np
import torch


def saliency(model, batch_fn, target_fn, device,
             atmos_var_names=(), surf_var_names=()):
    """Vanilla gradient saliency: dy/dx for a target scalar y.

    Args:
        model:
            Aurora model in eval mode with gradient checkpointing already
            enabled and all parameters frozen (requires_grad=False).

        batch_fn:
            callable(requires_grad: bool) -> aurora.Batch
            When requires_grad=True, the returned batch's tensors for every
            name in atmos_var_names / surf_var_names must be *leaf* tensors
            on `device` with requires_grad=True.

        target_fn:
            callable(pred: aurora.Batch) -> scalar torch.Tensor
            Extracts the attribution target from the model prediction.
            Must be differentiable w.r.t. the input batch.

        device:
            torch device string, e.g. "cuda" or "cpu".

        atmos_var_names:
            Names of atmospheric variables (keys in batch.atmos_vars) for
            which gradients are requested.  Defaults to empty — pass a list
            when targeting an atmospheric output variable is appropriate, but
            note that self-attribution (same variable in input and target)
            collapses to a near-delta-function via the Perceiver decoder.

        surf_var_names:
            Names of surface variables (keys in batch.surf_vars) for which
            gradients are requested.  Prefer this when the target is an
            atmospheric variable: surface inputs bypass the Perceiver, so
            d(atmos_out)/d(surf_in) is spatially distributed and meaningful.

    Returns:
        dict with:
            "grads": dict[str -> np.ndarray | None]
                Signed gradients dy/dx per variable (atmos and surf combined).
                Atmospheric shape: (1, 2, n_levels, H, W).
                Surface shape:     (1, 2, H, W).
                None if the variable produced no gradient.
            "score": float
                Value of target_fn at the actual input.
    """
    batch = batch_fn(requires_grad=True)

    # Keep Aurora's own backbone-only autocast. Global bfloat16 autocast also
    # covers the decoder and can produce NaN atmospheric gradients through the
    # Perceiver; leaving the decoder in float32 keeps backward stable.
    _orig_model_autocast = getattr(model, "autocast", False)
    model.autocast = True

    with torch.enable_grad():
        pred = model.forward(batch)
        if isinstance(pred, tuple):
            pred = pred[0]
        score = target_fn(pred)
    score.float().backward()

    model.autocast = _orig_model_autocast

    grads = {}
    for k in atmos_var_names:
        g = batch.atmos_vars[k].grad
        grads[k] = g.detach().float().cpu().numpy() if g is not None else None
    for k in surf_var_names:
        g = batch.surf_vars[k].grad
        grads[k] = g.detach().float().cpu().numpy() if g is not None else None

    score_val = score.detach().float().item()

    del batch, pred, score
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"grads": grads, "score": score_val}
