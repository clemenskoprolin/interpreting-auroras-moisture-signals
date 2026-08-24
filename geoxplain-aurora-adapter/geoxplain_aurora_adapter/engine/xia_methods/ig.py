"""
Integrated Gradients — general implementation.

IG_k(x, x_bl) = (x_k - x_bl_k) * (1/N) * sum_{i=0}^{N-1} dy/dx_k |_{x_bl + (i+0.5)/N * (x-x_bl)}

Uses the midpoint Riemann rule.  Gradient sign is preserved throughout;
the caller decides how to reduce (abs, max, etc.) for display.

Supports attribution w.r.t. atmospheric variables, surface variables, or
both.  batch_fn must return a Batch whose relevant tensors are leaf
tensors on `device` with requires_grad=True when requires_grad=True.
"""

import gc
import numpy as np
import torch


def integrated_gradients(
    model,
    batch_fn,
    target_fn,
    atmos_actual=None,
    atmos_baseline=None,
    atmos_var_names=(),
    surf_actual=None,
    surf_baseline=None,
    surf_var_names=(),
    device="cuda",
    n_steps=10,
    progress_callback=None,
):
    """Integrated Gradients attribution.

    Args:
        model:
            Aurora model in eval mode with gradient checkpointing already
            enabled and all parameters frozen (requires_grad=False).

        batch_fn:
            callable(alpha: float, requires_grad: bool) -> aurora.Batch
            alpha=0.0 → baseline, alpha=1.0 → actual input.
            When requires_grad=True, batch.atmos_vars[k] / batch.surf_vars[k]
            must be *leaf* tensors on `device` with requires_grad=True for
            every k in atmos_var_names / surf_var_names.  The tensor's values
            at alpha should equal:
                baseline[k] + alpha * (actual[k] - baseline[k])
            Variables that are not being attributed remain fixed at their
            original input values in the shipped adapter runner.

        target_fn:
            callable(pred) -> scalar torch.Tensor
            Differentiable target extracted from the model prediction.

        atmos_actual / atmos_baseline:
            dict[str -> torch.Tensor]  (CPU, float32), optional.
            Actual and baseline values for each atmospheric variable in
            atmos_var_names.  Shapes must match batch.atmos_vars[k].

        atmos_var_names:
            Tuple/list of atmospheric variable names to attribute.

        surf_actual / surf_baseline:
            dict[str -> torch.Tensor]  (CPU, float32), optional.
            Actual and baseline values for each surface variable in
            surf_var_names.  Shapes must match batch.surf_vars[k].

        surf_var_names:
            Tuple/list of surface variable names to attribute.

        device:
            torch device string.

        n_steps:
            Number of integration steps (midpoint Riemann rule).

    Returns:
        dict with:
            "ig": dict[str -> np.ndarray]
                Signed IG attribution per variable (atmos and surf merged).
                ig[k] = (actual[k] - baseline[k]) * mean_grad[k]
            "mean_grads": dict[str -> np.ndarray]
                Accumulated mean gradients (1/N * sum grad), before delta
                multiplication.
            "score": float
                target_fn value at the last IG step (alpha ≈ 1.0 for large N).
    """
    atmos_var_names = tuple(atmos_var_names or ())
    surf_var_names = tuple(surf_var_names or ())

    if not atmos_var_names and not surf_var_names:
        raise ValueError(
            "integrated_gradients requires at least one variable in "
            "atmos_var_names or surf_var_names."
        )

    grad_accum = {}
    for var in atmos_var_names:
        grad_accum[var] = np.zeros(atmos_actual[var].shape, dtype=np.float64)
    for var in surf_var_names:
        grad_accum[var] = np.zeros(surf_actual[var].shape, dtype=np.float64)

    score_val = None

    # Keep Aurora's own backbone-only autocast. Global bfloat16 autocast also
    # covers the decoder and can produce NaN atmospheric gradients through the
    # Perceiver; leaving the decoder in float32 keeps backward stable.
    _orig_model_autocast = getattr(model, "autocast", False)
    model.autocast = True

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps  # midpoint rule
        batch = batch_fn(alpha=alpha, requires_grad=True)

        with torch.enable_grad():
            pred = model.forward(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            score = target_fn(pred)
            score.float().backward()

        if step == n_steps - 1:
            score_val = score.detach().float().item()

        for var in atmos_var_names:
            g = batch.atmos_vars[var].grad
            if g is not None:
                grad_accum[var] += g.detach().float().cpu().numpy()
        for var in surf_var_names:
            g = batch.surf_vars[var].grad
            if g is not None:
                grad_accum[var] += g.detach().float().cpu().numpy()

        # Do not call torch.cuda.empty_cache() inside this loop. It can release
        # memory still needed by checkpointed backward/cuBLAS workspaces and
        # trigger a later CUDA "illegal memory access"; dropping references is
        # enough, and the allocator reuses cached blocks across steps.
        del batch, pred, score

        if (step + 1) % 5 == 0 or step == n_steps - 1:
            print(f"  IG step {step + 1}/{n_steps} done")
        if progress_callback is not None:
            progress_callback(step + 1, n_steps)

    model.autocast = _orig_model_autocast

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_grads = {var: grad_accum[var] / n_steps for var in grad_accum}

    ig = {}
    for var in atmos_var_names:
        delta = (atmos_actual[var] - atmos_baseline[var]).numpy()
        ig[var] = delta * mean_grads[var]
    for var in surf_var_names:
        delta = (surf_actual[var] - surf_baseline[var]).numpy()
        ig[var] = delta * mean_grads[var]

    return {"ig": ig, "mean_grads": mean_grads, "score": score_val}
