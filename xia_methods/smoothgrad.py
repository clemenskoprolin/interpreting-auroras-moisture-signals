"""
SmoothGrad — average vanilla saliency over N noisy input copies.

Smilkov et al. 2017, "SmoothGrad: removing noise by adding noise". Given a
vanilla-gradient saliency map is often visually noisy, averaging it across
inputs perturbed with Gaussian noise gives a smoother, more localized map.

Implementation note: noise is injected by the caller's `batch_fn`, not by
this module. Each call to `batch_fn(requires_grad=True)` must return a freshly-
noised batch (so calling it N times yields N independent samples). This keeps
variable-specific noise-scaling logic out of the generic method code.
"""

import numpy as np

from .saliency import saliency as _saliency


def smoothgrad(model, batch_fn, target_fn, device,
               atmos_var_names=(), surf_var_names=(),
               n_samples=16):
    """SmoothGrad attribution via N-sample gradient averaging.

    Args:
        model, batch_fn, target_fn, device, atmos_var_names, surf_var_names:
            Same semantics as `xia_methods.saliency.saliency`. Crucially, each
            call to `batch_fn(requires_grad=True)` must resample the input
            noise so that successive calls produce independent noisy inputs.
        n_samples:
            Number of noisy-gradient samples to average. Typical range 10-50.

    Returns:
        dict with:
            "grads": dict[str -> np.ndarray | None]
                Mean signed gradient across the N samples, per variable.
            "score": float
                Mean target value across the N samples.
    """
    if n_samples < 1:
        raise ValueError(f"smoothgrad: n_samples must be >= 1, got {n_samples}")

    accum: dict[str, np.ndarray | None] = {
        k: None for k in list(atmos_var_names) + list(surf_var_names)
    }
    scores: list[float] = []

    for _ in range(n_samples):
        out = _saliency(
            model=model,
            batch_fn=batch_fn,
            target_fn=target_fn,
            device=device,
            atmos_var_names=atmos_var_names,
            surf_var_names=surf_var_names,
        )
        for k, g in out["grads"].items():
            if g is None:
                continue
            accum[k] = g if accum[k] is None else accum[k] + g
        scores.append(out["score"])

    grads = {
        k: (v / n_samples) if v is not None else None
        for k, v in accum.items()
    }
    return {"grads": grads, "score": float(np.mean(scores))}
