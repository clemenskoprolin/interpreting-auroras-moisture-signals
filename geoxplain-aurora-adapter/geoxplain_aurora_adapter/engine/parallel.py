"""Multi-GPU fan-out for forward-only attribution methods (RISE, ViT-CX).

Both methods reduce to a large set of *independent, batch-1* forward passes
whose scalar results are combined by a fixed-order numpy reduction.  This module
distributes only the forward passes across every visible CUDA device — one model
replica per device, driven by a work-stealing thread pool — while the reduction
stays byte-for-byte the single-GPU code path.

When fewer than two GPUs are visible (or compute is on CPU), callers fall back to
the original path and this module adds no overhead.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import torch


# Per-process cache of extra model replicas, keyed by the primary model object
# so a long-lived listener builds the (expensive) replicas once and reuses them
# across requests.  ``id(primary)`` is stable for the singleton model.
_replica_lock = threading.Lock()
_replica_cache: dict[int, dict[str, object]] = {}


def _normalize_device(device) -> str:
    """Return a canonical ``"cuda:N"`` / ``"cpu"`` string for ``device``."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return f"cuda:{dev.index if dev.index is not None else torch.cuda.current_device()}"
    return dev.type


def num_visible_gpus() -> int:
    """Number of CUDA devices to use, honouring an optional cap.

    Defaults to ``torch.cuda.device_count()`` (which reflects the SLURM
    allocation, e.g. ``--gpus-per-task``).  ``GEOXPLAIN_AURORA_ADAPTER_MAX_GPUS``
    caps it lower — primarily so tests can force the single-GPU path in a
    multi-GPU process for an exact comparison.
    """
    try:
        if not torch.cuda.is_available():
            return 0
        n = torch.cuda.device_count()
    except Exception:
        return 0
    cap = os.environ.get("GEOXPLAIN_AURORA_ADAPTER_MAX_GPUS")
    if cap:
        try:
            n = min(n, max(0, int(cap)))
        except ValueError:
            pass
    return n


def attribution_devices(primary_device) -> list[str]:
    """Devices to fan out across (primary first), or ``[]`` to stay sequential.

    Returns ``[]`` whenever fewer than two GPUs are usable, so callers keep the
    original single-GPU code path with zero replica/threading overhead.
    """
    n = num_visible_gpus()
    if n <= 1:
        return []
    devices = [f"cuda:{i}" for i in range(n)]
    primary = _normalize_device(primary_device)
    # Keep the primary device first so its slot reuses the already-loaded model.
    if primary in devices:
        devices.remove(primary)
        devices.insert(0, primary)
    return devices


def get_replicas(primary_model, primary_device, devices: list[str]) -> list[tuple]:
    """Return ``[(model, device_str), ...]`` aligned with ``devices``.

    The slot whose device equals the primary's reuses ``primary_model``; the
    remaining devices get weight-identical replicas, built once and cached for
    reuse across requests.
    """
    from .model import replicate_model

    primary_str = _normalize_device(primary_device)
    out: list[tuple] = []
    with _replica_lock:
        cache = _replica_cache.setdefault(id(primary_model), {})
        cache[primary_str] = primary_model
        for dev in devices:
            if dev == primary_str:
                out.append((primary_model, dev))
                continue
            replica = cache.get(dev)
            if replica is None:
                replica = replicate_model(primary_model, dev)
                cache[dev] = replica
            out.append((replica, dev))
    return out


def parallel_map_scores(
    task: Callable[[int, object, str], float],
    n_items: int,
    replicas: list[tuple],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[float]:
    """Run ``task(i, model, device)`` for ``i`` in ``range(n_items)`` across replicas.

    One worker thread per replica pulls indices off a shared, lock-guarded
    counter (dynamic load balancing for uneven per-forward times) and runs the
    task on its own device.  Results are returned in item order so the caller can
    reduce them exactly as the sequential path would.  ``progress_callback`` is
    invoked under the lock with a monotonically increasing ``(done, total)``.
    """
    results: list[Optional[float]] = [None] * n_items
    state = {"next": 0, "done": 0}
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(model, device):
        dev_index = torch.device(device).index
        try:
            with torch.cuda.device(dev_index):
                while True:
                    with lock:
                        if errors:
                            return
                        i = state["next"]
                        if i >= n_items:
                            return
                        state["next"] = i + 1
                    value = task(i, model, device)
                    results[i] = value
                    with lock:
                        state["done"] += 1
                        done = state["done"]
                        if progress_callback is not None:
                            progress_callback(done, n_items)
        except BaseException as exc:  # noqa: BLE001 — surfaced on the main thread
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(model, device), daemon=True)
        for model, device in replicas
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
    return [float(v) for v in results]  # type: ignore[arg-type]
