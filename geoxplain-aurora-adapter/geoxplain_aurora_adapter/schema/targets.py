"""Target function builder + geometry helpers for geoxplain_aurora_adapter.

This module is intentionally *generic*: it does not ship any named regions
(ticino, california, ...).  Those are domain-specific case studies and
belong in user code or a project-level config — not in a reusable library.
See ``geoxplain_aurora_adapter.spec.Target`` for how to build point / box targets.

Provided:
- ``build_target_fn`` — convert a ``TargetSpec`` + ``CaseData`` into a
                        differentiable ``target_fn(pred) → scalar`` suitable
                        for gradient-based XIA methods.
- Geometry helpers: ``box_indices``, ``nearest_gridpoint_indices``,
  ``great_circle_km``, ``cos_lat_weights``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..engine.data import CaseData
    from .spec import TargetSpec


# ── Geometry helpers ──────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km.  Scalars or broadcastable arrays."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_gridpoint_indices(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[int, int]:
    """Return the nearest grid-point indices to a requested lat/lon.

    Longitude matching uses the shortest angular distance so either
    -180..180 or 0..360 convention is accepted.
    """
    lat_idx = int(np.argmin(np.abs(lat_vals - lat)))
    lon_wrapped = lon % 360.0
    lon_delta = np.abs((lon_vals - lon_wrapped + 180.0) % 360.0 - 180.0)
    lon_idx = int(np.argmin(lon_delta))
    return lat_idx, lon_idx


def box_indices(
    south: float,
    north: float,
    west: float,
    east: float,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Return ``(lat_idx_min, lat_idx_max, lon_indices)`` for the box
    ``[south, north] × [west, east]`` (inclusive).

    Latitude is a contiguous ``lat_idx_min..lat_idx_max`` span; ERA5 descending
    latitudes are handled.  Longitude is returned as an *explicit ascending
    index array* rather than a ``min:max`` pair: a box may straddle the
    0°/360° (or ±180°) seam — e.g. centred on the prime meridian — and no
    single contiguous slice can express such a wrap.

    Longitude matching is convention-independent: ``west``/``east`` and
    ``lon_vals`` may each use the ``-180..180`` or ``0..360`` convention.  A
    column is selected when its angular offset east of ``west`` falls within
    the box's longitudinal span, which wraps across the seam automatically.
    """
    if lat_vals[0] > lat_vals[-1]:  # ERA5: descending (90 → -90)
        lat_idx_min = int(np.where(lat_vals <= north)[0][0])
        lat_idx_max = int(np.where(lat_vals >= south)[0][-1])
    else:
        lat_idx_min = int(np.where(lat_vals >= south)[0][0])
        lat_idx_max = int(np.where(lat_vals <= north)[0][-1])

    span = (east - west) % 360.0
    offset = (np.asarray(lon_vals, dtype=np.float64) - west) % 360.0
    lon_indices = np.where(offset <= span)[0].astype(int)
    return lat_idx_min, lat_idx_max, lon_indices


def cos_lat_weights(lat_vals: np.ndarray, W: int) -> np.ndarray:
    """(H, W) cosine-latitude weights, broadcast along longitude."""
    w_lat = np.cos(np.radians(lat_vals)).astype(np.float32)
    w_lat = np.clip(w_lat, 0.0, None)
    return np.broadcast_to(w_lat[:, None], (lat_vals.shape[0], W)).copy()


# ── Target function builder ───────────────────────────────────────────────────

def _level_index(case: "CaseData", level_hpa: int) -> int:
    """Return the pressure-level index for ``level_hpa`` in case.pressure_levels."""
    levels = np.asarray(case.pressure_levels)
    matches = np.where(levels == level_hpa)[0]
    if matches.size != 1:
        raise ValueError(
            f"Level {level_hpa} hPa not found in case pressure levels "
            f"{case.pressure_levels}. Available: {list(case.pressure_levels)}"
        )
    return int(matches[0])


def build_target_fn(target: "TargetSpec", case: "CaseData"):
    """Build a differentiable ``target_fn(pred) → scalar torch.Tensor``.

    Parameters
    ----------
    target: TargetSpec describing the scalar to explain.
    case:   CaseData loaded by ``load_case()``.

    Returns
    -------
    Callable ``target_fn`` that accepts an Aurora ``Batch`` prediction and
    returns a scalar ``torch.Tensor``.  The function is differentiable with
    respect to all tensors in the prediction.
    """
    lat_vals = case.lat_vals
    lon_vals = case.lon_vals

    # Resolve level index once
    level_idx: int | None = None
    if target.level is not None:
        level_idx = _level_index(case, target.level)

    if target.mode == "point":
        lat_idx, lon_idx = nearest_gridpoint_indices(
            lat_vals, lon_vals, target.lat, target.lon
        )

        def target_fn(pred):
            var = target.var
            if var in pred.atmos_vars:
                return pred.atmos_vars[var].float()[0, 0, level_idx, lat_idx, lon_idx]
            if var in pred.surf_vars:
                return pred.surf_vars[var].float()[0, 0, lat_idx, lon_idx]
            raise ValueError(f"Variable {var!r} not found in model prediction")

        return target_fn

    if target.mode == "box":
        south, north, west, east = target.box_bounds()
        # box_indices is convention-independent and wrap-aware, so the box
        # bounds are passed through untouched (no manual 0..360 normalization).
        lat_imin, lat_imax, lon_idx = box_indices(
            south, north, west, east, lat_vals, lon_vals
        )
        lon_cols = lon_idx.tolist()

        def target_fn(pred):
            var = target.var
            if var in pred.atmos_vars:
                field = pred.atmos_vars[var].float()[
                    0, 0, level_idx, lat_imin:lat_imax + 1, :
                ]
                return field[:, lon_cols].mean()
            if var in pred.surf_vars:
                field = pred.surf_vars[var].float()[
                    0, 0, lat_imin:lat_imax + 1, :
                ]
                return field[:, lon_cols].mean()
            raise ValueError(f"Variable {var!r} not found in model prediction")

        return target_fn

    raise ValueError(f"Unknown target mode: {target.mode!r}")
