"""Local overlay pulls — raw ERA5 fields, no model and no GPU.

``_pull_overlay_local`` reads requested weather fields straight from the local
data store into an ``OverlayResult``.  It never runs Aurora, which is why the
sbatch listener can compute overlays in-process on the login node instead of
submitting a GPU job (see ``remote/local_overlay.py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._common import _parse_init_time
from .data import ATMOS_VARS, AURORA_LEVELS, SURF_VARS, load_overlay_field
from ..schema.metadata import (
    OVERLAY_COLORMAPS,
    default_overlay_colormap,
    default_overlay_label,
    default_overlay_unit,
)
from ..schema.overlay import OverlayFrame, OverlayResult


def _validate_overlay_request(variable: str, level: Optional[int]) -> None:
    if variable in ATMOS_VARS:
        if level is None:
            raise ValueError(f"Atmospheric variable {variable!r} requires level=... in hPa.")
        if level not in AURORA_LEVELS:
            raise ValueError(
                f"Unsupported level {level!r} for {variable!r}. "
                f"Supported levels: {list(AURORA_LEVELS)}"
            )
        return
    if variable in SURF_VARS:
        if level is not None:
            raise ValueError(f"Surface variable {variable!r} does not accept level=.")
        return
    raise ValueError(
        f"Unknown overlay variable {variable!r}. "
        f"ATMOS_VARS={list(ATMOS_VARS)}, SURF_VARS={list(SURF_VARS)}"
    )


def _pull_overlay_local(
    variable: str,
    timestamps: list[str],
    *,
    level: Optional[int] = None,
    name: Optional[str] = None,
    unit: Optional[str] = None,
    colormap: Optional[str] = None,
    visible: bool = True,
) -> OverlayResult:
    """Pull raw ERA5 fields from the local data store into an OverlayResult."""

    if colormap is None:
        colormap = default_overlay_colormap(variable)
    if colormap not in OVERLAY_COLORMAPS:
        raise ValueError(f"colormap must be one of {OVERLAY_COLORMAPS}. Got: {colormap!r}")
    if not timestamps:
        raise ValueError("timestamps is empty - overlay requests need at least one timestamp.")
    _validate_overlay_request(variable, level)

    frames: list[OverlayFrame] = []
    lat_vals = None
    lon_vals = None
    for timestamp in timestamps:
        init_time = _parse_init_time(timestamp)
        arr, lat_vals, lon_vals = load_overlay_field(init_time, variable, level=level)
        frames.append(OverlayFrame(timestamp=timestamp, data=arr.astype(np.float32, copy=True)))

    return OverlayResult(
        variable=variable,
        level=level,
        frames=frames,
        label=name or default_overlay_label(variable, level),
        unit=default_overlay_unit(variable) if unit is None else unit,
        colormap=colormap,
        visible=visible,
        lat=np.asarray(lat_vals, dtype=np.float32) if lat_vals is not None else None,
        lon=np.asarray(lon_vals, dtype=np.float32) if lon_vals is not None else None,
        meta={"source": "ERA5", "cadence": "6h"},
    )
