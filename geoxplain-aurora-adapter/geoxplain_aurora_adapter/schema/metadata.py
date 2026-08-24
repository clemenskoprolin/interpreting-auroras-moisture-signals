"""Lightweight overlay metadata — display names, units, and default colormaps.

Kept dependency-free (no torch/numpy/xarray) so the client-side
``pull_overlay`` dispatch can infer overlay metadata without importing the
compute stack.  ``compute`` re-uses these tables on the server side.
"""

from __future__ import annotations

from typing import Optional

# Aurora's 13 pressure levels (hPa), in descending pressure / ascending
# altitude order.  Canonical home is here (dependency-free) so the client-side
# dispatch can validate a user's ``levels=`` selection without importing the
# compute stack.  ``data`` re-exports this for the server side.
AURORA_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

# Colormaps the GeoXplain viewer accepts for weather overlays.
OVERLAY_COLORMAPS = ("viridis", "plasma", "thermal", "sequential")

OVERLAY_NAMES = {
    "q": "Specific Humidity",
    "t": "Temperature",
    "z": "Geopotential",
    "u": "U Wind",
    "v": "V Wind",
    "2t": "2 m Temperature",
    "10u": "10 m U Wind",
    "10v": "10 m V Wind",
    "msl": "Mean Sea Level Pressure",
}

OVERLAY_UNITS = {
    "q": "kg/kg",
    "t": "K",
    "z": "m^2 s^-2",
    "zwd": "mm",
    "u": "m/s",
    "v": "m/s",
    "2t": "K",
    "10u": "m/s",
    "10v": "m/s",
    "msl": "Pa",
}

# Per-variable default colormap, chosen from the viewer's supported set:
# temperature → thermal, wind → plasma, pressure → sequential, everything
# else (humidity, geopotential) → viridis.  Unknown variables fall back to
# ``DEFAULT_OVERLAY_COLORMAP``.
OVERLAY_COLORMAP_DEFAULTS = {
    "t": "thermal",
    "2t": "thermal",
    "u": "plasma",
    "v": "plasma",
    "10u": "plasma",
    "10v": "plasma",
    "msl": "sequential",
    "q": "viridis",
    "z": "viridis",
}

DEFAULT_OVERLAY_COLORMAP = "viridis"


def default_overlay_label(variable: str, level: Optional[int]) -> str:
    """Human-readable overlay name, e.g. ``"Specific Humidity 850 hPa"``."""
    label = OVERLAY_NAMES.get(variable, variable)
    if level is not None:
        return f"{label} {level} hPa"
    return label


def default_overlay_unit(variable: str) -> str:
    """SI-ish unit string for ``variable`` (``""`` when unknown)."""
    return OVERLAY_UNITS.get(variable, "")


def default_overlay_colormap(variable: str) -> str:
    """Meaningful default colormap for ``variable``."""
    return OVERLAY_COLORMAP_DEFAULTS.get(variable, DEFAULT_OVERLAY_COLORMAP)


# ── XIA method display names ─────────────────────────────────────────────────
# ``method`` is a machine id used for branching in the compute/dispatch layers
# (``if method == "ig"``); these are the human-readable names the viewer shows.
METHOD_DISPLAY_NAMES = {
    "saliency": "Saliency",
    "ig": "Integrated Gradients",
    "rise": "RISE",
    "vit_cx": "ViT-CX",
}


def method_display_name(method: str) -> str:
    """Human-readable name for an XIA ``method`` id (``method`` itself if unknown)."""
    return METHOD_DISPLAY_NAMES.get(method, method)
