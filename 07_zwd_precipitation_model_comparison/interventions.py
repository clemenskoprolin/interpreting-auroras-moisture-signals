"""
interventions.py — Precipitation intervention generation for 07_zwd_precipitation_model_comparison.

Provides spatial masks and intervention builders for manipulating precipitation
fields before feeding them to Aurora models.

All masks use the convention: 0 = fully removed/replaced, 1 = fully kept.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from comparison_config import DEFAULT_DOSES_MM, PRECIP_DISK_KM, PRECIP_TAPER_KM


# ---------------------------------------------------------------------------
# Earth geometry (reuse pattern from searchlight_tasks.py)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def _great_circle_km(lat1: float, lon1: float, lat2_arr, lon2_arr) -> np.ndarray:
    """Great-circle distance from a single point to an array of points (km)."""
    lat1_r = math.radians(lat1)
    lat2_r = np.radians(lat2_arr)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Spatial mask
# ---------------------------------------------------------------------------

def make_removal_mask(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    disk_km: float = PRECIP_DISK_KM,
    taper_km: float = PRECIP_TAPER_KM,
) -> np.ndarray:
    """Return a float [0,1] mask of shape (H, W).

    Convention:
      - 0  = fully removed (inside disk_km)
      - 1  = fully kept   (beyond taper_km)
      - smooth cosine taper between disk_km and taper_km

    Args:
        lat_vals: (H,) latitude array.
        lon_vals: (W,) longitude array.
        center_lat: Disk center latitude (degrees).
        center_lon: Disk center longitude (degrees, 0..360 convention).
        disk_km: Radius at which removal is 100%.
        taper_km: Radius at which the mask reaches 1.0 (no effect).

    Returns:
        (H, W) float32 array with values in [0, 1].
    """
    if disk_km >= taper_km:
        raise ValueError(f"disk_km ({disk_km}) must be < taper_km ({taper_km})")

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")  # (H, W)
    # Normalize longitude difference (handle wrap-around)
    dlon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    lon_adj = center_lon + dlon

    dist = _great_circle_km(center_lat, center_lon, lat_grid, lon_adj)

    mask = np.ones_like(dist, dtype=np.float32)

    # Inside disk: 0
    inside = dist <= disk_km
    mask[inside] = 0.0

    # Taper region: smooth cosine
    taper_zone = (dist > disk_km) & (dist < taper_km)
    if taper_zone.any():
        t = (dist[taper_zone] - disk_km) / (taper_km - disk_km)   # 0..1
        # Cosine taper: 0 at t=0, 1 at t=1
        mask[taper_zone] = 0.5 * (1.0 - np.cos(np.pi * t))

    return mask


# ---------------------------------------------------------------------------
# Precipitation field manipulation
# ---------------------------------------------------------------------------

def apply_precip_removal(
    precip_tensor: torch.Tensor,
    mask: np.ndarray,
    timestep: int,
) -> torch.Tensor:
    """Apply a removal mask to the precipitation tensor.

    Multiplies precip by mask (0 = remove, 1 = keep) at the specified timestep.
    mask = 0 → precip zeroed out. No baseline replacement — precipitation is set
    to zero where the mask is 0, since zero is the natural "no rain" baseline.

    Args:
        precip_tensor: (1, 2, H, W) CPU float32 precipitation tensor.
        mask: (H, W) float32 array in [0, 1].
        timestep: 0 for t0, 1 for t1, -1 for both.

    Returns:
        New (1, 2, H, W) tensor with precip replaced.
    """
    out = precip_tensor.clone()
    mask_t = torch.from_numpy(mask.astype(np.float32))  # (H, W)

    if timestep == -1:
        out[0, 0] = out[0, 0] * mask_t
        out[0, 1] = out[0, 1] * mask_t
    elif timestep in (0, 1):
        out[0, timestep] = out[0, timestep] * mask_t
    else:
        raise ValueError(f"timestep must be 0, 1, or -1 (both), got {timestep}")

    return out


def apply_precip_dose(
    precip_tensor: torch.Tensor,
    mask: np.ndarray,
    dose_mm: float,
    timestep: int,
    subtract: bool = False,
) -> torch.Tensor:
    """Add (or subtract) a spatially-tapered dose to the precipitation tensor.

    The dose is applied proportionally to (1 - mask): pixels inside the disk
    (mask=0) get the full dose; pixels outside the taper (mask=1) get zero dose.
    This keeps the dose spatially consistent with the removal region.

    Args:
        precip_tensor: (1, 2, H, W) CPU float32 precipitation tensor.
        mask: (H, W) float32 removal mask (0=removed, 1=kept).
        dose_mm: Dose magnitude in mm.
        timestep: 0 for t0, 1 for t1, -1 for both.
        subtract: If True, subtract the dose instead of adding.

    Returns:
        New (1, 2, H, W) tensor with dose applied. Values clipped to >= 0.
    """
    out = precip_tensor.clone()
    sign = -1.0 if subtract else 1.0
    # dose_field: full dose where mask=0, zero where mask=1
    dose_field = torch.from_numpy(
        (dose_mm * (1.0 - mask.astype(np.float32)))
    )  # (H, W)

    if timestep == -1:
        out[0, 0] = (out[0, 0] + sign * dose_field).clamp(min=0.0)
        out[0, 1] = (out[0, 1] + sign * dose_field).clamp(min=0.0)
    elif timestep in (0, 1):
        out[0, timestep] = (out[0, timestep] + sign * dose_field).clamp(min=0.0)
    else:
        raise ValueError(f"timestep must be 0, 1, or -1 (both), got {timestep}")

    return out


# ---------------------------------------------------------------------------
# All-in-one intervention builder
# ---------------------------------------------------------------------------

def build_all_interventions(
    precip_tensor: torch.Tensor,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    doses_mm: tuple[float, ...] = DEFAULT_DOSES_MM,
    disk_km: float = PRECIP_DISK_KM,
    taper_km: float = PRECIP_TAPER_KM,
) -> Dict[str, torch.Tensor]:
    """Build all intervention variants for a single event location.

    Args:
        precip_tensor: (1, 2, H, W) CPU float32 precipitation tensor at (t0, t1).
        lat_vals: (H,) latitude array.
        lon_vals: (W,) longitude array.
        center_lat: Disk center latitude.
        center_lon: Disk center longitude (0..360 convention).
        doses_mm: Iterable of dose magnitudes (e.g. (1.0, 5.0, 10.0)).
        disk_km: Radius of full removal (default PRECIP_DISK_KM).
        taper_km: Radius at which effect tapers to zero (default PRECIP_TAPER_KM).

    Returns:
        Dict mapping intervention name → (1, 2, H, W) CPU float32 tensor.

        Keys:
          "actual"        — original precipitation (unmodified)
          "remove_t1"     — zero out precip in disk at t1 only
          "remove_t0"     — zero out precip in disk at t0 only
          "remove_both"   — zero out precip in disk at both t0 and t1
          "dose_plus_{d}mm"  for d in doses_mm
          "dose_minus_{d}mm" for d in doses_mm
    """
    mask = make_removal_mask(
        lat_vals, lon_vals, center_lat, center_lon,
        disk_km=disk_km, taper_km=taper_km,
    )

    interventions: Dict[str, torch.Tensor] = {
        "actual":      precip_tensor.clone(),
        "remove_t1":   apply_precip_removal(precip_tensor, mask, timestep=1),
        "remove_t0":   apply_precip_removal(precip_tensor, mask, timestep=0),
        "remove_both": apply_precip_removal(precip_tensor, mask, timestep=-1),
    }

    for d in doses_mm:
        key_plus = f"dose_plus_{d:.0f}mm" if d == int(d) else f"dose_plus_{d}mm"
        key_minus = f"dose_minus_{d:.0f}mm" if d == int(d) else f"dose_minus_{d}mm"
        interventions[key_plus] = apply_precip_dose(
            precip_tensor, mask, d, timestep=1, subtract=False
        )
        interventions[key_minus] = apply_precip_dose(
            precip_tensor, mask, d, timestep=1, subtract=True
        )

    return interventions
