"""
Task / case / searchlight-mask definitions for the ZWD searchlight benchmark.

The benchmark is organised as follows:

    Target regions  -->  Cases (target + init_time + role)  -->  Scales
    -->  Searchlight masks (near + remote)

A `Case` fixes an init_time and a target region (with a fixed q850 box).
A `ScaleConfig` picks the Gaussian σ (in degrees of latitude) that defines
the searchlight mask radius AND the baseline-smoothing kernel for RISE /
ViT-CX / IG.  For each case and scale, we lay a coarse grid of mask centers
inside a 2500 km radius of the target box center (the "near" set), plus a
few "remote" centers beyond 5000 km that act as a sanity control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------
# Target regions
# ------------------------------------------------------------------
@dataclass(frozen=True)
class TargetRegion:
    name: str
    short: str
    box_lat: tuple[float, float]     # (south, north) in degrees
    box_lon: tuple[float, float]     # (west, east)  in degrees, 0..360 convention
    map_extent: tuple[float, float, float, float]  # (lon_w, lon_e, lat_s, lat_n)

    @property
    def center_lat(self) -> float:
        return 0.5 * (self.box_lat[0] + self.box_lat[1])

    @property
    def center_lon(self) -> float:
        return 0.5 * (self.box_lon[0] + self.box_lon[1])


TARGETS: dict[str, TargetRegion] = {
    "ticino": TargetRegion(
        name="Ticino / Southern Alps",
        short="ticino",
        box_lat=(45.5, 47.0),
        box_lon=(7.5, 10.0),
        map_extent=(-15.0, 45.0, 30.0, 65.0),
    ),
    "california": TargetRegion(
        name="Northern California AR corridor",
        short="california",
        box_lat=(37.0, 39.5),
        box_lon=(236.0, 240.0),              # 0..360 convention
        map_extent=(-145.0, -100.0, 25.0, 55.0),
    ),
    "japan": TargetRegion(
        name="Central Honshu Pacific side",
        short="japan",
        box_lat=(34.5, 36.5),
        box_lon=(137.0, 140.5),
        map_extent=(120.0, 160.0, 20.0, 50.0),
    ),
    # --- additional Alps (to reinforce/falsify the Ticino finding) ---
    "alps_east": TargetRegion(
        name="Eastern Alps / Innsbruck–Salzburg",
        short="alps_east",
        box_lat=(47.0, 48.5),
        box_lon=(11.0, 13.5),
        map_extent=(-5.0, 35.0, 35.0, 62.0),
    ),
    "valais": TargetRegion(
        name="Western Alps / Valais–Rhône",
        short="valais",
        box_lat=(45.8, 47.2),
        box_lon=(6.5, 8.5),
        map_extent=(-15.0, 25.0, 35.0, 62.0),
    ),
    # --- other mountain ranges ---
    "rockies": TargetRegion(
        name="Colorado Rocky Mountains",
        short="rockies",
        box_lat=(38.5, 40.5),
        box_lon=(254.0, 257.0),              # ~103–106 W
        map_extent=(-130.0, -85.0, 25.0, 55.0),
    ),
    "himalayas": TargetRegion(
        name="Central Himalayas / Nepal",
        short="himalayas",
        box_lat=(27.0, 29.5),
        box_lon=(83.0, 87.0),
        map_extent=(65.0, 105.0, 15.0, 45.0),
    ),
    # --- flat-terrain controls (should show no beyond-humidity signal) ---
    "netherlands": TargetRegion(
        name="Netherlands / Low Countries (flat control)",
        short="netherlands",
        box_lat=(51.5, 53.5),
        box_lon=(4.0, 7.0),
        map_extent=(-15.0, 35.0, 40.0, 65.0),
    ),
    "great_plains": TargetRegion(
        name="US Great Plains / Kansas (flat control)",
        short="great_plains",
        box_lat=(38.0, 40.0),
        box_lon=(262.0, 265.0),              # ~95–98 W
        map_extent=(-130.0, -80.0, 25.0, 55.0),
    ),
    "ganges_plain": TargetRegion(
        name="Ganges Plain / Bangladesh (flat tropical control)",
        short="ganges_plain",
        box_lat=(23.5, 25.5),
        box_lon=(87.0, 91.0),
        map_extent=(70.0, 105.0, 10.0, 38.0),
    ),
    # --- additional mountain ranges for conditional-mechanism falsification ---
    "andes": TargetRegion(
        name="Central Andes / Chile-Argentina",
        short="andes",
        box_lat=(-34.0, -31.0),
        box_lon=(288.0, 291.5),             # ~72-68.5 W
        map_extent=(-90.0, -45.0, -50.0, -15.0),
    ),
    "cascades_sierra": TargetRegion(
        name="Cascades / Northern Sierra Nevada",
        short="cascades_sierra",
        box_lat=(40.0, 44.0),
        box_lon=(238.0, 242.0),             # ~122-118 W
        map_extent=(-140.0, -105.0, 25.0, 55.0),
    ),
    "pyrenees": TargetRegion(
        name="Pyrenees",
        short="pyrenees",
        box_lat=(42.0, 43.5),
        box_lon=(0.0, 3.0),
        map_extent=(-15.0, 20.0, 32.0, 52.0),
    ),
    "atlas": TargetRegion(
        name="Atlas Mountains / Morocco",
        short="atlas",
        box_lat=(31.0, 34.0),
        box_lon=(352.0, 356.0),             # ~8-4 W
        map_extent=(-20.0, 15.0, 20.0, 42.0),
    ),
    "new_zealand_alps": TargetRegion(
        name="New Zealand Southern Alps",
        short="new_zealand_alps",
        box_lat=(-45.5, -42.5),
        box_lon=(168.0, 172.0),
        map_extent=(155.0, 185.0, -55.0, -32.0),
    ),
    "caucasus": TargetRegion(
        name="Greater Caucasus",
        short="caucasus",
        box_lat=(41.0, 43.5),
        box_lon=(42.0, 46.0),
        map_extent=(25.0, 60.0, 32.0, 52.0),
    ),
    # --- paired flat/coastal controls for the added mountain regions ---
    "pampas": TargetRegion(
        name="Argentine Pampas (flat control for Andes)",
        short="pampas",
        box_lat=(-36.0, -33.0),
        box_lon=(300.0, 304.0),             # ~60-56 W
        map_extent=(-75.0, -45.0, -45.0, -25.0),
    ),
    "pacific_nw_coast": TargetRegion(
        name="Pacific Northwest coast (coastal control for Cascades)",
        short="pacific_nw_coast",
        box_lat=(44.0, 47.0),
        box_lon=(235.0, 238.0),             # ~125-122 W
        map_extent=(-140.0, -110.0, 32.0, 55.0),
    ),
    "aquitaine_basin": TargetRegion(
        name="Aquitaine Basin (flat control for Pyrenees)",
        short="aquitaine_basin",
        box_lat=(44.0, 46.0),
        box_lon=(0.0, 2.5),
        map_extent=(-15.0, 20.0, 35.0, 55.0),
    ),
    "sahara_plain": TargetRegion(
        name="Northwest Sahara plain (dry flat control for Atlas)",
        short="sahara_plain",
        box_lat=(28.0, 31.0),
        box_lon=(350.0, 354.0),             # ~10-6 W
        map_extent=(-20.0, 15.0, 18.0, 40.0),
    ),
    "canterbury_plain": TargetRegion(
        name="Canterbury Plain (flat control for New Zealand Alps)",
        short="canterbury_plain",
        box_lat=(-44.5, -42.5),
        box_lon=(172.0, 174.5),
        map_extent=(160.0, 185.0, -52.0, -35.0),
    ),
    "caspian_lowland": TargetRegion(
        name="Caspian Lowland (flat control for Caucasus)",
        short="caspian_lowland",
        box_lat=(45.0, 47.0),
        box_lon=(46.0, 50.0),
        map_extent=(30.0, 65.0, 35.0, 55.0),
    ),
    "amazon_interior": TargetRegion(
        name="Amazon interior (low-residual moist control)",
        short="amazon_interior",
        box_lat=(-6.0, -3.0),
        box_lon=(295.0, 300.0),            # 65-60 W
        map_extent=(-80.0, -40.0, -20.0, 10.0),
    ),
}


# ------------------------------------------------------------------
# Cases (target x init_time x role)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    target: str            # key into TARGETS
    init_time: datetime    # second (t1) of the two input timesteps
    role: str              # "strong" or "secondary"

    @property
    def case_id(self) -> str:
        return (
            f"{self.target}_{self.init_time.strftime('%Y%m%d%H')}_{self.role}"
        )


def default_cases() -> list[Case]:
    """Hard-coded v1 cases.

    Dates are placeholders (2020-01-01 and 2021-01-01 at 22 UTC init);
    swap via `load_cases_from_json` once a proper selection has been done.
    """
    t0 = datetime(2020, 1, 1, 22, 0)
    t1 = datetime(2021, 1, 1, 22, 0)
    out: list[Case] = []
    for short in ("ticino", "california", "japan"):
        out.append(Case(target=short, init_time=t0, role="strong"))
        out.append(Case(target=short, init_time=t1, role="secondary"))
    return out


def load_cases_from_json(path: str) -> list[Case]:
    import json
    with open(path, "r") as f:
        raw = json.load(f)
    out = []
    for entry in raw:
        out.append(Case(
            target=entry["target"],
            init_time=datetime.fromisoformat(entry["init_time"]),
            role=entry.get("role", "strong"),
        ))
    return out


# ------------------------------------------------------------------
# Scales
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ScaleConfig:
    name: str          # "local" / "synoptic"
    sigma_deg: float   # Gaussian σ in degrees of latitude
    grid_stride_deg: float  # spacing between mask centers in the near grid


SCALES: dict[str, ScaleConfig] = {
    "local":    ScaleConfig(name="local",    sigma_deg=2.5, grid_stride_deg=2.5),
    "synoptic": ScaleConfig(name="synoptic", sigma_deg=6.0, grid_stride_deg=5.0),
}


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km.  Scalars or broadcastable arrays."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ------------------------------------------------------------------
# Searchlight mask centers
# ------------------------------------------------------------------
@dataclass
class MaskSpec:
    scale: str
    role: str            # "near" or "remote"
    center_lat: float
    center_lon: float    # 0..360 convention
    mask_id: int

    @property
    def key(self) -> str:
        return f"{self.scale}_{self.role}_{self.mask_id:03d}"


def generate_mask_centers(
    target: TargetRegion,
    scale: ScaleConfig,
    near_radius_km: float = 2500.0,
    remote_min_km: float = 5000.0,
    n_remote: int = 4,
) -> list[MaskSpec]:
    """Lay out searchlight centers for one (target, scale) pair.

    Near centers: regular lat-lon grid at `grid_stride_deg` spacing,
    intersected with a 2500 km disk around the target box center.

    Remote centers: deterministic set of points at fixed bearings
    (N/E/S/W) just beyond `remote_min_km`.  These are used as a remote-
    control sanity check.
    """
    specs: list[MaskSpec] = []
    clat, clon = target.center_lat, target.center_lon
    stride = scale.grid_stride_deg

    # Roughly symmetric lat window; lon window widened by 1/cos(lat) to
    # compensate for convergence of meridians.
    lat_halfwidth = near_radius_km / 111.0
    lon_halfwidth = lat_halfwidth / max(math.cos(math.radians(clat)), 0.2)

    lats = np.arange(
        clat - lat_halfwidth, clat + lat_halfwidth + 1e-6, stride
    )
    lons = np.arange(
        clon - lon_halfwidth, clon + lon_halfwidth + 1e-6, stride
    )

    mid = 0
    for la in lats:
        for lo in lons:
            lo_wrapped = lo % 360.0
            d = great_circle_km(clat, clon, la, lo_wrapped)
            if float(d) <= near_radius_km:
                specs.append(MaskSpec(
                    scale=scale.name, role="near",
                    center_lat=float(la), center_lon=float(lo_wrapped),
                    mask_id=mid,
                ))
                mid += 1

    # Remote control: cardinal bearings just past remote_min_km.
    # We step outwards along great circles from the target center.
    for b_idx, bearing_deg in enumerate((0.0, 90.0, 180.0, 270.0)):
        if b_idx >= n_remote:
            break
        # Convert bearing + distance to destination lat/lon.
        br = math.radians(bearing_deg)
        ang = (remote_min_km + 500.0) / EARTH_RADIUS_KM
        phi1 = math.radians(clat)
        lam1 = math.radians(clon)
        phi2 = math.asin(
            math.sin(phi1) * math.cos(ang)
            + math.cos(phi1) * math.sin(ang) * math.cos(br)
        )
        lam2 = lam1 + math.atan2(
            math.sin(br) * math.sin(ang) * math.cos(phi1),
            math.cos(ang) - math.sin(phi1) * math.sin(phi2),
        )
        rlat = math.degrees(phi2)
        rlon = math.degrees(lam2) % 360.0
        # Clamp to valid lat range; if a bearing would put us past the
        # pole, fall back to a modest poleward offset.
        rlat = float(np.clip(rlat, -85.0, 85.0))
        specs.append(MaskSpec(
            scale=scale.name, role="remote",
            center_lat=rlat, center_lon=rlon, mask_id=mid,
        ))
        mid += 1

    return specs


# ------------------------------------------------------------------
# Gaussian masks on the lat-lon grid
# ------------------------------------------------------------------
def gaussian_mask(
    spec: MaskSpec,
    sigma_deg: float,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> np.ndarray:
    """Return an (H, W) float32 Gaussian mask centered at spec.center_*.

    Approximates geodesic distance by (lat_deg, cos(lat)*lon_deg) which
    is accurate enough for mask radii << Earth radius.  The mask is
    normalised so its peak value equals 1.
    """
    H, W = lat_vals.shape[0], lon_vals.shape[0]
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    dlat = lat_grid - spec.center_lat
    dlon = (lon_grid - spec.center_lon + 180.0) % 360.0 - 180.0  # wrap
    dlon *= np.cos(np.radians(spec.center_lat))

    r2 = dlat * dlat + dlon * dlon
    mask = np.exp(-0.5 * r2 / (sigma_deg * sigma_deg))
    return mask.astype(np.float32)


def cos_lat_weights(lat_vals: np.ndarray, W: int) -> np.ndarray:
    """(H, W) cosine-latitude weights, broadcast along longitude."""
    w_lat = np.cos(np.radians(lat_vals)).astype(np.float32)
    w_lat = np.clip(w_lat, 0.0, None)
    return np.broadcast_to(w_lat[:, None], (lat_vals.shape[0], W)).copy()


def nearest_gridpoint_indices(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[int, int]:
    """Return the nearest grid-point indices to a requested lat/lon.

    Longitude is matched on the wrapped 0..360 grid using the shortest angular
    distance, so callers can pass either -180..180 or 0..360 convention.
    """
    lat_idx = int(np.argmin(np.abs(lat_vals - lat)))
    lon_wrapped = lon % 360.0
    lon_delta = np.abs((lon_vals - lon_wrapped + 180.0) % 360.0 - 180.0)
    lon_idx = int(np.argmin(lon_delta))
    return lat_idx, lon_idx


def box_indices(
    target: TargetRegion,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return (lat_idx_min, lat_idx_max, lon_idx_min, lon_idx_max) for the
    target box, inclusive indices.  Handles ERA5 descending latitudes.
    """
    lo_s, lo_n = target.box_lat
    lo_w, lo_e = target.box_lon

    # ERA5 lat is descending (90 .. -90): min index = northernmost.
    if lat_vals[0] > lat_vals[-1]:
        lat_idx_min = int(np.where(lat_vals <= lo_n)[0][0])
        lat_idx_max = int(np.where(lat_vals >= lo_s)[0][-1])
    else:
        lat_idx_min = int(np.where(lat_vals >= lo_s)[0][0])
        lat_idx_max = int(np.where(lat_vals <= lo_n)[0][-1])

    lon_idx_min = int(np.where(lon_vals >= lo_w)[0][0])
    lon_idx_max = int(np.where(lon_vals <= lo_e)[0][-1])
    return lat_idx_min, lat_idx_max, lon_idx_min, lon_idx_max
