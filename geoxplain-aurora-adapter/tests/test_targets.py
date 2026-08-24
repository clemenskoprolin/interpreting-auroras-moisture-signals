"""Geometry tests for target box selection (``geoxplain_aurora_adapter.schema.targets``).

Focus: longitude handling must be convention-independent and wrap correctly
across the 0°/360° (and ±180°) seam.  A box straddling the prime meridian used
to produce ``lon_idx_min > lon_idx_max`` (an empty/inverted slice), which made
the box-mean target collapse to ``NaN`` and poisoned the whole attribution.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from geoxplain_aurora_adapter.schema.spec import Target
from geoxplain_aurora_adapter.schema.targets import box_indices, build_target_fn

# 1° full-globe grids in both longitude conventions.
LON_0_360 = np.arange(0.0, 360.0, 1.0)     # 0, 1, ..., 359
LON_180 = np.arange(-180.0, 180.0, 1.0)    # -180, ..., 179
LAT_DESC = np.linspace(90.0, -90.0, 181)   # ERA5 / Aurora ordering
LAT_ASC = np.linspace(-90.0, 90.0, 181)


def test_box_indices_no_wrap_0_360():
    """A box well inside the grid stays a contiguous, correctly-placed span."""
    lat_min, lat_max, lon_idx = box_indices(45.0, 47.0, 6.0, 10.0, LAT_DESC, LON_0_360)
    assert [LAT_DESC[lat_min], LAT_DESC[lat_max]] == [47.0, 45.0]  # descending
    assert LON_0_360[lon_idx].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0]
    # non-wrapping selection is contiguous and ascending
    assert lon_idx.tolist() == list(range(int(lon_idx[0]), int(lon_idx[-1]) + 1))


def test_box_indices_ascending_latitude():
    """Latitude selection works for ascending grids too."""
    lat_min, lat_max, _ = box_indices(45.0, 47.0, 6.0, 10.0, LAT_ASC, LON_0_360)
    assert LAT_ASC[lat_min] == 45.0 and LAT_ASC[lat_max] == 47.0


def test_box_indices_wraps_prime_meridian():
    """Box centred on lon=0 (bounds [-2, 2]) wraps both sides of the seam.

    Regression: previously returned an empty/inverted slice -> NaN target.
    """
    _, _, lon_idx = box_indices(45.0, 47.0, -2.0, 2.0, LAT_DESC, LON_0_360)
    assert lon_idx.size > 0
    assert sorted(LON_0_360[lon_idx].tolist()) == [0.0, 1.0, 2.0, 358.0, 359.0]


def test_box_indices_negative_bounds_on_0_360_grid():
    """California-style bounds may be given as 0..360 on a 0..360 grid."""
    _, _, lon_idx = box_indices(37.0, 39.0, 236.0, 240.0, LAT_DESC, LON_0_360)
    assert sorted(LON_0_360[lon_idx].tolist()) == [236.0, 237.0, 238.0, 239.0, 240.0]


def test_box_indices_convention_independent():
    """The same physical box selects the same longitudes regardless of the
    convention used for the bounds *and* the grid."""
    # bounds + grid in -180..180
    _, _, idx_a = box_indices(37.0, 39.0, -124.0, -120.0, LAT_DESC, LON_180)
    selected_a = sorted(LON_180[idx_a].tolist())
    # same box, bounds + grid in 0..360, mapped back to -180..180 for comparison
    _, _, idx_b = box_indices(37.0, 39.0, 236.0, 240.0, LAT_DESC, LON_0_360)
    selected_b = sorted((((LON_0_360[idx_b] + 180.0) % 360.0) - 180.0).tolist())
    assert selected_a == selected_b == [-124.0, -123.0, -122.0, -121.0, -120.0]


def test_build_target_fn_box_wrap_is_finite():
    """End-to-end: a prime-meridian box yields a finite mean over the correct,
    wrapped region (not NaN)."""
    torch = pytest.importorskip("torch")

    case = SimpleNamespace(
        lat_vals=LAT_DESC, lon_vals=LON_0_360, pressure_levels=np.array([850]),
    )
    target = Target.box(
        var="q", level=None, lat=46.0, lon=0.0, size=(2.0, 4.0),
        timestamp="2024-01-01T00:00:00Z",
    )
    fn = build_target_fn(target, case)

    H, W = LAT_DESC.size, LON_0_360.size
    field = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W)
    pred = SimpleNamespace(atmos_vars={}, surf_vars={"q": field})

    out = fn(pred)
    assert torch.isfinite(out)

    lat_min, lat_max, lon_idx = box_indices(45.0, 47.0, -2.0, 2.0, LAT_DESC, LON_0_360)
    expected = field[0, 0, lat_min:lat_max + 1, :][:, lon_idx.tolist()].mean()
    assert torch.allclose(out, expected)
