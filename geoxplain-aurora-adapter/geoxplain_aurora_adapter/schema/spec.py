"""Target specification for XIA computations.

A `TargetSpec` fully describes the scalar that an XIA method should explain:
which variable, at which pressure level, over which spatial region, and at
which timestamp.

Two spatial modes are supported:

- ``"point"`` — nearest grid point to ``(lat, lon)``.
- ``"box"``   — mean over a lat/lon box centered at ``(lat, lon)`` with
  half-widths derived from ``size = (dlat, dlon)``.  The default size
  ``DEFAULT_BOX_SIZE = (2.0, 3.0)`` (degrees lat × lon) is the median of
  the case-study boxes used in the ZWD searchlight benchmark; override
  per call via ``size=``.

Named case-study regions (ticino, california, ...) are *not* shipped with
this library — they are domain-specific data that belongs in the calling
project.  Build them in user code, e.g.::

    TICINO = ax.Target.box(var="q", level=850,
                           lat=46.25, lon=8.75, size=(1.5, 2.5),
                           timestamp="2024-03-20T00:00:00Z")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Default box full extent in degrees. Override with ``Target.box(..., size=...)``.
DEFAULT_BOX_SIZE: tuple[float, float] = (2.0, 3.0)


@dataclass
class TargetSpec:
    """Fully-resolved specification of the XIA attribution target.

    Fields
    ------
    var:        Variable name in the model output (e.g. ``"q"``, ``"t"``, ``"zwd"``).
    level:      Pressure level in hPa (e.g. ``850``).  ``None`` for surface-only vars.
    mode:       Spatial selection mode: ``"point"`` or ``"box"``.
    timestamp:  ISO-8601 string for the second (t1) input timestep, e.g.
                ``"2024-03-20T00:00:00Z"``.  This is what you pass when
                *constructing* a target, and it is preserved unchanged as the
                frame's displayed timestamp (``frame.target.timestamp`` ==
                ``frame.timestamp`` == t1).  The explained prediction is the
                6 h-ahead step t2 = t1 + lead, recorded in the frame's
                ``lead_hours`` metadata rather than by shifting the timestamp.
    lat/lon:    Point coordinates (``mode="point"``) or *box center*
                (``mode="box"``).  Longitudes accepted in either
                ``-180..180`` or ``0..360`` convention.
    size:       Box full extent in degrees ``(dlat, dlon)`` (``mode="box"``
                only).  The actual bounds are
                ``[lat-dlat/2, lat+dlat/2] × [lon-dlon/2, lon+dlon/2]``.
    """

    var: str
    level: Optional[int]
    mode: str
    timestamp: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    size: Optional[tuple[float, float]] = None

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def point(
        cls,
        *,
        var: str,
        level: Optional[int],
        lat: float,
        lon: float,
        timestamp: str,
    ) -> "TargetSpec":
        """Single-grid-point target."""
        return cls(
            var=var, level=level, mode="point", timestamp=timestamp,
            lat=lat, lon=lon,
        )

    @classmethod
    def box(
        cls,
        *,
        var: str,
        level: Optional[int],
        lat: float,
        lon: float,
        timestamp: str,
        size: tuple[float, float] = DEFAULT_BOX_SIZE,
    ) -> "TargetSpec":
        """Box-mean target centered at ``(lat, lon)`` with extent ``size``."""
        return cls(
            var=var, level=level, mode="box", timestamp=timestamp,
            lat=lat, lon=lon, size=tuple(size),
        )

    # ── derived properties ────────────────────────────────────────────────

    def box_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(south, north, west, east)`` for ``mode="box"``."""
        if self.mode != "box":
            raise ValueError(f"box_bounds() requires mode='box', got {self.mode!r}")
        dlat, dlon = self.size if self.size is not None else DEFAULT_BOX_SIZE
        return (
            self.lat - dlat / 2.0,
            self.lat + dlat / 2.0,
            self.lon - dlon / 2.0,
            self.lon + dlon / 2.0,
        )

    # ── serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "var": self.var,
            "level": self.level,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "lat": self.lat,
            "lon": self.lon,
            "size": list(self.size) if self.size is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetSpec":
        size = d.get("size")
        return cls(
            var=d["var"],
            level=d.get("level"),
            mode=d["mode"],
            timestamp=d["timestamp"],
            lat=d.get("lat"),
            lon=d.get("lon"),
            size=tuple(size) if size is not None else None,
        )

    # ── widget integration ────────────────────────────────────────────────

    def as_widget_dict(self) -> dict:
        """Return the target in the model-agnostic GeoXplain viewer format."""
        if self.mode == "point":
            return {
                "type": "point",
                "lat": float(self.lat),
                "lon": float(self.lon),
            }
        if self.mode == "box":
            s, n, w, e = self.box_bounds()
            return {
                "type": "box",
                "south": float(s), "north": float(n),
                "west": float(w),  "east": float(e),
            }
        return {}

    def __repr__(self) -> str:
        if self.mode == "point":
            return (
                f"TargetSpec(var={self.var!r}, level={self.level}, mode='point', "
                f"lat={self.lat}, lon={self.lon}, timestamp={self.timestamp!r})"
            )
        if self.mode == "box":
            return (
                f"TargetSpec(var={self.var!r}, level={self.level}, mode='box', "
                f"lat={self.lat}, lon={self.lon}, size={self.size}, "
                f"timestamp={self.timestamp!r})"
            )
        return f"TargetSpec(var={self.var!r}, mode={self.mode!r})"


class Target:
    """Namespace for TargetSpec factory methods.

    Usage::

        import geoxplain_aurora_adapter as ax

        # Single grid point
        target = ax.Target.point(var="q", level=850, lat=46.2, lon=8.8,
                                 timestamp="2024-03-20T00:00:00Z")

        # Box of default size (2.0° lat × 3.0° lon) centered at (lat, lon)
        target = ax.Target.box(var="q", level=850, lat=46.25, lon=8.75,
                               timestamp="2024-03-20T00:00:00Z")

        # Box of custom size
        target = ax.Target.box(var="q", level=850, lat=46.25, lon=8.75,
                               size=(1.5, 2.5),
                               timestamp="2024-03-20T00:00:00Z")
    """

    point = staticmethod(TargetSpec.point)
    box = staticmethod(TargetSpec.box)
