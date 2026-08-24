"""OverlayResult - raw ERA5 weather-field overlays for the visualization.

File format: ``.overlay.npz`` (``format_version == 3``) - a zip archive
containing:
- ``meta.json``: variable metadata (including ``overlay_offset_hours`` and
  ``time_label``) and a frame list.
- ``lat.npy`` / ``lon.npy``: optional coordinate vectors.
- ``f{i}.npy``: one raw float32 2-D array per overlay frame.

``overlay_offset_hours`` (added in version 2, default 0) records how far the
overlay's field data was shifted relative to each frame's displayed timestamp
(see ``pull_overlay(overlay_time=...)``). The displayed frame is Aurora's
most-recent input step ``t1``, so: ``0`` for the frame's own time (the default
``t1`` input field), ``-6`` for the prior input step ``t0``, ``+6`` for the
forecast valid time ``t2``. Version 1 bundles predate the field and are loaded
as ``overlay_offset_hours = 0``.

``time_label`` (added in version 3, default ``None``) is an optional free-text
annotation the viewer shows alongside the offset, e.g. ``"Forecast valid time
t2"``. Version 1/2 bundles predate the field and load as ``time_label = None``.

Wire format: msgpack envelope with float32 arrays embedded as raw bytes.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

OVERLAY_FORMAT_VERSION = 3


@dataclass
class OverlayFrame:
    """One timestamped raw weather-field overlay frame."""

    timestamp: str
    data: np.ndarray


@dataclass
class OverlayResult:
    """Self-contained weather-field overlay bundle.

    ``frames`` are raw ERA5/Aurora-grid arrays. The visualization package applies
    its existing preprocessing pipeline when the overlay is added to a widget.
    """

    variable: str
    level: Optional[int]
    frames: list[OverlayFrame]
    label: str
    unit: str = ""
    colormap: str = "viridis"
    visible: bool = True
    #: Hours the field data is shifted relative to each frame's displayed
    #: timestamp, which is Aurora's input step t1 (0 = t1, the frame's own time;
    #: -6 = prior input step t0; +6 = forecast valid time t2).
    overlay_offset_hours: int = 0
    #: Optional free-text annotation shown by the viewer alongside the offset
    #: (e.g. "Forecast valid time t2"). ``None`` shows only the offset, if any.
    time_label: Optional[str] = None
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def timestamps(self) -> list[str]:
        return [frame.timestamp for frame in self.frames]

    def arrays(self) -> np.ndarray:
        return np.stack([np.asarray(frame.data, dtype=np.float32) for frame in self.frames])

    def _meta_payload(self) -> dict:
        return {
            "format_version": OVERLAY_FORMAT_VERSION,
            "variable": self.variable,
            "level": self.level,
            "label": self.label,
            "unit": self.unit,
            "colormap": self.colormap,
            "visible": self.visible,
            "overlay_offset_hours": self.overlay_offset_hours,
            "time_label": self.time_label,
            "meta": self.meta,
            "has_lat": self.lat is not None,
            "has_lon": self.lon is not None,
            "frames": [
                {
                    "timestamp": frame.timestamp,
                    "member": f"f{i}.npy",
                    "shape": list(np.asarray(frame.data).shape),
                }
                for i, frame in enumerate(self.frames)
            ],
        }

    def save(self, path: str | os.PathLike) -> None:
        """Save to a ``.overlay.npz`` archive."""

        path = os.fspath(path)
        if not path.endswith(".overlay.npz"):
            path = path + ".overlay.npz"

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(self._meta_payload(), indent=2))
            if self.lat is not None:
                buf = io.BytesIO()
                np.save(buf, np.asarray(self.lat, dtype=np.float32))
                zf.writestr("lat.npy", buf.getvalue())
            if self.lon is not None:
                buf = io.BytesIO()
                np.save(buf, np.asarray(self.lon, dtype=np.float32))
                zf.writestr("lon.npy", buf.getvalue())
            for i, frame in enumerate(self.frames):
                buf = io.BytesIO()
                np.save(buf, np.asarray(frame.data, dtype=np.float32))
                zf.writestr(f"f{i}.npy", buf.getvalue())

    @classmethod
    def load(cls, path: str | os.PathLike) -> "OverlayResult":
        """Load from a ``.overlay.npz`` archive."""

        path = os.fspath(path)
        with zipfile.ZipFile(path, "r") as zf:
            meta_payload = json.loads(zf.read("meta.json"))
            frames = []
            for frame_meta in meta_payload.get("frames", []):
                buf = io.BytesIO(zf.read(frame_meta["member"]))
                frames.append(
                    OverlayFrame(
                        timestamp=frame_meta.get("timestamp", ""),
                        data=np.load(buf).astype(np.float32, copy=False),
                    )
                )
            lat = None
            lon = None
            if meta_payload.get("has_lat") and "lat.npy" in zf.namelist():
                lat = np.load(io.BytesIO(zf.read("lat.npy"))).astype(np.float32, copy=False)
            if meta_payload.get("has_lon") and "lon.npy" in zf.namelist():
                lon = np.load(io.BytesIO(zf.read("lon.npy"))).astype(np.float32, copy=False)

        return cls(
            variable=meta_payload["variable"],
            level=meta_payload.get("level"),
            frames=frames,
            label=meta_payload.get("label", meta_payload["variable"]),
            unit=meta_payload.get("unit", ""),
            colormap=meta_payload.get("colormap", "viridis"),
            visible=meta_payload.get("visible", True),
            overlay_offset_hours=int(meta_payload.get("overlay_offset_hours", 0)),
            time_label=meta_payload.get("time_label"),
            lat=lat,
            lon=lon,
            meta=meta_payload.get("meta", {}),
        )

    def to_msgpack(self) -> bytes:
        """Serialize to msgpack bytes for HTTP transport."""

        import msgpack  # type: ignore[import]

        payload = {
            "format_version": OVERLAY_FORMAT_VERSION,
            "variable": self.variable,
            "level": self.level,
            "label": self.label,
            "unit": self.unit,
            "colormap": self.colormap,
            "visible": self.visible,
            "overlay_offset_hours": self.overlay_offset_hours,
            "time_label": self.time_label,
            "meta": self.meta,
            "lat": None if self.lat is None else np.asarray(self.lat, dtype=np.float32).tobytes(),
            "lat_shape": None if self.lat is None else list(np.asarray(self.lat).shape),
            "lon": None if self.lon is None else np.asarray(self.lon, dtype=np.float32).tobytes(),
            "lon_shape": None if self.lon is None else list(np.asarray(self.lon).shape),
            "frames": [
                {
                    "timestamp": frame.timestamp,
                    "data": np.asarray(frame.data, dtype=np.float32).tobytes(),
                    "shape": list(np.asarray(frame.data).shape),
                    "dtype": "float32",
                }
                for frame in self.frames
            ],
        }
        return msgpack.packb(payload, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "OverlayResult":
        """Deserialize bytes produced by :meth:`to_msgpack`."""

        import msgpack  # type: ignore[import]

        payload = msgpack.unpackb(data, raw=False)
        frames = [
            OverlayFrame(
                timestamp=frame_payload.get("timestamp", ""),
                data=(
                    np.frombuffer(frame_payload["data"], dtype=np.float32)
                    .reshape(tuple(frame_payload["shape"]))
                    .copy()
                ),
            )
            for frame_payload in payload["frames"]
        ]
        lat = None
        lon = None
        if payload.get("lat") is not None:
            lat = (
                np.frombuffer(payload["lat"], dtype=np.float32)
                .reshape(tuple(payload["lat_shape"]))
                .copy()
            )
        if payload.get("lon") is not None:
            lon = (
                np.frombuffer(payload["lon"], dtype=np.float32)
                .reshape(tuple(payload["lon_shape"]))
                .copy()
            )
        return cls(
            variable=payload["variable"],
            level=payload.get("level"),
            frames=frames,
            label=payload.get("label", payload["variable"]),
            unit=payload.get("unit", ""),
            colormap=payload.get("colormap", "viridis"),
            visible=payload.get("visible", True),
            overlay_offset_hours=int(payload.get("overlay_offset_hours", 0)),
            time_label=payload.get("time_label"),
            lat=lat,
            lon=lon,
            meta=payload.get("meta", {}),
        )

    def summary(self) -> str:
        level = "surface" if self.level is None else f"{self.level} hPa"
        span = "-"
        timestamps = self.timestamps
        if len(timestamps) == 1:
            span = timestamps[0]
        elif timestamps:
            span = f"{timestamps[0]}..{timestamps[-1]}"
        return (
            f"OverlayResult(variable={self.variable!r}, level={level}, "
            f"n_frames={len(self.frames)}, timestamps={span})"
        )

    def __repr__(self) -> str:
        return self.summary()
