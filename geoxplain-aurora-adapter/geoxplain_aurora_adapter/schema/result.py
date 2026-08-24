"""Result containers and serialization for GeoXplain XIA bundles.

A ``XiaResult`` contains one or more ``XiaFrame`` objects, each with a target,
timestamp, attribution maps, and display metadata. Results can be saved as
``.xia.npz`` archives or packed as msgpack bytes for remote transport.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .spec import TargetSpec

FORMAT_VERSION = 2


@dataclass
class XiaFrame:
    """A single time step of an :class:`XiaResult` bundle.

    Attributes
    ----------
    target:
        The scalar that was explained (what variable, where, when).
    timestamp:
        ISO-8601 string of the **requested time** — Aurora's most-recent input
        step t1 — which is what the viewer displays and equals
        ``target.timestamp``. The explained prediction is the 6 h-ahead step
        t2 = t1 + lead; ``meta["lead_hours"]`` records the lead, and
        ``meta["input_timestamp"]`` echoes t1 for convenience.
    attributions:
        ``attributions[wrt_var][level_key]`` is a ``(721, 1440)`` float32
        array recording the sensitivity of the target to the input field
        ``wrt_var`` at ``level_key``.  Atmospheric level keys have the form
        ``"z-{N}"`` (higher ``N`` = higher in the tool); surface variables
        use ``"sfc"``.
    diverging:
        Whether the attribution maps contain significant negative values.
        Auto-detected from the sign distribution; controls whether the
        visualization widget uses a diverging colormap.
    meta:
        Optional per-frame diagnostic metadata: ``"target_score"`` (scalar
        value being explained, saliency/IG only), ``"runtime_s"``, etc.
    """

    target: TargetSpec
    timestamp: str
    attributions: dict[str, dict[str, np.ndarray]]
    diverging: bool
    meta: dict = field(default_factory=dict)

    def as_widget_dict(self) -> dict:
        """Return this frame's target as a geoxplain-compatible dict.

        The visualization side's importer calls ``as_widget_dict()`` on each
        *frame* (not on ``frame.target``) to place the target point/box.  When
        an in-memory :class:`XiaResult` is handed straight to the widget — the
        live/remote path, with no ``.xia.npz`` round-trip — the frame object is
        this class, so it must expose the method itself; otherwise the target
        is silently dropped.  Delegates to :meth:`TargetSpec.as_widget_dict`.
        """
        if self.target is None:
            return {}
        return self.target.as_widget_dict()


@dataclass
class XiaResult:
    """Self-describing XIA attribution bundle (one method, one or more frames).

    Attributes
    ----------
    method:
        XIA method id: ``"saliency"``, ``"ig"``, ``"rise"``, or ``"vit_cx"``.
        A stable machine identifier (the compute/dispatch layers branch on it).
    method_label:
        Human-readable method name for display, e.g. ``"Integrated Gradients"``.
        Empty when unknown; consumers fall back to ``method`` in that case.
    frames:
        One :class:`XiaFrame` per time step.  Single-frame results are the
        common case; use :meth:`single` to build one ergonomically.
    layer_labels:
        Optional ``{level_key: display_name}`` map shared across frames, e.g.
        ``{"z-2": "850 hPa", "sfc": "Surface"}``.  Absent keys fall back to the
        bare number (``"z-2"`` → ``"2"``) or ``"Surface"`` for ``"sfc"``.
    meta:
        Bundle-level diagnostic metadata: ``"checkpoint_hash"``, ``"host"``,
        ``"slurm_job_id"``, etc.
    """

    method: str
    frames: list[XiaFrame]
    layer_labels: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    method_label: str = ""

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def single(
        cls,
        method: str,
        target: TargetSpec,
        timestamp: str,
        attributions: dict[str, dict[str, np.ndarray]],
        diverging: bool,
        meta: dict | None = None,
        *,
        layer_labels: dict[str, str] | None = None,
        frame_meta: dict | None = None,
        method_label: str = "",
    ) -> "XiaResult":
        """Build a one-frame bundle.

        ``meta`` becomes the bundle-level metadata; ``frame_meta`` (if given)
        is attached to the single frame.  For the common case where there is
        only one frame, passing diagnostics via ``meta`` is fine.
        """
        return cls(
            method=method,
            method_label=method_label,
            frames=[
                XiaFrame(
                    target=target,
                    timestamp=timestamp,
                    attributions=attributions,
                    diverging=diverging,
                    meta=frame_meta or {},
                )
            ],
            layer_labels=dict(layer_labels or {}),
            meta=dict(meta or {}),
        )

    # ── persistence ──────────────────────────────────────────────────────────

    def _meta_payload(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "method": self.method,
            "method_label": self.method_label,
            "meta": self.meta,
            "layer_labels": self.layer_labels,
            "frames": [
                {
                    "timestamp": frame.timestamp,
                    "target": frame.target.to_dict(),
                    "diverging": frame.diverging,
                    "meta": frame.meta,
                    "attributions_index": {
                        wrt_var: list(levels.keys())
                        for wrt_var, levels in frame.attributions.items()
                    },
                }
                for frame in self.frames
            ],
        }

    def save(self, path: str | os.PathLike) -> None:
        """Save to a ``.xia.npz`` archive.

        If *path* does not end with ``".xia.npz"``, the suffix is appended
        automatically.
        """
        path = os.fspath(path)
        if not path.endswith(".xia.npz"):
            path = path + ".xia.npz"

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(self._meta_payload(), indent=2))
            for i, frame in enumerate(self.frames):
                for wrt_var, levels in frame.attributions.items():
                    for level_key, arr in levels.items():
                        buf = io.BytesIO()
                        np.save(buf, arr.astype(np.float32))
                        zf.writestr(f"f{i}__{wrt_var}__{level_key}.npy", buf.getvalue())

    @classmethod
    def load(cls, path: str | os.PathLike) -> "XiaResult":
        """Load from a ``.xia.npz`` archive (``format_version == 2``)."""
        path = os.fspath(path)
        with zipfile.ZipFile(path, "r") as zf:
            meta_payload = json.loads(zf.read("meta.json"))
            frames: list[XiaFrame] = []
            for i, frame_meta in enumerate(meta_payload["frames"]):
                index: dict[str, list[str]] = frame_meta["attributions_index"]
                attributions: dict[str, dict[str, np.ndarray]] = {}
                for wrt_var, level_keys in index.items():
                    attributions[wrt_var] = {}
                    for level_key in level_keys:
                        member = f"f{i}__{wrt_var}__{level_key}.npy"
                        buf = io.BytesIO(zf.read(member))
                        attributions[wrt_var][level_key] = np.load(buf)
                frames.append(
                    XiaFrame(
                        target=TargetSpec.from_dict(frame_meta["target"]),
                        timestamp=frame_meta["timestamp"],
                        attributions=attributions,
                        diverging=frame_meta["diverging"],
                        meta=frame_meta.get("meta", {}),
                    )
                )

        return cls(
            method=meta_payload["method"],
            method_label=meta_payload.get("method_label", ""),
            frames=frames,
            layer_labels=meta_payload.get("layer_labels", {}),
            meta=meta_payload.get("meta", {}),
        )

    # ── wire format ──────────────────────────────────────────────────────────

    def to_msgpack(self) -> bytes:
        """Serialize to msgpack bytes for HTTP transport.

        Arrays are embedded as raw float32 LE bytes + shape info.  The
        result is designed to be deserialised by ``XiaResult.from_msgpack``.
        """
        import msgpack  # type: ignore[import]

        frames_payload: list[dict] = []
        for frame in self.frames:
            arrays: dict[str, dict] = {}
            for wrt_var, levels in frame.attributions.items():
                for level_key, arr in levels.items():
                    a32 = np.asarray(arr, dtype=np.float32)
                    arrays[f"{wrt_var}__{level_key}"] = {
                        "data": a32.tobytes(),
                        "shape": list(a32.shape),
                        "dtype": "float32",
                    }
            frames_payload.append(
                {
                    "target": frame.target.to_dict(),
                    "timestamp": frame.timestamp,
                    "diverging": frame.diverging,
                    "meta": frame.meta,
                    "arrays": arrays,
                }
            )

        payload = {
            "format_version": FORMAT_VERSION,
            "method": self.method,
            "method_label": self.method_label,
            "meta": self.meta,
            "layer_labels": self.layer_labels,
            "frames": frames_payload,
        }
        return msgpack.packb(payload, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "XiaResult":
        """Deserialize from msgpack bytes produced by ``XiaResult.to_msgpack``."""
        import msgpack  # type: ignore[import]

        payload = msgpack.unpackb(data, raw=False)

        frames: list[XiaFrame] = []
        for frame_payload in payload["frames"]:
            attributions: dict[str, dict[str, np.ndarray]] = {}
            for key, arr_data in frame_payload["arrays"].items():
                wrt_var, level_key = key.split("__", 1)
                shape = tuple(arr_data["shape"])
                arr = (
                    np.frombuffer(arr_data["data"], dtype=np.float32)
                    .reshape(shape)
                    .copy()
                )
                attributions.setdefault(wrt_var, {})[level_key] = arr
            frames.append(
                XiaFrame(
                    target=TargetSpec.from_dict(frame_payload["target"]),
                    timestamp=frame_payload["timestamp"],
                    attributions=attributions,
                    diverging=frame_payload["diverging"],
                    meta=frame_payload.get("meta", {}),
                )
            )

        return cls(
            method=payload["method"],
            method_label=payload.get("method_label", ""),
            frames=frames,
            layer_labels=payload.get("layer_labels", {}),
            meta=payload.get("meta", {}),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_maps = sum(
            len(levels)
            for frame in self.frames
            for levels in frame.attributions.values()
        )
        timestamps = [f.timestamp for f in self.frames]
        if len(timestamps) == 1:
            span = timestamps[0]
        elif timestamps:
            span = f"{timestamps[0]}..{timestamps[-1]}"
        else:
            span = "-"
        return (
            f"XiaResult(method={self.method!r}, n_frames={len(self.frames)}, "
            f"timestamps={span}, n_maps={n_maps})"
        )

    def __repr__(self) -> str:
        return self.summary()
