"""On-the-fly XIA computation for Microsoft Aurora.

The public API exposes target builders, attribution runners, overlay pulls, and
result containers. Computation can run in-process or through a remote listener.
"""

from .schema.spec import Target, TargetSpec
from .schema.result import XiaResult, XiaFrame
from .schema.overlay import OverlayResult, OverlayFrame
from .api.dispatch import session_timestamps
from .api.listener import listen_for_request
from .api.methods import (
    pull_overlay,
    run_ig,
    run_rise,
    run_rollout,
    run_saliency,
    run_vit_cx,
)

__all__ = [
    "Target",
    "TargetSpec",
    "XiaResult",
    "XiaFrame",
    "OverlayResult",
    "OverlayFrame",
    "pull_overlay",
    "session_timestamps",
    "run_saliency",
    "run_ig",
    "run_rise",
    "run_vit_cx",
    "run_rollout",
    "listen_for_request",
]
