"""
Shared constants, path wiring, and helpers for attribution method validation.

All submodule scripts import from here so they can also be run standalone.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

# ------------------------------------------------------------------
# Path wiring — must happen before any benchmark imports
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BM_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

for _p in (_BM_DIR, _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ------------------------------------------------------------------
# Result directories
# ------------------------------------------------------------------
_RESULTS_ROOT = os.environ.get(
    "AURORA_XAI_RESULTS_DIR",
    os.path.join(_ROOT, "results"),
)

# Corrected-precision 6 h point benchmark used by the validation suite.
EVENT_SUITE_DIR = os.environ.get(
    "ZWD_SEARCHLIGHT_RESULTS_DIR",
    os.path.join(_RESULTS_ROOT, "searchlight_igfix_6h_point"),
)
OUTPUT_DIR = os.environ.get(
    "ATTRIBUTION_VALIDATION_RESULTS_DIR",
    os.path.join(_RESULTS_ROOT, "attribution_method_validation"),
)

# ------------------------------------------------------------------
# Representative case for stability / randomization / completeness
# ------------------------------------------------------------------
REP_CASE_TARGET   = "ticino"
REP_CASE_INIT     = datetime(2020, 4, 20, 12, 0)
REP_CASE_ID_POINT = "ticino_2020042012_strong__point"
REP_SCALE         = "local"

# ------------------------------------------------------------------
# Event-suite cases used by the RISE convergence diagnostic
# ------------------------------------------------------------------
EVENT_CASES: list[tuple] = [
    ("ticino",     datetime(2020,  4, 20, 12, 0), "strong"),
    ("ticino",     datetime(2020, 11,  2, 12, 0), "secondary"),
    ("california", datetime(2021, 10, 24, 12, 0), "strong"),
    ("california", datetime(2023,  3, 10, 12, 0), "secondary"),
    ("japan",      datetime(2020,  2, 16, 12, 0), "strong"),
    ("japan",      datetime(2023, 12, 12,  0, 0), "secondary"),
]


# ------------------------------------------------------------------
# Lazy benchmark import (triggers Aurora + torch; call only when GPU needed)
# ------------------------------------------------------------------
def import_benchmark():
    """Import all heavy benchmark symbols. Call once per process."""
    from searchlight_benchmark import (  # noqa: E402
        setup_model, run_saliency, run_ig,
        make_q850_target, _forward,
        _saved_tensors_cpu_context, _RolloutForwardWrapper,
    )
    from searchlight_data import load_case, make_batch  # noqa: E402
    from searchlight_tasks import (  # noqa: E402
        TARGETS, SCALES, generate_mask_centers, gaussian_mask,
        cos_lat_weights,
    )
    from searchlight_ground_truth import smoothed_zwd_baseline  # noqa: E402
    from searchlight_metrics import pool_attribution, spearman  # noqa: E402
    from xia_methods.ig import integrated_gradients as xia_ig  # noqa: E402
    return (
        setup_model, run_saliency, run_ig, make_q850_target, _forward,
        _saved_tensors_cpu_context, _RolloutForwardWrapper,
        load_case, make_batch,
        TARGETS, SCALES, generate_mask_centers, gaussian_mask,
        cos_lat_weights, smoothed_zwd_baseline,
        pool_attribution, spearman,
        xia_ig,
    )


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.flatten(), b.flatten())
    return float(r)


def load_attr(result_dir: str, case_id: str, scale: str, method: str) -> np.ndarray:
    path = os.path.join(result_dir, "per_case", case_id, scale, f"{method}_attr.npy")
    return np.load(path).astype(np.float32)


def load_gt_json(result_dir: str, case_id: str, scale: str, mode: str = "plain") -> dict:
    fname = "ground_truth.json" if mode == "plain" else f"ground_truth_{mode}.json"
    path = os.path.join(result_dir, "per_case", case_id, scale, fname)
    with open(path) as f:
        return json.load(f)


def save_json(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved: {path}")
