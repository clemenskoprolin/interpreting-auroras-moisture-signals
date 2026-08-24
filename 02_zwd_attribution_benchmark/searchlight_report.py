"""
Reporting for the ZWD searchlight benchmark.

Writes per-case files under:
    <output_dir>/per_case/<case_id>/<scale>/ground_truth.json
    <output_dir>/per_case/<case_id>/<scale>/<method>_attr.npy
    <output_dir>/per_case/<case_id>/<scale>/<method>_metrics.json
    <output_dir>/per_case/<case_id>/<scale>/<method>_pooled.npy

Leaderboard aggregation lives in generate_leaderboard.py.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np


def _to_jsonable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_ground_truth(
    *,
    output_dir: str,
    case_id: str,
    base_case_id: str,
    scale: str,
    masks,
    G: np.ndarray,
    S: np.ndarray,
    f_plus: np.ndarray,
    f_minus: np.ndarray,
    mode: str = "plain",
    target_mode: str = "box",
    target_meta: dict | None = None,
) -> str:
    out_dir = os.path.join(output_dir, "per_case", case_id, scale)
    os.makedirs(out_dir, exist_ok=True)
    fname = "ground_truth.json" if mode == "plain" else f"ground_truth_{mode}.json"
    path = os.path.join(out_dir, fname)
    payload = {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "scale": scale,
        "gt_mode": mode,
        "target_mode": target_mode,
        "target_meta": target_meta or {},
        "masks": [
            {
                "key": m.key, "role": m.role,
                "center_lat": m.center_lat, "center_lon": m.center_lon,
            }
            for m in masks
        ],
        "G": G.tolist(),
        "S": S.tolist(),
        "f_plus": f_plus.tolist(),
        "f_minus": f_minus.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_method_result(
    *,
    output_dir: str,
    case_id: str,
    base_case_id: str,
    scale: str,
    method: str,
    attr_map: np.ndarray,
    metrics,
    runtime_s: float,
    target_mode: str = "box",
    target_meta: dict | None = None,
) -> None:
    out_dir = os.path.join(output_dir, "per_case", case_id, scale)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{method}_attr.npy"),
            attr_map.astype(np.float32))
    np.save(os.path.join(out_dir, f"{method}_pooled_Amag.npy"),
            metrics.pooled_A_mag.astype(np.float64))
    np.save(os.path.join(out_dir, f"{method}_pooled_Asign.npy"),
            metrics.pooled_A_sign.astype(np.float64))
    payload = {
        "method": method,
        "case_id": case_id,
        "base_case_id": base_case_id,
        "scale": scale,
        "target_mode": target_mode,
        "target_meta": target_meta or {},
        "runtime_s": runtime_s,
        "rho_mag": metrics.rho_mag,
        "rho_signed": metrics.rho_signed,
        "ndcg_at_10": metrics.ndcg_at_10,
        "top10_recall": metrics.top10_recall,
        "remote_gap": metrics.remote_gap,
        "n_masks": metrics.n_masks,
        "n_remote": metrics.n_remote,
    }
    with open(os.path.join(out_dir, f"{method}_metrics.json"), "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)




def plot_case_scatter(
    *,
    output_dir: str,
    case_id: str,
    scale: str,
    method: str,
    pooled_A_mag: np.ndarray,
    G: np.ndarray,
    masks,
) -> None:
    """Scatter of pooled |attribution| vs ground-truth G per mask."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  plot skipped ({exc})")
        return

    roles = np.array([m.role for m in masks])
    fig, ax = plt.subplots(figsize=(5, 4))
    for role, color in (("near", "C0"), ("remote", "C3")):
        sel = roles == role
        if sel.any():
            ax.scatter(G[sel], pooled_A_mag[sel], label=role, s=20, alpha=0.8)
    ax.set_xlabel("Ground-truth |Δf|/2 (G_r)")
    ax.set_ylabel(f"Pooled |attr_zwd| ({method})")
    ax.set_title(f"{case_id} — {scale} — {method}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(output_dir, "per_case", case_id, scale)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{method}_scatter.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
