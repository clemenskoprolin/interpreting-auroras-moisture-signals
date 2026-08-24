"""
conditional_stages.py — Stage A/B/C/D implementations.

  A  run_stage_a   Output ablation: score(mode) × lead
  B  run_stage_b   Attribution: ZWD vs q saliency overlap
  C  run_stage_c   Timing: t0/t1 split + rollout curve
  D  run_stage_d   Routing: hidden-state RMS + dec_s1 attention contrast
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any

import numpy as np  # noqa: F401 (used in _attribution_overlap_metrics and savez)
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SEARCHLIGHT_DIR)

from searchlight_benchmark import (  # noqa: E402
    _RolloutForwardWrapper, _forward, _gpu_sync_and_gc,
)
from searchlight_data import CaseData, make_batch  # noqa: E402
from searchlight_tasks import Case, TARGETS, TargetRegion  # noqa: E402
from xia_methods.saliency import saliency as _xia_saliency  # noqa: E402

from conditional_data import (  # noqa: E402
    _append_csv, _ensure_dir,
    build_localized_residual_masks,
    make_localized_residual_zwd,
    make_qhat_zwd,
    make_random_same_month_zwd,
)

DEC_S1_LAYER_IDX = 1


def _clock() -> float:
    return time.perf_counter()


# ─── Shared forward helper ────────────────────────────────────────────────────

def run_single_forward(
    model,
    case: CaseData,
    device: str,
    zwd_override: torch.Tensor | None,
    target_fn,
    lead_steps: int,
) -> float:
    batch = make_batch(case, device, zwd_override=zwd_override)
    with torch.no_grad():
        if lead_steps == 1:
            pred = _forward(model, batch)
        else:
            wrapper = _RolloutForwardWrapper(model, lead_steps)
            pred = wrapper.forward(batch)
    # `target_fn` may be a dict of named scalar fns; they all read the SAME
    # prediction, so extra diagnostics cost no additional forward pass.
    if isinstance(target_fn, dict):
        score = {k: float(fn(pred).detach().float().cpu().item())
                 for k, fn in target_fn.items() if fn is not None}
    else:
        score = float(target_fn(pred).detach().float().cpu().item())
    del batch, pred
    _gpu_sync_and_gc()
    return score


# ─── Stage A: output ablation ─────────────────────────────────────────────────

def run_stage_a(
    model,
    case: CaseData,
    case_obj: Case,
    target_fn,
    device: str,
    replacements: dict[str, torch.Tensor | None],
    leads_hours: list[int],
    out_dir: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_id = case_obj.case_id

    _PRIMARY = "q850"
    for lead_h in leads_hours:
        lead_steps = lead_h // 6
        all_scores: dict[str, dict[str, float]] = {}
        for mode, zwd_ovr in replacements.items():
            t0 = _clock()
            sc = run_single_forward(model, case, device, zwd_ovr, target_fn, lead_steps)
            if not isinstance(sc, dict):
                sc = {_PRIMARY: sc}
            all_scores[mode] = sc
            extra = "  ".join(f"{k}={v:.6f}" for k, v in sc.items() if k != _PRIMARY)
            print(f"    [{case_id}] lead={lead_h}h mode={mode} "
                  f"score={sc[_PRIMARY]:.6f} {extra} ({_clock()-t0:.1f}s)")

        scores = {m: s[_PRIMARY] for m, s in all_scores.items()}
        s_true = scores.get("true", float("nan"))
        for mode, score_mode in scores.items():
            row = {
                "case_id": case_id,
                "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(),
                "role": case_obj.role,
                "lead_h": lead_h,
                "mode": mode,
                "score": score_mode,
                "delta_vs_true": score_mode - s_true,
                "score_true": s_true,
                "delta_beyond_humidity": s_true - scores.get("qhat", float("nan")),
                "delta_total_zwd": s_true - scores.get("climatology", float("nan")),
                "delta_residual": scores.get("residual_only", float("nan")) - scores.get("climatology", float("nan")),
                "delta_humidity_matched_swap": scores.get("matched_swap", float("nan")) - scores.get("qhat", float("nan")),
            }
            # Secondary diagnostics from the same forward pass (e.g. the
            # above-ground-only q850 mean, which is the physically meaningful
            # one over high terrain).
            for k, v in all_scores[mode].items():
                if k == _PRIMARY:
                    continue
                row[f"score_{k}"] = v
                row[f"delta_vs_true_{k}"] = v - all_scores.get("true", {}).get(k, float("nan"))
            rows.append(row)

        qhat_rows = []
        for mode, score_mode in scores.items():
            if not mode.startswith("qhat"):
                continue
            delta = s_true - score_mode
            qhat_rows.append({
                "case_id": case_id,
                "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(),
                "role": case_obj.role,
                "lead_h": lead_h,
                "qhat_mode": mode,
                "score_true": s_true,
                "score_qhat": score_mode,
                "delta_true_minus_qhat": delta,
                "rel_abs_delta": abs(delta) / (abs(s_true) + 1e-10),
            })
        _append_csv(os.path.join(out_dir, "qhat_robustness.csv"), qhat_rows)

    _append_csv(os.path.join(out_dir, "ablation_scores.csv"), rows)
    return rows


# ─── Random-donor null: repeated same-month controls ─────────────────────────

def run_random_null_stage(
    model,
    case: CaseData,
    case_obj: Case,
    target_fn,
    device: str,
    ref_zwds: list[np.ndarray],
    rng: np.random.Generator,
    leads_hours: list[int],
    out_dir: str,
    *,
    n_random: int,
    qhat_zwd: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if n_random <= 0:
        return [], []
    if qhat_zwd is None:
        qhat_zwd = make_qhat_zwd(case)

    case_id = case_obj.case_id
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for lead_h in leads_hours:
        lead_steps = lead_h // 6
        s_true = run_single_forward(model, case, device, None, target_fn, lead_steps)
        s_qhat = run_single_forward(model, case, device, qhat_zwd, target_fn, lead_steps)
        observed = s_true - s_qhat

        null_deltas = []
        for donor_idx in range(n_random):
            donor_zwd = make_random_same_month_zwd(case, ref_zwds, rng)
            score_donor = run_single_forward(model, case, device, donor_zwd, target_fn, lead_steps)
            delta = score_donor - s_qhat
            null_deltas.append(delta)
            rows.append({
                "case_id": case_id,
                "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(),
                "role": case_obj.role,
                "lead_h": lead_h,
                "donor_idx": donor_idx,
                "score_true": s_true,
                "score_qhat": s_qhat,
                "score_donor": score_donor,
                "observed_true_minus_qhat": observed,
                "null_donor_minus_qhat": delta,
            })

        arr = np.asarray(null_deltas, dtype=np.float64)
        abs_p = float((1 + np.sum(np.abs(arr) >= abs(observed))) / (arr.size + 1))
        signed_rank = float(np.mean(arr <= observed)) if arr.size else float("nan")
        summary_rows.append({
            "case_id": case_id,
            "target": case_obj.target,
            "init_time": case_obj.init_time.isoformat(),
            "role": case_obj.role,
            "lead_h": lead_h,
            "n_random": int(arr.size),
            "score_true": s_true,
            "score_qhat": s_qhat,
            "observed_true_minus_qhat": observed,
            "null_mean": float(arr.mean()) if arr.size else float("nan"),
            "null_std": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "null_p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
            "null_p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
            "null_p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
            "abs_two_sided_empirical_p": abs_p,
            "signed_percentile": signed_rank,
        })
        print(f"    [{case_id}] null lead={lead_h}h: "
              f"obs={observed:+.4f} null_mean={summary_rows[-1]['null_mean']:+.4f} "
              f"p_abs={abs_p:.3f}")

    _append_csv(os.path.join(out_dir, "random_null_scores.csv"), rows)
    _append_csv(os.path.join(out_dir, "random_null_summary.csv"), summary_rows)
    return rows, summary_rows


# ─── Stage E: localized residual mechanism ───────────────────────────────────

def run_stage_e_localized_residual(
    model,
    case: CaseData,
    case_obj: Case,
    target_fn,
    device: str,
    qhat_zwd: torch.Tensor,
    leads_hours: list[int],
    out_dir: str,
) -> list[dict[str, Any]]:
    case_id = case_obj.case_id
    masks = build_localized_residual_masks(case, case_obj)
    rows: list[dict[str, Any]] = []

    for lead_h in leads_hours:
        lead_steps = lead_h // 6
        s_true = run_single_forward(model, case, device, None, target_fn, lead_steps)
        s_qhat = run_single_forward(model, case, device, qhat_zwd, target_fn, lead_steps)
        full_delta = s_true - s_qhat

        full_t1_score = None
        for mask_name, mask_hw in masks.items():
            zwd_loc = make_localized_residual_zwd(case, qhat_zwd, mask_hw, timestep=1)
            score = run_single_forward(model, case, device, zwd_loc, target_fn, lead_steps)
            if mask_name == "full_t1":
                full_t1_score = score
            rows.append({
                "case_id": case_id,
                "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(),
                "role": case_obj.role,
                "lead_h": lead_h,
                "mask_name": mask_name,
                "mask_mean": float(mask_hw.float().mean().item()),
                "mask_pixels": int((mask_hw > 0).sum().item()),
                "score_true": s_true,
                "score_qhat": s_qhat,
                "score_localized": score,
                "delta_true_minus_qhat": full_delta,
                "delta_localized_minus_qhat": score - s_qhat,
                "fraction_of_true_minus_qhat": (
                    (score - s_qhat) / full_delta if abs(full_delta) > 1e-10 else float("nan")
                ),
                "score_full_t1": None,
                "fraction_of_full_t1": None,
            })

        if full_t1_score is not None:
            full_t1_delta = full_t1_score - s_qhat
            for r in rows:
                if r["case_id"] == case_id and r["lead_h"] == lead_h:
                    r["score_full_t1"] = full_t1_score
                    r["fraction_of_full_t1"] = (
                        (r["delta_localized_minus_qhat"] / full_t1_delta)
                        if abs(full_t1_delta) > 1e-10 else float("nan")
                    )
        print(f"    [{case_id}] Stage E lead={lead_h}h: "
              f"full={full_delta:+.4f} full_t1={(full_t1_score - s_qhat) if full_t1_score is not None else float('nan'):+.4f}")

    _append_csv(os.path.join(out_dir, "localized_residual.csv"), rows)
    return rows


# ─── Stage B: attribution mechanism ──────────────────────────────────────────

def _compute_saliency_maps(
    model,
    case: CaseData,
    device: str,
    zwd_override: torch.Tensor | None,
    target_fn,
) -> dict[str, np.ndarray | None]:
    def batch_fn(requires_grad: bool):
        return make_batch(
            case, device,
            requires_grad_surf=("zwd",) if requires_grad else (),
            requires_grad_atmos=("q",) if requires_grad else (),
            zwd_override=zwd_override,
        )
    result = _xia_saliency(model, batch_fn, target_fn, device,
                           atmos_var_names=("q",), surf_var_names=("zwd",))
    return result["grads"]


def _attribution_overlap_metrics(
    grad_zwd: np.ndarray,   # (1, 2, H, W)
    grad_q: np.ndarray,     # (1, 2, L, H, W)
    level_idx_850: int,
    target: TargetRegion,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> dict[str, float]:
    from scipy.stats import spearmanr
    from searchlight_tasks import great_circle_km

    g_zwd = np.abs(grad_zwd[0, 1]).ravel()
    g_q850 = np.abs(grad_q[0, 1, level_idx_850]).ravel()

    r, pval = spearmanr(g_zwd, g_q850)

    k = max(1, int(0.01 * g_zwd.size))
    topk_overlap = len(
        set(np.argpartition(g_zwd, -k)[-k:]) & set(np.argpartition(g_q850, -k)[-k:])
    ) / k

    def _com(g_flat):
        g_hw = g_flat.reshape(lat_vals.size, lon_vals.size)
        g_hw = g_hw / (g_hw.sum() + 1e-30)
        return float(np.sum(lat_vals[:, None] * g_hw)), float(np.sum(lon_vals[None, :] * g_hw))

    lat_z, lon_z = _com(g_zwd)
    lat_q, lon_q = _com(g_q850)
    com_shift = float(great_circle_km(lat_z, lon_z, lat_q, lon_q))

    def _dist_to_target(lat, lon):
        return float(great_circle_km(target.center_lat, target.center_lon, lat, lon))

    return {
        "spearman_r": float(r),
        "spearman_pval": float(pval),
        "topk_overlap": float(topk_overlap),
        "com_shift_km": com_shift,
        "com_lat_zwd": lat_z,
        "com_lon_zwd": lon_z,
        "com_lat_q850": lat_q,
        "com_lon_q850": lon_q,
        "com_dist_target_zwd_km": _dist_to_target(lat_z, lon_z),
        "com_dist_target_q850_km": _dist_to_target(lat_q, lon_q),
    }


def run_stage_b(
    model,
    case: CaseData,
    case_obj: Case,
    target_fn,
    device: str,
    replacements: dict[str, torch.Tensor | None],
    level_idx_850: int,
    out_dir: str,
) -> list[dict[str, Any]]:
    target = TARGETS[case_obj.target]
    case_id = case_obj.case_id
    npz_dir = os.path.join(out_dir, "attribution_maps", case_id)
    _ensure_dir(npz_dir)
    rows: list[dict[str, Any]] = []

    for mode in ("true", "qhat", "residual_only"):
        if mode not in replacements:
            continue
        t0 = _clock()
        grads = _compute_saliency_maps(model, case, device, replacements[mode], target_fn)
        elapsed = _clock() - t0

        grad_zwd = grads.get("zwd")
        grad_q = grads.get("q")
        if grad_zwd is None or grad_q is None:
            print(f"    [{case_id}] Stage B {mode}: gradient is None, skipping.")
            continue

        np.savez_compressed(os.path.join(npz_dir, f"saliency_{mode}.npz"),
                            grad_zwd=grad_zwd, grad_q=grad_q)

        metrics = _attribution_overlap_metrics(
            grad_zwd, grad_q, level_idx_850, target, case.lat_vals, case.lon_vals,
        )
        rows.append({
            "case_id": case_id, "target": case_obj.target,
            "init_time": case_obj.init_time.isoformat(), "role": case_obj.role,
            "mode": mode, "elapsed_s": elapsed, **metrics,
        })
        print(f"    [{case_id}] Stage B {mode}: "
              f"spearman_r={metrics['spearman_r']:.3f} "
              f"topk_overlap={metrics['topk_overlap']:.3f} ({elapsed:.1f}s)")

    _append_csv(os.path.join(out_dir, "attribution_mechanism.csv"), rows)
    return rows


# ─── Stage C: timing mechanism ────────────────────────────────────────────────

def _t_only_override(
    case: CaseData,
    replacement: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    out = case.surf_cpu["zwd"].clone()
    out[0, timestep] = replacement[0, timestep]
    return out


def run_stage_c(
    model,
    case: CaseData,
    case_obj: Case,
    target_fn,
    device: str,
    qhat_zwd: torch.Tensor,
    leads_hours: list[int],
    out_dir: str,
) -> list[dict[str, Any]]:
    case_id = case_obj.case_id
    rows: list[dict[str, Any]] = []

    overrides = {
        "true":        None,
        "qhat_both":   qhat_zwd,
        "qhat_t0only": _t_only_override(case, qhat_zwd, 0),
        "qhat_t1only": _t_only_override(case, qhat_zwd, 1),
    }

    for lead_h in leads_hours:
        lead_steps = lead_h // 6
        scores = {
            lbl: run_single_forward(model, case, device, ovr, target_fn, lead_steps)
            for lbl, ovr in overrides.items()
        }
        s_true, s_both, s_t0, s_t1 = (
            scores["true"], scores["qhat_both"],
            scores["qhat_t0only"], scores["qhat_t1only"],
        )
        d_both = s_true - s_both
        t1_frac = abs(s_true - s_t1) / (abs(d_both) + 1e-10) if abs(d_both) > 1e-10 else float("nan")

        rows.append({
            "case_id": case_id, "target": case_obj.target,
            "init_time": case_obj.init_time.isoformat(), "role": case_obj.role,
            "lead_h": lead_h,
            "score_true": s_true, "score_qhat_both": s_both,
            "score_qhat_t0only": s_t0, "score_qhat_t1only": s_t1,
            "delta_both": d_both,
            "delta_t0_only": s_true - s_t0,
            "delta_t1_only": s_true - s_t1,
            "t1_fraction_of_both": t1_frac,
        })
        print(f"    [{case_id}] Stage C lead={lead_h}h: "
              f"Δboth={d_both:.4f} Δt0={s_true-s_t0:.4f} "
              f"Δt1={s_true-s_t1:.4f} t1_frac={t1_frac:.2f}")

    _append_csv(os.path.join(out_dir, "timing_mechanism.csv"), rows)
    return rows


# ─── Stage D: routing mechanism ───────────────────────────────────────────────

def _block_list(model) -> list[tuple[Any, str, str]]:
    specs = []
    for s, layer in enumerate(model.backbone.encoder_layers):
        for b, blk in enumerate(layer.blocks):
            specs.append((blk, f"enc_s{s}_b{b:02d}", "encoder"))
    for s, layer in enumerate(model.backbone.decoder_layers):
        for b, blk in enumerate(layer.blocks):
            specs.append((blk, f"dec_s{s}_b{b:02d}", "decoder"))
    return specs


def _collect_hidden_states(model, batch, block_specs) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(key):
        def _hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[key] = t.detach().float().cpu()
        return _hook

    for mod, key, _ in block_specs:
        handles.append(mod.register_forward_hook(_make_hook(key)))
    try:
        with torch.no_grad():
            _forward(model, batch)
    finally:
        for h in handles:
            h.remove()
    return captures


def _dec_s1_attention_contrast(
    model, case: CaseData, device: str, qhat_zwd: torch.Tensor,
    case_id: str, case_obj: Case,
) -> list[dict[str, Any]]:
    """Per-head attention delta for dec_s1: qhat vs true."""
    dec_s1 = model.backbone.decoder_layers[DEC_S1_LAYER_IDX]
    blocks = list(dec_s1.blocks)
    if not blocks:
        return []

    captured: dict[str, torch.Tensor] = {}
    patch_info = []

    for b_idx, blk in enumerate(blocks):
        wa = getattr(blk, "attn", None)
        if wa is None:
            continue
        orig_fwd = wa.forward
        key = f"dec_s1_b{b_idx:02d}_attn"

        def _make_patched(orig_fn, capture_key):
            def _patched(*args, **kwargs):
                orig_sdpa = F.scaled_dot_product_attention

                def _explicit(q, kk, v, attn_mask=None, dropout_p=0.0, is_causal=False):
                    scale = q.shape[-1] ** -0.5
                    qk = torch.matmul(q, kk.transpose(-2, -1)) * scale
                    if attn_mask is not None:
                        qk = qk + attn_mask
                    a = torch.softmax(qk, dim=-1)
                    captured[capture_key] = a.detach().float().cpu()
                    return torch.matmul(a, v)

                F.scaled_dot_product_attention = _explicit
                try:
                    return orig_fn(*args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = orig_sdpa
            return _patched

        wa.forward = _make_patched(orig_fwd, key)
        patch_info.append((wa, orig_fwd))

    rows: list[dict[str, Any]] = []
    try:
        batch_true = make_batch(case, device, zwd_override=None)
        with torch.no_grad():
            _forward(model, batch_true)
        A_true = {k: v.clone() for k, v in captured.items()}
        captured.clear()
        del batch_true
        _gpu_sync_and_gc()

        batch_qhat = make_batch(case, device, zwd_override=qhat_zwd)
        with torch.no_grad():
            _forward(model, batch_qhat)
        A_qhat = dict(captured)
        captured.clear()
        del batch_qhat
        _gpu_sync_and_gc()

        for b_idx in range(len(blocks)):
            key = f"dec_s1_b{b_idx:02d}_attn"
            if key not in A_true or key not in A_qhat:
                continue
            dA = (A_qhat[key] - A_true[key]).float()
            _nan = float("nan")
            base_row = {
                "case_id": case_id, "target": case_obj.target,
                "init_time": case_obj.init_time.isoformat(), "role": case_obj.role,
                "block_key": f"dec_s1_b{b_idx:02d}", "family": "dec_s1_attn",
                # hidden-state fields not applicable here
                "baseline_rms": _nan, "delta_rms": _nan,
                "relative_rms": _nan, "cosine_sim": _nan,
            }
            if dA.dim() == 4:  # (windows, heads, Q, K)
                for h in range(dA.shape[1]):
                    rows.append({**base_row, "head_idx": h,
                                 "attn_delta_rms": float(torch.sqrt((dA[:, h] ** 2).mean()).item())})
            else:
                rows.append({**base_row, "head_idx": -1,
                             "attn_delta_rms": float(torch.sqrt((dA ** 2).mean()).item())})
    finally:
        for wa, orig in patch_info:
            wa.forward = orig

    return rows


def run_stage_d(
    model,
    case: CaseData,
    case_obj: Case,
    device: str,
    qhat_zwd: torch.Tensor,
    out_dir: str,
) -> list[dict[str, Any]]:
    case_id = case_obj.case_id
    specs = _block_list(model)

    print(f"    [{case_id}] Stage D: collecting hidden states (true)…")
    states_true = _collect_hidden_states(model, make_batch(case, device), specs)
    _gpu_sync_and_gc()

    print(f"    [{case_id}] Stage D: collecting hidden states (qhat)…")
    states_qhat = _collect_hidden_states(model, make_batch(case, device, zwd_override=qhat_zwd), specs)
    _gpu_sync_and_gc()

    rows: list[dict[str, Any]] = []
    for _, key, family in specs:
        if key not in states_true or key not in states_qhat:
            continue
        h_t = states_true[key]
        delta = states_qhat[key] - h_t
        base_rms = float(torch.sqrt((h_t ** 2).mean()).item())
        d_rms = float(torch.sqrt((delta ** 2).mean()).item())
        h_f = h_t.reshape(-1).double()
        q_f = states_qhat[key].reshape(-1).double()
        cos_sim = float((torch.dot(h_f, q_f) / (torch.norm(h_f) * torch.norm(q_f) + 1e-30)).item())
        rows.append({
            "case_id": case_id, "target": case_obj.target,
            "init_time": case_obj.init_time.isoformat(), "role": case_obj.role,
            "block_key": key, "family": family,
            "head_idx": None,          # only set for dec_s1_attn rows
            "attn_delta_rms": None,    # only set for dec_s1_attn rows
            "baseline_rms": base_rms, "delta_rms": d_rms,
            "relative_rms": d_rms / base_rms if base_rms > 1e-10 else float("nan"),
            "cosine_sim": cos_sim,
        })

    rows += _dec_s1_attention_contrast(model, case, device, qhat_zwd, case_id, case_obj)

    _append_csv(os.path.join(out_dir, "routing_mechanism.csv"), rows)

    by_fam: dict[str, list[float]] = {}
    for r in rows:
        v = r.get("relative_rms")
        if isinstance(v, float) and not (v != v):  # not NaN
            by_fam.setdefault(r.get("family", "?"), []).append(v)
    for fam, vals in sorted(by_fam.items()):
        print(f"    [{case_id}] Stage D {fam}: max_rel_rms={max(vals):.4f}")

    return rows
