"""
comparison_models.py — Model loading and inference for 07_zwd_precipitation_model_comparison.

Provides:
  - load_model(model_spec, device): loads Aurora checkpoint with grad checkpointing
  - rollout_independent(model, batch, n_steps): autoregressive rollout
  - extract_scalar(pred, target_spec, lat_vals, lon_vals, target_region): box mean
  - extract_q850_box / extract_precip_box: low-level box mean helpers
  - audit_model_spec(checkpoint_path): inspects state dict for surf_vars, architecture
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sys
import types
from typing import Optional

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class _Stub:
    """Flexible stub for unknown classes in pickled checkpoints.

    Supports being called, iterated, used as a context manager, and
    attribute access — all silently returning empty/None results.
    This lets torch.load reconstruct checkpoints whose training codebase
    is not present, without disrupting the state_dict tensors we need.
    """
    def __init__(self, *a, **kw):
        pass
    def __call__(self, *a, **kw):
        return _Stub()
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0
    def __getattr__(self, name):
        return _Stub()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __repr__(self):
        return "_Stub()"


import pickle as _pickle


class _LenientUnpickler(_pickle.Unpickler):
    """Unpickler that returns _Stub for any class that can't be imported.

    The precip_small checkpoints reference utils.losses and other training
    codebase modules.  By overriding find_class we let torch.load succeed
    even when those modules are absent; the state_dict tensors we actually
    need are plain tensors and load correctly regardless.
    """
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, ModuleNotFoundError, AttributeError):
            return _Stub


class _LenientPickleModule:
    """Minimal pickle-module interface accepted by torch.load(pickle_module=...)."""
    Unpickler = _LenientUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _LenientUnpickler(file).load()

    @staticmethod
    def loads(data, **kwargs):
        import io as _io
        return _LenientUnpickler(_io.BytesIO(data)).load()



# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

for _p in (_SEARCHLIGHT_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparison_config import ModelSpec, TargetSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Gradient checkpointing helpers (reused from searchlight_benchmark.py)
# ---------------------------------------------------------------------------

def _enable_swin_grad_checkpointing(model) -> None:
    """Wrap Swin3DTransformerBlock.forward with activation checkpointing."""
    for layer in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        for blk in layer.blocks:
            orig = blk.forward

            def _make_cp(fn):
                def _cp(*a, **kw):
                    if kw:
                        return torch_checkpoint(
                            lambda *args: fn(*args, **kw),
                            *a,
                            use_reentrant=True,
                        )
                    return torch_checkpoint(fn, *a, use_reentrant=True)
                return _cp

            blk.forward = _make_cp(orig)


def _enable_level_agg_checkpointing(model) -> None:
    """Wrap encoder.level_agg with activation checkpointing."""
    for enc in [model.encoder] + list(getattr(model, "encoders", [])):
        if hasattr(enc, "level_agg"):
            orig_la = enc.level_agg.forward

            def _make_la_cp(fn):
                def _cp(*a, **kw):
                    if kw:
                        return torch_checkpoint(
                            lambda *args: fn(*args, **kw),
                            *a,
                            use_reentrant=True,
                        )
                    return torch_checkpoint(fn, *a, use_reentrant=True)
                return _cp

            enc.level_agg.forward = _make_la_cp(orig_la)


def _enable_decoder_rollout_checkpointing(model) -> None:
    """Wrap decoder.level_decoder.forward with conditional checkpointing.

    Only activates when grad is enabled and an input requires grad.
    Required for multi-step rollout backward passes.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return
    level_decoder = getattr(decoder, "level_decoder", None)
    if level_decoder is None:
        return
    if getattr(level_decoder, "_rollout_checkpoint_wrapped", False):
        return

    orig_forward = level_decoder.forward

    def _checkpointed_forward(latents, x):
        if torch.is_grad_enabled() and (
            getattr(latents, "requires_grad", False)
            or getattr(x, "requires_grad", False)
        ):
            return torch_checkpoint(orig_forward, latents, x, use_reentrant=True)
        return orig_forward(latents, x)

    level_decoder.forward = _checkpointed_forward
    level_decoder._rollout_checkpoint_wrapped = True


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_spec: ModelSpec, device):
    """Load an Aurora checkpoint specified by model_spec.

    Applies gradient checkpointing on all Swin3D blocks, level_agg, and
    the decoder level_decoder — following the same pattern as
    searchlight_benchmark.setup_model().

    Returns the model frozen, grad-checkpointed, in eval mode on device.
    """
    from aurora import Aurora

    print(f"Loading model '{model_spec.name}' from {model_spec.checkpoint_path}")

    model = Aurora(
        surf_vars=model_spec.surf_vars,
        static_vars=model_spec.static_vars,
        atmos_vars=model_spec.atmos_vars,
        encoder_depths=model_spec.encoder_depths,
        encoder_num_heads=model_spec.encoder_num_heads,
        decoder_depths=model_spec.decoder_depths,
        decoder_num_heads=model_spec.decoder_num_heads,
        embed_dim=model_spec.embed_dim,
        num_heads=model_spec.num_heads,
        autocast=True,
        use_lora=False,
        num_ensemble=1,
        encoder_activation_checkpointing=False,
    )

    ckpt = torch.load(
        model_spec.checkpoint_path, map_location="cpu",
        weights_only=False, pickle_module=_LenientPickleModule,
    )
    # Support both raw state_dict and Lightning-style {state_dict: {...}}
    if "state_dict" in ckpt:
        raw_state = ckpt["state_dict"]
        # Lightning prepends "net." or "model." — strip first 4 chars if needed
        first_key = next(iter(raw_state.keys()), "")
        if first_key.startswith("net."):
            state = {k[4:]: v for k, v in raw_state.items()}
        elif first_key.startswith("model."):
            state = {k[6:]: v for k, v in raw_state.items()}
        else:
            # Try the 4-char strip as in the original searchlight code
            state = {k[4:]: v for k, v in raw_state.items()}
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    _enable_swin_grad_checkpointing(model)
    _enable_level_agg_checkpointing(model)
    _enable_decoder_rollout_checkpointing(model)

    print(f"  Model '{model_spec.name}' loaded, frozen, grad-checkpointed.")
    return model


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

def _forward(model, batch):
    """Call model.forward and unwrap (pred, std, preds) tuple if needed."""
    out = model.forward(batch)
    if isinstance(out, tuple):
        out = out[0]
    return out


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

def rollout_independent(model, batch, n_steps: int) -> list:
    """Run n_steps autoregressive forward passes, returning list of predictions.

    Each step produces one prediction; the prediction is fed back as input
    for the next step. The input batch is not modified.

    Returns:
        List of n_steps Aurora Batch (prediction) objects.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    p = next(model.parameters())
    current = batch.type(p.dtype)

    if model.use_resolution_specific_patch_tokenizers:
        patch_size = model.patch_tokenizer_identifier.get_patch_size(
            current.metadata.grid_resolution
        )
    else:
        patch_size = model.patch_size
    current = current.crop(patch_size=patch_size)
    current = current.to(p.device)

    preds = []
    for _ in range(n_steps):
        pred = _forward(model, current)
        preds.append(pred)
        current = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([current.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([current.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

    return preds


# ---------------------------------------------------------------------------
# Scalar extractors
# ---------------------------------------------------------------------------

def extract_q850_box(
    pred,
    level_idx: int,
    lat_imin: int,
    lat_imax: int,
    lon_imin: int,
    lon_imax: int,
) -> float:
    """Extract cosine-lat-weighted box mean of q at 850 hPa from a prediction.

    Returns value in g/kg (multiplied by 1e3 to match searchlight convention).
    """
    q = pred.atmos_vars["q"].float()
    return float(q[0, 0, level_idx, lat_imin:lat_imax + 1, lon_imin:lon_imax + 1].mean().item()) * 1e3


def extract_precip_box(
    pred,
    lat_imin: int,
    lat_imax: int,
    lon_imin: int,
    lon_imax: int,
    precip_var: str = "total_precipitation_MSWEP",
) -> float:
    """Extract box mean precipitation from a prediction (mm or model units)."""
    if precip_var not in pred.surf_vars:
        return float("nan")
    p = pred.surf_vars[precip_var].float()
    return float(p[0, 0, lat_imin:lat_imax + 1, lon_imin:lon_imax + 1].mean().item())


def extract_scalar(
    pred,
    target_spec: TargetSpec,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    target_region,
) -> float:
    """Extract a box-mean scalar from a prediction for the given target_spec.

    target_region must be a TargetRegion with box_lat / box_lon attributes
    (as defined in searchlight_tasks.py).

    Returns the box-mean scalar value.
    """
    from searchlight_tasks import box_indices

    lat_imin, lat_imax, lon_imin, lon_imax = box_indices(
        target_region, lat_vals, lon_vals
    )

    if target_spec.level_hpa is not None:
        # Atmospheric variable
        levels = pred.metadata.atmos_levels if hasattr(pred, "metadata") else None
        if levels is None:
            raise ValueError("Cannot determine pressure levels from prediction metadata")
        levels_arr = np.asarray(levels)
        matches = np.where(levels_arr == target_spec.level_hpa)[0]
        if matches.size != 1:
            raise ValueError(
                f"Expected 1 level at {target_spec.level_hpa} hPa, "
                f"found {matches.tolist()}"
            )
        level_idx = int(matches[0])

        if target_spec.output_var == "q":
            return extract_q850_box(pred, level_idx, lat_imin, lat_imax, lon_imin, lon_imax)
        # Generic atmos variable
        arr = pred.atmos_vars[target_spec.output_var].float()
        return float(arr[0, 0, level_idx, lat_imin:lat_imax + 1, lon_imin:lon_imax + 1].mean().item())
    else:
        # Surface variable
        return extract_precip_box(pred, lat_imin, lat_imax, lon_imin, lon_imax, target_spec.output_var)


# ---------------------------------------------------------------------------
# Checkpoint auditor
# ---------------------------------------------------------------------------

def audit_model_spec(checkpoint_path: str) -> dict:
    """Inspect a checkpoint state dict to determine model architecture.

    Loads the checkpoint (CPU only) and infers:
      - surf_vars: from encoder.surf_token_embeds or similar keys
      - atmos_vars: from encoder.atmos_token_embeds keys
      - embed_dim: from a linear weight shape
      - encoder_depths: number of blocks per stage (count unique block indices)
      - whether the output includes precip and ZWD

    Returns a dict with the inferred fields. Call this in audit_checkpoints.py.
    """
    print(f"Auditing: {checkpoint_path}")
    ckpt = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=False, pickle_module=_LenientPickleModule,
    )

    if "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        first_key = next(iter(raw.keys()), "")
        if first_key.startswith("net.") or first_key.startswith("model."):
            prefix_len = 4 if first_key.startswith("net.") else 6
            state = {k[prefix_len:]: v for k, v in raw.items()}
        else:
            state = {k[4:]: v for k, v in raw.items()}
    else:
        state = ckpt

    result = {
        "checkpoint_path": checkpoint_path,
        "surf_vars": [],
        "atmos_vars": [],
        "embed_dim": None,
        "encoder_depths": [],
        "decoder_depths": [],
        "has_precip": False,
        "has_zwd": False,
        "has_q": False,
        "inferred_notes": [],
    }

    # Infer surf_vars / atmos_vars from decoder head keys.
    # Keys in `state` have the "net." prefix already stripped, so patterns are:
    #   decoder.surf_heads.{var}.{layer}.{weight|bias}
    #   decoder.atmos_heads.{var}.{layer}.{weight|bias}
    import re as _re2
    surf_var_keys: set[str] = set()
    atmos_var_keys: set[str] = set()

    for key in state.keys():
        m = _re2.match(r"decoder\.surf_heads\.([^.]+)\.", key)
        if m:
            surf_var_keys.add(m.group(1))
        m = _re2.match(r"decoder\.atmos_heads\.([^.]+)\.", key)
        if m:
            atmos_var_keys.add(m.group(1))

    if surf_var_keys:
        result["surf_vars"] = sorted(surf_var_keys)
        result["has_zwd"] = "zwd" in surf_var_keys
        result["has_precip"] = "tp_mswep" in surf_var_keys or "precip" in " ".join(surf_var_keys)
    if atmos_var_keys:
        result["atmos_vars"] = sorted(atmos_var_keys)
        result["has_q"] = "q" in atmos_var_keys

    # Infer embed_dim from backbone.time_mlp.0.weight (shape [D, D], prefix stripped)
    time_mlp_key = "backbone.time_mlp.0.weight"
    if time_mlp_key in state:
        v = state[time_mlp_key]
        if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == v.shape[1]:
            result["embed_dim"] = int(v.shape[0])
            result["inferred_notes"].append(
                f"embed_dim from {time_mlp_key}: shape {list(v.shape)}"
            )

    # Infer encoder/decoder depths by counting unique block indices per stage
    import re as _re
    enc_blocks: dict[int, set[int]] = {}
    dec_blocks: dict[int, set[int]] = {}

    for key in state.keys():
        # encoder_layers.{stage}.blocks.{block_idx}
        m = _re.search(r"backbone\.encoder_layers\.(\d+)\.blocks\.(\d+)", key)
        if m:
            stage, blk = int(m.group(1)), int(m.group(2))
            enc_blocks.setdefault(stage, set()).add(blk)
        m = _re.search(r"backbone\.decoder_layers\.(\d+)\.blocks\.(\d+)", key)
        if m:
            stage, blk = int(m.group(1)), int(m.group(2))
            dec_blocks.setdefault(stage, set()).add(blk)

    if enc_blocks:
        result["encoder_depths"] = [
            len(enc_blocks[s]) for s in sorted(enc_blocks.keys())
        ]
    if dec_blocks:
        result["decoder_depths"] = [
            len(dec_blocks[s]) for s in sorted(dec_blocks.keys())
        ]

    print(f"  surf_vars: {result['surf_vars']}")
    print(f"  atmos_vars: {result['atmos_vars']}")
    print(f"  embed_dim: {result['embed_dim']}")
    print(f"  encoder_depths: {result['encoder_depths']}")
    print(f"  decoder_depths: {result['decoder_depths']}")
    print(f"  has_precip={result['has_precip']}  has_zwd={result['has_zwd']}  has_q={result['has_q']}")
    if result["inferred_notes"]:
        for note in result["inferred_notes"]:
            print(f"  note: {note}")

    return result


# ---------------------------------------------------------------------------
# GPU sync helper
# ---------------------------------------------------------------------------

def _gpu_sync_and_gc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
