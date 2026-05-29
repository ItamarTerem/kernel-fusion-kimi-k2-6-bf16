"""
Offline weight fusion for Kimi-K2.6 / DeepSeek-V2-Lite MLA models.

Bakes RMSNorm gamma into downstream Linear weights (W_new = W * gamma) and sets
each absorbed norm's gamma to 1.  The result is a standard HuggingFace
checkpoint (same module names, no FusedRMSNorm* ops) that is mathematically
equivalent to runtime fusion when the original forward runs norm → linear.

Used by scripts/export_fused_weights.py before NVFP4 PTQ (Model Optimizer /
llm-compressor).  Do NOT run PTQ on patch_kimi_model() — quant tools expect
nn.Linear checkpoints, not custom CUDA fused modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.weight_transforms.weight_transform import compute_fused_weights


def _disable_fused_norm(rms_norm: nn.Module) -> None:
    with torch.no_grad():
        rms_norm.weight.fill_(1.0)


def _kv_a_linear(attn: nn.Module) -> nn.Linear:
    if hasattr(attn, "kv_a_proj_with_mqa"):
        return attn.kv_a_proj_with_mqa
    if hasattr(attn, "kv_a_proj"):
        return attn.kv_a_proj
    raise AttributeError(
        f"{type(attn).__name__} has neither kv_a_proj_with_mqa nor kv_a_proj"
    )


def fuse_decoder_layer_weights(layer: nn.Module, *, inplace: bool = True) -> None:
    """
    Fuse all four MLA norm→linear sites on one decoder layer (in-place on weights).

    Fusion sites (same as kimi_patch.py):
      input_layernorm  → q_a_proj, kv_a_proj(_with_mqa)
      q_a_layernorm    → q_b_proj
      kv_a_layernorm   → kv_b_proj
    """
    attn = layer.self_attn
    kv_a = _kv_a_linear(attn)

    compute_fused_weights(layer.input_layernorm, attn.q_a_proj, inplace=inplace)
    compute_fused_weights(layer.input_layernorm, kv_a, inplace=inplace)
    _disable_fused_norm(layer.input_layernorm)

    compute_fused_weights(attn.q_a_layernorm, attn.q_b_proj, inplace=inplace)
    _disable_fused_norm(attn.q_a_layernorm)

    compute_fused_weights(attn.kv_a_layernorm, attn.kv_b_proj, inplace=inplace)
    _disable_fused_norm(attn.kv_a_layernorm)


def fuse_model_weights(model: nn.Module, *, inplace: bool = True) -> nn.Module:
    """Apply fuse_decoder_layer_weights to every decoder layer. Returns model."""
    for layer in model.model.layers:
        fuse_decoder_layer_weights(layer, inplace=inplace)
    return model
