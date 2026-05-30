"""
Offline weight fusion for Kimi-K2.6 / DeepSeek-V2-Lite MLA models.

Bakes RMSNorm gamma into downstream Linear weights (W_new = W * gamma) and sets
each absorbed norm's gamma to 1.  The result is a standard HuggingFace
checkpoint (same module names, no FusedRMSNorm* ops) that is mathematically
equivalent to runtime fusion when the original forward runs norm → linear.

Two modes:
  fuse_model_weights()        — in-memory fusion on a loaded nn.Module (original).
  fuse_safetensors_inplace()  — shard-by-shard fusion directly on safetensors files.
                                Peak memory ~one shard (~5 GB). No second copy of
                                the checkpoint needed. Designed for DGX Spark
                                (128 GB unified memory, 2 TB model).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from src.weight_transforms.weight_transform import compute_fused_weights


# ---------------------------------------------------------------------------
# In-memory fusion (original, used when model is already loaded)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shard-by-shard in-place fusion (DGX Spark / memory-constrained systems)
# ---------------------------------------------------------------------------

def _build_fusion_map(weight_map: dict[str, str], n_layers: int) -> tuple[dict, dict]:
    """
    Returns:
        linear_to_norm: linear_weight_key -> norm_weight_key
        norm_keys:      set of all norm weight keys that should be reset to 1
    """
    linear_to_norm: dict[str, str] = {}
    norm_keys: set[str] = set()

    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        attn = f"{prefix}.self_attn"

        input_norm = f"{prefix}.input_layernorm.weight"
        q_a_norm   = f"{attn}.q_a_layernorm.weight"
        kv_a_norm  = f"{attn}.kv_a_layernorm.weight"

        # kv_a proj name varies by model variant
        kv_a_proj = f"{attn}.kv_a_proj_with_mqa.weight"
        if kv_a_proj not in weight_map:
            kv_a_proj = f"{attn}.kv_a_proj.weight"

        linear_to_norm[f"{attn}.q_a_proj.weight"] = input_norm
        linear_to_norm[kv_a_proj]                 = input_norm
        linear_to_norm[f"{attn}.q_b_proj.weight"] = q_a_norm
        linear_to_norm[f"{attn}.kv_b_proj.weight"] = kv_a_norm

        norm_keys.update([input_norm, q_a_norm, kv_a_norm])

    return linear_to_norm, norm_keys


def fuse_safetensors_inplace(model_path: str) -> None:
    """
    Fuse RMSNorm gammas into Linear weights directly on the safetensors shards.

    Modifies the checkpoint directory in-place — no second copy of the model is
    written.  Peak memory is ~one shard at a time (~5 GB for 5 GB shards).

    Algorithm:
      1. Load all norm gamma vectors into memory (few MB total — they are tiny).
      2. For each shard that contains linear weights needing fusion:
           load shard → multiply W by gamma → write shard back (temp+rename).
      3. For each shard that contains norm weights:
           load shard → set gamma to 1 → write shard back (temp+rename).
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    model_dir = Path(model_path)
    index_path = model_dir / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No model.safetensors.index.json found in {model_path}. "
            "Single-shard models are not supported by this path."
        )

    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]  # tensor_name -> shard_filename

    # Detect number of layers
    layer_indices = set()
    for name in weight_map:
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            try:
                layer_indices.add(int(parts[2]))
            except ValueError:
                pass
    n_layers = max(layer_indices) + 1
    print(f"Detected {n_layers} decoder layers.")

    linear_to_norm, norm_keys = _build_fusion_map(weight_map, n_layers)

    # ---- Step 1: load all norm gammas (tiny vectors, fits easily in RAM) ----
    print("Loading norm gamma vectors ...")
    gammas: dict[str, torch.Tensor] = {}
    norm_shards: set[str] = set(weight_map[k] for k in norm_keys if k in weight_map)
    for shard_name in sorted(norm_shards):
        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in norm_keys:
                    gammas[name] = f.get_tensor(name).clone()
    print(f"  Loaded {len(gammas)} norm gamma tensors.")

    # ---- Step 2: fuse linear weights shard by shard ----
    linear_shards: dict[str, list[str]] = defaultdict(list)
    for linear_key in linear_to_norm:
        if linear_key in weight_map:
            linear_shards[weight_map[linear_key]].append(linear_key)

    print(f"Fusing linear weights across {len(linear_shards)} shards ...")
    for shard_idx, (shard_name, linear_keys) in enumerate(sorted(linear_shards.items()), 1):
        shard_path = model_dir / shard_name
        print(f"  [{shard_idx}/{len(linear_shards)}] {shard_name} ({len(linear_keys)} linears) ...", end=" ", flush=True)

        # Load full shard
        all_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                all_tensors[name] = f.get_tensor(name)

        # Apply fusions
        for linear_key in linear_keys:
            norm_key = linear_to_norm[linear_key]
            if norm_key not in gammas:
                print(f"\n  WARNING: gamma not found for {norm_key}, skipping {linear_key}")
                continue
            gamma = gammas[norm_key].to(all_tensors[linear_key].dtype)
            all_tensors[linear_key].mul_(gamma)

        # Write back atomically
        tmp_path = shard_path.with_suffix(".safetensors.tmp")
        save_file(all_tensors, str(tmp_path))
        tmp_path.rename(shard_path)
        print("done")

    # ---- Step 3: reset norm gammas to 1 shard by shard ----
    print(f"Resetting norm gammas to 1 across {len(norm_shards)} shards ...")
    for shard_idx, shard_name in enumerate(sorted(norm_shards), 1):
        shard_path = model_dir / shard_name
        print(f"  [{shard_idx}/{len(norm_shards)}] {shard_name} ...", end=" ", flush=True)

        all_tensors = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in norm_keys:
                    t = torch.ones_like(t)
                all_tensors[name] = t

        tmp_path = shard_path.with_suffix(".safetensors.tmp")
        save_file(all_tensors, str(tmp_path))
        tmp_path.rename(shard_path)
        print("done")

    print("All shards fused and written back.")
