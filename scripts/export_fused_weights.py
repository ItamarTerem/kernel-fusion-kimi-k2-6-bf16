#!/usr/bin/env python3
"""
Export a fused-weight HuggingFace checkpoint for NVFP4 quantization.

This does NOT use patch_kimi_model() or FusedRMSNormLinear CUDA modules.
It writes gamma-absorbed weights into the original nn.Linear tensors and
sets absorbed RMSNorm gammas to 1, then saves a new directory your teammate
(or Model Optimizer / llm-compressor) can PTQ as a normal HF model.

Usage:
    python scripts/export_fused_weights.py \\
        --model-path models/Kimi-K2.6-bf16 \\
        --output-dir models/Kimi-K2.6-bf16-fused-weights

Requires ~2 TB free disk for a full Kimi export (new copy of weights).
Run on a multi-GPU node (device_map=auto); use tmux for long saves.

After export, quantize ONLY the output directory, e.g. Model Optimizer:
    python examples/llm_ptq/hf_ptq.py \\
        --pyt_ckpt_path models/Kimi-K2.6-bf16-fused-weights \\
        --qformat nvfp4_mlp_only \\
        --export_path models/Kimi-K2.6-NVFP4-fused-weights \\
        ...
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.weight_transforms.fuse_model_weights import fuse_model_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fused-weight BF16 checkpoint for NVFP4 PTQ"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Input HF checkpoint (unfused BF16), e.g. models/Kimi-K2.6-bf16",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for fused-weight checkpoint",
    )
    args = parser.parse_args()

    if os.path.abspath(args.model_path) == os.path.abspath(args.output_dir):
        raise SystemExit("Refusing to export in-place: --output-dir must differ from --model-path")

    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    print(f"Fusing weights on {n_layers} decoder layers (in-place on module tensors) ...")
    fuse_model_weights(model, inplace=True)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving fused checkpoint to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")

    print("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")
    print(f"  Fused BF16 weights: {args.output_dir}")
    print("  Next: run NVFP4 PTQ on this directory (not on patch_kimi_model output).")


if __name__ == "__main__":
    main()
