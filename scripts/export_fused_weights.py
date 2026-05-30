#!/usr/bin/env python3
"""
In-place offline weight fusion for Kimi-K2.6 safetensors checkpoint.

Fuses RMSNorm gammas into downstream Linear weights (W_new = W * gamma) and
resets each absorbed norm's gamma to 1, directly on the safetensors shards.

No second copy of the checkpoint is created — the model directory is modified
in-place.  Peak memory is ~one shard (~5 GB), well within the 128 GB unified
memory of a DGX Spark.

Usage:
    python scripts/export_fused_weights.py --model-path /home/nvidia/Kimi-K2.6-bf16

After fusion, quantize the same directory with llm-compressor:
    python nvfp4_quant_kimi.py   # update model_stub to point at model-path
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.weight_transforms.fuse_model_weights import fuse_safetensors_inplace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-place offline RMSNorm→Linear weight fusion for Kimi-K2.6"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF checkpoint directory to fuse in-place (e.g. /home/nvidia/Kimi-K2.6-bf16)",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        raise SystemExit(f"--model-path does not exist or is not a directory: {model_path}")

    print(f"Fusing weights in-place at: {model_path}")
    print("WARNING: this modifies the checkpoint directory directly.")
    print("         Make sure you have a backup if the original is needed.")
    print()

    fuse_safetensors_inplace(model_path)

    print()
    print("Done.")
    print(f"  Fused checkpoint: {model_path}")
    print("  Next: run NVFP4 PTQ on this directory with nvfp4_quant_kimi.py")


if __name__ == "__main__":
    main()
