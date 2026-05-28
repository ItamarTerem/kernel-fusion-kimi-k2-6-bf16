"""
verify_fusion.py — Sanity-check the RMSNorm+Linear fusion on Kimi-K2.6.

Runs five tests in order:
  1. CUDA extension  — denominator_cuda loads and kernels are callable
  2. Layer coverage  — all decoder layers have the three fused modules
  3. Norm disabled   — absorbed gamma vectors are all ones
  4. Numerical diff  — logit max-abs-diff fused vs unfused < threshold
  5. Throughput      — tokens/sec with and without fusion (quick 50-token run)

Usage:
    python verify_fusion.py --model-path /path/to/Kimi-K2.6-bf16 [--variant V2]

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import argparse
import sys
import time
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
INFO = "\033[94m INFO\033[0m"

results: list[tuple[str, bool, str]] = []


def log(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    print(f"[{tag}] {name}" + (f"  —  {detail}" if detail else ""))
    results.append((name, ok, detail))


# ── Test 1: CUDA extension ────────────────────────────────────────────────────

def test_cuda_extension() -> bool:
    print("\n── Test 1: CUDA extension ──────────────────────────────────────")
    try:
        from src.load_cuda import denominator_cuda
        has_v1 = hasattr(denominator_cuda, "rmsnorm_normalize")
        has_v3 = hasattr(denominator_cuda, "rmsnorm_normalize_512")
        log("extension loads", True)
        log("rmsnorm_normalize (V1) present", has_v1)
        log("rmsnorm_normalize_512 (V3) present", has_v3)

        # Quick kernel smoke-test with a tiny tensor
        if has_v1:
            T, h, d = 2, 64, 128
            x   = torch.randn(T, h, device="cuda", dtype=torch.bfloat16)
            out = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
            b   = torch.zeros(d, device="cuda", dtype=torch.bfloat16)
            denominator_cuda.rmsnorm_normalize(x, out, b, h, 1e-6)
            log("V1 kernel runs on BF16", True)
        if has_v3:
            denominator_cuda.rmsnorm_normalize_512(x, out, b, h, 1e-6)
            log("V3 kernel runs on BF16", True)

        return has_v1 and has_v3
    except Exception as e:
        log("CUDA extension", False, str(e))
        return False


# ── Test 2: Layer coverage ────────────────────────────────────────────────────

def test_layer_coverage(model: nn.Module) -> bool:
    print("\n── Test 2: Layer coverage ──────────────────────────────────────")
    layers = model.model.layers
    n_layers = len(layers)
    print(f"  Decoder layers: {n_layers}")

    missing = []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        for attr in ("fused_input_ln", "fused_q_b", "fused_kv_b"):
            if not hasattr(attn, attr):
                missing.append(f"layer {i}: missing {attr}")

    ok = len(missing) == 0
    log(f"all {n_layers} layers have fused modules", ok,
        f"{len(missing)} missing" if not ok else "")
    if missing:
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
    return ok


# ── Test 3: Norm disabled (gamma = ones) ─────────────────────────────────────

def test_norm_disabled(model: nn.Module, unfused_model: nn.Module) -> bool:
    print("\n── Test 3: Norm weights disabled ───────────────────────────────")
    bad = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for norm_name in ("q_a_layernorm", "kv_a_layernorm"):
            if hasattr(attn, norm_name):
                g = getattr(attn, norm_name).weight.data
                if not torch.allclose(g, torch.ones_like(g), atol=1e-4):
                    bad.append(f"layer {i} {norm_name}: gamma not ones "
                                f"(max_dev={( g - 1).abs().max().item():.2e})")

        # input_layernorm lives on the layer itself
        if hasattr(layer, "input_layernorm"):
            g = layer.input_layernorm.weight.data
            if not torch.allclose(g, torch.ones_like(g), atol=1e-4):
                bad.append(f"layer {i} input_layernorm: gamma not ones")

    ok = len(bad) == 0
    log("absorbed norms have gamma = 1", ok,
        f"{len(bad)} deviations" if not ok else "")
    if bad:
        for b in bad[:3]:
            print(f"    {b}")
    return ok


# ── Test 4: Numerical equivalence ────────────────────────────────────────────

def test_numerical(
    fused_model: nn.Module,
    unfused_model: nn.Module,
    tokenizer,
    prompt: str = "Explain the concept of kernel fusion in one paragraph.",
    max_abs_tol: float = 5e-2,   # BF16 has ~0.01 precision; 5e-2 gives headroom
) -> bool:
    print("\n── Test 4: Numerical equivalence ───────────────────────────────")
    inputs = tokenizer(prompt, return_tensors="pt").to(fused_model.device)

    with torch.no_grad():
        fused_logits   = fused_model(**inputs).logits[:, -1, :]
        unfused_logits = unfused_model(**inputs).logits[:, -1, :]

    diff = (fused_logits - unfused_logits).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    # Top-1 token agreement (most practical signal)
    fused_top1   = fused_logits.argmax(dim=-1)
    unfused_top1 = unfused_logits.argmax(dim=-1)
    top1_match = (fused_top1 == unfused_top1).all().item()

    print(f"  Max  |Δlogit| : {max_diff:.4f}  (threshold {max_abs_tol})")
    print(f"  Mean |Δlogit| : {mean_diff:.6f}")
    print(f"  Top-1 token   : {'match' if top1_match else 'MISMATCH'}")

    ok = max_diff < max_abs_tol
    log("logit max-abs-diff < threshold", ok, f"max={max_diff:.4f}")
    log("top-1 token matches", top1_match)
    return ok and top1_match


# ── Test 5: Throughput ────────────────────────────────────────────────────────

def test_throughput(
    fused_model: nn.Module,
    unfused_model: nn.Module,
    tokenizer,
    n_tokens: int = 50,
    prompt: str = "Hello, world!",
) -> bool:
    print("\n── Test 5: Throughput ──────────────────────────────────────────")
    inputs = tokenizer(prompt, return_tensors="pt").to(fused_model.device)

    def measure(model, label):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = n_tokens / elapsed
        print(f"  {label:<12}: {tps:.1f} tok/s  ({elapsed:.2f}s for {n_tokens} tokens)")
        return tps

    tps_unfused = measure(unfused_model, "unfused")
    tps_fused   = measure(fused_model,   "fused")

    speedup = tps_fused / tps_unfused
    print(f"  Speedup       : {speedup:.3f}×")

    # We expect at least no regression (>= 0.98×); any speedup is a win
    ok = speedup >= 0.98
    log("throughput >= 0.98× unfused", ok, f"speedup={speedup:.3f}×")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify Kimi-K2.6 fusion")
    parser.add_argument("--model-path", required=True,
                        help="Path to bullerwins/Kimi-K2.6-bf16 weights")
    parser.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"],
                        help="Fusion variant to test (default: V2)")
    parser.add_argument("--skip-numerical", action="store_true",
                        help="Skip numerical equivalence test (saves memory; "
                             "requires loading model twice otherwise)")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print(f" Kimi-K2.6 fusion verification  (variant={args.variant})")
    print(f" Model path: {args.model_path}")
    print("=" * 60)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\n{INFO} Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # ── Test 1 (no model needed) ──────────────────────────────────────────────
    ext_ok = test_cuda_extension()

    # ── Load unfused model ────────────────────────────────────────────────────
    print(f"\n{INFO} Loading unfused model...")
    unfused_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    unfused_model.eval()

    # ── Load fused model ──────────────────────────────────────────────────────
    print(f"\n{INFO} Loading fused model (variant={args.variant})...")
    from kimi_patch import patch_kimi_model

    fused_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    fused_model = patch_kimi_model(fused_model, variant=args.variant)
    fused_model.eval()

    # ── Tests 2 & 3 ───────────────────────────────────────────────────────────
    cov_ok  = test_layer_coverage(fused_model)
    norm_ok = test_norm_disabled(fused_model, unfused_model)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    if args.skip_numerical:
        print(f"\n{INFO} Skipping numerical equivalence test (--skip-numerical)")
        num_ok = True
    else:
        num_ok = test_numerical(fused_model, unfused_model, tokenizer)

    # ── Test 5 ────────────────────────────────────────────────────────────────
    if args.skip_throughput:
        print(f"\n{INFO} Skipping throughput test (--skip-throughput)")
        thr_ok = True
    else:
        thr_ok = test_throughput(fused_model, unfused_model, tokenizer)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Results")
    print("=" * 60)
    all_passed = True
    for name, ok, detail in results:
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name}" + (f"  —  {detail}" if detail else ""))
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("\033[92m All tests passed. Fusion is correct.\033[0m")
        sys.exit(0)
    else:
        print("\033[91m One or more tests FAILED. Check output above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()