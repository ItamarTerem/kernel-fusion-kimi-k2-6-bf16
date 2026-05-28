"""
verify_fusion.py — Sanity-check the RMSNorm+Linear fusion on Kimi-K2.6.

Strategy: load the model ONCE, measure the unfused baseline, then patch
in-place and compare.  Peak GPU memory = one model (~2 TB BF16).

Pipeline
────────
  Phase A — pre-patch (unfused model)
    1. CUDA extension  — denominator_cuda loads; V1 + V3 kernels callable
    2. Reference logits — forward pass saved to CPU before any weights change
    3. Unfused throughput — tokens/sec baseline (optional)

  Phase B — patch in-place
    patch_kimi_model(model, variant=...)

  Phase C — post-patch (fused model)
    4. Layer coverage  — all 61 layers have fused_input_ln / fused_q_b / fused_kv_b
    5. Norm disabled   — absorbed gamma vectors are all ones
    6. Numerical diff  — fused logits vs saved reference logits < threshold
    7. Fused throughput — tokens/sec after fusion; speedup vs baseline reported

Usage:
    python tests/verify_fusion.py --model-path /path/to/Kimi-K2.6-bf16 [--variant V2]

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"

results: list[tuple[str, bool, str]] = []


def log(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f"  —  {detail}" if detail else ""))
    results.append((name, ok, detail))
    return ok


# ── Phase A tests ─────────────────────────────────────────────────────────────

def test_cuda_extension() -> bool:
    print("\n── Test 1: CUDA extension ──────────────────────────────────────")
    try:
        from src.load_cuda import denominator_cuda

        has_v1 = hasattr(denominator_cuda, "rmsnorm_normalize")
        has_v3 = hasattr(denominator_cuda, "rmsnorm_normalize_512")
        log("extension loads without error", True)
        log("rmsnorm_normalize (V1) present", has_v1)
        log("rmsnorm_normalize_512 (V3) present", has_v3)

        # Smoke-test: call each kernel on a real BF16 tensor
        T, h, d = 2, 64, 128
        x   = torch.randn(T, h, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
        b   = torch.zeros(d,  device="cuda", dtype=torch.bfloat16)

        if has_v1:
            denominator_cuda.rmsnorm_normalize(x, out, b, h, 1e-6)
            log("V1 kernel executes on BF16 tensor", True)
        if has_v3:
            denominator_cuda.rmsnorm_normalize_512(x, out, b, h, 1e-6)
            log("V3 kernel executes on BF16 tensor", True)

        return has_v1 and has_v3

    except Exception as e:
        log("CUDA extension", False, str(e))
        return False


def save_reference_logits(
    model: nn.Module,
    tokenizer,
    prompt: str,
) -> torch.Tensor:
    """
    Run one forward pass on the unfused model and return the last-token
    logits on CPU.  Called before patch_kimi_model() modifies any weights.
    """
    print(f"\n── Saving reference logits (unfused) ───────────────────────────")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].cpu()
    print(f"  Logits shape : {tuple(logits.shape)}")
    print(f"  Top-1 token  : {logits.argmax(dim=-1).item()}")
    return logits


def measure_throughput(model: nn.Module, tokenizer, prompt: str, n_tokens: int, label: str) -> float:
    """Generate n_tokens and return tokens/sec."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = n_tokens / elapsed
    print(f"  {label:<16}: {tps:.1f} tok/s  ({elapsed:.2f}s for {n_tokens} tokens)")
    return tps


# ── Phase C tests ─────────────────────────────────────────────────────────────

def test_layer_coverage(model: nn.Module) -> bool:
    print("\n── Test 2: Layer coverage ──────────────────────────────────────")
    layers = model.model.layers
    n = len(layers)
    print(f"  Decoder layers: {n}")

    missing = []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        for attr in ("fused_input_ln", "fused_q_b", "fused_kv_b"):
            if not hasattr(attn, attr):
                missing.append(f"layer {i}: missing self_attn.{attr}")

    ok = len(missing) == 0
    log(f"all {n} layers have fused_input_ln + fused_q_b + fused_kv_b", ok,
        f"{len(missing)} missing" if not ok else "")
    for m in missing[:5]:
        print(f"    {m}")
    if len(missing) > 5:
        print(f"    ... and {len(missing) - 5} more")
    return ok


def test_norm_disabled(model: nn.Module) -> bool:
    print("\n── Test 3: Absorbed norms are disabled (gamma = 1) ─────────────")
    bad = []

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # Inner norms (q_a, kv_a)
        for norm_name in ("q_a_layernorm", "kv_a_layernorm"):
            if hasattr(attn, norm_name):
                g = getattr(attn, norm_name).weight.data
                dev = (g - 1.0).abs().max().item()
                if dev > 1e-4:
                    bad.append(f"layer {i} self_attn.{norm_name}: max_dev={dev:.2e}")
        # input_layernorm lives on the layer
        if hasattr(layer, "input_layernorm"):
            g = layer.input_layernorm.weight.data
            dev = (g - 1.0).abs().max().item()
            if dev > 1e-4:
                bad.append(f"layer {i} input_layernorm: max_dev={dev:.2e}")

    ok = len(bad) == 0
    log("all absorbed gamma vectors equal 1", ok,
        f"{len(bad)} deviations" if not ok else "")
    for b in bad[:3]:
        print(f"    {b}")
    return ok


def test_numerical(
    model: nn.Module,
    tokenizer,
    reference_logits: torch.Tensor,
    prompt: str,
    max_abs_tol: float = 5e-2,
) -> bool:
    print("\n── Test 4: Numerical equivalence ───────────────────────────────")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        fused_logits = model(**inputs).logits[:, -1, :].cpu()

    diff     = (fused_logits - reference_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ref_top1   = reference_logits.argmax(dim=-1).item()
    fused_top1 = fused_logits.argmax(dim=-1).item()
    top1_match = ref_top1 == fused_top1

    print(f"  Max  |Δlogit| : {max_diff:.4f}  (threshold {max_abs_tol})")
    print(f"  Mean |Δlogit| : {mean_diff:.6f}")
    print(f"  Top-1 token   : ref={ref_top1}  fused={fused_top1}  "
          f"{'✔ match' if top1_match else '✘ MISMATCH'}")

    ok = max_diff < max_abs_tol
    log(f"max |Δlogit| < {max_abs_tol}", ok, f"max={max_diff:.4f}")
    log("top-1 token unchanged", top1_match)
    return ok and top1_match


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify Kimi-K2.6 RMSNorm fusion")
    parser.add_argument("--model-path", required=True,
                        help="Path to bullerwins/Kimi-K2.6-bf16 weights")
    parser.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"],
                        help="Fusion variant to verify (default: V2)")
    parser.add_argument("--prompt",
                        default="Explain the concept of kernel fusion in one paragraph.",
                        help="Prompt used for numerical + throughput tests")
    parser.add_argument("--throughput-tokens", type=int, default=50,
                        help="Tokens generated for throughput measurement (default: 50)")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput measurement")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Kimi-K2.6 fusion verification")
    print(f"  Model  : {args.model_path}")
    print(f"  Variant: {args.variant}")
    print("=" * 60)

    # ── Test 1: CUDA extension (no model needed) ──────────────────────────────
    ext_ok = test_cuda_extension()
    if not ext_ok:
        print("\nERROR: CUDA extension failed to load — cannot continue.")
        sys.exit(1)

    # ── Load model ONCE (unfused) ─────────────────────────────────────────────
    print(f"\n[{INFO}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print(f"[{INFO}] Loading model (unfused)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[{INFO}] Model loaded — peak GPU memory: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── Phase A: unfused baseline ─────────────────────────────────────────────
    reference_logits = save_reference_logits(model, tokenizer, args.prompt)

    tps_unfused = None
    if not args.skip_throughput:
        print("\n── Unfused throughput (pre-patch) ──────────────────────────────")
        tps_unfused = measure_throughput(
            model, tokenizer, args.prompt, args.throughput_tokens, "unfused"
        )

    # ── Phase B: patch in-place ───────────────────────────────────────────────
    print(f"\n[{INFO}] Applying fusion (variant={args.variant})...")
    from kimi_patch import patch_kimi_model
    patch_kimi_model(model, variant=args.variant)
    torch.cuda.empty_cache()
    print(f"[{INFO}] Patch applied — peak GPU memory: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── Phase C: post-patch tests ─────────────────────────────────────────────
    cov_ok  = test_layer_coverage(model)
    norm_ok = test_norm_disabled(model)
    num_ok  = test_numerical(model, tokenizer, reference_logits, args.prompt)

    thr_ok = True
    if not args.skip_throughput:
        print("\n── Fused throughput (post-patch) ───────────────────────────────")
        tps_fused = measure_throughput(
            model, tokenizer, args.prompt, args.throughput_tokens, "fused"
        )
        speedup = tps_fused / tps_unfused
        print(f"  Speedup          : {speedup:.3f}×")
        thr_ok = log("throughput >= 0.98× unfused (no regression)", speedup >= 0.98,
                     f"speedup={speedup:.3f}×")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    all_passed = True
    for name, ok, detail in results:
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name}" + (f"  —  {detail}" if detail else ""))
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("\033[92m  All tests passed. Fusion is correct.\033[0m")
        sys.exit(0)
    else:
        print("\033[91m  One or more tests FAILED. See output above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()

    