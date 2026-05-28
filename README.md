# kernel-fusion-kimi-k2-6-bf16

Fused RMSNorm + Linear CUDA kernels for **Kimi-K2.6** (1T MoE, BF16).

---

## What This Project Does

Kimi-K2.6 uses Multi-head Latent Attention (MLA), which introduces additional normalization layers inside each attention block:

```
input_layernorm → q_a_proj  → q_a_layernorm  → q_b_proj  → Q
input_layernorm → kv_a_proj → kv_a_layernorm → kv_b_proj → K, V
```

Each norm-then-linear pair is normally two separate operations. This project **fuses them into one** by absorbing the norm's gamma (γ) scale factor into the weight matrix offline:

```
W_new = W × γ          (done once at load time)

runtime: out = (x @ W_new.T) / rms(x) + bias
         ↑ one kernel instead of norm + matmul
```

Across 61 decoder layers × 4 fusion points each, this removes 244 redundant norm passes per forward step.

### Fusion Variants

| Variant | Kernel | Best for |
|---------|--------|----------|
| V1 | Custom CUDA, 256 threads | h < 4096, baseline |
| V2 | PyTorch ops + side CUDA stream | Blackwell B200 / DGX Spark (recommended) |
| V3 | Custom CUDA, 512 threads | h = 7168 (Kimi hidden dim), H100 |

### Fusion Points Per Layer

| Module | Norm absorbed | Type |
|--------|--------------|------|
| `fused_input_ln` | `input_layernorm` → `[q_a_proj, kv_a_proj_with_mqa]` | fan-out (1 rms → 2 matmuls) |
| `fused_q_b` | `q_a_layernorm` → `q_b_proj` | single |
| `fused_kv_b` | `kv_a_layernorm` → `kv_b_proj` | single |

---

## Project Structure

```
kernel-fusion-kimi-k2-6-bf16/
├── csrc/
│   ├── denominator.cpp          # pybind11 bindings (V1 + V3 kernels)
│   └── denominator_kernel.cu    # CUDA kernels: RMSNorm V1 (256t) + V3 (512t)
│
├── ops/
│   ├── __init__.py
│   └── fused_rmsnorm_linear.py  # FusedRMSNormLinearV1/V2/V3 + MultiLinear classes
│
├── src/
│   ├── __init__.py
│   ├── load_cuda.py             # JIT-compiles the CUDA extension via torch.utils
│   └── weight_transforms/
│       ├── __init__.py
│       └── weight_transform.py  # compute_fused_weights: W_new = W × γ
│
├── tests/
│   ├── verify_fusion.py         # 5-test correctness + throughput (save-then-patch, Kimi)
│   └── verify_fusion_proxy.py   # dual-load numerical test on DeepSeek-V2-Lite proxy
│
├── scripts/
│   ├── setup_runtime.sh         # create venv, install PyTorch + dependencies
│   ├── download_model.sh        # download bullerwins/Kimi-K2.6-bf16 (~2 TB)
│   ├── download_proxy_model.sh  # download deepseek-ai/DeepSeek-V2-Lite (~31 GB)
│   └── build_engine.sh          # orchestrate env + download + fusion + verify
│
├── kimi_patch.py                # monkey-patches the HF model with fused modules
├── run.py                       # entry point: load → patch → smoke test
└── requirements.txt
```

---

## Hardware Requirements

| Resource | Minimum |
|----------|---------|
| GPU | NVIDIA Hopper (H100) or Blackwell (B200 / DGX Spark) |
| GPU memory | ~140 GB for BF16 weights (8× H100 80 GB or DGX Spark GB10) |
| Disk space | ~2 TB for model weights |
| CUDA toolkit | 12.4 or newer (12.8 recommended for Blackwell) |
| OS | Ubuntu 22.04 / 24.04 |

---

## Supported GPU Architectures

The CUDA extension is compiled for all of these automatically via `load_cuda.py`:

| Architecture | sm target | Example hardware |
|---|---|---|
| Ada Lovelace | sm_89 | RTX 6000 Ada, RTX 4000 series |
| Hopper | sm_90 | H100, H200, H110 |
| Blackwell DC | sm_100 | B100, B200, GB200, DGX Spark GB10 |
| Blackwell CC | sm_120 | RTX PRO 6000, RTX 5000 series |

---

## Quickstart

### Step 1 — Set up the environment

```bash
bash scripts/setup_runtime.sh
```

This installs system dependencies (`cmake`, `ninja-build`, `build-essential`),
creates a Python virtual environment at `.venv/`, installs PyTorch with the
correct CUDA wheel for your hardware, and installs all project dependencies
from `requirements.txt`.

### Step 2 — Activate the environment

```bash
source .venv/bin/activate
```

### Step 3 — Download the model weights

```bash
bash scripts/download_model.sh
```

Downloads **bullerwins/Kimi-K2.6-bf16** (~2 TB) to `models/Kimi-K2.6-bf16/`.
The script checks available disk space before starting and enables
`hf_transfer` for maximum download speed.

Model source: https://huggingface.co/bullerwins/Kimi-K2.6-bf16
(INT4 → BF16 upcast of the official `moonshotai/Kimi-K2.6` weights)

### Step 4 — Apply fusion and verify

```bash
bash scripts/build_engine.sh
```

This runs `run.py` to apply the fusion, then `tests/verify_fusion.py` to
confirm correctness. You should see all five tests pass.

> **Memory note:** the verification script loads the model only once.
> It saves reference logits before patching, applies the fusion in-place,
> then compares fused logits against the saved reference.
> Peak GPU memory = one model (~2 TB BF16), not two.

---

## What the Verification Tests Check

`tests/verify_fusion.py` runs five tests automatically after fusion:

| # | Test | Pass condition |
|---|------|----------------|
| 1 | **CUDA extension** — `denominator_cuda` loads; V1 and V3 kernels callable on a real BF16 tensor | No exception |
| 2 | **Layer coverage** — all 61 decoder layers have `fused_input_ln`, `fused_q_b`, `fused_kv_b` | 0 modules missing |
| 3 | **Norm disabled** — absorbed gamma vectors are all ones (norm is identity after fusion) | Max deviation < 1e-4 |
| 4 | **Numerical equivalence** — logit max-abs-diff between fused and reference logits; top-1 token unchanged is the hard gate (BF16 operation reordering causes expected diff of 0.1–0.35; V2 typically higher than V1/V3) | top-1 token match; max diff < 4e-1 |
| 5 | **Throughput** — tokens/sec fused vs unfused (50-token generation) | ≥ 0.98× (no regression) |

Run them independently at any time:

```bash
python tests/verify_fusion.py \
    --model-path models/Kimi-K2.6-bf16 \
    --variant V2
```

Options:
- `--variant` — `V1`, `V2` (default), or `V3`
- `--skip-throughput` — skip the tokens/sec benchmark
- `--prompt` — override the test prompt
- `--throughput-tokens` — number of tokens to generate for the throughput test (default: 50)

---

## Proxy Model Testing (DeepSeek-V2-Lite)

Running a clean dual-load numerical test on Kimi-K2.6 requires ~4 TB of GPU memory — two full copies of a 2 TB model. **DeepSeek-V2-Lite** (15.7B params, ~31 GB BF16) is a drop-in proxy: it uses the *exact same* MLA architecture with the same attribute names, so `patch_kimi_model()` applies to it without modification.

Two copies of DeepSeek-V2-Lite fit on a single H100 80 GB (~62 GB combined), enabling a clean numerical comparison where the reference model and the fused model are loaded independently and never share weight state.

### Step 1 — Download the proxy model

```bash
bash scripts/download_proxy_model.sh
```

Downloads `deepseek-ai/DeepSeek-V2-Lite` (~31 GB) to `models/DeepSeek-V2-Lite/`.

### Step 2 — Run the dual-load proxy test

```bash
python tests/verify_fusion_proxy.py \
    --model-path models/DeepSeek-V2-Lite \
    --variant V2
```

This runs the same five tests as `verify_fusion.py`, but with a clean dual-load approach:
- Phase A loads an unfused reference copy and saves reference logits
- Phase B loads a second independent copy and applies the patch
- Test 4 compares the two copies' logits directly — no shared state

Options match `verify_fusion.py`: `--variant`, `--skip-throughput`, `--prompt`, `--throughput-tokens`.

### Relationship to Kimi-K2.6

| Property | DeepSeek-V2-Lite | Kimi-K2.6 |
|---|---|---|
| Parameters | 15.7B | ~1T |
| Attention | MLA (identical structure) | MLA (identical structure) |
| Hidden dim | 2048 | 7168 |
| Layers | 27 | 61 |
| `attn.scaling` | absent (computed inline) | present |
| `apply_rotary_pos_emb` | module-level function | method on attn class |

Both differences are handled automatically by `kimi_patch.py`'s `_resolve_rope_fn()` and `_resolve_scale()` helpers, which resolve the correct callable once at patch time.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VENV` | `.venv` | Path to the Python virtual environment |
| `MODEL_DIR` | `models/Kimi-K2.6-bf16` | Local path for model weights |
| `MODEL_ID` | `bullerwins/Kimi-K2.6-bf16` | HuggingFace repo ID |
| `VARIANT` | `V2` | Fusion variant (`V1` / `V2` / `V3`) |
| `CUDA_VERSION` | auto-detected | Override CUDA toolkit version |
| `TORCH_VERSION` | `2.7.0` | PyTorch version to install |
| `FUSED_LN_BUILD_DIR` | `.jit_build` | JIT build cache (avoids recompilation) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enables fast C-based HF downloads |
| `SKIP_VERIFY` | `0` | Set to `1` to skip verification in `build_engine.sh` |

---

## Future Work

- **SwiGLU fusion** — fuse `post_attention_layernorm` with the MoE expert projections. Requires patching the MoE dispatcher; planned as a separate file once the dispatch design is finalised.
- **SGLang serving** — launch the fused model behind an OpenAI-compatible API for production inference and full latency/throughput benchmarking.
- **FP4 quantization** — apply NVIDIA ModelOpt NVFP4 quantization to the fused model for Blackwell GB200/B200 native FP4 throughput.
