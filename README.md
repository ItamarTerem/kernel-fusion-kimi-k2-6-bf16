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
│   └── verify_fusion.py         # 5-test correctness + throughput verification
│
├── scripts/
│   ├── setup_runtime.sh         # create venv, install PyTorch + dependencies
│   ├── download_model.sh        # download bullerwins/Kimi-K2.6-bf16 (~2 TB)
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

---

## What the Verification Tests Check

`tests/verify_fusion.py` runs five tests automatically after fusion:

| # | Test | Pass condition |
|---|------|----------------|
| 1 | **CUDA extension** — `denominator_cuda` loads; V1 and V3 kernels callable on a real BF16 tensor | No exception |
| 2 | **Layer coverage** — all 61 decoder layers have `fused_input_ln`, `fused_q_b`, `fused_kv_b` | 0 modules missing |
| 3 | **Norm disabled** — absorbed gamma vectors are all ones (norm is identity after fusion) | Max deviation < 1e-4 |
| 4 | **Numerical equivalence** — logit max-abs-diff between fused and unfused model on the same prompt | < 5e-2 (BF16 tolerance) |
| 5 | **Throughput** — tokens/sec fused vs unfused (50-token generation) | ≥ 0.98× (no regression) |

Run them independently at any time:

```bash
python tests/verify_fusion.py \
    --model-path models/Kimi-K2.6-bf16 \
    --variant V2
```

Options:
- `--variant` — `V1`, `V2` (default), or `V3`
- `--skip-numerical` — skip the logit comparison (saves loading the model twice)
- `--skip-throughput` — skip the tokens/sec benchmark

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