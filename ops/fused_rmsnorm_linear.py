"""
Fused RMSNorm+Linear forward pass for Kimi-K2.6 (MoE, BF16).

Kimi-K2.6 uses RMSNorm — no mean subtraction, no norm bias (beta):
    rms(x) = sqrt(mean(x^2) + eps) = sqrt(||x||^2 / h + eps)
    out    = x / rms(x) * gamma          (gamma absorbed into W_new offline)

The matmul (x @ W_new.T) runs on the default CUDA stream while the RMS
denominator kernel (sqrt(||x||^2 / h + eps)) runs on a side stream
concurrently, hiding the reduction latency behind the matmul.

Offline weight precomputation (done once, before inference):
    W_new = W * gamma          # absorb RMSNorm scale into weight rows
    b_new = b                  # linear bias passed through (may be None)

Kimi-K2.6 architectural details:
    Hidden dim (attention / MoE input): 7168
    MoE expert hidden dim:              2048
    RMSNorm epsilon:                    1e-6
    RMSNorm bias (beta):                None  (RMSNorm has no beta)

Fan-out pattern: a single RMSNorm feeds multiple linear layers:
    - Attention pre-norm  → q_proj, kv_proj       (MLA)
    - MoE pre-norm        → gate_proj + experts
Use FusedRMSNormMultiLinearKimi / ...V2 for these cases.
"""

import os
import torch
import torch.nn.functional as F
from src.load_cuda import denominator_cuda

# Set FUSED_LN_NVTX=1 to emit NVTX ranges visible in Nsight Systems
_USE_NVTX = os.environ.get("FUSED_LN_NVTX", "0") == "1"


def _nvtx_range(name):
    """NVTX range annotation (no-op when profiling is disabled)."""
    if _USE_NVTX:
        return torch.cuda.nvtx.range(name)
    return _NullContext()


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros_bias_like(W: torch.Tensor) -> torch.Tensor:
    """Return a zero bias vector matching W's output dim and dtype/device."""
    return torch.zeros(W.size(0), dtype=W.dtype, device=W.device)


# ---------------------------------------------------------------------------
# V1 — RMSNorm+Linear, single linear, no streams (CUDA kernel path)
# ---------------------------------------------------------------------------

class FusedRMSNormLinearV1(torch.nn.Module):
    """
    V1: Fused RMSNorm+Linear, single downstream linear, no concurrent streams.

    Uses the custom CUDA kernel (rmsnorm_normalize) which:
      1. Reads x to compute rms = sqrt(mean(x^2) + eps)
      2. Divides raw_output (= x @ W_new.T) in-place by rms
      3. Adds b_new in-place (if present)

    Kimi-K2.6 change vs original LayerNorm version:
      - Denominator is sqrt(mean(x^2) + eps), NOT sqrt(mean((x-mu)^2) + eps)
      - b_new may be None (RMSNorm has no beta; linear bias optional)
    """

    def __init__(
        self,
        W_new: torch.Tensor,          # [d_out, d_in], gamma already absorbed
        b_new: torch.Tensor | None,   # [d_out] or None
        h: int,                       # input hidden dim (e.g. 7168)
        eps: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("W_new", W_new)
        # Kernel expects a bias tensor; pass zeros when there is no bias so we
        # avoid branching inside the hot path.
        bias = b_new if b_new is not None else _zeros_bias_like(W_new)
        self.register_buffer("b_new", bias)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormLinear_V1"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))          # [T, h]

            raw_output = F.linear(x_2d, self.W_new)   # [T, d_out]

            # In-place: raw_output = raw_output / rms(x) + b_new
            # Kernel reads x_2d for RMS; denominator is sqrt(mean(x^2)+eps)
            denominator_cuda.rmsnorm_normalize(
                x_2d, raw_output, self.b_new, self.h, self.eps
            )

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


# ---------------------------------------------------------------------------
# V2 — RMSNorm+Linear, single linear, WITH concurrent CUDA streams
# ---------------------------------------------------------------------------

class FusedRMSNormLinearV2(torch.nn.Module):
    """
    V2: Fused RMSNorm+Linear with concurrent CUDA streams.

    Timeline (both kernels enqueued before either completes):
      Stream default │ ══════════ matmul (x @ W_new.T) ══════════╗
      Stream side    │ ═══ rms reduction ═══╗                     ║
                     │                      ║ sync ───────────────╝
                     │                      └──► out = raw / rms + b_new

    The matmul dominates for large d_out, so the cheap RMS reduction
    (a single-pass sum-of-squares reduction over h) is fully hidden.

    Uses PyTorch ops for the RMS reduction on the side stream so that the
    result is a plain tensor and no extra kernel ABI is needed.
    """

    def __init__(
        self,
        W_new: torch.Tensor,
        b_new: torch.Tensor | None,
        h: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("W_new", W_new)
        if b_new is not None:
            self.register_buffer("b_new", b_new)
        else:
            self.b_new = None
        self.h = h
        self.eps = eps
        # A persistent side stream avoids stream-creation overhead per call
        self._side_stream = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormLinear_V2"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))           # [T, h]

            # ── Default stream: matmul ──────────────────────────────────────
            # Enqueued immediately; GPU executes asynchronously.
            with _nvtx_range("matmul"):
                raw = F.linear(x_2d, self.W_new)       # [T, d_out]

            # ── Side stream: RMS denominator ────────────────────────────────
            # Launched while the matmul is running on the default stream.
            # Both kernels read x_2d (no write conflict → safe concurrency).
            with torch.cuda.stream(self._side_stream):
                with _nvtx_range("rms_reduction"):
                    # rms shape: [T, 1] — one scalar per token
                    rms = torch.sqrt(
                        x_2d.to(torch.float32)          # accumulate in FP32
                            .pow(2)
                            .mean(dim=-1, keepdim=True)
                        + self.eps
                    ).to(x_2d.dtype)                    # cast back to BF16

            # ── Synchronise: default stream waits for side stream ───────────
            torch.cuda.current_stream().wait_stream(self._side_stream)

            # ── Combine ──────────────────────────────────────────────────────
            with _nvtx_range("apply_norm"):
                out = raw / rms
                if self.b_new is not None:
                    out = out + self.b_new

            out_shape = orig_shape[:-1] + (raw.size(-1),)
            return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# V3 — RMSNorm+Linear, single linear, 512-thread CUDA kernel, no streams
# ---------------------------------------------------------------------------

class FusedRMSNormLinearV3(torch.nn.Module):
    """
    V3: Same as V1 but uses the 512-thread kernel variant (rmsnorm_normalize_512).

    Prefer V3 when h >= 4096 (e.g. Kimi's h=7168) where the wider warp
    occupancy of 512 threads amortises the reduction overhead better.
    """

    def __init__(
        self,
        W_new: torch.Tensor,
        b_new: torch.Tensor | None,
        h: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("W_new", W_new)
        bias = b_new if b_new is not None else _zeros_bias_like(W_new)
        self.register_buffer("b_new", bias)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormLinear_V3"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_new)
            denominator_cuda.rmsnorm_normalize_512(
                x_2d, raw_output, self.b_new, self.h, self.eps
            )

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


# ---------------------------------------------------------------------------
# Kimi-K2.6 fan-out: one RMSNorm → N linear layers (no streams)
# ---------------------------------------------------------------------------

class FusedRMSNormMultiLinearKimi(torch.nn.Module):
    """
    Kimi-K2.6 fan-out pattern: a single RMSNorm feeds N linear layers.

    RMS is computed once from x, then each matmul output is divided by it.
    Gamma is pre-absorbed into every W_new_i offline.

    Use cases in Kimi-K2.6:
      Attention pre-norm  (h=7168, eps=1e-6):
          → q_proj  [d_attn_q, 7168]
          → kv_proj [d_attn_kv, 7168]   (MLA compressed KV)
      MoE pre-norm (h=7168, eps=1e-6):
          → gate_proj [n_experts, 7168]  (router)
          → expert up/gate projs         (handled per-expert by MoE dispatcher)

    Args:
        W_new_list: list of [d_out_i, h] weight tensors (gamma absorbed)
        b_new_list: list of [d_out_i] bias tensors or None per linear
        h:          input hidden dim (7168 for Kimi-K2.6 transformer layers)
        eps:        RMSNorm epsilon (default 1e-6)
    """

    def __init__(
        self,
        W_new_list: list[torch.Tensor],
        b_new_list: list[torch.Tensor | None],
        h: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(W_new_list) == len(b_new_list), (
            "W_new_list and b_new_list must have the same length"
        )
        self.h = h
        self.eps = eps
        self.n = len(W_new_list)

        for i, W in enumerate(W_new_list):
            self.register_buffer(f"W_{i}", W)

        self._has_bias = []
        for i, b in enumerate(b_new_list):
            if b is not None:
                self.register_buffer(f"b_{i}", b)
                self._has_bias.append(True)
            else:
                self._has_bias.append(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with _nvtx_range("FusedRMSNormMultiLinear_Kimi"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))           # [T, h]

            # Compute RMS once, reused across all downstream linears
            rms = torch.sqrt(
                x_2d.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps
            ).to(x_2d.dtype)                            # [T, 1], BF16

            outputs = []
            for i in range(self.n):
                W = getattr(self, f"W_{i}")
                raw = F.linear(x_2d, W)                # [T, d_out_i]
                out = raw / rms
                if self._has_bias[i]:
                    out = out + getattr(self, f"b_{i}")
                out_shape = orig_shape[:-1] + (out.size(-1),)
                outputs.append(out.reshape(out_shape))

            return outputs


# ---------------------------------------------------------------------------
# Kimi-K2.6 fan-out WITH concurrent streams (V2 variant)
# ---------------------------------------------------------------------------

class FusedRMSNormMultiLinearKimiV2(torch.nn.Module):
    """
    Kimi-K2.6 fan-out with concurrent CUDA streams.

    The first matmul and the RMS reduction run concurrently. Subsequent
    matmuls are pipelined after the sync point, so the RMS latency is
    fully hidden behind the first (and often largest) matmul.

    Timeline:
      Default stream │ ══ matmul_0 ══╗  matmul_1 ══  matmul_2 ══ ...
      Side stream    │ ═ rms ═╗      ║
                     │        ║sync──╝
                     │              └──► out_i = raw_i / rms + b_i

    Args: same as FusedRMSNormMultiLinearKimi.
    """

    def __init__(
        self,
        W_new_list: list[torch.Tensor],
        b_new_list: list[torch.Tensor | None],
        h: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(W_new_list) == len(b_new_list)
        self.h = h
        self.eps = eps
        self.n = len(W_new_list)

        for i, W in enumerate(W_new_list):
            self.register_buffer(f"W_{i}", W)

        self._has_bias = []
        for i, b in enumerate(b_new_list):
            if b is not None:
                self.register_buffer(f"b_{i}", b)
                self._has_bias.append(True)
            else:
                self._has_bias.append(False)

        self._side_stream = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with _nvtx_range("FusedRMSNormMultiLinear_KimiV2"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))           # [T, h]

            # ── Default stream: first matmul ────────────────────────────────
            W0 = getattr(self, "W_0")
            with _nvtx_range("matmul_0"):
                raw0 = F.linear(x_2d, W0)

            # ── Side stream: RMS reduction (concurrent with matmul_0) ───────
            with torch.cuda.stream(self._side_stream):
                with _nvtx_range("rms_reduction"):
                    rms = torch.sqrt(
                        x_2d.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
                        + self.eps
                    ).to(x_2d.dtype)                   # [T, 1]

            # ── Sync: default stream waits for rms to be ready ───────────────
            torch.cuda.current_stream().wait_stream(self._side_stream)

            # ── Apply norm + collect outputs ─────────────────────────────────
            outputs = []
            raws = [raw0] + [
                F.linear(x_2d, getattr(self, f"W_{i}"))
                for i in range(1, self.n)
            ]
            for i, raw in enumerate(raws):
                out = raw / rms
                if self._has_bias[i]:
                    out = out + getattr(self, f"b_{i}")
                out_shape = orig_shape[:-1] + (out.size(-1),)
                outputs.append(out.reshape(out_shape))

            return outputs


# ---------------------------------------------------------------------------
# Factory: select the right module for a given use case
# ---------------------------------------------------------------------------

def make_fused_rmsnorm_linear(
    W_new_list: list[torch.Tensor],
    b_new_list: list[torch.Tensor | None],
    h: int,
    eps: float = 1e-6,
    use_streams: bool = True,
    use_512_threads: bool = True,
) -> torch.nn.Module:
    """
    Return the appropriate fused module for the given weight configuration.

    Single linear  + streams=True               → FusedRMSNormLinearV2
    Single linear  + streams=False, 512t=True   → FusedRMSNormLinearV3
    Single linear  + streams=False, 512t=False  → FusedRMSNormLinearV1
    Multiple linears + streams=True             → FusedRMSNormMultiLinearKimiV2
    Multiple linears + streams=False            → FusedRMSNormMultiLinearKimi

    Recommended defaults for Kimi-K2.6 on Blackwell (B200):
        h=7168, eps=1e-6, use_streams=True, use_512_threads=True
    """
    n = len(W_new_list)
    if n == 1:
        W, b = W_new_list[0], b_new_list[0]
        if use_streams:
            return FusedRMSNormLinearV2(W, b, h, eps)
        elif use_512_threads:
            return FusedRMSNormLinearV3(W, b, h, eps)
        else:
            return FusedRMSNormLinearV1(W, b, h, eps)
    else:
        if use_streams:
            return FusedRMSNormMultiLinearKimiV2(W_new_list, b_new_list, h, eps)
        else:
            return FusedRMSNormMultiLinearKimi(W_new_list, b_new_list, h, eps)