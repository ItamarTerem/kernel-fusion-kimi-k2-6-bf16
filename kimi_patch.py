"""
Monkey-patch Kimi-K2.6 (and compatible MLA models) to use fused RMSNorm+Linear.

Compatible models
─────────────────────────────────────────────────────────────────────────────
  Kimi-K2.6  (bullerwins/Kimi-K2.6-bf16)   — primary target
  DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite) — proxy model for testing

Both use MLA (Multi-head Latent Attention) with the same attribute names and
the same two-stage projection structure:

  input_layernorm → q_a_proj  → q_a_layernorm  → q_b_proj  → Q
  input_layernorm → kv_a_proj → kv_a_layernorm → kv_b_proj → K, V

This patch fuses all norm+linear pairs (γ absorbed into W_new offline via
compute_fused_weights; runtime via ops/fused_rmsnorm_linear.py):

  input_layernorm + [q_a_proj, kv_a_proj_with_mqa]  →  fused_input_ln  (fan-out)
  q_a_layernorm   + q_b_proj                         →  fused_q_b       (single)
  kv_a_layernorm  + kv_b_proj                        →  fused_kv_b      (single)

The fan-out (fused_input_ln) computes rms(hidden_states) once and feeds both
matmuls — saving one full RMSNorm computation per token per layer.

MoE post_attention_layernorm is unchanged (expert fusion is future work).

Model differences handled automatically
─────────────────────────────────────────────────────────────────────────────
  attn.scaling            — present on Kimi; absent on DeepSeek-V2-Lite
                            (resolved once at patch time via hasattr)
  attn.apply_rotary_pos_emb — method on Kimi; standalone function on DeepSeek
                            (resolved once at patch time via importlib)

Variants
─────────────────────────────────────────────────────────────────────────────
  "V1" — single: CUDA kernel 256 threads  |  fan-out: MultiLinear (no streams)
  "V2" — single: PyTorch + side stream    |  fan-out: MultiLinearV2 (streaming)
         Recommended for Blackwell B200 / DGX Spark
  "V3" — single: CUDA kernel 512 threads  |  fan-out: MultiLinear (no streams)
         Preferred for h=7168 (Kimi-K2.6 hidden dim)
"""

import math
import importlib
import torch
import torch.nn as nn

from ops.fused_rmsnorm_linear import (
    FusedRMSNormLinearV1,
    FusedRMSNormLinearV2,
    FusedRMSNormLinearV3,
    FusedRMSNormMultiLinearKimi,
    FusedRMSNormMultiLinearKimiV2,
)
from src.weight_transforms.weight_transform import compute_fused_weights

# ---------------------------------------------------------------------------
# Variant → (single-linear class, multi-linear fan-out class)
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, tuple[type, type]] = {
    "V1": (FusedRMSNormLinearV1, FusedRMSNormMultiLinearKimi),
    "V2": (FusedRMSNormLinearV2, FusedRMSNormMultiLinearKimiV2),
    "V3": (FusedRMSNormLinearV3, FusedRMSNormMultiLinearKimi),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(t: torch.Tensor | None, device) -> torch.Tensor | None:
    """Move tensor to device, or pass through None (no-bias linear)."""
    return t.to(device) if t is not None else None


def _disable_fused_norm(rms_norm: nn.Module) -> None:
    """
    Set absorbed RMSNorm gamma to ones after fusion.
    The patched forwards bypass these norm modules entirely, so this is a
    safety annotation rather than a functional requirement.
    """
    with torch.no_grad():
        rms_norm.weight.fill_(1.0)


def _resolve_rope_fn(attn: nn.Module):
    """
    Return the apply_rotary_pos_emb callable for this attention module.

    Kimi-K2.6:       it is a method on the attention class.
    DeepSeek-V2-Lite: it is a standalone module-level function.

    Resolved once at patch time and captured in the closure — never called
    via getattr on every forward pass.
    """
    if hasattr(attn, 'apply_rotary_pos_emb'):
        return attn.apply_rotary_pos_emb          # Kimi: bound method
    # DeepSeek-V2-Lite (and similar): look up the function in the modeling module
    mod = importlib.import_module(type(attn).__module__)
    fn  = getattr(mod, 'apply_rotary_pos_emb', None)
    if fn is None:
        raise AttributeError(
            f"Could not find apply_rotary_pos_emb on {type(attn).__name__} "
            f"or in module {type(attn).__module__}. "
            "Add a manual import or set attn.apply_rotary_pos_emb before patching."
        )
    return fn


def _resolve_scale(attn: nn.Module) -> float:
    """
    Return the softmax scale for this attention module.

    Kimi-K2.6:       stored as attn.scaling.
    DeepSeek-V2-Lite: not stored; must compute from qk_head_dim.

    Resolved once at patch time and captured in the closure.
    """
    if hasattr(attn, 'scaling'):
        return float(attn.scaling)
    return 1.0 / math.sqrt(attn.qk_head_dim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_kimi_model(
    model,
    device=None,
    variant: str = "V2",
) -> nn.Module:
    """
    Patch all decoder layers in a Kimi-K2.6 (or DeepSeek-V2-Lite) model
    to use fused RMSNorm+Linear modules.

    Args:
        model:   HuggingFace causal LM loaded with trust_remote_code=True
        device:  target device (defaults to model's current device)
        variant: kernel variant — "V1" | "V2" | "V3"
                 Default "V2" (PyTorch + streaming, best on Blackwell / DGX Spark)

    Returns:
        The patched model (modified in-place).
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {list(_VARIANTS)}"
        )

    if device is None:
        device = next(model.parameters()).device

    single_cls, multi_cls = _VARIANTS[variant]

    for layer in model.model.layers:
        _patch_decoder_layer(layer, device, single_cls, multi_cls)

    return model


# ---------------------------------------------------------------------------
# Per-layer patching
# ---------------------------------------------------------------------------

def _patch_decoder_layer(
    layer: nn.Module,
    device: torch.device,
    single_cls: type,
    multi_cls: type,
) -> None:
    attn = layer.self_attn

    # ── Fan-out: input_layernorm → [q_a_proj, kv_a_proj_with_mqa] ──────────
    W_qa,  b_qa,  h_in, eps_in = compute_fused_weights(
        layer.input_layernorm, attn.q_a_proj
    )
    W_kva, b_kva, _,    _      = compute_fused_weights(
        layer.input_layernorm, attn.kv_a_proj_with_mqa
    )
    attn.fused_input_ln = multi_cls(
        W_new_list=[W_qa.to(device),  W_kva.to(device)],
        b_new_list=[_to_device(b_qa, device), _to_device(b_kva, device)],
        h=h_in,
        eps=eps_in,
    )
    _disable_fused_norm(layer.input_layernorm)

    # ── Single: q_a_layernorm → q_b_proj ────────────────────────────────────
    W_qb, b_qb, h_q, eps_q = compute_fused_weights(
        attn.q_a_layernorm, attn.q_b_proj
    )
    attn.fused_q_b = single_cls(
        W_new=W_qb.to(device),
        b_new=_to_device(b_qb, device),
        h=h_q,
        eps=eps_q,
    )
    _disable_fused_norm(attn.q_a_layernorm)

    # ── Single: kv_a_layernorm → kv_b_proj ──────────────────────────────────
    W_kvb, b_kvb, h_kv, eps_kv = compute_fused_weights(
        attn.kv_a_layernorm, attn.kv_b_proj
    )
    attn.fused_kv_b = single_cls(
        W_new=W_kvb.to(device),
        b_new=_to_device(b_kvb, device),
        h=h_kv,
        eps=eps_kv,
    )
    _disable_fused_norm(attn.kv_a_layernorm)

    _patch_mla_forward(attn)
    _patch_layer_forward(layer)


# ---------------------------------------------------------------------------
# Patched MLA attention forward
# ---------------------------------------------------------------------------

def _patch_mla_forward(attn: nn.Module) -> None:
    """
    Replace the MLA attention forward to use fused norm+projection modules.

    Model differences are resolved HERE, once, before the closure is created:
      _apply_rope — callable that works for both Kimi and DeepSeek-V2-Lite
      _scale      — softmax scale (float) that works for both models
    Both are captured by the closure and never re-resolved on each call.
    """

    # ── Resolve model-specific callables once ────────────────────────────────
    _apply_rope: callable = _resolve_rope_fn(attn)
    _scale: float         = _resolve_scale(attn)

    # ── Closure ──────────────────────────────────────────────────────────────

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:

        bsz, q_len, _ = hidden_states.shape

        # Stage 1: input_layernorm fused with q_a_proj and kv_a_proj
        q_a_out, kv_a_with_rope = attn.fused_input_ln(hidden_states)

        kv_lora_rank     = attn.kv_lora_rank
        qk_rope_head_dim = attn.qk_rope_head_dim

        kv_a_out, k_pe = torch.split(
            kv_a_with_rope,
            [kv_lora_rank, qk_rope_head_dim],
            dim=-1,
        )
        k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)

        # Stage 2: inner norms fused with up-projections
        q  = attn.fused_q_b(q_a_out)
        kv = attn.fused_kv_b(kv_a_out)

        num_heads   = attn.num_heads
        qk_head_dim = attn.qk_head_dim
        q = q.view(bsz, q_len, num_heads, qk_head_dim).transpose(1, 2)

        num_kv_heads = attn.num_key_value_heads
        v_head_dim   = attn.v_head_dim
        k_nope, v = torch.split(
            kv.view(bsz, q_len, num_kv_heads, qk_head_dim + v_head_dim),
            [qk_head_dim, v_head_dim],
            dim=-1,
        )
        k_nope = k_nope.transpose(1, 2)
        v      = v.transpose(1, 2)

        # RoPE
        q_nope, q_pe = torch.split(
            q, [qk_head_dim - qk_rope_head_dim, qk_rope_head_dim], dim=-1
        )
        cos, sin = attn.rotary_emb(v, position_ids)
        q_pe, k_pe = _apply_rope(q_pe, k_pe, cos, sin)   # ← resolved at patch time

        k_pe = k_pe.expand(-1, num_kv_heads, -1, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, attn.layer_idx, cache_kwargs)

        # GQA expand
        if num_heads != num_kv_heads:
            reps = num_heads // num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=attn.attention_dropout if attn.training else 0.0,
            scale=_scale,                                 # ← resolved at patch time
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs

    attn.forward = patched_forward


# ---------------------------------------------------------------------------
# Patched decoder layer forward
# ---------------------------------------------------------------------------

def _patch_layer_forward(layer: nn.Module) -> None:
    """
    Replace the decoder layer forward.

    input_layernorm is SKIPPED — baked into fused_input_ln inside self_attn.
    post_attention_layernorm is KEPT — MoE expert fusion is future work.
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:

        residual = hidden_states

        # input_layernorm intentionally skipped — fused into fused_input_ln
        attn_outputs = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        attn_out           = attn_outputs[0]
        attn_weights       = attn_outputs[1] if output_attentions else None
        present_key_value  = (
            attn_outputs[2 if output_attentions else 1] if use_cache else None
        )

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    layer.forward = patched_forward