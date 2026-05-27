"""
Monkey-patch Kimi-K2.6 to use fused RMSNorm+Linear modules.

Kimi-K2.6 MLA attention has inner norms before the up-projections:
  input_layernorm → q_a_proj  → q_a_layernorm  → q_b_proj  → Q
  input_layernorm → kv_a_proj → kv_a_layernorm → kv_b_proj → K,V

This patch fuses the two inner pairs (offline W_new via weight_transform.py,
runtime via ops/fused_rmsnorm_linear.py):

  q_a_layernorm  → q_b_proj   →  fused_q_b
  kv_a_layernorm → kv_b_proj  →  fused_kv_b

input_layernorm, q_a_proj, and kv_a_proj_with_mqa stay on the original path.
MoE post_attention_layernorm is unchanged (expert fusion is future work).

Variants:
  "V1" — custom CUDA kernel, 256 threads, no streams
  "V2" — PyTorch ops, concurrent CUDA streams  (recommended for Blackwell B200)
  "V3" — custom CUDA kernel, 512 threads, no streams
"""

import torch
import torch.nn as nn

from ops.fused_rmsnorm_linear import (
    FusedRMSNormLinearV1,
    FusedRMSNormLinearV2,
    FusedRMSNormLinearV3,
)
from src.weight_transforms.weight_transform import compute_fused_weights

# ---------------------------------------------------------------------------
# Variant → fused single-linear class
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, type] = {
    "V1": FusedRMSNormLinearV1,
    "V2": FusedRMSNormLinearV2,
    "V3": FusedRMSNormLinearV3,
}


def _disable_fused_norm(rms_norm: nn.Module) -> None:
    """Set absorbed RMSNorm gamma to ones (identity) after fusion."""
    with torch.no_grad():
        rms_norm.weight.fill_(1.0)


def _bias_for_fused(linear: nn.Linear, b: torch.Tensor) -> torch.Tensor | None:
    """Pass None to fused modules when the original linear has no bias."""
    return b if linear.bias is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_kimi_model(
    model,
    device=None,
    variant: str = "V2",
) -> nn.Module:
    """
    Patch all decoder layers in a Kimi-K2.6 model to use fused RMSNorm+Linear.

    Args:
        model:   HuggingFace KimiForCausalLM (loaded with trust_remote_code=True)
        device:  target device (defaults to model's current device)
        variant: kernel variant — "V1" | "V2" | "V3"

    Returns:
        The patched model (modified in-place).
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {list(_VARIANTS)}"
        )

    if device is None:
        device = next(model.parameters()).device

    fused_cls = _VARIANTS[variant]

    for layer in model.model.layers:
        _patch_decoder_layer(layer, device, fused_cls)

    return model


# ---------------------------------------------------------------------------
# Per-layer patching
# ---------------------------------------------------------------------------

def _patch_decoder_layer(
    layer: nn.Module,
    device: torch.device,
    fused_cls: type,
) -> None:
    """
    Patch a single Kimi-K2.6 decoder layer.

    Computes fused weights, installs fused modules on layer.self_attn, and
    replaces the attention and decoder-layer forward methods.
    """
    attn = layer.self_attn

    # q_a_layernorm → q_b_proj
    W_qb, b_qb, h_q, eps_q = compute_fused_weights(
        attn.q_a_layernorm,
        attn.q_b_proj,
    )
    attn.fused_q_b = fused_cls(
        W_new=W_qb.to(device),
        b_new=_bias_for_fused(attn.q_b_proj, b_qb.to(device)),
        h=h_q,
        eps=eps_q,
    )
    _disable_fused_norm(attn.q_a_layernorm)

    # kv_a_layernorm → kv_b_proj
    W_kvb, b_kvb, h_kv, eps_kv = compute_fused_weights(
        attn.kv_a_layernorm,
        attn.kv_b_proj,
    )
    attn.fused_kv_b = fused_cls(
        W_new=W_kvb.to(device),
        b_new=_bias_for_fused(attn.kv_b_proj, b_kvb.to(device)),
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
    Replace MLA attention forward to use fused inner norm+projection modules.

    Expects hidden_states after input_layernorm (applied in decoder layer).
    Skips q_a_layernorm/q_b_proj and kv_a_layernorm/kv_b_proj in favor of
    fused_q_b and fused_kv_b.

    Attribute names follow DeepSeek V3 / Kimi-K2 custom modeling code:
      kv_a_proj_with_mqa — projects normed hidden_states → [kv_lora, k_pe]
      kv_b_proj          — up-project latent KV (fused with kv_a_layernorm)
    """

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

        # Stage 1: low-rank projections (input_layernorm already applied)
        q_a_out = attn.q_a_proj(hidden_states)
        kv_a_with_rope = attn.kv_a_proj_with_mqa(hidden_states)

        kv_lora_rank = attn.kv_lora_rank
        qk_rope_head_dim = attn.qk_rope_head_dim

        kv_a_out, k_pe = torch.split(
            kv_a_with_rope,
            [kv_lora_rank, qk_rope_head_dim],
            dim=-1,
        )
        k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)

        # Stage 2: fused inner norm + up-projections
        q = attn.fused_q_b(q_a_out)
        kv = attn.fused_kv_b(kv_a_out)

        num_heads = attn.num_heads
        qk_head_dim = attn.qk_head_dim
        q = q.view(bsz, q_len, num_heads, qk_head_dim).transpose(1, 2)

        num_kv_heads = attn.num_key_value_heads
        v_head_dim = attn.v_head_dim
        k_nope, v = torch.split(
            kv.view(bsz, q_len, num_kv_heads, qk_head_dim + v_head_dim),
            [qk_head_dim, v_head_dim],
            dim=-1,
        )
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)

        q_nope, q_pe = torch.split(
            q, [qk_head_dim - qk_rope_head_dim, qk_rope_head_dim], dim=-1
        )

        cos, sin = attn.rotary_emb(v, position_ids)
        q_pe, k_pe = attn.apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        k_pe = k_pe.expand(-1, num_kv_heads, -1, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, attn.layer_idx, cache_kwargs)

        if num_heads != num_kv_heads:
            reps = num_heads // num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=attn.attention_dropout if attn.training else 0.0,
            scale=attn.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        return attn_output, None

    attn.forward = patched_forward


# ---------------------------------------------------------------------------
# Patched decoder layer forward
# ---------------------------------------------------------------------------

def _patch_layer_forward(layer: nn.Module) -> None:
    """
    Replace the Kimi-K2.6 decoder layer forward.

    input_layernorm is kept; fused modules live inside self_attn.
    post_attention_layernorm + MoE are unchanged.
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
        hidden_states = layer.input_layernorm(hidden_states)

        attn_out, attn_weights = layer.self_attn(
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
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    layer.forward = patched_forward
