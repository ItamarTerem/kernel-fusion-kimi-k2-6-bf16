"""
Monkey-patch Kimi-K2.6 to use fused RMSNorm+Linear modules.

Kimi-K2.6 differs from Llama in two fundamental ways:
  1. MLA (Multi-head Latent Attention) instead of GQA.
     MLA compresses Q and KV through two-stage projections with inner norms:
       input_layernorm → q_a_proj  → q_a_layernorm → q_b_proj  → Q
       input_layernorm → kv_a_proj → kv_a_layernorm → kv_b_proj → K,V
     This gives FOUR norm+linear fusion points per attention block.

  2. MoE instead of MLP.
     post_attention_layernorm feeds a router gate AND a shared expert.
     The 384 routed experts receive dynamically dispatched tokens, making
     per-expert norm fusion impractical here (tracked as future work).

Fusions applied (γ absorbed into W_new offline via transform_kimi_layer):

  ┌─ input_layernorm ─┬──► q_a_proj          ┐  fused_input_ln
  │   (h=7168)        └──► kv_a_proj_with_mqa ┘  (MultiLinear fan-out, single rms)
  │
  ├─ q_a_layernorm  ──────► q_b_proj              fused_q_b   (single)
  │   (h=q_lora_rank)
  │
  └─ kv_a_layernorm ──────► kv_b_proj             fused_kv_b  (single)
      (h=kv_lora_rank)

NOT fused (future work):
  post_attention_layernorm + MoE experts  — requires patching the MoE dispatcher
  down_proj                               — follows SwiGLU activation, not a norm

Variants:
  "V1" — custom CUDA kernel, 256 threads, no streams
  "V2" — PyTorch ops, concurrent CUDA streams  (recommended for Blackwell B200)
  "V3" — custom CUDA kernel, 512 threads, no streams  (preferred for h=7168)

Expected output of transform_kimi_layer(layer):
  {
    "input_ln_q_a":  (W_new [d_qa,  7168], b_new|None, h=7168,        eps),
    "input_ln_kv_a": (W_new [d_kva, 7168], b_new|None, h=7168,        eps),
    "q_a_ln_q_b":    (W_new [d_q,  d_qa],  b_new|None, h=q_lora_rank, eps),
    "kv_a_ln_kv_b":  (W_new [d_kv, d_kva], b_new|None, h=kv_lora_rank,eps),
  }
"""

import torch
import torch.nn as nn
from ops.fused_rmsnorm_linear import (
    FusedRMSNormLinearV1,
    FusedRMSNormLinearV2,
    FusedRMSNormLinearV3,
    FusedRMSNormMultiLinearKimi,
    FusedRMSNormMultiLinearKimiV2,
)
from src.weight_transform import transform_kimi_layer

# ---------------------------------------------------------------------------
# Variant → (single-linear class, multi-linear class)
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, tuple[type, type]] = {
    "V1": (FusedRMSNormLinearV1,  FusedRMSNormMultiLinearKimi),
    "V2": (FusedRMSNormLinearV2,  FusedRMSNormMultiLinearKimiV2),
    "V3": (FusedRMSNormLinearV3,  FusedRMSNormMultiLinearKimi),
}


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
                 Recommended: "V2" (streams, Blackwell) or "V3" (512-thread kernel)

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

    for layer_idx, layer in enumerate(model.model.layers):
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
    """
    Patch a single Kimi-K2.6 decoder layer.

    Installs fused modules on layer.self_attn and monkey-patches the
    attention and decoder-layer forward methods.
    """
    weights = transform_kimi_layer(layer)

    # ── Fan-out: input_layernorm → [q_a_proj, kv_a_proj_with_mqa] ──────────
    W_qa,  b_qa,  h_in, eps = weights["input_ln_q_a"]
    W_kva, b_kva, _,    _   = weights["input_ln_kv_a"]

    layer.self_attn.fused_input_ln = multi_cls(
        W_new_list=[W_qa.to(device),  W_kva.to(device)],
        b_new_list=[b_qa.to(device) if b_qa  is not None else None,
                    b_kva.to(device) if b_kva is not None else None],
        h=h_in,
        eps=eps,
    )

    # ── Single: q_a_layernorm → q_b_proj ────────────────────────────────────
    W_qb, b_qb, h_q, eps_q = weights["q_a_ln_q_b"]
    layer.self_attn.fused_q_b = single_cls(
        W_new=W_qb.to(device),
        b_new=b_qb.to(device) if b_qb is not None else None,
        h=h_q,
        eps=eps_q,
    )

    # ── Single: kv_a_layernorm → kv_b_proj ──────────────────────────────────
    W_kvb, b_kvb, h_kv, eps_kv = weights["kv_a_ln_kv_b"]
    layer.self_attn.fused_kv_b = single_cls(
        W_new=W_kvb.to(device),
        b_new=b_kvb.to(device) if b_kvb is not None else None,
        h=h_kv,
        eps=eps_kv,
    )

    # ── Patch forwards ───────────────────────────────────────────────────────
    _patch_mla_forward(layer.self_attn)
    _patch_layer_forward(layer)


# ---------------------------------------------------------------------------
# Patched MLA attention forward
# ---------------------------------------------------------------------------

def _patch_mla_forward(attn: nn.Module) -> None:
    """
    Replace the MLA attention forward to use fused norm+projection modules.

    The patched forward receives RAW hidden_states (input_layernorm is skipped
    in the decoder layer forward and its effect is baked into fused_input_ln).

    Inner norms (q_a_layernorm, kv_a_layernorm) are similarly bypassed;
    fused_q_b and fused_kv_b absorb them into their weight matrices.

    Attribute names follow the DeepSeek V3 / Kimi-K2 custom modeling code:
      kv_a_proj_with_mqa  — projects hidden_states → [kv_lora, k_pe]
      kv_b_proj           — projects kv_lora       → interleaved K,V
    Adjust the split sizes below if your checkpoint uses different names.
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

        # ── Stage 1: input_layernorm fused with q_a_proj and kv_a_proj ──────
        # fused_input_ln returns [q_a_out, kv_a_with_rope_out]
        # Both outputs already include 1/rms(hidden_states) * gamma scaling.
        q_a_out, kv_a_with_rope = attn.fused_input_ln(hidden_states)

        # Split kv_a_with_rope → latent KV part and RoPE key part
        # kv_lora_rank: rank of the compressed KV latent space
        # qk_rope_head_dim: dim of the RoPE-only key component
        kv_lora_rank    = attn.kv_lora_rank           # e.g. 512
        qk_rope_head_dim = attn.qk_rope_head_dim      # e.g. 64

        kv_a_out, k_pe = torch.split(
            kv_a_with_rope,
            [kv_lora_rank, qk_rope_head_dim],
            dim=-1,
        )
        k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)

        # ── Stage 2: inner norms fused with up-projections ───────────────────
        # fused_q_b  absorbs q_a_layernorm  into q_b_proj weights
        # fused_kv_b absorbs kv_a_layernorm into kv_b_proj weights

        q = attn.fused_q_b(q_a_out)        # [bsz, q_len, n_heads * qk_head_dim]
        kv = attn.fused_kv_b(kv_a_out)     # [bsz, q_len, n_kv_heads * (k_dim + v_dim)]

        # Reshape Q
        num_heads  = attn.num_heads
        qk_head_dim = attn.qk_head_dim    # may differ from v_head_dim in MLA
        q = q.view(bsz, q_len, num_heads, qk_head_dim).transpose(1, 2)

        # Split interleaved K and V from kv_b_proj output
        num_kv_heads = attn.num_key_value_heads
        v_head_dim   = attn.v_head_dim
        k_nope, v = torch.split(
            kv.view(bsz, q_len, num_kv_heads, qk_head_dim + v_head_dim),
            [qk_head_dim, v_head_dim],
            dim=-1,
        )
        k_nope = k_nope.transpose(1, 2)   # [bsz, n_kv_heads, q_len, qk_head_dim]
        v      = v.transpose(1, 2)         # [bsz, n_kv_heads, q_len, v_head_dim]

        # ── RoPE: split Q into NoPE and RoPE parts, apply, recombine ─────────
        q_nope, q_pe = torch.split(q, [qk_head_dim - qk_rope_head_dim, qk_rope_head_dim], dim=-1)

        cos, sin = attn.rotary_emb(v, position_ids)
        q_pe, k_pe = attn.apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # Expand k_pe to match num_kv_heads and concatenate
        k_pe = k_pe.expand(-1, num_kv_heads, -1, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # ── KV cache update ───────────────────────────────────────────────────
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, attn.layer_idx, cache_kwargs)

        # ── Scaled dot-product attention ──────────────────────────────────────
        # Repeat K/V for GQA if num_heads > num_kv_heads
        if num_heads != num_kv_heads:
            reps = num_heads // num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=attn.attention_dropout if attn.training else 0.0,
            scale=attn.scaling,
        )

        # ── Output projection (NOT fused — follows attention, not a norm) ─────
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        return attn_output, None   # (output, attn_weights=None)

    attn.forward = patched_forward


# ---------------------------------------------------------------------------
# Patched decoder layer forward
# ---------------------------------------------------------------------------

def _patch_layer_forward(layer: nn.Module) -> None:
    """
    Replace the Kimi-K2.6 decoder layer forward.

    Key change vs original:
      - input_layernorm is SKIPPED (its effect lives in fused_input_ln inside attn)
      - post_attention_layernorm is KEPT (MoE expert fusion is future work)
      - MoE forward is otherwise unchanged (receives pre-normalized hidden states)
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

        # ── Self-attention block ───────────────────────────────────────────────
        residual = hidden_states

        # input_layernorm is INTENTIONALLY SKIPPED here:
        # fused_input_ln inside self_attn handles normalization + projection.
        attn_out, attn_weights = layer.self_attn(
            hidden_states=hidden_states,       # raw (un-normed) hidden states
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

        # ── MoE block ─────────────────────────────────────────────────────────
        residual = hidden_states

        # post_attention_layernorm is applied explicitly here.
        # MoE expert fusion is left for future work (requires patching dispatch).
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    layer.forward = patched_forward
