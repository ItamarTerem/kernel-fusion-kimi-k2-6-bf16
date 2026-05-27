"""
weight_transform.py — Kimi K2.6 RMSNorm + Linear weight fusion.

WHAT THIS FILE DOES:
    Computes the fused weight W_new = W * gamma (offline, once at load time).
    Called by kimi_patch.py when patching the model before inference.

HOW IT FITS IN:
    kimi_patch.py calls:
        W_new, b, h, eps = compute_fused_weights(norm, linear)
    Then passes W_new (and b / None) to fused_rmsnorm_linear.py.

WHAT CHANGES IN THE WEIGHTS:
    Before:
        q_a_layernorm.weight = gamma  [1536]          (scales normalized x)
        q_b_proj.weight      = W      [12288, 1536]   (projects normalized x)

    After (fused):
        W_new = W * gamma             [12288, 1536]   (gamma absorbed into W)
        q_a_layernorm becomes a no-op (gamma set to ones by kimi_patch.py)

    Same math, one fewer elementwise multiply at runtime.

KIMI K2.6 FUSION TARGETS (called 244 times by kimi_patch.py):
    Layers 0-60, per layer (4 fusions):
      input_layernorm  -> q_a_proj          (norm_shape=[7168],  W_shape=[1536, 7168])
      input_layernorm  -> kv_a_proj_with_mqa(norm_shape=[7168],  W_shape=[576,  7168])
      q_a_layernorm    -> q_b_proj          (norm_shape=[1536],  W_shape=[12288,1536])
      kv_a_layernorm   -> kv_b_proj         (norm_shape=[512],   W_shape=[16384,512])

MEMORY NOTE:
    By default, compute_fused_weights returns a *new* tensor (copy of W scaled
    by gamma). The original linear.weight stays in GPU memory even though it is
    bypassed, so memory for those weights is temporarily doubled.

    To avoid this, pass inplace=True: the fusion is applied directly to
    linear.weight.data, no extra tensor is allocated, and the fused module
    receives a view of the (now-modified) weight.  Only do this if you are
    certain the original linear layer will never be called again (which is
    always the case after kimi_patch.py replaces the forward methods).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_eps(rms_norm) -> float:
    """
    Read epsilon from an RMSNorm module.

    Kimi-K2.6 uses DeepseekV3RMSNorm which stores epsilon as
    .variance_epsilon. Standard PyTorch RMSNorm uses .eps.
    We handle both.
    """
    if hasattr(rms_norm, "variance_epsilon"):
        return rms_norm.variance_epsilon
    if hasattr(rms_norm, "eps"):
        return rms_norm.eps
    raise AttributeError(
        f"{type(rms_norm).__name__} has neither .variance_epsilon nor .eps"
    )


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------

def compute_fused_weights(
    rms_norm,
    linear: nn.Linear,
    *,
    inplace: bool = False,
):
    """
    Compute the fused weight W_new = W * gamma for one RMSNorm + Linear pair.

    Args:
        rms_norm:  RMSNorm layer (e.g. q_a_layernorm, kv_a_layernorm,
                   input_layernorm). Must expose a .weight attribute (gamma).
        linear:    Linear layer immediately after the norm
                   (e.g. q_b_proj, kv_b_proj, q_a_proj).
        inplace:   If True, modify linear.weight.data in-place and return a
                   view — avoids allocating a second copy of W on the GPU.
                   Safe only after kimi_patch.py has replaced all forward
                   methods (the original linear is never called again).

    Returns:
        W_new:  Weight matrix [out, h] with gamma absorbed in.
                Same dtype as linear.weight (BF16 for Kimi-K2.6).
        b:      Bias vector [out], or None if linear has no bias.
                Callers should pass this directly; fused modules handle None.
        h:      Hidden dimension (in_features). Needed by the CUDA kernel to
                compute rms(x) = sqrt(mean(x²) + eps).
        eps:    Norm epsilon. Needed by the CUDA kernel.

    Example (called from kimi_patch.py):
        W_new, b, h, eps = compute_fused_weights(
            layer.self_attn.q_a_layernorm,
            layer.self_attn.q_b_proj,
        )
    """
    gamma = rms_norm.weight.data          # [h]           — may be FP32 or BF16
    W     = linear.weight.data            # [out, h]       — BF16 in Kimi-K2.6
    h     = gamma.shape[0]
    eps   = _get_eps(rms_norm)

    # Cast gamma to W's dtype before multiplying.
    # RMSNorm weights are sometimes stored in FP32 even in BF16 models (higher
    # precision norms).  Without this cast, W * gamma silently upcasts to FP32,
    # producing a FP32 W_new that mismatches every downstream BF16 tensor.
    gamma_cast = gamma.to(W.dtype)        # [h], now matches W dtype

    # Core fusion: each row of W (one output neuron) is scaled by gamma.
    # Broadcasting: [out, h] * [h] → [out, h]
    if inplace:
        W.mul_(gamma_cast)                # modifies linear.weight.data directly
        W_new = W                         # return the (now-modified) view
    else:
        W_new = W * gamma_cast            # new tensor; original W unchanged

    # Return None for bias when the linear has no bias term.
    # Fused modules check `if b_new is not None` and create internal zeros
    # when needed — do not substitute zeros here, it breaks that contract.
    b = linear.bias.data if linear.bias is not None else None

    return W_new, b, h, eps


# ---------------------------------------------------------------------------
# Multi-linear convenience wrapper
# ---------------------------------------------------------------------------

def compute_fused_weights_multi(
    rms_norm,
    linears: list[nn.Linear],
    *,
    inplace: bool = False,
) -> list[tuple]:
    """
    Compute fused weights for multiple Linear layers sharing one RMSNorm.

    Useful when one norm fans out into several projections (e.g. input_layernorm
    → [q_a_proj, kv_a_proj_with_mqa]).  Each linear gets its own W_new with
    gamma absorbed; h and eps are identical for all since they share the norm.

    Returns:
        List of (W_new, b, h, eps) tuples in the same order as `linears`.

    Example:
        results = compute_fused_weights_multi(
            layer.input_layernorm,
            [attn.q_a_proj, attn.kv_a_proj_with_mqa],
        )
        (W_qa, b_qa, h, eps), (W_kva, b_kva, _, _) = results
    """
    return [
        compute_fused_weights(rms_norm, lin, inplace=inplace)
        for lin in linears
    ]