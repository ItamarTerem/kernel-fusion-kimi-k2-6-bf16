"""
weight_transform.py — Kimi K2.6 RMSNorm + Linear weight fusion.

WHAT THIS FILE DOES:
    Computes the new fused weight W_new = W * gamma.
    This is called by kimi_patch.py when patching the model.

HOW IT FITS IN:
    kimi_patch.py calls:
        W_new, b, h, eps = compute_fused_weights(norm, linear)
    Then passes W_new to fused_rmsnorm_linear.py to build the new layer.

WHAT CHANGES IN THE WEIGHTS:
    Before:
        q_a_layernorm has weight = gamma vector [1536]
        q_b_proj      has weight = W matrix    [12288, 1536]

    After:
        q_a_layernorm weight -> set to ones (becomes identity, no-op)
        q_b_proj      weight -> W_new = W * gamma (gamma absorbed in)

    Same math, one fewer operation at runtime.

KIMI K2.6 FUSION TARGETS (called 122 times total by kimi_patch.py):
    Layer 0-60:  q_a_layernorm  -> q_b_proj    (norm_shape=[1536], W_shape=[12288,1536])
    Layer 0-60:  kv_a_layernorm -> kv_b_proj   (norm_shape=[512],  W_shape=[16384,512])
"""

import torch
import torch.nn as nn


def _get_eps(rms_norm) -> float:
    """
    Read epsilon from RMSNorm.

    Kimi K2.6 uses DeepseekV3RMSNorm which stores epsilon as
    .variance_epsilon (not .eps like standard PyTorch RMSNorm).
    We handle both so this works regardless of which norm class is used.
    """
    if hasattr(rms_norm, 'eps'):
        return rms_norm.eps
    if hasattr(rms_norm, 'variance_epsilon'):
        return rms_norm.variance_epsilon
    raise AttributeError(
        f"{type(rms_norm).__name__} has neither .eps nor .variance_epsilon"
    )


def compute_fused_weights(rms_norm, linear: nn.Linear):
    """
    Compute fused weight W_new = W * gamma for one RMSNorm + Linear pair.

    THIS IS THE MAIN FUNCTION — kimi_patch.py calls this.

    Args:
        rms_norm:  The RMSNorm layer (e.g. q_a_layernorm or kv_a_layernorm).
                   Must have a .weight attribute (gamma vector).
        linear:    The Linear layer immediately after the norm
                   (e.g. q_b_proj or kv_b_proj).

    Returns:
        W_new:  New weight matrix [out, h] with gamma absorbed in.
                Same dtype as original (BF16). Pass to fused_rmsnorm_linear.py.
        b:      Bias vector [out]. Unchanged. Zeros if no bias.
        h:      Hidden dimension (in_features). Needed by the CUDA kernel
                to compute rms(x) = sqrt(mean(x^2) + eps).
        eps:    Norm epsilon. Needed by the CUDA kernel.

    Example (called from kimi_patch.py):
        W_new, b, h, eps = compute_fused_weights(
            layer.self_attn.q_a_layernorm,
            layer.self_attn.q_b_proj
        )
    """
    gamma = rms_norm.weight.data                  # [h]
    W     = linear.weight.data                    # [out, h]
    b     = (linear.bias.data
             if linear.bias is not None
             else torch.zeros(W.size(0), dtype=W.dtype, device=W.device))
    h     = gamma.shape[0]
    eps   = _get_eps(rms_norm)

    # Core fusion math: absorb gamma into W
    # Each row of W (one output neuron) gets scaled by the corresponding gamma
    # [out, h] * [h]  ->  [out, h]
    W_new = W * gamma

    return W_new, b, h, eps


def compute_fused_weights_multi(rms_norm, linears: list) -> list:
    """
    Compute fused weights for multiple Linear layers sharing one RMSNorm.
    Returns a list of (W_new, b, h, eps) in the same order as input linears.

    Use this if kimi_patch.py ever needs to fuse one norm into multiple
    projections at once.
    """
    return [compute_fused_weights(rms_norm, lin) for lin in linears]