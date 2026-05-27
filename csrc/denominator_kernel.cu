/*
 * denominator_kernel.cu
 *
 * CUDA kernels for fused RMSNorm+Linear targeting Kimi-K2.6 (MoE, BF16).
 *
 * Kimi-K2.6 uses RMSNorm exclusively — no LayerNorm, no mean subtraction,
 * no Welford algorithm.  RMSNorm needs only a single-pass sum-of-squares.
 *
 *   rms(x) = sqrt( mean(x²) + eps ) = sqrt( ||x||₂² / h + eps )
 *
 * Gamma (γ) is pre-absorbed into W_new offline:
 *   W_new = W * γ
 * The kernel receives raw_output = x @ W_new.T (computed upstream by cuBLAS)
 * and modifies it in-place:
 *   raw_output[:] = raw_output / rms(x) + b_new
 *
 * Kernels provided
 * ─────────────────
 *  V1  (256 threads / block) — rmsnorm_normalize_{fp32,fp16,bf16}_kernel
 *      One block per token.  Prefer when h < 4096.
 *
 *  V3  (512 threads / block) — rmsnorm_normalize_512_{fp32,fp16,bf16}_kernel
 *      One block per token.  Better occupancy for large h (e.g. Kimi h=7168).
 *
 * Supported dtypes
 * ─────────────────
 *  FP32  — full native CUDA float
 *  FP16  — __half  (standard half)
 *  BF16  — __nv_bfloat16
 *
 * Host dispatchers (called from denominator.cpp)
 * ───────────────────────────────────────────────
 *  rmsnorm_normalize_cuda        — launches V1 (256-thread) kernel
 *  rmsnorm_normalize_512_cuda    — launches V3 (512-thread) kernel
 *
 * NOT implemented here (future files)
 * ─────────────────────────────────────
 *  SwiGLU fusion  — requires MoE dispatch redesign; will be a separate file.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ===========================================================================
// Constants
// ===========================================================================

static constexpr int WARP_SIZE    = 32;
static constexpr int BLOCK_SIZE   = 256;   // V1
static constexpr int BLOCK_SIZE_512 = 512; // V3

// ===========================================================================
// Warp- and block-level reduction utilities
// ===========================================================================

// Sum across all lanes in a warp using butterfly shuffles.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Sum across all warps in a block.
// shared[] must have at least (blockDim.x / WARP_SIZE) elements.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // Only the first warp participates in the second reduction
    val = (threadIdx.x < BLOCK / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

// ===========================================================================
// V1 RMSNorm kernels — 256 threads per block
// ===========================================================================
// One block per token (row of x).
// Pass 1: accumulate sum of x² across h elements (strided over threads).
// Result: rms = sqrt(sum_sq / h + eps).
// Pass 2: divide each raw_output element by rms and add bias.
//
// Dtype template keeps all three variants in one place; the compiler
// instantiates FP32 / FP16 / BF16 paths separately.

// ── FP32 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_fp32_kernel(
    const float* __restrict__ x,          // [T, h]
    float*       __restrict__ raw_output, // [T, d_out]  modified in-place
    const float* __restrict__ b_new,      // [d_out]
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    const int tok = blockIdx.x;
    const float* x_row   = x          + (int64_t)tok * h;
    float*       out_row  = raw_output + (int64_t)tok * d_out;

    // ── Pass 1: sum of squares ───────────────────────────────────────────────
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE)
        sum_sq += x_row[i] * x_row[i];

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    // ── Pass 2: normalize and add bias ───────────────────────────────────────
    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE)
        out_row[i] = out_row[i] * inv_rms + b_new[i];
}

// ── FP16 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_fp16_kernel(
    const __half* __restrict__ x,
    __half*       __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    const int tok = blockIdx.x;
    const __half* x_row  = x          + (int64_t)tok * h;
    __half*       out_row = raw_output + (int64_t)tok * d_out;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE)
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_rms
                                  + __half2float(b_new[i]));
}

// ── BF16 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    const int tok = blockIdx.x;
    const __nv_bfloat16* x_row  = x          + (int64_t)tok * h;
    __nv_bfloat16*       out_row = raw_output + (int64_t)tok * d_out;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE)
        out_row[i] = __float2bfloat16(
                         __bfloat162float(out_row[i]) * inv_rms
                         + __bfloat162float(b_new[i]));
}

// ===========================================================================
// V3 RMSNorm kernels — 512 threads per block
// ===========================================================================
// Identical logic to V1; doubled thread count improves occupancy for
// large hidden dims (Kimi-K2.6: h = 7168).

// ── FP32 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_512_fp32_kernel(
    const float* __restrict__ x,
    float*       __restrict__ raw_output,
    const float* __restrict__ b_new,
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    const int tok = blockIdx.x;
    const float* x_row  = x          + (int64_t)tok * h;
    float*       out_row = raw_output + (int64_t)tok * d_out;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE_512)
        sum_sq += x_row[i] * x_row[i];

    sum_sq = block_reduce_sum<BLOCK_SIZE_512>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE_512)
        out_row[i] = out_row[i] * inv_rms + b_new[i];
}

// ── FP16 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_512_fp16_kernel(
    const __half* __restrict__ x,
    __half*       __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    const int tok = blockIdx.x;
    const __half* x_row  = x          + (int64_t)tok * h;
    __half*       out_row = raw_output + (int64_t)tok * d_out;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE_512) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE_512>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE_512)
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_rms
                                  + __half2float(b_new[i]));
}

// ── BF16 ────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_normalize_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int h,
    int d_out,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    const int tok = blockIdx.x;
    const __nv_bfloat16* x_row  = x          + (int64_t)tok * h;
    __nv_bfloat16*       out_row = raw_output + (int64_t)tok * d_out;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < h; i += BLOCK_SIZE_512) {
        float v = __bfloat162float(x_row[i]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE_512>(sum_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0)
        rms_shared = rsqrtf(sum_sq / (float)h + eps);
    __syncthreads();

    const float inv_rms = rms_shared;

    for (int i = threadIdx.x; i < d_out; i += BLOCK_SIZE_512)
        out_row[i] = __float2bfloat16(
                         __bfloat162float(out_row[i]) * inv_rms
                         + __bfloat162float(b_new[i]));
}

// ===========================================================================
// Host dispatchers
// ===========================================================================

// ── V1 (256 threads) ────────────────────────────────────────────────────────

void rmsnorm_normalize_cuda(
    torch::Tensor x,          // [T, h]
    torch::Tensor raw_output, // [T, d_out]
    torch::Tensor b_new,      // [d_out]
    int h,
    float eps
) {
    const int T     = (int)x.size(0);
    const int d_out = (int)raw_output.size(1);

    const dim3 grid(T);
    const dim3 block(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        x.scalar_type(), "rmsnorm_normalize_cuda",
        [&]() {
            // Map scalar_t → correct kernel specialisation
            if constexpr (std::is_same_v<scalar_t, float>) {
                rmsnorm_normalize_fp32_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    raw_output.data_ptr<float>(),
                    b_new.data_ptr<float>(),
                    h, d_out, eps);
            } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
                rmsnorm_normalize_fp16_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
                    h, d_out, eps);
            } else {
                // BFloat16
                rmsnorm_normalize_bf16_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
                    h, d_out, eps);
            }
        }
    );
}

// ── V3 (512 threads) ────────────────────────────────────────────────────────

void rmsnorm_normalize_512_cuda(
    torch::Tensor x,          // [T, h]
    torch::Tensor raw_output, // [T, d_out]
    torch::Tensor b_new,      // [d_out]
    int h,
    float eps
) {
    const int T     = (int)x.size(0);
    const int d_out = (int)raw_output.size(1);

    const dim3 grid(T);
    const dim3 block(BLOCK_SIZE_512);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        x.scalar_type(), "rmsnorm_normalize_512_cuda",
        [&]() {
            if constexpr (std::is_same_v<scalar_t, float>) {
                rmsnorm_normalize_512_fp32_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    raw_output.data_ptr<float>(),
                    b_new.data_ptr<float>(),
                    h, d_out, eps);
            } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
                rmsnorm_normalize_512_fp16_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
                    h, d_out, eps);
            } else {
                rmsnorm_normalize_512_bf16_kernel<<<grid, block, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
                    h, d_out, eps);
            }
        }
    );
}