/*
 * denominator.cpp
 *
 * Python/pybind11 bindings for the fused RMSNorm+Linear CUDA kernels
 * targeting Kimi-K2.6 (MoE, BF16).
 *
 * Kimi-K2.6 uses RMSNorm exclusively — no LayerNorm, no Welford algorithm.
 * RMSNorm requires only a single-pass sum-of-squares with no mean subtraction,
 * so the Welford / two-pass complexity is unnecessary here.
 *
 *  ── RMSNorm (Kimi-K2.6) ─────────────────────────────────────────────────
 *    rms(x) = sqrt(mean(x²) + eps) = sqrt(||x||₂² / h + eps)
 *  Gamma (γ) is pre-absorbed into W_new offline; the kernel divides
 *  raw_output (= x @ W_new.T) in-place by rms(x) and adds bias b_new.
 *
 *    rmsnorm_normalize             V1  256 threads, single downstream linear
 *    rmsnorm_normalize_512         V3  512 threads, single downstream linear
 *
 *  V2 (streaming) computes RMS via PyTorch ops on a side CUDA stream and
 *  does NOT call any kernel here — it is handled entirely in Python.
 *  The multi-linear fan-out variants also use PyTorch ops for RMS.
 *
 *  SwiGLU fusion (RMSNorm + MoE expert FFN) is not implemented yet.
 *  It will be added as a separate file once the MoE dispatch design is finalised.
 *
 * Tensor conventions:
 *   x          — 2D [T, h]        BF16 or FP32,  CUDA, contiguous
 *   raw_output — 2D [T, d_out]    same dtype,     CUDA, contiguous, modified in-place
 *   b_new      — 1D [d_out]       same dtype,     CUDA, contiguous
 *                                  Pass a zeros tensor when there is no bias
 *                                  (Python layer does this automatically).
 *   h          — int  == x.size(1)
 *   eps        — float > 0
 */

#include <torch/extension.h>

// ===========================================================================
// CUDA kernel forward declarations
// ===========================================================================
void rmsnorm_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);
void rmsnorm_normalize_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);

// ===========================================================================
// Validation macros
// ===========================================================================

#define CHECK_CUDA(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// Kimi-K2.6 uses BF16; FP32 is accepted for testing/debugging
#define CHECK_BF16_OR_FP32(x) \
    TORCH_CHECK( \
        (x).scalar_type() == at::kBFloat16 || \
        (x).scalar_type() == at::kFloat,      \
        #x " must be BFloat16 or Float32, got ", toString((x).scalar_type()) \
    )

#define CHECK_CUDA_CONTIGUOUS(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)

// Validate x and raw_output share dtype, and that b_new matches too
#define CHECK_DTYPE_MATCH(ref, other) \
    TORCH_CHECK( \
        (other).scalar_type() == (ref).scalar_type(), \
        #other " dtype (", toString((other).scalar_type()), \
        ") must match " #ref " dtype (", toString((ref).scalar_type()), ")" \
    )

// Shared validation for all rmsnorm_normalize-family calls
// (inline so the macro expansion is readable)
static inline void validate_rmsnorm_args(
    const torch::Tensor& x,
    const torch::Tensor& raw_output,
    const torch::Tensor& b_new,
    int h,
    float eps,
    const char* fn_name
) {
    CHECK_CUDA_CONTIGUOUS(x);
    CHECK_CUDA_CONTIGUOUS(raw_output);
    CHECK_CUDA_CONTIGUOUS(b_new);
    CHECK_BF16_OR_FP32(x);
    CHECK_DTYPE_MATCH(x, raw_output);
    CHECK_DTYPE_MATCH(x, b_new);

    TORCH_CHECK(x.dim() == 2,
        fn_name, ": x must be 2D [T, h], got ", x.dim(), "D");
    TORCH_CHECK(x.size(1) == (int64_t)h,
        fn_name, ": x.size(1)=", x.size(1), " must equal h=", h);

    TORCH_CHECK(raw_output.dim() == 2,
        fn_name, ": raw_output must be 2D [T, d_out], got ", raw_output.dim(), "D");
    TORCH_CHECK(raw_output.size(0) == x.size(0),
        fn_name, ": raw_output.size(0)=", raw_output.size(0),
        " must equal x.size(0)=", x.size(0), " (T tokens)");

    TORCH_CHECK(b_new.dim() == 1,
        fn_name, ": b_new must be 1D [d_out], got ", b_new.dim(), "D");
    TORCH_CHECK(b_new.size(0) == raw_output.size(1),
        fn_name, ": b_new.size(0)=", b_new.size(0),
        " must match raw_output.size(1)=", raw_output.size(1));

    TORCH_CHECK(eps > 0.0f,
        fn_name, ": eps must be positive, got ", eps);
}

// ===========================================================================
// RMSNorm wrappers — Kimi-K2.6 (no mean subtraction)
// ===========================================================================
// Called by FusedRMSNormLinearV1 / V3 in fused_rmsnorm_linear_kimi.py.
// V2 (streaming) uses PyTorch ops on a side stream and does not call here.
// Multi-linear fan-out variants also use PyTorch ops — no new bindings needed.

void rmsnorm_normalize(
    torch::Tensor x,          // [T, h]      input for rms(x) computation
    torch::Tensor raw_output, // [T, d_out]  x @ W_new.T, normalized in-place
    torch::Tensor b_new,      // [d_out]     bias (zeros tensor if no bias)
    int h,
    float eps
) {
    validate_rmsnorm_args(x, raw_output, b_new, h, eps, "rmsnorm_normalize");
    rmsnorm_normalize_cuda(x, raw_output, b_new, h, eps);
}

void rmsnorm_normalize_512(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    // Prefer this over rmsnorm_normalize when h >= 4096 (e.g. Kimi h=7168):
    // 512 threads amortise the warp-level reduction overhead better.
    validate_rmsnorm_args(x, raw_output, b_new, h, eps, "rmsnorm_normalize_512");
    rmsnorm_normalize_512_cuda(x, raw_output, b_new, h, eps);
}

// ===========================================================================
// Module registration
// ===========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // ── RMSNorm (Kimi-K2.6) ────────────────────────────────────────────────
    // Called by FusedRMSNormLinearV1 / V3 in fused_rmsnorm_linear_kimi.py.
    // Denominator: rms(x) = sqrt(mean(x²) + eps)  — NO mean subtraction.
    // raw_output is modified in-place: raw_output = raw_output / rms(x) + b_new.
    // Pass b_new=zeros([d_out]) when there is no bias.

    m.def("rmsnorm_normalize", &rmsnorm_normalize,
        "RMSNorm V1 [Kimi-K2.6, 256 threads]: fused RMSNorm normalize in-place.\n"
        "rms(x) = sqrt(mean(x²) + eps). No mean subtraction.\n"
        "raw_output[:] = raw_output / rms(x) + b_new  (in-place).\n"
        "Use when h < 4096 or as baseline. For h=7168, prefer rmsnorm_normalize_512.",
        py::arg("x"),
        py::arg("raw_output"),
        py::arg("b_new"),
        py::arg("h"),
        py::arg("eps"));

    m.def("rmsnorm_normalize_512", &rmsnorm_normalize_512,
        "RMSNorm V3 [Kimi-K2.6, 512 threads]: fused RMSNorm normalize in-place.\n"
        "Same semantics as rmsnorm_normalize; 512 threads give better occupancy\n"
        "for large hidden dims (e.g. Kimi h=7168).\n"
        "raw_output[:] = raw_output / rms(x) + b_new  (in-place).",
        py::arg("x"),
        py::arg("raw_output"),
        py::arg("b_new"),
        py::arg("h"),
        py::arg("eps"));
}



