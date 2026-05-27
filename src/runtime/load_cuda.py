"""
Load the CUDA denominator extension, building via JIT if needed.

Automatically detects all visible CUDA devices and compiles the kernel
with the appropriate architecture flags for each.

Supported targets:
  sm_89  — Ada Lovelace     (RTX 6000 Ada Generation, RTX 4000 series)
  sm_90  — Hopper           (H100, H200, H110)
  sm_100 — Blackwell DC     (B100, B200, GB200, DGX Spark GB10)
  sm_120 — Blackwell CC     (RTX PRO 6000 Blackwell, RTX 5000 series)

Environment variables:
  FUSED_LN_BUILD_DIR   — directory for JIT build artifacts
                         (default: <repo_root>/.jit_build)
                         Reusing the same directory avoids recompilation
                         on subsequent imports.
  FUSED_LN_VERBOSE     — set to "1" to print NVCC compiler output
  FUSED_LN_ARCHS       — override detected architectures with a
                         comma-separated list of sm strings, e.g.
                         "sm_90,sm_100" to force specific targets.
"""

import os
import torch
import torch.utils.cpp_extension as ext

# ---------------------------------------------------------------------------
# Bypass CUDA version check
# NVCC toolkit 12.x is forward-compatible with driver 13.x; the check is
# overly strict and blocks valid configurations.
# ---------------------------------------------------------------------------
_orig_check = ext._check_cuda_version
ext._check_cuda_version = lambda *a, **k: None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# GPU compute capability → NVCC sm target
# ---------------------------------------------------------------------------
_CAP_TO_SM: dict[tuple[int, int], str] = {
    (8, 9):  "sm_89",   # Ada Lovelace   — RTX 6000 Ada, RTX 4000 series
    (9, 0):  "sm_90",   # Hopper         — H100, H200, H110
    (10, 0): "sm_100",  # Blackwell DC   — B100, B200, GB200, DGX Spark GB10
    (12, 0): "sm_120",  # Blackwell CC   — RTX PRO 6000, RTX 5000 series
}

# Default architecture set used when no CUDA device is visible at build time
# (e.g. compiling on a CPU-only head node for later deployment on GPU nodes).
_DEFAULT_ARCHS = ["sm_89", "sm_90", "sm_100", "sm_120"]


def _detect_sm_targets() -> list[str]:
    """
    Return a deduplicated, sorted list of sm_XX strings covering all visible
    CUDA devices, or the override list from FUSED_LN_ARCHS if set.
    """
    # Manual override
    override = os.environ.get("FUSED_LN_ARCHS", "").strip()
    if override:
        targets = [s.strip() for s in override.split(",") if s.strip()]
        return sorted(set(targets))

    # No CUDA visible — fall back to default multi-arch build
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print(
            "[load_cuda] No CUDA devices visible; compiling for default "
            f"architectures: {_DEFAULT_ARCHS}"
        )
        return _DEFAULT_ARCHS

    targets: set[str] = set()
    for dev in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(dev)
        sm = _CAP_TO_SM.get(cap)
        if sm is None:
            # Unknown capability — construct sm string and warn
            sm = f"sm_{cap[0]}{cap[1]}"
            print(
                f"[load_cuda] Device {dev}: unknown capability {cap}, "
                f"falling back to {sm}. Add it to _CAP_TO_SM if incorrect."
            )
        targets.add(sm)

    return sorted(targets)


def _build_nvcc_arch_flags(sm_targets: list[str]) -> list[str]:
    """
    Convert a list of sm strings into NVCC -gencode flags.

    Each sm target produces one flag of the form:
        -gencode=arch=compute_XX,code=sm_XX

    Using -gencode (rather than -arch) allows mixing multiple targets in
    a single binary and avoids conflicts when more than one GPU is present.
    """
    flags = []
    for sm in sm_targets:
        compute = sm.replace("sm_", "compute_")
        flags.append(f"-gencode=arch={compute},code={sm}")
    return flags


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

_sm_targets   = _detect_sm_targets()
_arch_flags   = _build_nvcc_arch_flags(_sm_targets)
_build_dir    = os.environ.get(
    "FUSED_LN_BUILD_DIR",
    os.path.join(_ROOT, ".jit_build"),
)
_verbose      = os.environ.get("FUSED_LN_VERBOSE", "0") == "1"

os.makedirs(_build_dir, exist_ok=True)

if _verbose:
    print(f"[load_cuda] Compiling for architectures: {_sm_targets}")
    print(f"[load_cuda] NVCC arch flags: {_arch_flags}")
    print(f"[load_cuda] Build directory: {_build_dir}")

try:
    denominator_cuda = ext.load(
        name="denominator_cuda",
        sources=[
            os.path.join(_ROOT, "csrc", "denominator.cpp"),
            os.path.join(_ROOT, "csrc", "denominator_kernel.cu"),
        ],
        extra_cuda_cflags=_arch_flags + ["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        build_directory=_build_dir,
        verbose=_verbose,
    )
finally:
    # Always restore the original check, even if compilation fails
    ext._check_cuda_version = _orig_check