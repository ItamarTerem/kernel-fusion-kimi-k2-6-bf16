#!/usr/bin/env bash
# scripts/download.sh
#
# Download Kimi-K2.6 BF16 weights from HuggingFace.
#
# Model : bullerwins/Kimi-K2.6-bf16
# Source: https://huggingface.co/bullerwins/Kimi-K2.6-bf16
# Size  : ~2 TB  (1T parameters × 2 bytes BF16)
# Notes : Upcasted from INT4 → BF16 by bullerwins.
#         Based on the official moonshotai/Kimi-K2.6 weights.
#         Model is NOT gated — no HF token required.
#
# Usage:
#   bash scripts/download.sh
#
# Overrides:
#   MODEL_ID   — HuggingFace repo ID   (default: bullerwins/Kimi-K2.6-bf16)
#   MODEL_DIR  — local save path       (default: <repo_root>/models/Kimi-K2.6-bf16)
#   REVISION   — git revision / branch (default: main)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-bullerwins/Kimi-K2.6-bf16}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/Kimi-K2.6-bf16}"
REVISION="${REVISION:-main}"

# Approximate size in GiB (1T params × 2 bytes BF16, plus config/tokenizer overhead)
APPROX_SIZE_GIB=2000

echo "======================================"
echo " Kimi-K2.6 BF16 downloader"
echo " MODEL_ID  : $MODEL_ID"
echo " MODEL_DIR : $MODEL_DIR"
echo " REVISION  : $REVISION"
echo " Est. size : ~${APPROX_SIZE_GIB} GiB (~2 TB)"
echo "======================================"

# ── Check hf CLI ─────────────────────────────────────────────────────────────
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found."
    echo "Activate your venv and ensure huggingface_hub is installed:"
    echo "  source .venv/bin/activate"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# ── Disk space check ─────────────────────────────────────────────────────────
PARENT_DIR="$(dirname "$MODEL_DIR")"
mkdir -p "$PARENT_DIR"

AVAILABLE_GIB=$(df -BG "$PARENT_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Available disk space at $PARENT_DIR: ${AVAILABLE_GIB} GiB"

if (( AVAILABLE_GIB < APPROX_SIZE_GIB )); then
    echo ""
    echo "ERROR: Insufficient disk space."
    echo "  Required : ~${APPROX_SIZE_GIB} GiB"
    echo "  Available: ${AVAILABLE_GIB} GiB"
    echo ""
    echo "Free up space or set MODEL_DIR to a path with enough capacity:"
    echo "  MODEL_DIR=/path/to/large/disk bash scripts/download.sh"
    exit 1
fi
echo "✔ Disk space OK"

# ── HF token warning (not required for this model) ───────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "NOTE: HF_TOKEN is not set."
    echo "bullerwins/Kimi-K2.6-bf16 is public — no token is needed."
    echo "If you see 401 errors, run: hf auth login"
fi

# ── Enable fast transfer if hf_transfer is installed ─────────────────────────
# hf_transfer is a C-based downloader; 5-10× faster than the default Python client.
# It is installed by requirements.txt and persisted in the venv activate.
# Exporting here covers the case where the user runs this script without activating.
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Prepare destination ───────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Starting download — this will take a while (~2 TB)..."
echo ""

CMD=(
    hf download "$MODEL_ID"
    --local-dir "$MODEL_DIR"
    --revision "$REVISION"
    --local-dir-use-symlinks False   # store real files; avoids cache symlink issues
)

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Download complete ✔"
echo " Weights saved to: $MODEL_DIR"
echo ""
echo " Next step — launch the SGLang server:"
echo "   source .venv/bin/activate"
echo "   python -m sglang.launch_server \\"
echo "     --model-path $MODEL_DIR \\"
echo "     --trust-remote-code \\"
echo "     --dtype bfloat16 \\"
echo "     --tp 8 \\"
echo "     --port 30000"
echo "======================================"