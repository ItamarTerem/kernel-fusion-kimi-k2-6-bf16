#!/usr/bin/env bash
# scripts/build_engine.sh
#
# End-to-end pipeline:
#   1. Create Python environment  (calls scripts/setup_runtime.sh if needed)
#   2. Download model weights      (calls scripts/download.sh if needed)
#   3. Apply RMSNorm+Linear fusion (runs run.py)
#   4. Verify fusion correctness   (runs verify_fusion.py)
#
# Usage:
#   bash scripts/build_engine.sh
#
# Overrides:
#   VENV        — venv path        (default: <root>/.venv)
#   MODEL_DIR   — model weights    (default: <root>/models/Kimi-K2.6-bf16)
#   MODEL_ID    — HF repo          (default: bullerwins/Kimi-K2.6-bf16)
#   VARIANT     — fusion variant   (default: V2)
#   SKIP_VERIFY — set to 1 to skip verification step

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="${ROOT}/scripts"

VENV="${VENV:-${ROOT}/.venv}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/Kimi-K2.6-bf16}"
MODEL_ID="${MODEL_ID:-bullerwins/Kimi-K2.6-bf16}"
VARIANT="${VARIANT:-V2}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "======================================"
echo " Kimi-K2.6 build engine"
echo " Root      : $ROOT"
echo " Venv      : $VENV"
echo " Model dir : $MODEL_DIR"
echo " Variant   : $VARIANT"
echo "======================================"

# ── [1/4] Environment ─────────────────────────────────────────────────────────
echo ""
echo "[1/4] Python environment..."

if [[ -d "$VENV" && -f "$VENV/bin/activate" ]]; then
    echo "  Reusing existing venv at $VENV"
else
    echo "  Venv not found — running setup_runtime.sh..."
    bash "${SCRIPTS}/setup_runtime.sh"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
echo "  Active Python: $(which python) — $(python --version)"
echo "  ✔ Environment ready"

# ── [2/4] Model weights ───────────────────────────────────────────────────────
echo ""
echo "[2/4] Model weights..."

# Check for at least one safetensors shard (sufficient to detect a completed download)
if ls "$MODEL_DIR"/*.safetensors &>/dev/null 2>&1; then
    echo "  Model already downloaded at $MODEL_DIR"
    echo "  ✔ Skipping download"
else
    echo "  Model not found — running download.sh..."
    MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" bash "${SCRIPTS}/download.sh"
fi

# ── [3/4] Apply fusion ────────────────────────────────────────────────────────
echo ""
echo "[3/4] Applying RMSNorm+Linear fusion (variant=$VARIANT)..."

cd "$ROOT"
python run.py \
    --model-path "$MODEL_DIR" \
    --variant "$VARIANT"

echo "  ✔ Fusion applied"

# ── [4/4] Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying fusion..."

if [[ "$SKIP_VERIFY" == "1" ]]; then
    echo "  SKIP_VERIFY=1 — skipping verification"
else
    python tests/verify_fusion.py \
        --model-path "$MODEL_DIR" \
        --variant "$VARIANT"

    VERIFY_EXIT=$?
    if [[ $VERIFY_EXIT -ne 0 ]]; then
        echo ""
        echo "ERROR: Fusion verification failed (exit $VERIFY_EXIT)."
        echo "Check the output above for which test(s) failed."
        exit $VERIFY_EXIT
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Build complete ✅"
echo ""
echo " To launch the SGLang server:"
echo "   source $VENV/bin/activate"
echo "   python -m sglang.launch_server \\"
echo "     --model-path $MODEL_DIR \\"
echo "     --trust-remote-code \\"
echo "     --dtype bfloat16 \\"
echo "     --tp 8 \\"
echo "     --port 30000"
echo ""
echo " To run verification again at any time:"
echo "   source $VENV/bin/activate"
echo "   python verify_fusion.py --model-path $MODEL_DIR --variant $VARIANT"
echo "======================================"