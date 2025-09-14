#!/usr/bin/env bash
# Complete pipeline: train → eval → threshold → export → serve
# One script to rule them all!

set -e

# Configuration
CONFIG_FILE="${1:-configs/train.yaml}"
DATA_ROOT="${2:-data_cars}"
TARGET_PRECISION="${3:-0.95}"

echo "=========================================="
echo "DirtyCar Complete Training Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Data: $DATA_ROOT"
echo "Target precision (clean): $TARGET_PRECISION"
echo "=========================================="

# Check prerequisites
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory not found: $DATA_ROOT"
    echo "Expected structure:"
    echo "  $DATA_ROOT/"
    echo "    train/clean/"
    echo "    train/dirty/"
    echo "    val/clean/"
    echo "    val/dirty/"
    echo "    test/clean/ (optional)"
    echo "    test/dirty/ (optional)"
    exit 1
fi

# Create output directories
mkdir -p artifacts logs runs

echo ""
echo "Step 1/5: Training model with DDP..."
echo "======================================"
bash scripts/launch_ddp.sh "$CONFIG_FILE"

if [ ! -f "artifacts/best.pt" ]; then
    echo "Error: Training failed - best.pt not found"
    exit 1
fi

echo ""
echo "Step 2/5: Evaluating model..."
echo "=============================="
python src/eval.py \
    --config "$CONFIG_FILE" \
    --checkpoint artifacts/best.pt \
    --data-root "$DATA_ROOT" \
    --datasets val test

echo ""
echo "Step 3/5: Selecting optimal threshold..."
echo "========================================"
python src/threshold_select.py \
    --config "$CONFIG_FILE" \
    --checkpoint artifacts/best.pt \
    --probs-source val \
    --target-precision-clean "$TARGET_PRECISION" \
    --out artifacts/threshold.json

if [ ! -f "artifacts/threshold.json" ]; then
    echo "Error: Threshold selection failed"
    exit 1
fi

echo ""
echo "Step 4/5: Exporting to ONNX..."
echo "==============================="
python src/export_onnx.py \
    --config "$CONFIG_FILE" \
    --checkpoint artifacts/best.pt \
    --out artifacts/best.onnx \
    --imgsz 256 \
    --half \
    --dynamic \
    --validate \
    --benchmark

if [ ! -f "artifacts/best.onnx" ]; then
    echo "Error: ONNX export failed"
    exit 1
fi

echo ""
echo "Step 5/5: Starting API server..."
echo "================================="
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/healthz"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Set environment variables and start server
export MODEL_PATH=artifacts/best.onnx
export IMGSZ=256

python -m uvicorn src.serve:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload

echo ""
echo "Pipeline completed successfully!"
echo "==============================="
echo "Artifacts created:"
echo "  - artifacts/best.pt (PyTorch model)"
echo "  - artifacts/best.onnx (ONNX model)"
echo "  - artifacts/threshold.json (optimal threshold)"
echo "  - artifacts/reports/ (evaluation reports)"
echo "  - logs/ (training logs)"
echo "  - runs/ (TensorBoard logs)"
