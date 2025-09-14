#!/usr/bin/env bash
# Launch distributed training on 2×RTX 3090 GPUs

set -e

# Configuration
CONFIG_FILE="${1:-configs/train.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "Starting distributed training..."
echo "  Config: $CONFIG_FILE"
echo "  GPUs: $NPROC_PER_NODE"
echo "  Master port: $MASTER_PORT"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU training may not work."
fi

# Set environment variables
export MASTER_PORT=$MASTER_PORT
export CUDA_VISIBLE_DEVICES=0,1

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/train.py \
    --config "$CONFIG_FILE"

echo "Training completed!"
