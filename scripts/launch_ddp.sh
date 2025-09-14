#!/usr/bin/env bash
# Launch training on GPU 1

set -e

# Configuration
CONFIG_FILE="${1:-configs/train.yaml}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES=1

echo "Starting YOLO training on GPU 1..."
echo "  Data root: ../dataset_raw/data_cars_converted"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Model: YOLOv8n"

# Check if dataset exists
if [ ! -d "../dataset_raw/data_cars_converted" ]; then
    echo "Error: Dataset not found. Converting from raw data..."
    cd ../dataset_raw
    python convert_yolo_to_classification.py --input dataset_raw --output data_cars_converted
    cd ../dirtycar
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
