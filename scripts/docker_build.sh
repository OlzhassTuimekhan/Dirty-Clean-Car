#!/usr/bin/env bash
# Docker build and deployment script for DirtyCar

set -e

# Configuration
IMAGE_NAME="${1:-dirtycar}"
TAG="${2:-latest}"
DOCKERFILE="${3:-docker/Dockerfile}"

echo "Building DirtyCar Docker image..."
echo "  Image: $IMAGE_NAME:$TAG"
echo "  Dockerfile: $DOCKERFILE"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is available
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA Container Toolkit not properly configured."
    echo "GPU support may not work. Please install NVIDIA Container Toolkit."
fi

# Build image
echo "Building Docker image..."
docker build \
    -f "$DOCKERFILE" \
    -t "$IMAGE_NAME:$TAG" \
    .

echo "Docker image built successfully: $IMAGE_NAME:$TAG"

# Test the image
echo "Testing Docker image..."
docker run --rm "$IMAGE_NAME:$TAG" python --version

echo "Build completed successfully!"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 8000:8000 -v \$(pwd)/artifacts:/app/artifacts $IMAGE_NAME:$TAG"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose -f docker/docker-compose.yml up"
