# DirtyCar Binary Classification - Production Docker Image
# Base: NVIDIA CUDA 12.4 runtime with cuDNN support

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels (compatible with CUDA 12.4 runtime)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

# Install other Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY README.md .

# Create directories for artifacts and data
RUN mkdir -p /app/artifacts /app/logs /app/runs /app/data_cars

# Set environment variables for the application
ENV MODEL_PATH=/app/artifacts/best.onnx
ENV IMGSZ=256
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 7439

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command - run FastAPI server
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
