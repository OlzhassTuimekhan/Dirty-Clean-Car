#!/usr/bin/env bash
# Environment setup script for DirtyCar project

set -e

echo "Setting up DirtyCar development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "Python version check passed: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and tools
echo "Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 wheels
echo "Installing PyTorch with CUDA 12.1 support..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

# Install other dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p artifacts logs runs data_cars/train/clean data_cars/train/dirty data_cars/val/clean data_cars/val/dirty data_cars/test/clean data_cars/test/dirty

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
fi

echo ""
echo "Environment setup completed successfully!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Place your data in the data_cars/ directory structure"
echo "2. Run training: bash scripts/run_all.sh"
echo "3. Or run individual steps as needed"
