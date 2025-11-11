#!/bin/bash

# Parakeet STT API Server - Installation Script (Linux/macOS)

set -e

echo "========================================"
echo "Parakeet STT API Server - Installation"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_CMD="python3"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Check if CUDA is available
GPU_DETECTED=false
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n1
    GPU_DETECTED=true
else
    echo "⚠ nvidia-smi not found. No NVIDIA GPU detected."
fi

echo ""

# Prompt user for GPU support
if [ "$GPU_DETECTED" = true ]; then
    echo "Do you want to install PyTorch with CUDA support for GPU acceleration?"
    echo "This is recommended for 5-10x faster inference."
    read -p "Install with GPU support? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        INSTALL_GPU=false
        echo "→ Installing CPU-only version"
    else
        INSTALL_GPU=true
        echo "→ Installing GPU (CUDA 12.6) version"
    fi
else
    echo "No GPU detected. Installing CPU-only version."
    INSTALL_GPU=false
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo ""
echo "========================================"
echo "Installing PyTorch..."
echo "========================================"
echo ""

# Install PyTorch with or without CUDA
if [ "$INSTALL_GPU" = true ]; then
    echo "Installing PyTorch with CUDA 12.6 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "========================================"
echo "Installing sherpa-onnx..."
echo "========================================"
echo ""

# Install sherpa-onnx with GPU support if needed
if [ "$INSTALL_GPU" = true ]; then
    echo "Installing sherpa-onnx with CUDA 12.x support..."
    pip install sherpa-onnx==1.12.13+cuda12.cudnn9 -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
    echo ""
    echo "Installing CUDNN 9 (required for sherpa-onnx GPU support)..."
    pip install nvidia-cudnn-cu12
else
    echo "Installing sherpa-onnx (CPU-only)..."
    pip install "sherpa-onnx>=1.10.0"
fi

echo ""
echo "========================================"
echo "Installing other dependencies..."
echo "========================================"
echo ""

# Install other dependencies (sherpa-onnx and CUDNN already installed above)
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""

# Verify GPU support if installed
if [ "$INSTALL_GPU" = true ]; then
    echo "Verifying GPU support..."
    python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'✓ CUDA available: {cuda_available}'); print(f'✓ CUDA version: {torch.version.cuda if cuda_available else \"N/A\"}'); exit(0 if cuda_available else 1)" && echo "" || (echo "⚠ Warning: CUDA not available. GPU may not be properly configured." && echo "")
fi

echo "To start the server, run:"
echo "  ./start.sh"
echo ""
echo "Or activate the virtual environment and run manually:"
echo "  source venv/bin/activate"
echo "  python server.py"
echo ""
