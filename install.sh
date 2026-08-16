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
echo "[+] Found Python $PYTHON_VERSION"

# Check if CUDA is available
GPU_DETECTED=false
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n1
    GPU_DETECTED=true
else
    echo "[!] nvidia-smi not found. No NVIDIA GPU detected."
fi

echo ""

# Use a single PyTorch wheel channel for torch/torchvision/torchaudio so the
# venv does not end up with mixed CUDA builds after dependency installation.
GPU_TORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
GPU_TORCH_LABEL=""
CPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
DETECTED_DRIVER_CUDA_VERSION=""
DETECTED_TOOLKIT_CUDA_VERSION=""
RECOMMENDED_GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
RECOMMENDED_GPU_TORCH_LABEL="CUDA 13.0"
RECOMMENDED_GPU_TORCH_CHOICE="1"
RECOMMENDED_GPU_TORCH_REASON="best default for current NVIDIA drivers"

describe_gpu_torch_index() {
    case "$1" in
        "https://download.pytorch.org/whl/cu130")
            echo "CUDA 13.0"
            ;;
        "https://download.pytorch.org/whl/cu128")
            echo "CUDA 12.8"
            ;;
        "https://download.pytorch.org/whl/cu126")
            echo "CUDA 12.6"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

version_ge() {
    local detected="$1"
    local minimum="$2"
    [ -n "$detected" ] || return 1
    "$PYTHON_CMD" - "$detected" "$minimum" <<'PY'
import sys

detected = tuple(int(part) for part in sys.argv[1].split("."))
minimum = tuple(int(part) for part in sys.argv[2].split("."))
sys.exit(0 if detected >= minimum else 1)
PY
}

detect_cuda_versions() {
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_DRIVER_CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
    fi

    if command -v nvcc &> /dev/null; then
        DETECTED_TOOLKIT_CUDA_VERSION="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -n1)"
    fi
}

recommend_gpu_torch_index() {
    local detected_version=""

    if [ -n "$DETECTED_DRIVER_CUDA_VERSION" ]; then
        detected_version="$DETECTED_DRIVER_CUDA_VERSION"
        RECOMMENDED_GPU_TORCH_REASON="driver reports CUDA $DETECTED_DRIVER_CUDA_VERSION"
    elif [ -n "$DETECTED_TOOLKIT_CUDA_VERSION" ]; then
        detected_version="$DETECTED_TOOLKIT_CUDA_VERSION"
        RECOMMENDED_GPU_TORCH_REASON="local CUDA toolkit is $DETECTED_TOOLKIT_CUDA_VERSION"
    else
        RECOMMENDED_GPU_TORCH_REASON="best default for current NVIDIA drivers"
    fi

    if version_ge "$detected_version" "13.0"; then
        RECOMMENDED_GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
        RECOMMENDED_GPU_TORCH_LABEL="CUDA 13.0"
        RECOMMENDED_GPU_TORCH_CHOICE="1"
    elif version_ge "$detected_version" "12.8"; then
        RECOMMENDED_GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        RECOMMENDED_GPU_TORCH_LABEL="CUDA 12.8"
        RECOMMENDED_GPU_TORCH_CHOICE="2"
    else
        RECOMMENDED_GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
        RECOMMENDED_GPU_TORCH_LABEL="CUDA 12.6"
        RECOMMENDED_GPU_TORCH_CHOICE="3"
    fi
}

choose_gpu_torch_index() {
    echo "Recommended build for this system: $RECOMMENDED_GPU_TORCH_LABEL ($RECOMMENDED_GPU_TORCH_REASON)"
    echo "Choose the PyTorch GPU build:"
    echo "  1) CUDA 13.0"
    echo "  2) CUDA 12.8"
    echo "  3) CUDA 12.6"
    read -p "Pick 1, 2, or 3 [$RECOMMENDED_GPU_TORCH_CHOICE]: " GPU_BUILD_CHOICE

    if [ -z "$GPU_BUILD_CHOICE" ]; then
        GPU_BUILD_CHOICE="$RECOMMENDED_GPU_TORCH_CHOICE"
    fi

    case "$GPU_BUILD_CHOICE" in
        1)
            GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
            GPU_TORCH_LABEL="CUDA 13.0"
            ;;
        2)
            GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
            GPU_TORCH_LABEL="CUDA 12.8"
            ;;
        3)
            GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
            GPU_TORCH_LABEL="CUDA 12.6"
            ;;
        *)
            echo "Invalid choice. Using $RECOMMENDED_GPU_TORCH_LABEL."
            GPU_TORCH_INDEX_URL="$RECOMMENDED_GPU_TORCH_INDEX_URL"
            GPU_TORCH_LABEL="$RECOMMENDED_GPU_TORCH_LABEL"
            ;;
    esac
}

# Prompt user for GPU support
if [ "$GPU_DETECTED" = true ]; then
    detect_cuda_versions
    recommend_gpu_torch_index

    if [ -n "$DETECTED_DRIVER_CUDA_VERSION" ]; then
        echo "[i] NVIDIA driver reports CUDA $DETECTED_DRIVER_CUDA_VERSION"
    fi
    if [ -n "$DETECTED_TOOLKIT_CUDA_VERSION" ]; then
        echo "[i] Local CUDA toolkit: $DETECTED_TOOLKIT_CUDA_VERSION"
    fi

    echo "Do you want GPU acceleration for FP32 transcription?"
    echo "This is recommended for much faster inference."
    read -p "Install with GPU support? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        INSTALL_GPU=false
        echo "Installing CPU-only version"
    else
        INSTALL_GPU=true
        if [ -n "$GPU_TORCH_INDEX_URL" ]; then
            GPU_TORCH_LABEL="$(describe_gpu_torch_index "$GPU_TORCH_INDEX_URL")"
            echo "Using PyTorch GPU build from PYTORCH_INDEX_URL: $GPU_TORCH_LABEL"
        else
            choose_gpu_torch_index
        fi
        echo "Installing GPU version: $GPU_TORCH_LABEL"
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
    echo "[+] Virtual environment created"
else
    echo "[+] Virtual environment already exists"
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
    echo "Installing PyTorch with $GPU_TORCH_LABEL support..."
    echo "  $GPU_TORCH_INDEX_URL"
    pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url "$GPU_TORCH_INDEX_URL"
else
    echo "Installing PyTorch (CPU-only)..."
    pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url "$CPU_TORCH_INDEX_URL"
fi

echo ""
echo "========================================"
echo "Installing sherpa-onnx..."
echo "========================================"
echo ""

# INT8 is a CPU-only fallback; keep its dependencies independent from the
# selected PyTorch CUDA runtime used by the FP32 NeMo backend.
echo "Installing sherpa-onnx CPU backend..."
pip install --upgrade --force-reinstall --no-deps "sherpa-onnx>=1.10.0"

echo ""
echo "========================================"
echo "Installing other dependencies..."
echo "========================================"
echo ""

# Install other dependencies (sherpa-onnx is already installed above)
pip install -r requirements.txt

echo ""
# Verify the exact runtime and prepare the selected model before first startup.
if [ "$INSTALL_GPU" = true ]; then
    echo "Verifying GPU runtime compatibility..."
    python startup_state.py check-runtime --require-cuda
    echo "Preparing FP32 NeMo model cache..."
    python model_downloader.py --precision fp32
else
    echo "Verifying CPU runtime compatibility..."
    python startup_state.py check-runtime
    echo "Preparing INT8 ONNX model files..."
    python model_downloader.py --precision int8
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""

echo "To start the server, run:"
echo "  ./start.sh"
echo ""
echo "Or activate the virtual environment and run manually:"
echo "  source venv/bin/activate"
echo "  python server.py"
echo ""
