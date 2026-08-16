"""
Configuration for Parakeet STT API Server
"""
import os
from pathlib import Path

# Base directories (use absolute paths)
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

# Model information
MODEL_NAME = "parakeet-tdt-0.6b-v3"
MODEL_DESCRIPTION = "Parakeet TDT 0.6B v3 (Multilingual)"
MODEL_SIZE = "600M parameters"
MODEL_LANGUAGES = "25 European languages"

# Available precision levels
# FP32: GPU and CPU support (NeMo backend)
# INT8: CPU-only support (ONNX backend)
AVAILABLE_PRECISION = ["fp32", "int8"]

# ONNX Model Download URLs (only INT8 available)
ONNX_MODEL_URLS = {
    'int8': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2',
}

# ONNX model directory mapping
ONNX_MODEL_DIRS = {
    'int8': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8',
}

# NeMo model ID
NEMO_MODEL_ID = 'nvidia/parakeet-tdt-0.6b-v3'

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8022"))

# Model Configuration
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_THREADS = 0  # 0 = auto-detect (uses physical cores, following ONNX Runtime convention)
DEFAULT_PRECISION = 'fp32'
