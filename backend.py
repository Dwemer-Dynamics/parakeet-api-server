"""
STT Backend Abstraction
Supports multiple inference engines (NeMo, ONNX) with different precision levels

Supported Configurations:
- FP32: GPU and CPU support (NeMo backend)
- INT8: CPU-only support (ONNX backend)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import numpy as np


class STTBackend(ABC):
    """Abstract base class for STT backends"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Transcribe audio array

        Args:
            audio: Audio array (float32, mono, 16kHz)

        Returns:
            Dictionary with transcription results:
            {
                "text": str,
                "tokens": list (optional),
                "timestamps": list (optional),
                "language": str
            }
        """
        pass

    @abstractmethod
    def transcribe_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe audio from file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription results
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the expected sample rate for this backend"""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get the backend name (e.g., 'NeMo', 'ONNX')"""
        pass


def create_backend(
    precision: str,
    model_dir: Optional[Path] = None,
    num_threads: int = 0,
    force_cpu: bool = False
) -> STTBackend:
    """
    Factory function to create appropriate STT backend

    Args:
        precision: Precision level ('fp32' for GPU/CPU, 'int8' for CPU-only)
        model_dir: Directory containing model files (for ONNX)
        num_threads: Number of threads for CPU inference (0 = auto-detect)
        force_cpu: Force CPU usage even if GPU is available (FP32 only, INT8 is always CPU)

    Returns:
        Appropriate STT backend instance
    """
    import config

    if precision not in config.AVAILABLE_PRECISION:
        raise ValueError(
            f"Unsupported precision: {precision}. "
            f"Choose from: {', '.join(config.AVAILABLE_PRECISION)}"
        )

    # INT8 uses ONNX backend (pre-quantized model)
    if precision == 'int8':
        from inference_onnx import ONNXBackend
        return ONNXBackend(
            precision=precision,
            model_dir=model_dir,
            num_threads=num_threads,
            force_cpu=force_cpu
        )

    # FP32 uses NeMo backend (native PyTorch)
    elif precision == 'fp32':
        from inference_nemo import NeMoBackend
        return NeMoBackend(
            precision=precision,
            force_cpu=force_cpu,
            num_threads=num_threads
        )

    else:
        raise ValueError(f"Unsupported precision: {precision}")
