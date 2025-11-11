"""
NeMo-based STT Backend
Supports FP32 precision for Parakeet TDT 0.6B v3
"""
import numpy as np
import torch
from pathlib import Path
from typing import Union
import soundfile as sf
import librosa

from backend import STTBackend
import config


class NeMoBackend(STTBackend):
    """NeMo-based STT backend for FP32 inference"""

    def __init__(self, precision: str = 'fp32', force_cpu: bool = False, num_threads: int = 0):
        """
        Initialize NeMo backend

        Args:
            precision: 'fp32' (only supported precision)
            force_cpu: Force CPU usage even if GPU is available
            num_threads: Number of threads for CPU inference (0 = auto-detect)
        """
        if precision != 'fp32':
            raise ValueError(f"NeMo backend only supports fp32, got: {precision}")

        self.precision = precision
        self._sample_rate = config.DEFAULT_SAMPLE_RATE

        # Import NeMo (lazy import to avoid dependency if using ONNX only)
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo is not installed. Install with: "
                "pip install nemo_toolkit[asr]"
            )

        # Determine device before loading model
        if force_cpu:
            self.device = torch.device('cpu')
            print(f"  Using CPU (forced)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"  Using GPU (CUDA)")
        else:
            self.device = torch.device('cpu')
            print(f"  Using CPU")

        # Configure CPU threading for optimal performance
        if self.device.type == 'cpu':
            import os
            # Auto-detect thread count if not specified
            # Following ONNX Runtime convention: use physical cores, not logical cores
            if num_threads == 0:
                # Try to get physical cores; fallback to logical cores / 2 (rough approximation)
                try:
                    import psutil
                    num_threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
                except ImportError:
                    # Approximate physical cores as half of logical cores (assumes hyperthreading)
                    num_threads = (os.cpu_count() // 2) or 4

            # Set PyTorch threading
            torch.set_num_threads(num_threads)
            # Inter-op parallelism: typically 1-2 threads is optimal
            torch.set_num_interop_threads(min(2, max(1, num_threads // 4)))

            print(f"  CPU Optimization:")
            print(f"    - Intra-op threads: {num_threads} (physical cores)")
            print(f"    - Inter-op threads: {torch.get_num_interop_threads()}")

        print(f"Loading NeMo model: {config.NEMO_MODEL_ID} ({precision})...")

        # Load model from HuggingFace - it may load to GPU by default
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.NEMO_MODEL_ID
        )

        # Move to appropriate device
        self.model = self.model.to(self.device)

        # If forcing CPU, clear any CUDA memory that might have been allocated
        if force_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

        print(f"  Precision: FP32")

        # Set to eval mode
        self.model.eval()

        print("✓ NeMo model loaded successfully!\n")

    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Transcribe audio array

        Args:
            audio: Audio array (float32, mono, 16kHz)

        Returns:
            Dictionary with transcription results
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # NeMo expects a list of audio arrays
        # Use inference_mode() for better performance than no_grad()
        with torch.inference_mode():
            transcriptions = self.model.transcribe([audio])

        # Extract text from transcription result
        if isinstance(transcriptions, list):
            result = transcriptions[0] if transcriptions else ""
            # Extract text from Hypothesis object if needed
            if hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)
        else:
            text = str(transcriptions)

        return {
            "text": text,
            "tokens": [],  # NeMo doesn't expose tokens easily
            "language": "multilingual",  # Parakeet TDT 0.6b v3 supports 25 languages
        }

    def transcribe_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe audio from file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription results
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Resample if necessary
        if sr != self._sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sample_rate)

        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return self.transcribe(audio)

    @property
    def sample_rate(self) -> int:
        """Get the expected sample rate"""
        return self._sample_rate

    @property
    def backend_name(self) -> str:
        """Get the backend name"""
        return f"NeMo ({self.precision.upper()})"
