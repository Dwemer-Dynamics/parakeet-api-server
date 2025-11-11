"""
ONNX-based STT Backend using sherpa-onnx
Supports INT8 quantized Parakeet TDT 0.6B v3 (CPU-only)
"""
import numpy as np
import sherpa_onnx
from pathlib import Path
from typing import Union
import soundfile as sf
import librosa

from backend import STTBackend
import config


class ONNXBackend(STTBackend):
    """ONNX-based STT backend for quantized inference (CPU-only)"""

    def __init__(
        self,
        precision: str,
        model_dir: Path,
        num_threads: int = 0,
        force_cpu: bool = False
    ):
        """
        Initialize ONNX backend (CPU-only)

        Args:
            precision: 'int8' (only supported precision, CPU-only)
            model_dir: Path to model directory containing ONNX files
            num_threads: Number of threads for inference (0 = auto-detect)
            force_cpu: Ignored (INT8 is always CPU-only)
        """
        if precision != 'int8':
            raise ValueError(
                f"ONNX backend only supports int8, got: {precision}"
            )

        self.precision = precision
        self.model_dir = Path(model_dir).resolve()
        self._sample_rate = config.DEFAULT_SAMPLE_RATE

        # Auto-detect thread count if not specified
        # Following ONNX Runtime convention: use physical cores, not logical cores
        if num_threads == 0:
            import os
            # Try to get physical cores; fallback to logical cores / 2 (rough approximation)
            try:
                import psutil
                num_threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
            except ImportError:
                # Approximate physical cores as half of logical cores (assumes hyperthreading)
                num_threads = (os.cpu_count() // 2) or 4

        # INT8 model files
        model_files = {
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        }

        # Resolve paths
        encoder_path = str((self.model_dir / model_files["encoder"]).resolve())
        decoder_path = str((self.model_dir / model_files["decoder"]).resolve())
        joiner_path = str((self.model_dir / model_files["joiner"]).resolve())
        tokens_path = str((self.model_dir / model_files["tokens"]).resolve())

        # Verify all files exist
        required_files = [encoder_path, decoder_path, joiner_path, tokens_path]
        for path in required_files:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

        # INT8 is CPU-only - always use CPU provider
        provider = "cpu"

        print(f"Initializing ONNX backend ({precision.upper()}, CPU-only)...")
        print(f"  Encoder: {Path(encoder_path).name}")
        print(f"  Decoder: {Path(decoder_path).name}")
        print(f"  Joiner: {Path(joiner_path).name}")
        print(f"  Threads: {num_threads}")

        # Initialize recognizer using the factory method (CPU-only)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=num_threads,
            sample_rate=self._sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider=provider,
            debug=False,
        )

        print(f"  Provider: CPU (INT8 is optimized for CPU inference)")

        print("✓ ONNX model loaded successfully!\n")

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

        # Create stream
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, audio)

        # Run recognition
        self.recognizer.decode_stream(stream)
        result = stream.result

        return {
            "text": result.text,
            "tokens": result.tokens if hasattr(result, 'tokens') else [],
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
        return f"ONNX ({self.precision.upper()})"
