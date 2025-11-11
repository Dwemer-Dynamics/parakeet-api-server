"""
Parakeet STT API Server
FastAPI-based server with OpenAI-compatible endpoints
"""
import argparse
import io
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from backend import STTBackend, create_backend
from model_downloader import download_models


# Global model instance
stt_model: Optional[STTBackend] = None
model_info_global: dict = {}


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str


class OpenAITranscriptionResponse(BaseModel):
    """OpenAI-compatible response model"""
    text: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    precision: str
    backend: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global stt_model

    # Startup: Check if model is initialized
    if stt_model is None:
        print("ERROR: Model not initialized. This should not happen.")
        sys.exit(1)

    yield  # Server is running

    # Shutdown: cleanup if needed
    # (currently no cleanup needed, but this is where it would go)


# Initialize FastAPI app
app = FastAPI(
    title="Parakeet STT API",
    description="Speech-to-Text API using NVIDIA Parakeet TDT 0.6b-v3",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Parakeet STT API Server",
        "version": "1.0.0",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/transcribe",
            "openai_compatible": "/v1/audio/transcriptions",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=config.MODEL_DESCRIPTION,
        precision=model_info_global.get('precision', 'unknown'),
        backend=stt_model.backend_name if stt_model else 'not_loaded',
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    timestamps: bool = Form(False),
):
    """
    Transcribe audio file

    Args:
        file: Audio file (wav, mp3, flac, ogg, etc.)
        language: Language code (not used - model auto-detects)
        timestamps: Return word-level timestamps

    Returns:
        Transcription result
    """
    if stt_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Load audio
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != config.DEFAULT_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.DEFAULT_SAMPLE_RATE)

        # Transcribe
        if timestamps:
            result = stt_model.transcribe_with_timestamps(audio)
        else:
            result = stt_model.transcribe(audio)

        return TranscriptionResponse(text=result["text"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/transcriptions", response_model=OpenAITranscriptionResponse)
async def openai_transcribe(
    file: UploadFile = File(...),
    model: str = Form("parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """
    OpenAI-compatible transcription endpoint

    Compatible with OpenAI's audio transcription API.

    Args:
        file: Audio file
        model: Model to use (ignored - uses Parakeet)
        language: Language code (not used - model auto-detects)
        prompt: Optional prompt (not used)
        response_format: Response format (json, text, srt, vtt)
        temperature: Temperature (not used)

    Returns:
        Transcription in requested format
    """
    if stt_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Load audio
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != config.DEFAULT_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.DEFAULT_SAMPLE_RATE)

        # Transcribe
        if response_format in ["srt", "vtt"]:
            result = stt_model.transcribe_with_timestamps(audio)
        else:
            result = stt_model.transcribe(audio)

        text = result["text"]

        # Format response based on response_format
        if response_format == "text":
            return text
        elif response_format == "json":
            return OpenAITranscriptionResponse(text=text)
        elif response_format == "srt":
            # Simple SRT format (without precise timestamps)
            srt = f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"
            return srt
        elif response_format == "vtt":
            # Simple VTT format (without precise timestamps)
            vtt = f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"
            return vtt
        else:
            return OpenAITranscriptionResponse(text=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def initialize_model(
    precision: str = config.DEFAULT_PRECISION,
    num_threads: int = 4,
    auto_download: bool = True,
    force_cpu: bool = False
):
    """
    Initialize the STT model

    Args:
        precision: Precision level ('fp32', 'int8')
        num_threads: Number of threads for inference (ONNX only)
        auto_download: Auto-download models if not present
        force_cpu: Force CPU usage even if GPU is available
    """
    global stt_model, model_info_global

    # Validate precision
    if precision not in config.AVAILABLE_PRECISION:
        print(f"ERROR: Invalid precision: {precision}")
        print(f"Available precision levels: {', '.join(config.AVAILABLE_PRECISION)}")
        sys.exit(1)

    # Store model info globally
    model_info_global['precision'] = precision

    # Check if ONNX models need to be downloaded (only INT8)
    model_dir = None
    if precision == 'int8':
        model_dir_name = config.ONNX_MODEL_DIRS.get(precision)
        if model_dir_name:
            model_dir = config.MODELS_DIR / model_dir_name

            # Check if models exist
            required_files = ['encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt']

            all_exist = model_dir.exists() and all((model_dir / f).exists() for f in required_files)

            if not all_exist:
                if auto_download:
                    print(f"Models not found. Downloading {precision} models...")
                    download_models(precision)
                else:
                    print(f"ERROR: Models not found at {model_dir}")
                    print("Run with --download flag to download models automatically")
                    sys.exit(1)

    # Initialize backend
    print(f"\nInitializing {config.MODEL_DESCRIPTION} ({precision.upper()})...")
    stt_model = create_backend(
        precision=precision,
        model_dir=model_dir,
        num_threads=num_threads,
        force_cpu=force_cpu
    )
    print("✓ Server ready!\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Parakeet STT API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings (FP32)
  python server.py

  # Start server with FP32 precision (best quality, GPU/CPU, NeMo backend)
  python server.py --precision fp32

  # Start server with INT8 (CPU-only, ONNX backend)
  python server.py --precision int8

  # Start server on custom host/port
  python server.py --host 0.0.0.0 --port 9000

  # Start server without auto-download
  python server.py --no-download

  # Force CPU usage for FP32 (even with GPU available)
  python server.py --precision fp32 --cpu
        """
    )

    parser.add_argument(
        "--precision",
        "-p",
        choices=config.AVAILABLE_PRECISION,
        default=config.DEFAULT_PRECISION,
        help=f"Precision level (default: {config.DEFAULT_PRECISION})"
    )
    parser.add_argument(
        "--host",
        default=config.API_HOST,
        help=f"Host to bind to (default: {config.API_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help=f"Port to bind to (default: {config.API_PORT})"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=config.DEFAULT_NUM_THREADS,
        help=f"Number of threads for CPU inference, 0=auto-detect (default: {config.DEFAULT_NUM_THREADS})"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't auto-download models (fail if not present)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )

    args = parser.parse_args()

    print("="*70)
    print("Parakeet STT API Server")
    print("="*70)
    print(f"Model: {config.MODEL_DESCRIPTION}")
    print(f"Size: {config.MODEL_SIZE}")
    print(f"Languages: {config.MODEL_LANGUAGES}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    if args.threads == 0:
        import os
        # Show physical cores (approximation)
        try:
            import psutil
            detected_threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
        except ImportError:
            detected_threads = (os.cpu_count() // 2) or 4
        print(f"CPU Threads: {detected_threads} physical cores (auto-detected)")
    else:
        print(f"CPU Threads: {args.threads}")
    if args.cpu:
        print(f"Mode: CPU (forced)")
    print("="*70)
    print()

    # Initialize model
    initialize_model(
        precision=args.precision,
        num_threads=args.threads,
        auto_download=not args.no_download,
        force_cpu=args.cpu
    )

    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1/audio/transcriptions")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
