# Parakeet STT API Server

A high-performance Speech-to-Text API server using NVIDIA's Parakeet TDT 0.6B v3 model with CPU and ONNX int8 support.

## Features

- **Flexible Precision**: FP32 (Best quality, GPU/CPU) or INT8 (Slightly worse quality, CPU-only)
- **Easy Setup**: Clone and run with a single command
- **Auto-Download**: Models download automatically on first run
- **OpenAI Compatible**: Drop-in replacement for OpenAI's Whisper API
- **Multilingual**: Supports 25 European languages (auto-detected)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Dolyfin/parakeet-api.git
cd parakeet-api

# Run installation script (auto-detects GPU)
# Linux/macOS:
./install.sh

# Windows:
install.bat
```

### Starting the Server

```bash
# Linux/macOS
./start.sh

# Windows
start.bat

# Or run directly with options
python server.py --precision fp32  # GPU/CPU, best quality (default)
python server.py --precision int8  # CPU-only, fastest
```

## Usage

### Basic Transcription

```bash
curl -X POST "http://localhost:8022/v1/transcribe" \
  -F "file=@audio.wav"
```

### OpenAI-Compatible Endpoint

```bash
curl -X POST "http://localhost:8022/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=parakeet-tdt-0.6b-v3"
```

### Python Client

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8022/v1/audio/transcriptions",
        files={"file": f},
        data={"model": "parakeet-tdt-0.6b-v3"}
    )

print(response.json()["text"])
```

### OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8022/v1",
    api_key="not-needed"
)

with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=audio_file
    )

print(transcript.text)
```

## API Endpoints

- **POST /v1/transcribe** - Standard transcription endpoint
- **POST /v1/audio/transcriptions** - OpenAI-compatible endpoint
- **GET /health** - Health check
- **GET /docs** - Interactive API documentation

### Response Formats

Supports `json`, `text`, `srt`, and `vtt` formats. Add timestamps with `timestamps=true`.

## Command Line Options

```bash
# Precision selection
python server.py --precision fp32   # Best quality, GPU/CPU (default)
python server.py --precision int8   # Fastest, CPU-only

# Server configuration
python server.py --host 0.0.0.0 --port 9000

# CPU threading (applies to both FP32 and INT8)
python server.py --threads 8        # Use 8 threads
python server.py --threads 0        # Auto-detect (default, uses all CPU cores)

# Force CPU for FP32 with optimized threading
python server.py --precision fp32 --cpu --threads 0
```

## GPU Setup

The install script auto-detects your GPU. To check GPU support manually:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If GPU is not detected, reinstall PyTorch with CUDA:

```bash
# For CUDA 12.6 (RTX 3000/4000 series and newer)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Model Information

**NVIDIA Parakeet TDT 0.6B v3**
- Size: 600M parameters
- Languages: 25 European languages (auto-detected)
- Architecture: FastConformer-TDT

| Precision | Memory | Backend | Device Support |
|-----------|--------|---------|----------------|
| FP32      | ~2.5GB | NeMo (PyTorch) | GPU/CPU |
| INT8      | ~640MB | ONNX | CPU-only |

## Benchmarking

Test Word Error Rate (WER) and performance:

```bash
# Linux/macOS
./benchmark.sh ./test_data 3 results.csv

# Windows
benchmark.bat .\test_data 3 results.csv

# Manual
python benchmark.py --dataset ./test_data --iterations 3
```

Dataset format: Pairs of audio files and `.txt` transcriptions in the same folder.

### Benchmark Results

Tested on RTX 3060 and i5-11600K on a 2.4hr dataset:

| Precision | Device | WER (%) | RTF | Time (s) | VRAM (MB) | Delta from FP32 GPU |
|-----------|--------|---------|-----|----------|-----------|---------------------|
| FP32      | GPU    | 11.80   | 0.01 | 77.3    | 2434      | -                   |
| FP32      | CPU    | 11.79   | 0.10 | 779.1   | -         | -                   |
| INT8      | CPU    | 15.49   | 0.04 | 362.1   | -         | +3.69%              |

*Note: Custsom test dataset was not perfect and could inflate WER values. Results show relative performance differences between configurations.*
*FP32 CPU can benefit from higher `--threads` than default by 10-20% which isn't tested.*

## Troubleshooting

### GPU Not Being Used

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA support (see GPU Setup above)

### Out of Memory

- Use INT8: `python server.py --precision int8`
- Force CPU for FP32: `python server.py --precision fp32 --cpu`

### Windows: scipy Build Error (Python 3.9)

Upgrade to Python 3.10+ or install pre-built wheels:
```batch
pip install numpy==1.23.5 scipy==1.11.4
```

## Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A and more. Audio is automatically resampled to 16kHz and converted to mono.

## Docker

*not tested

```bash
# Build
docker build -t parakeet-api .

# Run with GPU
docker-compose up

# Run CPU-only
docker run -p 8022:8022 parakeet-api --precision int8
```

## Project Structure

```
parakeet-api/
├── server.py              # FastAPI server
├── backend.py             # Backend abstraction layer
├── inference_nemo.py      # NeMo backend (FP32)
├── inference_onnx.py      # ONNX backend (INT8)
├── model_downloader.py    # Model auto-download
├── benchmark.py           # WER & performance testing
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── install.sh/bat         # Installation scripts
├── start.sh/bat           # Startup scripts
└── benchmark.sh/bat       # Benchmark launchers
```

## License

This project uses the NVIDIA Parakeet model licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
You agree to the license by downloading the model.

This project is licensed under GNU General Public License v3.0.

## Credits

- **Model**: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **ONNX Runtime**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review API docs at `/docs`
3. Open an issue on GitHub
