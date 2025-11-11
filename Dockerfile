# Parakeet STT API Server - Docker Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY model_downloader.py .
COPY backend.py .
COPY inference_nemo.py .
COPY inference_onnx.py .
COPY server.py .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8022

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8022/health')"

# Run server
ENTRYPOINT ["python3", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8022"]
