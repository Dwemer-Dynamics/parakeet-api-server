#!/bin/bash

# Benchmark launcher script for Parakeet STT API (Linux/macOS)

set -e

echo "========================================"
echo "Parakeet STT Benchmark Tool"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run the installation script first:"
    echo "  ./install.sh"
    echo ""
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Get dataset path
if [ -z "$1" ]; then
    echo "Usage: ./benchmark.sh <dataset_folder> [iterations] [output_csv] [device] [debug_log]"
    echo ""
    echo "Examples:"
    echo "  ./benchmark.sh ./test_data 3 results.csv"
    echo "  ./benchmark.sh ./test_data 3 results.csv both errors.txt"
    echo "  ./benchmark.sh ./test_data 1 results.csv gpu"
    echo ""
    echo "Arguments:"
    echo "  dataset_folder  - Path to folder containing .wav and .txt files (required)"
    echo "  iterations      - Number of test iterations for averaging (default: 1)"
    echo "  output_csv      - Output CSV filename (default: benchmark_results.csv)"
    echo "  device          - Device to test: cpu, gpu, or both (default: auto)"
    echo "  debug_log       - Optional debug log file for transcription errors"
    echo ""
    exit 1
fi

DATASET=$1
ITERATIONS=${2:-1}
OUTPUT=${3:-benchmark_results.csv}
DEVICE=${4:-}
DEBUG_LOG=${5:-}

echo "Dataset: $DATASET"
echo "Iterations: $ITERATIONS"
echo "Output: $OUTPUT"
[ -n "$DEVICE" ] && echo "Device: $DEVICE"
[ -n "$DEBUG_LOG" ] && echo "Debug Log: $DEBUG_LOG"
echo ""

# Build command
CMD="python benchmark.py --dataset \"$DATASET\" --iterations \"$ITERATIONS\" --output \"$OUTPUT\""
[ -n "$DEVICE" ] && CMD="$CMD --device \"$DEVICE\""
[ -n "$DEBUG_LOG" ] && CMD="$CMD --debug-log \"$DEBUG_LOG\""

# Run benchmark
eval $CMD

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "Results saved to: $OUTPUT"
echo "========================================"
