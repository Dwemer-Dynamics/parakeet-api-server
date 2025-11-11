#!/bin/bash

# Parakeet STT API Server - Startup Script (Linux/macOS)

set -e

echo "========================================"
echo "Parakeet STT API Server"
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
echo "Activating virtual environment..."
source venv/bin/activate

# Start the server with any provided arguments
echo "Starting server..."
echo ""

# Pass all command line arguments directly to server.py
python server.py "$@"
