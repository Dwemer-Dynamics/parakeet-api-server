@echo off
setlocal EnableDelayedExpansion
REM Parakeet STT API Server - Installation Script (Windows)

echo ========================================
echo Parakeet STT API Server - Installation
echo ========================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

set PYTHON_CMD=python

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [+] Found Python %PYTHON_VERSION%

REM Detect Python 3.9 for special handling
echo %PYTHON_VERSION% | findstr /C:"3.9" >nul
if not errorlevel 1 (
    set USE_PY39_REQUIREMENTS=1
    echo [i] Detected Python 3.9 - will use compatible package versions
) else (
    set USE_PY39_REQUIREMENTS=0
)

REM Check if NVIDIA GPU is available
set GPU_DETECTED=0
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [!] nvidia-smi not found. No NVIDIA GPU detected.
    goto :skip_gpu_detect
)

echo [+] NVIDIA GPU detected
set GPU_DETECTED=1
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
    echo     %%g
    goto :skip_gpu_detect
)

:skip_gpu_detect

echo.

REM Prompt user for GPU support
set INSTALL_GPU=0
if !GPU_DETECTED!==1 (
    echo Do you want to install PyTorch with CUDA support for GPU acceleration?
    echo This is recommended for 5-10x faster inference.
    set /p GPU_CHOICE="Install with GPU support? [Y/n]: "

    if /i "!GPU_CHOICE!"=="" set INSTALL_GPU=1
    if /i "!GPU_CHOICE!"=="y" set INSTALL_GPU=1
    if /i "!GPU_CHOICE!"=="yes" set INSTALL_GPU=1

    if !INSTALL_GPU!==1 (
        echo [i] Installing GPU ^(CUDA 12.6^) version
    ) else (
        echo [i] Installing CPU-only version
    )
) else (
    echo No GPU detected. Installing CPU-only version.
)

echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    echo [+] Virtual environment created
) else (
    echo [+] Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip >nul 2>&1

REM Clear pip cache for clean install
echo Clearing pip cache...
pip cache purge >nul 2>&1

echo.
echo ========================================
echo Installing PyTorch...
echo ========================================
echo.

REM Install PyTorch with or without CUDA
if !INSTALL_GPU!==1 (
    echo Installing PyTorch with CUDA 12.6 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else (
    echo Installing PyTorch ^(CPU-only^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo ========================================
echo Installing sherpa-onnx...
echo ========================================
echo.

REM Install sherpa-onnx with GPU support if needed
if !INSTALL_GPU!==1 (
    echo Installing sherpa-onnx with CUDA 12.x support...
    pip install sherpa-onnx==1.12.13+cuda12.cudnn9 -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
    echo.
    echo Installing CUDNN 9 ^(required for sherpa-onnx GPU support^)...
    pip install nvidia-cudnn-cu12
) else (
    echo Installing sherpa-onnx ^(CPU-only^)...
    pip install "sherpa-onnx>=1.10.0"
)

echo.
echo ========================================
echo Installing other dependencies...
echo ========================================
echo.

REM Install dependencies with appropriate requirements file (sherpa-onnx and CUDNN already installed above)
if "!USE_PY39_REQUIREMENTS!"=="1" (
    echo [i] Using Python 3.9 compatible requirements with pre-built wheels
    if exist "requirements-windows-py39.txt" (
        pip install --no-cache-dir -r requirements-windows-py39.txt
    ) else (
        echo WARNING: requirements-windows-py39.txt not found, using default
        pip install --no-cache-dir -r requirements.txt
    )
) else (
    pip install --no-cache-dir -r requirements.txt
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.

REM Verify GPU support if installed
if !INSTALL_GPU!==1 (
    echo Verifying GPU support...
    python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'[+] CUDA available: {cuda_available}'); print(f'[+] CUDA version: {torch.version.cuda if cuda_available else \"N/A\"}'); exit(0 if cuda_available else 1)" && (echo.) || (echo [!] Warning: CUDA not available. GPU may not be properly configured. && echo.)
)

echo To start the server, run:
echo   start.bat
echo.
echo Or activate the virtual environment and run manually:
echo   venv\Scripts\activate.bat
echo   python server.py
echo.

pause
