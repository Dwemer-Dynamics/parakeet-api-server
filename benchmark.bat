@echo off
REM Benchmark launcher script for Parakeet STT API (Windows)

echo ========================================
echo Parakeet STT Benchmark Tool
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run the installation script first:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

REM Get dataset path
if "%~1"=="" (
    echo Usage: benchmark.bat ^<dataset_folder^> [iterations] [output_csv] [device] [debug_log]
    echo.
    echo Examples:
    echo   benchmark.bat .\test_data 3 results.csv
    echo   benchmark.bat .\test_data 3 results.csv both errors.txt
    echo   benchmark.bat .\test_data 1 results.csv gpu
    echo.
    echo Arguments:
    echo   dataset_folder  - Path to folder containing .wav and .txt files ^(required^)
    echo   iterations      - Number of test iterations for averaging ^(default: 1^)
    echo   output_csv      - Output CSV filename ^(default: benchmark_results.csv^)
    echo   device          - Device to test: cpu, gpu, or both ^(default: auto^)
    echo   debug_log       - Optional debug log file for transcription errors
    echo.
    pause
    exit /b 1
)

set DATASET=%~1
set ITERATIONS=%~2
set OUTPUT=%~3
set DEVICE=%~4
set DEBUG_LOG=%~5

REM Set defaults
if "%ITERATIONS%"=="" set ITERATIONS=1
if "%OUTPUT%"=="" set OUTPUT=benchmark_results.csv

echo Dataset: %DATASET%
echo Iterations: %ITERATIONS%
echo Output: %OUTPUT%
if not "%DEVICE%"=="" echo Device: %DEVICE%
if not "%DEBUG_LOG%"=="" echo Debug Log: %DEBUG_LOG%
echo.

REM Build command
set CMD="%CD%\venv\Scripts\python.exe" benchmark.py --dataset "%DATASET%" --iterations %ITERATIONS% --output "%OUTPUT%"
if not "%DEVICE%"=="" set CMD=%CMD% --device "%DEVICE%"
if not "%DEBUG_LOG%"=="" set CMD=%CMD% --debug-log "%DEBUG_LOG%"

REM Run benchmark
%CMD%

echo.
echo ========================================
echo Benchmark Complete!
echo Results saved to: %OUTPUT%
echo ========================================
echo.

pause
