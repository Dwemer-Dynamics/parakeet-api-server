@echo off
REM Parakeet STT API Server - Startup Script (Windows)

echo ========================================
echo Parakeet STT API Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run the installation script first:
    echo   install.bat
    echo.
    echo Looking for: venv\Scripts\python.exe
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and start server
echo Activating virtual environment...
echo Using Python: %CD%\venv\Scripts\python.exe
echo.

REM Start the server with any provided arguments
echo Starting server...
echo.

REM Use full path to venv Python
"%CD%\venv\Scripts\python.exe" server.py %*

pause
