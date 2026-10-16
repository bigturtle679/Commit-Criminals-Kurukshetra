@echo off
REM NeuroScope 3D — Windows Launch Script
REM Usage: start-neuroscope.bat [port]

set PORT=%1
if "%PORT%"=="" set PORT=8000

echo ============================================
echo  NeuroScope 3D - Brain Tumor Segmentation
echo  Research Demonstration Only
echo ============================================
echo.

REM Activate venv if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: No .venv found. Using system Python.
)

echo Starting server on http://127.0.0.1:%PORT%
echo Press Ctrl+C to stop.
echo.

python -m uvicorn src.app:app --host 127.0.0.1 --port %PORT%
