@echo off
title NeuroScope 3D - Download Model Weights
echo ========================================================
echo  NeuroScope 3D - Automated Checkpoint Downloader
echo ========================================================
echo.
python scripts\download_weights.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Download encountered an issue. Please verify internet connection.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo [SUCCESS] Model weights are ready.
pause
