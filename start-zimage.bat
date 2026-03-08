@echo off
chcp 65001 >nul
title Z-Image-Turbo Web UI

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

:: Quick check: venv exists?
if not exist "%VENV_DIR%" (
    echo ❌ Virtual environment not found!
    echo.
    echo Please run setup-zimage.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo ❌ Python not found in venv!
    echo.
    echo Please run setup-zimage.bat to repair the installation.
    echo.
    pause
    exit /b 1
)

echo ===========================================
echo    Z-Image-Turbo Web UI
echo ===========================================
echo.
echo    🚀 LAUNCHING APPLICATION
echo.
echo    Local:  http://localhost:7860
echo.
echo    To stop: close this window
echo.

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" launch.py --load-on-start %*

echo.
echo ===========================================
echo    Application stopped (code: %ERRORLEVEL%)
echo ===========================================
pause
