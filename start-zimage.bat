@echo off
chcp 65001 >nul
title Z-Image-Turbo Web UI

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

echo ===========================================
echo    Z-Image-Turbo Web UI - Launcher
echo ===========================================
echo.

:: 1. Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Python not found!
    echo    Install Python 3.9-3.12 from https://python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do echo    ✅ %%a

:: 2. Create venv if needed
echo.
echo [2/6] Checking venv...
if not exist "%VENV_DIR%" (
    echo    Creating venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Unable to create venv
        pause
        exit /b 1
    )
    echo    ✅ Venv created
) else (
    echo    ✅ Venv exists
)

:: Check that python.exe exists in venv
if not exist "%PYTHON_EXE%" (
    echo.
    echo ❌ ERROR: Python not found in venv
    pause
    exit /b 1
)

:: 3. Clean invalid residual distributions
echo.
echo [3/6] Cleaning invalid distributions...
if exist "%VENV_DIR%\Lib\site-packages\~orch*" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch" 2>nul
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch-2.10.0.dist-info" 2>nul
    echo    🧹 Corrupted distributions cleaned
) else (
    echo    ✅ No invalid distributions
)
if exist "%VENV_DIR%\Lib\site-packages\~unctorch" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~unctorch" 2>nul
    echo    🧹 Corrupted functorch cleaned
)

:: 4. Check/Install PyTorch with CUDA
echo.
echo [4/6] Checking PyTorch + CUDA...
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo    📦 Installing PyTorch 2.6 + CUDA 12.4...
    echo    (downloading ~2.5 GB, this may take several minutes)
    echo.
    "%PIP_EXE%" install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: PyTorch installation failed!
        pause
        exit /b 1
    )
    echo    ✅ PyTorch + CUDA installed
) else (
    for /f "tokens=*" %%a in ('"%PYTHON_EXE%" -c "import torch; print(f\'PyTorch {torch.__version__} - CUDA {torch.version.cuda}\')" 2^>^&1') do echo    ✅ %%a
)

:: 5. Check/Install other dependencies
echo.
echo [5/6] Checking other dependencies...
"%PYTHON_EXE%" -c "import gradio, diffusers, PIL, transformers" 2>nul
if errorlevel 1 (
    echo    📦 Installing dependencies...
    "%PIP_EXE%" install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Dependencies installation failed!
        pause
        exit /b 1
    )
    echo    ✅ Dependencies installed
) else (
    echo    ✅ Dependencies OK
)

:: 6. Check GPU
echo.
echo [6/6] Checking GPU...
"%PYTHON_EXE%" -c "import torch; cuda=torch.cuda.is_available(); print('   GPU:', torch.cuda.get_device_name(0) if cuda else 'CPU only')" 2>nul

:: Launch
echo.
echo ===========================================
echo    🚀 LAUNCHING APPLICATION
echo ===========================================
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
