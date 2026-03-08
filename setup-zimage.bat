@echo off
chcp 65001 >nul
title Z-Image-Turbo Web UI - Setup / Clean Update

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

echo ===========================================
echo    Z-Image-Turbo Web UI - Clean Setup
echo ===========================================
echo.
echo This script will:
echo   - Check Python installation
echo   - Create/update virtual environment
echo   - Pull latest changes from GitHub
echo   - CLEAN old/unused packages
echo   - Install/update PyTorch with CUDA
echo   - Install/update all dependencies
echo.
echo ⚠️  This may take several minutes.
echo.
pause
echo.

:: 1. Check Python
echo [1/8] Checking Python...
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
echo [2/8] Checking venv...
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
    echo    Deleting corrupted venv...
    rd /s /q "%VENV_DIR%"
    echo    Please run this script again.
    pause
    exit /b 1
)

:: 3. Update pip itself
echo.
echo [3/8] Updating pip...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1
echo    ✅ Pip updated

:: 4. Update from GitHub
echo.
echo [4/8] Updating from GitHub...
if exist "%PROJECT_DIR%\.git" (
    cd /d "%PROJECT_DIR%"
    :: Add safe directory for network drives (ignore ownership errors)
    git config --global --add safe.directory "%PROJECT_DIR%" 2>nul
    git pull 2>nul
    if errorlevel 1 (
        echo    ⚠️  Git pull skipped (network drive or permission issue)
    ) else (
        echo    ✅ Repository updated
    )
) else (
    echo    ℹ️  Not a git repository (skipping)
)

:: 5. Clean invalid residual distributions
echo.
echo [5/8] Cleaning corrupted packages...
if exist "%VENV_DIR%\Lib\site-packages\~orch*" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch" 2>nul
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch-2.10.0.dist-info" 2>nul
    echo    🧹 Corrupted distributions cleaned
)
if exist "%VENV_DIR%\Lib\site-packages\~unctorch" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~unctorch" 2>nul
    echo    🧹 Corrupted functorch cleaned
)
echo    ✅ Cleanup done

:: 6. Uninstall packages not in requirements.txt (clean environment)
echo.
echo [6/8] Synchronizing packages with requirements.txt...
echo    📋 Installing requirements...
"%PIP_EXE%" install -r "%PROJECT_DIR%requirements.txt" --upgrade --quiet
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Dependencies installation failed!
    pause
    exit /b 1
)
echo    ✅ Dependencies synced

:: 7. Check/Install PyTorch with CUDA
echo.
echo [7/8] Checking PyTorch + CUDA...
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
    "%PYTHON_EXE%" -c "import torch; print(f'PyTorch {torch.__version__} - CUDA {torch.version.cuda}')" 2>nul
    echo    ✅ PyTorch + CUDA OK
)

:: 8. Verify installation
echo.
echo [8/8] Verifying installation...
"%PYTHON_EXE%" -c "import torch, gradio, diffusers" 2>nul
if errorlevel 1 (
    echo    ⚠️  Some packages may be missing - check errors above
) else (
    echo    ✅ All core packages OK
)

:: Check GPU
echo.
"%PYTHON_EXE%" -c "import torch; cuda=torch.cuda.is_available(); print('   GPU:', torch.cuda.get_device_name(0) if cuda else 'CPU only')" 2>nul

:: Done
echo.
echo ===========================================
echo    ✅ SETUP COMPLETE
echo ===========================================
echo.
echo You can now run start-zimage.bat to launch
echo.
pause
