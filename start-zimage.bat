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

:: 1. Verifier Python
echo [1/6] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ ERREUR: Python non trouve!
    echo    Installez Python 3.9-3.12 depuis https://python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do echo    ✅ %%a

:: 2. Creer le venv si besoin
echo.
echo [2/6] Verification du venv...
if not exist "%VENV_DIR%" (
    echo    Creation du venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ❌ ERREUR: Impossible de creer le venv
        pause
        exit /b 1
    )
    echo    ✅ Venv cree
) else (
    echo    ✅ Venv existe
)

:: Verifier que python.exe existe dans le venv
if not exist "%PYTHON_EXE%" (
    echo.
    echo ❌ ERREUR: Python introuvable dans le venv
    pause
    exit /b 1
)

:: 3. Nettoyer les distributions invalides residuelles
echo.
echo [3/6] Nettoyage des distributions invalides...
if exist "%VENV_DIR%\Lib\site-packages\~orch*" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch" 2>nul
    rd /s /q "%VENV_DIR%\Lib\site-packages\~orch-2.10.0.dist-info" 2>nul
    echo    🧹 Distributions corrompues nettoyees
) else (
    echo    ✅ Pas de distributions invalides
)
if exist "%VENV_DIR%\Lib\site-packages\~unctorch" (
    rd /s /q "%VENV_DIR%\Lib\site-packages\~unctorch" 2>nul
    echo    🧹 Functorch corrompu nettoye
)

:: 4. Verifier/Installer PyTorch avec CUDA
echo.
echo [4/6] Verification de PyTorch + CUDA...
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo    📦 Installation de PyTorch 2.6 + CUDA 12.4...
    echo    (telechargement ~2.5 GB, cela peut prendre plusieurs minutes)
    echo.
    "%PIP_EXE%" install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo.
        echo ❌ ERREUR: Installation de PyTorch echouee!
        pause
        exit /b 1
    )
    echo    ✅ PyTorch + CUDA installes
) else (
    for /f "tokens=*" %%a in ('"%PYTHON_EXE%" -c "import torch; print(f\'PyTorch {torch.__version__} - CUDA {torch.version.cuda}\')" 2^>^&1') do echo    ✅ %%a
)

:: 5. Verifier/Installer les autres dependances
echo.
echo [5/6] Verification des autres dependances...
"%PYTHON_EXE%" -c "import gradio, diffusers, PIL, transformers" 2>nul
if errorlevel 1 (
    echo    📦 Installation des dependances...
    "%PIP_EXE%" install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 (
        echo.
        echo ❌ ERREUR: Installation des dependances echouee!
        pause
        exit /b 1
    )
    echo    ✅ Dependances installees
) else (
    echo    ✅ Dependances OK
)

:: 6. Verifier CUDA
echo.
echo [6/6] Verification GPU...
"%PYTHON_EXE%" -c "import torch; cuda=torch.cuda.is_available(); print('   GPU:', torch.cuda.get_device_name(0) if cuda else 'CPU seulement')" 2>nul

:: Lancer
echo.
echo ===========================================
echo    🚀 LANCEMENT DE L'APPLICATION
echo ===========================================
echo.
echo    Local:  http://localhost:7860
echo.
echo    Pour arreter: fermez cette fenetre
echo.

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" launch.py --load-on-start %*

echo.
echo ===========================================
echo    Application arretee (code: %ERRORLEVEL%)
echo ===========================================
pause
