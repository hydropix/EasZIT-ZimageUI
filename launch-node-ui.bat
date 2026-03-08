@echo off
chcp 65001 >nul
:: Z-Image-Turbo Node-Based UI Launcher
:: Lance l'interface avec le système de nœuds

title Z-Image-Turbo Node UI

echo ========================================
echo  Z-Image-Turbo Node-Based UI
echo ========================================
echo.

:: Vérifier si on est dans le bon dossier
if not exist "app.py" (
    echo [ERREUR] app.py non trouvé !
    echo Veuillez lancer ce script depuis le dossier Z-Image-Turbo.
    pause
    exit /b 1
)

:: Vérifier l'environnement virtuel
if not exist "venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel non trouvé !
    echo Veuillez d'abord exécuter setup-zimage.bat
    pause
    exit /b 1
)

echo [1/3] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)

echo [2/3] Chargement des nœuds...
echo.

:: Afficher les options
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

:: Configuration par défaut
set HOST=0.0.0.0
set PORT=7860
set EXTRA_ARGS=

:: Parser les arguments
:parse_args
if "%~1"=="" goto :launch
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--local" (
    set HOST=127.0.0.1
    shift
    goto :parse_args
)
if "%~1"=="--share" (
    set EXTRA_ARGS=%EXTRA_ARGS% --share
    shift
    goto :parse_args
)
if "%~1"=="--no-custom-nodes" (
    set EXTRA_ARGS=%EXTRA_ARGS% --no-custom-nodes
    shift
    goto :parse_args
)
if "%~1"=="--auth" (
    set EXTRA_ARGS=%EXTRA_ARGS% --auth %~2
    shift
    shift
    goto :parse_args
)
set EXTRA_ARGS=%EXTRA_ARGS% %~1
shift
goto :parse_args

:launch
echo [3/3] Démarrage du serveur...
echo.
echo Configuration :
echo   Host : %HOST%
echo   Port : %PORT%
echo   Args : %EXTRA_ARGS%
echo.
echo ========================================
echo.

python app.py --host %HOST% --port %PORT% %EXTRA_ARGS%

if errorlevel 1 (
    echo.
    echo [ERREUR] L'application s'est arrêtée avec une erreur.
    pause
)

goto :eof

:help
echo Usage: launch-node-ui.bat [OPTIONS]
echo.
echo Options:
echo   --port PORT           Port du serveur (défaut: 7860)
echo   --host HOST           Adresse d'écoute (défaut: 0.0.0.0)
echo   --local               N'écouter que sur localhost (127.0.0.1)
echo   --share               Créer un lien public temporaire
echo   --no-custom-nodes     Désactiver le chargement des nœuds personnalisés
echo   --auth USER:PASS      Activer l'authentification
echo   -h, --help            Afficher cette aide
echo.
echo Exemples:
echo   launch-node-ui.bat                    ^(Lancement standard^)
echo   launch-node-ui.bat --port 8080        ^(Port 8080^)
echo   launch-node-ui.bat --local            ^(Local uniquement^)
echo   launch-node-ui.bat --share            ^(Lien public^)
echo.
pause
