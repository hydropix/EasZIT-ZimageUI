@echo off
:: Lancement rapide sans affichage détaillé

if not exist "venv\Scripts\activate.bat" (
    echo Environnement virtuel non trouvé. Lancement de setup...
    call setup-zimage.bat
)

call venv\Scripts\activate.bat >nul 2>&1
python app.py --load-on-start %*
