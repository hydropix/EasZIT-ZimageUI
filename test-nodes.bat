@echo off
chcp 65001 >nul
:: Test rapide du système de nœuds

title Z-Image-Turbo - Test Nodes

echo ========================================
echo  Z-Image-Turbo - Test du système de nœuds
echo ========================================
echo.

:: Vérifier l'environnement virtuel
if not exist "venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel non trouvé !
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [TEST 1/4] Test d'importation des nœuds...
python -c "from zimage.nodes import NodeRegistry; print('  Nodes enregistrés:', len(NodeRegistry.list_nodes()))"
if errorlevel 1 (
    echo [ECHEC] Erreur d'importation
    pause
    exit /b 1
)
echo [OK] Importation réussie
echo.

echo [TEST 2/4] Liste des nœuds disponibles...
python -c "from zimage.nodes import NodeRegistry; [print('  -', n) for n in NodeRegistry.list_nodes()]"
echo [OK]
echo.

echo [TEST 3/4] Création d'un workflow txt2img...
python -c "from zimage.workflows.builder import create_txt2img_workflow; w = create_txt2img_workflow('test', '', 512, 512, 1, 42); print('  Workflow créé avec', len(w), 'nœuds')"
if errorlevel 1 (
    echo [ECHEC] Erreur de création de workflow
    pause
    exit /b 1
)
echo [OK] Workflow créé
echo.

echo [TEST 4/4] Test du moteur d'exécution...
python -c "from zimage.engine.executor import ExecutionEngine; e = ExecutionEngine(); print('  Moteur initialisé')"
if errorlevel 1 (
    echo [ECHEC] Erreur d'initialisation du moteur
    pause
    exit /b 1
)
echo [OK] Moteur prêt
echo.

echo ========================================
echo  TOUS LES TESTS ONT RÉUSSI !
echo ========================================
echo.
echo Vous pouvez maintenant lancer l'application avec :
echo   launch-node-ui.bat
echo.
pause
