#!/usr/bin/env python3
"""
Script de diagnostic pour Z-Image-Turbo Web UI
Exécute ce script pour vérifier l'installation
"""

import sys
import os
import subprocess
from pathlib import Path

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python():
    print_section("Python")
    print(f"Version: {sys.version}")
    print(f"Exécutable: {sys.executable}")
    print(f"Platform: {sys.platform}")

def check_venv():
    print_section("Environnement Virtuel")
    
    # Vérifier si on est dans un venv
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    print(f"Dans un venv: {'✅ Oui' if in_venv else '❌ Non'}")
    
    if in_venv:
        print(f"  Prefix: {sys.prefix}")
        print(f"  Base Prefix: {getattr(sys, 'base_prefix', 'N/A')}")
    
    # Chercher le venv local
    project_dir = Path(__file__).parent
    venv_dir = project_dir / "venv"
    print(f"\nVenv dans le projet: {venv_dir}")
    print(f"  Existe: {'✅ Oui' if venv_dir.exists() else '❌ Non'}")
    
    if venv_dir.exists():
        python_exe = venv_dir / "Scripts" / "python.exe"
        print(f"  Python.exe: {'✅ Trouvé' if python_exe.exists() else '❌ Introuvable'}")

def check_dependencies():
    print_section("Dépendances")
    
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("gradio", "Gradio"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for module, name in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'inconnue')
            print(f"  ✅ {name}: {version}")
        except ImportError as e:
            print(f"  ❌ {name}: NON INSTALLÉ ({e})")
            all_ok = False
    
    return all_ok

def check_cuda():
    print_section("CUDA / GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponible")
            print(f"  Version CUDA (PyTorch): {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"  ⚠️  CUDA non disponible")
            print(f"  PyTorch utilisera le CPU (lent)")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")

def check_files():
    print_section("Fichiers du projet")
    
    project_dir = Path(__file__).parent
    files = ["app.py", "launch.py", "requirements.txt"]
    
    for f in files:
        path = project_dir / f
        print(f"  {'✅' if path.exists() else '❌'} {f}")

def test_launch():
    print_section("Test de lancement")
    
    project_dir = Path(__file__).parent
    venv_python = project_dir / "venv" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print("  ❌ Python du venv introuvable")
        return
    
    print("  Test d'import de launch.py...")
    try:
        # Tester si on peut importer les modules de launch.py
        result = subprocess.run(
            [str(venv_python), "-c", 
             "import sys; sys.path.insert(0, '.'); " +
             "import torch; import gradio; import diffusers; " +
             "print('Tous les imports OK')"],
            cwd=project_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ {result.stdout.strip()}")
        else:
            print(f"  ❌ Erreur:")
            print(f"     stdout: {result.stdout}")
            print(f"     stderr: {result.stderr}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")

def main():
    print("\n" + "="*60)
    print("  DIAGNOSTIC Z-Image-Turbo Web UI")
    print("="*60)
    
    check_python()
    check_venv()
    deps_ok = check_dependencies()
    check_cuda()
    check_files()
    test_launch()
    
    print_section("Résumé")
    if deps_ok:
        print("  ✅ Toutes les dépendances sont installées")
        print("  ✅ Vous pouvez lancer: python launch.py")
    else:
        print("  ❌ Certaines dépendances sont manquantes")
        print("  📦 Exécutez: pip install -r requirements.txt")
    
    print()
    input("Appuyez sur Entrée pour fermer...")

if __name__ == "__main__":
    main()
