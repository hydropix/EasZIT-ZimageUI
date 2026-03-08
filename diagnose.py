#!/usr/bin/env python3
"""
Diagnostic script for Z-Image-Turbo Web UI
Run this script to check the installation
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
    print(f"Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")

def check_venv():
    print_section("Virtual Environment")
    
    # Check if in venv
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    print(f"In a venv: {'✅ Yes' if in_venv else '❌ No'}")
    
    if in_venv:
        print(f"  Prefix: {sys.prefix}")
        print(f"  Base Prefix: {getattr(sys, 'base_prefix', 'N/A')}")
    
    # Look for local venv
    project_dir = Path(__file__).parent
    venv_dir = project_dir / "venv"
    print(f"\nVenv in project: {venv_dir}")
    print(f"  Exists: {'✅ Yes' if venv_dir.exists() else '❌ No'}")
    
    if venv_dir.exists():
        python_exe = venv_dir / "Scripts" / "python.exe"
        print(f"  Python.exe: {'✅ Found' if python_exe.exists() else '❌ Not found'}")

def check_dependencies():
    print_section("Dependencies")
    
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
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError as e:
            print(f"  ❌ {name}: NOT INSTALLED ({e})")
            all_ok = False
    
    return all_ok

def check_cuda():
    print_section("CUDA / GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available")
            print(f"  CUDA version (PyTorch): {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"  ⚠️  CUDA not available")
            print(f"  PyTorch will use CPU (slow)")
    except Exception as e:
        print(f"  ❌ Error: {e}")

def check_files():
    print_section("Project Files")
    
    project_dir = Path(__file__).parent
    files = ["app.py", "launch.py", "requirements.txt"]
    
    for f in files:
        path = project_dir / f
        print(f"  {'✅' if path.exists() else '❌'} {f}")

def test_launch():
    print_section("Launch Test")
    
    project_dir = Path(__file__).parent
    venv_python = project_dir / "venv" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print("  ❌ Venv Python not found")
        return
    
    print("  Testing launch.py imports...")
    try:
        # Test if we can import modules from launch.py
        result = subprocess.run(
            [str(venv_python), "-c", 
             "import sys; sys.path.insert(0, '.'); " +
             "import torch; import gradio; import diffusers; " +
             "print('All imports OK')"],
            cwd=project_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ {result.stdout.strip()}")
        else:
            print(f"  ❌ Error:")
            print(f"     stdout: {result.stdout}")
            print(f"     stderr: {result.stderr}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")

def main():
    print("\n" + "="*60)
    print("  Z-Image-Turbo Web UI DIAGNOSTIC")
    print("="*60)
    
    check_python()
    check_venv()
    deps_ok = check_dependencies()
    check_cuda()
    check_files()
    test_launch()
    
    print_section("Summary")
    if deps_ok:
        print("  ✅ All dependencies are installed")
        print("  ✅ You can launch: python launch.py")
    else:
        print("  ❌ Some dependencies are missing")
        print("  📦 Run: pip install -r requirements.txt")
    
    print()
    input("Press Enter to close...")

if __name__ == "__main__":
    main()
