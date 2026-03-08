#!/usr/bin/env python3
"""
Launch script for Z-Image-Turbo Web UI
Launches the application with optimal parameters for remote access
"""

import os
import sys
import argparse
import subprocess
import socket


def get_local_ip():
    """Gets the local IP to display the access URL"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def check_cuda():
    """Checks if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️ CUDA not available - CPU mode (slow)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Launch Z-Image-Turbo Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python launch.py                    # Basic launch
  python launch.py --load-on-start    # Load model at startup
  python launch.py --share            # Create temporary public link
  python launch.py --port 8080        # Use port 8080
  python launch.py --auth admin:pass  # Enable authentication
        """
    )
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to listen on (default: 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to use (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link (72h max)")
    parser.add_argument("--load-on-start", action="store_true",
                        help="Load model automatically at startup")
    parser.add_argument("--auth", type=str,
                        help="Basic authentication 'user:password'")
    parser.add_argument("--no-autolaunch", action="store_true",
                        help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           ⚡ Z-Image-Turbo Web UI Launcher ⚡                ║
║                                                              ║
║   Web interface for image generation with Z-Image            ║
║   Model: Tongyi-MAI/Z-Image-Turbo                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Checks
    print("🔍 Preliminary checks...")
    print("-" * 50)
    
    # Check Python
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio not installed. Run: pip install -r requirements.txt")
        return 1
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("❌ Diffusers not installed. Run: pip install -r requirements.txt")
        return 1
    
    # Build command
    cmd = [sys.executable, "app.py"]
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])
    
    if args.share:
        cmd.append("--share")
    
    if args.load_on_start:
        cmd.append("--load-on-start")
    
    if args.auth:
        cmd.extend(["--auth", args.auth])
    
    # Display access info
    print("\n" + "=" * 50)
    print("🌐 Access Information:")
    print("=" * 50)
    
    local_ip = get_local_ip()
    print(f"📍 Local:    http://localhost:{args.port}")
    print(f"🌐 Network:  http://{local_ip}:{args.port}")
    
    if args.host == "0.0.0.0":
        print(f"\n💡 Application is accessible from the local network!")
        print(f"   Use address: http://{local_ip}:{args.port}")
    
    if args.share:
        print(f"\n🌍 A public link will be created (valid 72h)")
    
    if args.auth:
        print(f"\n🔒 Authentication enabled: {args.auth}")
    
    print("\n⚠️  To stop: press CTRL+C")
    print("=" * 50 + "\n")
    
    # Launch application
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Stop requested. Goodbye!")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
