#!/usr/bin/env python3
"""
Script de lancement pour Z-Image-Turbo Web UI
Lance l'application avec les paramètres optimaux pour accès distant
"""

import os
import sys
import argparse
import subprocess
import socket


def get_local_ip():
    """Récupère l'IP locale pour afficher l'URL d'accès"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def check_cuda():
    """Vérifie si CUDA est disponible"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️ CUDA non disponible - Mode CPU (lent)")
            return False
    except ImportError:
        print("❌ PyTorch non installé")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Lance Z-Image-Turbo Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python launch.py                    # Lancement basique
  python launch.py --load-on-start    # Charge le modèle au démarrage
  python launch.py --share            # Crée un lien public temporaire
  python launch.py --port 8080        # Utilise le port 8080
  python launch.py --auth admin:pass  # Active l'authentification
        """
    )
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host à écouter (default: 0.0.0.0 pour accès réseau)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port à utiliser (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Créer un lien public Gradio (72h max)")
    parser.add_argument("--load-on-start", action="store_true",
                        help="Charger le modèle automatiquement au démarrage")
    parser.add_argument("--auth", type=str,
                        help="Authentification basique 'user:password'")
    parser.add_argument("--no-autolaunch", action="store_true",
                        help="Ne pas ouvrir le navigateur automatiquement")
    
    args = parser.parse_args()
    
    # Bannière
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           ⚡ Z-Image-Turbo Web UI Launcher ⚡                ║
║                                                              ║
║   Interface web pour génération d'images avec Z-Image        ║
║   Modèle: Tongyi-MAI/Z-Image-Turbo                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Vérifications
    print("🔍 Vérifications préliminaires...")
    print("-" * 50)
    
    # Vérifier Python
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Vérifier CUDA
    has_cuda = check_cuda()
    
    # Vérifier les dépendances
    print("\n📦 Vérification des dépendances...")
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio non installé. Exécutez: pip install -r requirements.txt")
        return 1
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("❌ Diffusers non installé. Exécutez: pip install -r requirements.txt")
        return 1
    
    # Construire la commande
    cmd = [sys.executable, "app.py"]
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])
    
    if args.share:
        cmd.append("--share")
    
    if args.load_on_start:
        cmd.append("--load-on-start")
    
    if args.auth:
        cmd.extend(["--auth", args.auth])
    
    # Afficher les infos d'accès
    print("\n" + "=" * 50)
    print("🌐 Informations d'accès:")
    print("=" * 50)
    
    local_ip = get_local_ip()
    print(f"📍 Local:    http://localhost:{args.port}")
    print(f"🌐 Réseau:   http://{local_ip}:{args.port}")
    
    if args.host == "0.0.0.0":
        print(f"\n💡 L'application est accessible depuis le réseau local!")
        print(f"   Utilisez l'adresse: http://{local_ip}:{args.port}")
    
    if args.share:
        print(f"\n🌍 Un lien public sera créé (valide 72h)")
    
    if args.auth:
        print(f"\n🔒 Authentification activée: {args.auth}")
    
    print("\n⚠️  Pour arrêter: appuyez sur CTRL+C")
    print("=" * 50 + "\n")
    
    # Lancer l'application
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé. Au revoir!")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
