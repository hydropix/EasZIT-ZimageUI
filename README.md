# ⚡ Z-Image-Turbo Web UI

Interface web pour générer des images avec **Z-Image-Turbo**, le modèle de diffusion de 6 milliards de paramètres développé par Alibaba/Tongyi. Optimisé pour s'exécuter sur un PC avec GPU et être accessible à distance via le réseau.

## 🎯 Caractéristiques

- **Modèle**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B paramètres)
- **Vitesse**: Génération en ~8 étapes (quelques secondes)
- **Qualité**: Rendu photoréaliste avec support texte bilingue (EN/ZH)
- **Interface**: Gradio moderne et intuitive
- **Accès**: Local + réseau local + tunnel public (optionnel)
- **Optimisations**: Flash Attention, compilation torch, CPU offloading

## 📋 Prérequis

### Configuration minimale
- **GPU**: NVIDIA avec 8GB+ VRAM (16GB recommandé)
- **RAM**: 16GB système
- **OS**: Windows 10/11, Linux, macOS (CPU lent)
- **Python**: 3.9 - 3.12

### Configuration recommandée
- **GPU**: RTX 3090/4090 (24GB VRAM)
- **RAM**: 32GB
- **CUDA**: 12.1+

## 🚀 Installation

### 1. Cloner ou créer le projet

```bash
cd Z-Image-Turbo-WebUI
```

### 2. Créer un environnement virtuel (recommandé)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note**: L'installation de `diffusers` depuis Git peut prendre quelques minutes.

### 4. (Optionnel) Installer Flash Attention

Pour une accélération supplémentaire sur les GPUs récents:

```bash
pip install flash-attn --no-build-isolation
```

> ⚠️ Nécessite CUDA toolkit et peut prendre longtemps à compiler.

## 🎮 Utilisation

### Lancement simple

```bash
python launch.py
```

L'interface sera accessible sur:
- Local: http://localhost:7860
- Réseau: http://[IP_DU_PC]:7860

### Options avancées

```bash
# Charger le modèle automatiquement au démarrage
python launch.py --load-on-start

# Changer le port
python launch.py --port 8080

# Créer un lien public (72h max)
python launch.py --share

# Activer l'authentification
python launch.py --auth admin:votremotdepasse

# Combiner les options
python launch.py --load-on-start --port 8080 --auth user:pass
```

### Accès depuis un autre PC

1. Sur le PC avec GPU (serveur):
```bash
python launch.py --host 0.0.0.0 --port 7860
```

2. Sur l'autre PC (client), ouvrir le navigateur et aller sur:
```
http://[IP_DU_SERVEUR]:7860
```

Pour trouver l'IP du serveur:
- Windows: `ipconfig` dans CMD
- Linux: `ip addr` ou `hostname -I`

## 🖼️ Interface

### Onglet Génération
- **Prompt**: Description de l'image souhaitée
- **Prompt négatif**: Éléments à éviter
- **Dimensions**: Largeur/Hauteur (512-2048)
- **Étapes**: 1-50 (9 recommandé pour Turbo)
- **Guidance Scale**: 0.0 recommandé pour Turbo
- **Seed**: Pour reproduire les résultats
- **Batch Size**: Générer plusieurs images

### Onglet Gestion du Modèle
- Chargement/déchargement du modèle
- Configuration du dtype (bfloat16/float16/float32)
- Backend d'attention (SDPA/Flash Attention)
- CPU offloading pour GPUs avec peu de VRAM
- Compilation torch pour accélération

### Exemples de prompts

1. Portrait traditionnel chinois:
```
Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, 
red floral forehead pattern. Elaborate high bun, golden phoenix headdress, 
red flowers, beads. Soft-lit outdoor night background, silhouetted tiered 
pagoda, blurred colorful distant lights.
```

2. Cyberpunk:
```
Futuristic cyberpunk city at night, neon signs in Chinese and English, 
flying vehicles, rain-slicked streets reflecting colorful lights, 
cinematic composition
```

3. Nature:
```
A serene Japanese garden with cherry blossoms in full bloom, a small 
wooden bridge over a koi pond, soft morning light filtering through 
the trees, photorealistic
```

## ⚙️ Configuration avancée

### Variables d'environnement

```bash
# Forcer le device
export CUDA_VISIBLE_DEVICES=0  # GPU spécifique

# Optimisation mémoire
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Compilation
export TORCH_COMPILE_BACKEND=inductor
```

### Utilisation directe (sans launcher)

```bash
python app.py --help

# Toutes les options
python app.py --host 0.0.0.0 --port 7860 --load-on-start --share
```

## 🔧 Dépannage

### "CUDA out of memory"
- Activez "CPU Offloading" dans l'interface
- Réduisez la résolution (512x512)
- Utilisez `float16` au lieu de `bfloat16`
- Fermez les autres applications utilisant le GPU

### "No module named 'diffusers'"
```bash
pip install git+https://github.com/huggingface/diffusers
```

### Modèle très lent
- Vérifiez que CUDA est utilisé: `torch.cuda.is_available()`
- Activez la compilation du modèle
- Essayez Flash Attention si compatible

### Impossible d'accéder depuis le réseau
- Vérifiez le firewall Windows (autoriser Python)
- Utilisez `--host 0.0.0.0`
- Vérifiez que le port n'est pas utilisé: `netstat -an | findstr 7860`

## 📁 Structure du projet

```
Z-Image-Turbo-WebUI/
├── app.py              # Application principale Gradio
├── launch.py           # Script de lancement
├── requirements.txt    # Dépendances Python
├── README.md          # Ce fichier
├── outputs/           # Images générées
│   └── images/
└── venv/              # Environnement virtuel
```

## 🌐 Accès distant avancé

### Avec ngrok (tunnel sécurisé)

1. Installer ngrok: https://ngrok.com/download
2. Créer un compte gratuit et configurer le token
3. Lancer:
```bash
ngrok http 7860
```

### Avec Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:7860
```

### Avec Tailscale (VPN mesh)

Idéal pour accès sécurisé entre vos appareils sans exposition publique.

## 📚 Ressources

- [Z-Image-Turbo sur HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Documentation Diffusers](https://huggingface.co/docs/diffusers)
- [Gradio Documentation](https://gradio.app/docs)

## 📝 Licence

Ce projet utilise Z-Image-Turbo sous licence Apache 2.0.

## 🙏 Crédits

- Modèle: [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) (Alibaba)
- UI: Gradio
- Pipeline: HuggingFace Diffusers
