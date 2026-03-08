#!/usr/bin/env python3
"""
Z-Image-Turbo Web UI
Interface Gradio pour générer des images avec Z-Image-Turbo
Exécute sur PC avec GPU, accès distant via navigateur
"""

import os
import sys
import time
import uuid
import argparse
from pathlib import Path
from datetime import datetime

import torch
import gradio as gr
from PIL import Image

# Configuration
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 9
DEFAULT_SEED = -1
OUTPUT_DIR = Path("outputs")

# Global variable pour le pipeline
pipe = None
model_loaded = False
model_loading = False


def setup_output_dir():
    """Crée le dossier de sortie s'il n'existe pas"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)


def get_available_device():
    """Détecte le meilleur device disponible"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(
    dtype_choice="bfloat16",
    enable_cpu_offload=False,
    compile_model=False,
    attention_backend="sdpa",
    progress=gr.Progress()
):
    """Charge le modèle Z-Image-Turbo"""
    global pipe, model_loaded, model_loading
    
    if model_loading:
        return "⏳ Chargement déjà en cours..."
    
    if model_loaded:
        return "✅ Modèle déjà chargé!"
    
    model_loading = True
    
    try:
        from diffusers import ZImagePipeline
        
        progress(0.1, desc="Import des bibliothèques...")
        
        # Détection du dtype
        if dtype_choice == "bfloat16" and torch.cuda.is_available():
            dtype = torch.bfloat16
        elif dtype_choice == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        device = get_available_device()
        progress(0.2, desc=f"Chargement du modèle sur {device.upper()}...")
        print(f"📥 Téléchargement du modèle depuis HuggingFace (~13 GB)...")
        
        # Chargement du pipeline
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        
        progress(0.6, desc="Configuration du modèle...")
        
        # Configuration de l'attention
        if attention_backend == "flash" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("flash")
                print("✅ Flash Attention 2 activé")
            except Exception as e:
                print(f"⚠️ Flash Attention non disponible: {e}")
        elif attention_backend == "flash3" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("_flash_3")
                print("✅ Flash Attention 3 activé")
            except Exception as e:
                print(f"⚠️ Flash Attention 3 non disponible: {e}")
        
        # CPU offloading pour mémoire limitée
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offloading activé")
        else:
            pipe.to(device)
        
        # Compilation pour accélération
        if compile_model and hasattr(torch, 'compile'):
            try:
                progress(0.8, desc="Compilation du modèle (peut prendre du temps)...")
                pipe.transformer.compile()
                print("✅ Modèle compilé")
            except Exception as e:
                print(f"⚠️ Compilation désactivée: {e}")
                print("   Le modèle fonctionnera sans compilation.")
        
        progress(1.0, desc="Prêt!")
        model_loaded = True
        model_loading = False
        
        device_info = f"{device.upper()}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        
        return f"✅ Modèle chargé avec succès sur {device_info} avec {dtype_choice}!"
        
    except Exception as e:
        model_loading = False
        return f"❌ Erreur lors du chargement: {str(e)}"


def unload_model():
    """Décharge le modèle pour libérer la VRAM"""
    global pipe, model_loaded
    
    if pipe is not None:
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        model_loaded = False
        return "🗑️ Modèle déchargé. VRAM libérée."
    
    return "Aucun modèle à décharger."


def generate_image(
    prompt,
    negative_prompt="",
    width=1024,
    height=1024,
    num_steps=9,
    guidance_scale=0.0,
    seed=-1,
    batch_size=1,
    progress=gr.Progress()
):
    """Génère une image avec Z-Image-Turbo"""
    global pipe, model_loaded
    
    if not model_loaded or pipe is None:
        return None, "❌ Le modèle n'est pas chargé. Veuillez d'abord charger le modèle."
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Veuillez entrer un prompt."
    
    try:
        # Gestion du seed
        if seed == -1:
            seed = int(time.time()) % 2**32
        
        generator = torch.Generator(get_available_device()).manual_seed(seed)
        
        # Ajustement pour batch
        actual_steps = max(1, num_steps)
        
        progress(0, desc="Génération en cours...")
        
        start_time = time.time()
        
        # Génération
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=actual_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size,
            callback_on_step_end=lambda pipe_obj, step, timestep, kwargs: (
                progress(step / actual_steps, desc=f"Étape {step}/{actual_steps}...") 
                or kwargs
            ) if step % 2 == 0 else kwargs,
        )
        
        elapsed = time.time() - start_time
        
        # Sauvegarde
        images_to_return = []
        saved_paths = []
        
        for i, image in enumerate(result.images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zimage_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
            filepath = OUTPUT_DIR / "images" / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
            images_to_return.append(image)
        
        progress(1.0, desc="Terminé!")
        
        info = f"✅ Génération terminée en {elapsed:.2f}s | Seed: {seed} | Sauvegardé dans: {OUTPUT_DIR / 'images'}"
        
        if len(images_to_return) == 1:
            return images_to_return[0], info
        else:
            return images_to_return, info
            
    except Exception as e:
        return None, f"❌ Erreur de génération: {str(e)}"


def get_system_info():
    """Retourne les informations système"""
    info = []
    
    # PyTorch
    info.append(f"PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        info.append(f"CUDA disponible: ✅")
        info.append(f"Version CUDA: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        info.append("CUDA disponible: ❌ (utilisation CPU)")
    
    # Device courant
    info.append(f"Device actif: {get_available_device().upper()}")
    
    return "\n".join(info)


def create_ui():
    """Crée l'interface Gradio"""
    
    custom_css = """
    .container {
        max-width: 1400px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .header h1 {
        color: #ff6b35;
        font-weight: 700;
    }
    .gallery {
        min-height: 400px;
    }
    """
    
    # JavaScript pour la confirmation de fermeture/rechargement
    confirm_exit_js = """
    <script>
    window.addEventListener('beforeunload', function (e) {
        e.preventDefault();
        e.returnValue = '';
    });
    </script>
    """
    
    with gr.Blocks(css=custom_css, title="Z-Image-Turbo Web UI") as app:
        
        # Script de confirmation avant fermeture
        gr.HTML(confirm_exit_js)
        
        # Header
        gr.Markdown("""
        # ⚡ Z-Image-Turbo Web UI
        
        Interface web pour générer des images avec **Z-Image-Turbo** (Alibaba/Tongyi)
        
        > Modèle de 6B paramètres | Génération en 8 étapes | Support bilingue EN/ZH
        """)
        
        with gr.Tabs():
            
            # Onglet Génération
            with gr.Tab("🎨 Génération", id="generate"):
                with gr.Row():
                    # Colonne gauche: Paramètres
                    with gr.Column(scale=1):
                        
                        prompt = gr.Textbox(
                            label="📝 Prompt",
                            placeholder="Décrivez l'image que vous souhaitez générer...",
                            lines=4,
                            value=""
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="⛔ Prompt négatif (optionnel)",
                            placeholder="Éléments à éviter...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            width = gr.Slider(
                                label="📐 Largeur",
                                minimum=512,
                                maximum=2048,
                                step=64,
                                value=1024
                            )
                            height = gr.Slider(
                                label="📐 Hauteur",
                                minimum=512,
                                maximum=2048,
                                step=64,
                                value=1024
                            )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="🔄 Étapes d'inférence",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=9,
                                info="9 = ~8 forwards DiT (recommandé)"
                            )
                            guidance_scale = gr.Slider(
                                label="⚖️ Guidance Scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=0.0,
                                info="0.0 pour Turbo (recommandé)"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="🎲 Seed (-1 = aléatoire)",
                                value=-1,
                                precision=0
                            )
                            batch_size = gr.Slider(
                                label="📦 Batch Size",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=1
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button(
                                "🚀 Générer l'image",
                                variant="primary",
                                scale=2
                            )
                            clear_btn = gr.Button(
                                "🗑️ Effacer",
                                scale=1
                            )
                        
                    
                    # Colonne droite: Résultat
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="🖼️ Image générée",
                            type="pil",
                            height=600
                        )
                        output_info = gr.Textbox(
                            label="ℹ️ Informations",
                            lines=3,
                            interactive=False
                        )
                        

                
                # Event handlers
                generate_btn.click(
                    fn=generate_image,
                    inputs=[prompt, negative_prompt, width, height, num_steps, guidance_scale, seed, batch_size],
                    outputs=[output_image, output_info]
                )
                
                clear_btn.click(
                    fn=lambda: [None, "", "", 1024, 1024, 9, 0.0, -1, 1],
                    outputs=[output_image, prompt, negative_prompt, width, height, num_steps, guidance_scale, seed, batch_size]
                )
            
            # Onglet Modèle
            with gr.Tab("⚙️ Gestion du Modèle", id="model"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔧 Configuration du Modèle")
                        
                        dtype_choice = gr.Radio(
                            choices=["bfloat16", "float16", "float32"],
                            value="bfloat16",
                            label="Précision (dtype)",
                            info="bfloat16 recommandé pour les GPUs récents"
                        )
                        
                        attention_backend = gr.Radio(
                            choices=["sdpa", "flash", "flash3"],
                            value="sdpa",
                            label="Backend d'attention",
                            info="Flash Attention nécessite installation séparée"
                        )
                        
                        enable_cpu_offload = gr.Checkbox(
                            label="Activer CPU Offloading",
                            value=False,
                            info="Pour GPUs avec peu de VRAM"
                        )
                        
                        compile_model = gr.Checkbox(
                            label="Compiler le modèle (torch.compile) [Experimental]",
                            value=False,
                            info="+30%% perf mais peut causer des erreurs (desactive si probleme)"
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button("📥 Charger le modèle", variant="primary")
                            unload_btn = gr.Button("🗑️ Décharger le modèle", variant="stop")
                        
                        model_status = gr.Textbox(
                            label="📊 Statut",
                            value="Modèle non chargé",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 💻 Informations Système")
                        system_info = gr.Textbox(
                            label="🖥️ Informations",
                            value=get_system_info(),
                            interactive=False,
                            lines=10
                        )
                        refresh_info_btn = gr.Button("🔄 Rafraîchir")
                
                load_btn.click(
                    fn=load_model,
                    inputs=[dtype_choice, enable_cpu_offload, compile_model, attention_backend],
                    outputs=model_status
                )
                
                unload_btn.click(
                    fn=unload_model,
                    outputs=model_status
                )
                
                refresh_info_btn.click(
                    fn=get_system_info,
                    outputs=system_info
                )
            
            # Onglet Paramètres avancés
            with gr.Tab("🔬 Paramètres avancés", id="advanced"):
                gr.Markdown("""
                ### 🎯 Paramètres avancés de génération
                
                Ces paramètres permettent un contrôle fin sur le processus de génération.
                """
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📐 Ratios d'aspect prédéfinis")
                        aspect_1_1 = gr.Button("1:1 Carré (1024×1024)")
                        aspect_16_9 = gr.Button("16:9 Paysage (1024×576)")
                        aspect_9_16 = gr.Button("9:16 Portrait (576×1024)")
                        aspect_4_3 = gr.Button("4:3 (1024×768)")
                        aspect_3_4 = gr.Button("3:4 (768×1024)")
                        aspect_21_9 = gr.Button("21:9 Ultrawide (1024×448)")
                
                def set_dimensions(w, h):
                    return w, h
                
                # Ces boutons mettraient à jour les sliders dans l'onglet génération
                # (nécessiterait des composants partagés ou state)
        
        # Footer
        gr.Markdown("""
        ---
        
        **Z-Image-Turbo Web UI** | Développé avec [Gradio](https://gradio.app) et [Diffusers](https://huggingface.co/docs/diffusers)
        
        Modèle: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
        """)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host à écouter (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port à utiliser (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Créer un lien public Gradio")
    parser.add_argument("--load-on-start", action="store_true", help="Charger le modèle au démarrage")
    parser.add_argument("--auth", type=str, help="Authentification user:pass")
    parser.add_argument("--ssl-cert", type=str, help="Chemin vers le certificat SSL")
    parser.add_argument("--ssl-key", type=str, help="Chemin vers la clé SSL")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚡ Z-Image-Turbo Web UI")
    print("=" * 60)
    
    # Setup
    setup_output_dir()
    
    # Info système
    print("\n🖥️ Informations système:")
    print(get_system_info())
    
    # Création de l'UI
    app = create_ui()
    
    # Configuration SSL si fournie
    ssl_verify = False
    if args.ssl_cert and args.ssl_key:
        ssl_verify = (args.ssl_cert, args.ssl_key)
    
    # Authentification
    auth = None
    if args.auth:
        user, password = args.auth.split(":")
        auth = (user, password)
    
    print(f"\n🌐 Lancement du serveur...")
    print(f"   URL locale: http://localhost:{args.port}")
    print(f"   URL réseau: http://{args.host}:{args.port}")
    
    if args.share:
        print("   Lien public Gradio sera créé...")
    
    if auth:
        print(f"   Authentification activée: {auth[0]}")
    
    print("\n" + "=" * 60)
    
    # Lancement
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        ssl_verify=ssl_verify if ssl_verify else False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
