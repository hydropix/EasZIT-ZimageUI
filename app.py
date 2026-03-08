#!/usr/bin/env python3
"""
Z-Image-Turbo Web UI
Gradio interface for generating images with Z-Image-Turbo
Runs on PC with GPU, remote access via browser
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

# Global variable for the pipeline
pipe = None
model_loaded = False
model_loading = False


def setup_output_dir():
    """Creates the output folder if it doesn't exist"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)


def get_available_device():
    """Detects the best available device"""
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
    """Loads the Z-Image-Turbo model"""
    global pipe, model_loaded, model_loading
    
    if model_loading:
        return "⏳ Loading already in progress..."
    
    if model_loaded:
        return "✅ Model already loaded!"
    
    model_loading = True
    
    try:
        from diffusers import ZImagePipeline
        
        progress(0.1, desc="Importing libraries...")
        
        # Detect dtype
        if dtype_choice == "bfloat16" and torch.cuda.is_available():
            dtype = torch.bfloat16
        elif dtype_choice == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        device = get_available_device()
        progress(0.2, desc=f"Loading model on {device.upper()}...")
        print(f"📥 Downloading model from HuggingFace (~13 GB)...")
        
        # Load pipeline
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        
        progress(0.6, desc="Configuring model...")
        
        # Configure attention
        if attention_backend == "flash" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("flash")
                print("✅ Flash Attention 2 enabled")
            except Exception as e:
                print(f"⚠️ Flash Attention not available: {e}")
        elif attention_backend == "flash3" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("_flash_3")
                print("✅ Flash Attention 3 enabled")
            except Exception as e:
                print(f"⚠️ Flash Attention 3 not available: {e}")
        
        # CPU offloading for limited memory
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offloading enabled")
        else:
            pipe.to(device)
        
        # Compilation for acceleration
        if compile_model and hasattr(torch, 'compile'):
            try:
                progress(0.8, desc="Compiling model (may take time)...")
                pipe.transformer.compile()
                print("✅ Model compiled")
            except Exception as e:
                print(f"⚠️ Compilation disabled: {e}")
                print("   Model will run without compilation.")
        
        progress(1.0, desc="Ready!")
        model_loaded = True
        model_loading = False
        
        device_info = f"{device.upper()}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        
        return f"✅ Model loaded successfully on {device_info} with {dtype_choice}!"
        
    except Exception as e:
        model_loading = False
        return f"❌ Error during loading: {str(e)}"


def unload_model():
    """Unloads the model to free VRAM"""
    global pipe, model_loaded
    
    if pipe is not None:
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        model_loaded = False
        return "🗑️ Model unloaded. VRAM freed."
    
    return "No model to unload."


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
    """Generates an image with Z-Image-Turbo"""
    global pipe, model_loaded
    
    if not model_loaded or pipe is None:
        return None, "❌ Model is not loaded. Please load the model first."
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Please enter a prompt."
    
    try:
        # Handle seed
        if seed == -1:
            seed = int(time.time()) % 2**32
        
        generator = torch.Generator(get_available_device()).manual_seed(seed)
        
        # Adjust for batch
        actual_steps = max(1, num_steps)
        
        progress(0, desc="Generation in progress...")
        
        start_time = time.time()
        
        # Generation
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
                progress(step / actual_steps, desc=f"Step {step}/{actual_steps}...") 
                or kwargs
            ) if step % 2 == 0 else kwargs,
        )
        
        elapsed = time.time() - start_time
        
        # Save
        images_to_return = []
        saved_paths = []
        
        for i, image in enumerate(result.images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zimage_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
            filepath = OUTPUT_DIR / "images" / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
            images_to_return.append(image)
        
        progress(1.0, desc="Done!")
        
        info = f"✅ Generation completed in {elapsed:.2f}s | Seed: {seed} | Saved to: {OUTPUT_DIR / 'images'}"
        
        if len(images_to_return) == 1:
            return images_to_return[0], info
        else:
            return images_to_return, info
            
    except Exception as e:
        return None, f"❌ Generation error: {str(e)}"


def get_system_info():
    """Returns system information"""
    info = []
    
    # PyTorch
    info.append(f"PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        info.append(f"CUDA available: ✅")
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        info.append("CUDA available: ❌ (CPU mode)")
    
    # Current device
    info.append(f"Active device: {get_available_device().upper()}")
    
    return "\n".join(info)


def create_ui():
    """Creates the Gradio interface"""
    
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
    
    # JavaScript for close/reload confirmation
    confirm_exit_js = """
    <script>
    window.addEventListener('beforeunload', function (e) {
        e.preventDefault();
        e.returnValue = '';
    });
    </script>
    """
    
    with gr.Blocks(css=custom_css, title="Z-Image-Turbo Web UI") as app:
        
        # Close confirmation script
        gr.HTML(confirm_exit_js)
        
        # Header
        gr.Markdown("""
        # ⚡ Z-Image-Turbo Web UI
        
        Web interface for generating images with **Z-Image-Turbo** (Alibaba/Tongyi)
        
        > 6B parameter model | 8-step generation | Bilingual EN/ZH support
        """)
        
        with gr.Tabs():
            
            # Generation Tab
            with gr.Tab("🎨 Generation", id="generate"):
                with gr.Row():
                    # Left column: Parameters
                    with gr.Column(scale=1):
                        
                        prompt = gr.Textbox(
                            label="📝 Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=4,
                            value=""
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="⛔ Negative prompt (optional)",
                            placeholder="Elements to avoid...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            width = gr.Slider(
                                label="📐 Width",
                                minimum=512,
                                maximum=2048,
                                step=64,
                                value=1024
                            )
                            height = gr.Slider(
                                label="📐 Height",
                                minimum=512,
                                maximum=2048,
                                step=64,
                                value=1024
                            )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="🔄 Inference steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=9,
                                info="9 = ~8 forwards DiT (recommended)"
                            )
                            guidance_scale = gr.Slider(
                                label="⚖️ Guidance Scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=0.0,
                                info="0.0 for Turbo (recommended)"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="🎲 Seed (-1 = random)",
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
                                "🚀 Generate Image",
                                variant="primary",
                                scale=2
                            )
                            clear_btn = gr.Button(
                                "🗑️ Clear",
                                scale=1
                            )
                        
                    
                    # Right column: Result
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="🖼️ Generated Image",
                            type="pil",
                            height=600
                        )
                        output_info = gr.Textbox(
                            label="ℹ️ Information",
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
            
            # Model Tab
            with gr.Tab("⚙️ Model Management", id="model"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔧 Model Configuration")
                        
                        dtype_choice = gr.Radio(
                            choices=["bfloat16", "float16", "float32"],
                            value="bfloat16",
                            label="Precision (dtype)",
                            info="bfloat16 recommended for recent GPUs"
                        )
                        
                        attention_backend = gr.Radio(
                            choices=["sdpa", "flash", "flash3"],
                            value="sdpa",
                            label="Attention backend",
                            info="Flash Attention requires separate installation"
                        )
                        
                        enable_cpu_offload = gr.Checkbox(
                            label="Enable CPU Offloading",
                            value=False,
                            info="For GPUs with limited VRAM"
                        )
                        
                        compile_model = gr.Checkbox(
                            label="Compile model (torch.compile) [Experimental]",
                            value=False,
                            info="+30%% perf but may cause errors (disable if issues)"
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button("📥 Load Model", variant="primary")
                            unload_btn = gr.Button("🗑️ Unload Model", variant="stop")
                        
                        model_status = gr.Textbox(
                            label="📊 Status",
                            value="Model not loaded",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 💻 System Information")
                        system_info = gr.Textbox(
                            label="🖥️ Information",
                            value=get_system_info(),
                            interactive=False,
                            lines=10
                        )
                        refresh_info_btn = gr.Button("🔄 Refresh")
                
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
            
            # Advanced Parameters Tab
            with gr.Tab("🔬 Advanced Parameters", id="advanced"):
                gr.Markdown("""
                ### 🎯 Advanced generation parameters
                
                These parameters allow fine control over the generation process.
                """
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📐 Preset aspect ratios")
                        aspect_1_1 = gr.Button("1:1 Square (1024×1024)")
                        aspect_16_9 = gr.Button("16:9 Landscape (1024×576)")
                        aspect_9_16 = gr.Button("9:16 Portrait (576×1024)")
                        aspect_4_3 = gr.Button("4:3 (1024×768)")
                        aspect_3_4 = gr.Button("3:4 (768×1024)")
                        aspect_21_9 = gr.Button("21:9 Ultrawide (1024×448)")
                
                def set_dimensions(w, h):
                    return w, h
                
                # These buttons would update sliders in generation tab
                # (would require shared components or state)
        
        # Footer
        gr.Markdown("""
        ---
        
        **Z-Image-Turbo Web UI** | Built with [Gradio](https://gradio.app) and [Diffusers](https://huggingface.co/docs/diffusers)
        
        Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
        """)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to use (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--load-on-start", action="store_true", help="Load model at startup")
    parser.add_argument("--auth", type=str, help="Authentication user:pass")
    parser.add_argument("--ssl-cert", type=str, help="Path to SSL certificate")
    parser.add_argument("--ssl-key", type=str, help="Path to SSL key")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚡ Z-Image-Turbo Web UI")
    print("=" * 60)
    
    # Setup
    setup_output_dir()
    
    # System info
    print("\n🖥️ System information:")
    print(get_system_info())
    
    # Create UI
    app = create_ui()
    
    # SSL configuration if provided
    ssl_verify = False
    if args.ssl_cert and args.ssl_key:
        ssl_verify = (args.ssl_cert, args.ssl_key)
    
    # Authentication
    auth = None
    if args.auth:
        user, password = args.auth.split(":")
        auth = (user, password)
    
    print(f"\n🌐 Starting server...")
    print(f"   Local URL: http://localhost:{args.port}")
    print(f"   Network URL: http://{args.host}:{args.port}")
    
    if args.share:
        print("   Public Gradio link will be created...")
    
    if auth:
        print(f"   Authentication enabled: {auth[0]}")
    
    print("\n" + "=" * 60)
    
    # Launch
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
