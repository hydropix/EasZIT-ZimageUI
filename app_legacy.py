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
text_encoder_on_cpu = False  # Track if text encoder is offloaded to CPU
auto_generating = False  # Track if auto-generation is active

# NOTE: SeedVarianceEnhancer has been temporarily removed due to implementation issues.
# A proper implementation matching ComfyUI's approach will be added in a future update.
# For now, users can achieve diversity by:
# 1. Using different seeds
# 2. Varying the prompt slightly
# 3. Adjusting inference steps


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
    attention_backend="sdpa",
    text_encoder_cpu_offload=False,
    progress=gr.Progress()
):
    """Loads the Z-Image-Turbo model"""
    global pipe, model_loaded, model_loading, text_encoder_on_cpu
    
    if model_loading:
        return "Loading already in progress..."
    
    if model_loaded:
        return "Model already loaded!"
    
    model_loading = True
    start_time = time.time()
    last_time = start_time
    
    def log_step(step_name):
        nonlocal last_time
        current = time.time()
        print(f"  [{current - start_time:5.1f}s | +{current - last_time:5.1f}s] {step_name}")
        last_time = current
    
    try:
        print(f"\n{'='*50}")
        print(f"LOADING MODEL (~13GB from disk)")
        print(f"{'='*50}")
        
        log_step("Importing libraries...")
        from diffusers import ZImagePipeline
        
        # Detect dtype
        if dtype_choice == "bfloat16" and torch.cuda.is_available():
            dtype = torch.bfloat16
        elif dtype_choice == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        device = get_available_device()
        progress(0.1, desc="Loading from disk...")
        
        log_step("Loading pipeline (reading 13GB)...")
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        log_step("Pipeline loaded")
        
        progress(0.5, desc="Loading VAE and text encoder...")
        print(f"  Components loaded in {time.time() - start_time:.1f}s")
        
        progress(0.6, desc="Configuring model...")
        log_step("Configuring attention backend...")
        
        # Configure attention
        if attention_backend == "flash" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("flash")
                print("  Flash Attention 2 enabled")
            except Exception as e:
                print(f"  Flash Attention not available: {e}")
        elif attention_backend == "flash3" and hasattr(pipe.transformer, 'set_attention_backend'):
            try:
                pipe.transformer.set_attention_backend("_flash_3")
                print("  Flash Attention 3 enabled")
            except Exception as e:
                print(f"  Flash Attention 3 not available: {e}")
        
        log_step("Setting up device placement...")
        # CPU offloading for limited memory
        if enable_cpu_offload:
            log_step("  Enabling model CPU offload...")
            pipe.enable_model_cpu_offload()
            print("  CPU offloading enabled")
            text_encoder_on_cpu = False  # Managed by accelerate
        else:
            log_step(f"  Moving model to {device}...")
            pipe.to(device)
            
            # Optional: Text encoder specific CPU offloading
            if text_encoder_cpu_offload and hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                pipe.text_encoder.to("cpu")
                text_encoder_on_cpu = True
                print("  Text encoder offloaded to CPU (saves ~8GB VRAM)")
            else:
                text_encoder_on_cpu = False
        
        log_step("Finalizing...")
        progress(1.0, desc="Ready!")
        model_loaded = True
        model_loading = False
        load_time = time.time() - start_time
        
        device_info = f"{device.upper()}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        
        print(f"\n✓ Model loaded in {load_time:.1f}s")
        print(f"{'='*50}\n")
        
        return f"Model loaded successfully on {device_info} with {dtype_choice}! (took {load_time:.1f}s)"
        
    except Exception as e:
        model_loading = False
        return f"Error during loading: {str(e)}"


def unload_model():
    """Unloads the model to free VRAM"""
    global pipe, model_loaded, text_encoder_on_cpu
    
    if pipe is not None:
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        model_loaded = False
        text_encoder_on_cpu = False
        return "Model unloaded. VRAM freed."
    
    return "No model to unload."


def generate_image(
    prompt,
    width=1024,
    height=1024,
    num_steps=9,
    guidance_scale=0.0,
    seed=-1,
    batch_size=1,
    auto_enabled=False,
    progress=gr.Progress()
):
    """
    Generates an image with Z-Image-Turbo.
    
    Args:
        prompt: Text prompt for image generation
        width: Image width in pixels
        height: Image height in pixels  
        num_steps: Number of inference steps (8-10 recommended for Turbo)
        guidance_scale: CFG scale (0.0 recommended for Turbo)
        seed: Random seed (-1 for random)
        batch_size: Number of images to generate
        auto_enabled: Internal flag for auto-generation mode
        progress: Gradio progress callback
        
    Returns:
        PIL Image or list of PIL Images
    """
    global pipe, model_loaded
    
    if not model_loaded or pipe is None:
        gr.Warning("Model is not loaded. Please load the model first.")
        return None
    
    if not prompt or prompt.strip() == "":
        gr.Warning("Please enter a prompt.")
        return None
    
    try:
        # Handle seed
        if seed == -1:
            seed = int(time.time()) % 2**32
        
        device = get_available_device()
        generator = torch.Generator(device).manual_seed(seed)
        
        # Adjust for batch
        actual_steps = max(1, num_steps)
        
        progress(0, desc="Generation in progress...")
        
        start_time = time.time()
        
        # Progress callback
        def step_callback(pipe_obj, step, timestep, callback_kwargs):
            if step % 2 == 0:
                progress(step / actual_steps, desc=f"Step {step}/{actual_steps}...")
            return callback_kwargs
        
        # Generation parameters
        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": actual_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": batch_size,
            "callback_on_step_end": step_callback,
        }
        
        # Generate
        result = pipe(**pipe_kwargs)
        
        elapsed = time.time() - start_time
        
        # Save images
        images_to_return = []
        saved_paths = []
        
        for i, image in enumerate(result.images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zimage_{timestamp}_{uuid.uuid4().hex[:8]}_{i}.png"
            filepath = OUTPUT_DIR / "images" / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
            images_to_return.append(image)
        
        progress(1.0, desc=f"Done! ({elapsed:.1f}s)")
        
        if len(images_to_return) == 1:
            return images_to_return[0]
        else:
            return images_to_return
            
    except Exception as e:
        gr.Warning(f"Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def stop_auto_generation():
    """Stops the auto-generation loop."""
    global auto_generating
    auto_generating = False
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=False, interactive=True)


def generate_with_auto(
    prompt,
    width=1024,
    height=1024,
    num_steps=9,
    guidance_scale=0.0,
    seed=-1,
    batch_size=1,
    auto_enabled=False
):
    """
    Wrapper for generate_image that handles auto-generation mode.
    Yields results continuously if auto_enabled is True.
    Returns tuple: (image, generate_btn_visible, stop_btn_visible, auto_checkbox_interactive)
    """
    global auto_generating
    
    # If auto is enabled, start the loop
    if auto_enabled:
        auto_generating = True
        iteration = 0
        
        # Show stop button, hide generate button
        yield None, gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)
        
        while auto_generating:
            iteration += 1
            print(f"Auto-generation iteration #{iteration}")
            
            # Generate image with random seed
            result = generate_image(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=-1,  # Always random in auto mode
                batch_size=batch_size,
                auto_enabled=True
            )
            
            if not auto_generating:
                break
            
            # Yield the result
            yield result, gr.update(visible=False), gr.update(visible=True), gr.update(value=True, interactive=False)
        
        # Loop ended - restore UI
        final_result = result if 'result' in locals() else None
        yield final_result, gr.update(visible=True), gr.update(visible=False), gr.update(value=False, interactive=True)
    
    else:
        # Single generation
        result = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_size=batch_size,
            auto_enabled=False
        )
        yield result, gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)


def get_system_info():
    """Returns system information"""
    info = []
    
    # PyTorch
    info.append(f"PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        info.append(f"CUDA available: Yes")
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        info.append("CUDA available: No (CPU mode)")
    
    # Current device
    info.append(f"Active device: {get_available_device().upper()}")
    
    return "\n".join(info)


def get_vram_info():
    """Returns detailed VRAM usage for this process"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device = 0
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    
    lines = []
    lines.append(f"GPU: {torch.cuda.get_device_name(device)}")
    lines.append(f"Total VRAM: {total / 1e9:.2f} GB")
    lines.append("")
    lines.append("Current Usage:")
    lines.append(f"  Allocated: {allocated / 1e9:.2f} GB ({allocated/total*100:.1f}%)")
    lines.append(f"  Reserved:  {reserved / 1e9:.2f} GB ({reserved/total*100:.1f}%)")
    lines.append(f"  Free:      {(total - reserved) / 1e9:.2f} GB ({(total-reserved)/total*100:.1f}%)")
    lines.append("")
    lines.append("Peak Usage (this session):")
    lines.append(f"  Max Allocated: {max_allocated / 1e9:.2f} GB ({max_allocated/total*100:.1f}%)")
    lines.append(f"  Max Reserved:  {max_reserved / 1e9:.2f} GB ({max_reserved/total*100:.1f}%)")
    
    if reserved > 0:
        fragmentation = (reserved - allocated) / reserved * 100
        lines.append("")
        lines.append(f"PyTorch Cache: {fragmentation:.1f}% of reserved is not actively used")
    
    lines.append("")
    if model_loaded:
        lines.append("Model Status: LOADED")
    else:
        lines.append("Model Status: NOT LOADED")
    
    return "\n".join(lines)


def get_detailed_vram_breakdown():
    """
    Returns detailed breakdown of VRAM usage by component.
    Only works when model is loaded.
    """
    global pipe, text_encoder_on_cpu
    
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    if not model_loaded or pipe is None:
        return "Model not loaded. Load model first to see breakdown."
    
    lines = []
    lines.append("VRAM Breakdown by Component")
    lines.append("=" * 50)
    
    if text_encoder_on_cpu:
        lines.append("Note: Text Encoder is on CPU (saves ~8GB VRAM)")
        lines.append("")
    
    device = 0
    
    # Function to get memory of a module
    def get_module_memory(module, name):
        if module is None:
            return 0
        try:
            mem = sum(p.numel() * p.element_size() for p in module.parameters())
            mem += sum(b.numel() * b.element_size() for b in module.buffers())
            return mem
        except Exception:
            return 0
    
    # Check each component
    components = [
        ("Transformer (DiT)", getattr(pipe, 'transformer', None)),
        ("Text Encoder", getattr(pipe, 'text_encoder', None)),
        ("Text Encoder 2", getattr(pipe, 'text_encoder_2', None)),
        ("VAE", getattr(pipe, 'vae', None)),
    ]
    
    total_model_mem = 0
    for name, module in components:
        if module is not None:
            mem = get_module_memory(module, name)
            total_model_mem += mem
            lines.append(f"{name}: {mem / 1e9:.2f} GB")
        else:
            lines.append(f"{name}: Not present")
    
    lines.append("")
    lines.append(f"Sum of model weights: {total_model_mem / 1e9:.2f} GB")
    
    # Compare with actual allocated
    allocated = torch.cuda.memory_allocated(device)
    overhead = allocated - total_model_mem
    lines.append(f"Actual allocated: {allocated / 1e9:.2f} GB")
    lines.append(f"Overhead (buffers, caches, etc.): {overhead / 1e9:.2f} GB")
    
    # Check for float32 leakage
    lines.append("")
    lines.append("Precision Check:")
    if pipe.transformer is not None:
        transformer_dtypes = set(str(p.dtype) for p in pipe.transformer.parameters())
        lines.append(f"  Transformer dtypes: {', '.join(transformer_dtypes)}")
    if pipe.text_encoder is not None:
        te_dtypes = set(str(p.dtype) for p in pipe.text_encoder.parameters())
        lines.append(f"  Text Encoder dtypes: {', '.join(te_dtypes)}")
    
    # Memory optimization suggestions
    lines.append("")
    lines.append("Optimization Suggestions:")
    if 'torch.float32' in str(transformer_dtypes if pipe.transformer else []):
        lines.append("  - WARNING: Transformer has float32 weights!")
        lines.append("    Use bfloat16 or float16 to save VRAM.")
    if pipe.text_encoder is not None:
        te_mem = get_module_memory(pipe.text_encoder, "text_encoder")
        if te_mem > 4e9:  # > 4GB
            lines.append(f"  - Text encoder uses {te_mem/1e9:.1f} GB.")
            lines.append("    Consider enabling 'Text Encoder CPU Offload' option.")
    
    lines.append("")
    lines.append("Expected vs Actual:")
    lines.append("  Z-Image-Turbo DiT (6B bf16): ~12 GB")
    lines.append("  Text Encoder (Qwen3-4B bf16): ~8 GB")
    lines.append("  VAE: ~0.5 GB")
    lines.append("  Expected total: ~20-21 GB")
    
    return "\n".join(lines)


def warmup_cuda():
    """Pre-initializes CUDA to avoid delay during first model load."""
    if torch.cuda.is_available():
        print("  Pre-warming CUDA...")
        start = time.time()
        try:
            # Create a small tensor to force CUDA initialization
            _ = torch.randn(1, 1).cuda()
            torch.cuda.synchronize()
            print(f"  ✓ CUDA ready in {time.time() - start:.1f}s")
        except Exception as e:
            print(f"  ⚠ CUDA warmup failed: {e}")


def create_ui():
    """Creates the Gradio interface"""
    
    print("\n" + "="*50)
    print("Z-IMAGE-TURBO WEB UI")
    print("="*50)
    
    # Pre-warm CUDA to avoid 20-30s delay on first model load
    warmup_cuda()
    
    print(f"\n  Model: ~13GB (will be read from disk on first load)")
    print(f"  Expected load time: ~25-35s depending on SSD speed")
    
    # Custom black & white theme
    theme = gr.themes.Default(
        primary_hue="neutral",
        secondary_hue="gray",
        neutral_hue="gray",
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        color_accent="*neutral_600",
        color_accent_soft="*neutral_100",
        color_accent_soft_dark="*neutral_800",
        button_primary_background_fill="*neutral_900",
        button_primary_background_fill_hover="*neutral_700",
        button_primary_background_fill_dark="*neutral_100",
        button_primary_background_fill_hover_dark="*neutral_300",
        button_primary_text_color="white",
        button_primary_text_color_dark="black",
        button_secondary_background_fill="*neutral_200",
        button_secondary_background_fill_hover="*neutral_300",
        button_secondary_background_fill_dark="*neutral_800",
        button_secondary_background_fill_hover_dark="*neutral_700",
        button_cancel_background_fill="*neutral_400",
        button_cancel_background_fill_hover="*neutral_500",
        checkbox_background_color="*neutral_900",
        checkbox_background_color_dark="*neutral_100",
        checkbox_border_color="*neutral_400",
        checkbox_border_color_dark="*neutral_600",
        checkbox_border_color_focus="*neutral_700",
        checkbox_border_color_focus_dark="*neutral_400",
        slider_color="*neutral_700",
        slider_color_dark="*neutral_300",
        input_background_fill="white",
        input_background_fill_dark="*neutral_900",
        input_border_color="*neutral_300",
        input_border_color_dark="*neutral_600",
        input_border_color_focus="*neutral_500",
        input_border_color_focus_dark="*neutral_400",
        block_border_color="*neutral_200",
        block_border_color_dark="*neutral_800",
        
    )
    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    .material-icons {
        font-family: 'Material Symbols Rounded';
        font-weight: normal;
        font-style: normal;
        font-size: 18px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        vertical-align: middle;
        margin-right: 4px;
    }
    
    .container {
        max-width: 1400px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .header h1 {
        color: #000000;
        font-weight: 700;
    }
    .gallery {
        min-height: 400px;
    }
    /* Style for primary buttons */
    button.primary {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    button.primary:hover {
        background-color: #333333 !important;
    }
    /* ============================================
       SQUARE SLIDER STYLE (Progress Fill) - DARK MODE
       ============================================ */
    input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        width: 100%;
        height: 6px;
        background: #374151;
        border-radius: 0;
        outline: none;
    }
    
    input[type="range"]:focus {
        outline: none;
    }
    
    /* Square Thumb - WebKit (Chrome, Safari, Edge) */
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        background: #e5e7eb;
        border-radius: 0;
        cursor: pointer;
        border: 3px solid #1f2937;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
        margin-top: -7px;
        transition: all 0.1s ease;
    }
    
    input[type="range"]::-webkit-slider-thumb:hover {
        background: #f3f4f6;
        transform: scale(1.08);
    }
    
    input[type="range"]::-webkit-slider-thumb:active {
        transform: scale(0.95);
    }
    
    /* Track WebKit */
    input[type="range"]::-webkit-slider-runnable-track {
        width: 100%;
        height: 6px;
        cursor: pointer;
        background: #374151;
        border-radius: 0;
    }
    
    /* Square Thumb - Firefox */
    input[type="range"]::-moz-range-thumb {
        width: 20px;
        height: 20px;
        background: #e5e7eb;
        border-radius: 0;
        cursor: pointer;
        border: 3px solid #1f2937;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
        transition: all 0.1s ease;
    }
    
    input[type="range"]::-moz-range-thumb:hover {
        background: #f3f4f6;
    }
    
    /* Track Firefox */
    input[type="range"]::-moz-range-track {
        width: 100%;
        height: 6px;
        cursor: pointer;
        background: #374151;
        border-radius: 0;
    }
    
    /* ============================================
       SQUARE CHECKBOX STYLE - DARK MODE
       ============================================ */
    input[type="checkbox"] {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        background: #1f2937;
        border: 2px solid #6b7280;
        border-radius: 0;
        cursor: pointer;
        position: relative;
        transition: all 0.15s ease;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    input[type="checkbox"]:hover {
        border-color: #9ca3af;
    }
    
    input[type="checkbox"]:checked {
        background: #e5e7eb;
        border-color: #e5e7eb;
    }
    
    /* Checkmark */
    input[type="checkbox"]:checked::after {
        content: '';
        position: absolute;
        left: 5px;
        top: 1px;
        width: 6px;
        height: 11px;
        border: solid #1f2937;
        border-width: 0 2px 2px 0;
        transform: rotate(45deg);
    }
    
    input[type="checkbox"]:focus {
        outline: 2px solid #e5e7eb;
        outline-offset: 2px;
    }
    
    /* ============================================
       SQUARE RADIO BUTTON STYLE - DARK MODE
       ============================================ */
    input[type="radio"] {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        background: #1f2937;
        border: 2px solid #6b7280;
        border-radius: 0;
        cursor: pointer;
        position: relative;
        transition: all 0.15s ease;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    input[type="radio"]:hover {
        border-color: #9ca3af;
    }
    
    input[type="radio"]:checked {
        background: #e5e7eb;
        border-color: #e5e7eb;
    }
    
    /* Inner square indicator */
    input[type="radio"]:checked::after {
        content: '';
        position: absolute;
        left: 4px;
        top: 4px;
        width: 8px;
        height: 8px;
        background: #1f2937;
        border-radius: 0;
    }
    
    input[type="radio"]:focus {
        outline: 2px solid #e5e7eb;
        outline-offset: 2px;
    }
    
    /* ============================================
       PARAMETERS GROUP - Unified block (DARK MODE)
       ============================================ */
    .parameters-group {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .parameters-group .parameter-row {
        margin-bottom: 0 !important;
        padding: 6px 0;
    }
    
    .parameters-group .parameter-row:first-child {
        padding-top: 0;
    }
    
    .parameters-group .parameter-row:last-child {
        padding-bottom: 0;
    }
    
    /* Remove borders between rows for seamless look */
    .parameters-group .parameter-row > div {
        border: none !important;
        margin-bottom: 0;
    }
    
    
    /* ============================================
       RESIZABLE PROMPT TEXTBOX - Vertical only
       ============================================ */
    #prompt-textbox textarea {
        resize: vertical !important;
        min-height: 100px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    #prompt-textbox textarea::-webkit-resizer {
        background: linear-gradient(135deg, transparent 50%, #9ca3af 50%);
        border-bottom-right-radius: 2px;
    }
    
    /* ============================================
       RESIZABLE OUTPUT IMAGE - Vertical only
       ============================================ */
    #output-image {
        resize: vertical !important;
        overflow: auto;
        min-height: 300px;
        max-height: 900px;
    }
    
    #output-image img {
        height: 100%;
        object-fit: contain;
    }
    
    #output-image::-webkit-resizer {
        background: linear-gradient(135deg, transparent 50%, #9ca3af 50%);
        border-bottom-right-radius: 2px;
    }
    
    /* ============================================
       AUTO GENERATE CHECKBOX STYLE
       ============================================ */
    #auto-generate-checkbox input[type="checkbox"] {
        width: 18px;
        height: 18px;
        background: #1f2937;
        border: 2px solid #fbbf24;
        border-radius: 4px;
    }
    
    #auto-generate-checkbox input[type="checkbox"]:checked {
        background: #fbbf24;
        border-color: #fbbf24;
    }
    
    #auto-generate-checkbox input[type="checkbox"]:checked::after {
        border-color: #1f2937;
        width: 4px;
        height: 8px;
        left: 5px;
        top: 2px;
    }
    
    #auto-generate-checkbox label {
        color: #fbbf24 !important;
        font-weight: 500;
    }
    
    /* ============================================
       STOP BUTTON STYLE (Red)
       ============================================ */
    #stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border-color: #dc2626 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        animation: pulse-stop 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse-stop {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    }
    
    #stop-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        transform: translateY(-1px);
    }
    
    /* ============================================
       FORCE DARK MODE
       ============================================ */
    .gradio-container {
        color-scheme: dark !important;
    }
    
    body {
        background-color: #0f0f0f !important;
        color: #f3f4f6 !important;
    }
    
    /* Override all background colors for dark mode */
    .gradio-container,
    .gradio-container .fillable,
    .gradio-container .contain,
    .gradio-container .block,
    .gradio-container .padded {
        background-color: #0f0f0f !important;
    }
    
    /* Input fields dark */
    input[type="text"],
    textarea,
    .input-text {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border-color: #374151 !important;
    }
    
    /* Labels dark */
    label,
    .label,
    .field-label {
        color: #f3f4f6 !important;
    }
    
    /* ============================================
       YELLOW GENERATE BUTTON
       ============================================ */
    #generate-btn {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        border-color: #f59e0b !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    #generate-btn:hover {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border-color: #d97706 !important;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.4) !important;
        transform: translateY(-1px);
    }
    
    #generate-btn:active {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(251, 191, 36, 0.3) !important;
    }
    
    /* Dark mode adjustments for other elements */
    .parameters-group {
        background: #1f2937 !important;
        border-color: #374151 !important;
    }
    
    input[type="checkbox"] {
        background: #1f2937;
        border-color: #6b7280;
    }
    input[type="checkbox"]:checked {
        background: #e5e7eb;
        border-color: #e5e7eb;
    }
    input[type="checkbox"]:checked::after {
        border-color: #1f2937;
    }
    input[type="checkbox"]:focus {
        outline-color: #e5e7eb;
    }
    
    input[type="radio"] {
        background: #1f2937;
        border-color: #6b7280;
    }
    input[type="radio"]:checked {
        background: #e5e7eb;
        border-color: #e5e7eb;
    }
    input[type="radio"]:checked::after {
        background: #1f2937;
    }
    input[type="radio"]:focus {
        outline-color: #e5e7eb;
    }
    
    input[type="range"] {
        background: #374151;
    }
    input[type="range"]::-webkit-slider-thumb {
        background: #e5e7eb;
        border-color: #1f2937;
    }
    input[type="range"]::-webkit-slider-runnable-track {
        background: #374151;
    }
    input[type="range"]::-moz-range-thumb {
        background: #e5e7eb;
        border-color: #1f2937;
    }
    input[type="range"]::-moz-range-track {
        background: #374151;
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
    
    with gr.Blocks(css=custom_css, theme=theme, title="Z-Image-Turbo Web UI") as app:
        
        # Close confirmation script
        gr.HTML(confirm_exit_js)
        
        # Header
        gr.Markdown("""
        # Z-Image-Turbo Web UI
        
        Web interface for generating images with **Z-Image-Turbo** (Alibaba/Tongyi)
        
        > 6B parameter model | 8-step generation | Bilingual EN/ZH support
        """)
        
        with gr.Tabs():
            
            # Generation Tab
            with gr.Tab("Generation", id="generate"):
                with gr.Row():
                    # Left column: Prompt + Image (2/3 of space)
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=4,
                            value="",
                            elem_id="prompt-textbox"
                        )
                        
                        output_image = gr.Image(
                            label="Generated Image",
                            type="pil",
                            height=600,
                            elem_id="output-image"
                        )
                    
                    # Right column: Parameters (1/3 of space)
                    with gr.Column(scale=1):
                        
                        # Action buttons at top
                        with gr.Row():
                            auto_generate = gr.Checkbox(
                                label="Auto",
                                value=False,
                                scale=1,
                                elem_id="auto-generate-checkbox",
                                info="Continuous generation"
                            )
                            generate_btn = gr.Button(
                                "Generate Image",
                                variant="primary",
                                scale=2,
                                elem_id="generate-btn",
                                icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/rocket/default/24px.svg"
                            )
                            stop_btn = gr.Button(
                                "Stop",
                                scale=1,
                                variant="stop",
                                visible=False,
                                elem_id="stop-btn",
                                icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/stop/default/24px.svg"
                            )
                            clear_btn = gr.Button(
                                "Clear",
                                scale=1,
                                icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/delete/default/24px.svg"
                            )
                        
                        # Parameters block - grouped visually
                        with gr.Group(elem_classes="parameters-group"):
                            with gr.Row(elem_classes="parameter-row"):
                                width = gr.Slider(
                                    label="Width",
                                    minimum=512,
                                    maximum=2048,
                                    step=64,
                                    value=1024
                                )
                                height = gr.Slider(
                                    label="Height",
                                    minimum=512,
                                    maximum=2048,
                                    step=64,
                                    value=1024
                                )
                            
                            with gr.Row(elem_classes="parameter-row"):
                                num_steps = gr.Slider(
                                    label="Inference steps",
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=9,
                                    info="9 = ~8 forwards DiT (recommended)"
                                )
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=0.0,
                                    maximum=10.0,
                                    step=0.1,
                                value=0.0,
                                    info="0.0 for Turbo (recommended)"
                                )
                            
                            with gr.Row(elem_classes="parameter-row"):
                                seed = gr.Number(
                                    label="Seed (-1 = random)",
                                    value=-1,
                                    precision=0
                                )
                                batch_size = gr.Slider(
                                    label="Batch Size",
                                    minimum=1,
                                    maximum=4,
                                    step=1,
                                    value=1
                                )
                        
                # Event handlers
                generate_btn.click(
                    fn=generate_with_auto,
                    inputs=[prompt, width, height, num_steps, guidance_scale, seed, batch_size, auto_generate],
                    outputs=[output_image, generate_btn, stop_btn, auto_generate]
                )
                
                stop_btn.click(
                    fn=stop_auto_generation,
                    outputs=[generate_btn, stop_btn, auto_generate]
                )
                
                clear_btn.click(
                    fn=lambda: [None, "", 1024, 1024, 9, 0.0, -1, 1, False],
                    outputs=[output_image, prompt, width, height, num_steps, guidance_scale, seed, batch_size, auto_generate]
                )
            
            # Diversity Enhancement Tab (Feature temporarily disabled)
            with gr.Tab("Diversity Enhancement", id="diversity"):
                gr.Markdown("""
                # Diversity Enhancement - Temporarily Unavailable
                
                ## Status
                
                The SeedVarianceEnhancer feature has been **temporarily removed** due to implementation issues.
                The previous implementation was not working correctly with Z-Image-Turbo's specific pipeline format.
                
                ## Why It Was Removed
                
                Z-Image-Turbo uses a different internal format than standard Stable Diffusion:
                - Variable-length embeddings (not padded tensors)
                - List-based prompt encoding
                - Custom latent space handling
                
                The previous implementation was causing:
                - Slow generation times
                - Artifacts and quality issues
                - Inconsistent results
                
                ## Workarounds for Now
                
                To achieve diversity in your generations:
                
                1. **Use different seeds** - Each seed produces different random noise
                2. **Vary your prompts slightly** - Change adjectives, lighting, or composition terms
                3. **Adjust inference steps** - Try 8, 9, or 10 steps for different results
                4. **Use batch generation** - Generate multiple images at once with different seeds
                
                ## Future Plans
                
                A proper implementation will be added in a future update, based on ComfyUI's proven approach:
                - Correct handling of Z-Image-Turbo's embedding format
                - Efficient latent space perturbation
                - Better integration with the pipeline's denoising loop
                
                For updates, check the project repository.
                """)
            
            # Model Tab
            with gr.Tab("Model Management", id="model"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Configuration")
                        
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
                        
                        text_encoder_cpu_offload = gr.Checkbox(
                            label="Text Encoder CPU Offload",
                            value=False,
                            info="Frees ~8GB VRAM by running Qwen3-4B text encoder on CPU. Adds 1-2s delay for prompt encoding. Recommended for 16GB GPUs."
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button(
                                "Load Model", 
                                variant="primary",
                                icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/download/default/24px.svg"
                            )
                            unload_btn = gr.Button(
                                "Unload Model", 
                                variant="stop",
                                icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/delete_forever/default/24px.svg"
                            )
                        
                        model_status = gr.Textbox(
                            label="Status",
                            value="Model not loaded",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column():
                        gr.Markdown("### System Information")
                        system_info = gr.Textbox(
                            label="Information",
                            value=get_system_info(),
                            interactive=False,
                            lines=10
                        )
                        refresh_info_btn = gr.Button(
                            "Refresh",
                            icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/refresh/default/24px.svg"
                        )
                        
                        gr.Markdown("### VRAM Usage (This Process)")
                        vram_info = gr.Textbox(
                            label="VRAM Info",
                            value=get_vram_info,
                            interactive=False,
                            lines=12
                        )
                        refresh_vram_btn = gr.Button(
                            "Refresh VRAM",
                            icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/memory/default/24px.svg"
                        )
                        
                        vram_breakdown_btn = gr.Button(
                            "Detailed Breakdown",
                            variant="secondary",
                            icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/analytics/default/24px.svg"
                        )
                        vram_breakdown_info = gr.Textbox(
                            label="Component Breakdown",
                            value="Click 'Detailed Breakdown' to analyze VRAM usage by component",
                            interactive=False,
                            lines=20
                        )
                
                load_btn.click(
                    fn=load_model,
                    inputs=[dtype_choice, enable_cpu_offload, attention_backend, text_encoder_cpu_offload],
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
                
                refresh_vram_btn.click(
                    fn=get_vram_info,
                    outputs=vram_info
                )
                
                vram_breakdown_btn.click(
                    fn=get_detailed_vram_breakdown,
                    outputs=vram_breakdown_info
                )
        
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
    print("Z-Image-Turbo Web UI")
    print("=" * 60)
    
    # Setup
    setup_output_dir()
    
    # System info
    print("\nSystem information:")
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
    
    print(f"\nStarting server...")
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
