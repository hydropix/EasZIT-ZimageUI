"""
Gradio UI for Z-Image-Turbo - Original Style with Node Backend
Preserves the original UX while using node-based execution engine
"""

import gradio as gr
import os
from typing import Optional, Tuple
from pathlib import Path

from .engine.executor import ExecutionEngine
from .workflows.builder import (
    create_txt2img_workflow,
    create_img2img_workflow,
)
from .nodes.core import *  # Register all core nodes
from .nodes.diffusers_nodes import *  # Register diffusers nodes


class ZImageUI:
    """Gradio interface for Z-Image-Turbo with original styling."""
    
    def __init__(self):
        self.engine = ExecutionEngine(cache_enabled=True)
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs("outputs/images", exist_ok=True)
    
    def _get_theme(self):
        """Get the original black & white theme."""
        return gr.themes.Default(
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
    
    def _get_css(self):
        """Get the original custom CSS."""
        return """
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
        
        /* SQUARE SLIDER STYLE - DARK MODE */
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
        
        input[type="range"]::-webkit-slider-runnable-track {
            width: 100%;
            height: 6px;
            cursor: pointer;
            background: #374151;
            border-radius: 0;
        }
        
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
        
        input[type="range"]::-moz-range-track {
            width: 100%;
            height: 6px;
            cursor: pointer;
            background: #374151;
            border-radius: 0;
        }
        
        /* SQUARE CHECKBOX STYLE */
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
        
        /* PARAMETERS GROUP - Unified block */
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
        
        .parameters-group .parameter-row > div {
            border: none !important;
            margin-bottom: 0;
        }
        
        /* RESIZABLE PROMPT TEXTBOX */
        #prompt-textbox textarea {
            resize: vertical !important;
            min-height: 100px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        /* RESIZABLE OUTPUT IMAGE */
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
        
        /* YELLOW GENERATE BUTTON */
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
        
        /* FORCE DARK MODE */
        .gradio-container {
            color-scheme: dark !important;
        }
        
        body {
            background-color: #0f0f0f !important;
            color: #f3f4f6 !important;
        }
        
        .gradio-container,
        .gradio-container .fillable,
        .gradio-container .contain,
        .gradio-container .block,
        .gradio-container .padded {
            background-color: #0f0f0f !important;
        }
        
        input[type="text"],
        textarea,
        .input-text {
            background-color: #1f2937 !important;
            color: #f3f4f6 !important;
            border-color: #374151 !important;
        }
        
        label,
        .label,
        .field-label {
            color: #f3f4f6 !important;
        }
        """
    
    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface with original styling."""
        
        theme = self._get_theme()
        custom_css = self._get_css()
        
        with gr.Blocks(css=custom_css, theme=theme, title="Z-Image-Turbo Web UI") as app:
            
            # Header
            gr.Markdown("""
            # Z-Image-Turbo Web UI (Node-Based)
            
            Web interface for generating images with **Z-Image-Turbo** (Alibaba/Tongyi)
            
            > 6B parameter model | 8-step generation | Node-based backend
            """)
            
            with gr.Tabs():
                
                # Generation Tab - Original Style
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
                                type="filepath",
                                height=600,
                                elem_id="output-image"
                            )
                        
                        # Right column: Parameters (1/3 of space)
                        with gr.Column(scale=1):
                            
                            # Action buttons at top
                            with gr.Row():
                                generate_btn = gr.Button(
                                    "Generate Image",
                                    variant="primary",
                                    scale=3,
                                    elem_id="generate-btn",
                                    icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/rocket/default/24px.svg"
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
                                    seed = gr.Number(
                                        label="Seed (-1 = random)",
                                        value=-1,
                                        precision=0
                                    )
                            
                            # Status
                            status_text = gr.Textbox(
                                label="Status",
                                value="Ready - Model will load on first generation",
                                interactive=False
                            )
                
                # Image to Image Tab
                with gr.Tab("Image to Image", id="img2img"):
                    with gr.Row():
                        # Left column
                        with gr.Column(scale=2):
                            init_image = gr.Image(
                                label="Input Image",
                                type="filepath",
                                height=400
                            )
                            prompt_img2img = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe what you want...",
                                lines=3
                            )
                            output_image_img2img = gr.Image(
                                label="Generated Image",
                                type="filepath",
                                height=500
                            )
                        
                        # Right column
                        with gr.Column(scale=1):
                            with gr.Row():
                                generate_btn_img2img = gr.Button(
                                    "Generate",
                                    variant="primary",
                                    elem_id="generate-btn",
                                    icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/rocket/default/24px.svg"
                                )
                            
                            with gr.Group(elem_classes="parameters-group"):
                                strength = gr.Slider(
                                    label="Denoising Strength",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                    value=0.75,
                                    info="0.75 recommended"
                                )
                                num_steps_img2img = gr.Slider(
                                    label="Steps",
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=9
                                )
                                seed_img2img = gr.Number(
                                    label="Seed (-1 = random)",
                                    value=-1,
                                    precision=0
                                )
                
                # Model Tab
                with gr.Tab("Model", id="model"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Model Settings")
                            dtype_dropdown = gr.Dropdown(
                                label="Data Type",
                                choices=["bfloat16", "float16", "float32"],
                                value="bfloat16"
                            )
                            clear_cache_btn = gr.Button("Clear Cache")
                            cache_status = gr.Textbox(
                                label="Cache Status",
                                interactive=False,
                                value="Ready"
                            )
            
            # Event handlers - Text to Image
            generate_btn.click(
                fn=self.generate_txt2img,
                inputs=[prompt, width, height, num_steps, seed],
                outputs=[output_image, status_text]
            )
            
            clear_btn.click(
                fn=lambda: [None, "", 1024, 1024, 9, -1],
                outputs=[output_image, prompt, width, height, num_steps, seed]
            )
            
            # Event handlers - Image to Image
            generate_btn_img2img.click(
                fn=self.generate_img2img,
                inputs=[prompt_img2img, init_image, strength, num_steps_img2img, seed_img2img],
                outputs=[output_image_img2img, status_text]
            )
            
            # Event handlers - Model
            clear_cache_btn.click(
                fn=self.clear_cache,
                outputs=[cache_status]
            )
        
        return app
    
    def generate_txt2img(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
    ) -> Tuple[Optional[str], str]:
        """Generate image from text."""
        if not prompt or prompt.strip() == "":
            return None, "Please enter a prompt"
        
        try:
            import time
            start_time = time.time()
            
            # Build workflow
            workflow = create_txt2img_workflow(
                prompt=prompt,
                negative_prompt="",
                width=width,
                height=height,
                steps=steps,
                seed=seed,
            )
            
            # Execute
            result = self.engine.execute(workflow)
            
            if result.error:
                return None, f"Error: {result.error}"
            
            # Get output
            for node_id, output in result.outputs.items():
                if isinstance(output, tuple) and len(output) > 0:
                    filepath = output[0]
                    if isinstance(filepath, str) and os.path.exists(filepath):
                        elapsed = time.time() - start_time
                        return filepath, f"Generated in {elapsed:.1f}s"
            
            return None, "No output generated"
            
        except Exception as e:
            import traceback
            print(f"Error: {traceback.format_exc()}")
            return None, f"Error: {str(e)}"
    
    def generate_img2img(
        self,
        prompt: str,
        init_image: Optional[str],
        strength: float,
        steps: int,
        seed: int,
    ) -> Tuple[Optional[str], str]:
        """Generate image from image."""
        if not init_image:
            return None, "Please provide an input image"
        if not prompt:
            return None, "Please enter a prompt"
        
        try:
            import time
            start_time = time.time()
            
            # Build workflow
            workflow = create_img2img_workflow(
                prompt=prompt,
                negative_prompt="",
                image_path=init_image,
                strength=strength,
                steps=steps,
                seed=seed,
            )
            
            # Execute
            result = self.engine.execute(workflow)
            
            if result.error:
                return None, f"Error: {result.error}"
            
            # Get output
            for node_id, output in result.outputs.items():
                if isinstance(output, tuple) and len(output) > 0:
                    filepath = output[0]
                    if isinstance(filepath, str) and os.path.exists(filepath):
                        elapsed = time.time() - start_time
                        return filepath, f"Generated in {elapsed:.1f}s"
            
            return None, "No output generated"
            
        except Exception as e:
            import traceback
            print(f"Error: {traceback.format_exc()}")
            return None, f"Error: {str(e)}"
    
    def clear_cache(self) -> str:
        """Clear execution cache."""
        self.engine.clear_cache()
        return "Cache cleared"


def create_app() -> gr.Blocks:
    """Create and return the Gradio app."""
    ui = ZImageUI()
    return ui.create_ui()
