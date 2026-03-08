"""
Nodes using Diffusers pipeline for actual generation
These nodes provide real image generation using the diffusers library
"""

from typing import Tuple, Dict, Any, Optional
import os
from datetime import datetime

from .base import ZNode, ZType, register_node


@register_node("GenerateImage", "generation")
class GenerateImage(ZNode):
    """
    Complete image generation using diffusers pipeline.
    This node handles the full generation pipeline internally.
    """
    
    DESCRIPTION = "Generates image using Z-Image-Turbo pipeline"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (ZType.STRING, {"multiline": True, "default": ""}),
                "width": (ZType.INT, {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": (ZType.INT, {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "num_steps": (ZType.INT, {"default": 9, "min": 1, "max": 50}),
                "guidance_scale": (ZType.FLOAT, {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed": (ZType.INT, {"default": -1}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "attention": (["sdpa", "flash_attention_2"], {"default": "sdpa"}),
            },
            "optional": {
                "negative_prompt": (ZType.STRING, {"multiline": True, "default": ""}),
                "enable_cpu_offload": (ZType.BOOLEAN, {"default": False}),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, prompt: str, width: int, height: int, num_steps: int,
                guidance_scale: float, seed: int, dtype: str, attention: str,
                negative_prompt: str = "", enable_cpu_offload: bool = False) -> Tuple[Any]:
        """Generate image using diffusers pipeline."""
        import torch
        from diffusers import AutoPipelineForText2Image
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # Set dtype
        if dtype == "bfloat16" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        print(f"Loading pipeline on {device} with {dtype}...")
        
        # Load pipeline
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        
        # Apply attention backend
        if attention == "flash_attention_2":
            try:
                from diffusers.models.attention_processor import FlashAttnProcessor2_0
                pipeline.unet.set_attn_processor(FlashAttnProcessor2_0())
            except ImportError:
                print("Flash Attention 2 not available")
        
        # Handle device
        if enable_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"Generating with seed {seed}, steps {num_steps}...")
        
        # Generate
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        image = result.images[0]
        
        # Convert PIL to tensor [1, H, W, C]
        import numpy as np
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return (image_tensor,)


@register_node("GenerateImageToImage", "generation")
class GenerateImageToImage(ZNode):
    """
    Image-to-image generation using diffusers pipeline.
    """
    
    DESCRIPTION = "Image-to-image generation using Z-Image-Turbo"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (ZType.STRING, {"multiline": True, "default": ""}),
                "init_image": (ZType.IMAGE,),
                "strength": (ZType.FLOAT, {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_steps": (ZType.INT, {"default": 9, "min": 1, "max": 50}),
                "seed": (ZType.INT, {"default": -1}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "negative_prompt": (ZType.STRING, {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, prompt: str, init_image: Any, strength: float, 
                num_steps: int, seed: int, dtype: str,
                negative_prompt: str = "") -> Tuple[Any]:
        """Generate image from image."""
        import torch
        import numpy as np
        from PIL import Image
        from diffusers import AutoPipelineForImage2Image
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set dtype
        torch_dtype = getattr(torch, dtype) if dtype != "bfloat16" or device == "cuda" else torch.float32
        
        # Convert tensor to PIL
        if init_image.dim() == 4:
            init_image = init_image.squeeze(0)
        if init_image.shape[-1] == 3:
            init_image = init_image.numpy()
        else:
            init_image = init_image.numpy()
        
        init_image = (init_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(init_image)
        
        # Load pipeline
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch_dtype,
            use_safetensors=True,
        ).to(device)
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image=pil_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=0.0,  # Turbo uses 0
                generator=generator,
            )
        
        image = result.images[0]
        
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return (image_tensor,)


@register_node("SaveImageTensor", "image")
class SaveImageTensor(ZNode):
    """Save image tensor to disk."""
    
    DESCRIPTION = "Saves image tensor to disk"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (ZType.IMAGE,),
                "filename_prefix": (ZType.STRING, {"default": "zimage"}),
            }
        }
    
    RETURN_TYPES = (ZType.STRING,)
    RETURN_NAMES = ("filepath",)
    
    def execute(self, image: Any, filename_prefix: str) -> Tuple[str]:
        """Save image to disk."""
        import torch
        import numpy as np
        from PIL import Image
        
        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle tensor
        if torch.is_tensor(image):
            if image.dim() == 4:
                image = image.squeeze(0)
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image
        
        pil_image = Image.fromarray(image_np)
        
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        pil_image.save(filepath)
        
        print(f"Saved: {filepath}")
        
        return (filepath,)
