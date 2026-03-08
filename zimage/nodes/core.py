"""
Core nodes for Z-Image-Turbo
Includes model loading, text encoding, sampling, and image I/O
"""

from typing import Tuple, Dict, Any, TYPE_CHECKING
import os
from datetime import datetime

from .base import ZNode, ZType, register_node

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image


@register_node("LoadZImageTurbo", "loaders")
class LoadZImageTurbo(ZNode):
    """Load Z-Image-Turbo model pipeline."""
    
    DESCRIPTION = "Loads the Z-Image-Turbo diffusion model"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "attention": (["sdpa", "flash_attention_2"], {"default": "sdpa"}),
            },
            "optional": {
                "enable_cpu_offload": (ZType.BOOLEAN, {"default": False}),
                "compile_model": (ZType.BOOLEAN, {"default": False}),
            }
        }
    
    RETURN_TYPES = (ZType.MODEL, ZType.CLIP, ZType.VAE)
    RETURN_NAMES = ("model", "clip", "vae")
    
    def execute(self, dtype: str, attention: str, enable_cpu_offload: bool = False,
                compile_model: bool = False) -> Tuple[Any, Any, Any]:
        """Load Z-Image-Turbo and return components."""
        import torch
        from diffusers import AutoPipelineForText2Image
        from ..core.model_manager import get_model_manager
        
        model_manager = get_model_manager()
        model_key = f"zimage_turbo_{dtype}_{attention}_{enable_cpu_offload}_{compile_model}"
        
        def load_pipeline():
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=getattr(torch, dtype),
                use_safetensors=True,
            )
            
            # Apply attention backend
            if attention == "flash_attention_2":
                try:
                    from diffusers.models.attention_processor import FlashAttnProcessor2_0
                    pipeline.unet.set_attn_processor(FlashAttnProcessor2_0())
                except ImportError:
                    print("Flash Attention 2 not available, using default")
            
            # Compile for performance
            if compile_model and hasattr(torch, 'compile'):
                try:
                    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
                except Exception as e:
                    print(f"Could not compile model: {e}")
            
            # Handle device placement
            if enable_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline = pipeline.to(model_manager.device)
            
            return pipeline
        
        pipeline = model_manager.load_model(model_key, load_pipeline)
        
        # Return components (unet, text_encoder, vae)
        return (pipeline.unet, pipeline.text_encoder, pipeline.vae)


@register_node("CLIPTextEncode", "conditioning")
class CLIPTextEncode(ZNode):
    """Encode text prompts using CLIP."""
    
    DESCRIPTION = "Encodes text prompts into conditioning tensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (ZType.STRING, {"multiline": True, "default": ""}),
                "clip": (ZType.CLIP,),
            }
        }
    
    RETURN_TYPES = (ZType.CONDITIONING,)
    
    def execute(self, text: str, clip: Any) -> Tuple[Any]:
        """Encode text using CLIP."""
        import torch
        
        if text is None or text.strip() == "":
            text = ""
        
        # Get tokenizer and text encoder from clip
        tokenizer = clip.tokenizer if hasattr(clip, 'tokenizer') else None
        text_encoder = clip if hasattr(clip, 'text_model') else None
        
        if tokenizer is None or text_encoder is None:
            # Fallback for diffusers pipeline
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = clip
        
        # Tokenize
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = text_encoder(
                tokens.input_ids.to(text_encoder.device)
            )[0]
        
        return (text_embeddings,)


@register_node("EmptyLatentImage", "latent")
class EmptyLatentImage(ZNode):
    """Create empty latent image for generation."""
    
    DESCRIPTION = "Creates an empty latent image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (ZType.INT, {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": (ZType.INT, {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "batch_size": (ZType.INT, {"default": 1, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = (ZType.LATENT,)
    
    def execute(self, width: int, height: int, batch_size: int) -> Tuple[Dict]:
        """Create empty latent."""
        import torch
        
        latent_width = width // 8
        latent_height = height // 8
        
        latent = torch.zeros(
            batch_size, 4, latent_height, latent_width,
            dtype=torch.float32
        )
        
        return ({"samples": latent},)


@register_node("KSampler", "sampling")
class KSampler(ZNode):
    """Run diffusion sampling."""
    
    DESCRIPTION = "Runs the diffusion sampling process"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (ZType.MODEL,),
                "positive": (ZType.CONDITIONING,),
                "negative": (ZType.CONDITIONING,),
                "latent_image": (ZType.LATENT,),
                "seed": (ZType.INT, {"default": -1}),
                "steps": (ZType.INT, {"default": 9, "min": 1, "max": 50}),
                "cfg": (ZType.FLOAT, {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "sampler_name": (["euler", "dpmpp_2m", "ddim"], {"default": "euler"}),
                "scheduler": (["simple", "karras", "normal"], {"default": "simple"}),
            },
            "optional": {
                "denoise": (ZType.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = (ZType.LATENT,)
    
    def execute(self, model: Any, positive: Any, negative: Any,
                latent_image: Dict, seed: int, steps: int, cfg: float,
                sampler_name: str, scheduler: str, denoise: float = 1.0) -> Tuple[Dict]:
        """Run sampling."""
        import torch
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        
        # Get latent
        latent = latent_image["samples"].to(model.device)
        
        # Add noise based on denoise strength (for img2img)
        if denoise < 1.0:
            noise = torch.randn_like(latent)
            latent = latent * (1 - denoise) + noise * denoise
        else:
            noise = torch.randn_like(latent)
            latent = noise
        
        # Simplified sampling
        # In a full implementation, this would use actual sampling
        return ({"samples": latent},)


@register_node("VAEDecode", "latent")
class VAEDecode(ZNode):
    """Decode latent to image."""
    
    DESCRIPTION = "Decodes latent to image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (ZType.LATENT,),
                "vae": (ZType.VAE,),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, samples: Dict, vae: Any) -> Tuple[Any]:
        """Decode latent to image."""
        import torch
        
        latent = samples["samples"]
        
        # Move to VAE device
        latent = latent.to(vae.device)
        
        with torch.no_grad():
            # Decode
            images = vae.decode(latent / vae.config.scaling_factor).sample
        
        # Convert to [B, H, W, C] format and normalize to [0, 1]
        images = images.permute(0, 2, 3, 1)
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return (images,)


@register_node("VAEEncode", "latent")
class VAEEncode(ZNode):
    """Encode image to latent."""
    
    DESCRIPTION = "Encodes image to latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": (ZType.IMAGE,),
                "vae": (ZType.VAE,),
            }
        }
    
    RETURN_TYPES = (ZType.LATENT,)
    
    def execute(self, pixels: Any, vae: Any) -> Tuple[Dict]:
        """Encode image to latent."""
        import torch
        
        # Convert from [B, H, W, C] to [B, C, H, W]
        if pixels.dim() == 4 and pixels.shape[-1] == 3:
            pixels = pixels.permute(0, 3, 1, 2)
        
        # Normalize from [0, 1] to [-1, 1]
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.to(vae.device)
        
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
        
        return ({"samples": latent},)


@register_node("SaveImage", "image")
class SaveImage(ZNode):
    """Save image to disk."""
    
    DESCRIPTION = "Saves image to disk"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (ZType.IMAGE,),
                "filename_prefix": (ZType.STRING, {"default": "zimage"}),
            }
        }
    
    RETURN_TYPES = ()
    
    def execute(self, images: Any, filename_prefix: str) -> Dict:
        """Save images to disk."""
        import numpy as np
        from PIL import Image
        
        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image_tensor in enumerate(images):
            # Convert tensor to PIL
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Generate filename
            filename = f"{filename_prefix}_{timestamp}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save
            pil_image.save(filepath)
            saved_files.append(filepath)
        
        return {"ui": {"images": saved_files}}


@register_node("LoadImage", "image")
class LoadImage(ZNode):
    """Load image from disk."""
    
    DESCRIPTION = "Loads image from disk"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": (ZType.STRING, {"default": ""}),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE, ZType.MASK)
    RETURN_NAMES = ("image", "mask")
    
    def execute(self, image_path: str) -> Tuple[Any, Any]:
        """Load image from disk."""
        import torch
        import numpy as np
        from PIL import Image
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        pil_image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to tensor [H, W, C]
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Add batch dim
        
        # Create empty mask (all ones)
        mask = torch.ones((1, image_np.shape[0], image_np.shape[1]))
        
        return (image_tensor, mask)


@register_node("ImageScale", "image")
class ImageScale(ZNode):
    """Scale image to target size."""
    
    DESCRIPTION = "Scales image to target size"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (ZType.IMAGE,),
                "width": (ZType.INT, {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "height": (ZType.INT, {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "mode": (["bilinear", "nearest", "bicubic"], {"default": "bilinear"}),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, image: Any, width: int, height: int, mode: str) -> Tuple[Any]:
        """Scale image."""
        import torch
        
        # image is [B, H, W, C]
        # Need to convert to [B, C, H, W] for interpolation
        image = image.permute(0, 3, 1, 2)
        
        mode_map = {
            "bilinear": "bilinear",
            "nearest": "nearest",
            "bicubic": "bicubic",
        }
        
        scaled = torch.nn.functional.interpolate(
            image, size=(height, width), mode=mode_map.get(mode, "bilinear")
        )
        
        # Convert back to [B, H, W, C]
        scaled = scaled.permute(0, 2, 3, 1)
        
        return (scaled,)


@register_node("ImageToImage", "image")
class ImageToImage(ZNode):
    """Image-to-image generation node."""
    
    DESCRIPTION = "Image-to-image generation using Z-Image-Turbo"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (ZType.MODEL,),
                "vae": (ZType.VAE,),
                "positive": (ZType.CONDITIONING,),
                "negative": (ZType.CONDITIONING,),
                "init_image": (ZType.IMAGE,),
                "seed": (ZType.INT, {"default": -1}),
                "steps": (ZType.INT, {"default": 9, "min": 1, "max": 50}),
                "cfg": (ZType.FLOAT, {"default": 0.0, "min": 0.0, "max": 20.0}),
                "denoise": (ZType.FLOAT, {"default": 0.75, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, model, vae, positive, negative, init_image, seed, steps, cfg, denoise):
        """Run image-to-image generation."""
        import torch
        
        # Encode image to latent
        encode_node = VAEEncode()
        latent_dict = encode_node.execute(init_image, vae)[0]
        
        # Add noise based on denoise strength
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        
        latent = latent_dict["samples"]
        noise = torch.randn_like(latent)
        latent = latent * (1 - denoise) + noise * denoise
        
        # Note: Full img2img would require full pipeline integration
        # For now, return the noised latent decoded back
        
        decode_node = VAEDecode()
        result = decode_node.execute({"samples": latent}, vae)
        
        return result
