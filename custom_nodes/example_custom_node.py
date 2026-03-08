"""
Example custom node for Z-Image-Turbo

This demonstrates how to create custom nodes.
Place this file in custom_nodes/ directory.
"""

import torch
from zimage.nodes import ZNode, ZType, register_node


@register_node("ExampleFilter", "custom")
class ExampleFilter(ZNode):
    """
    Example custom node that applies a simple filter to images.
    
    This is a template for creating your own nodes.
    """
    
    DESCRIPTION = "Applies a simple brightness/contrast filter"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (ZType.IMAGE,),
                "brightness": (ZType.FLOAT, {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0,
                    "step": 0.1
                }),
                "contrast": (ZType.FLOAT, {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    RETURN_NAMES = ("filtered_image",)
    
    def execute(self, image: torch.Tensor, brightness: float, contrast: float):
        """
        Apply brightness and contrast adjustments.
        
        Args:
            image: Input image tensor [B, H, W, C] in range [0, 1]
            brightness: Brightness multiplier
            contrast: Contrast multiplier
        
        Returns:
            Filtered image tensor
        """
        # Apply contrast
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        image = (image - mean) * contrast + mean
        
        # Apply brightness
        image = image * brightness
        
        # Clamp to valid range
        image = image.clamp(0, 1)
        
        return (image,)


@register_node("ExampleDebug", "custom")
class ExampleDebug(ZNode):
    """
    Debug node that prints image information.
    """
    
    DESCRIPTION = "Prints debug information about the image"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (ZType.IMAGE,),
            }
        }
    
    RETURN_TYPES = ()
    
    def execute(self, image: torch.Tensor):
        """Print image info."""
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
        print(f"Image mean: {image.mean():.4f}")
        print(f"Image std: {image.std():.4f}")
        
        return {"ui": {"message": f"Shape: {image.shape}, Range: [{image.min():.4f}, {image.max():.4f}]"}}
