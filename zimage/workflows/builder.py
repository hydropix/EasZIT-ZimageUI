"""
Workflow builder for creating node graphs from simple parameters
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class WorkflowBuilder:
    """
    Builds workflows from simple parameters.
    Creates the JSON structure expected by the execution engine.
    """
    
    def __init__(self):
        self.nodes = {}
        self.node_counter = 0
    
    def _next_id(self) -> str:
        """Generate next node ID."""
        self.node_counter += 1
        return str(self.node_counter)
    
    def add_node(self, class_type: str, inputs: Dict[str, Any], 
                 is_output: bool = False) -> str:
        """Add a node to the workflow."""
        node_id = self._next_id()
        self.nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "is_output": is_output,
        }
        return node_id
    
    def build(self) -> Dict[str, Dict]:
        """Build and return the workflow."""
        return self.nodes.copy()
    
    def save(self, filepath: str):
        """Save workflow to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.nodes, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Dict]:
        """Load workflow from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def clear(self):
        """Clear all nodes."""
        self.nodes.clear()
        self.node_counter = 0


def create_txt2img_workflow(
    prompt: str,
    negative_prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    seed: int = -1,
    cfg: float = 0.0,
    dtype: str = "bfloat16",
    attention: str = "sdpa",
    filename_prefix: str = "zimage"
) -> Dict[str, Dict]:
    """
    Create a text-to-image workflow using GenerateImage node.
    
    Returns workflow dict ready for execution.
    """
    builder = WorkflowBuilder()
    
    # Generate image using diffusers pipeline
    generate = builder.add_node("GenerateImage", {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_steps": steps,
        "guidance_scale": cfg,
        "seed": seed,
        "dtype": dtype,
        "attention": attention,
        "enable_cpu_offload": False,
    })
    
    # Save the result
    builder.add_node("SaveImageTensor", {
        "image": [generate, 0],
        "filename_prefix": filename_prefix,
    }, is_output=True)
    
    return builder.build()


def create_img2img_workflow(
    prompt: str,
    negative_prompt: str,
    image_path: str,
    strength: float = 0.75,
    steps: int = 9,
    seed: int = -1,
    dtype: str = "bfloat16",
    filename_prefix: str = "zimage_img2img"
) -> Dict[str, Dict]:
    """
    Create an image-to-image workflow.
    """
    builder = WorkflowBuilder()
    
    # Load input image
    load_image = builder.add_node("LoadImage", {
        "image_path": image_path,
    })
    
    # Generate from image
    generate = builder.add_node("GenerateImageToImage", {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_image": [load_image, 0],
        "strength": strength,
        "num_steps": steps,
        "seed": seed,
        "dtype": dtype,
    })
    
    # Save result
    builder.add_node("SaveImageTensor", {
        "image": [generate, 0],
        "filename_prefix": filename_prefix,
    }, is_output=True)
    
    return builder.build()


def create_inpainting_workflow(
    prompt: str,
    negative_prompt: str,
    image_path: str,
    mask_path: str,
    strength: float = 1.0,
    steps: int = 9,
    seed: int = -1,
    dtype: str = "bfloat16",
    filename_prefix: str = "zimage_inpaint"
) -> Dict[str, Dict]:
    """
    Create an inpainting workflow.
    Note: For now uses img2img - full inpainting needs specialized support.
    """
    # For now, reuse img2img workflow
    return create_img2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        strength=strength,
        steps=steps,
        seed=seed,
        dtype=dtype,
        filename_prefix=filename_prefix
    )
