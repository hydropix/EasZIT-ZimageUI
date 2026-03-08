# Z-Image-Turbo Adaptation Plan

## Executive Summary

This document outlines a strategic plan to integrate key ComfyUI architectural concepts into the Z-Image-Turbo project. The goal is to enhance Z-Image-Turbo with ComfyUI's flexibility while maintaining simplicity and focusing on Z-Image-Turbo's core strengths.

---

## Current State Analysis

### Z-Image-Turbo Architecture (Current)

```
┌─────────────────────────────────────────────────────────────┐
│                CURRENT Z-IMAGE-TURBO                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Gradio UI                                                   │
│       │                                                      │
│       ▼                                                      │
│  Direct Pipeline (app.py)                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Load Model (Z-Image-Turbo from HuggingFace)    │   │
│  │  2. Encode Prompt (CLIP)                           │   │
│  │  3. Generate (Diffusers Pipeline)                  │   │
│  │  4. Decode & Save                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                      │
│       ▼                                                      │
│  Output Image                                                │
│                                                              │
│  Features:                                                   │
│  ✓ Simple, fast setup                                       │
│  ✓ Optimized for Z-Image-Turbo                              │
│  ✓ Gradio-based web UI                                      │
│  ✗ Limited extensibility                                    │
│  ✗ No node-based workflow                                   │
│  ✗ Fixed pipeline                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              TARGET Z-IMAGE-TURBO WITH COMFYUI INSPIRATION          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 1: User Interface (Gradio + Future Node Editor)              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Simple Mode: Current Gradio UI (default)                │    │
│  │  • Advanced Mode: Optional node-based workflow editor      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  LAYER 2: Execution Engine (New)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Graph topology & dependency resolution                  │    │
│  │  • Node execution orchestration                            │    │
│  │  • Result caching for iterative workflows                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  LAYER 3: Node Library (New)                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Core Nodes:                                                │    │
│  │    • Load Model (Z-Image-Turbo, SDXL, etc.)                │    │
│  │    • CLIP Text Encode                                      │    │
│  │    • KSampler (with multiple samplers)                     │    │
│  │    • VAE Encode/Decode                                     │    │
│  │    • Image I/O                                             │    │
│  │                                                             │    │
│  │  Enhancement Nodes:                                         │    │
│  │    • LoRA Loader                                           │    │
│  │    • Image to Image                                        │    │
│  │    • Inpainting                                            │    │
│  │    • ControlNet (future)                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  LAYER 4: Core Pipeline (Enhanced)                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Diffusers integration                                   │    │
│  │  • Memory management (VRAM optimization)                   │    │
│  │  • Model patching system                                   │    │
│  │  • Sampling algorithms                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Establish core infrastructure

#### Tasks

1. **Create Node System**
   ```
   zimage/
   ├── nodes/
   │   ├── __init__.py
   │   ├── base.py           # Base node class
   │   ├── registry.py       # Node registration
   │   └── types.py          # Type definitions
   ```

2. **Implement Base Node Class**
   ```python
   class ZNode:
       """Base class for Z-Image-Turbo nodes."""
       
       @classmethod
       def INPUT_TYPES(cls) -> Dict:
           return {"required": {}}
       
       RETURN_TYPES = ()
       FUNCTION = "execute"
       CATEGORY = "zimage"
       
       def execute(self, **kwargs):
           raise NotImplementedError
   ```

3. **Create Type System**
   ```python
   class ZType(Enum):
       IMAGE = "IMAGE"
       LATENT = "LATENT"
       MODEL = "MODEL"
       CLIP = "CLIP"
       VAE = "VAE"
       STRING = "STRING"
       INT = "INT"
       FLOAT = "FLOAT"
       MASK = "MASK"
   ```

4. **Simple Execution Engine**
   ```python
   class SimpleExecutor:
       """Lightweight execution engine."""
       
       def execute(self, workflow: Dict) -> Dict:
           # 1. Topological sort
           # 2. Execute nodes in order
           # 3. Cache results
           pass
   ```

**Deliverables**:
- Node base class and type system
- Simple execution engine
- 3-5 basic nodes (LoadModel, CLIPEncode, Sampler, VAEDecode, SaveImage)

---

### Phase 2: Core Features (Weeks 3-4)

**Goal**: Implement essential ComfyUI-inspired features

#### Tasks

1. **Enhanced Memory Management**
   ```python
   class MemoryManager:
       """Adapted from ComfyUI's model_management."""
       
       def __init__(self):
           self.device = self.detect_device()
           self.vram_state = self.detect_vram_state()
           self.loaded_models = {}
       
       def load_model(self, model_id, factory):
           # LRU cache with automatic offloading
           pass
       
       def get_free_memory(self):
           # Platform-specific memory query
           pass
   ```

2. **Model Patching System**
   ```python
   class ModelPatcher:
       """Enable LoRA and other modifications."""
       
       def __init__(self, model, load_device, offload_device):
           self.model = model
           self.patches = {}
           self.backup = {}
       
       def add_patches(self, patches: Dict):
           # Add weight modifications
           pass
       
       def patch_model(self):
           # Apply patches
           pass
   ```

3. **Caching System**
   ```python
   class ResultCache:
       """Simple LRU cache for node outputs."""
       
       def __init__(self, max_size=50):
           self.cache = OrderedDict()
           self.max_size = max_size
       
       def get(self, key):
           # LRU get
           pass
       
       def set(self, key, value):
           # LRU set with size limit
           pass
   ```

4. **Sampling Enhancements**
   - Multiple samplers (Euler, DPM++, DDIM)
   - Multiple schedulers (Simple, Karras, Beta)
   - CFG guidance

**Deliverables**:
- Memory management with auto-offloading
- LoRA support
- Result caching
- Multiple samplers/schedulers

---

### Phase 3: Advanced Features (Weeks 5-6)

**Goal**: Add image-to-image, inpainting, and workflow features

#### Tasks

1. **Image-to-Image**
   ```python
   class ImageToImageNode(ZNode):
       """Img2img workflow node."""
       
       @classmethod
       def INPUT_TYPES(cls):
           return {
               "required": {
                   "model": (ZType.MODEL,),
                   "image": (ZType.IMAGE,),
                   "prompt": (ZType.STRING,),
                   "strength": (ZType.FLOAT, {"default": 0.75, "min": 0.0, "max": 1.0}),
               }
           }
       
       RETURN_TYPES = (ZType.IMAGE,)
   ```

2. **Inpainting Support**
   - Mask loading and processing
   - Inpaint model support
   - Masked sampling

3. **Workflow System**
   - Save/load workflows as JSON
   - Workflow validation
   - Preset workflows

4. **Extension API**
   ```python
   class Extension:
       """Base class for extensions."""
       
       def register_nodes(self):
           return []
       
       def register_samplers(self):
           return {}
   ```

**Deliverables**:
- Image-to-image support
- Basic inpainting
- Workflow save/load
- Extension API

---

### Phase 4: UI Enhancement (Weeks 7-8)

**Goal**: Improved user experience with optional advanced mode

#### Tasks

1. **Dual-Mode Interface**
   - Simple Mode: Current Gradio UI
   - Advanced Mode: Node workflow editor

2. **Workflow Visualization**
   - Graph view of node connections
   - Real-time execution preview
   - Progress indicators

3. **Template System**
   - Pre-built workflow templates
   - Community template sharing

**Deliverables**:
- Dual-mode UI
- Workflow visualization
- Template library

---

## Detailed Implementation

### 1. Node System

```python
# zimage/nodes/base.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
from enum import Enum
import torch

class ZType(str, Enum):
    """Node data types."""
    IMAGE = "IMAGE"
    LATENT = "LATENT"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    MASK = "MASK"
    SAMPLER = "SAMPLER"
    SCHEDULER = "SCHEDULER"
    ANY = "*"


class ZNode(ABC):
    """Base class for all Z-Image-Turbo nodes."""
    
    # Node metadata
    NAME: str = ""
    CATEGORY: str = "zimage"
    DESCRIPTION: str = ""
    
    # Input/output definition
    INPUT_TYPES: Dict = {"required": {}}
    RETURN_TYPES: Tuple[ZType, ...] = ()
    RETURN_NAMES: Tuple[str, ...] = ()
    
    # Execution
    FUNCTION: str = "execute"
    OUTPUT_NODE: bool = False
    
    def __init__(self):
        self.id = None
        self.inputs = {}
        self.outputs = {}
    
    @abstractmethod
    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """Execute node logic."""
        pass
    
    def validate_inputs(self, inputs: Dict) -> Optional[str]:
        """Validate inputs before execution. Return error message or None."""
        return None
    
    def is_changed(self, **kwargs) -> Any:
        """Return value that changes when node should re-execute."""
        return False


# Decorator for easy node registration
def register_node(name: str = None, category: str = "zimage"):
    """Decorator to register a node class."""
    def decorator(cls):
        cls.NAME = name or cls.__name__
        cls.CATEGORY = category
        NodeRegistry.register(cls.NAME, cls)
        return cls
    return decorator


class NodeRegistry:
    """Global node registry."""
    _nodes: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, node_class: type):
        cls._nodes[name] = node_class
    
    @classmethod
    def get(cls, name: str) -> type:
        return cls._nodes.get(name)
    
    @classmethod
    def list_nodes(cls) -> Dict[str, type]:
        return cls._nodes.copy()
```

### 2. Execution Engine

```python
# zimage/engine/executor.py

from typing import Dict, List, Any, Optional
from collections import OrderedDict, deque
import torch
import time

class ExecutionResult:
    def __init__(self, outputs: Dict[str, Any], execution_time: float):
        self.outputs = outputs
        self.execution_time = execution_time


class SimpleExecutor:
    """Simplified execution engine for Z-Image-Turbo."""
    
    def __init__(self, cache_enabled: bool = True, max_cache_size: int = 50):
        self.cache_enabled = cache_enabled
        self.cache = OrderedDict() if cache_enabled else None
        self.max_cache_size = max_cache_size
        self.execution_history = []
    
    def execute(self, workflow: Dict[str, Dict], 
                progress_callback=None) -> ExecutionResult:
        """
        Execute a workflow.
        
        Args:
            workflow: Dict of node_id -> node_config
            progress_callback: Function(node_id, progress) for updates
        
        Returns:
            ExecutionResult with outputs and timing
        """
        start_time = time.time()
        
        # 1. Topological sort to determine execution order
        execution_order = self._topological_sort(workflow)
        
        # 2. Execute nodes
        results = {}
        for i, node_id in enumerate(execution_order):
            node_config = workflow[node_id]
            
            if progress_callback:
                progress_callback(node_id, i / len(execution_order))
            
            # Check cache
            cache_key = self._get_cache_key(node_id, node_config, results)
            if self.cache_enabled and cache_key in self.cache:
                results[node_id] = self.cache[cache_key]
                continue
            
            # Execute node
            result = self._execute_node(node_id, node_config, results)
            results[node_id] = result
            
            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)
        
        execution_time = time.time() - start_time
        
        # Collect output nodes
        outputs = {
            node_id: results[node_id]
            for node_id in workflow
            if workflow[node_id].get("is_output", False)
        }
        
        return ExecutionResult(outputs, execution_time)
    
    def _topological_sort(self, workflow: Dict) -> List[str]:
        """Kahn's algorithm for topological sorting."""
        # Build adjacency list
        in_degree = {node_id: 0 for node_id in workflow}
        dependents = {node_id: [] for node_id in workflow}
        
        for node_id, config in workflow.items():
            for input_name, input_value in config.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    # It's a link [source_node, output_index]
                    source_node = input_value[0]
                    in_degree[node_id] += 1
                    dependents[source_node].append(node_id)
        
        # Find nodes with no dependencies
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for dependent in dependents[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(workflow):
            raise ValueError("Workflow contains cycles")
        
        return result
    
    def _execute_node(self, node_id: str, config: Dict, 
                      available_results: Dict) -> Any:
        """Execute a single node."""
        from .registry import NodeRegistry
        
        node_class = NodeRegistry.get(config["class_type"])
        if node_class is None:
            raise ValueError(f"Unknown node type: {config['class_type']}")
        
        # Resolve inputs
        inputs = {}
        for input_name, input_value in config.get("inputs", {}).items():
            if isinstance(input_value, list) and len(input_value) == 2:
                # Link to another node's output
                source_node, output_idx = input_value
                source_result = available_results[source_node]
                if isinstance(source_result, tuple):
                    inputs[input_name] = source_result[output_idx]
                else:
                    inputs[input_name] = source_result
            else:
                # Constant value
                inputs[input_name] = input_value
        
        # Create and execute node
        node = node_class()
        result = getattr(node, node_class.FUNCTION)(**inputs)
        
        # Ensure result is tuple
        if not isinstance(result, tuple):
            result = (result,)
        
        return result
    
    def _get_cache_key(self, node_id: str, config: Dict, 
                       available_results: Dict) -> str:
        """Generate cache key for node execution."""
        # Include node type and input values
        key_parts = [config["class_type"]]
        
        for input_name in sorted(config.get("inputs", {}).keys()):
            input_value = config["inputs"][input_name]
            if isinstance(input_value, list) and len(input_value) == 2:
                # For links, use source node's cache key
                source_node = input_value[0]
                key_parts.append(f"{input_name}=@{source_node}")
            else:
                key_parts.append(f"{input_name}={input_value}")
        
        return "|".join(key_parts)
    
    def _cache_result(self, key: str, result: Any):
        """Cache execution result with LRU eviction."""
        if key in self.cache:
            # Move to end (most recently used)
            del self.cache[key]
        elif len(self.cache) >= self.max_cache_size:
            # Evict oldest
            self.cache.popitem(last=False)
        
        self.cache[key] = result
    
    def clear_cache(self):
        """Clear execution cache."""
        if self.cache:
            self.cache.clear()
```

### 3. Core Nodes

```python
# zimage/nodes/core.py

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image
import numpy as np
from .base import ZNode, ZType, register_node


@register_node("LoadZImageTurbo", "loaders")
class LoadZImageTurbo(ZNode):
    """Load Z-Image-Turbo model."""
    
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
    
    def __init__(self):
        super().__init__()
        self.loaded_pipeline = None
    
    def execute(self, dtype, attention, enable_cpu_offload=False, 
                compile_model=False):
        from ..core.model_manager import ModelManager
        
        # Use model manager for caching
        model_key = f"zimage_turbo_{dtype}_{attention}"
        
        def load_model():
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=getattr(torch, dtype),
                use_safetensors=True,
            )
            
            if attention == "flash_attention_2":
                pipeline.unet.set_attn_processor(FlashAttnProcessor2_0())
            
            if compile_model:
                pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            
            if enable_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline = pipeline.to("cuda")
            
            return pipeline
        
        pipeline = ModelManager.load_model(model_key, load_model)
        
        # Return components
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
    
    def execute(self, text, clip):
        # Tokenize and encode
        tokens = clip.tokenizer(
            text,
            padding="max_length",
            max_length=clip.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = clip(tokens.input_ids.to(clip.device))[0]
        
        return (text_embeddings,)


@register_node("EmptyLatentImage", "latent")
class EmptyLatentImage(ZNode):
    """Create empty latent image."""
    
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
    
    def execute(self, width, height, batch_size):
        # Calculate latent dimensions (assuming 8x downsampling)
        latent_width = width // 8
        latent_height = height // 8
        
        # Create empty latent
        latent = torch.zeros(
            batch_size, 4, latent_height, latent_width,
            dtype=torch.float32
        )
        
        return ({"samples": latent},)


@register_node("KSampler", "sampling")
class KSampler(ZNode):
    """Run diffusion sampling."""
    
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
    
    def execute(self, model, positive, negative, latent_image, seed, steps, 
                cfg, sampler_name, scheduler, denoise=1.0):
        from ..core.sampling import sample_latents
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        
        # Get latent
        latent = latent_image["samples"].to(model.device)
        
        # Add noise for img2img
        if denoise < 1.0:
            noise = torch.randn_like(latent)
            latent = latent * (1 - denoise) + noise * denoise
        else:
            noise = torch.randn_like(latent)
            latent = noise
        
        # Run sampling
        samples = sample_latents(
            model=model,
            latents=latent,
            positive=positive,
            negative=negative,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        
        return ({"samples": samples},)


@register_node("VAEDecode", "latent")
class VAEDecode(ZNode):
    """Decode latent to image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (ZType.LATENT,),
                "vae": (ZType.VAE,),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, samples, vae):
        latent = samples["samples"]
        
        with torch.no_grad():
            images = vae.decode(latent / vae.config.scaling_factor).sample
        
        # Convert to [B, H, W, C] format
        images = images.permute(0, 2, 3, 1)
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return (images,)


@register_node("SaveImage", "image")
class SaveImage(ZNode):
    """Save image to disk."""
    
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
    
    def execute(self, images, filename_prefix):
        import os
        from datetime import datetime
        
        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for i, image_tensor in enumerate(images):
            # Convert tensor to PIL
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save
            pil_image.save(filepath)
            saved_files.append(filepath)
        
        return {"ui": {"images": saved_files}}
```

### 4. Integration with Existing App

```python
# zimage/gradio_interface.py

import gradio as gr
from .engine.executor import SimpleExecutor
from .nodes.core import *
from .nodes.registry import NodeRegistry

class GradioInterface:
    """Gradio interface using node system internally."""
    
    def __init__(self):
        self.executor = SimpleExecutor(cache_enabled=True)
    
    def create_ui(self):
        with gr.Blocks(title="Z-Image-Turbo") as app:
            gr.Markdown("# Z-Image-Turbo with Node Backend")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt...",
                        lines=3
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt...",
                        lines=2
                    )
                    
                    with gr.Row():
                        width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                        height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                    
                    steps = gr.Slider(1, 50, 9, step=1, label="Steps")
                    seed = gr.Number(-1, label="Seed (-1 for random)")
                    
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Generated Image")
                    info_text = gr.Textbox(label="Info", interactive=False)
            
            generate_btn.click(
                fn=self.generate,
                inputs=[prompt, negative_prompt, width, height, steps, seed],
                outputs=[output_image, info_text]
            )
        
        return app
    
    def generate(self, prompt, negative_prompt, width, height, steps, seed):
        """Generate image using node system."""
        # Build workflow
        workflow = {
            "1": {
                "class_type": "LoadZImageTurbo",
                "inputs": {
                    "dtype": "bfloat16",
                    "attention": "sdpa",
                    "enable_cpu_offload": False,
                    "compile_model": False,
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],  # From node 1, output 1
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": 0.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2],
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "zimage",
                },
                "is_output": True,
            }
        }
        
        # Execute
        result = self.executor.execute(workflow)
        
        # Get output
        output_node = result.outputs.get("7")
        if output_node:
            images = output_node.get("ui", {}).get("images", [])
            if images:
                return images[0], f"Generated in {result.execution_time:.2f}s"
        
        return None, "Generation failed"
```

---

## Migration Strategy

### Step 1: Backend First
- Implement node system and executor
- Keep existing Gradio UI
- Use nodes internally

### Step 2: Feature Parity
- Port all existing features to nodes
- Ensure same performance
- Add new capabilities (LoRA, img2img)

### Step 3: Optional Advanced UI
- Add node editor as optional advanced mode
- Keep simple mode as default
- Allow workflow save/load

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Extensibility | Fixed pipeline | Modular nodes |
| Features | Basic | Advanced (LoRA, img2img, etc.) |
| Customization | Limited | Unlimited workflow combinations |
| Community | None | Can share workflows/nodes |
| Maintenance | Monolithic | Modular, testable components |

---

## Files to Create/Modify

### New Files
```
zimage/
├── nodes/
│   ├── __init__.py
│   ├── base.py
│   ├── core.py
│   ├── image_nodes.py
│   ├── conditioning_nodes.py
│   └── registry.py
├── engine/
│   ├── __init__.py
│   ├── executor.py
│   └── caching.py
├── core/
│   ├── model_manager.py
│   ├── sampling.py
│   └── memory_management.py
├── extensions/
│   └── __init__.py
└── workflows/
    └── templates/
```

### Modified Files
```
app.py                    # Integrate node system
requirements.txt          # Add dependencies
```

---

## Conclusion

This adaptation plan brings ComfyUI's architectural strengths to Z-Image-Turbo while maintaining its simplicity. The phased approach ensures:

1. **Incremental development** with working versions at each phase
2. **Backward compatibility** with existing functionality
3. **Future extensibility** through the node system
4. **Community engagement** through shareable workflows

The result will be a powerful yet accessible image generation tool optimized for Z-Image-Turbo's speed and quality.
