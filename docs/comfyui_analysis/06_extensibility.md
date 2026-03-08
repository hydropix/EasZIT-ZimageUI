# ComfyUI Extensibility System

## Overview

ComfyUI's extensibility is one of its greatest strengths. The system allows third-party developers to add new functionality through custom nodes without modifying core code.

---

## Custom Node Architecture

### Directory Structure

```
custom_nodes/
├── my_custom_node/
│   ├── __init__.py          # Node registration
│   ├── my_node.py           # Node implementation
│   ├── utils.py             # Helper functions
│   └── requirements.txt     # Dependencies
├── another_node_pack/
│   └── ...
└── ComfyUI-Manager/         # Node manager (optional)
```

### Basic Custom Node

```python
# custom_nodes/my_node/__init__.py

from .my_node import MyNode, MyOtherNode

NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
    "MyOtherNode": MyOtherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "My Custom Node",
    "MyOtherNode": "My Other Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

```python
# custom_nodes/my_node/my_node.py

import torch
import numpy as np
from PIL import Image

class MyNode:
    """
    Description of what this node does.
    This appears as a tooltip in the UI.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "process"
    CATEGORY = "custom/category"
    
    def process(self, image, strength, mask=None):
        """Main processing function."""
        # Process image
        output = image * strength
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return (output,)
```

---

## Node Hooks and Callbacks

### Model Patcher Hooks

```python
# Hook into model forward pass
class ModelHookNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_hook"
    
    def apply_hook(self, model):
        # Add wrapper function
        def my_wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]
            
            # Modify inputs before forward pass
            input_x = self.modify_input(input_x)
            
            # Call original model
            output = apply_model(input_x, timestep, **c)
            
            # Modify output
            output = self.modify_output(output)
            
            return output
        
        # Add to model options
        model_options = model.model_options.copy()
        if "model_function_wrapper" not in model_options:
            model_options["model_function_wrapper"] = []
        model_options["model_function_wrapper"].append(my_wrapper)
        
        model.model_options = model_options
        return (model,)
```

### CFG Hooks

```python
# Custom CFG function
class CustomCFGNode:
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_scale": ("FLOAT", {"default": 7.0}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_cfg"
    
    def apply_cfg(self, model, cfg_scale):
        def custom_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            
            # Custom CFG formula
            result = uncond + (cond - uncond) * (cfg_scale ** 0.5)
            
            return result
        
        model_options = model.model_options.copy()
        model_options["sampler_cfg_function"] = custom_cfg
        model.model_options = model_options
        
        return (model,)
```

---

## Advanced Custom Nodes

### Nodes with State

```python
class StatefulNode:
    """Node that maintains state between executions."""
    
    def __init__(self):
        self.cache = {}
        self.counter = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_data": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    
    def process(self, input_data):
        # Use cached state
        if input_data in self.cache:
            return (self.cache[input_data],)
        
        # Process and cache
        result = self.expensive_operation(input_data)
        self.cache[input_data] = result
        self.counter += 1
        
        return (result,)
```

### Nodes with File I/O

```python
import os
import json
import folder_paths

class LoadConfigNode:
    """Load configuration from file."""
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) 
                 if f.endswith('.json')]
        
        return {
            "required": {
                "config_file": (sorted(files),),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("name", "count", "scale")
    FUNCTION = "load_config"
    
    def load_config(self, config_file):
        config_path = folder_paths.get_annotated_filepath(config_file)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return (
            config.get("name", ""),
            config.get("count", 0),
            config.get("scale", 1.0),
        )
    
    @classmethod
    def IS_CHANGED(s, config_file):
        """Invalidate cache when file changes."""
        config_path = folder_paths.get_annotated_filepath(config_file)
        mtime = os.path.getmtime(config_path)
        return float(mtime)
```

### Nodes with WebSocket Output

```python
class ProgressNode:
    """Send progress updates to client."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 10}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    
    def generate(self, steps, prompt=None, extra_pnginfo=None, unique_id=None):
        # Get server reference
        from server import PromptServer
        server = PromptServer.instance
        
        images = []
        for i in range(steps):
            # Do work...
            image = self.generate_step(i)
            images.append(image)
            
            # Send progress
            server.send_sync("progress", {
                "node_id": unique_id,
                "step": i,
                "total": steps,
                "prompt_id": prompt.get("prompt_id") if prompt else None,
            }, server.client_id)
        
        # Return with UI data
        return {
            "ui": {
                "images": [self.tensor_to_image(img) for img in images]
            },
            "result": (torch.stack(images),)
        }
```

---

## Node Packs

### Pack Structure

```python
# custom_nodes/my_node_pack/__init__.py

# Import all nodes
from .image_nodes import *
from .text_nodes import *
from .model_nodes import *

# Combine mappings
NODE_CLASS_MAPPINGS = {
    **image_nodes.NODE_CLASS_MAPPINGS,
    **text_nodes.NODE_CLASS_MAPPINGS,
    **model_nodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **image_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **text_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **model_nodes.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./web"  # Custom JS/CSS
```

### Custom JavaScript

```javascript
// custom_nodes/my_node_pack/web/my_extension.js

import { app } from "../../scripts/app.js";

// Register custom node appearance
app.registerExtension({
    name: "MyNodePack.MyExtension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MyNode") {
            // Customize node appearance
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                // Add custom widget
                this.addWidget("button", "Refresh", null, () => {
                    this.refreshData();
                });
            };
        }
    },
    
    // Add context menu items
    async setup(app) {
        const getExtraMenuOptions = app.canvas.getExtraMenuOptions;
        app.canvas.getExtraMenuOptions = function(_, options) {
            options.push({
                content: "My Custom Action",
                callback: () => {
                    console.log("Custom action triggered!");
                }
            });
            return getExtraMenuOptions?.apply(this, arguments);
        };
    }
});
```

---

## Popular Extension Patterns

### LoRA Loader Pattern

```python
class LoraLoader:
    """Load and apply LoRA weights."""
    
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        # Cache loaded LoRA
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        
        # Apply to model and clip
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        
        return (model_lora, clip_lora)
```

### ControlNet Pattern

```python
class ControlNetApply:
    """Apply ControlNet conditioning."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    
    def apply_controlnet(self, positive, negative, control_net, image, strength):
        if strength == 0:
            return (positive, negative)
        
        # Prepare control hint
        control_hint = image.movedim(-1, 1)  # [B,H,W,C] -> [B,C,H,W]
        
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                
                # Set control hint
                c_net = control_net.copy().set_cond_hint(
                    control_hint, strength
                )
                
                # Chain with previous controlnet if exists
                if 'control' in d:
                    c_net.set_previous_controlnet(d['control'])
                
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        
        return (out[0], out[1])
```

---

## Extension Management

### ComfyUI-Manager

Popular extension manager for ComfyUI:

```bash
# Install manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Features:
# - Install custom nodes from GitHub
# - Update all nodes
# - Install missing nodes from workflow
# - Search node registry
```

### Node Registry

```python
# Register node metadata for discovery
NODE_CLASS_MAPPINGS = {...}
NODE_DISPLAY_NAME_MAPPINGS = {...}

# Optional metadata
NODE_DESCRIPTIONS = {
    "MyNode": "Detailed description for documentation"
}

NODE_TAGS = {
    "MyNode": ["image", "processing", "utility"]
}
```

---

## Best Practices

### DO

- ✅ Use type hints for clarity
- ✅ Provide comprehensive docstrings
- ✅ Implement IS_CHANGED for file inputs
- ✅ Cache expensive operations
- ✅ Handle errors gracefully
- ✅ Follow naming conventions
- ✅ Test with different batch sizes

### DON'T

- ❌ Modify global state
- ❌ Block the main thread
- ❌ Use large hardcoded constants
- ❌ Ignore VRAM constraints
- ❌ Break backward compatibility without deprecation

---

## Z-Image-Turbo Extension Strategy

### Simplified Extension API

```python
# zimage/extensions/base.py

class ZExtension:
    """Base class for Z-Image-Turbo extensions."""
    
    name = "extension_name"
    version = "1.0.0"
    
    def register(self, app):
        """Register extension with the app."""
        pass
    
    def get_nodes(self):
        """Return list of node classes."""
        return []
    
    def get_samplers(self):
        """Return dict of sampler functions."""
        return {}
    
    def get_schedulers(self):
        """Return dict of scheduler functions."""
        return {}

# Example extension
class MyExtension(ZExtension):
    name = "my_extension"
    
    def get_nodes(self):
        return [MyCustomNode, MyOtherNode]
    
    def get_samplers(self):
        return {
            "my_sampler": my_sampler_function
        }

# Register
app.register_extension(MyExtension())
```

### Extension Loading

```python
# zimage/extensions/loader.py

import importlib
import pkgutil
from pathlib import Path

def load_extensions(extension_dir="extensions"):
    """Load all extensions from directory."""
    extensions = []
    
    for finder, name, ispkg in pkgutil.iter_modules([extension_dir]):
        try:
            module = importlib.import_module(f"{extension_dir}.{name}")
            
            if hasattr(module, 'register'):
                module.register(app)
                extensions.append(name)
        except Exception as e:
            logger.error(f"Failed to load extension {name}: {e}")
    
    return extensions
```

---

## Related Files

| File | Purpose |
|------|---------|
| `nodes.py` | Core node definitions (reference) |
| `comfy/comfy_types/` | Type definitions |
| `folder_paths.py` | Path utilities |
| `server.py` | WebSocket server |
