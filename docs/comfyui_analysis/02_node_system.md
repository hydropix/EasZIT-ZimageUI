# ComfyUI Node System Deep Dive

## Overview

The node system is the foundation of ComfyUI's modularity. Each node is a self-contained computational unit that transforms inputs into outputs according to a specific function.

---

## Node Definition Structure

### Basic Node Template

```python
class NodeName(ComfyNodeABC):
    """
    Node description - appears as tooltip in UI.
    Explain what this node does.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # REQUIRED ATTRIBUTES
    # ═══════════════════════════════════════════════════════════════
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        """Define all inputs for this node."""
        return {
            "required": {
                # Inputs that MUST be connected
                "input_name": (IO.TYPE, {
                    "default": default_value,
                    "min": minimum_value,
                    "max": maximum_value,
                }),
            },
            "optional": {
                # Inputs that CAN be connected
                "optional_input": (IO.TYPE, {}),
            },
            "hidden": {
                # Special internal inputs
                "prompt": "PROMPT",           # Full workflow JSON
                "unique_id": "UNIQUE_ID",     # Node instance ID
                "extra_pnginfo": "EXTRA_PNGINFO",  # Metadata
            }
        }
    
    RETURN_TYPES = (IO.TYPE1, IO.TYPE2)  # Output types tuple
    FUNCTION = "execute"                  # Method to call
    CATEGORY = "category/subcategory"     # Menu organization
    
    # ═══════════════════════════════════════════════════════════════
    # OPTIONAL ATTRIBUTES
    # ═══════════════════════════════════════════════════════════════
    
    RETURN_NAMES = ("output1", "output2")  # Custom output names
    OUTPUT_TOOLTIPS = ("Tooltip 1", "Tooltip 2")  # Output tooltips
    
    DESCRIPTION = "Node description for UI tooltip"
    SEARCH_ALIASES = ["search", "terms"]   # Search keywords
    
    OUTPUT_NODE = True      # Is this an output node?
    INPUT_IS_LIST = False   # Does this handle list inputs?
    OUTPUT_IS_LIST = (False, False)  # Which outputs are lists?
    
    EXPERIMENTAL = False    # Mark as experimental
    DEPRECATED = False      # Mark as deprecated
    DEV_ONLY = False        # Only show in dev mode
    
    # ═══════════════════════════════════════════════════════════════
    # EXECUTION METHOD
    # ═══════════════════════════════════════════════════════════════
    
    def execute(self, input_name, optional_input=None, 
                prompt=None, unique_id=None, extra_pnginfo=None):
        """Main execution logic."""
        # Process inputs
        result = self.process(input_name)
        # Return tuple matching RETURN_TYPES
        return (result,)
```

---

## Input Types Reference

### Primitive Types

| Type | Python Type | UI Widget | Use Case |
|------|-------------|-----------|----------|
| `IO.INT` | `int` | Number slider | Steps, seeds, dimensions |
| `IO.FLOAT` | `float` | Float slider | Strengths, scales |
| `IO.STRING` | `str` | Text input | Prompts, filenames |
| `IO.BOOLEAN` | `bool` | Checkbox | Toggles |

### Tensor Types

| Type | PyTorch Type | Description |
|------|--------------|-------------|
| `IO.IMAGE` | `Tensor[B,H,W,C]` | RGB images [0,1] |
| `IO.MASK` | `Tensor[B,H,W]` | Single channel masks |
| `IO.LATENT` | `Dict[str, Tensor]` | Latent representations |
| `IO.CONDITIONING` | `List[Tuple[Tensor, Dict]]` | Encoded prompts |

### Model Types

| Type | Description |
|------|-------------|
| `IO.MODEL` | Diffusion UNet model |
| `IO.CLIP` | Text encoder model |
| `IO.VAE` | VAE encoder/decoder |
| `IO.CONTROL_NET` | ControlNet model |
| `IO.LORA_MODEL` | LoRA weights |
| `IO.UPSCALE_MODEL` | Super-resolution model |

### Utility Types

| Type | Description |
|------|-------------|
| `IO.SAMPLER` | Sampling algorithm |
| `IO.SIGMAS` | Noise schedule |
| `IO.COMBO` | Dropdown selection |
| `IO.ANY` | Wildcard type (*) |

---

## Input Options

### Number Options (INT/FLOAT)

```python
"steps": (IO.INT, {
    "default": 20,           # Default value
    "min": 1,                # Minimum value
    "max": 10000,           # Maximum value
    "step": 1,              # Increment step
    "round": 0.5,           # Round to this value
    "tooltip": "Explanation", # Hover tooltip
    "advanced": True,       # Collapse in advanced section
})
```

### String Options

```python
"prompt": (IO.STRING, {
    "default": "",
    "multiline": True,       # Multi-line text area
    "dynamicPrompts": True,  # Support {wildcards|syntax}
    "placeholder": "Enter prompt...",
})
```

### Combo Options

```python
"scheduler": (IO.COMBO, {
    "options": ["normal", "karras", "simple", "ddim_uniform"],
    "default": "normal",
})

# Or using list format (legacy):
"scheduler": (["normal", "karras", "simple"], {"default": "normal"})
```

### File Input Options

```python
"image": (IO.STRING, {
    "image_upload": True,           # Enable image upload
    "image_folder": "input",         # "input", "output", or "temp"
})
```

---

## Node Categories

### Core Categories

| Category | Description | Example Nodes |
|----------|-------------|---------------|
| `loaders` | Model loading | CheckpointLoader, VAELoader |
| `conditioning` | Prompt processing | CLIPTextEncode, ConditioningCombine |
| `latent` | Latent operations | EmptyLatentImage, VAEDecode |
| `sampling` | Diffusion sampling | KSampler, KSamplerAdvanced |
| `image` | Image operations | LoadImage, SaveImage, ImageScale |
| `_for_testing` | Experimental nodes | Various test nodes |

### Category Nesting

```python
CATEGORY = "conditioning/advanced"      # Sub-menu
CATEGORY = "loaders/video_models"       # Deep nesting
CATEGORY = "essentials"                 # Top-level
```

---

## List Handling

### Output List Nodes

Nodes that output lists for batch processing:

```python
class ImageBatch:
    INPUT_TYPES = {...}
    RETURN_TYPES = (IO.IMAGE,)
    OUTPUT_IS_LIST = (True,)  # Output is a list of images
    
    def execute(self, image_list):
        # Returns List[Tensor], not Tensor
        return (image_list,)
```

### Input List Nodes

Nodes that consume lists:

```python
class ImageProcessor:
    INPUT_TYPES = {...}
    RETURN_TYPES = (IO.IMAGE,)
    INPUT_IS_LIST = True  # Receive all items at once
    
    def execute(self, images):  # images is List[Tensor]
        processed = [self.process(img) for img in images]
        return (processed,)
```

### Automatic Iteration

When a list-output connects to a non-list-input node:

```
List Output Node ──┐
                   ├──▶ Non-List Node executes N times
Single Output ────┘    (once per list item)
```

---

## Lazy Evaluation

### Concept

Lazy evaluation delays input computation until actually needed:

```python
class LazyNode:
    INPUT_TYPES = {
        "required": {
            "always_needed": (IO.MODEL,),
            "rarely_needed": (IO.IMAGE, {"lazy": True}),  # Lazy input
        }
    }
    
    def check_lazy_status(self, always_needed, rarely_needed=None):
        """Return list of inputs that should be evaluated."""
        needed = ["always_needed"]
        if self.should_use_rarely_needed():
            needed.append("rarely_needed")
        return needed
    
    def execute(self, always_needed, rarely_needed=None):
        # rarely_needed is None if not requested in check_lazy_status
        return (result,)
```

### Use Cases

- **Conditional branches**: Only compute if branch is taken
- **Memory optimization**: Avoid loading unused models
- **Performance**: Skip expensive operations when cached

---

## Validation

### Static Validation

```python
@classmethod
def VALIDATE_INPUTS(s, input_name, another_input):
    """Validate inputs before execution."""
    if input_name not in ["allowed", "values"]:
        return "Invalid input_name"
    return True  # Validation passed
```

### Change Detection

```python
@classmethod
def IS_CHANGED(s, image_path):
    """Return value that changes when input changes.
    
    Used to invalidate cache.
    """
    return os.path.getmtime(image_path)  # Timestamp
    # Or return float("NaN") to always re-execute
```

---

## Advanced Node Patterns

### Node with Side Effects (Output Node)

```python
class SaveImage:
    INPUT_TYPES = {...}
    RETURN_TYPES = ()  # No outputs
    OUTPUT_NODE = True  # Must execute even if unconnected
    
    def execute(self, images, filename_prefix):
        for img in images:
            self.save_image(img, filename_prefix)
        return {"ui": {"images": image_metadata}}  # UI update
```

### Node with Multiple Outputs

```python
class CheckpointLoader:
    RETURN_TYPES = (IO.MODEL, IO.CLIP, IO.VAE)
    RETURN_NAMES = ("model", "clip", "vae")
    
    def load_checkpoint(self, ckpt_name):
        model, clip, vae = load_models(ckpt_name)
        return (model, clip, vae)  # Tuple matches RETURN_TYPES
```

### Dynamic Input Types

```python
class FlexibleNode:
    @classmethod
    def INPUT_TYPES(s):
        # Dynamic based on available files
        files = get_available_files()
        return {
            "required": {
                "file": (files,),
            }
        }
```

---

## Node Registration

### Automatic Registration

Nodes are automatically discovered from `nodes.py`:

```python
# At end of nodes.py
NODE_CLASS_MAPPINGS = {
    "CheckpointLoader": CheckpointLoader,
    "CLIPTextEncode": CLIPTextEncode,
    # ... all nodes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoader": "Load Checkpoint",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)",
    # ... display names
}
```

### Custom Node Registration

```python
# In custom_node/__init__.py
from .my_node import MyCustomNode

NODE_CLASS_MAPPINGS = {"MyCustomNode": MyCustomNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyCustomNode": "My Custom Node"}
```

---

## Type System Internals

### Type Matching

```python
# Type compatibility check
def validate_node_input(received_type, input_type):
    if input_type == "*":  # ANY type accepts all
        return True
    if received_type == input_type:
        return True
    # Multi-type support: "FLOAT,INT" accepts both
    return received_type in input_type.split(",")
```

### Type Safety

ComfyUI validates connections at:
1. **Workflow validation**: Before execution starts
2. **Runtime validation**: During node execution
3. **Type coercion**: Automatic conversion when safe

---

## Best Practices

### DO

- ✅ Use descriptive node names
- ✅ Provide tooltips for all inputs/outputs
- ✅ Validate inputs early
- ✅ Handle edge cases gracefully
- ✅ Use appropriate input types
- ✅ Implement IS_CHANGED for file inputs
- ✅ Set OUTPUT_NODE for side-effect nodes

### DON'T

- ❌ Use ANY type unnecessarily
- ❌ Modify inputs in-place
- ❌ Access global state directly
- ❌ Block the execution thread
- ❌ Return mismatched output types

---

## Z-Image-Turbo Adaptation Notes

### Critical Features to Implement

1. **Typed Node System**: Enforce input/output types
2. **Category Organization**: Logical grouping
3. **Validation Framework**: Pre-execution checks
4. **List Handling**: Batch processing support
5. **Lazy Evaluation**: Conditional computation

### Simplified Node Base

```python
class ZNode:
    """Simplified base class for Z-Image-Turbo nodes."""
    
    INPUT_TYPES = {}
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "zimage"
    
    def validate_inputs(self, **kwargs):
        """Override for custom validation."""
        return True
    
    def execute(self, **kwargs):
        """Override in subclasses."""
        raise NotImplementedError
```

---

## Related Files

| File | Purpose |
|------|---------|
| `nodes.py` | Core node definitions |
| `comfy/comfy_types/node_typing.py` | Type definitions |
| `comfy_execution/validation.py` | Input validation |
| `comfy_api/internal/` | Node API internals |
