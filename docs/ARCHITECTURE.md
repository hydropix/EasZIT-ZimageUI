# Z-Image-Turbo Architecture

## Overview

Z-Image-Turbo now uses a **hybrid architecture**:
- **Frontend**: Simple Gradio UI (tabs for txt2img, img2img, inpainting)
- **Backend**: Node-based execution engine inspired by ComfyUI

This design provides the power and flexibility of ComfyUI's node system while maintaining an accessible user interface.

```
┌─────────────────────────────────────────────┐
│              GRADIO UI                       │
│  ┌─────────┬──────────┬──────────┐          │
│  │ txt2img │ img2img  │ inpaint  │          │
│  └────┬────┴────┬─────┴────┬─────┘          │
└───────┼─────────┼──────────┼────────────────┘
        │         │          │
        ▼         ▼          ▼
┌─────────────────────────────────────────────┐
│         WORKFLOW BUILDER                     │
│  (Converts UI params to JSON workflow)       │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│      EXECUTION ENGINE                        │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ Topological │  │    Cache    │           │
│  │    Sort     │  │   (LRU)     │           │
│  └──────┬──────┘  └─────────────┘           │
│         │                                   │
│         ▼                                   │
│  ┌─────────────────────┐                    │
│  │   NODE EXECUTION    │                    │
│  │  LoadZImageTurbo    │                    │
│  │  CLIPTextEncode     │                    │
│  │  EmptyLatentImage   │                    │
│  │  KSampler           │                    │
│  │  VAEDecode          │                    │
│  │  SaveImage          │                    │
│  └─────────────────────┘                    │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
zimage/
├── nodes/                    # Node system
│   ├── base.py              # ZNode base class, ZType enum
│   ├── registry.py          # NodeRegistry with custom_nodes support
│   └── core.py              # Core nodes (LoadModel, CLIP, Sampler, etc.)
├── engine/                   # Execution engine
│   └── executor.py          # Workflow execution with caching
├── core/                     # Core utilities
│   └── model_manager.py     # Model loading with LRU cache
├── workflows/                # Workflow management
│   ├── builder.py           # Workflow builder from simple params
│   └── templates.py         # Pre-built workflow templates
├── gradio_ui.py             # Gradio interface
└── extensions/              # Extension system

custom_nodes/                 # User custom nodes (auto-loaded)
```

## Node System

### Base Node Class

```python
from zimage.nodes import ZNode, ZType, register_node

@register_node("MyNode", "category")
class MyNode(ZNode):
    INPUT_TYPES = {
        "required": {
            "input_name": (ZType.IMAGE,),
        }
    }
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, input_name):
        # Node logic
        return (result,)
```

### Available Types

- `ZType.IMAGE` - Image tensor [B, H, W, C]
- `ZType.LATENT` - Latent representation
- `ZType.MODEL` - Diffusion model
- `ZType.CLIP` - Text encoder
- `ZType.VAE` - VAE encoder/decoder
- `ZType.CONDITIONING` - Encoded prompt
- `ZType.STRING`, `ZType.INT`, `ZType.FLOAT`, `ZType.BOOLEAN`

## Creating Custom Nodes

1. Create a Python file in `custom_nodes/`:

```python
# custom_nodes/my_custom_node.py
from zimage.nodes import ZNode, ZType, register_node

@register_node("MyCustomNode", "custom")
class MyCustomNode(ZNode):
    """My custom node description."""
    
    INPUT_TYPES = {
        "required": {
            "image": (ZType.IMAGE,),
            "strength": (ZType.FLOAT, {"default": 1.0, "min": 0.0, "max": 2.0}),
        }
    }
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, image, strength):
        # Your logic here
        result = image * strength
        return (result,)
```

2. Restart the application
3. The node is automatically registered and available

## Workflow JSON Format

Workflows are represented as JSON:

```json
{
  "1": {
    "class_type": "LoadZImageTurbo",
    "inputs": {
      "dtype": "bfloat16",
      "attention": "sdpa"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "a beautiful landscape",
      "clip": ["1", 1]
    }
  },
  "3": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  }
}
```

Links are represented as `[source_node_id, output_index]`.

## Running the Application

```bash
# Standard launch
python app.py

# With options
python app.py --port 8080 --host 0.0.0.0

# Without custom nodes
python app.py --no-custom-nodes

# With authentication
python app.py --auth user:pass
```

## Importing ComfyUI Workflows

You can import ComfyUI workflow JSON files:

```python
from zimage.engine.executor import ExecutionEngine
from zimage.workflows.builder import WorkflowBuilder

# Load ComfyUI workflow
workflow = WorkflowBuilder.load("path/to/workflow.json")

# Execute
engine = ExecutionEngine()
result = engine.execute(workflow)
```

Note: Only nodes registered in Z-Image-Turbo will work. Custom ComfyUI nodes need to be ported.

## Key Differences from ComfyUI

1. **No visual node editor** - UI is Gradio-based with simple tabs
2. **Simplified execution** - Synchronous execution, no async complexity
3. **Focused on Z-Image-Turbo** - Optimized for this specific model
4. **JSON workflows** - Can import/export workflows as JSON

## Future Enhancements

- [ ] Full sampling implementation (using diffusers schedulers)
- [ ] Inpainting with mask support
- [ ] LoRA loading nodes
- [ ] ControlNet support
- [ ] Batch processing
- [ ] Workflow save/load from UI
