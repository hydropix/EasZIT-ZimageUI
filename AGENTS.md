# Z-Image-Turbo Web UI - Agent Documentation

## Project Overview

Z-Image-Turbo Web UI is a web interface for generating images using **Z-Image-Turbo**, a 6 billion parameter diffusion model developed by Alibaba/Tongyi. The application provides a modern Gradio-based UI that runs locally on a PC with GPU and can be accessed remotely via network.

**Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
**Key Features**:
- Fast generation in ~8 steps (few seconds)
- Photorealistic rendering with bilingual text support (EN/ZH)
- Optimizations: Flash Attention, torch compilation, CPU offloading
- Network accessible (local + LAN + public tunnel)
- **Node-based backend** inspired by ComfyUI architecture

---

## Architecture Overview

### Hybrid Architecture (New)

The application now uses a **hybrid architecture**:
- **Frontend**: Simple Gradio UI with tabs (preserves original UX)
- **Backend**: Node-based execution engine inspired by ComfyUI

```
┌─────────────────────────────────────────────┐
│              GRADIO UI                       │
│  • Text to Image tab                        │
│  • Image to Image tab                       │
│  • Model Management tab                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         WORKFLOW BUILDER (JSON)             │
│  Converts UI parameters to node workflow    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      EXECUTION ENGINE                        │
│  • Topological sort (Kahn's algorithm)      │
│  • Node dependency resolution               │
│  • LRU caching for node outputs             │
│  • Progress callbacks                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  NODE SYSTEM                                 │
│  ┌───────────────────────────────────────┐  │
│  │ Core Nodes (13 registered)            │  │
│  │ • GenerateImage (txt2img)             │  │
│  │ • GenerateImageToImage (img2img)      │  │
│  │ • LoadImage, SaveImageTensor          │  │
│  │ • ImageScale, VAEEncode, etc.         │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ Custom Nodes (from custom_nodes/)     │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Key Differences from Original App

| Aspect | Original | Current (Node-based) |
|--------|----------|---------------------|
| Backend | Monolithic pipeline | Modular node system |
| Extensibility | Fixed | Custom nodes supported |
| Caching | None | LRU cache per node |
| ComfyUI import | No | Yes (via adapter) |
| Architecture | Direct diffusers | Node execution engine |

---

## Technology Stack

| Component | Version/Requirement |
|-----------|---------------------|
| Python | 3.9 - 3.12 |
| PyTorch | 2.6.0 (with CUDA 12.4) |
| Gradio | >= 6.0.0 |
| Diffusers | GitHub main branch |
| Transformers | >= 4.49.0 |
| Accelerate | >= 1.4.0 |
| PIL/Pillow | >= 11.0.0 |
| NumPy | >= 2.0.0 |

**Hardware Requirements**:
- Minimum: NVIDIA GPU with 8GB VRAM, 16GB RAM
- Recommended: RTX 3090/4090 (24GB VRAM), 32GB RAM

---

## Project Structure

```
Z-Image-Turbo-WebUI/
├── app.py                      # NEW: Entry point with node backend
├── app_legacy.py               # ORIGINAL: Legacy app (backup)
├── launch.py                   # Launcher script with pre-flight checks
├── setup-zimage.bat            # Windows batch for full setup/update
├── start-zimage.bat            # Windows batch for fast launch
├── launch-node-ui.bat          # NEW: Launcher for node-based UI
├── test-nodes.bat              # NEW: Test node system
├── quick-launch.bat            # NEW: Quick launcher
├── diagnose.py                 # Diagnostic script
├── requirements.txt            # Python dependencies
├── README.md                   # User documentation
├── AGENTS.md                   # This file
├── .gitignore
├── outputs/images/             # Generated images directory
├── venv/                       # Virtual environment
│
├── zimage/                     # NEW: Node-based system package
│   ├── __init__.py
│   ├── gradio_ui.py           # Gradio interface (original styling)
│   │
│   ├── nodes/                 # Node system
│   │   ├── __init__.py
│   │   ├── base.py            # ZNode base class, ZType enum
│   │   ├── registry.py        # NodeRegistry with custom_nodes support
│   │   ├── core.py            # Core nodes (VAE, Image ops)
│   │   └── diffusers_nodes.py # NEW: Generation nodes using diffusers
│   │
│   ├── engine/                # Execution engine
│   │   ├── __init__.py
│   │   └── executor.py        # Workflow execution with caching
│   │
│   ├── core/                  # Core utilities
│   │   ├── __init__.py
│   │   └── model_manager.py   # Model LRU cache
│   │
│   ├── workflows/             # Workflow management
│   │   ├── __init__.py
│   │   ├── builder.py         # Workflow builder from parameters
│   │   └── templates.py       # Pre-built workflow templates
│   │
│   └── utils/                 # NEW: Utilities
│       ├── __init__.py
│       └── workflow_adapter.py # ComfyUI workflow adapter
│
├── custom_nodes/              # NEW: User custom nodes directory
│   └── example_custom_node.py # Example custom node
│
└── docs/                      # Documentation
    ├── ARCHITECTURE.md        # Architecture documentation
    └── comfyui_analysis/      # ComfyUI adaptation docs
        ├── README.md
        ├── 07_adaptation_plan.md
        ├── 08_implementation_summary.md
        └── 09_updated_plan.md
```

---

## Build and Run Commands

### Quick Start (Windows) - Node-based UI

```bash
# First time: Setup
setup-zimage.bat

# Launch node-based UI
launch-node-ui.bat

# Or quick launch (minimal output)
quick-launch.bat
```

### Launch Options (Node-based)

```bash
# Standard launch
python app.py

# Port different
launch-node-ui.bat --port 8080

# Local only (no network)
launch-node-ui.bat --local

# Without custom nodes
launch-node-ui.bat --no-custom-nodes

# With authentication
launch-node-ui.bat --auth user:password

# Public share link
launch-node-ui.bat --share
```

### Testing Node System

```bash
# Test node system before launch
test-nodes.bat

# Or manually:
python -c "from zimage.nodes import NodeRegistry; print(list(NodeRegistry.list_nodes().keys()))"
```

---

## Current Implementation Status

### ✅ Completed Features

1. **Node System Foundation**
   - Base node class with `INPUT_TYPES`, `RETURN_TYPES`
   - Node registry with auto-discovery
   - 13 core nodes registered

2. **Execution Engine**
   - Topological sort for dependency resolution
   - LRU caching (50 entries max)
   - Progress callbacks
   - Error handling

3. **Generation Nodes**
   - `GenerateImage` - Full txt2img pipeline using diffusers
   - `GenerateImageToImage` - Full img2img pipeline
   - `SaveImageTensor` - Save output to disk

4. **UI**
   - Original Gradio styling (black/white theme, square sliders)
   - Text to Image tab (working)
   - Image to Image tab (basic)
   - Model Management tab

5. **Workflow System**
   - `create_txt2img_workflow()` - Working
   - `create_img2img_workflow()` - Basic
   - JSON export/import

6. **Utilities**
   - ComfyUI workflow adapter (`workflow_adapter.py`)

### 🔄 In Progress / Needs Work

1. **Image-to-Image** (Priority: HIGH)
   - Current: Basic implementation loads pipeline twice
   - Needed: Reuse loaded model between generations
   - File: `zimage/nodes/diffusers_nodes.py` → `GenerateImageToImage`

2. **Model Persistence** (Priority: HIGH)
   - Current: Pipeline loaded on every generation
   - Needed: Cache loaded pipeline in `ModelManager`
   - See: `zimage/core/model_manager.py`

3. **Inpainting** (Priority: MEDIUM)
   - Current: Basic (uses img2img)
   - Needed: Proper mask processing
   - Requires: `LoadMask` node, masked sampling

### 📋 Next Steps / TODO

#### Phase 1: Core Improvements

1. **Fix Image-to-Image**
   ```python
   # Current issue: Loads pipeline twice
   # Solution: Use ModelManager to cache pipeline
   # After GenerateImage, cache pipeline in ModelManager
   # GenerateImageToImage should check cache first
   ```

2. **Model Caching**
   ```python
   # In GenerateImage.execute():
   model_key = f"zimage_pipeline_{dtype}_{attention}"
   pipeline = model_manager.load_model(model_key, load_fn)
   ```

3. **Batch Processing**
   - Support batch_size > 1 in GenerateImage
   - Return list of images

#### Phase 2: Advanced Features

4. **LoRA Support**
   - Create `LoadLoRA` node
   - Create `ApplyLoRA` node
   - Integrate with diffusers LoRA loading

5. **ControlNet**
   - Create `LoadControlNet` node
   - Create `ApplyControlNet` node
   - Requires: ControlNet model download

6. **Better Inpainting**
   - Proper mask loading
   - Masked latent generation
   - Inpaint-specific model support

#### Phase 3: Workflow & UX

7. **Workflow Import/Export**
   - Save workflow from UI
   - Load ComfyUI workflow (partial support exists)
   - Visual workflow editor (future)

8. **Gallery View**
   - Show generated images history
   - Metadata display (prompt, seed, params)
   - Re-run from history

9. **Advanced Parameters**
   - Scheduler selection
   - Sampler selection (Euler, DPM++, etc.)
   - Negative prompt strength

---

## Node Development Guide

### Creating a Custom Node

```python
# custom_nodes/my_node.py
from zimage.nodes import ZNode, ZType, register_node
from typing import Tuple, Any

@register_node("MyNode", "category")
class MyNode(ZNode):
    """Description of what this node does."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (ZType.IMAGE,),
                "strength": (ZType.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "mask": (ZType.MASK,),
            }
        }
    
    RETURN_TYPES = (ZType.IMAGE,)
    RETURN_NAMES = ("output_image",)
    
    def execute(self, image: Any, strength: float, mask: Any = None) -> Tuple[Any]:
        """Node logic here."""
        import torch
        result = image * strength
        return (result,)
```

### Node Type Reference

| Type | Description | Format |
|------|-------------|--------|
| `ZType.IMAGE` | Image tensor | `[B, H, W, C]` range `[0, 1]` |
| `ZType.LATENT` | Latent representation | Dict with `"samples"` key |
| `ZType.MODEL` | Diffusion model | UNet/Transformer |
| `ZType.CLIP` | Text encoder | Text encoder model |
| `ZType.VAE` | VAE model | Encoder/decoder |
| `ZType.CONDITIONING` | Encoded prompt | Text embeddings tensor |
| `ZType.MASK` | Mask tensor | `[B, H, W]` range `[0, 1]` |
| `ZType.STRING` | Text | Python string |
| `ZType.INT` | Integer | Python int |
| `ZType.FLOAT` | Float | Python float |
| `ZType.BOOLEAN` | Boolean | Python bool |

---

## ComfyUI Compatibility

### Workflow Adapter Usage

```bash
# Analyze a ComfyUI workflow
python -c "
from zimage.utils.workflow_adapter import analyze_comfyui_workflow
analyze_comfyui_workflow('workflow.json')
"

# Adapt workflow to our format
python -c "
from zimage.utils.workflow_adapter import ComfyUIWorkflowAdapter
adapter = ComfyUIWorkflowAdapter()
result = adapter.adapt_workflow('comfy_workflow.json', 'adapted.json')

# Print missing node templates
for name, code in result['missing_node_templates'].items():
    print(f'=== {name} ===')
    print(code)
"
```

### Node Mapping

| ComfyUI Node | Our Node | Status |
|--------------|----------|--------|
| `CheckpointLoaderSimple` | `LoadZImageTurbo` | ✅ Mapped |
| `KSampler` | `GenerateImage` | ✅ Mapped |
| `CLIPTextEncode` | (included in GenerateImage) | ✅ Absorbed |
| `SaveImage` | `SaveImageTensor` | ✅ Mapped |
| `LoadImage` | `LoadImage` | ✅ Mapped |
| `VAEEncode` | `VAEEncode` | ✅ Mapped |
| `VAEDecode` | (included in GenerateImage) | ✅ Absorbed |
| `ControlNetApply` | - | ❌ Not implemented |
| `LoRALoader` | - | ❌ Not implemented |

---

## Configuration Options

### Model Loading Options (UI)
| Option | Choices | Description |
|--------|---------|-------------|
| dtype | bfloat16, float16, float32 | Precision for model weights |
| attention | sdpa, flash_attention_2 | Attention mechanism |
| enable_cpu_offload | checkbox | For GPUs with limited VRAM |

### Generation Parameters
| Parameter | Range | Default | Note |
|-----------|-------|---------|------|
| width | 512-2048 | 1024 | Step 64 |
| height | 512-2048 | 1024 | Step 64 |
| num_steps | 1-50 | 9 | 9 = ~8 forwards DiT (recommended for Turbo) |
| guidance_scale | 0.0-10.0 | 0.0 | 0.0 recommended for Turbo |
| seed | -1 or int | -1 | -1 = random |
| strength | 0.0-1.0 | 0.75 | For img2img (0=no change, 1=full generation) |

---

## Development Conventions

### Language
**ALL content must be written in English** - No exceptions:
- **Code comments**: English
- **User-facing UI**: English
- **Documentation (README, AGENTS.md, etc.)**: English
- **Commit messages**: English
- **Variable names**: English
- **Docstrings**: English
- **Error messages**: English

> ⚠️ This project is now **English-only**. Do not use French or any other language.

### Coding Style
- **English for all UI labels and user messages**
- English for technical/internal variable names
- Docstrings in English
- Type hints not used

### Icons
- **Use Google Material Icons** (Material Symbols Rounded) instead of emojis
- For buttons: Use the `icon` parameter with direct SVG URLs from Google Fonts
- Available icons: https://fonts.google.com/icons
- **Do not use emojis** in the UI or console output

### Lazy Imports
- **Always use lazy imports** in nodes to avoid slow startup:
```python
# BAD (slows startup)
import torch
from diffusers import Pipeline

# GOOD (fast startup)
def execute(self, ...):
    import torch
    from diffusers import Pipeline
    # ... use imports here
```

### Output Handling
- Generated images saved to: `outputs/images/zimage_{timestamp}_{index}.png`
- Timestamps use format: `%Y%m%d_%H%M%S`

---

## Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Enable CPU offloading, reduce to 512x512, use float16 |
| "No module named 'diffusers'" | Run `pip install -r requirements.txt` |
| Model loads slowly every time | Model caching not implemented yet (see TODO) |
| Custom node not loading | Check `custom_nodes/` directory exists, file has `.py` extension |
| Workflow import fails | Check node mapping in `workflow_adapter.py` |

---

## External Dependencies

- **Model**: Auto-downloaded from HuggingFace on first run (Tongyi-MAI/Z-Image-Turbo)
- **Flash Attention**: Optional, requires separate install with CUDA toolkit
- **ComfyUI Nodes**: Can be adapted using `workflow_adapter.py`

---

## Legacy App

The original monolithic app is preserved as `app_legacy.py` for reference.
Use it if you need:
- Model loading/unloading UI
- Auto-generation mode
- Detailed VRAM breakdown

To run legacy app:
```bash
python app_legacy.py
```

---

*Last updated: March 2026 - Node-based architecture implemented*
