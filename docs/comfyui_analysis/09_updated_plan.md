# Updated Adaptation Plan - Simplified Architecture

## Revised Approach

After reviewing the requirements, the plan has been updated to focus on:
1. **Simple UX**: Clean Gradio tabs, no visual node editor
2. **Powerful Backend**: Full node-based execution with caching
3. **Extensibility**: Custom nodes support for future expansion
4. **ComfyUI Compatibility**: Can import workflows from ComfyUI

## Implementation Status

### ✅ Phase 1: Foundation (Complete)

#### Created Components:

1. **Node System** (`zimage/nodes/`)
   - `base.py`: ZNode abstract class, ZType enum
   - `registry.py`: NodeRegistry with custom_nodes auto-discovery
   - `core.py`: 10 core nodes for image generation

2. **Execution Engine** (`zimage/engine/`)
   - `executor.py`: Topological sort, caching, execution

3. **Model Management** (`zimage/core/`)
   - `model_manager.py`: LRU cache, memory management

4. **Workflow System** (`zimage/workflows/`)
   - `builder.py`: Programmatic workflow creation
   - `templates.py`: Pre-built workflows (txt2img, img2img, inpaint)

5. **UI** (`zimage/gradio_ui.py`)
   - 4 tabs: txt2img, img2img, inpainting, model

### 📋 Phase 2: Core Features (Next)

#### TODO:

1. **Complete Sampling Implementation**
   - Integrate with diffusers schedulers
   - Implement actual denoising loop in KSampler
   - Add proper Euler, DPM++, DDIM samplers

2. **Image-to-Image (Full)**
   - Proper strength control
   - Image encoding/decoding pipeline

3. **Inpainting (Full)**
   - Mask loading and processing
   - Inpainting model support
   - Masked sampling

4. **LoRA Support**
   - LoRALoader node
   - Model patching system

5. **Testing**
   - Unit tests for nodes
   - Integration tests for workflows
   - Performance benchmarks

### 📋 Phase 3: Advanced Features (Future)

1. **ControlNet Support**
2. **Batch Processing**
3. **Workflow Save/Load from UI**
4. **Extension API improvements**

## Architecture Comparison

### Original Plan (Full ComfyUI Clone)
```
Node Editor UI (React/Canvas)
    ↓
Complex Execution Engine
    ↓
Extensive Node Library
```

### New Plan (Hybrid)
```
Simple Gradio UI (Tabs)
    ↓
Workflow Builder (JSON)
    ↓
Node Execution Engine
    ↓
Core + Custom Nodes
```

### Benefits of New Approach:

| Aspect | Full ComfyUI | Hybrid (New) |
|--------|-------------|--------------|
| UX Complexity | High (learning curve) | Low (intuitive) |
| Development Time | 8+ weeks | 2-3 weeks |
| Extensibility | Full | Good (custom nodes) |
| Maintenance | Complex | Simple |
| ComfyUI Import | N/A | Yes |
| Mobile Friendly | No | Yes (Gradio) |

## File Structure

```
Z-Image-Turbo/
├── app.py                      # Entry point
├── app_legacy.py               # Original app (backup)
├── zimage/                     # New node-based system
│   ├── nodes/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── core.py
│   ├── engine/
│   │   └── executor.py
│   ├── core/
│   │   └── model_manager.py
│   ├── workflows/
│   │   ├── builder.py
│   │   └── templates.py
│   └── gradio_ui.py
├── custom_nodes/               # User extensions
│   └── example_custom_node.py
└── docs/
    ├── ARCHITECTURE.md
    └── comfyui_analysis/
        └── 08_implementation_summary.md
```

## Usage Examples

### Creating a Workflow

```python
from zimage.workflows.builder import create_txt2img_workflow
from zimage.engine.executor import ExecutionEngine

workflow = create_txt2img_workflow(
    prompt="a beautiful landscape",
    negative_prompt="blur, low quality",
    width=1024,
    height=1024,
    steps=9,
    seed=-1,
)

engine = ExecutionEngine()
result = engine.execute(workflow)
```

### Creating a Custom Node

```python
# custom_nodes/my_node.py
from zimage.nodes import ZNode, ZType, register_node

@register_node("MyNode", "custom")
class MyNode(ZNode):
    INPUT_TYPES = {
        "required": {
            "image": (ZType.IMAGE,),
        }
    }
    RETURN_TYPES = (ZType.IMAGE,)
    
    def execute(self, image):
        return (image,)
```

### Importing ComfyUI Workflow

```python
from zimage.engine.executor import ExecutionEngine
from zimage.workflows.builder import WorkflowBuilder

workflow = WorkflowBuilder.load("comfyui_workflow.json")
engine = ExecutionEngine()
result = engine.execute(workflow)
```

## Migration Path

1. **Current**: New system works side-by-side with legacy app
2. **Testing**: Both apps can be run independently
3. **Switch**: When ready, replace app.py with new version
4. **Legacy**: app_legacy.py kept for reference/rollback

## Conclusion

The revised plan focuses on delivering a working system faster while maintaining extensibility. The hybrid approach gives users a simple interface while providing the power of a node-based backend.

Key achievements:
- ✅ Node system with 10 core nodes
- ✅ Execution engine with caching
- ✅ Custom nodes support
- ✅ Gradio UI with tabs
- ✅ Workflow JSON import/export

Next priorities:
1. Complete sampling implementation
2. Full img2img/inpainting support
3. LoRA support
4. Comprehensive testing
