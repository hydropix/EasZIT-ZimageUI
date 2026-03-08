# ComfyUI Memory Management

## Overview

ComfyUI implements sophisticated memory management to handle large diffusion models within limited VRAM constraints. The system automatically adapts to available hardware and provides multiple strategies for memory optimization.

---

## VRAM State System

### State Hierarchy

```python
class VRAMState(Enum):
    DISABLED = 0    # No GPU: CPU-only mode
    NO_VRAM = 1     # Minimal VRAM: maximum offloading
    LOW_VRAM = 2    # Limited VRAM: partial offloading
    NORMAL_VRAM = 3 # Standard operation
    HIGH_VRAM = 4   # Keep models in VRAM
    SHARED = 5      # Unified memory (Apple Silicon, etc.)
```

### Automatic Detection

```python
def detect_vram_state():
    """Automatically determine optimal VRAM state."""
    total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
    
    if args.cpu:
        return VRAMState.DISABLED
    elif args.lowvram:
        return VRAMState.LOW_VRAM
    elif args.novram:
        return VRAMState.NO_VRAM
    elif args.highvram or args.gpu_only:
        return VRAMState.HIGH_VRAM
    elif cpu_state == CPUState.MPS:
        return VRAMState.SHARED
    else:
        return VRAMState.NORMAL_VRAM
```

---

## Model Loading Strategies

### Strategy Matrix

| VRAM State | Model Loading | Patch Application | Behavior |
|------------|---------------|-------------------|----------|
| `DISABLED` | CPU only | CPU | Slowest, no VRAM used |
| `NO_VRAM` | CPU + partial GPU | On-demand | Constant offloading |
| `LOW_VRAM` | Partial GPU | Incremental | Aggressive offloading |
| `NORMAL_VRAM` | Full GPU | Batch | Standard operation |
| `HIGH_VRAM` | Full GPU + keep | Pre-computed | Fastest, max VRAM use |
| `SHARED` | Unified memory | Mixed | Platform-specific |

### Implementation

```python
class LoadedModel:
    """Wrapper for a loaded model with memory management."""
    
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
    
    def model_load(self, lowvram_model_memory=0):
        """Load model to device with VRAM optimization."""
        patch_model_to = self.device
        
        # Determine load strategy based on VRAM state
        if vram_state == VRAMState.NO_VRAM:
            # Keep weights in RAM, compute on GPU
            patch_model_to = torch.device("cpu")
        elif vram_state == VRAMState.LOW_VRAM:
            # Partial loading based on available memory
            lowvram_model_memory = self.calculate_lowvram_memory()
        
        # Apply patches (LoRA, etc.) and move to device
        self.real_model = self.model.patch_model(
            device_to=patch_model_to,
            lowvram_model_memory=lowvram_model_memory
        )
        
        return self.real_model
    
    def model_unload(self, memory_used=0):
        """Unload model to free VRAM."""
        if self.real_model is not None:
            self.model.unpatch_model()
            self.real_model = None
```

---

## Low VRAM Mode

### Concept

Low VRAM mode keeps model weights in system RAM and only transfers needed layers to GPU during computation.

```
┌─────────────────────────────────────────────────────────────┐
│                    LOW VRAM MODE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  System RAM (16GB+)         GPU VRAM (4-8GB)                │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │ Full Model      │        │ Current Layer   │             │
│  │ Weights         │───────▶│ Being Computed  │             │
│  │ (FP16/BF16)     │        │                 │             │
│  └─────────────────┘        └─────────────────┘             │
│         │                           │                       │
│         │                           ▼                       │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │ LoRA Weights    │        │ Output Activations          │
│  │ (FP16)          │        │                 │             │
│  └─────────────────┘        └─────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class LowVramPatch:
    """Patch for on-demand weight loading."""
    
    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func
        self.set_func = set_func
    
    def __call__(self, weight):
        """Apply patches when weight is accessed."""
        return calculate_weight(
            self.patches[self.key], 
            weight, 
            self.key,
            intermediate_dtype=weight.dtype
        )

def apply_lowvram_patches(model, patches):
    """Replace weights with lazy-loading wrappers."""
    for key, patch_list in patches.items():
        weight, set_func, convert_func = get_key_weight(model, key)
        
        # Create wrapper that applies patches on access
        patch = LowVramPatch(key, patches, convert_func, set_func)
        
        # Replace weight with patched version
        if set_func is not None:
            set_func(patch(weight))
        else:
            # Use weight_function hook
            if not hasattr(module, "weight_function"):
                module.weight_function = []
            module.weight_function.append(patch)
```

### Memory Estimation

```python
def low_vram_patch_estimate_vram(model, key):
    """Estimate VRAM needed for patch computation."""
    weight, _, _ = get_key_weight(model, key)
    if weight is None:
        return 0
    
    model_dtype = getattr(model, "manual_cast_dtype", torch.float32)
    if model_dtype is None:
        model_dtype = weight.dtype
    
    # Estimate: weight_size * dtype_size * computation_factor
    return weight.numel() * model_dtype.itemsize * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR
```

---

## Smart Memory Management

### Cleanup Strategies

```python
def cleanup_models_gc(aggressive=False):
    """Garbage collect unused models."""
    global current_loaded_models
    
    # Remove unused models from list
    new_loaded_models = []
    for loaded_model in current_loaded_models:
        if loaded_model.currently_used:
            new_loaded_models.append(loaded_model)
        else:
            # Unload to free memory
            loaded_model.model_unload()
    
    current_loaded_models = new_loaded_models
    
    # Force Python garbage collection
    gc.collect()
    
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def unload_all_models():
    """Emergency unload all models."""
    global current_loaded_models
    for loaded_model in current_loaded_models:
        loaded_model.model_unload(remove_model=True)
    current_loaded_models = []
    gc.collect()
    soft_empty_cache(True)
```

### Automatic Cleanup

```python
def handle_execution_error(ex):
    """Handle errors, with OOM special case."""
    if isinstance(ex, OOM_EXCEPTION):
        logging.error("Out of memory error - unloading models")
        logging.info("Memory summary: %s", debug_memory_summary())
        unload_all_models()
        return "Out of VRAM. Models unloaded. Retry with lower resolution."
```

---

## Device Management

### Device Selection

```python
def get_torch_device():
    """Get optimal device for computation."""
    global directml_enabled, cpu_state
    
    if directml_enabled:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    
    # GPU selection
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    elif is_ascend_npu():
        return torch.device("npu", torch.npu.current_device())
    elif is_mlu():
        return torch.device("mlu", torch.mlu.current_device())
    else:
        return torch.device(torch.cuda.current_device())
```

### Memory Queries

```python
def get_free_memory(device=None):
    """Get available memory on device."""
    if device is None:
        device = get_torch_device()
    
    if hasattr(device, 'type') and device.type == 'cuda':
        stats = torch.cuda.memory_stats(device)
        mem_reserved = stats['reserved_bytes.all.current']
        _, mem_total_cuda = torch.cuda.mem_get_info(device)
        return mem_total_cuda - mem_reserved
    elif device.type == 'mps':
        # MPS doesn't expose free memory directly
        return psutil.virtual_memory().available
    else:
        return psutil.virtual_memory().available
```

---

## ModelPatcher Memory Features

### Weight Offloading

```python
class ModelPatcher:
    def __init__(self, model, load_device, offload_device, ...):
        self.model = model
        self.load_device = load_device      # GPU device
        self.offload_device = offload_device # CPU/RAM device
        self.patches = {}                   # Weight modifications
        self.backup = {}                    # Original weight backups
    
    def patch_model(self, device_to=None, lowvram_model_memory=0):
        """Apply patches and move to device."""
        if device_to is None:
            device_to = self.load_device
        
        # Apply weight patches
        for key, patches in self.patches.items():
            weight, set_func, _ = get_key_weight(self.model, key)
            
            # Move to device if needed
            if weight.device != device_to:
                weight = weight.to(device_to)
            
            # Apply patches
            patched_weight = self.patch_weight(key, weight)
            
            # Set patched weight
            if set_func is not None:
                set_func(patched_weight)
            else:
                # Backup original
                self.backup[key] = weight
                # Replace
                setattr(module, param_name, patched_weight)
        
        return self.model
    
    def unpatch_model(self, device_to=None):
        """Restore original weights and move to device."""
        if device_to is None:
            device_to = self.offload_device
        
        # Restore backups
        for key, backup_weight in self.backup.items():
            weight, set_func, _ = get_key_weight(self.model, key)
            if set_func is not None:
                set_func(backup_weight.to(device_to))
            else:
                setattr(module, param_name, backup_weight.to(device_to))
        
        self.backup.clear()
```

### Partial Loading

```python
def partially_load_model(self, device_to, memory_to_free=0):
    """Load only part of model to GPU."""
    # Calculate how much we can load
    free_memory = get_free_memory(device_to)
    target_memory = free_memory - memory_to_free
    
    # Sort modules by priority
    modules_to_load = self.get_modules_by_priority()
    
    loaded_memory = 0
    for module in modules_to_load:
        module_memory = estimate_module_memory(module)
        
        if loaded_memory + module_memory < target_memory:
            # Load to GPU
            module.to(device_to)
            loaded_memory += module_memory
        else:
            # Keep on CPU
            module.to(self.offload_device)
```

---

## Attention Optimization

### PyTorch Attention

```python
# Enable PyTorch's native attention (PyTorch 2.0+)
ENABLE_PYTORCH_ATTENTION = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

### xFormers (Optional)

```python
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False

def optimized_attention(q, k, v, ...):
    if XFORMERS_IS_AVAILABLE:
        return xformers.ops.memory_efficient_attention(q, k, v)
    else:
        # Fallback to PyTorch
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

---

## Quantization Support

### FP8 Support

```python
SUPPORT_FP8_OPS = args.supports_fp8_compute

if SUPPORT_FP8_OPS and is_nvidia():
    # Enable FP8 computation
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

def quantize_model(model, dtype=torch.float8_e4m3fn):
    """Quantize model to FP8."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.to(dtype)
```

---

## Memory Profiling

### Debug Utilities

```python
def debug_memory_summary():
    """Generate memory usage report."""
    summary = []
    
    if torch.cuda.is_available():
        summary.append("=== CUDA Memory Summary ===")
        summary.append(torch.cuda.memory_summary())
    
    summary.append("=== Loaded Models ===")
    for i, loaded_model in enumerate(current_loaded_models):
        size_mb = loaded_model.model.model_size() / (1024 * 1024)
        device = loaded_model.device
        summary.append(f"  {i}: {size_mb:.0f}MB on {device}")
    
    summary.append("=== System Memory ===")
    mem = psutil.virtual_memory()
    summary.append(f"  Total: {mem.total / (1024**3):.1f}GB")
    summary.append(f"  Available: {mem.available / (1024**3):.1f}GB")
    summary.append(f"  Percent: {mem.percent}%")
    
    return "\n".join(summary)
```

---

## Z-Image-Turbo Adaptation

### Simplified Memory Manager

```python
class MemoryManager:
    """Lightweight memory management for Z-Image-Turbo."""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = self.detect_device()
        self.vram_state = self.detect_vram_state()
    
    def detect_device(self):
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def detect_vram_state(self):
        """Detect available VRAM."""
        if self.device.type == "cpu":
            return "CPU"
        
        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = total_vram / (1024**3)
            
            if total_vram_gb < 6:
                return "LOW_VRAM"
            elif total_vram_gb < 12:
                return "NORMAL_VRAM"
            else:
                return "HIGH_VRAM"
        
        return "SHARED"
    
    def load_model(self, model_id, model_factory):
        """Load model with automatic offloading."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # Check available memory
        free_memory = self.get_free_memory()
        model_memory = estimate_model_memory(model_id)
        
        if model_memory > free_memory * 0.8:
            # Need to free memory
            self.unload_oldest_model()
        
        # Load model
        model = model_factory()
        
        if self.vram_state == "LOW_VRAM":
            # Enable CPU offloading
            model.enable_sequential_cpu_offload()
        elif self.vram_state == "CPU":
            model = model.to("cpu")
        else:
            model = model.to(self.device)
        
        self.loaded_models[model_id] = {
            "model": model,
            "last_used": time.time(),
            "memory": model_memory
        }
        
        return model
    
    def unload_oldest_model(self):
        """Unload least recently used model."""
        if not self.loaded_models:
            return
        
        oldest_id = min(
            self.loaded_models.keys(),
            key=lambda k: self.loaded_models[k]["last_used"]
        )
        
        del self.loaded_models[oldest_id]
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_free_memory(self):
        """Get free memory in bytes."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated()
        else:
            return psutil.virtual_memory().available
```

### Key Adaptations

1. **Simplified VRAM States**: Fewer states for easier management
2. **LRU Eviction**: Simple least-recently-used model unloading
3. **Automatic Offloading**: Use diffusers' built-in offloading
4. **Memory Estimation**: Pre-emptive OOM prevention
5. **Aggressive Cleanup**: Clear cache after each generation

---

## Related Files

| File | Purpose |
|------|---------|
| `comfy/model_management.py` | Main memory management |
| `comfy/model_patcher.py` | Model patching and offloading |
| `comfy/memory_management.py` | Low-level memory operations |
| `comfy/pinned_memory.py` | Pinned memory for transfers |
