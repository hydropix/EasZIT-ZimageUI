# Z-Image-Turbo Web UI - Agent Documentation

## Project Overview

Z-Image-Turbo Web UI is a web interface for generating images using **Z-Image-Turbo**, a 6 billion parameter diffusion model developed by Alibaba/Tongyi. The application provides a modern Gradio-based UI that runs locally on a PC with GPU and can be accessed remotely via network.

**Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
**Key Features**:
- Fast generation in ~8 steps (few seconds)
- Photorealistic rendering with bilingual text support (EN/ZH)
- Optimizations: Flash Attention, torch compilation, CPU offloading
- Network accessible (local + LAN + public tunnel)

---

## Technology Stack

| Component | Version/Requirement |
|-----------|---------------------|
| Python | 3.9 - 3.12 |
| PyTorch | 2.6.0 (with CUDA 12.4) |
| Gradio | >= 5.20.0 |
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
├── app.py              # Main Gradio application - UI and image generation logic
├── launch.py           # Launcher script with pre-flight checks
├── start-zimage.bat    # Windows batch script for one-click setup and launch
├── diagnose.py         # Diagnostic script for troubleshooting installation
├── requirements.txt    # Python dependencies
├── README.md          # User documentation (English)
├── AGENTS.md          # This file - AI agent documentation
├── .gitignore         # Git ignore patterns
├── outputs/           # Generated images directory (created at runtime)
│   └── images/
└── venv/              # Virtual environment (created by setup scripts)
```

---

## Build and Run Commands

### Quick Start (Windows)
```bash
# One-click setup and launch (handles venv, PyTorch CUDA, dependencies)
start-zimage.bat
```

### Manual Launch (All Platforms)
```bash
# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
python launch.py
```

### Launch Options
```bash
python launch.py --load-on-start    # Auto-load model at startup
python launch.py --port 8080        # Change port (default: 7860)
python launch.py --share            # Create public link (72h max)
python launch.py --auth user:pass   # Enable authentication
python launch.py --host 0.0.0.0     # Allow network access
```

### Direct Launch (without launcher checks)
```bash
python app.py [options]
```

### Diagnostics
```bash
python diagnose.py    # Run installation diagnostics
```

---

## Code Organization

### Main Modules

**`app.py`** (~575 lines):
- Global configuration constants (MODEL_ID, DEFAULT_WIDTH/HEIGHT, OUTPUT_DIR)
- Model management functions:
  - `load_model()` - Loads Z-Image-Turbo with dtype/attention/compilation options
  - `unload_model()` - Frees GPU memory
  - `get_available_device()` - Detects CUDA/MPS/CPU
- Image generation: `generate_image()` - Main inference pipeline
- System info: `get_system_info()` - PyTorch/CUDA/GPU details
- UI creation: `create_ui()` - Gradio interface with 3 tabs:
  - 🎨 Génération: Main generation interface (prompt, parameters, output)
  - ⚙️ Gestion du Modèle: Model load/unload and configuration
  - 🔬 Paramètres avancés: Advanced settings and aspect ratio shortcuts
- Entry point: `main()` - Argument parsing and server launch

**`launch.py`** (~156 lines):
- `get_local_ip()` - Gets local IP for network access display
- `check_cuda()` - Verifies PyTorch CUDA availability
- `main()` - Pre-flight checks, dependency verification, launches app.py

**`start-zimage.bat`** (Windows):
- Automated setup: Python check → venv creation → PyTorch CUDA install → dependency install
- Cleanup of corrupted distributions
- Launches: `python launch.py --load-on-start`

**`diagnose.py`**:
- Comprehensive diagnostic tool for troubleshooting
- Checks Python version, virtual environment, all dependencies, CUDA/GPU, project files
- Provides summary with actionable next steps

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
- Global variables for model state: `pipe`, `model_loaded`, `model_loading`
- French naming for UI labels and user messages
- English for technical/internal variable names
- Docstrings in French
- Type hints not used

### Output Handling
- Generated images saved to: `outputs/images/zimage_{timestamp}_{uuid}_{index}.png`
- Timestamps use format: `%Y%m%d_%H%M%S`
- UUID truncated to 8 characters for filename

---

## Configuration Options

### Model Loading Options (UI)
| Option | Choices | Description |
|--------|---------|-------------|
| dtype | bfloat16, float16, float32 | Precision for model weights |
| attention_backend | sdpa, flash, flash3 | Attention mechanism (Flash Attention requires separate install) |
| enable_cpu_offload | checkbox | For GPUs with limited VRAM |
| compile_model | checkbox | torch.compile for +30% performance (experimental) |

### Generation Parameters
| Parameter | Range | Default | Note |
|-----------|-------|---------|------|
| width | 512-2048 | 1024 | Step 64 |
| height | 512-2048 | 1024 | Step 64 |
| num_steps | 1-50 | 9 | 9 = ~8 forwards DiT (recommended for Turbo) |
| guidance_scale | 0.0-10.0 | 0.0 | 0.0 recommended for Turbo |
| seed | -1 or int | -1 | -1 = random |
| batch_size | 1-4 | 1 | Number of images per generation |

### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0              # Force specific GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
TORCH_COMPILE_BACKEND=inductor      # Compilation backend
```

---

## Testing Strategy

**No formal test suite exists.** Testing is manual:

1. **Diagnostic script**: `python diagnose.py` - Verifies installation
2. **Launch verification**: `python launch.py` - Checks all components
3. **Generation test**: Load model → Generate image with default settings
4. **Memory monitoring**: Check CUDA memory during generation

### Manual Test Checklist
- [ ] Diagnostic script shows all ✅
- [ ] Model loads without errors
- [ ] Image generates in < 30 seconds (GPU) or completes (CPU)
- [ ] Output saved to outputs/images/
- [ ] Network access works (if `--host 0.0.0.0` used)

---

## Deployment Considerations

### Network Access
- Default binds to `0.0.0.0:7860` (accessible from LAN)
- Use `--host 127.0.0.1` for localhost-only
- Firewall: Allow Python through Windows Defender

### Security
- No authentication by default
- Use `--auth user:pass` for basic auth
- SSL support via `--ssl-cert` and `--ssl-key`
- Public sharing via `--share` creates temporary Gradio link (72h max)

### Memory Management
- Model downloads ~13GB from HuggingFace on first load
- Generated images accumulate in `outputs/images/` (not auto-cleaned)
- Use "Décharger le modèle" button to free VRAM
- CPU offloading available for 8GB VRAM GPUs

### Error Handling
- Model not loaded → Clear error message in UI
- CUDA OOM → Suggest CPU offloading, reduce resolution, use float16
- Missing dependencies → Diagnostic script identifies missing packages

---

## Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Enable CPU offloading, reduce to 512x512, use float16 |
| "No module named 'diffusers'" | Run `pip install -r requirements.txt` |
| Model very slow | Verify CUDA available with `torch.cuda.is_available()` |
| Cannot access from network | Check firewall, use `--host 0.0.0.0`, verify port not in use |

---

## External Dependencies

- **Model**: Auto-downloaded from HuggingFace on first run (Tongyi-MAI/Z-Image-Turbo)
- **Flash Attention**: Optional, requires separate install with CUDA toolkit (slow compilation)
- **Ngrok/Cloudflare**: Optional for secure tunneling (not included in requirements)
