# ⚡ Z-Image-Turbo Web UI

Web interface for generating images with **Z-Image-Turbo**, the 6 billion parameter diffusion model developed by Alibaba/Tongyi. Optimized to run on a PC with GPU and accessible remotely via network.

## 🎯 Features

- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B parameters)
- **Speed**: Generation in ~8 steps (a few seconds)
- **Quality**: Photorealistic rendering with bilingual text support (EN/ZH)
- **Interface**: Modern and intuitive Gradio UI
- **Access**: Local + local network + public tunnel (optional)
- **Optimizations**: Flash Attention, torch compilation, CPU offloading

## 📋 Prerequisites

### Minimum Configuration
- **GPU**: NVIDIA with 8GB+ VRAM (16GB recommended)
- **RAM**: 16GB system
- **OS**: Windows 10/11, Linux, macOS (CPU slow)
- **Python**: 3.9 - 3.12

### Recommended Configuration
- **GPU**: RTX 3090/4090 (24GB VRAM)
- **RAM**: 32GB
- **CUDA**: 12.1+

## 🚀 Installation

### 1. Clone or create the project

```bash
cd Z-Image-Turbo-WebUI
```

### 2. Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Installing `diffusers` from Git may take a few minutes.

### 4. (Optional) Install Flash Attention

For additional acceleration on recent GPUs:

```bash
pip install flash-attn --no-build-isolation
```

> ⚠️ Requires CUDA toolkit and may take a long time to compile.

## 🎮 Usage

### Simple Launch

```bash
python launch.py
```

The interface will be accessible at:
- Local: http://localhost:7860
- Network: http://[PC_IP]:7860

### Advanced Options

```bash
# Load model automatically at startup
python launch.py --load-on-start

# Change port
python launch.py --port 8080

# Create public link (72h max)
python launch.py --share

# Enable authentication
python launch.py --auth admin:yourpassword

# Combine options
python launch.py --load-on-start --port 8080 --auth user:pass
```

### Access from another PC

1. On the PC with GPU (server):
```bash
python launch.py --host 0.0.0.0 --port 7860
```

2. On the other PC (client), open the browser and go to:
```
http://[SERVER_IP]:7860
```

To find the server IP:
- Windows: `ipconfig` in CMD
- Linux: `ip addr` or `hostname -I`

## 🖼️ Interface

### Generation Tab
- **Prompt**: Description of the desired image
- **Negative Prompt**: Elements to avoid
- **Dimensions**: Width/Height (512-2048)
- **Steps**: 1-50 (9 recommended for Turbo)
- **Guidance Scale**: 0.0 recommended for Turbo
- **Seed**: To reproduce results
- **Batch Size**: Generate multiple images

### Model Management Tab
- Load/unload model
- Configure dtype (bfloat16/float16/float32)
- Attention backend (SDPA/Flash Attention)
- CPU offloading for GPUs with limited VRAM
- Torch compilation for acceleration

### Prompt Examples

1. Traditional Chinese portrait:
```
Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, 
red floral forehead pattern. Elaborate high bun, golden phoenix headdress, 
red flowers, beads. Soft-lit outdoor night background, silhouetted tiered 
pagoda, blurred colorful distant lights.
```

2. Cyberpunk:
```
Futuristic cyberpunk city at night, neon signs in Chinese and English, 
flying vehicles, rain-slicked streets reflecting colorful lights, 
cinematic composition
```

3. Nature:
```
A serene Japanese garden with cherry blossoms in full bloom, a small 
wooden bridge over a koi pond, soft morning light filtering through 
the trees, photorealistic
```

## ⚙️ Advanced Configuration

### Environment Variables

```bash
# Force device
export CUDA_VISIBLE_DEVICES=0  # Specific GPU

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Compilation
export TORCH_COMPILE_BACKEND=inductor
```

### Direct Usage (without launcher)

```bash
python app.py --help

# All options
python app.py --host 0.0.0.0 --port 7860 --load-on-start --share
```

## 🔧 Troubleshooting

### "CUDA out of memory"
- Enable "CPU Offloading" in the interface
- Reduce resolution (512x512)
- Use `float16` instead of `bfloat16`
- Close other applications using the GPU

### "No module named 'diffusers'"
```bash
pip install git+https://github.com/huggingface/diffusers
```

### Model very slow
- Verify CUDA is being used: `torch.cuda.is_available()`
- Enable model compilation
- Try Flash Attention if compatible

### Cannot access from network
- Check Windows firewall (allow Python)
- Use `--host 0.0.0.0`
- Check that port is not in use: `netstat -an | findstr 7860`

## 📁 Project Structure

```
Z-Image-Turbo-WebUI/
├── app.py              # Main Gradio application
├── launch.py           # Launch script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── outputs/           # Generated images
│   └── images/
└── venv/              # Virtual environment
```

## 🌐 Advanced Remote Access

### With ngrok (secure tunnel)

1. Install ngrok: https://ngrok.com/download
2. Create a free account and configure token
3. Launch:
```bash
ngrok http 7860
```

### With Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:7860
```

### With Tailscale (mesh VPN)

Ideal for secure access between your devices without public exposure.

## 📚 Resources

- [Z-Image-Turbo on HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Gradio Documentation](https://gradio.app/docs)

## 📝 License

This project uses Z-Image-Turbo under Apache 2.0 license.

## 🙏 Credits

- Model: [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) (Alibaba)
- UI: Gradio
- Pipeline: HuggingFace Diffusers
