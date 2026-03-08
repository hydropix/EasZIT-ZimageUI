# ComfyUI Sampling Pipeline

## Overview

The sampling pipeline is the core of the image generation process in ComfyUI. It handles the diffusion denoising process, from initial noise to final image, with support for various samplers, schedulers, and conditioning strategies.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAMPLING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                                                              │
│  ├── Model (UNet)                                                    │
│  ├── Positive Conditioning (CLIP encoded prompt)                    │
│  ├── Negative Conditioning                                            │
│  ├── Latent Image (noise or initial image)                          │
│  └── Parameters (steps, cfg, sampler, scheduler, seed)              │
│                                                                      │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    KSAMPLER                                  │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  1. Prepare Sigmas (Scheduler)                        │  │    │
│  │  │     └── [sigma_max, ..., sigma_min, 0]                │  │    │
│  │  ├───────────────────────────────────────────────────────┤  │    │
│  │  │  2. Add Noise (if not disabled)                       │  │    │
│  │  │     └── latent = latent_image * mask + noise * (1-mask)│  │    │
│  │  ├───────────────────────────────────────────────────────┤  │    │
│  │  │  3. Denoising Loop                                    │  │    │
│  │  │     for i in range(steps):                           │  │    │
│  │  │         sigma = sigmas[i]                             │  │    │
│  │  │         sigma_next = sigmas[i+1]                      │  │    │
│  │  │                                                          │  │    │
│  │  │         # CFG Prediction                               │  │    │
│  │  │         noise_pred = calc_cond_batch(                  │  │    │
│  │  │             model, [positive, negative],               │  │    │
│  │  │             latent, sigma, ...                        │  │    │
│  │  │         )                                              │  │    │
│  │  │                                                          │  │    │
│  │  │         # Denoise Step                                 │  │    │
│  │  │         latent = sampler_step(                         │  │    │
│  │  │             latent, noise_pred, sigma, sigma_next     │  │    │
│  │  │         )                                              │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  OUTPUT                                                              │
│  └── Denoised Latent                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. KSampler

Main sampling interface.

```python
class KSampler:
    def __init__(self, model, steps, device, sampler, scheduler, 
                 denoise=1.0, model_options=None):
        self.model = model
        self.steps = steps
        self.device = device
        self.sampler = sampler  # "euler", "dpmpp_2m", etc.
        self.scheduler = scheduler  # "normal", "karras", etc.
        self.denoise = denoise
        self.model_options = model_options or {}
    
    def sample(self, noise, positive, negative, cfg=1.0, 
               latent_image=None, start_step=None, last_step=None,
               force_full_denoise=False, denoise_mask=None,
               sigmas=None, callback=None, disable_pbar=False, seed=None):
        """
        Main sampling function.
        
        Args:
            noise: Initial noise tensor
            positive: Positive conditioning
            negative: Negative conditioning
            cfg: Classifier-free guidance scale
            latent_image: Starting image (for img2img)
            start_step: Start at specific step
            last_step: End at specific step
            denoise_mask: Mask for inpainting
            sigmas: Custom noise schedule
            callback: Progress callback
            seed: Random seed
        
        Returns:
            Denoised latent tensor
        """
        # Get noise schedule
        if sigmas is None:
            sigmas = self.get_sigmas(self.steps)
        
        # Prepare latent
        if latent_image is not None:
            latent = latent_image * denoise_mask + noise * (1 - denoise_mask)
        else:
            latent = noise
        
        # Get sampler function
        sampler_function = self.get_sampler_function(self.sampler)
        
        # Run denoising
        samples = sampler_function(
            model=self.model,
            x=latent,
            sigmas=sigmas,
            extra_args={
                "cond": positive,
                "uncond": negative,
                "cond_scale": cfg,
            },
            callback=callback,
            disable=disable_pbar,
        )
        
        return samples
```

### 2. Schedulers

Generate noise schedules (sigma values).

```python
# Simple scheduler (evenly spaced)
def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs.append(float(s.sigmas[-(1 + int(x * ss))]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)

# Karras scheduler (log-space spacing)
def karras_scheduler(n, sigma_min, sigma_max, rho=7.):
    """Karras et al. 2022 noise schedule."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas

# Exponential scheduler
def exponential_scheduler(n, sigma_min, sigma_max):
    sigmas = torch.exp(torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n
    ))
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas

# Beta scheduler (for Turbo models)
def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    """Beta distribution scheduler for few-step sampling."""
    total_timesteps = len(model_sampling.sigmas) - 1
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    
    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs.append(float(model_sampling.sigmas[int(t)]))
        last_t = t
    sigs.append(0.0)
    return torch.FloatTensor(sigs)
```

### 3. Samplers

Denoising algorithms.

```python
# Euler sampler
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Euler method - simple first-order sampler."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Predict noise
        denoised = model(x, sigma * s_in, **extra_args)
        
        # Callback for progress
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        # Euler step
        d = (x - denoised) / sigma
        dt = sigma_next - sigma
        x = x + d * dt
    
    return x

# Euler ancestral (adds stochasticity)
def sample_euler_ancestral(model, x, sigmas, extra_args=None, 
                           callback=None, disable=None, eta=1.0):
    """Euler method with ancestral sampling."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_hat = sigma * (1 + eta * (sigma_next / sigma - 1))
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        # Add noise
        eps = (x - denoised) / sigma
        x = denoised + (sigma_hat ** 2 - sigma_next ** 2) ** 0.5 * eps
        
        if sigma_next != 0:
            x = x + sigma_next * torch.randn_like(x)
    
    return x

# DPM++ 2M Karras (high quality, 2nd order)
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM++ 2M sampler - good quality, fast."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        # 2nd order update
        if old_denoised is not None and sigma_next != 0:
            t, t_next = sigma, sigma_next
            h = t_next - t
            r = h / (t - sigmas[i - 1]) if i > 0 else 0
            
            x = (t_next / t) * x - (2 * h) * denoised + h * old_denoised
        else:
            # 1st order fallback
            x = (sigma_next / sigma) * x - (sigma_next - sigma) * denoised
        
        old_denoised = denoised
    
    return x

# DDIM (deterministic, good for img2img)
def sample_ddim(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.0):
    """DDIM sampling - deterministic when eta=0."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'denoised': denoised})
        
        # DDIM step
        alpha = 1 / (1 + sigma ** 2) ** 0.5
        alpha_next = 1 / (1 + sigma_next ** 2) ** 0.5
        
        sigma_eta = eta * (sigma_next ** 2 - sigma ** 2 * alpha_next ** 2 / alpha ** 2) ** 0.5
        
        x = alpha_next * denoised + (1 - alpha_next ** 2 - sigma_eta ** 2) ** 0.5 * (x - alpha * denoised) / (1 - alpha ** 2) ** 0.5
        
        if sigma_next != 0 and eta > 0:
            x = x + sigma_eta * torch.randn_like(x)
    
    return x
```

### 4. Conditioning

Handle positive and negative prompts.

```python
def calc_cond_batch(model, conds, x_in, timestep, model_options):
    """
    Calculate conditioning for a batch.
    
    Args:
        model: The diffusion model
        conds: List of [positive_conds, negative_conds]
        x_in: Input latent
        timestep: Current sigma/timestep
        model_options: Additional options
    
    Returns:
        List of conditioned outputs
    """
    out_conds = []
    out_counts = []
    
    for i, cond in enumerate(conds):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)
        
        if cond is not None:
            for cond_dict in cond:
                # Get area and mask
                area = cond_dict.get('area')
                strength = cond_dict.get('strength', 1.0)
                mask = cond_dict.get('mask')
                
                # Process conditioning
                model_conds = cond_dict["model_conds"]
                conditioning = {}
                for c in model_conds:
                    conditioning[c] = model_conds[c].process_cond(
                        batch_size=x_in.shape[0], 
                        area=area
                    )
                
                # Get control if present
                control = cond_dict.get('control')
                if control is not None:
                    conditioning['control'] = control.get_control(
                        x_in, timestep, conditioning, ...
                    )
                
                # Apply model
                output = model.apply_model(x_in, timestep, **conditioning)
                
                # Apply area/mask weighting
                if area is not None:
                    # Weight by area
                    out_conds[i] += output * strength
                    out_counts[i] += strength
                else:
                    out_conds[i] += output * strength
                    out_counts[i] += strength
    
    # Normalize
    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]
    
    return out_conds
```

### 5. CFG (Classifier-Free Guidance)

```python
def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, 
                 timestep, model_options=None, cond=None, uncond=None):
    """
    Apply classifier-free guidance.
    
    formula: output = uncond + (cond - uncond) * cond_scale
    """
    if model_options is None:
        model_options = {}
    
    # Custom CFG function
    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        # Standard CFG
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    
    # Post-CFG hooks
    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "cond_scale": cond_scale,
            "sigma": timestep,
        }
        cfg_result = fn(args)
    
    return cfg_result
```

---

## Sampling Configurations

### Text-to-Image

```python
# Standard configuration
t2i_config = {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20,
    "cfg": 7.0,
    "seed": 42,
    "width": 512,
    "height": 512,
}

# Turbo configuration (few steps)
turbo_config = {
    "sampler": "dpmpp_sde",
    "scheduler": "karras",
    "steps": 8,  # Fewer steps for Turbo models
    "cfg": 1.0,  # Lower CFG for Turbo
    "seed": 42,
}

# High quality configuration
quality_config = {
    "sampler": "dpmpp_2m",
    "scheduler": "karras",
    "steps": 30,
    "cfg": 7.5,
    "seed": 42,
}
```

### Image-to-Image

```python
# Standard img2img
img2img_config = {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20,
    "cfg": 7.0,
    "denoise": 0.75,  # Strength of transformation
    "seed": 42,
}

# Subtle changes
subtle_config = {
    "denoise": 0.4,  # Keep more of original
    "steps": 15,
}

# Major transformation
major_config = {
    "denoise": 0.9,  # Almost full redenoising
    "steps": 25,
}
```

### Inpainting

```python
# Inpainting configuration
inpaint_config = {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20,
    "cfg": 7.0,
    "denoise": 1.0,
    "use_mask": True,
    "mask_blur": 8,
}
```

---

## Advanced Features

### Regional Conditioning

```python
# Different prompts for different areas
regional_conds = [
    {
        "prompt": "blue sky",
        "area": (256, 512, 0, 0),  # (height, width, y, x)
        "strength": 1.0,
    },
    {
        "prompt": "green grass",
        "area": (256, 512, 256, 0),
        "strength": 1.0,
    },
]
```

### Masked Conditioning

```python
# Apply prompt only to masked region
masked_cond = {
    "prompt": "red car",
    "mask": mask_tensor,  # [B, H, W]
    "mask_strength": 1.0,
}
```

### Timestep-based Conditioning

```python
# Prompt active only during specific steps
scheduled_cond = {
    "prompt": "detailed texture",
    "start_percent": 0.0,  # Start at beginning
    "end_percent": 0.5,    # End at 50% of steps
}
```

---

## Z-Image-Turbo Integration

### Simplified Sampler Interface

```python
class SimpleSampler:
    """Lightweight sampler for Z-Image-Turbo."""
    
    def __init__(self, model, scheduler="simple", sampler="euler"):
        self.model = model
        self.scheduler = scheduler
        self.sampler = sampler
        self.device = model.device
    
    @torch.no_grad()
    def sample(self, prompt, negative_prompt=None, 
               width=1024, height=1024, steps=9, 
               cfg=0.0, seed=None, latent=None):
        """
        Simple sampling interface.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (optional)
            width: Image width
            height: Image height
            steps: Number of steps
            cfg: Guidance scale
            seed: Random seed
            latent: Initial latent (for img2img)
        
        Returns:
            Generated image tensor
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode prompts
        positive = self.encode_prompt(prompt)
        negative = self.encode_prompt(negative_prompt or "")
        
        # Prepare latent
        if latent is None:
            latent = torch.randn(
                1, 4, height // 8, width // 8,
                device=self.device, dtype=self.model.dtype
            )
        
        # Get sigmas
        sigmas = self.get_sigmas(steps)
        
        # Add noise if img2img
        if latent is not None and steps < len(sigmas):
            noise = torch.randn_like(latent)
            latent = latent * (1 - denoise) + noise * denoise
            sigmas = sigmas[-(steps + 1):]
        
        # Sampling loop
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            # Predict noise
            noise_pred = self.predict_noise(latent, sigma, positive, negative, cfg)
            
            # Denoise step
            latent = self.denoise_step(latent, noise_pred, sigma, sigma_next)
        
        # Decode to image
        image = self.decode_latent(latent)
        
        return image
    
    def predict_noise(self, latent, sigma, positive, negative, cfg):
        """Predict noise with CFG."""
        # Conditional prediction
        cond_pred = self.model(latent, sigma, positive)
        
        if cfg > 1.0:
            # Unconditional prediction
            uncond_pred = self.model(latent, sigma, negative)
            # CFG
            return uncond_pred + (cond_pred - uncond_pred) * cfg
        else:
            return cond_pred
    
    def denoise_step(self, latent, noise_pred, sigma, sigma_next):
        """Apply Euler denoising step."""
        # Euler method
        d = (latent - noise_pred) / sigma
        dt = sigma_next - sigma
        return latent + d * dt
```

### Turbo-Optimized Configuration

```python
# Z-Image-Turbo optimized settings
ZIMAGE_TURBO_CONFIG = {
    # Model-specific
    "model_id": "Tongyi-MAI/Z-Image-Turbo",
    "dtype": "bfloat16",
    "attention_backend": "sdpa",
    
    # Sampling
    "steps": 9,           # ~8 forward passes
    "cfg": 0.0,           # No CFG needed for Turbo
    "sampler": "euler",
    "scheduler": "simple",
    
    # Resolution
    "width": 1024,
    "height": 1024,
    
    # Optimization
    "compile_model": True,
    "enable_cpu_offload": False,  # Auto-detect based on VRAM
    "batch_size": 1,
}
```

---

## Sampler Comparison

| Sampler | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| euler | Fast | Good | General purpose |
| euler_ancestral | Fast | Good | Creative/stochastic |
| dpmpp_2m | Medium | Excellent | High quality |
| dpmpp_sde | Medium | Excellent | Turbo models |
| ddim | Slow | Good | img2img (deterministic) |
| uni_pc | Fast | Good | Few steps |

---

## Related Files

| File | Purpose |
|------|---------|
| `comfy/sample.py` | Main sampling interface |
| `comfy/samplers.py` | Sampler implementations |
| `comfy/sampler_helpers.py` | Sampling utilities |
| `comfy/k_diffusion/sampling.py` | K-diffusion samplers |
| `comfy/extra_samplers/` | Additional samplers |
