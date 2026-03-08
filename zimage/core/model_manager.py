"""
Model management with caching and memory optimization
Simplified version inspired by ComfyUI's model_management
"""

import gc
from typing import Dict, Any, Callable, Optional
from collections import OrderedDict


class ModelManager:
    """
    Manages model loading with LRU caching.
    Simplified version of ComfyUI's model management.
    """
    
    _instance = None
    _models: OrderedDict = OrderedDict()
    _max_cache_size: int = 2  # Keep max 2 models in VRAM
    _current_device: str = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detect_device()
        return cls._instance
    
    def _detect_device(self):
        """Detect best available device."""
        import torch
        if torch.cuda.is_available():
            self._current_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._current_device = "mps"
        else:
            self._current_device = "cpu"
    
    @property
    def device(self) -> str:
        return self._current_device
    
    def get_free_memory(self) -> int:
        """Get free VRAM/RAM in bytes."""
        import torch
        if self._current_device == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.get_free_memory(self._current_device)
        return 8 * 1024 * 1024 * 1024  # Assume 8GB for CPU
    
    def load_model(self, model_key: str, factory: Callable, device: str = None) -> Any:
        """
        Load a model with caching.
        
        Args:
            model_key: Unique identifier for the model
            factory: Function that creates/loads the model
            device: Device to load model on (defaults to detected device)
        
        Returns:
            Loaded model
        """
        if device is None:
            device = self._current_device
        
        # Check cache
        if model_key in self._models:
            # Move to end (most recently used)
            model = self._models.pop(model_key)
            self._models[model_key] = model
            return model
        
        # Evict oldest if cache is full
        while len(self._models) >= self._max_cache_size:
            oldest_key = next(iter(self._models))
            oldest_model = self._models.pop(oldest_key)
            del oldest_model
            self._cleanup_memory()
        
        # Load new model
        model = factory()
        
        # Move to device if needed
        if hasattr(model, 'to'):
            model = model.to(device)
        
        self._models[model_key] = model
        return model
    
    def unload_model(self, model_key: str):
        """Unload a specific model from cache."""
        if model_key in self._models:
            model = self._models.pop(model_key)
            del model
            self._cleanup_memory()
    
    def unload_all(self):
        """Unload all models."""
        for model in self._models.values():
            del model
        self._models.clear()
        self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up GPU/CPU memory."""
        import torch
        gc.collect()
        if self._current_device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def is_cached(self, model_key: str) -> bool:
        """Check if model is in cache."""
        return model_key in self._models


def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance."""
    return ModelManager()
