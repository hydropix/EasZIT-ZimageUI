"""
Base node class and type system for Z-Image-Turbo
Inspired by ComfyUI's node system
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
from enum import Enum


class ZType(str, Enum):
    """Node data types - matching ComfyUI conventions."""
    IMAGE = "IMAGE"
    LATENT = "LATENT"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    MASK = "MASK"
    SAMPLER = "SAMPLER"
    SCHEDULER = "SCHEDULER"
    ANY = "*"


class ZNode(ABC):
    """
    Base class for all Z-Image-Turbo nodes.
    Similar to ComfyUI's node base class.
    """
    
    # Node metadata
    NAME: str = ""
    CATEGORY: str = "zimage"
    DESCRIPTION: str = ""
    
    # Input/output definition
    INPUT_TYPES: Dict = {"required": {}}
    RETURN_TYPES: Tuple[ZType, ...] = ()
    RETURN_NAMES: Tuple[str, ...] = ()
    
    # Execution
    FUNCTION: str = "execute"
    OUTPUT_NODE: bool = False
    
    def __init__(self):
        self.id = None
        self.inputs = {}
        self.outputs = {}
    
    @abstractmethod
    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """Execute node logic. Must be implemented by subclasses."""
        pass
    
    def validate_inputs(self, inputs: Dict) -> Optional[str]:
        """Validate inputs before execution. Return error message or None."""
        return None
    
    def is_changed(self, **kwargs) -> Any:
        """
        Return value that changes when node should re-execute.
        Used for caching - if this value matches cached, skip execution.
        """
        return hash(str(kwargs))
    
    @classmethod
    def get_input_types(cls):
        """Get all input types including required, optional, and hidden."""
        result = {}
        if "required" in cls.INPUT_TYPES:
            result["required"] = cls.INPUT_TYPES["required"]
        if "optional" in cls.INPUT_TYPES:
            result["optional"] = cls.INPUT_TYPES["optional"]
        if "hidden" in cls.INPUT_TYPES:
            result["hidden"] = cls.INPUT_TYPES["hidden"]
        return result


def register_node(name: str = None, category: str = "zimage"):
    """Decorator to register a node class."""
    def decorator(cls):
        cls.NAME = name or cls.__name__
        cls.CATEGORY = category
        from .registry import NodeRegistry
        NodeRegistry.register(cls.NAME, cls)
        return cls
    return decorator
