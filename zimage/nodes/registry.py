"""
Node registry with support for custom nodes
Inspired by ComfyUI's node registration system
"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, Type, Optional
from pathlib import Path


class NodeRegistry:
    """Global node registry with custom node support."""
    
    _nodes: Dict[str, Type] = {}
    _categories: Dict[str, list] = {}
    
    @classmethod
    def register(cls, name: str, node_class: Type):
        """Register a node class."""
        cls._nodes[name] = node_class
        category = getattr(node_class, 'CATEGORY', 'zimage')
        if category not in cls._categories:
            cls._categories[category] = []
        if name not in cls._categories[category]:
            cls._categories[category].append(name)
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a node class by name."""
        return cls._nodes.get(name)
    
    @classmethod
    def list_nodes(cls) -> Dict[str, Type]:
        """List all registered nodes."""
        return cls._nodes.copy()
    
    @classmethod
    def list_categories(cls) -> Dict[str, list]:
        """List nodes grouped by category."""
        return cls._categories.copy()
    
    @classmethod
    def load_custom_nodes(cls, custom_nodes_path: str = "custom_nodes"):
        """
        Load custom nodes from directory.
        Similar to ComfyUI's custom node loading.
        """
        if not os.path.exists(custom_nodes_path):
            return
        
        custom_nodes_dir = Path(custom_nodes_path)
        
        # Add custom_nodes to path
        if str(custom_nodes_dir) not in sys.path:
            sys.path.insert(0, str(custom_nodes_dir))
        
        for item in custom_nodes_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                # Try to load as a package
                init_file = item / "__init__.py"
                if init_file.exists():
                    try:
                        module = importlib.import_module(item.name)
                        print(f"Loaded custom node package: {item.name}")
                    except Exception as e:
                        print(f"Failed to load custom node {item.name}: {e}")
                else:
                    # Try to load .py files
                    for py_file in item.glob("*.py"):
                        if not py_file.name.startswith('__'):
                            try:
                                spec = importlib.util.spec_from_file_location(
                                    py_file.stem, py_file
                                )
                                module = importlib.util.module_from_spec(spec)
                                sys.modules[py_file.stem] = module
                                spec.loader.exec_module(module)
                                print(f"Loaded custom node: {py_file.stem}")
                            except Exception as e:
                                print(f"Failed to load custom node {py_file}: {e}")
            elif item.is_file() and item.suffix == '.py' and not item.name.startswith('__'):
                # Load single .py file
                try:
                    spec = importlib.util.spec_from_file_location(
                        item.stem, item
                    )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[item.stem] = module
                    spec.loader.exec_module(module)
                    print(f"Loaded custom node: {item.stem}")
                except Exception as e:
                    print(f"Failed to load custom node {item}: {e}")


def register_node(name: str = None, category: str = "zimage"):
    """Decorator to register a node class."""
    def decorator(cls):
        cls.NAME = name or cls.__name__
        cls.CATEGORY = category
        NodeRegistry.register(cls.NAME, cls)
        return cls
    return decorator
