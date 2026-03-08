"""
Execution engine for node-based workflows
Simplified version inspired by ComfyUI's execution.py
"""

import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import OrderedDict, deque
from pathlib import Path


class ExecutionResult:
    """Result of workflow execution."""
    def __init__(self, outputs: Dict[str, Any], execution_time: float, success: bool = True):
        self.outputs = outputs
        self.execution_time = execution_time
        self.success = success
        self.error = None


class ExecutionCache:
    """LRU cache for node execution results."""
    
    def __init__(self, max_size: int = 50):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Any:
        """Get cached result."""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def set(self, key: str, value: Any):
        """Cache result with LRU eviction."""
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.max_size:
            # Evict oldest
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


class ExecutionEngine:
    """
    Simplified execution engine for Z-Image-Turbo workflows.
    Inspired by ComfyUI's execution system.
    """
    
    def __init__(self, cache_enabled: bool = True, max_cache_size: int = 50):
        self.cache_enabled = cache_enabled
        self.cache = ExecutionCache(max_cache_size) if cache_enabled else None
        self.execution_history = []
        self.loaded_models = {}  # Keep track of loaded models
    
    def execute(self, workflow: Dict[str, Dict], 
                progress_callback: Optional[Callable] = None) -> ExecutionResult:
        """
        Execute a workflow.
        
        Args:
            workflow: Dict of node_id -> node_config
                     Format: {
                         "node_id": {
                             "class_type": "NodeName",
                             "inputs": {"input_name": value_or_link},
                             ...
                         }
                     }
            progress_callback: Function(node_id, progress) for updates
        
        Returns:
            ExecutionResult with outputs and timing
        """
        start_time = time.time()
        
        try:
            # 1. Topological sort to determine execution order
            execution_order = self._topological_sort(workflow)
            
            # 2. Execute nodes
            results = {}
            for i, node_id in enumerate(execution_order):
                node_config = workflow[node_id]
                
                if progress_callback:
                    progress_callback(node_id, i / len(execution_order))
                
                # Generate cache key
                cache_key = self._get_cache_key(node_id, node_config, results)
                
                # Check cache
                if self.cache_enabled:
                    cached = self.cache.get(cache_key)
                    if cached is not None:
                        results[node_id] = cached
                        continue
                
                # Execute node
                result = self._execute_node(node_id, node_config, results)
                results[node_id] = result
                
                # Cache result
                if self.cache_enabled:
                    self.cache.set(cache_key, result)
            
            execution_time = time.time() - start_time
            
            # Collect output nodes
            outputs = {}
            for node_id, config in workflow.items():
                if config.get("is_output", False) or self._is_output_node(config):
                    if node_id in results:
                        outputs[node_id] = results[node_id]
            
            return ExecutionResult(outputs, execution_time, success=True)
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = ExecutionResult({}, execution_time, success=False)
            result.error = str(e)
            return result
    
    def _topological_sort(self, workflow: Dict) -> List[str]:
        """
        Kahn's algorithm for topological sorting.
        Determines execution order based on node dependencies.
        """
        # Build adjacency list
        in_degree = {node_id: 0 for node_id in workflow}
        dependents = {node_id: [] for node_id in workflow}
        
        for node_id, config in workflow.items():
            inputs = config.get("inputs", {})
            for input_name, input_value in inputs.items():
                # Check if it's a link [source_node, output_index]
                if isinstance(input_value, list) and len(input_value) == 2:
                    source_node = input_value[0]
                    if source_node in workflow:
                        in_degree[node_id] += 1
                        dependents[source_node].append(node_id)
        
        # Find nodes with no dependencies
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for dependent in dependents[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(workflow):
            raise ValueError("Workflow contains cycles - cannot execute")
        
        return result
    
    def _execute_node(self, node_id: str, config: Dict, 
                      available_results: Dict) -> Any:
        """Execute a single node."""
        from ..nodes.registry import NodeRegistry
        
        node_class = NodeRegistry.get(config["class_type"])
        if node_class is None:
            raise ValueError(f"Unknown node type: {config['class_type']}")
        
        # Resolve inputs
        inputs = {}
        for input_name, input_value in config.get("inputs", {}).items():
            if isinstance(input_value, list) and len(input_value) == 2:
                # Link to another node's output
                source_node, output_idx = input_value
                if source_node not in available_results:
                    raise ValueError(f"Missing dependency: {source_node} for {node_id}")
                source_result = available_results[source_node]
                if isinstance(source_result, tuple):
                    inputs[input_name] = source_result[output_idx]
                else:
                    inputs[input_name] = source_result
            else:
                # Constant value
                inputs[input_name] = input_value
        
        # Create and execute node
        node = node_class()
        func = getattr(node, node_class.FUNCTION, node.execute)
        result = func(**inputs)
        
        # Ensure result is tuple
        if not isinstance(result, tuple):
            result = (result,)
        
        return result
    
    def _get_cache_key(self, node_id: str, config: Dict, 
                       available_results: Dict) -> str:
        """Generate cache key for node execution."""
        key_parts = [config["class_type"]]
        
        for input_name in sorted(config.get("inputs", {}).keys()):
            input_value = config["inputs"][input_name]
            if isinstance(input_value, list) and len(input_value) == 2:
                # For links, use source node's cache key
                source_node = input_value[0]
                key_parts.append(f"{input_name}=@{source_node}")
            else:
                key_parts.append(f"{input_name}={input_value}")
        
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _is_output_node(self, config: Dict) -> bool:
        """Check if node is an output node."""
        from ..nodes.registry import NodeRegistry
        node_class = NodeRegistry.get(config["class_type"])
        if node_class:
            return getattr(node_class, 'OUTPUT_NODE', False)
        return False
    
    def clear_cache(self):
        """Clear execution cache."""
        if self.cache:
            self.cache.clear()
    
    def load_workflow(self, workflow_path: str) -> Dict:
        """Load workflow from JSON file."""
        path = Path(workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_workflow(self, workflow: Dict, workflow_path: str):
        """Save workflow to JSON file."""
        path = Path(workflow_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(workflow, f, indent=2)
