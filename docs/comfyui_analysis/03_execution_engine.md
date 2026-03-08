# ComfyUI Execution Engine

## Overview

The execution engine is responsible for orchestrating node execution according to the workflow graph's dependencies. It implements a **topological sort** with advanced features like caching, lazy evaluation, and async execution.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXECUTION FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Workflow JSON                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Parse &        │───▶│  Validate       │───▶│  Build Graph    │  │
│  │  Normalize      │    │  Inputs         │    │  Topology       │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                        │             │
│       ┌────────────────────────────────────────────────┘             │
│       ▼                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Check Cache    │───▶│  Execute Node   │───▶│  Store Result   │  │
│  │  (Hierarchical) │◀───│  (Async)        │    │  in Cache       │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│         │                     ▲                                      │
│         └─────────────────────┘                                      │
│              (Next Node)                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. PromptExecutor

Main entry point for workflow execution.

```python
class PromptExecutor:
    def __init__(self, server, cache_type=CacheType.CLASSIC, cache_args=None):
        self.server = server           # WebSocket for progress updates
        self.caches = CacheSet(...)    # Output and object caches
        self.cache_type = cache_type   # Caching strategy
        
    def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        """Execute a workflow."""
        asyncio.run(self.execute_async(prompt, prompt_id, extra_data, execute_outputs))
    
    async def execute_async(self, prompt, prompt_id, extra_data, execute_outputs):
        """Async execution with progress tracking."""
        # 1. Setup
        dynamic_prompt = DynamicPrompt(prompt)
        reset_progress_state(prompt_id, dynamic_prompt)
        
        # 2. Initialize caches
        for cache in self.caches.all:
            await cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
            cache.clean_unused()
        
        # 3. Build execution list
        execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)
        for node_id in execute_outputs:
            execution_list.add_node(node_id)
        
        # 4. Execute nodes
        while not execution_list.is_empty():
            node_id, error, ex = await execution_list.stage_node_execution()
            result, error, ex = await execute(
                self.server, dynamic_prompt, self.caches, 
                node_id, extra_data, executed, prompt_id, 
                execution_list, pending_subgraph_results, 
                pending_async_nodes, ui_node_outputs
            )
            # Handle result...
```

### 2. DynamicPrompt

Mutable graph representation that supports runtime modifications.

```python
class DynamicPrompt:
    def __init__(self, original_prompt):
        self.original_prompt = original_prompt    # Static user workflow
        self.ephemeral_prompt = {}                 # Runtime-created nodes
        self.ephemeral_parents = {}                # Parent relationships
        self.ephemeral_display = {}                # Display node mapping
    
    def get_node(self, node_id):
        """Get node definition (static or ephemeral)."""
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        return self.original_prompt[node_id]
    
    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        """Add a node created during execution (e.g., by expansion)."""
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id
    
    def get_real_node_id(self, node_id):
        """Trace back to original node (for error reporting)."""
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id
```

### 3. ExecutionList

Manages the execution queue with topological ordering.

```python
class ExecutionList(TopologicalSort):
    """
    Implements topological dissolve of the graph.
    Nodes can be staged, then returned if dependencies are added.
    """
    
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None
        self.execution_cache = {}  # Per-listener cache
    
    def add_node(self, node_unique_id, include_lazy=False):
        """Add node and its dependencies to the execution graph."""
        # Recursively add all upstream dependencies
        node_ids = [node_unique_id]
        while node_ids:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue
            
            self.pendingNodes[unique_id] = True
            self.blockCount[unique_id] = 0
            self.blocking[unique_id] = {}
            
            # Add dependencies
            inputs = self.dynprompt.get_node(unique_id)["inputs"]
            for input_name, value in inputs.items():
                if is_link(value):  # Connected to another node
                    from_node_id = value[0]
                    if not self.is_cached(from_node_id):
                        node_ids.append(from_node_id)
                    self.add_strong_link(from_node_id, input_name, unique_id)
    
    async def stage_node_execution(self):
        """Get next ready node for execution."""
        available = self.get_ready_nodes()  # Nodes with 0 block count
        
        if not available:
            # Check for cycles
            cycled_nodes = self.get_nodes_in_cycle()
            if cycled_nodes:
                raise DependencyCycleError("Cycle detected")
        
        # UX-friendly selection: prefer output nodes, then async nodes
        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None
    
    def complete_node_execution(self):
        """Mark staged node as complete."""
        node_id = self.staged_node_id
        self.pop_node(node_id)  # Remove and unblock dependents
        self.staged_node_id = None
    
    def unstage_node_execution(self):
        """Return staged node to pending (for lazy evaluation)."""
        self.staged_node_id = None
```

### 4. TopologicalSort

Base class for dependency resolution.

```python
class TopologicalSort:
    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self.pendingNodes = {}   # Nodes waiting to execute
        self.blockCount = {}     # Number of blocking dependencies
        self.blocking = {}       # Which nodes each node blocks
        self.externalBlocks = 0  # Async operation blocks
    
    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        """Create a dependency edge."""
        if not self.is_cached(from_node_id):
            self.add_node(from_node_id)
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}
                self.blockCount[to_node_id] += 1
            self.blocking[from_node_id][to_node_id][from_socket] = True
    
    def pop_node(self, unique_id):
        """Remove node and unblock its dependents."""
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]
    
    def get_ready_nodes(self):
        """Return nodes with no remaining dependencies."""
        return [node_id for node_id in self.pendingNodes 
                if self.blockCount[node_id] == 0]
```

---

## Execution Flow Details

### Main Execute Function

```python
async def execute(server, dynprompt, caches, current_item, extra_data, 
                  executed, prompt_id, execution_list, pending_subgraph_results,
                  pending_async_nodes, ui_outputs):
    """
    Execute a single node.
    
    Returns: (ExecutionResult, error_details, exception)
    """
    unique_id = current_item
    node = dynprompt.get_node(unique_id)
    class_type = node["class_type"]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    
    # 1. Check cache
    cached = caches.outputs.get(unique_id)
    if cached is not None:
        # Return cached result immediately
        server.send_sync("executed", { "node": unique_id, ... })
        execution_list.cache_update(unique_id, cached)
        return (ExecutionResult.SUCCESS, None, None)
    
    # 2. Gather inputs
    input_data_all, missing_keys, v3_data = get_input_data(
        node["inputs"], class_def, unique_id, execution_list, dynprompt, extra_data
    )
    
    # 3. Handle lazy evaluation
    if class_def has check_lazy_status:
        required_inputs = await check_lazy_status(...)
        if missing required inputs:
            execution_list.make_input_strong_link(unique_id, required_input)
            return (ExecutionResult.PENDING, None, None)
    
    # 4. Create/get node instance
    obj = caches.objects.get(unique_id)
    if obj is None:
        obj = class_def()
        caches.objects.set(unique_id, obj)
    
    # 5. Execute node function
    try:
        output_data, output_ui, has_subgraph, has_pending = await get_output_data(
            prompt_id, unique_id, obj, input_data_all, ...
        )
        
        if has_pending:
            # Async operation in progress
            pending_async_nodes[unique_id] = output_data
            return (ExecutionResult.PENDING, None, None)
        
        if has_subgraph:
            # Node expanded into subgraph
            handle_subgraph_expansion(output_data, execution_list, caches)
            return (ExecutionResult.PENDING, None, None)
        
    except Exception as ex:
        return (ExecutionResult.FAILURE, error_details, ex)
    
    # 6. Cache and return result
    cache_entry = CacheEntry(ui=ui_outputs.get(unique_id), outputs=output_data)
    caches.outputs.set(unique_id, cache_entry)
    executed.add(unique_id)
    
    return (ExecutionResult.SUCCESS, None, None)
```

### Input Data Resolution

```python
def get_input_data(inputs, class_def, unique_id, execution_list=None, 
                   dynprompt=None, extra_data=None):
    """
    Resolve all inputs for a node.
    
    Returns: (input_data_all, missing_keys, v3_data)
    """
    input_data_all = {}
    missing_keys = {}
    
    for input_name, input_value in inputs.items():
        if is_link(input_value):
            # Connected to another node's output
            from_node_id, output_index = input_value
            
            if execution_list is None:
                missing_keys[input_name] = True
                continue
            
            # Get from cache
            cached = execution_list.get_cache(from_node_id, unique_id)
            if cached is None or cached.outputs is None:
                missing_keys[input_name] = True
                continue
            
            obj = cached.outputs[output_index]
            input_data_all[input_name] = obj
        else:
            # Constant value
            input_data_all[input_name] = [input_value]
    
    # Add hidden inputs (prompt, unique_id, etc.)
    if "hidden" in valid_inputs:
        for hidden_name, hidden_type in valid_inputs["hidden"].items():
            if hidden_type == "PROMPT":
                input_data_all[hidden_name] = [dynprompt.get_original_prompt()]
            elif hidden_type == "UNIQUE_ID":
                input_data_all[hidden_name] = [unique_id]
            # ... etc
    
    return input_data_all, missing_keys, v3_data
```

---

## Caching System

### Cache Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    CACHE HIERARCHY                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: NullCache                                              │
│  └── No caching (debugging)                                 │
│                                                              │
│  L2: ClassicCache (HierarchicalCache + CacheKeySetID)       │
│  └── Cache by node ID + class type                          │
│      └── Subcache by parent node                            │
│                                                              │
│  L3: LRUCache                                               │
│  └── Fixed-size cache with eviction                         │
│                                                              │
│  L4: RAMPressureCache                                       │
│  └── Dynamic size based on available RAM                    │
│      └── Evicts based on OOM score                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Cache Key Generation

```python
class CacheKeySetInputSignature(CacheKeySet):
    """Generate cache keys based on input signatures."""
    
    async def get_node_signature(self, dynprompt, node_id):
        """Create unique signature for node state."""
        signature = []
        
        # Get ordered list of ancestor nodes
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        
        # Current node signature
        signature.append(self.get_immediate_node_signature(
            dynprompt, node_id, order_mapping
        ))
        
        # Ancestor signatures
        for ancestor_id in ancestors:
            signature.append(self.get_immediate_node_signature(
                dynprompt, ancestor_id, order_mapping
            ))
        
        return to_hashable(signature)
    
    def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order):
        """Signature for a single node."""
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        
        signature = [
            class_type,
            await self.is_changed_cache.get(node_id),  # IS_CHANGED result
        ]
        
        # Include node ID for non-idempotent nodes
        if getattr(class_def, 'NOT_IDEMPOTENT', False):
            signature.append(node_id)
        
        # Hash inputs
        for key in sorted(node["inputs"].keys()):
            value = node["inputs"][key]
            if is_link(value):
                # Reference to ancestor position
                ancestor_id, socket = value
                ancestor_index = ancestor_order[ancestor_id]
                signature.append((key, ("ANCESTOR", ancestor_index, socket)))
            else:
                # Constant value
                signature.append((key, value))
        
        return signature
```

### RAMPressureCache

```python
class RAMPressureCache(LRUCache):
    """Cache that evicts entries when RAM is low."""
    
    def poll(self, ram_headroom):
        """Check RAM and evict if necessary."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb > ram_headroom:
            return  # Sufficient RAM
        
        gc.collect()
        if available_gb > ram_headroom:
            return  # GC freed enough
        
        # Calculate OOM score for each cache entry
        clean_list = []
        for key, (outputs, _) in self.cache.items():
            oom_score = self.calculate_oom_score(key, outputs)
            bisect.insort(clean_list, (oom_score, timestamp, key))
        
        # Evict highest scores until RAM is sufficient
        while available_gb < ram_headroom * HYSTERESIS and clean_list:
            _, _, key = clean_list.pop()
            del self.cache[key]
            gc.collect()
    
    def calculate_oom_score(self, key, outputs):
        """Higher score = more likely to be evicted."""
        # Age factor: older entries more likely to evict
        age_score = OLD_WORKFLOW_MULTIPLIER ** (
            self.generation - self.used_generation[key]
        )
        
        # Size factor: larger entries more likely to evict
        ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
        for output in outputs:
            if isinstance(output, torch.Tensor):
                ram_usage += output.numel() * output.element_size() * 0.5
        
        return age_score * ram_usage
```

---

## Async and Parallel Execution

### Async Node Functions

```python
async def get_output_data(prompt_id, unique_id, obj, input_data_all, ...):
    """Execute node function with async support."""
    
    # Check if function is async
    func = getattr(obj, obj.FUNCTION)
    
    if inspect.iscoroutinefunction(func):
        # Async execution
        async def async_wrapper(f, prompt_id, unique_id, index, args):
            with CurrentNodeContext(prompt_id, unique_id, index):
                return await f(**args)
        
        task = asyncio.create_task(async_wrapper(
            func, prompt_id, unique_id, index, inputs
        ))
        
        # Give it a chance to complete synchronously
        await asyncio.sleep(0)
        
        if task.done():
            return task.result()
        else:
            # Return task for later await
            return [task], {}, False, True
    else:
        # Synchronous execution
        with CurrentNodeContext(prompt_id, unique_id, index):
            result = func(**inputs)
        return [result], {}, False, False
```

### Batch Processing

```python
async def _async_map_node_over_list(prompt_id, unique_id, obj, 
                                     input_data_all, func_name, ...):
    """Execute node over list of inputs."""
    
    # Check if node wants lists directly
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)
    
    # Get max input length
    if input_data_all:
        max_len = max(len(x) for x in input_data_all.values())
    else:
        max_len = 0
    
    def slice_dict(d, i):
        """Get i-th element from each input list."""
        return {k: v[i if len(v) > i else -1] for k, v in d.items()}
    
    results = []
    
    if input_is_list:
        # Pass entire lists to node
        await process_inputs(input_data_all, 0, input_is_list=True)
    else:
        # Process each item sequentially
        for i in range(max_len):
            input_dict = slice_dict(input_data_all, i)
            await process_inputs(input_dict, i)
    
    return results
```

---

## Error Handling

### Exception Types

```python
class ExecutionResult(Enum):
    SUCCESS = 0   # Node executed successfully
    FAILURE = 1   # Node failed with error
    PENDING = 2   # Node waiting for dependencies/async

class DependencyCycleError(Exception):
    """Graph contains a cycle."""
    pass

class NodeNotFoundError(Exception):
    """Referenced node doesn't exist."""
    pass

class NodeInputError(Exception):
    """Invalid node input."""
    pass
```

### Error Propagation

```python
def handle_execution_error(self, prompt_id, prompt, current_outputs, 
                           executed, error, ex):
    """Send error information to client."""
    node_id = error["node_id"]
    class_type = prompt[node_id]["class_type"]
    
    if isinstance(ex, InterruptProcessingException):
        # User interrupted
        mes = {
            "prompt_id": prompt_id,
            "node_id": node_id,
            "node_type": class_type,
            "executed": list(executed),
        }
        self.add_message("execution_interrupted", mes, broadcast=True)
    else:
        # Actual error
        mes = {
            "prompt_id": prompt_id,
            "node_id": node_id,
            "node_type": class_type,
            "executed": list(executed),
            "exception_message": error["exception_message"],
            "exception_type": error["exception_type"],
            "traceback": error["traceback"],
            "current_inputs": error["current_inputs"],
        }
        self.add_message("execution_error", mes, broadcast=False)
```

---

## Z-Image-Turbo Adaptation

### Simplified Execution Engine

```python
class SimpleExecutionEngine:
    """Lightweight execution engine for Z-Image-Turbo."""
    
    def __init__(self, cache_enabled=True):
        self.cache = {} if cache_enabled else None
        self.results = {}
    
    def execute(self, workflow):
        """Execute workflow graph."""
        # 1. Topological sort
        execution_order = self.topological_sort(workflow)
        
        # 2. Execute nodes
        for node_id in execution_order:
            if self.is_cached(node_id):
                continue
            
            inputs = self.resolve_inputs(node_id, workflow)
            result = self.execute_node(node_id, inputs, workflow)
            self.store_result(node_id, result)
        
        return self.results
    
    def topological_sort(self, workflow):
        """Kahn's algorithm for topological sorting."""
        # Build adjacency list
        in_degree = {node_id: 0 for node_id in workflow}
        
        for node_id, node in workflow.items():
            for input_value in node.get("inputs", {}).values():
                if self.is_link(input_value):
                    from_node = input_value[0]
                    in_degree[node_id] += 1
        
        # Start with nodes having no dependencies
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Reduce in-degree of dependents
            for other_id, other_node in workflow.items():
                for input_value in other_node.get("inputs", {}).values():
                    if self.is_link(input_value) and input_value[0] == node_id:
                        in_degree[other_id] -= 1
                        if in_degree[other_id] == 0:
                            queue.append(other_id)
        
        return result
```

### Key Features to Implement

1. **Topological Sort**: Efficient dependency resolution
2. **Caching**: Avoid redundant computations
3. **Error Handling**: Clear error messages
4. **Progress Tracking**: Real-time updates
5. **Async Support**: Non-blocking operations

---

## Performance Considerations

| Aspect | Strategy | Impact |
|--------|----------|--------|
| Cache Hits | Input signature hashing | 10-100x speedup |
| Lazy Evaluation | Skip unused branches | Reduces unnecessary work |
| Batch Processing | Process multiple items | Better GPU utilization |
| Async I/O | Non-blocking file ops | Improved responsiveness |
| Memory Pressure | Dynamic cache eviction | Prevents OOM errors |

---

## Related Files

| File | Purpose |
|------|---------|
| `execution.py` | Main execution logic |
| `comfy_execution/graph.py` | Graph topology |
| `comfy_execution/caching.py` | Caching strategies |
| `comfy_execution/validation.py` | Input validation |
| `comfy_execution/progress.py` | Progress tracking |
