"""
Workflow adapter for importing ComfyUI workflows
Analyzes JSON and creates compatible nodes
"""

import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


class ComfyUIWorkflowAdapter:
    """Adapts ComfyUI workflows to Z-Image-Turbo format."""
    
    # Mapping ComfyUI nodes -> Our nodes
    NODE_MAPPING = {
        "CheckpointLoaderSimple": "LoadZImageTurbo",
        "CheckpointLoader": "LoadZImageTurbo",
        "UNETLoader": "LoadZImageTurbo",
        "CLIPTextEncode": None,  # Included in GenerateImage
        "CLIPSetLastLayer": None,  # Not needed
        "KSampler": "GenerateImage",
        "KSamplerAdvanced": "GenerateImage",
        "VAELoader": None,  # Included in GenerateImage
        "VAEDecode": None,  # Included in GenerateImage
        "VAEEncode": "VAEEncode",
        "SaveImage": "SaveImageTensor",
        "PreviewImage": "SaveImageTensor",
        "LoadImage": "LoadImage",
        "EmptyLatentImage": None,  # Included in GenerateImage
        "LatentUpscale": None,  # Not implemented
        "ImageScale": "ImageScale",
        "ImageUpscaleWithModel": None,  # Not implemented
    }
    
    def __init__(self):
        self.missing_nodes: Set[str] = set()
        self.mapped_nodes: Dict[str, str] = {}
    
    def analyze(self, workflow_path: str) -> Dict:
        """Analyze a ComfyUI workflow and report what needs to be adapted."""
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        analysis = {
            "total_nodes": len(workflow),
            "mapped_nodes": [],
            "missing_nodes": [],
            "already_compatible": [],
        }
        
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "Unknown")
            
            if class_type in self.NODE_MAPPING:
                mapped = self.NODE_MAPPING[class_type]
                if mapped:
                    analysis["mapped_nodes"].append({
                        "id": node_id,
                        "original": class_type,
                        "mapped_to": mapped
                    })
                else:
                    analysis["already_compatible"].append({
                        "id": node_id,
                        "class_type": class_type,
                        "note": "Functionality included in other nodes"
                    })
            else:
                analysis["missing_nodes"].append({
                    "id": node_id,
                    "class_type": class_type,
                    "inputs": list(node.get("inputs", {}).keys()),
                })
                self.missing_nodes.add(class_type)
        
        return analysis
    
    def generate_missing_node_template(self, class_type: str) -> str:
        """Generate a template for a missing node."""
        return f'''"""
Custom node adapted from ComfyUI: {class_type}
"""
from typing import Tuple, Any
from zimage.nodes import ZNode, ZType, register_node


@register_node("{class_type}", "custom")
class {class_type.replace(" ", "").replace("-", "_")}(ZNode):
    """
    Adapted from ComfyUI {class_type}
    TODO: Implement the actual functionality
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {{
            "required": {{
                # TODO: Define inputs based on ComfyUI implementation
            }}
        }}
    
    RETURN_TYPES = (ZType.ANY,)  # TODO: Adjust return type
    
    def execute(self, **kwargs) -> Tuple[Any]:
        """TODO: Implement node logic"""
        raise NotImplementedError("This node needs to be implemented")
'''
    
    def adapt_workflow(self, workflow_path: str, output_path: str = None) -> Dict:
        """
        Adapt a ComfyUI workflow to our format.
        Returns the adapted workflow and saves missing node templates.
        """
        with open(workflow_path, 'r') as f:
            comfy_workflow = json.load(f)
        
        adapted = {}
        node_id_mapping = {}
        
        # First pass: create node mappings
        new_id = 1
        for old_id, node in comfy_workflow.items():
            class_type = node.get("class_type", "")
            
            if class_type in self.NODE_MAPPING and self.NODE_MAPPING[class_type]:
                node_id_mapping[old_id] = str(new_id)
                new_id += 1
            elif class_type in self.NODE_MAPPING and self.NODE_MAPPING[class_type] is None:
                pass
            else:
                node_id_mapping[old_id] = str(new_id)
                new_id += 1
        
        # Second pass: rebuild workflow
        for old_id, node in comfy_workflow.items():
            if old_id not in node_id_mapping:
                continue
            
            new_id = node_id_mapping[old_id]
            class_type = node.get("class_type", "")
            
            if class_type in self.NODE_MAPPING:
                new_class = self.NODE_MAPPING[class_type]
                if new_class is None:
                    continue
            else:
                new_class = class_type
            
            # Remap input links
            new_inputs = {}
            for input_name, input_value in node.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    old_source = input_value[0]
                    if old_source in node_id_mapping:
                        new_inputs[input_name] = [node_id_mapping[old_source], input_value[1]]
                else:
                    new_inputs[input_name] = input_value
            
            adapted[new_id] = {
                "class_type": new_class,
                "inputs": new_inputs,
                "is_output": class_type in ["SaveImage", "PreviewImage"]
            }
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(adapted, f, indent=2)
        
        templates = {}
        for missing in self.missing_nodes:
            templates[missing] = self.generate_missing_node_template(missing)
        
        return {
            "adapted_workflow": adapted,
            "missing_node_templates": templates,
        }


def analyze_comfyui_workflow(workflow_path: str) -> None:
    """Quick analysis function."""
    adapter = ComfyUIWorkflowAdapter()
    analysis = adapter.analyze(workflow_path)
    
    print("=" * 60)
    print("COMFYUI WORKFLOW ANALYSIS")
    print("=" * 60)
    print(f"\nTotal nodes: {analysis['total_nodes']}")
    
    print(f"\n✅ Compatible ({len(analysis['already_compatible'])}):")
    for node in analysis['already_compatible']:
        print(f"  [{node['id']}] {node['class_type']}")
    
    print(f"\n🔄 Mappable ({len(analysis['mapped_nodes'])}):")
    for node in analysis['mapped_nodes']:
        print(f"  [{node['id']}] {node['original']} → {node['mapped_to']}")
    
    print(f"\n❌ Missing ({len(analysis['missing_nodes'])}):")
    for node in analysis['missing_nodes']:
        print(f"  [{node['id']}] {node['class_type']}")
    
    if analysis['missing_nodes']:
        print("\n⚠️  These nodes need implementation.")
    else:
        print("\n✨ All nodes compatible!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_comfyui_workflow(sys.argv[1])
    else:
        print("Usage: python workflow_adapter.py <workflow.json>")
