#!/usr/bin/env python3
"""
Z-Image-Turbo Web UI - Node-based backend with simple Gradio UI

This version uses a node-based execution engine inspired by ComfyUI,
while maintaining a simple Gradio interface for users.

For the legacy version, see app_legacy.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zimage.nodes.registry import NodeRegistry
from zimage.gradio_ui import create_app


def load_custom_nodes():
    """Load custom nodes from custom_nodes directory."""
    custom_nodes_dir = Path("custom_nodes")
    if custom_nodes_dir.exists():
        print(f"Loading custom nodes from {custom_nodes_dir}...")
        NodeRegistry.load_custom_nodes(str(custom_nodes_dir))
        print(f"Registered {len(NodeRegistry.list_nodes())} nodes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--auth", help="Username:Password for authentication")
    parser.add_argument("--ssl-cert", help="SSL certificate file")
    parser.add_argument("--ssl-key", help="SSL key file")
    parser.add_argument("--no-custom-nodes", action="store_true", help="Skip loading custom nodes")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("outputs/images", exist_ok=True)
    
    # Load custom nodes
    if not args.no_custom_nodes:
        load_custom_nodes()
    
    # Prepare auth
    auth = None
    if args.auth:
        parts = args.auth.split(":")
        if len(parts) == 2:
            auth = tuple(parts)
        else:
            print("Warning: Auth format should be username:password")
    
    # Prepare SSL
    ssl_verify = True
    if args.ssl_cert and args.ssl_key:
        ssl_verify = (args.ssl_cert, args.ssl_key)
    
    # Create and launch app
    print(f"\n{'='*50}")
    print(f"Z-Image-Turbo Web UI (Node-based)")
    print(f"{'='*50}")
    print(f"Loading interface...")
    
    app = create_app()
    
    print(f"Starting server on {args.host}:{args.port}")
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        ssl_verify=ssl_verify if isinstance(ssl_verify, bool) else False,
        ssl_certfile=args.ssl_cert if args.ssl_cert else None,
        ssl_keyfile=args.ssl_key if args.ssl_key else None,
        show_error=True,
    )


if __name__ == "__main__":
    main()
