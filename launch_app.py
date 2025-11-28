#!/usr/bin/env python3
"""
Helper script to launch app.py with better error handling and port management.
"""

import os
import sys
import subprocess
import socket

def check_port(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

def kill_process_on_port(port):
    """Kill the process using the specified port."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"Killing process {pid} on port {port}...")
                subprocess.run(['kill', pid])
            return True
        return False
    except Exception as e:
        print(f"Could not kill process on port {port}: {e}")
        return False

def main():
    # Get port from environment or use default
    port = int(os.getenv('GRADIO_SERVER_PORT', 7860))
    
    print(f"Checking port {port}...")
    
    if not check_port(port):
        print(f"⚠️  Port {port} is already in use!")
        response = input(f"Kill process on port {port}? (y/n): ").strip().lower()
        if response == 'y':
            if kill_process_on_port(port):
                print(f"✅ Port {port} is now available")
            else:
                print(f"❌ Could not free port {port}")
                print(f"💡 Try using a different port: GRADIO_SERVER_PORT=7861 python app.py")
                sys.exit(1)
        else:
            print(f"💡 Using a different port: {port + 1}")
            port = port + 1
            os.environ['GRADIO_SERVER_PORT'] = str(port)
    
    print(f"🚀 Launching app on port {port}...")
    print(f"📡 MCP endpoint will be at: http://localhost:{port}/gradio_api/mcp/sse")
    print(f"🌐 Web UI will be at: http://localhost:{port}")
    print()
    
    # Import and launch
    import app
    
    # Override the port in the launch call
    import gradio as gr
    
    # Get the app instance
    app_instance = app.app
    
    # Launch with the specified port
    app_instance.launch(
        server_name="0.0.0.0",
        server_port=port,
        mcp_server=True,
        share=False
    )

if __name__ == "__main__":
    main()





