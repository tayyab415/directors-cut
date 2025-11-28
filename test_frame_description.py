#!/usr/bin/env python3
"""
Ask Qwen VL to describe the subject position in the frame
"""
import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

def describe_frame(frame_path):
    """Ask Qwen VL to describe where the subject is"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {frame_path}")
    print('='*60)
    
    with open(frame_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # First, ask it to describe the position
    response = requests.post(
        "https://api.studio.nebius.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NEBIUS_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Describe this image briefly:
1. What is the main subject? (person, object, text, etc.)
2. Where is the main subject positioned horizontally? (far left, left, center, right, far right)
3. Is the subject centered or off to one side?

Be specific about the horizontal position."""
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        description = result['choices'][0]['message']['content']
        print(f"\n📝 Qwen VL Description:\n{description}\n")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    # Test multiple frames
    frames = [
        "temp/frames/frame_0.00.jpg",
        "temp/frames/frame_5.20.jpg",
        "temp/frames/frame_12.70.jpg"
    ]
    
    for frame in frames:
        if os.path.exists(frame):
            describe_frame(frame)
