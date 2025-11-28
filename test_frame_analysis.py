#!/usr/bin/env python3
"""
Test Qwen VL on actual extracted frame to verify position detection
"""
import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

def test_frame(frame_path):
    """Test Qwen VL analysis on a specific frame"""
    
    print(f"Testing frame: {frame_path}")
    
    if not os.path.exists(frame_path):
        print(f"❌ Frame not found: {frame_path}")
        return
    
    # Read and encode the image
    with open(frame_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    try:
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
                                "text": """You are analyzing a video frame to detect where the MAIN SUBJECT is positioned horizontally.

TASK: Identify the center point of the main subject (person, face, or focal object).

INSTRUCTIONS:
1. Locate the main subject (prioritize: person > face > text > main object)
2. Find the CENTER POINT of that subject's bounding box
3. Measure where that center point is along the horizontal axis
4. Imagine the frame divided into 10 equal vertical strips (0%, 10%, 20%... 100%)
5. Return the position as a decimal between 0.0 and 1.0

POSITION SCALE:
0.0 = Far left edge (subject center at left 0%)
0.1 = Left edge (subject center at 10% from left)
0.2 = Left quarter (subject center at 20% from left)
0.3 = Left-center area (subject center at 30% from left)
0.4 = Slightly left of center (subject center at 40% from left)
0.5 = Perfect center (subject center at 50%)
0.6 = Slightly right of center (subject center at 60% from left)
0.7 = Right-center area (subject center at 70% from left)
0.8 = Right quarter (subject center at 80% from left)
0.9 = Right edge (subject center at 90% from left)
1.0 = Far right edge (subject center at right edge 100%)

EXAMPLES:
- Person's face/body mostly on far left: 0.15
- Person slightly left of center: 0.35
- Person perfectly centered: 0.5
- Person slightly right of center: 0.65
- Person's face/body mostly on far right: 0.85

IMPORTANT: 
- Be precise! Don't default to 0.5 unless truly centered
- Use the full range 0.0 to 1.0
- Measure the CENTER of the subject, not its edges

Return ONLY the decimal number (e.g., 0.35), nothing else."""
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=30
        )
        
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            position = result['choices'][0]['message']['content'].strip()
            print(f"📝 Qwen VL detected position: {position}")
            print(f"\nInterpretation:")
            pos_val = float(position)
            if pos_val < 0.3:
                print(f"  → Subject is on the LEFT side of frame")
            elif pos_val < 0.45:
                print(f"  → Subject is slightly LEFT of center")
            elif pos_val < 0.55:
                print(f"  → Subject is CENTERED")
            elif pos_val < 0.7:
                print(f"  → Subject is slightly RIGHT of center")
            else:
                print(f"  → Subject is on the RIGHT side of frame")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the most recent frame
    frame = "temp/frames/frame_12.70.jpg"
    test_frame(frame)
    
    print("\n" + "="*60)
    print("Now testing with a different frame...")
    print("="*60 + "\n")
    
    # Test another frame
    frame2 = "temp/frames/frame_0.00.jpg"
    test_frame(frame2)
