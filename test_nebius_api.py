#!/usr/bin/env python3
"""
Test script to verify Nebius Qwen VL API endpoint and response format
"""
import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

def test_nebius_api():
    """Test Nebius API with a real test image"""
    
    # Use the generated test image
    test_image_path = "/Users/tayyabkhan/.gemini/antigravity/brain/697af135-43c5-4f0d-a61d-479cfb1f89fb/test_crop_image_1763694307509.png"
    
    print("Testing Nebius Qwen VL API endpoint...")
    print(f"API Key present: {'Yes' if NEBIUS_API_KEY else 'No'}")
    print(f"Test image: {test_image_path}")
    
    if not NEBIUS_API_KEY:
        print("❌ NEBIUS_API_KEY not found in .env")
        return False
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found at {test_image_path}")
        return False
    
    # Read and encode the image
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
        test_image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
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
                                    "url": f"data:image/png;base64,{test_image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Where is the main subject in this image horizontally?
                                
Return a number between 0.0 and 1.0:
- 0.0 = far left
- 0.5 = center  
- 1.0 = far right

Return ONLY the number."""
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=30
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ API call successful!")
            print(f"Full response: {result}")
            
            # Try to extract the message
            try:
                message = result['choices'][0]['message']['content']
                print(f"\nExtracted message: {message}")
                return True
            except KeyError as e:
                print(f"\n⚠️ Response structure different than expected: {e}")
                print("This may need code adjustment")
                return False
        else:
            print(f"\n❌ API call failed")
            print(f"Response body: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nebius_api()
    print(f"\nTest {'PASSED ✅' if success else 'FAILED ❌'}")
