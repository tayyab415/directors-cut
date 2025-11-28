#!/usr/bin/env python3
"""
Test script for FLUX image generation via Nebius API.
This script isolates the image generation logic to troubleshoot issues.
"""

import os
import requests
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

if not NEBIUS_API_KEY:
    print("❌ ERROR: NEBIUS_API_KEY not found in environment")
    exit(1)

print(f"✅ Found NEBIUS_API_KEY: {NEBIUS_API_KEY[:10]}...")

# Test parameters
test_prompt = "High-energy social media intro card, vertical 9:16, large bold typography reading 'TEST TITLE', vibrant neon gradients, modern streetwear aesthetic, 4k resolution"
test_title = "TEST TITLE"
test_mood = "hype"

print(f"\n📝 Test Prompt: {test_prompt[:100]}...")
print(f"🎨 Mood: {test_mood}")
print(f"📐 Dimensions: 1080x1920")

# Test 1: Request with response_format="b64_json"
print("\n" + "="*60)
print("TEST 1: Requesting base64 format explicitly")
print("="*60)

try:
    response = requests.post(
        "https://api.studio.nebius.ai/v1/images/generations",
        headers={
            "Authorization": f"Bearer {NEBIUS_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "black-forest-labs/flux-schnell",
            "prompt": test_prompt,
            "width": 1080,
            "height": 1920,
            "num_inference_steps": 4,
            "response_format": "b64_json"
        },
        timeout=60
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code != 200:
        print(f"\n❌ API Error: {response.status_code}")
        print(f"Response Text: {response.text}")
        exit(1)

    result = response.json()
    print(f"\n✅ API Response Structure:")
    print(f"  Top-level keys: {list(result.keys())}")
    
    if 'data' in result:
        data = result['data']
        print(f"  Data array length: {len(data)}")
        
        if data:
            image_entry = data[0]
            print(f"  Image entry keys: {list(image_entry.keys())}")
            
            # Check for b64_json
            if 'b64_json' in image_entry:
                print(f"\n✅ Found b64_json in response!")
                b64_data = image_entry['b64_json']
                print(f"  Base64 data length: {len(b64_data)} characters")
                
                try:
                    # Decode base64
                    image_bytes = base64.b64decode(b64_data)
                    print(f"  Decoded image size: {len(image_bytes)} bytes")
                    
                    # Check magic bytes
                    if image_bytes.startswith(b'\xff\xd8\xff'):
                        print(f"  ✅ Valid JPEG (magic bytes: FF D8 FF)")
                    elif image_bytes.startswith(b'\x89PNG'):
                        print(f"  ✅ Valid PNG (magic bytes: 89 50 4E 47)")
                    elif image_bytes.startswith(b'RIFF'):
                        print(f"  ✅ Valid WebP (magic bytes: RIFF)")
                    else:
                        print(f"  ⚠️ Unknown format (first 10 bytes: {image_bytes[:10].hex()})")
                    
                    # Save test image
                    output_path = "temp/test_flux_image.jpg"
                    os.makedirs("temp", exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    
                    print(f"\n✅ Image saved to: {output_path}")
                    print(f"   File size: {os.path.getsize(output_path)} bytes")
                    
                except Exception as decode_error:
                    print(f"\n❌ Failed to decode base64: {decode_error}")
                    import traceback
                    traceback.print_exc()
            
            # Check for URL
            if 'url' in image_entry and image_entry.get('url') is not None:
                print(f"\n⚠️ Found URL in response (but we requested b64_json)")
                print(f"  URL: {image_entry['url']}")
                
                # Try downloading it anyway
                print(f"\n  Attempting to download from URL...")
                img_response = requests.get(image_entry['url'], timeout=30)
            elif 'url' in image_entry:
                print(f"\n⚠️ URL field exists but is None (expected when requesting b64_json)")
                print(f"  Download status: {img_response.status_code}")
                print(f"  Content-Type: {img_response.headers.get('Content-Type', 'N/A')}")
                print(f"  Content length: {len(img_response.content)} bytes")
                
                if img_response.status_code == 200:
                    content = img_response.content
                    # Check magic bytes
                    if content.startswith(b'\xff\xd8\xff'):
                        print(f"  ✅ Valid JPEG downloaded")
                    elif content.startswith(b'<?xml') or content.startswith(b'<Error'):
                        print(f"  ❌ XML Error received instead of image!")
                        print(f"  Error content: {content[:500].decode('utf-8', errors='ignore')}")
                    else:
                        print(f"  ⚠️ Unknown format (first 20 bytes: {content[:20].hex()})")
            
            # Check for other fields
            other_keys = [k for k in image_entry.keys() if k not in ['b64_json', 'url']]
            if other_keys:
                print(f"\n  Other keys in response: {other_keys}")
                for key in other_keys:
                    value = image_entry[key]
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
        else:
            print(f"\n❌ Data array is empty!")
            print(f"  Full response: {json.dumps(result, indent=2)}")
    else:
        print(f"\n❌ No 'data' key in response!")
        print(f"  Full response: {json.dumps(result, indent=2)}")

except Exception as e:
    print(f"\n❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Request without response_format (see what we get by default)
print("\n" + "="*60)
print("TEST 2: Requesting without response_format (default)")
print("="*60)

try:
    response2 = requests.post(
        "https://api.studio.nebius.ai/v1/images/generations",
        headers={
            "Authorization": f"Bearer {NEBIUS_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "black-forest-labs/flux-schnell",
            "prompt": test_prompt,
            "width": 1080,
            "height": 1920,
            "num_inference_steps": 4
            # No response_format parameter
        },
        timeout=60
    )

    print(f"Status Code: {response2.status_code}")
    
    if response2.status_code == 200:
        result2 = response2.json()
        if 'data' in result2 and result2['data']:
            image_entry2 = result2['data'][0]
            print(f"  Default response keys: {list(image_entry2.keys())}")
            
            if 'b64_json' in image_entry2:
                print(f"  ✅ Default includes b64_json")
            elif 'url' in image_entry2:
                print(f"  ⚠️ Default returns URL instead of b64_json")
            else:
                print(f"  ❌ Unknown default format")
    else:
        print(f"  ❌ Error: {response2.status_code}")

except Exception as e:
    print(f"  ⚠️ Test 2 failed: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

