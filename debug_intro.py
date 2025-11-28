
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.getcwd())

from src.showrunner import Showrunner

def test_intro_generation():
    print("🚀 Starting Intro Generation Debug")
    print("="*50)
    
    # Check API Key
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("❌ NEBIUS_API_KEY is missing in environment!")
        return
    print(f"✅ NEBIUS_API_KEY found (starts with: {api_key[:5]}...)")
    
    runner = Showrunner()
    
    # Test Parameters
    title = "DEBUG TEST"
    mood = "hype"
    
    print(f"\n📸 Attempting to generate image for: '{title}' ({mood})")
    
    # Potential model names to try
    models_to_try = [
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-dev",
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "stability-ai/sdxl" 
    ]
    
    image_path = None
    
    for model_name in models_to_try:
        print(f"\n🔄 Trying model: {model_name}")
        try:
            # Temporarily patch the runner's generate method to use this model
            # Or just copy the request logic here for testing
            import requests
            
            prompt = "High-energy social media intro card, vertical 9:16, large bold typography reading 'DEBUG TEST', vibrant neon colors"
            
            response = requests.post(
                "https://api.studio.nebius.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "width": 1080,
                    "height": 1920,
                    "num_inference_steps": 4
                }
            )
            
            if response.status_code == 200:
                print(f"✅ SUCCESS with model: {model_name}")
                result = response.json()
                image_url = result['data'][0]['url']
                
                img_response = requests.get(image_url)
                output_path = "temp/intro_image.jpg"
                os.makedirs("temp", exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                
                image_path = output_path
                break
            else:
                print(f"❌ Failed with {model_name}: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception with {model_name}: {e}")

    if image_path:
        print(f"\n✅ Image generated successfully at: {image_path}")
        # Check file size
        size = os.path.getsize(image_path)
        print(f"   File size: {size} bytes")
    else:
        print("\n❌ All models failed")
        return

    # 2. Create Clip
    print("\n🎬 Attempting to create video clip...")
    try:
        from moviepy.editor import ImageClip
        
        if os.path.exists(image_path):
            clip = ImageClip(image_path, duration=3.0)
            print(f"✅ Clip created successfully")
            print(f"   Duration: {clip.duration}")
            print(f"   Size: {clip.size}")
        else:
            print("❌ Image file not found for clip creation")
            
    except Exception as e:
        print(f"❌ Exception during clip creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intro_generation()
