#!/usr/bin/env python3
"""
Test Gemini API key and basic functionality
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is loaded"""
    api_key = os.getenv("VIDEO_API_KEY")
    if not api_key:
        print("❌ VIDEO_API_KEY not found in environment")
        return False
    print(f"✅ API key found (starts with: {api_key[:10]}...)")
    return True

def test_basic_generation():
    """Test basic text generation"""
    try:
        print("\n🔄 Testing basic text generation...")
        genai.configure(api_key=os.getenv("VIDEO_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        response = model.generate_content("Say 'hello' in one word")
        print(f"✅ Text generation works: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def test_video_upload():
    """Test video upload (without actual analysis)"""
    try:
        print("\n🔄 Testing video file upload...")
        
        # Check if there's a test video
        test_video = "temp/smart_cropped.mp4"
        if not os.path.exists(test_video):
            print(f"⚠️ Test video not found at {test_video}, skipping upload test")
            return None
        
        print(f"Uploading {test_video}...")
        video_file = genai.upload_file(test_video)
        print(f"✅ Video uploaded: {video_file.name}")
        
        # Wait for processing with timeout
        print("Waiting for Gemini to process video...")
        wait_time = 0
        max_wait = 30
        while video_file.state.name == "PROCESSING":
            if wait_time >= max_wait:
                print(f"⚠️ Processing timeout after {max_wait}s")
                return False
            time.sleep(2)
            wait_time += 2
            video_file = genai.get_file(video_file.name)
            print(f"  Status: {video_file.state.name} ({wait_time}s)")
        
        if video_file.state.name == "ACTIVE":
            print("✅ Video processed successfully")
            return True
        else:
            print(f"❌ Video processing failed with state: {video_file.state.name}")
            return False
            
    except Exception as e:
        print(f"❌ Video upload failed: {e}")
        return False

def main():
    print("=" * 60)
    print("GEMINI API TEST")
    print("=" * 60)
    
    if not test_api_key():
        return
    
    if not test_basic_generation():
        return
    
    result = test_video_upload()
    if result is None:
        print("\n⚠️ Video upload test skipped (no test file)")
    elif result:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Video upload/processing failed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
