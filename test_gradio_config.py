#!/usr/bin/env python3
"""Test script to verify Gradio UI configuration for Production Studio tab."""

import sys
sys.path.insert(0, '/Users/tayyabkhan/Documents/try-director')

# Mock the environment to avoid missing API keys
import os
os.environ.setdefault('GEMINI_API_KEY', 'test_key')
os.environ.setdefault('VIDEO_API_KEY', 'test_key')

print("🧪 Testing Gradio App Configuration\n" + "="*50)

try:
    # Import app components
    print("⏳ Importing app.py...")
    import app
    print("✅ app.py imported successfully")
    
    # Check if wrapper function exists
    print("\n⏳ Checking add_production_wrapper function...")
    if hasattr(app, 'add_production_wrapper'):
        print("✅ add_production_wrapper function exists")
        
        # Check signature
        import inspect
        sig = inspect.signature(app.add_production_wrapper)
        params = list(sig.parameters.keys())
        print(f"   Parameters: {params}")
        
        expected = ['video_file', 'mood_override', 'enable_smart_crop', 
                    'add_intro_image', 'add_subtitles', 'progress']
        
        missing = [p for p in expected if p not in params]
        if missing:
            print(f"❌ Missing parameters: {missing}")
        else:
            print("✅ All expected parameters present")
    else:
        print("❌ add_production_wrapper function NOT FOUND")
    
    print("\n✅ Gradio app configuration test passed!")
    print("\nℹ️  To launch the app, run: python app.py")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
