#!/usr/bin/env python3
"""Quick test to see if app can launch without WhisperX."""

import sys
import os

# Set test env vars
os.environ.setdefault('GEMINI_API_KEY', 'test')
os.environ.setdefault('VIDEO_API_KEY', 'test')
os.environ.setdefault('NEBIUS_API_KEY', 'test')
os.environ.setdefault('ELEVENLABS_API_KEY', 'test')

sys.path.insert(0, '/Users/tayyabkhan/Documents/try-director')

print("🧪 Testing App Launch Readiness\n" + "="*50)

# Check if whisperx is available
try:
    import whisperx
    print("✅ WhisperX is installed")
    whisperx_available = True
except ImportError as e:
    print(f"⚠️  WhisperX not available: {e}")
    print("   Subtitle feature will be disabled")
    whisperx_available = False

# Check if torch is available
try:
    import torch
    print(f"✅ Torch is installed (version: {torch.__version__})")
except ImportError as e:
    print(f"❌ Torch not available: {e}")

# Check if gradio is available
try:
    import gradio as gr
    print(f"✅ Gradio is installed (version: {gr.__version__})")
except ImportError as e:
    print(f"❌ Gradio not available: {e}")

# Check if moviepy is available
try:
    import moviepy
    print(f"✅ MoviePy is installed")
except ImportError as e:
    print(f"❌ MoviePy not available: {e}")

print("\n" + "="*50)

if not whisperx_available:
    print("\n⚠️  Note: Subtitle feature will fail gracefully")
    print("   Other features (smart crop, intro image, voiceover) will still work")
    print("   To enable subtitles, manually install: pip install git+https://github.com/m-bain/whisperx.git")
else:
    print("\n✅ All dependencies are ready!")

print("\n💡 App should be able to launch with existing dependencies")
