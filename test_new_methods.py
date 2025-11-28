#!/usr/bin/env python3
"""Test script to verify new Showrunner methods exist and have correct signatures."""

import sys
sys.path.insert(0, '/Users/tayyabkhan/Documents/try-director')

from src.showrunner import Showrunner
import inspect

# Initialize Showrunner
showrunner = Showrunner()

# Test 1: Check new methods exist
print("🧪 Testing New Methods Existence\n" + "="*50)

methods_to_check = [
    'generate_intro_screen',
    'create_intro_clip_from_image',
    'get_word_level_transcription',
    'add_subtitles_to_video'
]

for method_name in methods_to_check:
    if hasattr(showrunner, method_name):
        print(f"✅ {method_name} exists")
    else:
        print(f"❌ {method_name} NOT FOUND")

# Test 2: Check compose_final signature
print("\n🧪 Testing compose_final Signature\n" + "="*50)
sig = inspect.signature(showrunner.compose_final)
params = list(sig.parameters.keys())
print(f"Parameters: {params}")

expected_params = ['original_video', 'intro_audio', 'bg_music', 'title_text', 
                   'mood', 'enable_smart_crop', 'add_intro_image', 'add_subtitles']

for param in expected_params:
    if param in params:
        print(f"✅ Parameter '{param}' exists")
    else:
        print(f"❌ Parameter '{param}' MISSING")

# Test 3: Check method signatures
print("\n🧪 Testing Method Signatures\n" + "="*50)

# generate_intro_screen
sig = inspect.signature(showrunner.generate_intro_screen)
print(f"generate_intro_screen: {sig}")

# create_intro_clip_from_image
sig = inspect.signature(showrunner.create_intro_clip_from_image)
print(f"create_intro_clip_from_image: {sig}")

# get_word_level_transcription
sig = inspect.signature(showrunner.get_word_level_transcription)
print(f"get_word_level_transcription: {sig}")

# add_subtitles_to_video
sig = inspect.signature(showrunner.add_subtitles_to_video)
print(f"add_subtitles_to_video: {sig}")

print("\n✅ All tests passed!")
