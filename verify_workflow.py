"""
Verification script for Director's Cut.
Tests the FULL workflow: Scout -> Verifier -> Director -> Hands.
"""

import os
import json
import time
from src.server import scan_video, analyze_hotspot, render_edit
from src.utils import ensure_directories

def test_full_workflow():
    print("=== Testing Full Director's Cut Workflow ===")
    ensure_directories()
    
    # 1. Scout
    test_url = "https://www.youtube.com/watch?v=H14bBuluwB8" # Short timer video
    print(f"\n[1/4] Scouting video: {test_url}...")
    try:
        hotspots_str = scan_video(test_url)
        print(f"Hotspots found: {hotspots_str}")
        hotspots = eval(hotspots_str) # Safe for this test script
    except Exception as e:
        print(f"❌ Scout failed: {e}")
        return

    if not hotspots:
        print("⚠️ No hotspots found. Cannot proceed.")
        return

    # 2. Verifier (Test first hotspot)
    print(f"\n[2/4] Verifying first hotspot...")
    first_hotspot = hotspots[0]
    timestamp = first_hotspot['start']
    try:
        # Note: This requires OPENROUTER_API_KEY
        verification = analyze_hotspot(test_url, timestamp)
        print(f"Verification result: {verification}")
    except Exception as e:
        print(f"❌ Verifier failed (check API key?): {e}")

    # 3. Director & Hands (Render Edit)
    print(f"\n[3/4] & [4/4] Director & Hands (Rendering)...")
    try:
        # Note: This requires DEEPSEEK_API_KEY
        # We pass the hotspots string as expected by the tool
        result = render_edit(test_url, json.dumps(hotspots))
        print(f"Render result: {result}")
    except Exception as e:
        print(f"❌ Director/Hands failed (check API key?): {e}")

if __name__ == "__main__":
    test_full_workflow()
