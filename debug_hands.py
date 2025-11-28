
"""
Debug script to test the full workflow from verified hotspots to final render.
"""
import os
import json
from src.utils import download_video_segment, get_video_info
from src.director import Director
from src.hands import Hands
from dotenv import load_dotenv

load_dotenv()

def debug():
    url = "https://www.youtube.com/watch?v=H14bBuluwB8"
    video_id = url.split("v=")[-1]
    
    print("=== SIMULATING STEP 2: Download Clips ===")
    # Simulate verified hotspots from Step 2
    hotspots = [
        {'start': 10.0, 'end': 15.0, 'score': 0.9, 'type': 'audio_peak'},
        {'start': 50.0, 'end': 55.0, 'score': 0.8, 'type': 'audio_peak'}
    ]
    
    # Download clips
    for i, spot in enumerate(hotspots):
        start = spot['start']
        end = spot['end']
        print(f"\nDownloading clip {i+1}: {start}s - {end}s")
        clip_filename = os.path.join("assets/temp", f"{video_id}_clip_{i}")
        try:
            clip_path = download_video_segment(url, start, end, clip_filename)
            spot['clip_path'] = clip_path
            print(f"  Downloaded: {clip_path}")
        except Exception as e:
            print(f"  Error: {e}")
            return
    
    print("\n=== SIMULATING STEP 3: Director ===")
    director = Director()
    info = get_video_info(url)
    print(f"Video Info: {info.get('title', 'Unknown')}")
    
    plan = director.create_edit_plan(hotspots, info)
    print(f"\nDirector Plan ({len(plan)} clips):")
    print(json.dumps(plan, indent=2))
    
    # Re-attach clip paths
    print("\n=== Mapping Clips to Plan ===")
    for clip in plan:
        c_start = clip.get('start')
        for spot in hotspots:
            h_start = spot.get('start')
            h_end = spot.get('end')
            if h_start <= c_start + 0.5 and h_end >= c_start - 0.5:
                clip['clip_path'] = spot.get('clip_path')
                clip['source_start'] = h_start
                print(f"  Clip {c_start}s matched to hotspot {h_start}s-{h_end}s")
                print(f"    Path: {clip['clip_path']}")
                break
    
    print("\n=== SIMULATING STEP 4: Hands ===")
    hands = Hands()
    
    # Hands needs a video_path, but in our Audio-First workflow, we don't have one
    # So let's pass a dummy or use the first clip
    video_path = hotspots[0].get('clip_path', '')
    
    print(f"Calling hands.execute with video_path={video_path}")
    try:
        output = hands.execute(video_path, plan, "debug_output.mp4")
        print(f"\n✓ Success! Output: {output}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
