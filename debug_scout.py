
import os
import time
from src.utils import download_audio
from src.scout import SignalScout

def debug():
    url = "https://www.youtube.com/watch?v=H14bBuluwB8" # Short video for test
    print(f"1. Downloading {url}...")
    t0 = time.time()
    path = download_audio(url, "assets/temp/debug_audio")
    print(f"   Downloaded in {time.time()-t0:.2f}s: {path}")
    
    print("2. Initializing Scout...")
    scout = SignalScout()
    
    print("3. Running Analyze...")
    t0 = time.time()
    hotspots = scout.analyze(path)
    print(f"   Analyzed in {time.time()-t0:.2f}s")
    print(f"   Hotspots: {hotspots}")

if __name__ == "__main__":
    debug()
