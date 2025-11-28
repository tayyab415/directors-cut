
import os
from src.director import Director
from dotenv import load_dotenv

load_dotenv()

def debug():
    print("1. Initializing Director...")
    director = Director()
    
    # Mock data
    hotspots = [
        {'start': 10.0, 'end': 15.0, 'score': 0.9, 'type': 'audio_peak', 'verification': {'score': 8, 'reason': 'Good visual', 'type': 'talking_head'}},
        {'start': 50.0, 'end': 55.0, 'score': 0.8, 'type': 'audio_peak', 'verification': {'score': 7, 'reason': 'Interesting demo', 'type': 'demo'}}
    ]
    
    video_info = {
        'title': 'Test Video',
        'duration': 100,
        'uploader': 'TestUser'
    }
    
    print("2. Running create_edit_plan...")
    try:
        plan = director.create_edit_plan(hotspots, video_info)
        print(f"3. Result: {plan}")
    except Exception as e:
        print(f"3. Error: {e}")

if __name__ == "__main__":
    debug()
