"""
SignalScout: The "Scout" component of the Director's Cut architecture.
Responsible for analyzing raw video to find "Hotspots" using signal processing (audio/visual).
"""

import librosa
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from typing import List, Dict, Any
import os

class SignalScout:
    def __init__(self):
        pass

    def analyze(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a video file to find interesting segments (hotspots).
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            List of hotspots: [{start, end, score, type}]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Scouting video: {video_path}")
        
        # 1. Audio Analysis (Loudness/Energy)
        audio_hotspots = self._analyze_audio(video_path)
        
        # 2. Visual Analysis (Scene Changes)
        # SKIP for speed if input is audio-only or if we want fast scouting
        # scene_hotspots = self._analyze_scenes(video_path)
        
        return audio_hotspots

    def detect_bad_audio(self, video_path: str, silence_thresh: float = 0.01) -> List[Dict[str, Any]]:
        """
        Detect 'bad' audio segments to exclude (silence, noise).
        """
        print("  - Detecting bad audio segments...")
        try:
            from moviepy import AudioFileClip
            
            clip = AudioFileClip(video_path)
            duration = clip.duration
            chunk_size = 1.0 # 1 second precision for bad segments
            
            bad_segments = []
            current_bad_start = None
            
            # Iterate
            for t in np.arange(0, duration, chunk_size):
                end = min(t + chunk_size, duration)
                if end - t < 0.1: continue
                
                chunk = clip.subclipped(t, end).to_soundarray()
                rms = np.sqrt(np.mean(chunk**2))
                
                is_bad = rms < silence_thresh
                
                if is_bad:
                    if current_bad_start is None:
                        current_bad_start = t
                else:
                    if current_bad_start is not None:
                        # End of bad segment
                        bad_segments.append({
                            'start': float(current_bad_start),
                            'end': float(t),
                            'type': 'silence'
                        })
                        current_bad_start = None
            
            # Check if ended in bad segment
            if current_bad_start is not None:
                bad_segments.append({
                    'start': float(current_bad_start),
                    'end': float(duration),
                    'type': 'silence'
                })
                
            clip.close()
            print(f"    Found {len(bad_segments)} bad audio segments (silence).")
            return bad_segments
            
        except Exception as e:
            print(f"  ! Bad audio detection failed: {e}")
            return []

    def _analyze_audio(self, video_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Find loud moments in the audio using MoviePy (lazy loading).
        Returns top_n hotspots (default 10 for better coverage of long videos).
        """
        print("  - Analyzing audio (MoviePy)...")
        try:
            from moviepy import AudioFileClip
            
            # MoviePy reads via ffmpeg pipe, so it doesn't load the whole file
            clip = AudioFileClip(video_path)
            duration = clip.duration
            chunk_size = 5.0 # seconds
            
            segments = []
            print(f"    Audio duration: {duration:.1f}s ({duration/60:.1f} min)")
            
            # Iterate
            for t in np.arange(0, duration, chunk_size):
                end = min(t + chunk_size, duration)
                if end - t < 1.0: continue
                
                # Read chunk (returns numpy array)
                chunk = clip.subclipped(t, end).to_soundarray()
                
                # Calculate RMS
                # chunk is (N, 2) for stereo
                rms = np.sqrt(np.mean(chunk**2))
                
                segments.append({
                    'start': float(t),
                    'end': float(end),
                    'score': float(rms),
                    'type': 'audio_peak'
                })
                
                if int(t) % 60 == 0:
                    print(f"    Scanned {t:.0f}s...")
            
            clip.close()
            
            # Sort by energy
            segments.sort(key=lambda x: x['score'], reverse=True)
            
            # Normalize scores
            if segments:
                max_score = segments[0]['score']
                if max_score > 0:
                    for s in segments:
                        s['score'] = s['score'] / max_score
            
            # Return top N, but adapt to video length
            # For very long videos (>30 min), return more candidates
            adaptive_top_n = top_n
            if duration > 1800:  # 30+ minutes
                adaptive_top_n = min(15, top_n * 2)
            
            print(f"    Returning top {adaptive_top_n} audio hotspots")
            return segments[:adaptive_top_n]
            
        except Exception as e:
            print(f"  ! Audio analysis failed: {e}")
            return []

    def _analyze_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scene changes."""
        print("  - Analyzing scenes...")
        try:
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            
            scenes = []
            for scene in scene_list:
                start, end = scene
                scenes.append({
                    'start': start.get_seconds(),
                    'end': end.get_seconds(),
                    'score': 0.5, # Default score for a scene change
                    'type': 'scene_change'
                })
            return scenes
            
        except Exception as e:
            print(f"  ! Scene analysis failed: {e}")
            return []

if __name__ == "__main__":
    # Test stub
    scout = SignalScout()
    print("SignalScout initialized.")
