"""
Hands: The "Execution" component.
Uses MoviePy to cut, crop, and render the final video.
"""

from moviepy import VideoFileClip, concatenate_videoclips
import os
from typing import List, Dict

from src.paths import OUTPUT_DIR, ASSETS_TEMP_DIR, ensure_runtime_dirs

class Hands:
    def __init__(self):
        pass

    def execute(self, video_path: str, edit_plan: List[Dict], output_filename: str = "final_edit.mp4") -> str:
        """
        Execute the edit plan.
        
        Args:
            video_path: Path to source video (can be audio-only if all clips have clip_path).
            edit_plan: List of clips to extract.
            output_filename: Name of output file.
            
        Returns:
            Path to the rendered video.
        """
        print(f"  - Hands executing with {len(edit_plan)} planned clips")
        ensure_runtime_dirs()
        
        try:
            subclips = []
            clips_to_close = []  # Track clips we open so we can close them later
            
            # Check if we need the original video at all
            need_original = any(not clip.get('clip_path') for clip in edit_plan)
            original_clip = None
            
            if need_original:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Original video not found: {video_path}")
                if video_path.endswith('.mp3'):
                    raise ValueError("Original video is audio-only. Cannot render video without clip_path in plan.")
                original_clip = VideoFileClip(video_path)
            
            for clip_info in edit_plan:
                start = clip_info.get('start')
                end = clip_info.get('end')
                clip_path = clip_info.get('clip_path')
                source_start = clip_info.get('source_start')
                
                if start is None or end is None:
                    continue
                if start >= end:
                    continue
                    
                print(f"    > Processing clip: {start:.2f}s - {end:.2f}s")
                
                if clip_path and os.path.exists(clip_path):
                    # Use pre-downloaded clip (Sniper mode)
                    print(f"      Using cached clip: {clip_path}")
                    
                    # Calculate local timestamps
                    if source_start is not None:
                        local_start = start - source_start
                        local_end = end - source_start
                    else:
                        # Fallback: use entire clip
                        local_start = 0
                        local_end = end - start
                        
                    local_start = max(0, local_start)
                    
                    current_clip = VideoFileClip(clip_path)
                    clips_to_close.append(current_clip)
                    
                    # Clamp end time to clip duration
                    if local_end > current_clip.duration:
                        local_end = current_clip.duration
                    
                    # Skip if invalid range
                    if local_start >= local_end or local_start >= current_clip.duration:
                        print(f"      ! Invalid local range ({local_start:.2f}s - {local_end:.2f}s) for clip duration {current_clip.duration:.2f}s. Skipping.")
                        continue
                        
                    print(f"      Cutting local: {local_start:.2f}s - {local_end:.2f}s")
                    subclip = current_clip.subclipped(local_start, local_end)
                    subclips.append(subclip)
                    
                else:
                    # Use original video
                    if original_clip is None:
                        print("      ! No clip_path and no original video available. Skipping.")
                        continue
                        
                    print(f"      Using main video")
                    # Clamp to duration
                    if end > original_clip.duration:
                        end = original_clip.duration
                    subclip = original_clip.subclipped(start, end)
                    subclips.append(subclip)
            
            if not subclips:
                print("  ! No valid clips to render.")
                return "Error: No valid clips to render. Check that hotspot times are within video duration."
                
            print(f"  - Concatenating {len(subclips)} clips...")
            final_clip = concatenate_videoclips(subclips)
            
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            print(f"  - Rendering to {output_path}...")
            final_clip.write_videofile(
                output_path, 
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(ASSETS_TEMP_DIR, 'temp-audio.m4a'),
                remove_temp=True,
                logger=None
            )
            
            # Cleanup
            final_clip.close()
            if original_clip:
                original_clip.close()
            for clip in clips_to_close:
                clip.close()
            
            return output_path
            
        except Exception as e:
            print(f"  ! Hands failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Video rendering failed. {str(e)}"

if __name__ == "__main__":
    h = Hands()
    print("Hands initialized.")
