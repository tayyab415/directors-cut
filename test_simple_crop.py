#!/usr/bin/env python3
"""
Minimal test - just crop a video without any AI analysis
"""
import cv2
import os
from moviepy import VideoFileClip

def simple_9_16_crop(input_video, output_path="temp/simple_crop.mp4"):
    """
    Dead simple crop to 9:16 centered, no AI
    """
    print(f"Loading video: {input_video}")
    video = VideoFileClip(input_video)
    
    # Get dimensions
    width, height = video.size
    print(f"Original: {width}x{height}")
    
    # Calculate 9:16 crop (centered)
    target_aspect = 9 / 16
    current_aspect = width / height
    
    if current_aspect > target_aspect:
        # Too wide, crop sides
        new_width = int(height * target_aspect)
        x_center = width // 2
        x1 = x_center - new_width // 2
        x2 = x1 + new_width
        cropped = video.cropped(x1=x1, x2=x2)
    else:
        # Too tall, crop top/bottom
        new_height = int(width / target_aspect)
        y_center = height // 2
        y1 = y_center - new_height // 2
        y2 = y1 + new_height
        cropped = video.cropped(y1=y1, y2=y2)
    
    print(f"Cropped: {cropped.size[0]}x{cropped.size[1]}")
    print(f"Saving to: {output_path}")
    
    os.makedirs("temp", exist_ok=True)
    cropped.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        write_logfile=False
    )
    
    video.close()
    cropped.close()
    
    print(f"✅ Done! Cropped video saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test with existing video
    test_video = "temp/smart_cropped.mp4"
    if os.path.exists(test_video):
        print(f"Testing with: {test_video}")
        simple_9_16_crop(test_video)
    else:
        print(f"❌ Test video not found: {test_video}")
        print("Please provide a video file path as argument")
