from moviepy import VideoFileClip
import inspect

print("Inspecting VideoFileClip methods...")
print([m for m in dir(VideoFileClip) if 'audio' in m.lower()])
print([m for m in dir(VideoFileClip) if 'resize' in m.lower()])
