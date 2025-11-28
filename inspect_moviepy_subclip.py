from moviepy import AudioFileClip
import inspect

print("Inspecting AudioFileClip methods...")
print([m for m in dir(AudioFileClip) if 'clip' in m.lower()])
