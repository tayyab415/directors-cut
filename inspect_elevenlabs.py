import elevenlabs
from elevenlabs import ElevenLabs
import inspect

print(f"ElevenLabs version: {elevenlabs.__version__ if hasattr(elevenlabs, '__version__') else 'unknown'}")
print("Inspecting ElevenLabs class...")
print(dir(ElevenLabs))

try:
    client = ElevenLabs(api_key="test")
    print("Client attributes:")
    print(dir(client))
except Exception as e:
    print(f"Error creating client: {e}")
