from elevenlabs import ElevenLabs

try:
    client = ElevenLabs(api_key="test")
    print(f"Has text_to_speech: {hasattr(client, 'text_to_speech')}")
    print(f"Has generate: {hasattr(client, 'generate')}")
except Exception as e:
    print(f"Error: {e}")
