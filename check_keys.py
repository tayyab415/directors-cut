"""
Check API Keys Validity
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def check_openrouter():
    print("\n--- Checking OpenRouter ---")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("❌ OPENROUTER_API_KEY not found in env.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )
    try:
        # Minimal request
        client.models.list()
        print("✅ OpenRouter Key is VALID (Models list fetched).")
        
        # Check chat (cost money)
        print("   Attempting chat completion (minimal)...")
        client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-preview-02-05:free", # Use free model if possible to test auth
            messages=[{"role": "user", "content": "hi"}],
        )
        print("✅ OpenRouter Chat is WORKING.")
    except Exception as e:
        print(f"❌ OpenRouter Error: {e}")

def check_deepseek():
    print("\n--- Checking DeepSeek ---")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        print("❌ DEEPSEEK_API_KEY not found in env.")
        return

    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=key,
    )
    try:
        # Minimal request - check balance or models
        client.models.list()
        print("✅ DeepSeek Key is VALID (Models list fetched).")
        
        print("   Attempting chat completion...")
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        print("✅ DeepSeek Chat is WORKING.")
    except Exception as e:
        print(f"❌ DeepSeek Error: {e}")

if __name__ == "__main__":
    check_openrouter()
    check_deepseek()
