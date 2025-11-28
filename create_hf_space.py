#!/usr/bin/env python3
"""
Script to create a Hugging Face Space and connect it to the GitHub repository.
Requires HF_TOKEN environment variable or will prompt for it.
"""

import os
import sys
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError

def main():
    # Get token from command line, environment, or prompt
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ Hugging Face token required!")
        print("\nUsage:")
        print("  python3 create_hf_space.py YOUR_TOKEN")
        print("  OR")
        print("  HF_TOKEN=your_token python3 create_hf_space.py")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Login
    try:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)
    
    # Get username
    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name", "unknown")
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Could not get user info: {e}")
        sys.exit(1)
    
    # Space configuration
    space_id = "directors-cut"
    repo_id = f"{username}/{space_id}"
    github_repo = "tayyab415/directors-cut"
    
    print(f"\n📦 Creating Space: {repo_id}")
    print(f"🔗 Connecting to GitHub: {github_repo}")
    
    # Create Space
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,  # Set to True if you want it private
        )
        print(f"✅ Space created: https://huggingface.co/spaces/{repo_id}")
        
        # Update SDK version via repo update (if needed)
        # The SDK version is typically set in the README.md frontmatter
        print("   Note: SDK version 6.0.0 should be set in README.md frontmatter")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"⚠️  Space already exists: {repo_id}")
            print(f"   Access it at: https://huggingface.co/spaces/{repo_id}")
        else:
            print(f"❌ Failed to create Space: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    print("\n📝 Next steps:")
    print(f"1. Go to: https://huggingface.co/spaces/{repo_id}/settings")
    print("2. Connect your GitHub repository:")
    print(f"   - Repository: {github_repo}")
    print("   - Or clone the Space repo and push your code")
    print("\n3. Set environment variables (Settings → Variables and secrets):")
    print("   - GEMINI_API_KEY")
    print("   - VIDEO_API_KEY")
    print("   - ELEVENLABS_API_KEY (optional)")
    print("   - NEBIUS_API_KEY (optional)")
    print(f"\n4. Your Space will be available at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    main()

