"""
Director's Cut - MCP Server
Exposes video analysis and editing tools to the agent.

This server provides tools for:
- YouTube video analysis and transcript extraction
- Hotspot detection (finding engaging moments)
- AI-powered video editing
- Production enhancements (smart crop, intros, subtitles)
"""

from mcp.server.fastmcp import FastMCP
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import google.generativeai as genai
import os
import re
import json
import tempfile
import shutil
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Literal
import sys

# Add project root to sys.path to allow 'from src...' imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.paths import (
    PROJECT_ROOT as PATHS_PROJECT_ROOT,
    INPUT_DIR,
    OUTPUT_DIR,
    ensure_runtime_dirs,
)

PROJECT_ROOT = PATHS_PROJECT_ROOT

# Import local modules
from src.scout import SignalScout
from src.verifier import Verifier
from src.director import Director
from src.hands import Hands
from src.showrunner import Showrunner
from src.utils import download_video_segment, download_audio, get_video_info, ensure_directories, extract_video_id

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Director's Cut")

# Initialize Gemini API if key is available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Change to project root directory so relative paths work
os.chdir(PROJECT_ROOT)

# Initialize components
scout = SignalScout()
verifier = Verifier()
director = Director()
hands = Hands()

# Ensure runtime directories exist
ensure_runtime_dirs()

# --- Helper Functions ---

# extract_video_id moved to src/utils.py

# --- Existing Transcript Tools (from repo_b) ---

@mcp.tool()
def get_transcript(video_url_or_id: str, include_timestamps: bool = False) -> str:
    """Fetch video transcript by URL or video ID."""
    try:
        video_id = extract_video_id(video_url_or_id)
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = transcript_list.fetch()
        
        transcript_data = transcript.fetch()
        
        if include_timestamps:
            formatted = []
            for entry in transcript_data:
                start_time = int(entry.start if hasattr(entry, 'start') else entry['start'])
                text = entry.text if hasattr(entry, 'text') else entry['text']
                minutes = start_time // 60
                seconds = start_time % 60
                formatted.append(f"[{minutes:02d}:{seconds:02d}] {text}")
            return "\n".join(formatted)
        else:
            texts = [entry.text if hasattr(entry, 'text') else entry['text'] for entry in transcript_data]
            return " ".join(texts)
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

@mcp.tool()
def summarize_video(video_url_or_id: str, summary_length: str = "medium") -> str:
    """Summarize video using Gemini Flash Lite after fetching the transcript."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not set."
    try:
        transcript = get_transcript(video_url_or_id, include_timestamps=False)
        if transcript.startswith("Error"):
            return transcript
        
        prompt = f"Summarize this video transcript ({summary_length}):\n\n{transcript}"
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# --- New Video Editing Tools ---

@mcp.tool()
def scan_video(video_url: str) -> str:
    """
    Step 1: Scout.
    Downloads the audio and scans for hotspots using audio analysis.
    Returns a list of timestamps with high audio activity.
    """
    try:
        video_id = extract_video_id(video_url)

        print(f"Downloading audio for scanning: {video_id}")
        output_path = os.path.join(INPUT_DIR, video_id)

        # Download full audio for analysis (much faster than video)
        audio_path = output_path + ".mp3"
        if not os.path.exists(audio_path):
            try:
                audio_path = download_audio(video_url, output_path)
            except Exception as e:
                return f"Error: Cannot download audio from YouTube. {str(e)}"

        # Run SignalScout on the full audio
        hotspots = scout.analyze(audio_path)

        if not hotspots:
            return "Error: No hotspots found in video. The audio may be too quiet or uniform."

        return json.dumps(hotspots, indent=2)
    except Exception as e:
        return f"Error scanning video: {str(e)}"

@mcp.tool()
def analyze_hotspot(video_url: str, timestamp: float) -> str:
    """
    Step 2: Verifier.
    Extracts a frame from the hotspot and uses Vision AI to verify it.
    """
    try:
        video_id = extract_video_id(video_url)
        # Assuming video is already downloaded in the runtime input directory from step 1
        # In a real flow, we'd manage paths better
        video_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            return "Error: Video file not found. Run scan_video first."
            
        result = verifier.verify(video_path, timestamp)
        return str(result)
    except Exception as e:
        return f"Error verifying hotspot: {str(e)}"

@mcp.tool()
def render_edit(video_url: str, hotspots_str: str) -> str:
    """
    Step 3 & 4: Director & Hands.
    Takes verified hotspots, creates a plan, and renders it.

    Args:
        video_url: YouTube video URL
        hotspots_str: JSON string of hotspots, e.g. '[{"start": 10, "end": 20, "score": 0.9}]'
    """
    try:
        video_id = extract_video_id(video_url)
        video_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            return "Error: Video file not found. Run scan_video first to download the video."

        # Parse hotspots - handle both string and list input
        # MCP clients may send either a JSON string or a parsed list/dict
        if isinstance(hotspots_str, list):
            # Already a list, use directly
            hotspots = hotspots_str
        elif isinstance(hotspots_str, dict):
            # Single hotspot as dict, wrap in list
            hotspots = [hotspots_str]
        elif isinstance(hotspots_str, str):
            # JSON string, parse it
            try:
                hotspots = json.loads(hotspots_str)
            except json.JSONDecodeError as e:
                return f"Error: Invalid hotspots JSON string. Parse error: {str(e)}"
        else:
            return f"Error: Invalid hotspots type. Expected string or list, got {type(hotspots_str).__name__}"

        if not hotspots:
            return "Error: No hotspots provided."
            
        # Get video info
        info = get_video_info(video_url)
        
        # Director creates plan
        edit_plan = director.create_edit_plan(hotspots, info)
        if not edit_plan:
            return "Error: Director failed to create a plan."
            
        print(f"Edit Plan: {edit_plan}")
        
        # Hands execute plan
        output_path = hands.execute(video_path, edit_plan, output_filename=f"{video_id}_edit.mp4")
        
        return f"Rendered edit to: {output_path}"

    except Exception as e:
        return f"Error rendering edit: {str(e)}"


# --- New Showrunner Tools ---

@mcp.tool()
def get_youtube_info(video_url: str) -> dict:
    """
    Get metadata for a YouTube video without downloading.

    Args:
        video_url: YouTube video URL

    Returns:
        Dictionary with title, duration, uploader, description, tags, etc.
    """
    try:
        info = get_video_info(video_url)
        return {
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "duration_formatted": f"{info.get('duration', 0) // 60}m {info.get('duration', 0) % 60}s",
            "uploader": info.get("uploader", ""),
            "channel": info.get("channel", ""),
            "view_count": info.get("view_count", 0),
            "upload_date": info.get("upload_date", ""),
            "description": info.get("description", "")[:500] + "..." if len(info.get("description", "")) > 500 else info.get("description", ""),
            "tags": info.get("tags", [])[:10],
            "thumbnail": info.get("thumbnail", ""),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def create_viral_clip(
    video_url: str,
    target_duration: int = 30,
    verify_clips: bool = True
) -> str:
    """
    Full pipeline: Download YouTube video, find best moments, and create an edited clip.

    This runs the complete Director's Cut pipeline:
    1. Downloads audio and analyzes for hotspots (Scout)
    2. Downloads video segments and verifies quality (Verifier)
    3. Creates edit plan selecting best moments (Director)
    4. Renders final video (Hands)

    Args:
        video_url: YouTube video URL
        target_duration: Target duration in seconds (30-60 recommended)
        verify_clips: Whether to verify clips with Vision AI (slower but better quality)

    Returns:
        Path to the rendered video file, or error message
    """
    temp_dir = tempfile.mkdtemp(prefix="directors_cut_")

    try:
        video_id = extract_video_id(video_url)
        print(f"[1/5] Getting video info for {video_id}...")

        # Get video info
        video_info = get_video_info(video_url)
        if not video_info:
            return "Error: Could not fetch video info"

        print(f"[2/5] Downloading audio and scanning for hotspots...")

        # Download audio for scouting
        audio_path = os.path.join(temp_dir, "audio")
        actual_audio_path = download_audio(video_url, audio_path)

        # Run scout to find hotspots
        hotspots = scout.analyze(actual_audio_path)

        if not hotspots:
            return "Error: No hotspots found in video"

        # Sort by score and take top candidates
        hotspots.sort(key=lambda x: x.get('score', 0), reverse=True)
        candidates = hotspots[:8]

        print(f"[3/5] Found {len(candidates)} candidate hotspots, downloading clips...")

        # Download and optionally verify clips
        verified_hotspots = []
        clips_metadata = []

        for i, h in enumerate(candidates):
            start = max(0, h['start'] - 2)
            end = h['end'] + 2
            clip_path = os.path.join(temp_dir, f"clip_{i}")

            try:
                downloaded_path = download_video_segment(
                    video_url, start, end, clip_path, format_spec='best'
                )

                if verify_clips:
                    # Verify with Vision AI
                    clip_offset = h['start'] - start
                    result = verifier.verify(downloaded_path, clip_offset)
                    score = result.get('score', 0)

                    if score > 4:
                        h['verified_score'] = score
                        verified_hotspots.append(h)
                        clips_metadata.append({
                            'start': start,
                            'end': end,
                            'path': downloaded_path,
                            'hotspot': h
                        })
                else:
                    verified_hotspots.append(h)
                    clips_metadata.append({
                        'start': start,
                        'end': end,
                        'path': downloaded_path,
                        'hotspot': h
                    })

            except Exception as e:
                print(f"  Clip {i} failed: {e}")
                continue

        if not verified_hotspots:
            return "Error: No clips passed verification"

        print(f"[4/5] Creating edit plan from {len(verified_hotspots)} verified clips...")

        # Director creates plan
        edit_plan = director.create_edit_plan(verified_hotspots, video_info)

        if not edit_plan:
            return "Error: Director could not create edit plan"

        # Map plan to downloaded clips
        final_plan = []
        for item in edit_plan:
            item_start = item.get('start')
            item_end = item.get('end')

            if item_start is None or item_end is None:
                continue

            for clip in clips_metadata:
                clip_start = clip['start']
                clip_end = clip['end']

                if (clip_start - 1.0 <= item_start < clip_end) and (clip_start <= item_end <= clip_end + 1.0):
                    item['clip_path'] = clip['path']
                    item['source_start'] = clip['start']
                    final_plan.append(item)
                    break

        if not final_plan:
            return "Error: Could not map edit plan to clips"

        print(f"[5/5] Rendering final video...")

        # Hands render
        output_filename = f"viral_clip_{video_id}_{int(time.time())}.mp4"
        output_path = hands.execute(
            clips_metadata[0]['path'],
            final_plan,
            output_filename=output_filename
        )

        if output_path and os.path.exists(output_path):
            return f"Success! Video saved to: {output_path}"
        else:
            return "Error: Rendering failed"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@mcp.tool()
def smart_crop_video(video_path: str) -> str:
    """
    Apply AI-powered smart crop to convert video to 9:16 vertical format.

    Uses Gemini to detect scene changes and Qwen VL to track subject positions,
    then applies smooth interpolated cropping that follows the subject.

    Args:
        video_path: Path to the input video file

    Returns:
        Path to the cropped 9:16 video, or error message
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    try:
        showrunner = Showrunner()
        output_path = showrunner.smart_crop_pipeline(video_path)

        if output_path and os.path.exists(output_path):
            # Copy to output directory
            final_path = os.path.join(OUTPUT_DIR, f"cropped_{int(time.time())}.mp4")
            shutil.copy(output_path, final_path)
            return f"Success! Cropped video saved to: {final_path}"
        else:
            return "Error: Smart crop failed"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def add_production_value(
    video_path: str,
    mood: Literal["auto", "hype", "suspense", "chill"] = "auto",
    enable_smart_crop: bool = True,
    add_intro_image: bool = True,
    add_subtitles: bool = True
) -> str:
    """
    Transform a video with professional production enhancements.

    This applies the full Showrunner pipeline:
    - AI mood detection and creative direction
    - Smart vertical crop (9:16) that follows subjects
    - AI-generated intro screen (FLUX)
    - Professional voiceover intro (ElevenLabs)
    - Auto-generated subtitles (WhisperX)
    - Background music matching mood

    Args:
        video_path: Path to the input video file
        mood: Video mood - "auto" (AI detects), "hype", "suspense", or "chill"
        enable_smart_crop: Apply AI smart crop to 9:16 format
        add_intro_image: Generate AI intro screen
        add_subtitles: Add auto-generated subtitles

    Returns:
        Path to the polished video, or error message
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    try:
        showrunner = Showrunner()

        # Get creative direction
        if mood == "auto":
            direction = showrunner.analyze_video(video_path)
            detected_mood = direction.get('mood', 'hype')
            intro_script = direction.get('intro_script', 'Check this out!')
            title_text = direction.get('title_card', 'WATCH THIS')
        else:
            detected_mood = mood
            intro_script = "Get ready for something incredible..."
            title_text = "WATCH THIS"

        print(f"Creative Direction: mood={detected_mood}, title={title_text}")

        # Generate voiceover
        intro_audio = showrunner.generate_intro(intro_script)
        if not intro_audio or not os.path.exists(intro_audio):
            print("Warning: Voiceover generation failed, continuing without intro audio")
            intro_audio = None

        # Select background music
        bg_music = showrunner.select_music(detected_mood)

        # Compose final video with all enhancements
        final_video = showrunner.compose_final(
            video_path,
            intro_audio,
            bg_music,
            title_text=title_text,
            mood=detected_mood,
            enable_smart_crop=enable_smart_crop,
            add_intro_image=add_intro_image,
            add_subtitles=add_subtitles
        )

        if final_video and os.path.exists(final_video):
            return f"Success! Polished video saved to: {final_video}"
        else:
            return "Error: Production failed"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def generate_intro_voiceover(script_text: str) -> str:
    """
    Generate a professional voiceover audio file from text using ElevenLabs.

    Args:
        script_text: The text to convert to speech

    Returns:
        Path to the generated audio file, or error message
    """
    try:
        showrunner = Showrunner()
        audio_path = showrunner.generate_intro(script_text)

        if audio_path and os.path.exists(audio_path):
            return f"Success! Voiceover saved to: {audio_path}"
        else:
            return "Error: Voiceover generation failed. Check ELEVENLABS_API_KEY."

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def find_hotspots(video_url: str, top_n: int = 5) -> str:
    """
    Analyze a YouTube video and find the most engaging moments (hotspots).

    Downloads audio and uses signal analysis to detect:
    - High energy audio moments
    - Scene changes
    - Audio peaks

    Args:
        video_url: YouTube video URL
        top_n: Number of top hotspots to return

    Returns:
        JSON string with hotspot timestamps and scores
    """
    temp_dir = tempfile.mkdtemp(prefix="hotspots_")

    try:
        # Download audio
        audio_path = os.path.join(temp_dir, "audio")
        actual_audio_path = download_audio(video_url, audio_path)

        # Run scout
        hotspots = scout.analyze(actual_audio_path)

        # Sort and limit
        hotspots.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_hotspots = hotspots[:top_n]

        # Format nicely
        result = []
        for i, h in enumerate(top_hotspots, 1):
            result.append({
                "rank": i,
                "start": round(h.get('start', 0), 1),
                "end": round(h.get('end', 0), 1),
                "score": round(h.get('score', 0), 2),
                "type": h.get('type', 'unknown')
            })

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# --- MCP Resources ---

@mcp.resource("resource://supported_features")
def get_supported_features() -> dict:
    """List of features available in Director's Cut."""
    return {
        "video_analysis": ["transcript", "hotspot_detection", "summarization"],
        "video_editing": ["clip_extraction", "multi_clip_assembly", "intro_generation"],
        "production": ["smart_crop_9x16", "ai_subtitles", "background_music", "voiceover"],
        "ai_models": ["gemini_flash", "gemini_vision", "qwen_vl", "elevenlabs", "flux"]
    }


@mcp.resource("resource://mood_options")
def get_mood_options() -> dict:
    """Available mood options for video production."""
    return {
        "auto": "AI automatically detects the mood from video content",
        "hype": "High energy, exciting, action-packed",
        "suspense": "Tense, dramatic, building anticipation",
        "chill": "Relaxed, calm, ambient"
    }


# --- MCP Prompts ---

@mcp.prompt()
def viral_video_workflow() -> str:
    """Step-by-step guide for creating viral video content."""
    return """To create a viral video clip from YouTube:

1. **Get video info first:**
   get_youtube_info(video_url) - Check duration, title, content type

2. **Find engaging moments:**
   find_hotspots(video_url, top_n=5) - Get timestamps of best moments

3. **Create the clip:**
   create_viral_clip(video_url, target_duration=30) - Full auto pipeline

4. **Add production value (optional):**
   add_production_value(video_path, mood="auto") - Smart crop, intro, subtitles

Tips:
- Videos 5-30 minutes work best for finding good moments
- Target 30-60 second clips for social media
- Enable smart crop for vertical (TikTok/Reels) format"""


@mcp.prompt()
def production_workflow() -> str:
    """Guide for adding production value to existing videos."""
    return """To polish an existing video:

1. **Smart crop to vertical:**
   smart_crop_video(video_path) - AI tracks subjects, crops to 9:16

2. **Full production pipeline:**
   add_production_value(video_path, mood="auto", enable_smart_crop=True, add_intro_image=True, add_subtitles=True)

3. **Just voiceover:**
   generate_intro_voiceover("Your hook text here")

Features applied:
- Gemini analyzes scene changes
- Qwen VL tracks subject positions
- ElevenLabs generates voiceover
- FLUX creates AI intro image
- WhisperX adds word-level subtitles
- Background music matched to mood"""


def main():
    mcp.run()

if __name__ == "__main__":
    main()
