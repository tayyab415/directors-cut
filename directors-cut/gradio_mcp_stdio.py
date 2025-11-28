"""
Director's Cut - Gradio Tools as STDIO MCP Server

This wraps the Gradio server tools as a FastMCP STDIO server
so it works with Claude Desktop without requiring Node.js/npx.
"""

import os
import sys

# Add the directors-cut directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Literal
from mcp.server.fastmcp import FastMCP

# Import video processing modules
from video_processing.download import (
    download_youtube_video,
    get_video_metadata,
    get_playlist_info,
)
from video_processing.viral_detection import (
    analyze_video_segments,
    get_best_segment,
    VIRAL_WEIGHTS,
    to_dict as viral_to_dict,
)
from video_processing.conversion import (
    convert_to_vertical,
)
from video_processing.intro_generation import (
    generate_intro,
    get_intro_templates,
)
from video_processing.subtitle_generation import (
    add_subtitles,
)
from video_processing.ffmpeg_utils import (
    trim_video,
    concatenate_videos,
    get_video_info,
)

import uuid

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in [INPUTS_DIR, TEMP_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize FastMCP server
mcp = FastMCP("Director's Cut")

# Job tracking
PROCESSING_JOBS: dict[str, dict] = {}


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
def directors_cut_main(
    youtube_url: str,
    intro_text: str = "Check this out!",
    target_duration: int = 30,
) -> str:
    """
    Extract viral segment from YouTube video, convert to 9:16 vertical format,
    add AI-generated intro with custom text, and produce social media-ready content.

    Args:
        youtube_url: Full YouTube video URL to process
        intro_text: Custom text to display in the intro
        target_duration: Target duration in seconds for the final clip

    Returns:
        Path to the final processed video file
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # Step 1: Download video
        success, result = download_youtube_video(
            youtube_url,
            output_dir=job_dir,
            filename="original",
        )

        if not success:
            return f"Error downloading video: {result}"

        original_path = result

        # Step 2: Find best segment
        segment = get_best_segment(original_path, target_duration)

        if segment:
            start_time, end_time = segment
        else:
            start_time = 0
            end_time = target_duration

        # Step 3: Extract segment
        segment_path = os.path.join(job_dir, "segment.mp4")
        success, msg = trim_video(
            original_path,
            segment_path,
            start_time=start_time,
            duration=target_duration,
        )

        if not success:
            return f"Error extracting segment: {msg}"

        # Step 4: Convert to vertical
        vertical_path = os.path.join(job_dir, "vertical.mp4")
        success, result = convert_to_vertical(
            segment_path,
            vertical_path,
            crop_mode="smart",
        )

        if not success:
            return f"Error converting to vertical: {result}"

        # Step 5: Generate intro
        intro_path = os.path.join(job_dir, "intro.mp4")
        success, result = generate_intro(
            text=intro_text,
            duration=3,
            style="modern",
        )

        if not success:
            return f"Error generating intro: {result}"

        intro_path = result

        # Step 6: Concatenate
        final_path = os.path.join(OUTPUTS_DIR, f"directors_cut_{job_id}.mp4")
        success, msg = concatenate_videos([intro_path, vertical_path], final_path)

        if not success:
            return f"Error concatenating: {msg}"

        return final_path

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def analyze_video_segments_tool(
    youtube_url: str,
    segment_length: int = 30,
) -> dict:
    """
    Analyze video and return top 3 most engaging segments with timestamps and scores.

    Args:
        youtube_url: YouTube video URL to analyze
        segment_length: Length of each segment to analyze in seconds

    Returns:
        Dictionary with segments containing start, end, score, and reason
    """
    job_dir = os.path.join(TEMP_DIR, str(uuid.uuid4())[:8])
    os.makedirs(job_dir, exist_ok=True)

    try:
        success, result = download_youtube_video(
            youtube_url,
            output_dir=job_dir,
            filename="analyze",
        )

        if not success:
            return {"error": f"Download failed: {result}"}

        analysis = analyze_video_segments(
            result,
            segment_length=segment_length,
            top_n=3,
        )

        return viral_to_dict(analysis)

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def convert_to_vertical_tool(
    video_path: str,
    crop_mode: Literal["smart", "center", "left", "right"] = "smart",
) -> str:
    """
    Convert landscape video to 9:16 vertical format with intelligent subject tracking.

    Args:
        video_path: Path to the input video file
        crop_mode: Cropping mode - "smart", "center", "left", or "right"

    Returns:
        Path to the converted vertical video
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    output_path = os.path.join(
        OUTPUTS_DIR,
        f"vertical_{uuid.uuid4().hex[:8]}.mp4"
    )

    success, result = convert_to_vertical(
        video_path,
        output_path,
        crop_mode=crop_mode,
    )

    if success:
        return result
    return f"Error: {result}"


@mcp.tool()
def generate_custom_intro(
    text: str,
    duration: int = 3,
    style: Literal["modern", "energetic", "professional", "minimal"] = "modern",
) -> str:
    """
    Generate AI-powered video intro with custom text and background.

    Args:
        text: Text to display in the intro
        duration: Intro duration in seconds
        style: Visual style - "modern", "energetic", "professional", or "minimal"

    Returns:
        Path to the generated intro video file
    """
    output_path = os.path.join(
        OUTPUTS_DIR,
        f"intro_{uuid.uuid4().hex[:8]}.mp4"
    )

    success, result = generate_intro(
        text=text,
        duration=duration,
        style=style,
        output_path=output_path,
    )

    if success:
        return result
    return f"Error: {result}"


@mcp.tool()
def add_auto_subtitles(
    video_path: str,
    language: str = "en",
) -> str:
    """
    Automatically transcribe speech and add burned-in subtitles to video.

    Args:
        video_path: Path to the input video file
        language: Language code for transcription (default: "en")

    Returns:
        Path to the video with burned-in subtitles
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    output_path = os.path.join(
        OUTPUTS_DIR,
        f"subtitled_{uuid.uuid4().hex[:8]}.mp4"
    )

    success, result = add_subtitles(
        video_path,
        output_path,
        language=language,
    )

    if success:
        return result
    return f"Error: {result}"


@mcp.tool()
def get_youtube_info(youtube_url: str) -> dict:
    """
    Get metadata for a YouTube video without downloading.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Dictionary with title, duration, uploader, views, etc.
    """
    metadata = get_video_metadata(youtube_url)

    if metadata:
        return {
            "title": metadata.title,
            "duration": metadata.duration,
            "uploader": metadata.uploader,
            "view_count": metadata.view_count,
            "like_count": metadata.like_count,
            "upload_date": metadata.upload_date,
            "tags": metadata.tags[:10] if metadata.tags else [],
        }

    return {"error": "Could not fetch video metadata"}


# ============================================================================
# RESOURCES
# ============================================================================

@mcp.resource("resource://supported_formats")
def get_supported_formats() -> list[str]:
    """List of video formats Director's Cut can process."""
    return ["mp4", "mov", "avi", "mkv", "webm", "flv"]


@mcp.resource("resource://intro_templates")
def get_intro_templates_resource() -> dict:
    """Available intro templates with style descriptions."""
    return get_intro_templates()


@mcp.resource("resource://viral_detection_criteria")
def get_viral_criteria() -> dict:
    """Metrics and weights used for detecting engaging video segments."""
    return {
        "motion_intensity": {
            "weight": VIRAL_WEIGHTS["motion_intensity"],
            "description": "Amount of movement in frame",
        },
        "face_detection": {
            "weight": VIRAL_WEIGHTS["face_detection"],
            "description": "Presence and expressions of faces",
        },
        "scene_changes": {
            "weight": VIRAL_WEIGHTS["scene_changes"],
            "description": "Frequency of scene transitions",
        },
        "audio_peaks": {
            "weight": VIRAL_WEIGHTS["audio_peaks"],
            "description": "Loud or interesting audio moments",
        },
        "color_variance": {
            "weight": VIRAL_WEIGHTS["color_variance"],
            "description": "Visual interest from color changes",
        },
    }


@mcp.resource("resource://output_specifications")
def get_output_specs() -> dict:
    """Default output specifications for different social media platforms."""
    return {
        "tiktok": {
            "aspect_ratio": "9:16",
            "max_duration": 60,
            "resolution": "1080x1920",
            "fps": 30,
            "format": "mp4",
        },
        "instagram_reels": {
            "aspect_ratio": "9:16",
            "max_duration": 90,
            "resolution": "1080x1920",
            "fps": 30,
            "format": "mp4",
        },
        "youtube_shorts": {
            "aspect_ratio": "9:16",
            "max_duration": 60,
            "resolution": "1080x1920",
            "fps": 30,
            "format": "mp4",
        },
    }


# ============================================================================
# PROMPTS
# ============================================================================

@mcp.prompt()
def optimize_for_platform(
    platform: Literal["tiktok", "instagram_reels", "youtube_shorts"] = "tiktok",
) -> str:
    """Get optimization prompt for specific social media platform."""
    specs = get_output_specs()
    platform_spec = specs.get(platform, specs["tiktok"])

    platform_tips = {
        "tiktok": "Hook viewers in first 1-3 seconds. Use trending sounds. Add captions.",
        "instagram_reels": "Use hashtags strategically. Post during peak hours.",
        "youtube_shorts": "Use descriptive titles. Add end screens. Optimize for search.",
    }

    return f"""Optimize this video for {platform} by:
1. Ensuring aspect ratio is {platform_spec['aspect_ratio']}
2. Duration under {platform_spec['max_duration']} seconds
3. Resolution of {platform_spec['resolution']}
4. {platform_tips.get(platform, '')}
5. Export as {platform_spec['format']} format"""


@mcp.prompt()
def create_viral_compilation(
    topic: Literal["fails", "wins", "tutorials", "reactions"] = "fails",
) -> str:
    """Prompt for creating viral compilation videos."""
    topic_specifics = {
        "fails": "Focus on unexpected outcomes. Build tension before the fail moment.",
        "wins": "Highlight the achievement. Show the before and after.",
        "tutorials": "Break into clear steps. Use zoom for details.",
        "reactions": "Capture genuine emotions. Use split-screen when relevant.",
    }

    return f"""Create a {topic} compilation by:
1. Extract 5-10 best moments from source videos
2. Add countdown numbers or transitions between clips
3. Use fast transitions (0.2-0.5s) between segments
4. Add upbeat background music at 30% volume
5. {topic_specifics.get(topic, '')}
6. Keep total duration under 60 seconds"""


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    mcp.run()
