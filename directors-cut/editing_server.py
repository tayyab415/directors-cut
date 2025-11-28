"""
Director's Cut - FastMCP Editing Server

Fast local editing operations for iterative video refinements.
Transport: STDIO (local)

Run with: python editing_server.py
Or: mcp run editing_server.py
"""

import os
import uuid
from datetime import datetime
from typing import Literal, Optional
from dataclasses import dataclass, asdict

from mcp.server.fastmcp import FastMCP

# Import video processing modules
from video_processing.ffmpeg_utils import (
    trim_video as ffmpeg_trim,
    apply_video_filter,
    adjust_brightness as ffmpeg_brightness,
    change_speed,
    add_audio_to_video,
    get_video_info,
)
from video_processing.conversion import (
    change_aspect_ratio as convert_aspect,
)
from video_processing.intro_generation import (
    generate_intro,
    add_intro_to_video,
)


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FastMCP server
mcp = FastMCP("Director's Cut Editor")

# Editing history tracking
EDITING_HISTORY: list[dict] = []


def log_operation(operation: str, args: dict, input_path: str, output_path: str):
    """Log an editing operation to history."""
    EDITING_HISTORY.append({
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "args": args,
        "input": input_path,
        "output": output_path,
    })


def generate_output_path(prefix: str = "edited") -> str:
    """Generate a unique output path."""
    return os.path.join(OUTPUTS_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}.mp4")


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
def trim_video(
    video_path: str,
    seconds_to_cut: int,
    from_end: bool = True,
) -> str:
    """
    Remove seconds from start or end of video.

    Fast operation using stream copy (no re-encoding).

    Args:
        video_path: Path to the input video file
        seconds_to_cut: Number of seconds to remove
        from_end: If True, cut from end; if False, cut from start

    Returns:
        Path to the trimmed video file
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    info = get_video_info(video_path)
    if not info:
        return "Error: Could not get video info"

    output_path = generate_output_path("trimmed")

    if from_end:
        # Cut from end - set new duration
        new_duration = max(1, info.duration - seconds_to_cut)
        success, result = ffmpeg_trim(
            video_path,
            output_path,
            duration=new_duration,
            copy_codec=True,
        )
    else:
        # Cut from start - skip beginning
        success, result = ffmpeg_trim(
            video_path,
            output_path,
            start_time=seconds_to_cut,
            copy_codec=True,
        )

    if success:
        log_operation(
            "trim_video",
            {"seconds": seconds_to_cut, "from_end": from_end},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {result}"


@mcp.tool()
def change_intro_background(
    video_path: str,
    background_prompt: str,
    intro_duration: int = 3,
) -> str:
    """
    Regenerate video intro with new AI-generated background while keeping same text.

    This creates a new intro and replaces the first N seconds of the video.

    Args:
        video_path: Path to the video file
        background_prompt: Description for new background (e.g., "sunset beach", "city skyline")
        intro_duration: Duration of intro to replace in seconds (default: 3)

    Returns:
        Path to video with new intro
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    info = get_video_info(video_path)
    if not info:
        return "Error: Could not get video info"

    # Generate new intro (text would need to be extracted or provided)
    # For now, use a default text
    intro_text = "Check this out!"

    temp_intro = os.path.join(TEMP_DIR, f"new_intro_{uuid.uuid4().hex[:8]}.mp4")
    success, intro_result = generate_intro(
        text=intro_text,
        duration=intro_duration,
        style="modern",
        background_prompt=background_prompt,
        output_path=temp_intro,
    )

    if not success:
        return f"Error generating intro: {intro_result}"

    # Extract main content (skip original intro)
    temp_main = os.path.join(TEMP_DIR, f"main_{uuid.uuid4().hex[:8]}.mp4")
    success, trim_result = ffmpeg_trim(
        video_path,
        temp_main,
        start_time=intro_duration,
        copy_codec=True,
    )

    if not success:
        return f"Error extracting main content: {trim_result}"

    # Concatenate new intro with main content
    output_path = generate_output_path("new_intro")
    success, concat_result = add_intro_to_video(temp_intro, temp_main, output_path)

    # Clean up temp files
    for temp_file in [temp_intro, temp_main]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if success:
        log_operation(
            "change_intro_background",
            {"background_prompt": background_prompt},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {concat_result}"


@mcp.tool()
def adjust_brightness(
    video_path: str,
    brightness_level: float = 1.2,
) -> str:
    """
    Adjust video brightness/exposure.

    Args:
        video_path: Path to the input video file
        brightness_level: Brightness multiplier (1.0 = original, 1.5 = 50% brighter, 0.8 = 20% darker)

    Returns:
        Path to the adjusted video file
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    output_path = generate_output_path("brightness")

    success, result = ffmpeg_brightness(
        video_path,
        output_path,
        brightness=brightness_level,
    )

    if success:
        log_operation(
            "adjust_brightness",
            {"brightness_level": brightness_level},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {result}"


@mcp.tool()
def add_background_music(
    video_path: str,
    music_style: Literal["upbeat", "chill", "dramatic", "minimal", "none"] = "upbeat",
    volume: float = 0.3,
) -> str:
    """
    Add AI-generated or preset background music to video.

    Args:
        video_path: Path to the input video file
        music_style: Music style - "upbeat", "chill", "dramatic", "minimal", or "none" (removes music)
        volume: Music volume level 0.0-1.0 (mixed with original audio)

    Returns:
        Path to video with background music
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    if music_style == "none":
        # Remove audio track
        output_path = generate_output_path("no_music")
        from video_processing.ffmpeg_utils import run_ffmpeg
        success, result = run_ffmpeg(
            ["-an", "-c:v", "copy"],
            input_file=video_path,
            output_file=output_path,
        )
        if success:
            return output_path
        return f"Error: {result}"

    # For now, return a placeholder message
    # In production, this would fetch or generate music based on style
    return f"Music addition for style '{music_style}' at volume {volume} - music library integration pending"


@mcp.tool()
def change_aspect_ratio(
    video_path: str,
    aspect_ratio: Literal["9:16", "16:9", "1:1", "4:5"] = "9:16",
    crop_position: Literal["center", "top", "bottom", "left", "right", "smart"] = "center",
) -> str:
    """
    Change video aspect ratio with smart or manual cropping.

    Args:
        video_path: Path to the input video file
        aspect_ratio: Target aspect ratio - "9:16", "16:9", "1:1", or "4:5"
        crop_position: Crop alignment - "center", "top", "bottom", "left", "right", or "smart"

    Returns:
        Path to the cropped video file
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    output_path = generate_output_path(f"aspect_{aspect_ratio.replace(':', 'x')}")

    success, result = convert_aspect(
        video_path,
        output_path,
        aspect_ratio=aspect_ratio,
        crop_position=crop_position,
    )

    if success:
        log_operation(
            "change_aspect_ratio",
            {"aspect_ratio": aspect_ratio, "crop_position": crop_position},
            video_path,
            output_path,
        )
        return result

    return f"Error: {result}"


@mcp.tool()
def apply_filter(
    video_path: str,
    filter_name: Literal["sepia", "grayscale", "vintage", "vibrant", "blur", "sharpen"] = "vibrant",
) -> str:
    """
    Apply visual filter/effect to video.

    Args:
        video_path: Path to the input video file
        filter_name: Filter to apply - "sepia", "grayscale", "vintage", "vibrant", "blur", or "sharpen"

    Returns:
        Path to the filtered video file
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    output_path = generate_output_path(f"filter_{filter_name}")

    success, result = apply_video_filter(
        video_path,
        output_path,
        filter_name=filter_name,
    )

    if success:
        log_operation(
            "apply_filter",
            {"filter_name": filter_name},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {result}"


@mcp.tool()
def change_playback_speed(
    video_path: str,
    speed_factor: float = 1.0,
) -> str:
    """
    Change video playback speed for slow-motion or time-lapse.

    Args:
        video_path: Path to the input video file
        speed_factor: Speed multiplier (0.5 = half speed/slow-mo, 2.0 = double speed, 0.25 = very slow)

    Returns:
        Path to the speed-adjusted video file
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    if speed_factor <= 0:
        return "Error: Speed factor must be positive"

    output_path = generate_output_path(f"speed_{speed_factor}x")

    success, result = change_speed(
        video_path,
        output_path,
        speed_factor=speed_factor,
    )

    if success:
        log_operation(
            "change_playback_speed",
            {"speed_factor": speed_factor},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {result}"


@mcp.tool()
def extract_segment(
    video_path: str,
    start_time: int,
    end_time: int,
) -> str:
    """
    Extract specific segment from video by timestamp.

    Uses fast stream copy for quick extraction.

    Args:
        video_path: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Path to the extracted segment
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    if end_time <= start_time:
        return "Error: end_time must be greater than start_time"

    duration = end_time - start_time
    output_path = generate_output_path(f"segment_{start_time}_{end_time}")

    success, result = ffmpeg_trim(
        video_path,
        output_path,
        start_time=start_time,
        duration=duration,
        copy_codec=True,
    )

    if success:
        log_operation(
            "extract_segment",
            {"start_time": start_time, "end_time": end_time},
            video_path,
            output_path,
        )
        return output_path

    return f"Error: {result}"


# ============================================================================
# RESOURCES
# ============================================================================

@mcp.resource("resource://editing_history")
def get_editing_history() -> list[dict]:
    """
    Log of all editing operations performed in current session.

    Returns:
        List of operation records with timestamps, operations, args, inputs, and outputs
    """
    return EDITING_HISTORY


@mcp.resource("resource://video_metadata/{video_path}")
def get_video_metadata(video_path: str) -> dict:
    """
    Get detailed metadata for any video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with duration, resolution, fps, codec, bitrate, file size, and audio info
    """
    if not os.path.exists(video_path):
        return {"error": f"File not found: {video_path}"}

    info = get_video_info(video_path)

    if info:
        return {
            "duration": info.duration,
            "resolution": f"{info.width}x{info.height}",
            "fps": info.fps,
            "codec": info.codec,
            "bitrate": info.bitrate,
            "file_size_mb": info.file_size_mb,
            "has_audio": info.has_audio,
            "audio_codec": info.audio_codec,
        }

    return {"error": "Could not read video metadata"}


@mcp.resource("resource://available_filters")
def get_available_filters() -> dict:
    """
    List of all video filters with descriptions and preview images.

    Returns:
        Dictionary of filter names with descriptions and preview paths
    """
    return {
        "sepia": {
            "description": "Vintage warm tone",
            "preview": "/filters/sepia.jpg",
        },
        "grayscale": {
            "description": "Black and white",
            "preview": "/filters/grayscale.jpg",
        },
        "vintage": {
            "description": "Faded vintage look with curves",
            "preview": "/filters/vintage.jpg",
        },
        "vibrant": {
            "description": "Enhanced saturation for vivid colors",
            "preview": "/filters/vibrant.jpg",
        },
        "blur": {
            "description": "Gaussian blur effect",
            "preview": "/filters/blur.jpg",
        },
        "sharpen": {
            "description": "Sharpen edges for clarity",
            "preview": "/filters/sharpen.jpg",
        },
    }


# ============================================================================
# PROMPTS
# ============================================================================

@mcp.prompt()
def quick_social_edit() -> str:
    """
    Common editing sequence for social media optimization.

    Returns a prompt with step-by-step instructions for quick social media edits.

    Returns:
        Formatted prompt with editing steps
    """
    return """Quick social media edit sequence:
1. Trim to under 60 seconds - remove slow intro/outro
2. Boost brightness by 20% for better visibility on mobile
3. Add upbeat background music at 30% volume
4. Convert to 9:16 aspect ratio if not already vertical
5. Apply 'vibrant' filter for eye-catching colors

Run these commands in order:
- trim_video(video_path, seconds_to_cut=5, from_end=True)
- adjust_brightness(video_path, brightness_level=1.2)
- add_background_music(video_path, music_style="upbeat", volume=0.3)
- change_aspect_ratio(video_path, aspect_ratio="9:16", crop_position="smart")
- apply_filter(video_path, filter_name="vibrant")"""


@mcp.prompt()
def fix_common_issues(
    issue_type: Literal["dark", "shaky", "quiet", "blurry", "wrong_aspect"] = "dark",
) -> str:
    """
    Prompt for fixing common video problems.

    Args:
        issue_type: Type of issue - "dark", "shaky", "quiet", "blurry", or "wrong_aspect"

    Returns:
        Formatted prompt with fix instructions
    """
    fixes = {
        "dark": """Fix dark/underexposed video:
1. Increase brightness: adjust_brightness(video_path, brightness_level=1.4)
2. For severely dark videos, try brightness_level=1.6-1.8
3. Consider applying 'vibrant' filter to boost colors
4. Note: Extreme brightness adjustments may introduce noise""",

        "shaky": """Fix shaky video:
1. Unfortunately, stabilization requires re-encoding and specialized processing
2. Consider trimming to the steadiest portions: extract_segment(video_path, start, end)
3. For future: shoot with tripod or stabilizer
4. Slight blur can mask shake: apply_filter(video_path, filter_name="blur")""",

        "quiet": """Fix quiet/low audio video:
1. Add background music to fill gaps: add_background_music(video_path, "chill", 0.5)
2. For voiceover content, consider re-recording with better mic
3. Audio normalization requires specialized processing""",

        "blurry": """Fix blurry video:
1. Apply sharpen filter: apply_filter(video_path, filter_name="sharpen")
2. Note: Sharpening cannot recover lost detail, only enhance edges
3. For motion blur, try speed adjustment: change_playback_speed(video_path, 1.1)
4. Consider re-shooting with better focus/lighting""",

        "wrong_aspect": """Fix wrong aspect ratio:
1. For vertical content: change_aspect_ratio(video_path, "9:16", "smart")
2. For landscape: change_aspect_ratio(video_path, "16:9", "center")
3. For Instagram feed: change_aspect_ratio(video_path, "1:1", "center")
4. Use "smart" crop_position to automatically detect subject""",
    }

    return fixes.get(issue_type, f"Unknown issue type: {issue_type}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with STDIO transport (default for local/CLI use)
    mcp.run()
