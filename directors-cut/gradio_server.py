"""
Director's Cut - Gradio MCP Server

A dual MCP server system for autonomous video editing that transforms
YouTube videos into viral vertical content.

Launch with: python gradio_server.py
MCP endpoint: /gradio_api/mcp/sse
"""

import os
import uuid
import tempfile
from typing import Literal
from dataclasses import dataclass, asdict

import gradio as gr

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
    change_aspect_ratio,
)
from video_processing.intro_generation import (
    generate_intro,
    add_intro_to_video,
    get_intro_templates,
)
from video_processing.subtitle_generation import (
    add_subtitles,
    generate_srt,
)
from video_processing.ffmpeg_utils import (
    trim_video,
    concatenate_videos,
    get_video_info,
)


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories
for d in [INPUTS_DIR, TEMP_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Job status tracking
PROCESSING_JOBS: dict[str, dict] = {}


# ============================================================================
# TOOLS - AI decides when to call these
# ============================================================================

@gr.mcp.tool()
def directors_cut_main(
    youtube_url: str,
    intro_text: str = "Check this out!",
    target_duration: int = 30,
) -> str:
    """
    Extract viral segment from YouTube video, convert to 9:16 vertical format,
    add AI-generated intro with custom text, and produce social media-ready content.

    This is the main pipeline that combines all processing steps.

    Args:
        youtube_url: Full YouTube video URL to process
        intro_text: Custom text to display in the intro (default: "Check this out!")
        target_duration: Target duration in seconds for the final clip (default: 30)

    Returns:
        Path to the final processed video file ready for social media
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    PROCESSING_JOBS[job_id] = {"status": "downloading", "progress": 0}

    try:
        # Step 1: Download video
        PROCESSING_JOBS[job_id] = {"status": "downloading", "progress": 10}
        success, result = download_youtube_video(
            youtube_url,
            output_dir=job_dir,
            filename="original",
        )

        if not success:
            return f"Error downloading video: {result}"

        original_path = result
        PROCESSING_JOBS[job_id] = {"status": "analyzing", "progress": 25}

        # Step 2: Find best segment
        segment = get_best_segment(original_path, target_duration)

        if segment:
            start_time, end_time = segment
        else:
            # Fall back to first N seconds
            start_time = 0
            end_time = target_duration

        # Step 3: Extract segment
        PROCESSING_JOBS[job_id] = {"status": "extracting", "progress": 40}
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
        PROCESSING_JOBS[job_id] = {"status": "converting", "progress": 55}
        vertical_path = os.path.join(job_dir, "vertical.mp4")
        success, result = convert_to_vertical(
            segment_path,
            vertical_path,
            crop_mode="smart",
        )

        if not success:
            return f"Error converting to vertical: {result}"

        # Step 5: Generate intro
        PROCESSING_JOBS[job_id] = {"status": "generating_intro", "progress": 70}
        intro_path = os.path.join(job_dir, "intro.mp4")
        success, result = generate_intro(
            text=intro_text,
            duration=3,
            style="modern",
        )

        if not success:
            return f"Error generating intro: {result}"

        intro_path = result

        # Step 6: Concatenate intro + main video
        PROCESSING_JOBS[job_id] = {"status": "finalizing", "progress": 85}
        final_path = os.path.join(OUTPUTS_DIR, f"directors_cut_{job_id}.mp4")
        success, msg = concatenate_videos([intro_path, vertical_path], final_path)

        if not success:
            return f"Error concatenating: {msg}"

        PROCESSING_JOBS[job_id] = {"status": "completed", "progress": 100}

        return final_path

    except Exception as e:
        PROCESSING_JOBS[job_id] = {"status": "error", "progress": 0, "error": str(e)}
        return f"Error: {str(e)}"


@gr.mcp.tool()
def analyze_video_segments_tool(
    youtube_url: str,
    segment_length: int = 30,
) -> dict:
    """
    Analyze video and return top 3 most engaging segments with timestamps and scores.

    Uses AI-powered analysis to detect motion, faces, scene changes, audio peaks,
    and color variance to identify the most viral-worthy moments.

    Args:
        youtube_url: YouTube video URL to analyze
        segment_length: Length of each segment to analyze in seconds (default: 30)

    Returns:
        Dictionary with segments array containing start, end, score, and reason for each
    """
    job_dir = os.path.join(TEMP_DIR, str(uuid.uuid4())[:8])
    os.makedirs(job_dir, exist_ok=True)

    try:
        # Download video
        success, result = download_youtube_video(
            youtube_url,
            output_dir=job_dir,
            filename="analyze",
        )

        if not success:
            return {"error": f"Download failed: {result}"}

        # Analyze segments
        analysis = analyze_video_segments(
            result,
            segment_length=segment_length,
            top_n=3,
        )

        return viral_to_dict(analysis)

    except Exception as e:
        return {"error": str(e)}


@gr.mcp.tool()
def convert_to_vertical_tool(
    video_path: str,
    crop_mode: Literal["smart", "center", "left", "right"] = "smart",
) -> str:
    """
    Convert landscape video to 9:16 vertical format with intelligent subject tracking.

    Uses face detection to find the best crop position when crop_mode is "smart".

    Args:
        video_path: Path to the input video file
        crop_mode: Cropping mode - "smart" (AI detects subject), "center", "left", or "right"

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


@gr.mcp.tool()
def generate_custom_intro(
    text: str,
    duration: int = 3,
    style: Literal["modern", "energetic", "professional", "minimal"] = "modern",
    background_prompt: str = "abstract geometric shapes",
) -> str:
    """
    Generate AI-powered video intro with custom text and background.

    Creates a professional intro video with animated text overlay.

    Args:
        text: Text to display in the intro
        duration: Intro duration in seconds (default: 3)
        style: Visual style - "modern", "energetic", "professional", or "minimal"
        background_prompt: Description for AI background generation (currently uses style-based gradients)

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
        background_prompt=background_prompt,
        output_path=output_path,
    )

    if success:
        return result
    return f"Error: {result}"


@gr.mcp.tool()
def add_auto_subtitles(
    video_path: str,
    language: str = "en",
) -> str:
    """
    Automatically transcribe speech and add burned-in subtitles to video.

    Uses AI speech recognition to transcribe audio and burn subtitles directly into the video.

    Args:
        video_path: Path to the input video file
        language: Language code for transcription (default: "en" for English)

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


@gr.mcp.tool()
def batch_process_playlist(
    playlist_url: str,
    intro_text: str = "Check this out!",
    target_duration: int = 30,
    max_videos: int = 5,
) -> list[str]:
    """
    Process entire YouTube playlist with the same settings.

    Downloads and processes multiple videos from a playlist using the Director's Cut pipeline.

    Args:
        playlist_url: YouTube playlist URL
        intro_text: Text for intro overlay on all videos
        target_duration: Target duration for each clip in seconds
        max_videos: Maximum number of videos to process (default: 5)

    Returns:
        List of paths to processed video files
    """
    success, videos = get_playlist_info(playlist_url)

    if not success or not videos:
        return [f"Error: Could not get playlist info"]

    results = []

    for video in videos[:max_videos]:
        video_url = video.get("url", "")
        if video_url:
            result = directors_cut_main(
                youtube_url=video_url,
                intro_text=intro_text,
                target_duration=target_duration,
            )
            results.append(result)

    return results


# ============================================================================
# RESOURCES - User/app selects these for data access
# ============================================================================

@gr.mcp.resource("resource://supported_formats")
def get_supported_formats() -> list[str]:
    """
    List of video formats Director's Cut can process.

    Returns:
        List of supported file extensions
    """
    return ["mp4", "mov", "avi", "mkv", "webm", "flv"]


@gr.mcp.resource("resource://intro_templates")
def get_intro_templates_resource() -> dict:
    """
    Available intro templates with style descriptions and preview URLs.

    Returns:
        Dictionary of template configurations with descriptions and preview paths
    """
    return get_intro_templates()


@gr.mcp.resource("resource://viral_detection_criteria")
def get_viral_criteria() -> dict:
    """
    Metrics and weights used for detecting engaging video segments.

    Returns:
        Dictionary with metric names, weights, and descriptions
    """
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


@gr.mcp.resource("resource://processing_status/{job_id}")
def get_processing_status(job_id: str) -> dict:
    """
    Real-time status of video processing job.

    Args:
        job_id: The job ID returned when starting processing

    Returns:
        Dictionary with status, progress percentage, current step, and ETA
    """
    if job_id in PROCESSING_JOBS:
        job = PROCESSING_JOBS[job_id]
        return {
            "status": job.get("status", "unknown"),
            "progress": job.get("progress", 0),
            "current_step": job.get("status", "unknown"),
            "eta_seconds": max(0, (100 - job.get("progress", 0)) * 2),
        }

    return {
        "status": "not_found",
        "progress": 0,
        "current_step": "unknown",
        "eta_seconds": 0,
    }


@gr.mcp.resource("resource://output_specifications")
def get_output_specs() -> dict:
    """
    Default output specifications for different social media platforms.

    Returns:
        Dictionary of platform-specific video specifications
    """
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
# PROMPTS - User explicitly invokes these for LLM guidance
# ============================================================================

@gr.mcp.prompt()
def optimize_for_platform(
    platform: Literal["tiktok", "instagram_reels", "youtube_shorts"] = "tiktok",
) -> str:
    """
    Get optimization prompt for specific social media platform.

    Provides platform-specific guidelines for video optimization.

    Args:
        platform: Target platform - "tiktok", "instagram_reels", or "youtube_shorts"

    Returns:
        Optimization prompt with platform-specific tips
    """
    specs = get_output_specs()
    platform_spec = specs.get(platform, specs["tiktok"])

    platform_tips = {
        "tiktok": "Hook viewers in first 1-3 seconds. Use trending sounds. Add captions.",
        "instagram_reels": "Use hashtags strategically. Post during peak hours. Engage in comments.",
        "youtube_shorts": "Use descriptive titles. Add end screens. Optimize for search.",
    }

    return f"""Optimize this video for {platform} by:
1. Ensuring aspect ratio is {platform_spec['aspect_ratio']}
2. Duration under {platform_spec['max_duration']} seconds
3. Resolution of {platform_spec['resolution']}
4. {platform_tips.get(platform, '')}
5. Export as {platform_spec['format']} format"""


@gr.mcp.prompt()
def create_viral_compilation(
    topic: Literal["fails", "wins", "tutorials", "reactions"] = "fails",
) -> str:
    """
    Prompt for creating viral compilation videos.

    Provides structured guidance for creating engaging compilation content.

    Args:
        topic: Compilation topic - "fails", "wins", "tutorials", or "reactions"

    Returns:
        Structured prompt for compilation creation
    """
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
6. Keep total duration under 60 seconds for maximum engagement"""


@gr.mcp.prompt()
def tutorial_video_structure(
    subject: str = "product demo",
) -> str:
    """
    Structured prompt for creating tutorial videos.

    Provides a framework for creating effective tutorial content.

    Args:
        subject: Tutorial subject (e.g., "product demo", "cooking recipe", "tech review")

    Returns:
        Structured prompt for tutorial creation
    """
    return f"""Create a tutorial about {subject}:
1. Hook in first 3 seconds - show the end result or tease the value
2. Brief intro (5 seconds max) - what viewers will learn
3. Step-by-step with on-screen numbers (Steps 1, 2, 3...)
4. Zoom on important details - use 2-3x zoom for clarity
5. Add text overlays for key points
6. Summary at end - recap main steps
7. Call-to-action - follow, like, or try it yourself

Tips:
- Keep each step under 10 seconds
- Use clear, simple language
- Show, don't just tell
- Add subtle background music"""


# ============================================================================
# Gradio UI (Optional - for web interface)
# ============================================================================

def create_ui():
    """Create Gradio web interface."""

    with gr.Blocks(title="Director's Cut") as demo:
        gr.Markdown("# Director's Cut - Video Editor")
        gr.Markdown("Transform YouTube videos into viral vertical content")

        with gr.Tab("Quick Process"):
            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
            with gr.Row():
                intro_text = gr.Textbox(
                    label="Intro Text",
                    value="Check this out!",
                )
                duration = gr.Slider(
                    label="Target Duration (seconds)",
                    minimum=15,
                    maximum=60,
                    value=30,
                    step=5,
                )

            process_btn = gr.Button("Process Video", variant="primary")
            output = gr.Textbox(label="Result")

            process_btn.click(
                fn=directors_cut_main,
                inputs=[url_input, intro_text, duration],
                outputs=output,
            )

        with gr.Tab("Analyze"):
            analyze_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            segment_len = gr.Slider(
                label="Segment Length",
                minimum=10,
                maximum=60,
                value=30,
                step=5,
            )
            analyze_btn = gr.Button("Analyze Video")
            analyze_output = gr.JSON(label="Analysis Results")

            analyze_btn.click(
                fn=analyze_video_segments_tool,
                inputs=[analyze_url, segment_len],
                outputs=analyze_output,
            )

        with gr.Tab("Generate Intro"):
            intro_text_gen = gr.Textbox(
                label="Intro Text",
                value="Check this out!",
            )
            intro_duration = gr.Slider(
                label="Duration",
                minimum=2,
                maximum=5,
                value=3,
                step=1,
            )
            intro_style = gr.Dropdown(
                label="Style",
                choices=["modern", "energetic", "professional", "minimal"],
                value="modern",
            )
            intro_btn = gr.Button("Generate Intro")
            intro_output = gr.Textbox(label="Intro Path")

            intro_btn.click(
                fn=generate_custom_intro,
                inputs=[intro_text_gen, intro_duration, intro_style],
                outputs=intro_output,
            )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()

    # Launch with MCP server enabled
    demo.launch(
        mcp_server=True,
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
