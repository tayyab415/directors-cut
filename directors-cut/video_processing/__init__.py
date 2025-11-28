"""
Director's Cut - Video Processing Module

This module provides video processing utilities for the Director's Cut MCP servers.
"""

from .ffmpeg_utils import (
    run_ffmpeg,
    get_video_info,
    extract_frames,
    trim_video,
    crop_video,
    apply_video_filter,
    concatenate_videos,
    add_audio_to_video,
    burn_subtitles,
)

from .download import (
    download_youtube_video,
    get_video_metadata,
    download_playlist,
)

from .conversion import (
    convert_to_vertical,
    change_aspect_ratio,
    smart_crop,
)

from .viral_detection import (
    analyze_video_segments,
    score_segment,
    detect_faces,
    calculate_motion,
)

from .intro_generation import (
    generate_intro,
    create_text_overlay,
    generate_background,
)

from .subtitle_generation import (
    transcribe_audio,
    generate_srt,
    add_subtitles,
)

__all__ = [
    # FFmpeg utils
    "run_ffmpeg",
    "get_video_info",
    "extract_frames",
    "trim_video",
    "crop_video",
    "apply_video_filter",
    "concatenate_videos",
    "add_audio_to_video",
    "burn_subtitles",
    # Download
    "download_youtube_video",
    "get_video_metadata",
    "download_playlist",
    # Conversion
    "convert_to_vertical",
    "change_aspect_ratio",
    "smart_crop",
    # Viral detection
    "analyze_video_segments",
    "score_segment",
    "detect_faces",
    "calculate_motion",
    # Intro generation
    "generate_intro",
    "create_text_overlay",
    "generate_background",
    # Subtitle generation
    "transcribe_audio",
    "generate_srt",
    "add_subtitles",
]
