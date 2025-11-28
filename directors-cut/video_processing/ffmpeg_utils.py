"""
FFmpeg Utility Functions for Director's Cut

This module provides wrapper functions for FFmpeg operations.
Based on patterns from video-audio-mcp reference repository.
"""

import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata information."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: Optional[str]
    has_audio: bool
    audio_codec: Optional[str]
    file_size_mb: float


def check_ffmpeg_installed() -> bool:
    """Check if FFmpeg is installed and accessible."""
    return shutil.which("ffmpeg") is not None


def run_ffmpeg(
    args: list[str],
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    overwrite: bool = True,
    timeout: int = 600,
) -> tuple[bool, str]:
    """
    Run an FFmpeg command with the given arguments.

    Args:
        args: List of FFmpeg arguments (excluding ffmpeg binary, input, and output)
        input_file: Path to input file (optional, can be included in args)
        output_file: Path to output file (optional, can be included in args)
        overwrite: Whether to overwrite existing output files
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    cmd = ["ffmpeg"]

    if overwrite:
        cmd.append("-y")

    if input_file:
        cmd.extend(["-i", input_file])

    cmd.extend(args)

    if output_file:
        cmd.append(output_file)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"

        if output_file and os.path.exists(output_file):
            return True, f"Success: {output_file}"
        elif output_file:
            return False, "Output file was not created"

        return True, result.stdout or "Success"

    except subprocess.TimeoutExpired:
        return False, f"FFmpeg command timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error running FFmpeg: {str(e)}"


def get_video_info(video_path: str) -> Optional[VideoInfo]:
    """
    Get detailed metadata for a video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        VideoInfo object with video metadata, or None if failed
    """
    if not os.path.exists(video_path):
        return None

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream

        if not video_stream:
            return None

        # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den != 0 else 30.0
        except (ValueError, ZeroDivisionError):
            fps = 30.0

        format_info = data.get("format", {})
        file_size = int(format_info.get("size", 0))

        return VideoInfo(
            duration=float(format_info.get("duration", 0)),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=round(fps, 2),
            codec=video_stream.get("codec_name", "unknown"),
            bitrate=format_info.get("bit_rate"),
            has_audio=audio_stream is not None,
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            file_size_mb=round(file_size / (1024 * 1024), 2),
        )

    except Exception:
        return None


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    format: str = "jpg",
    quality: int = 2,
) -> tuple[bool, list[str]]:
    """
    Extract frames from a video at specified FPS.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1 frame per second)
        format: Output image format (jpg, png)
        quality: JPEG quality (1-31, lower is better, only for jpg)

    Returns:
        Tuple of (success: bool, list of frame paths)
    """
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, f"frame_%04d.{format}")

    args = [
        "-vf", f"fps={fps}",
    ]

    if format == "jpg":
        args.extend(["-qscale:v", str(quality)])

    success, message = run_ffmpeg(
        args,
        input_file=video_path,
        output_file=output_pattern,
    )

    if not success:
        return False, []

    # Get list of extracted frames
    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(f".{format}")
    ])

    return True, frames


def trim_video(
    video_path: str,
    output_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    duration: Optional[float] = None,
    copy_codec: bool = True,
) -> tuple[bool, str]:
    """
    Trim a video to specified time range.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (optional)
        duration: Duration in seconds (alternative to end_time)
        copy_codec: If True, use stream copy for speed (no re-encoding)

    Returns:
        Tuple of (success: bool, message: str)
    """
    args = []

    if start_time is not None:
        args.extend(["-ss", str(start_time)])

    args.extend(["-i", video_path])

    if end_time is not None:
        args.extend(["-to", str(end_time)])
    elif duration is not None:
        args.extend(["-t", str(duration)])

    if copy_codec:
        args.extend(["-c", "copy"])

    # Remove -i from args since we'll add it via input_file
    # Actually, we need to handle -ss before -i for seeking
    cmd = ["ffmpeg", "-y"]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", video_path])

    if end_time is not None:
        cmd.extend(["-to", str(end_time - (start_time or 0))])
    elif duration is not None:
        cmd.extend(["-t", str(duration)])

    if copy_codec:
        cmd.extend(["-c", "copy"])

    cmd.append(output_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"
        if os.path.exists(output_path):
            return True, output_path
        return False, "Output file not created"
    except Exception as e:
        return False, str(e)


def crop_video(
    video_path: str,
    output_path: str,
    width: int,
    height: int,
    x: int = 0,
    y: int = 0,
) -> tuple[bool, str]:
    """
    Crop a video to specified dimensions.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        width: Output width in pixels
        height: Output height in pixels
        x: X offset for crop (from left)
        y: Y offset for crop (from top)

    Returns:
        Tuple of (success: bool, message: str)
    """
    crop_filter = f"crop={width}:{height}:{x}:{y}"

    return run_ffmpeg(
        ["-vf", crop_filter, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def scale_video(
    video_path: str,
    output_path: str,
    width: int,
    height: int,
    maintain_aspect: bool = True,
) -> tuple[bool, str]:
    """
    Scale a video to specified dimensions.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        width: Target width (-1 to maintain aspect ratio)
        height: Target height (-1 to maintain aspect ratio)
        maintain_aspect: If True, use -1 for one dimension to maintain aspect ratio

    Returns:
        Tuple of (success: bool, message: str)
    """
    if maintain_aspect:
        # Scale to fit within dimensions while maintaining aspect ratio
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
    else:
        scale_filter = f"scale={width}:{height}"

    return run_ffmpeg(
        ["-vf", scale_filter, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def apply_video_filter(
    video_path: str,
    output_path: str,
    filter_name: str,
) -> tuple[bool, str]:
    """
    Apply a visual filter to a video.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        filter_name: Name of filter to apply
            - "sepia": Vintage warm tone
            - "grayscale": Black and white
            - "vintage": Faded vintage look
            - "vibrant": Enhanced saturation
            - "blur": Gaussian blur
            - "sharpen": Sharpen edges

    Returns:
        Tuple of (success: bool, message: str)
    """
    filter_map = {
        "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
        "grayscale": "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3",
        "vintage": "curves=vintage",
        "vibrant": "eq=saturation=1.5",
        "blur": "gblur=sigma=5",
        "sharpen": "unsharp=5:5:1.0:5:5:0.0",
    }

    if filter_name not in filter_map:
        return False, f"Unknown filter: {filter_name}. Available: {list(filter_map.keys())}"

    return run_ffmpeg(
        ["-vf", filter_map[filter_name], "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def adjust_brightness(
    video_path: str,
    output_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> tuple[bool, str]:
    """
    Adjust video brightness, contrast, and saturation.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        brightness: Brightness level (1.0 = original, 1.5 = 50% brighter)
        contrast: Contrast level (1.0 = original)
        saturation: Saturation level (1.0 = original)

    Returns:
        Tuple of (success: bool, message: str)
    """
    eq_filter = f"eq=brightness={brightness-1}:contrast={contrast}:saturation={saturation}"

    return run_ffmpeg(
        ["-vf", eq_filter, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def change_speed(
    video_path: str,
    output_path: str,
    speed_factor: float,
) -> tuple[bool, str]:
    """
    Change video playback speed.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        speed_factor: Speed multiplier (0.5 = half speed, 2.0 = double speed)

    Returns:
        Tuple of (success: bool, message: str)
    """
    if speed_factor <= 0:
        return False, "Speed factor must be positive"

    # setpts for video, atempo for audio
    video_filter = f"setpts={1/speed_factor}*PTS"

    # atempo only accepts values between 0.5 and 2.0, chain for extreme values
    audio_filters = []
    remaining = speed_factor

    while remaining > 2.0:
        audio_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        audio_filters.append("atempo=0.5")
        remaining /= 0.5

    audio_filters.append(f"atempo={remaining}")
    audio_filter = ",".join(audio_filters)

    return run_ffmpeg(
        ["-vf", video_filter, "-af", audio_filter],
        input_file=video_path,
        output_file=output_path,
    )


def concatenate_videos(
    video_paths: list[str],
    output_path: str,
    transition: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Concatenate multiple videos into one.

    Args:
        video_paths: List of paths to videos to concatenate
        output_path: Path for output video
        transition: Optional transition type (currently not implemented)

    Returns:
        Tuple of (success: bool, message: str)
    """
    if len(video_paths) < 2:
        return False, "Need at least 2 videos to concatenate"

    # Create concat file
    concat_file = output_path + ".concat.txt"

    try:
        with open(concat_file, "w") as f:
            for path in video_paths:
                # Escape special characters in path
                escaped_path = path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        success, message = run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy"],
            output_file=output_path,
        )

        # Clean up concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)

        return success, message

    except Exception as e:
        if os.path.exists(concat_file):
            os.remove(concat_file)
        return False, str(e)


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_volume: float = 1.0,
    mix_with_original: bool = True,
    original_volume: float = 1.0,
) -> tuple[bool, str]:
    """
    Add or mix audio with a video.

    Args:
        video_path: Path to input video
        audio_path: Path to audio file to add
        output_path: Path for output video
        audio_volume: Volume of added audio (0.0 to 1.0)
        mix_with_original: If True, mix with original audio; if False, replace
        original_volume: Volume of original audio when mixing

    Returns:
        Tuple of (success: bool, message: str)
    """
    if mix_with_original:
        # Mix both audio tracks
        filter_complex = (
            f"[0:a]volume={original_volume}[a0];"
            f"[1:a]volume={audio_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first[aout]"
        )
        args = [
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
        ]
    else:
        # Replace audio
        args = [
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-shortest",
        ]

    cmd = ["ffmpeg", "-y"] + args + [output_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"
        if os.path.exists(output_path):
            return True, output_path
        return False, "Output file not created"
    except Exception as e:
        return False, str(e)


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
) -> tuple[bool, str]:
    """
    Burn subtitles into a video (hardcode).

    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video
        font_size: Subtitle font size
        font_color: Subtitle text color
        outline_color: Subtitle outline color

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Escape the SRT path for FFmpeg filter
    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")

    subtitle_filter = (
        f"subtitles='{escaped_srt}':"
        f"force_style='FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2'"
    )

    return run_ffmpeg(
        ["-vf", subtitle_filter],
        input_file=video_path,
        output_file=output_path,
    )


def extract_audio(
    video_path: str,
    output_path: str,
    format: str = "mp3",
    bitrate: str = "192k",
) -> tuple[bool, str]:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to input video
        output_path: Path for output audio
        format: Output audio format (mp3, wav, aac)
        bitrate: Audio bitrate

    Returns:
        Tuple of (success: bool, message: str)
    """
    codec_map = {
        "mp3": "libmp3lame",
        "wav": "pcm_s16le",
        "aac": "aac",
    }

    codec = codec_map.get(format, "libmp3lame")

    return run_ffmpeg(
        ["-vn", "-acodec", codec, "-ab", bitrate],
        input_file=video_path,
        output_file=output_path,
    )


def create_video_from_image(
    image_path: str,
    output_path: str,
    duration: float,
    fps: int = 30,
) -> tuple[bool, str]:
    """
    Create a video from a static image.

    Args:
        image_path: Path to input image
        output_path: Path for output video
        duration: Video duration in seconds
        fps: Frames per second

    Returns:
        Tuple of (success: bool, message: str)
    """
    return run_ffmpeg(
        [
            "-loop", "1",
            "-i", image_path,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
        ],
        output_file=output_path,
    )


def add_text_overlay(
    video_path: str,
    output_path: str,
    text: str,
    x: str = "(w-text_w)/2",
    y: str = "(h-text_h)/2",
    font_size: int = 48,
    font_color: str = "white",
    start_time: float = 0,
    end_time: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Add text overlay to a video.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        text: Text to display
        x: X position expression (default: center)
        y: Y position expression (default: center)
        font_size: Font size
        font_color: Text color
        start_time: When to show text (seconds)
        end_time: When to hide text (seconds, None for entire video)

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Escape special characters in text
    escaped_text = text.replace("'", "'\\''").replace(":", "\\:")

    drawtext_filter = (
        f"drawtext=text='{escaped_text}':"
        f"fontsize={font_size}:"
        f"fontcolor={font_color}:"
        f"x={x}:y={y}"
    )

    if end_time is not None:
        drawtext_filter += f":enable='between(t,{start_time},{end_time})'"
    elif start_time > 0:
        drawtext_filter += f":enable='gte(t,{start_time})'"

    return run_ffmpeg(
        ["-vf", drawtext_filter, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )
