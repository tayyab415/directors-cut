"""
Video Format and Aspect Ratio Conversion for Director's Cut

This module provides functions for converting video formats and aspect ratios,
including smart cropping with subject detection.
"""

import os
import subprocess
from typing import Optional, Literal
from dataclasses import dataclass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .ffmpeg_utils import get_video_info, run_ffmpeg


@dataclass
class CropRegion:
    """Region to crop from the original video."""
    x: int
    y: int
    width: int
    height: int


def calculate_crop_for_aspect(
    src_width: int,
    src_height: int,
    target_aspect: str,
    position: str = "center",
) -> CropRegion:
    """
    Calculate crop region to achieve target aspect ratio.

    Args:
        src_width: Source video width
        src_height: Source video height
        target_aspect: Target aspect ratio (e.g., "9:16", "16:9", "1:1")
        position: Crop position ("center", "left", "right", "top", "bottom")

    Returns:
        CropRegion with calculated crop parameters
    """
    # Parse aspect ratio
    aspect_parts = target_aspect.split(":")
    target_w_ratio = int(aspect_parts[0])
    target_h_ratio = int(aspect_parts[1])
    target_ratio = target_w_ratio / target_h_ratio

    current_ratio = src_width / src_height

    if current_ratio > target_ratio:
        # Video is wider than target - crop width
        new_width = int(src_height * target_ratio)
        new_height = src_height

        if position == "left":
            x = 0
        elif position == "right":
            x = src_width - new_width
        else:  # center
            x = (src_width - new_width) // 2

        y = 0
    else:
        # Video is taller than target - crop height
        new_width = src_width
        new_height = int(src_width / target_ratio)

        x = 0

        if position == "top":
            y = 0
        elif position == "bottom":
            y = src_height - new_height
        else:  # center
            y = (src_height - new_height) // 2

    return CropRegion(x=x, y=y, width=new_width, height=new_height)


def detect_subject_position(video_path: str, sample_frames: int = 10) -> str:
    """
    Detect the primary subject position in the video using face detection.

    Args:
        video_path: Path to the video file
        sample_frames: Number of frames to sample for detection

    Returns:
        Position string: "left", "center", or "right"
    """
    if not HAS_CV2:
        return "center"

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "center"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if total_frames == 0:
            return "center"

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        positions = {"left": 0, "center": 0, "right": 0}
        step = max(1, total_frames // sample_frames)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_center = x + w // 2
                relative_pos = face_center / frame_width

                if relative_pos < 0.35:
                    positions["left"] += 1
                elif relative_pos > 0.65:
                    positions["right"] += 1
                else:
                    positions["center"] += 1

        cap.release()

        # Return the position with most detections
        if max(positions.values()) == 0:
            return "center"

        return max(positions, key=positions.get)

    except Exception:
        return "center"


def smart_crop(
    video_path: str,
    output_path: str,
    target_aspect: str = "9:16",
) -> tuple[bool, str]:
    """
    Smart crop video with automatic subject detection.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        target_aspect: Target aspect ratio

    Returns:
        Tuple of (success: bool, message: str)
    """
    info = get_video_info(video_path)
    if not info:
        return False, "Could not get video info"

    # Detect subject position
    position = detect_subject_position(video_path)

    # Calculate crop region
    crop = calculate_crop_for_aspect(
        info.width, info.height, target_aspect, position
    )

    # Apply crop with FFmpeg
    crop_filter = f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}"

    return run_ffmpeg(
        ["-vf", crop_filter, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def convert_to_vertical(
    video_path: str,
    output_path: str,
    crop_mode: Literal["smart", "center", "left", "right"] = "smart",
    output_resolution: tuple[int, int] = (1080, 1920),
) -> tuple[bool, str]:
    """
    Convert landscape video to 9:16 vertical format for social media.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        crop_mode: Cropping mode
            - "smart": AI detects subject position
            - "center": Center crop
            - "left": Left-aligned crop
            - "right": Right-aligned crop
        output_resolution: Target resolution (width, height)

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    info = get_video_info(video_path)
    if not info:
        return False, "Could not get video info"

    # Determine crop position
    if crop_mode == "smart":
        position = detect_subject_position(video_path)
    else:
        position = crop_mode

    # Calculate crop for 9:16 aspect ratio
    crop = calculate_crop_for_aspect(info.width, info.height, "9:16", position)

    # Build filter chain: crop then scale
    target_width, target_height = output_resolution

    filter_chain = (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        f"scale={target_width}:{target_height}"
    )

    success, message = run_ffmpeg(
        ["-vf", filter_chain, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )

    if success:
        return True, output_path
    return False, message


def change_aspect_ratio(
    video_path: str,
    output_path: str,
    aspect_ratio: Literal["9:16", "16:9", "1:1", "4:5", "4:3"] = "9:16",
    crop_position: Literal["center", "top", "bottom", "left", "right", "smart"] = "center",
    scale_to_fit: bool = True,
    target_resolution: Optional[tuple[int, int]] = None,
) -> tuple[bool, str]:
    """
    Change video aspect ratio with cropping.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        aspect_ratio: Target aspect ratio
        crop_position: How to crop the video
        scale_to_fit: Whether to scale to common resolution after cropping
        target_resolution: Custom target resolution (overrides scale_to_fit defaults)

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    info = get_video_info(video_path)
    if not info:
        return False, "Could not get video info"

    # Determine crop position
    if crop_position == "smart":
        position = detect_subject_position(video_path)
    else:
        position = crop_position

    # Calculate crop
    crop = calculate_crop_for_aspect(info.width, info.height, aspect_ratio, position)

    # Determine target resolution
    if target_resolution:
        target_width, target_height = target_resolution
    elif scale_to_fit:
        # Common resolutions for each aspect ratio
        resolution_map = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
            "4:5": (1080, 1350),
            "4:3": (1440, 1080),
        }
        target_width, target_height = resolution_map.get(aspect_ratio, (1080, 1920))
    else:
        target_width, target_height = crop.width, crop.height

    # Build filter chain
    filter_chain = (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        f"scale={target_width}:{target_height}"
    )

    success, message = run_ffmpeg(
        ["-vf", filter_chain, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )

    if success:
        return True, output_path
    return False, message


def add_letterbox(
    video_path: str,
    output_path: str,
    target_aspect: str = "9:16",
    background_color: str = "black",
    blur_background: bool = False,
) -> tuple[bool, str]:
    """
    Add letterboxing/pillarboxing to fit video in target aspect ratio without cropping.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        target_aspect: Target aspect ratio
        background_color: Background color for letterbox
        blur_background: If True, use blurred video as background instead of solid color

    Returns:
        Tuple of (success: bool, message: str)
    """
    info = get_video_info(video_path)
    if not info:
        return False, "Could not get video info"

    # Parse target aspect
    aspect_parts = target_aspect.split(":")
    target_w_ratio = int(aspect_parts[0])
    target_h_ratio = int(aspect_parts[1])

    # Calculate target dimensions that contain the original video
    if info.width / info.height > target_w_ratio / target_h_ratio:
        # Video is wider - use full width, add vertical padding
        output_width = info.width
        output_height = int(info.width * target_h_ratio / target_w_ratio)
    else:
        # Video is taller - use full height, add horizontal padding
        output_height = info.height
        output_width = int(info.height * target_w_ratio / target_h_ratio)

    if blur_background:
        # Create blurred, scaled background and overlay original
        filter_chain = (
            f"split[bg][fg];"
            f"[bg]scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
            f"crop={output_width}:{output_height},gblur=sigma=30[bg_out];"
            f"[fg]scale={output_width}:{output_height}:force_original_aspect_ratio=decrease[fg_out];"
            f"[bg_out][fg_out]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        # Simple padding with solid color
        filter_chain = (
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color={background_color}"
        )

    return run_ffmpeg(
        ["-vf", filter_chain, "-c:a", "copy"],
        input_file=video_path,
        output_file=output_path,
    )


def convert_format(
    video_path: str,
    output_path: str,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
) -> tuple[bool, str]:
    """
    Convert video to a different format/codec.

    Args:
        video_path: Path to input video
        output_path: Path for output video (extension determines container)
        codec: Video codec (libx264, libx265, vp9, etc.)
        crf: Constant Rate Factor (quality, lower = better, 18-28 typical)
        preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)

    Returns:
        Tuple of (success: bool, message: str)
    """
    return run_ffmpeg(
        ["-c:v", codec, "-crf", str(crf), "-preset", preset, "-c:a", "aac"],
        input_file=video_path,
        output_file=output_path,
    )
