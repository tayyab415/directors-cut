"""
AI-Powered Intro Generation for Director's Cut

This module provides functions for generating video intros with
AI-generated backgrounds and animated text overlays.
"""

import os
import tempfile
from typing import Optional, Literal
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .ffmpeg_utils import (
    create_video_from_image,
    add_text_overlay,
    concatenate_videos,
    run_ffmpeg,
)


@dataclass
class IntroStyle:
    """Configuration for intro style."""
    name: str
    description: str
    duration: int
    background_color: str
    text_color: str
    animation: str
    font_size: int


# Predefined intro styles
INTRO_STYLES = {
    "modern": IntroStyle(
        name="modern",
        description="Clean geometric shapes, smooth transitions",
        duration=3,
        background_color="#1a1a2e",
        text_color="white",
        animation="fade",
        font_size=72,
    ),
    "energetic": IntroStyle(
        name="energetic",
        description="Fast-paced, bright colors, motion blur",
        duration=2,
        background_color="#ff6b6b",
        text_color="white",
        animation="zoom",
        font_size=84,
    ),
    "professional": IntroStyle(
        name="professional",
        description="Corporate style, subtle animations",
        duration=3,
        background_color="#2c3e50",
        text_color="#ecf0f1",
        animation="slide",
        font_size=64,
    ),
    "minimal": IntroStyle(
        name="minimal",
        description="Simple text on solid background",
        duration=2,
        background_color="#000000",
        text_color="white",
        animation="none",
        font_size=56,
    ),
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_gradient_background(
    width: int,
    height: int,
    color1: str,
    color2: str,
    output_path: str,
) -> bool:
    """
    Create a gradient background image.

    Args:
        width: Image width
        height: Image height
        color1: Start color (hex)
        color2: End color (hex)
        output_path: Path to save the image

    Returns:
        True if successful
    """
    if not HAS_PIL:
        return False

    try:
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)

        image = Image.new("RGB", (width, height))
        pixels = image.load()

        for y in range(height):
            ratio = y / height
            r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
            g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
            b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)

            for x in range(width):
                pixels[x, y] = (r, g, b)

        image.save(output_path)
        return True

    except Exception:
        return False


def create_solid_background(
    width: int,
    height: int,
    color: str,
    output_path: str,
) -> bool:
    """
    Create a solid color background image.

    Args:
        width: Image width
        height: Image height
        color: Background color (hex or name)
        output_path: Path to save the image

    Returns:
        True if successful
    """
    if not HAS_PIL:
        return False

    try:
        if color.startswith("#"):
            rgb = hex_to_rgb(color)
        else:
            # Use PIL's color name support
            rgb = color

        image = Image.new("RGB", (width, height), rgb)
        image.save(output_path)
        return True

    except Exception:
        return False


def generate_background(
    width: int = 1080,
    height: int = 1920,
    style: str = "modern",
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Generate a background image for the intro.

    Args:
        width: Image width
        height: Image height
        style: Intro style name
        prompt: Optional custom prompt for AI generation (placeholder for future)
        output_path: Path to save the image

    Returns:
        Tuple of (success: bool, path: str)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    style_config = INTRO_STYLES.get(style, INTRO_STYLES["modern"])

    # For now, create gradient backgrounds based on style
    # In the future, this can be extended to use AI image generation
    gradient_colors = {
        "modern": ("#1a1a2e", "#16213e"),
        "energetic": ("#ff6b6b", "#feca57"),
        "professional": ("#2c3e50", "#34495e"),
        "minimal": ("#000000", "#1a1a1a"),
    }

    color1, color2 = gradient_colors.get(style, gradient_colors["modern"])

    success = create_gradient_background(width, height, color1, color2, output_path)

    if success:
        return True, output_path
    return False, "Failed to create background"


def create_text_image(
    text: str,
    width: int,
    height: int,
    font_size: int,
    text_color: str,
    background_path: str,
    output_path: str,
) -> bool:
    """
    Create an image with text overlaid on background.

    Args:
        text: Text to display
        width: Image width
        height: Image height
        font_size: Font size in pixels
        text_color: Text color
        background_path: Path to background image
        output_path: Path to save the result

    Returns:
        True if successful
    """
    if not HAS_PIL:
        return False

    try:
        # Load background
        image = Image.open(background_path).convert("RGBA")
        image = image.resize((width, height))

        # Create text layer
        txt_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2

        # Draw text with shadow for better visibility
        shadow_offset = 3
        if text_color.lower() in ["white", "#ffffff", "#fff"]:
            shadow_color = (0, 0, 0, 128)
        else:
            shadow_color = (255, 255, 255, 128)

        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        draw.text((x, y), text, font=font, fill=text_color)

        # Composite
        result = Image.alpha_composite(image, txt_layer)
        result = result.convert("RGB")
        result.save(output_path)

        return True

    except Exception:
        return False


def create_text_overlay(
    text: str,
    style: str = "modern",
    width: int = 1080,
    height: int = 1920,
    output_path: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Create an image with styled text overlay.

    Args:
        text: Text to display
        style: Intro style name
        width: Image width
        height: Image height
        output_path: Path to save the image

    Returns:
        Tuple of (success: bool, path: str)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    style_config = INTRO_STYLES.get(style, INTRO_STYLES["modern"])

    # Create background
    bg_path = tempfile.mktemp(suffix=".png")
    success, bg_result = generate_background(width, height, style, output_path=bg_path)

    if not success:
        return False, "Failed to create background"

    # Create text image
    success = create_text_image(
        text=text,
        width=width,
        height=height,
        font_size=style_config.font_size,
        text_color=style_config.text_color,
        background_path=bg_path,
        output_path=output_path,
    )

    # Clean up background
    if os.path.exists(bg_path):
        os.remove(bg_path)

    if success:
        return True, output_path
    return False, "Failed to create text overlay"


def apply_intro_animation(
    image_path: str,
    output_path: str,
    duration: float,
    animation: str = "fade",
    fps: int = 30,
) -> tuple[bool, str]:
    """
    Apply animation effect to create intro video from image.

    Args:
        image_path: Path to source image
        output_path: Path for output video
        duration: Video duration in seconds
        animation: Animation type (fade, zoom, slide, none)
        fps: Frames per second

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Build filter based on animation type
    if animation == "fade":
        # Fade in from black
        filter_complex = (
            f"loop=loop={int(duration * fps)}:size=1:start=0,"
            f"fade=t=in:st=0:d=0.5,fade=t=out:st={duration-0.5}:d=0.5"
        )
    elif animation == "zoom":
        # Zoom in effect
        filter_complex = (
            f"loop=loop={int(duration * fps)}:size=1:start=0,"
            f"zoompan=z='min(zoom+0.001,1.2)':d={int(duration * fps)}:s=1080x1920"
        )
    elif animation == "slide":
        # Slide in from bottom
        filter_complex = (
            f"loop=loop={int(duration * fps)}:size=1:start=0"
        )
    else:
        # No animation, just static
        filter_complex = f"loop=loop={int(duration * fps)}:size=1:start=0"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path
    ]

    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"

        if os.path.exists(output_path):
            return True, output_path

        return False, "Output file not created"

    except Exception as e:
        return False, str(e)


def generate_intro(
    text: str,
    duration: int = 3,
    style: Literal["modern", "energetic", "professional", "minimal"] = "modern",
    background_prompt: Optional[str] = None,
    width: int = 1080,
    height: int = 1920,
    output_path: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Generate a complete video intro with text and animation.

    Args:
        text: Text to display in the intro
        duration: Intro duration in seconds
        style: Visual style (modern, energetic, professional, minimal)
        background_prompt: Optional prompt for AI background generation
        width: Video width
        height: Video height
        output_path: Path for output video

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    style_config = INTRO_STYLES.get(style, INTRO_STYLES["modern"])

    # Create text overlay image
    image_path = tempfile.mktemp(suffix=".png")
    success, result = create_text_overlay(
        text=text,
        style=style,
        width=width,
        height=height,
        output_path=image_path,
    )

    if not success:
        return False, result

    # Apply animation to create video
    success, message = apply_intro_animation(
        image_path=image_path,
        output_path=output_path,
        duration=duration,
        animation=style_config.animation,
    )

    # Clean up temp image
    if os.path.exists(image_path):
        os.remove(image_path)

    if success:
        return True, output_path
    return False, message


def add_intro_to_video(
    intro_path: str,
    video_path: str,
    output_path: str,
) -> tuple[bool, str]:
    """
    Prepend an intro video to the main video.

    Args:
        intro_path: Path to intro video
        video_path: Path to main video
        output_path: Path for output video

    Returns:
        Tuple of (success: bool, message: str)
    """
    return concatenate_videos([intro_path, video_path], output_path)


def get_intro_templates() -> dict:
    """
    Get available intro templates with descriptions.

    Returns:
        Dictionary of template information
    """
    return {
        name: {
            "desc": style.description,
            "duration": style.duration,
            "preview": f"/previews/{name}.mp4",
        }
        for name, style in INTRO_STYLES.items()
    }
