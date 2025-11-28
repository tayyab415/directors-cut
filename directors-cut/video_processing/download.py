"""
YouTube Download Utilities for Director's Cut

This module provides functions for downloading videos from YouTube
using yt-dlp.
"""

import os
import subprocess
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video."""
    title: str
    description: str
    uploader: str
    duration: float
    view_count: int
    like_count: Optional[int]
    upload_date: str
    tags: list[str]
    thumbnail_url: str
    video_id: str


def get_video_metadata(url: str) -> Optional[VideoMetadata]:
    """
    Get metadata for a YouTube video without downloading.

    Args:
        url: YouTube video URL

    Returns:
        VideoMetadata object with video information, or None if failed
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        return VideoMetadata(
            title=data.get("title", "Unknown"),
            description=data.get("description", ""),
            uploader=data.get("uploader", "Unknown"),
            duration=float(data.get("duration", 0)),
            view_count=int(data.get("view_count", 0)),
            like_count=data.get("like_count"),
            upload_date=data.get("upload_date", ""),
            tags=data.get("tags", []),
            thumbnail_url=data.get("thumbnail", ""),
            video_id=data.get("id", ""),
        )

    except Exception:
        return None


def download_youtube_video(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
    format: str = "mp4",
    max_height: int = 1080,
    include_audio: bool = True,
) -> tuple[bool, str]:
    """
    Download a video from YouTube.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded video
        filename: Custom filename (without extension), uses video title if None
        format: Output format (mp4, webm, mkv)
        max_height: Maximum video height (1080, 720, 480, etc.)
        include_audio: Whether to include audio

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Build format selector
    if include_audio:
        format_selector = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    else:
        format_selector = f"bestvideo[height<={max_height}]/best[height<={max_height}]"

    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--merge-output-format", format,
        "-o", output_template,
        "--no-playlist",
        "--restrict-filenames",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return False, f"Download failed: {result.stderr}"

        # Find the downloaded file
        for line in result.stdout.split("\n"):
            if "Destination:" in line or "has already been downloaded" in line:
                # Extract path from output
                pass

        # Search for the downloaded file in output_dir
        for f in os.listdir(output_dir):
            if f.endswith(f".{format}"):
                return True, os.path.join(output_dir, f)

        # Try to find any video file
        for f in os.listdir(output_dir):
            if f.endswith((".mp4", ".webm", ".mkv")):
                return True, os.path.join(output_dir, f)

        return False, "Download completed but file not found"

    except subprocess.TimeoutExpired:
        return False, "Download timed out"
    except Exception as e:
        return False, str(e)


def download_youtube_segment(
    url: str,
    output_dir: str,
    start_time: float,
    duration: float,
    filename: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Download a specific segment of a YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded video
        start_time: Start time in seconds
        duration: Duration in seconds
        filename: Custom filename (without extension)

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title)s_segment.%(ext)s")

    # yt-dlp supports --download-sections for segment download
    section_spec = f"*{start_time}-{start_time + duration}"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--restrict-filenames",
        "--download-sections", section_spec,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return False, f"Download failed: {result.stderr}"

        # Find the downloaded file
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                return True, os.path.join(output_dir, f)

        return False, "Download completed but file not found"

    except subprocess.TimeoutExpired:
        return False, "Download timed out"
    except Exception as e:
        return False, str(e)


def download_playlist(
    playlist_url: str,
    output_dir: str,
    max_videos: Optional[int] = None,
    format: str = "mp4",
) -> tuple[bool, list[str]]:
    """
    Download all videos from a YouTube playlist.

    Args:
        playlist_url: YouTube playlist URL
        output_dir: Directory to save downloaded videos
        max_videos: Maximum number of videos to download (None for all)
        format: Output format

    Returns:
        Tuple of (success: bool, list of downloaded file paths)
    """
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(playlist_index)s_%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", format,
        "-o", output_template,
        "--restrict-filenames",
        playlist_url,
    ]

    if max_videos:
        cmd.extend(["--playlist-end", str(max_videos)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            return False, []

        # Find all downloaded files
        downloaded = []
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(f".{format}"):
                downloaded.append(os.path.join(output_dir, f))

        return True, downloaded

    except subprocess.TimeoutExpired:
        return False, []
    except Exception:
        return False, []


def get_playlist_info(playlist_url: str) -> tuple[bool, list[dict]]:
    """
    Get information about all videos in a playlist.

    Args:
        playlist_url: YouTube playlist URL

    Returns:
        Tuple of (success: bool, list of video info dicts)
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        playlist_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return False, []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "url": data.get("url") or f"https://www.youtube.com/watch?v={data.get('id')}",
                        "duration": data.get("duration"),
                    })
                except json.JSONDecodeError:
                    continue

        return True, videos

    except Exception:
        return False, []
