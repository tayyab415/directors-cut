"""
Utility functions for Director's Cut.
Includes video downloading logic integrated from crop-vid.
"""

import yt_dlp
import os
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Optional

from src.paths import ensure_runtime_dirs

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return the ID if already provided."""
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id
    try:
        parsed = urlparse(url_or_id)
        if parsed.hostname in ['youtu.be', 'www.youtu.be']:
            return parsed.path.lstrip('/')
        if 'youtube.com' in parsed.hostname or 'youtu.be' in parsed.hostname:
            if parsed.path == '/watch':
                query_params = parse_qs(parsed.query)
                if 'v' in query_params:
                    return query_params['v'][0]
            elif parsed.path.startswith('/embed/'):
                return parsed.path.split('/embed/')[1].split('?')[0]
            elif parsed.path.startswith('/v/'):
                return parsed.path.split('/v/')[1].split('?')[0]
    except Exception:
        pass
    return url_or_id

def ensure_directories():
    """Ensure that necessary directories exist."""
    ensure_runtime_dirs()

def download_video_segment(
    url: str,
    start_time: float,
    end_time: float,
    output_filename: str = 'clip',
    format_spec: str = 'best'
) -> str:
    """
    Download only a specific time range from a YouTube video.
    
    Args:
        url: YouTube video URL
        start_time: Start time in seconds (int or float)
        end_time: End time in seconds (int or float)
        output_filename: Output filename without extension (default: 'clip')
        format_spec: Format specification for yt-dlp (default: 'best')
    
    Returns:
        str: Path to the downloaded file
    """
    ydl_opts = {
        'format': format_spec,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'outtmpl': f'{output_filename}.%(ext)s',
        'force_keyframes_at_cuts': True,
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get('ext', 'mp4')
            output_path = f"{output_filename}.{ext}"
            print(f"✓ Downloaded segment: {output_path}")
            return output_path
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'proxy' in error_msg.lower() or '403' in error_msg or 'Forbidden' in error_msg:
            raise Exception(f"Network error: Cannot access YouTube. Your network may block YouTube access. Original error: {e}")
        elif 'unavailable' in error_msg.lower():
            raise Exception(f"Video unavailable: The video may be private, deleted, or region-locked. Original error: {e}")
        print(f"✗ Download failed: {e}")
        raise
    except OSError as e:
        if 'Tunnel connection failed' in str(e) or 'proxy' in str(e).lower():
            raise Exception(f"Network error: Cannot connect to YouTube through proxy. Error: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise

def download_audio(url: str, output_filename: str = 'audio') -> str:
    """
    Download audio only from a YouTube video (fast).
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_filename}.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            output_path = f"{output_filename}.mp3"
            print(f"✓ Downloaded audio: {output_path}")
            return output_path
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'proxy' in error_msg.lower() or '403' in error_msg or 'Forbidden' in error_msg:
            raise Exception(f"Network error: Cannot access YouTube. Your network may block YouTube access. Original error: {e}")
        elif 'unavailable' in error_msg.lower():
            raise Exception(f"Video unavailable: The video may be private, deleted, or region-locked. Original error: {e}")
        print(f"✗ Audio download failed: {e}")
        raise
    except OSError as e:
        if 'Tunnel connection failed' in str(e) or 'proxy' in str(e).lower():
            raise Exception(f"Network error: Cannot connect to YouTube through proxy. Error: {e}")
        raise
    except Exception as e:
        print(f"✗ Audio download failed: {e}")
        raise
def get_video_info(url: str) -> dict:
    """
    Get video information without downloading.
    
    Args:
        url: YouTube video URL
    
    Returns:
        dict: Video information dictionary with description, uploader, channel, etc.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', ''),
                'channel': info.get('channel', ''),
                'channel_id': info.get('channel_id', ''),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
                'categories': info.get('categories', []),
                'tags': info.get('tags', []),
                'formats': info.get('formats', []),
                'thumbnail': info.get('thumbnail', ''),
                'id': info.get('id', ''),
            }
    except Exception as e:
        print(f"✗ Failed to get video info: {e}")
        raise
