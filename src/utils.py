"""
Utility functions for Director's Cut.
Includes video downloading logic integrated from crop-vid.
"""

import yt_dlp
import os
import re
import requests
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Optional

from src.paths import ensure_runtime_dirs

# Proxy service URL (set via YOUTUBE_PROXY_URL env var)
YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").rstrip("/")

# Cookies file path (set via YOUTUBE_COOKIES_FILE env var or default location)
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "")

# Proxy configuration (set via YOUTUBE_PROXY env var, format: http://proxy:port or socks5://proxy:port)
YOUTUBE_PROXY = os.getenv("YOUTUBE_PROXY", "")

def _use_proxy_service() -> bool:
    """Check if proxy service is configured and available."""
    if not YOUTUBE_PROXY_URL:
        return False
    try:
        response = requests.get(f"{YOUTUBE_PROXY_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def _get_cookies_file() -> Optional[str]:
    """Get cookies file path if available."""
    if YOUTUBE_COOKIES_FILE and os.path.exists(YOUTUBE_COOKIES_FILE):
        return YOUTUBE_COOKIES_FILE
    # Check default locations
    default_paths = [
        "/tmp/youtube_cookies.txt",
        "youtube_cookies.txt",
        os.path.join(os.path.expanduser("~"), ".youtube_cookies.txt"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path
    return None

def _get_ydl_opts_with_cookies(base_opts: dict) -> dict:
    """Add cookies and proxy to yt-dlp options if available."""
    cookies_file = _get_cookies_file()
    if cookies_file:
        base_opts['cookiefile'] = cookies_file
        print(f"✓ Using cookies file: {cookies_file}")
    
    if YOUTUBE_PROXY:
        base_opts['proxy'] = YOUTUBE_PROXY
        print(f"✓ Using proxy: {YOUTUBE_PROXY}")
    
    return base_opts

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
    # Try proxy service first if available
    if _use_proxy_service():
        try:
            response = requests.post(
                f"{YOUTUBE_PROXY_URL}/api/download-segment",
                json={
                    "url": url,
                    "start_time": start_time,
                    "end_time": end_time
                },
                timeout=600,  # 10 minutes for video download
                stream=True
            )
            if response.status_code == 200:
                output_path = f"{output_filename}.mp4"
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded segment via proxy: {output_path}")
                return output_path
        except Exception as proxy_error:
            print(f"⚠️ Proxy service failed, falling back to direct: {proxy_error}")
    
    # Fall back to direct download
    ydl_opts = {
        'format': format_spec,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'outtmpl': f'{output_filename}.%(ext)s',
        'force_keyframes_at_cuts': True,
        'quiet': False,
        'no_warnings': False,
    }
    ydl_opts = _get_ydl_opts_with_cookies(ydl_opts)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get('ext', 'mp4')
            output_path = f"{output_filename}.{ext}"
            print(f"✓ Downloaded segment: {output_path}")
            return output_path
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if 'failed to resolve' in error_msg or 'no address associated with hostname' in error_msg:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
        if 'proxy' in error_msg or '403' in error_msg or 'Forbidden' in error_msg:
            raise Exception(f"Network error: Cannot access YouTube. Your network may block YouTube access. Original error: {e}")
        elif 'unavailable' in error_msg.lower():
            raise Exception(f"Video unavailable: The video may be private, deleted, or region-locked. Original error: {e}")
        print(f"✗ Download failed: {e}")
        raise
    except OSError as e:
        error_str = str(e).lower()
        if 'failed to resolve' in error_str or 'no address associated with hostname' in error_str:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
        if 'Tunnel connection failed' in str(e) or 'proxy' in str(e).lower():
            raise Exception(f"Network error: Cannot connect to YouTube through proxy. Error: {e}")
        raise
    except Exception as e:
        error_str = str(e).lower()
        if 'failed to resolve' in error_str or 'no address associated with hostname' in error_str:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
        print(f"✗ Unexpected error: {e}")
        raise

def download_audio(url: str, output_filename: str = 'audio') -> str:
    """
    Download audio only from a YouTube video (fast).
    """
    # Try proxy service first if available
    if _use_proxy_service():
        try:
            response = requests.post(
                f"{YOUTUBE_PROXY_URL}/api/download-audio",
                json={"url": url},
                timeout=300,  # 5 minutes for download
                stream=True
            )
            if response.status_code == 200:
                output_path = f"{output_filename}.mp3"
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded audio via proxy: {output_path}")
                return output_path
        except Exception as proxy_error:
            print(f"⚠️ Proxy service failed, falling back to direct: {proxy_error}")
    
    # Fall back to direct download
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
    ydl_opts = _get_ydl_opts_with_cookies(ydl_opts)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            output_path = f"{output_filename}.mp3"
            print(f"✓ Downloaded audio: {output_path}")
            return output_path
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if 'failed to resolve' in error_msg or 'no address associated with hostname' in error_msg:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
        if 'proxy' in error_msg or '403' in error_msg or 'Forbidden' in error_msg:
            raise Exception(f"Network error: Cannot access YouTube. Your network may block YouTube access. Original error: {e}")
        elif 'unavailable' in error_msg.lower():
            raise Exception(f"Video unavailable: The video may be private, deleted, or region-locked. Original error: {e}")
        print(f"✗ Audio download failed: {e}")
        raise
    except OSError as e:
        error_str = str(e).lower()
        if 'failed to resolve' in error_str or 'no address associated with hostname' in error_str:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
        if 'Tunnel connection failed' in str(e) or 'proxy' in str(e).lower():
            raise Exception(f"Network error: Cannot connect to YouTube through proxy. Error: {e}")
        raise
    except Exception as e:
        error_str = str(e).lower()
        if 'failed to resolve' in error_str or 'no address associated with hostname' in error_str:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube.\n\n"
                "**Workarounds:**\n"
                "1. Use the MCP server locally (run `python app.py` on your machine)\n"
                "2. Upload video files directly using the 'Upload Video' option\n"
                f"Original error: {e}"
            )
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
    # Try proxy service first if available
    if _use_proxy_service():
        try:
            response = requests.post(
                f"{YOUTUBE_PROXY_URL}/api/video-info",
                json={"url": url},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                # Convert to expected format
                return {
                    'title': data.get('title', ''),
                    'description': data.get('description', ''),
                    'duration': data.get('duration', 0),
                    'uploader': data.get('uploader', ''),
                    'channel': data.get('channel', ''),
                    'channel_id': '',
                    'upload_date': '',
                    'view_count': 0,
                    'categories': [],
                    'tags': [],
                    'formats': [],
                    'thumbnail': data.get('thumbnail', ''),
                    'id': data.get('id', ''),
                }
        except Exception as proxy_error:
            print(f"⚠️ Proxy service failed, falling back to direct: {proxy_error}")
    
    # Fall back to direct download
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    ydl_opts = _get_ydl_opts_with_cookies(ydl_opts)
    
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
        error_str = str(e).lower()
        # Check for DNS/network errors (common on HF Spaces)
        if 'failed to resolve' in error_str or 'no address associated with hostname' in error_str or 'name resolution' in error_str:
            raise Exception(
                "❌ Network Error: Cannot access YouTube. This is a known limitation on Hugging Face Spaces.\n\n"
                "HF Spaces restricts outbound network access to YouTube for security reasons.\n\n"
                "**Workarounds:**\n"
                "1. Deploy the YouTube proxy service (see youtube_proxy_service.py) and set YOUTUBE_PROXY_URL\n"
                "2. Use the MCP server locally (run `python app.py` on your machine)\n"
                "3. Upload video files directly using the 'Upload Video' option in the Production Studio tab\n"
                "4. Use the Space for MCP file downloads only (connect Claude Desktop to the Space URL)\n\n"
                f"Original error: {e}"
            )
        print(f"✗ Failed to get video info: {e}")
        raise
