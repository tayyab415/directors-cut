#!/usr/bin/env python3
"""
YouTube Proxy Service
A simple Flask service that downloads YouTube videos and serves them.
Deploy this separately (Railway, Render, Fly.io, etc.) to bypass HF Spaces restrictions.

Usage:
    pip install flask flask-cors yt-dlp
    python youtube_proxy_service.py

Environment variables:
    PORT: Server port (default: 5000)
    ALLOWED_ORIGINS: Comma-separated list of allowed origins (default: *)
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import os
import tempfile
import logging
from urllib.parse import urlparse, parse_qs
import re

app = Flask(__name__)

# CORS configuration
allowed_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, origins=allowed_origins)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL."""
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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

@app.route('/api/video-info', methods=['POST'])
def get_video_info():
    """Get video metadata without downloading."""
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL required"}), 400
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return jsonify({
                'title': info.get('title', ''),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', ''),
                'channel': info.get('channel', ''),
                'description': info.get('description', '')[:500],
                'thumbnail': info.get('thumbnail', ''),
                'id': info.get('id', ''),
            }), 200
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-audio', methods=['POST'])
def download_audio():
    """Download audio from YouTube video."""
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL required"}), 400
        
        video_id = extract_video_id(url)
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{video_id}.mp3")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
            'quiet': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if os.path.exists(output_path):
            return send_file(
                output_path,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name=f"{video_id}.mp3"
            )
        else:
            # Try to find the actual file
            for file in os.listdir(temp_dir):
                if file.endswith('.mp3'):
                    return send_file(
                        os.path.join(temp_dir, file),
                        mimetype='audio/mpeg',
                        as_attachment=True,
                        download_name=f"{video_id}.mp3"
                    )
            return jsonify({"error": "Download failed"}), 500
            
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-segment', methods=['POST'])
def download_segment():
    """Download a specific time segment from YouTube video."""
    try:
        data = request.json
        url = data.get('url')
        start_time = float(data.get('start_time', 0))
        end_time = float(data.get('end_time', 0))
        
        if not url:
            return jsonify({"error": "URL required"}), 400
        if end_time <= start_time:
            return jsonify({"error": "Invalid time range"}), 400
        
        video_id = extract_video_id(url)
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{video_id}_segment.mp4")
        
        ydl_opts = {
            'format': 'best',
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
            'outtmpl': output_path.replace('.mp4', '.%(ext)s'),
            'force_keyframes_at_cuts': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the actual downloaded file
        for file in os.listdir(temp_dir):
            if file.endswith(('.mp4', '.webm', '.mkv')):
                return send_file(
                    os.path.join(temp_dir, file),
                    mimetype='video/mp4',
                    as_attachment=True,
                    download_name=f"{video_id}_segment.mp4"
                )
        
        return jsonify({"error": "Download failed"}), 500
            
    except Exception as e:
        logger.error(f"Error downloading segment: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

