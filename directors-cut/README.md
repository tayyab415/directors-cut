# Director's Cut - Dual MCP Server for Video Editing

Transform YouTube videos into viral vertical content using AI-powered analysis and editing.

## Overview

Director's Cut is a dual MCP (Model Context Protocol) server system that provides autonomous video editing capabilities:

1. **Gradio MCP Server** (`gradio_server.py`) - Main video processing with web UI + MCP endpoint
2. **FastMCP Editing Server** (`editing_server.py`) - Fast local editing operations for iterative refinements

## Features

### Gradio Server Tools
- `directors_cut_main` - Full pipeline: download → viral detection → crop → intro → export
- `analyze_video_segments` - AI-powered segment analysis for engagement scoring
- `convert_to_vertical` - Smart 9:16 conversion with subject tracking
- `generate_custom_intro` - AI-generated video intros
- `add_auto_subtitles` - Automatic speech-to-text subtitles
- `batch_process_playlist` - Process entire YouTube playlists

### FastMCP Editing Tools
- `trim_video` - Cut from start or end
- `change_intro_background` - Regenerate intro with new background
- `adjust_brightness` - Exposure adjustment
- `add_background_music` - Background audio mixing
- `change_aspect_ratio` - Crop/resize with smart detection
- `apply_filter` - Visual filters (sepia, grayscale, vibrant, etc.)
- `change_playback_speed` - Slow-motion or time-lapse
- `extract_segment` - Extract specific timestamps

## Installation

```bash
cd directors-cut
pip install -r requirements.txt
```

### System Dependencies

Make sure you have FFmpeg installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

## Running the Servers

### Gradio MCP Server (Web UI + MCP)

```bash
python gradio_server.py
```

This starts:
- Web UI at `http://localhost:7860`
- MCP SSE endpoint at `http://localhost:7860/gradio_api/mcp/sse`
- MCP schema at `http://localhost:7860/gradio_api/mcp/schema`

### FastMCP Editing Server (STDIO)

```bash
python editing_server.py
# or
mcp run editing_server.py
```

## Claude Desktop Configuration

Add these servers to your Claude Desktop configuration file:

### Location
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Configuration

```json
{
  "mcpServers": {
    "directors-cut-gradio": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    },
    "directors-cut-editor": {
      "command": "python",
      "args": [
        "/path/to/directors-cut/editing_server.py"
      ]
    }
  }
}
```

**Important**: Replace `/path/to/directors-cut/` with the actual path to your installation.

### Alternative: Direct Python for Gradio Server

If you prefer not to use `mcp-remote`, you can also configure like this:

```json
{
  "mcpServers": {
    "directors-cut-gradio": {
      "command": "python",
      "args": [
        "/path/to/directors-cut/gradio_server.py"
      ],
      "env": {
        "GRADIO_SERVER_PORT": "7860"
      }
    },
    "directors-cut-editor": {
      "command": "python",
      "args": [
        "/path/to/directors-cut/editing_server.py"
      ]
    }
  }
}
```

## Usage Examples

### With Claude Desktop

Once configured, you can ask Claude:

```
"Download this YouTube video and create a 30-second vertical clip for TikTok: https://youtube.com/watch?v=..."

"Analyze this video and find the most engaging 30-second segment"

"Make the video brighter and add an intro that says 'Wait for it!'"

"Convert this landscape video to vertical format for Instagram Reels"
```

### Programmatic Usage

```python
# Use Gradio server tools directly
from gradio_server import directors_cut_main, analyze_video_segments

result = directors_cut_main(
    youtube_url="https://youtube.com/watch?v=example",
    intro_text="You won't believe this!",
    target_duration=30
)
print(f"Processed video: {result}")
```

## Resources (Read-Only Data)

### Gradio Server Resources
- `resource://supported_formats` - List of supported video formats
- `resource://intro_templates` - Available intro styles
- `resource://viral_detection_criteria` - Engagement scoring weights
- `resource://processing_status/{job_id}` - Job progress tracking
- `resource://output_specifications` - Platform-specific specs

### FastMCP Editor Resources
- `resource://editing_history` - Session operation log
- `resource://video_metadata/{video_path}` - Video file info
- `resource://available_filters` - Filter descriptions

## Prompts (LLM Guidance)

### Gradio Server Prompts
- `optimize_for_platform(platform)` - Platform-specific optimization tips
- `create_viral_compilation(topic)` - Compilation creation guide
- `tutorial_video_structure(subject)` - Tutorial video framework

### FastMCP Editor Prompts
- `quick_social_edit()` - Common social media edit sequence
- `fix_common_issues(issue_type)` - Troubleshooting guide

## Project Structure

```
directors-cut/
├── gradio_server.py              # Gradio MCP server (main pipeline)
├── editing_server.py             # FastMCP editing server
├── video_processing/
│   ├── __init__.py
│   ├── ffmpeg_utils.py          # FFmpeg wrapper functions
│   ├── download.py              # YouTube download with yt-dlp
│   ├── viral_detection.py       # AI segment analysis
│   ├── conversion.py            # Format/aspect conversion
│   ├── intro_generation.py      # AI intro creation
│   └── subtitle_generation.py   # Auto subtitle creation
├── inputs/                       # Downloaded videos
├── temp/                         # Processing temp files
├── outputs/                      # Final processed videos
├── requirements.txt
└── README.md
```

## Environment Variables

Optional environment variables for enhanced features:

```bash
# For OpenAI Whisper API (auto-subtitles)
export OPENAI_API_KEY=your_api_key

# For AI image generation (future)
export ANTHROPIC_API_KEY=your_api_key
```

## Troubleshooting

### "FFmpeg not found"
Install FFmpeg and ensure it's in your PATH.

### "Module not found"
Run `pip install -r requirements.txt` from the directors-cut directory.

### "MCP connection failed"
1. Ensure the Gradio server is running before connecting Claude Desktop
2. Check the port is not in use: `lsof -i :7860`
3. Verify the MCP endpoint: `curl http://localhost:7860/gradio_api/mcp/schema`

### "Video download failed"
- Check internet connection
- Verify YouTube URL is valid and video is public
- Update yt-dlp: `pip install -U yt-dlp`

## License

MIT License
