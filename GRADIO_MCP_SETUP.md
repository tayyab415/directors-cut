# Gradio MCP Server Setup

This document explains how the Gradio MCP server was built from `app.py`.

## Overview

The `app.py` file has been converted into a Gradio MCP server that exposes video editing tools via the Model Context Protocol (MCP). This allows AI assistants like Claude Desktop to interact with your video editing pipeline.

## Director's Cut Architecture

Director's Cut is an **autonomous video editing agent** that transforms long-form YouTube videos into viral short-form content (30-60 seconds) optimized for TikTok, Instagram Reels, and YouTube Shorts.

### Core Components

1. **Scout** (`src/scout.py`): Finds interesting moments using signal processing
   - Audio analysis (loudness, energy peaks)
   - Bad audio detection (silence, noise filtering)
   - Semantic trigger detection in transcripts

2. **Verifier** (`src/verifier.py`): Validates clip quality with vision AI
   - Standard verification for podcast pipeline
   - Deep verification (Gemini 2.0 Flash) for generic pipeline

3. **Director** (`src/director.py`): Creates edit plans using AI
   - Uses Gemini 2.0 Flash Lite to select best clips
   - Ensures 30-60 second total duration
   - Arranges clips for optimal flow

4. **Hands** (`src/hands.py`): Executes the edit plan
   - Concatenates selected clips
   - Handles timestamp mapping and offsets
   - Renders final video with MoviePy

5. **Showrunner** (`src/showrunner.py`): Adds production value
   - Smart crop (Gemini + Qwen VL subject tracking)
   - Intro generation (FLUX)
   - Voiceover (ElevenLabs)
   - Subtitles (WhisperX)
   - Background music

### Two Pipelines

**Podcast Pipeline** (semantic-heavy):
- Uses transcript semantic triggers to find interesting quotes
- Filters out bad audio segments
- Best for interviews, conversations, long-form discussions

**Generic Pipeline** (multimodal):
- Uses audio peaks + visual triggers
- Deep AI verification for viral potential
- Best for tutorials, demos, visual content

See `SMART_CROP_METHODOLOGY.md` and `FIXES_APPLIED.md` for detailed technical documentation.

## What Was Changed

### 1. Fixed Launch Bug
- **Before**: Line 1185 had `demo.launch()` but variable was `app`
- **After**: Fixed to use `app.launch()` with `mcp_server=True`

### 2. Added MCP Tools

The following functions are now exposed as MCP tools:

#### `process_video(url: str) -> str`
Main pipeline that processes a YouTube video through the full autonomous editing workflow:
- Classifies video as podcast or generic
- Finds hotspots using audio + transcript analysis
- Verifies clips with vision AI
- Creates edit plan
- Renders final video

#### `classify_video_type(youtube_url: str) -> str`
Classifies a YouTube video as 'podcast' or 'generic' based on metadata.

#### `smart_crop_video(video_path: str) -> str`
Applies AI-powered smart crop to convert video to 9:16 vertical format using:
- Gemini for scene change analysis
- Qwen VL for subject position tracking
- Smooth interpolated crop tracking

#### `add_production_value(video_path, mood, enable_smart_crop, add_intro_image, add_subtitles) -> str`
Adds professional production value with:
- Smart vertical crop (9:16)
- AI-generated intro images (FLUX)
- Auto-generated subtitles (WhisperX)
- Professional voiceover (ElevenLabs)
- Background music matched to mood

### 3. Added MCP Resources

#### `resource://video_classification_criteria`
Returns criteria used to classify videos (podcast channels, keywords, triggers).

#### `resource://supported_features`
Returns list of supported features, pipelines, and moods.

### 4. Added MCP Prompts

#### `podcast_editing_workflow()`
Template prompt for editing podcast-style videos.

#### `generic_video_editing_workflow()`
Template prompt for editing generic/tutorial videos.

## Installation

Make sure you have Gradio with MCP support installed:

```bash
pip install "gradio[mcp]"
```

Or if using requirements.txt, add:
```
gradio[mcp]>=4.0.0
```

## Running the Server

### Option 1: Direct Launch
```bash
python app.py
```

### Option 2: Using Helper Script (Recommended)
```bash
python launch_app.py
```

The helper script will:
- Check if port 7860 is available
- Offer to kill processes using the port if needed
- Automatically use a different port if 7860 is busy

### Option 3: Custom Port
```bash
GRADIO_SERVER_PORT=7861 python app.py
```

### Troubleshooting Port Conflicts

If you see `address already in use` error:

1. **Find and kill the process:**
   ```bash
   lsof -ti:7860 | xargs kill
   ```

2. **Or use a different port:**
   ```bash
   GRADIO_SERVER_PORT=7861 python app.py
   ```

This will start:
- **Web UI**: `http://localhost:7860` (or your custom port)
- **MCP SSE Endpoint**: `http://localhost:7860/gradio_api/mcp/sse`
- **MCP Schema**: `http://localhost:7860/gradio_api/mcp/schema`

## Connecting to Claude Desktop

Add this to your Claude Desktop configuration:

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "directors-cut": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  }
}
```

**Note**: You need `mcp-remote` installed:
```bash
npm install -g mcp-remote
```

## Alternative: Direct Python Connection

If you prefer not to use `mcp-remote`, you can use a wrapper script, but the Gradio MCP server is designed to work with SSE (Server-Sent Events) via `mcp-remote`.

## Testing the MCP Server

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Check MCP schema**:
   ```bash
   curl http://localhost:7860/gradio_api/mcp/schema
   ```

3. **Use in Claude Desktop**:
   Once configured, you can ask Claude:
   - "Process this YouTube video: https://youtube.com/watch?v=..."
   - "Classify this video type: https://youtube.com/watch?v=..."
   - "Add production value to this video: /path/to/video.mp4"

## MCP Tools Available

### 1. `process_video(url: str) -> str`
**Main autonomous editing pipeline** - Processes YouTube videos through the complete editing workflow:
- Automatically classifies as podcast or generic
- Finds hotspots using audio + transcript analysis
- Verifies clips with vision AI
- Creates edit plan with AI Director
- Renders final 30-60 second video

**Use when:** You want fully autonomous editing from YouTube URL to final video.

### 2. `classify_video_type(youtube_url: str) -> str`
**Video classification tool** - Determines if a video is "podcast" or "generic":
- Analyzes metadata (title, description, channel, tags, duration)
- Returns classification with reasoning
- Helps understand which pipeline will be used

**Use when:** You want to verify classification before processing or debug misclassifications.

### 3. `smart_crop_video(video_path: str) -> str`
**AI-powered smart crop** - Converts landscape videos to 9:16 portrait format:
- Uses Gemini 2.0 Flash Lite for scene analysis
- Uses Qwen 2.5-VL-72B for subject position detection
- Applies smooth interpolated crop tracking
- Preserves audio in output

**Use when:** Converting landscape videos for TikTok/Instagram Reels/YouTube Shorts.

### 4. `add_production_value(video_path, mood, enable_smart_crop, add_intro_image, add_subtitles) -> str`
**Production studio pipeline** - Adds professional enhancements:
- Smart vertical crop (9:16 with subject tracking)
- AI-generated intro images (FLUX)
- Professional voiceover (ElevenLabs)
- Background music (mood-matched)
- Auto-generated subtitles (WhisperX)

**Use when:** Polishing videos for social media with professional production value.

## MCP Resources Available

### 1. `resource://video_classification_criteria`
Provides transparency into classification system:
- Known podcast channels
- Podcast vs generic keywords
- Semantic triggers for each pipeline
- Classification notes and fallback behavior

**Use when:** Understanding why videos are classified a certain way or extending criteria.

### 2. `resource://supported_features`
Comprehensive feature overview:
- Detailed pipeline descriptions (podcast vs generic)
- Production feature specifications
- Mood options and descriptions
- Output format details
- API requirements

**Use when:** Understanding capabilities, requirements, or planning workflows.

## MCP Prompts Available

### 1. `podcast_editing_workflow`
Complete step-by-step guide for editing podcast-style videos:
- Classification process
- Pipeline explanation (semantic-heavy approach)
- Production enhancement options
- Best practices and tips

**Use when:** Learning how to edit podcast/interview content or as a reference.

### 2. `generic_video_editing_workflow`
Complete step-by-step guide for editing generic/tutorial videos:
- Classification process
- Pipeline explanation (multimodal with deep verification)
- Production enhancement options
- Best practices and tips

**Use when:** Learning how to edit tutorial/demo/visual content or as a reference.

## Notes

- The Gradio web UI continues to work as before
- MCP tools are exposed alongside the UI
- All existing functionality is preserved
- The server runs on `0.0.0.0:7860` by default

