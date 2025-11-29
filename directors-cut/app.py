"""
Director's Cut - HuggingFace Space Frontend
============================================

Lightweight Gradio frontend that calls Modal backend for all heavy processing.
This allows YouTube processing without network restrictions.

MCP endpoint: /gradio_api/mcp/sse
"""

import os
import json
import time
import tempfile

import gradio as gr
import requests

# ==============================================================================
# Configuration
# ==============================================================================

# Modal backend base URL (production deployment)
MODAL_BASE_URL = os.getenv(
    "MODAL_BASE_URL", 
    "https://tayyabkhn343--directors-cut"
)

def get_modal_url(endpoint: str) -> str:
    """Build Modal endpoint URL."""
    # Production Modal URLs: https://tayyabkhn343--directors-cut-{endpoint}.modal.run
    # Modal converts underscores to hyphens in endpoint names
    endpoint = endpoint.replace("_", "-")
    return f"{MODAL_BASE_URL}-{endpoint}.modal.run"

# Directories for temporary files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in [TEMP_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# Modal API Client Functions
# ==============================================================================

def call_modal_endpoint(endpoint: str, method: str = "GET", data: dict = None, timeout: int = 600) -> dict:
    """Call a Modal endpoint and return the response."""
    url = get_modal_url(endpoint)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=data or {}, timeout=timeout)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout}s"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from server"}


def download_video_from_modal(job_id: str) -> str | None:
    """Download video file from Modal backend."""
    url = get_modal_url("download") + f"?job_id={job_id}"
    
    try:
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()
        
        # Save to temp file
        output_path = os.path.join(OUTPUTS_DIR, f"directors_cut_{job_id}.mp4")
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    except Exception as e:
        print(f"Download error: {e}")
        return None


# ==============================================================================
# Tool Functions (exposed as MCP tools via gr.Interface)
# ==============================================================================

def directors_cut_main(
    youtube_url: str,
    num_hotspots: str = "5",
    enable_smart_crop: bool = True,
    add_intro: bool = True,
    add_subtitles: bool = True,
    mood: str = "auto",
) -> str:
    """
    Process YouTube video through the full Director's Cut pipeline on Modal cloud.
    
    Extracts viral segments, converts to 9:16 vertical format, adds AI intro,
    and applies subtitles. All heavy processing happens on Modal GPU servers.

    Args:
        youtube_url: Full YouTube video URL to process
        num_hotspots: Number of viral hotspots to detect (default: 5)
        enable_smart_crop: Enable AI subject tracking for 9:16 crop (default: True)
        add_intro: Add FLUX AI-generated intro (default: True)  
        add_subtitles: Add Whisper-transcribed subtitles (default: True)
        mood: Music/style mood - "auto", "hype", "suspense", or "chill"

    Returns:
        Status message with job ID and output path, or error message
    """
    if not youtube_url:
        return "❌ Error: Please provide a YouTube URL"
    
    # Call Modal full pipeline endpoint
    result = call_modal_endpoint(
        "process",
        method="POST",
        data={
            "url": youtube_url,
            "num_hotspots": int(num_hotspots),
            "enable_smart_crop": enable_smart_crop,
            "add_intro": add_intro,
            "add_subtitles": add_subtitles,
            "mood": mood,
        },
        timeout=1800,  # 30 minutes for full pipeline
    )
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    if result.get("success"):
        job_id = result.get("job_id", "unknown")
        stats = result.get("stats", {})
        return f"""✅ Success!

**Job ID:** {job_id}
**Duration:** {stats.get('total_duration', 'N/A')}

**Pipeline Stats:**
- Hotspots detected: {stats.get('hotspots_detected', 'N/A')}
- Clips verified: {stats.get('clips_verified', 'N/A')}
- Final duration: {stats.get('final_duration', 'N/A')}s

**Output:** Available via download endpoint with job_id={job_id}
"""
    else:
        errors = result.get("errors", [])
        return f"❌ Pipeline failed: {result.get('error', 'Unknown error')}\n\nErrors: {errors}"


def get_video_info(youtube_url: str) -> str:
    """
    Get information about a YouTube video without downloading.

    Retrieves title, duration, uploader, description preview, and other metadata.

    Args:
        youtube_url: Full YouTube video URL

    Returns:
        Formatted video information or error message
    """
    if not youtube_url:
        return "Please enter a YouTube URL"
    
    result = call_modal_endpoint(
        "video_info",
        method="POST",
        data={"url": youtube_url},
        timeout=60,
    )
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    duration = result.get("duration", 0)
    minutes = duration // 60
    seconds = duration % 60
    
    return f"""📺 **Video Information**

**Title:** {result.get('title', 'Unknown')}
**Duration:** {minutes}:{seconds:02d}
**Uploader:** {result.get('uploader', 'Unknown')}
**Channel:** {result.get('channel', 'Unknown')}
**View Count:** {result.get('view_count', 'N/A'):,}
**Upload Date:** {result.get('upload_date', 'Unknown')}

**Description Preview:**
{result.get('description', 'No description')[:300]}...
"""


def get_transcript(youtube_url: str, include_timestamps: bool = False) -> str:
    """
    Get the transcript/captions of a YouTube video.

    Uses YouTube's auto-generated or manual captions when available.

    Args:
        youtube_url: Full YouTube video URL
        include_timestamps: Include timestamps with each segment (default: False)

    Returns:
        Video transcript text or error message
    """
    if not youtube_url:
        return "Please enter a YouTube URL"
    
    result = call_modal_endpoint(
        "transcript",
        method="POST",
        data={
            "url": youtube_url,
            "include_timestamps": include_timestamps,
        },
        timeout=300,
    )
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    transcript = result.get("transcript", "No transcript available")
    return f"📝 **Transcript:**\n\n{transcript}"


def check_service_health() -> str:
    """
    Check if the Modal backend service is healthy and available.

    Returns:
        Service health status and timestamp
    """
    result = call_modal_endpoint("health", method="GET", timeout=10)
    
    if "error" in result:
        return f"❌ Service unavailable: {result['error']}"
    
    return f"""✅ **Service Status: Healthy**

**Service:** {result.get('service', 'directors-cut')}
**Timestamp:** {result.get('timestamp', 'N/A')}

Modal backend is running and ready to process videos!
"""


def get_pipeline_state() -> str:
    """
    Get current workflow/pipeline state from Modal backend.

    Returns:
        Current pipeline state and progress information
    """
    result = call_modal_endpoint("state", method="GET", timeout=30)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    current_step = result.get("current_step", "unknown")
    completed = result.get("completed_steps", [])
    
    return f"""📊 **Pipeline Status**

**Current Step:** {current_step}
**Completed Steps:** {', '.join(completed) if completed else 'None'}
**Video ID:** {result.get('video_id', 'N/A')}
**Title:** {result.get('title', 'N/A')}
"""


def list_outputs() -> str:
    """
    List all processed video outputs available for download.

    Returns:
        List of output files with their job IDs
    """
    result = call_modal_endpoint("outputs", method="GET", timeout=30)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    outputs = result.get("outputs", [])
    
    if not outputs:
        return "📁 No processed outputs available yet."
    
    output_list = "\n".join([f"- {f}" for f in outputs])
    return f"📁 **Available Outputs:**\n\n{output_list}"


# ==============================================================================
# Gradio UI
# ==============================================================================

def create_demo():
    """Create Gradio demo with MCP support."""
    
    with gr.Blocks(
        title="Director's Cut",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
# 🎬 Director's Cut

**AI-Powered Video Editor** - Transform YouTube videos into viral short-form content.

Powered by **Modal Labs** cloud GPUs for:
- ✅ Full YouTube access (no network restrictions)
- ✅ AI hotspot detection & verification
- ✅ FLUX AI intro generation
- ✅ Whisper GPU transcription
- ✅ Smart 9:16 subject tracking
        """)
        
        with gr.Tabs():
            # Main Processing Tab
            with gr.Tab("🚀 Process Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        url_input = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://youtube.com/watch?v=...",
                            lines=1,
                        )
                        
                        gr.Markdown("### Options")
                        
                        with gr.Row():
                            num_hotspots = gr.Dropdown(
                                choices=["3", "4", "5", "6", "7", "8", "9", "10"],
                                value="5",
                                label="Hotspots to Detect"
                            )
                            mood = gr.Dropdown(
                                choices=["auto", "hype", "suspense", "chill"],
                                value="auto",
                                label="Mood"
                            )
                        
                        with gr.Row():
                            smart_crop = gr.Checkbox(value=True, label="Smart Crop 9:16")
                            add_intro = gr.Checkbox(value=True, label="AI Intro")
                            add_subtitles = gr.Checkbox(value=True, label="Subtitles")
                        
                        process_btn = gr.Button(
                            "🎬 Process Video",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        output_status = gr.Markdown(label="Status")
                
                process_btn.click(
                    fn=directors_cut_main,
                    inputs=[url_input, num_hotspots, smart_crop, add_intro, add_subtitles, mood],
                    outputs=[output_status],
                    api_name="process_video"
                )
            
            # Tools Tab
            with gr.Tab("🛠️ Tools"):
                gr.Markdown("### Video Info")
                with gr.Row():
                    info_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
                    info_btn = gr.Button("Get Info")
                info_output = gr.Markdown()
                info_btn.click(
                    fn=get_video_info,
                    inputs=[info_url],
                    outputs=[info_output],
                    api_name="get_video_info"
                )
                
                gr.Markdown("---")
                gr.Markdown("### Transcript")
                with gr.Row():
                    trans_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
                    trans_timestamps = gr.Checkbox(label="Include Timestamps")
                    trans_btn = gr.Button("Get Transcript")
                trans_output = gr.Markdown()
                trans_btn.click(
                    fn=get_transcript,
                    inputs=[trans_url, trans_timestamps],
                    outputs=[trans_output],
                    api_name="get_transcript"
                )
                
                gr.Markdown("---")
                gr.Markdown("### Service Status")
                with gr.Row():
                    health_btn = gr.Button("Check Health")
                    state_btn = gr.Button("Pipeline State")
                    outputs_btn = gr.Button("List Outputs")
                status_output = gr.Markdown()
                health_btn.click(fn=check_service_health, outputs=[status_output], api_name="health_check")
                state_btn.click(fn=get_pipeline_state, outputs=[status_output], api_name="pipeline_state")
                outputs_btn.click(fn=list_outputs, outputs=[status_output], api_name="list_outputs")
            
            # API/MCP Tab
            with gr.Tab("📚 API & MCP"):
                gr.Markdown("""
### MCP Integration

Connect Claude Desktop or other MCP clients to this Space:

```json
{
    "mcpServers": {
        "directors-cut": {
            "url": "https://tyb343-directors-cut.hf.space/gradio_api/mcp/sse"
        }
    }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `process_video` | Full pipeline - extract viral clips from YouTube |
| `get_video_info` | Get video metadata without downloading |
| `get_transcript` | Get video transcript/captions |
| `health_check` | Check backend health |
| `pipeline_state` | Get current pipeline state |
| `list_outputs` | List available outputs |

### Direct API

POST to `/gradio_api/call/<tool_name>` with JSON body.
                """)
        
        gr.Markdown("""
---
*Powered by [Modal Labs](https://modal.com) • Built for the Anthropic MCP Hackathon*
        """)
    
    return demo


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        mcp_server=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
