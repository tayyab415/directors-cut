---
title: Director's Cut MCP Studio
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
tags:
  - building-mcp-track-customer
  - mcp-in-action-multimodal
license: mit
---

# Director's Cut MCP Studio

Director's Cut is a Gradio + MCP Space that transforms long-form YouTube videos
into viral-ready vertical edits. Claude Desktop (or any MCP client) can attach
to the Space, call the exposed `gr.mcp.tool` functions, and fetch fully rendered
clips that blend smart scouting, Gemini planning, MoviePy editing, and an
optional production polish pass (smart crop, AI intro, subtitles, music).

## Highlights

- **MCP-ready UI** – Every major workflow step is wrapped with `@gr.mcp.tool`
  so Claude and the web UI stay in sync.
- **Hugging Face friendly storage** – All runtime files live under
  `/tmp/directors-cut/...`, which keeps Spaces deployments stateless while
  preserving download links.
- **Multi-stage editing** – SignalScout + Verifier find hotspots, Director
  writes a plan, Hands renders it, and Showrunner can auto-produce intros,
  crops, subs, and soundtracks.
- **Smart logging** – Progress text streams through the Gradio UI and the
  terminal, so you can debug without touching the MCP plumbing.

## How It Works

1. **Scan & Classify** – Downloads audio, detects peaks, and labels the video as
   podcast or generic to choose the right pipeline.
2. **Verify & Plan** – Gemini vision / text models score hotspots and assemble a
   JSON edit plan.
3. **Render** – MoviePy stitches clips, writes to `/tmp/directors-cut/output`.
4. **Production Studio** – Optional Showrunner pass adds smart crop,
   FLUX-generated intro image, ElevenLabs hook, and Whisper-based subtitles.
5. **Smart Downloads** – Finished files are copied into the runtime output
   directory and exposed to Claude via MCP file responses.

## MCP Tools (excerpt)

| Tool | Purpose |
| --- | --- |
| `process_video` | Full autonomous pipeline end-to-end |
| `scan_video`, `render_edit`, `render_and_produce_mcp` | Step-by-step control |
| `smart_crop_video`, `add_production_value_mcp` | Production-only utilities |
| `get_youtube_info`, `classify_video_type` | Metadata helpers |

Full definitions live in `app.py` and mirror the Gradio tabs.

## Deployment Notes

- **Environment variables**: set `GEMINI_API_KEY`, `VIDEO_API_KEY`,
  `ELEVENLABS_API_KEY`, and `NEBIUS_API_KEY` (optional but recommended) via the
  Spaces Secrets UI. Without them, the corresponding features gracefully degrade
  (e.g., subtitles fall back to no-op).
- **Runtime storage**: override the default `/tmp/directors-cut` location by
  setting `DIRECTORS_CUT_RUNTIME_DIR` if needed.
- **Ports**: the app launches with `server_name="0.0.0.0"` and
  `server_port=7860`, matching HF Spaces defaults and Claude Desktop's MCP
  expectations.
- **Logging**: stdout logs stream to the Space console; Gradio progress boxes
  surface the same messages for MCP clients.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # create and fill secrets
python app.py
```

Claude Desktop can connect locally by pointing to the MCP server URL that
Gradio prints (or the `gradio_share_url.txt` file if you enable sharing).

## Resources

- [Official Gradio MCP guide](https://huggingface.co/blog/gradio-mcp)
- [Spaces as MCP servers](https://huggingface.co/docs/hub/spaces-mcp-servers)
- [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Hackathon details](https://huggingface.co/MCP-1st-Birthday)

Pull requests and feature ideas are welcome—just keep the file handling in
`src/paths.py` up to date so Spaces deployments remain portable.
