# Walkthrough - Production Studio Integration

## Changes Implemented
I have added a new "Production Studio" tab to the Director's Cut application, enabling users to enhance videos with AI-generated production elements.

### 1. New `Showrunner` Module
Created `src/showrunner.py` which handles the creative direction and asset generation:
- **Video Analysis**: Uses Gemini 2.0 Flash to determine mood (hype, suspense, chill) and generate scripts.
- **Voiceover**: Uses ElevenLabs API to generate professional intros.
- **Music Selection**: Selects background music from local assets based on mood.
- **Title Cards**: Uses Nebius FLUX API to generate high-quality title cards.
- **Composition**: Uses MoviePy to combine video, audio, and images into a polished final product.

### 2. App Integration
Updated `app.py` to:
- Import the `Showrunner` class.
- Restructure the UI into two tabs: "Create Clip" (existing) and "Production Studio" (new).
- Add the `add_production_wrapper` function to handle the Gradio interactions for the new tab.

### 3. Dependencies
- Added `elevenlabs` and `requests` to `pyproject.toml`.
- Updated `.env` with placeholders for `ELEVENLABS_API_KEY` and `NEBIUS_API_KEY`.

### 4. Assets
- Created directory structure `assets/music/{hype,suspense,chill}` for storing music files.

## Verification Results
### Automated Checks
- Verified that `app.py` and `src/showrunner.py` import correctly without errors.
- Confirmed that `moviepy` imports are compatible with the installed version (v2.2.1).

### Manual Verification Steps
To fully verify the feature, the user needs to:
1.  **Add API Keys**: Update `.env` with valid `ELEVENLABS_API_KEY` and `NEBIUS_API_KEY`.
2.  **Add Music**: Place MP3 files in `assets/music/hype`, `assets/music/suspense`, and `assets/music/chill`.
3.  **Run App**: Execute `python app.py`.
4.  **Test**: Upload a video to the "Production Studio" tab and click "Add Production Value".

## Screenshots
![Production Studio UI](/Users/tayyabkhan/.gemini/antigravity/brain/f0a595bd-5c00-4ab0-bb6d-0a15d0cb04b6/production_studio_ui.png)

**Status**: ✅ UI Verified. The "Production Studio" tab is present and functional.
