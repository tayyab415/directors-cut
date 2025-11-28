import google.generativeai as genai
import os
import time
import json
import random
import requests
import cv2
import base64
import gc
from elevenlabs import ElevenLabs, save
from moviepy import VideoFileClip, AudioFileClip, ColorClip, ImageClip, concatenate_videoclips, CompositeAudioClip, concatenate_audioclips, TextClip, CompositeVideoClip
import numpy as np
import logging
import subprocess
from dotenv import load_dotenv

from src.paths import WORK_DIR, FRAMES_DIR, STATIC_MUSIC_DIR, ensure_runtime_dirs

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Showrunner:
    def __init__(self):
        # Force reload .env to pick up any changes
        load_dotenv(override=True)
        ensure_runtime_dirs()

        self.video_api_key = os.getenv("VIDEO_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.nebius_api_key = os.getenv("NEBIUS_API_KEY")
        self.work_dir = WORK_DIR
        self.frames_dir = FRAMES_DIR
        self.music_root = STATIC_MUSIC_DIR

        if self.video_api_key:
            genai.configure(api_key=self.video_api_key)

    def analyze_video(self, video_path):
        """Uses Gemini to analyze video and decide creative direction."""
        logger.info(f"Analyzing video: {video_path}")
        try:
            video_file = genai.upload_file(video_path)

            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise Exception("Gemini video processing failed")

            model = genai.GenerativeModel("gemini-2.0-flash-exp")

            prompt = """
            You are The Showrunner - creative director for viral social media content.

            Watch this video compilation and make creative decisions.

            First, analyze the video content:
            - WHO is in the video? (people, personalities, characters)
            - WHAT are they doing/discussing? (main topic, action, conversation)
            - CONTEXT: What's the setting, tone, or situation?

            MOOD: Choose exactly ONE based on the video content:
            - hype: exciting, energetic, fast-paced content
            - suspense: mysterious, dramatic, intriguing content
            - chill: thoughtful, calm, insightful content

            INTRO_SCRIPT: Create a specific 15-20 word hook that REFERENCES THE ACTUAL VIDEO CONTENT.
            - Mention who is in the video or what they're discussing
            - Reference the specific topic, person, or situation shown
            - Make it feel like you watched the video and are teasing the content
            - This MUST be spoken in 6-7 seconds at a natural pace

            Good examples (context-aware):
            - "Joe Rogan just dropped some insane knowledge about AI. This take is gonna blow your mind, check it out..."
            - "Watch this street performer absolutely nail the most difficult guitar solo I've ever heard. The crowd's reaction says it all..."
            - "This debate between two top scientists about climate change gets heated real fast. Wait till you hear their main argument..."

            Bad examples:
            - "You won't believe what happens next..." (TOO GENERIC - doesn't reference video content)
            - "Check this out" (TOO SHORT and generic)
            - "In this video we will discuss..." (too formal)

            TITLE_CARD: 3-5 impactful words for text overlay that relate to the video content
            Examples: JOE ROGAN AI, EPIC GUITAR SOLO, HEATED DEBATE

            Return ONLY valid JSON with no markdown:
            {"mood": "suspense", "intro_script": "...", "title_card": "..."}
            """

            response = model.generate_content([video_file, prompt])
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            # Fallback
            return {
                "mood": "hype",
                "intro_script": "Check out this amazing video!",
                "title_card": "WATCH THIS"
            }

    def generate_intro(self, script):
        """Uses ElevenLabs to create voiceover."""
        logger.info("Generating intro voiceover...")
        if not self.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY missing, skipping voiceover")
            return None

        output_path = os.path.join(self.work_dir, "intro.mp3")

        try:
            client = ElevenLabs(api_key=self.elevenlabs_api_key)

            audio = client.text_to_speech.convert(
                text=script,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
                model_id="eleven_turbo_v2_5"  # Free tier compatible model
            )

            # Write audio to file
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            # Validate file was created and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Voiceover generated at: {output_path}")
                return output_path
            else:
                logger.error("Voiceover file is empty or wasn't created")
                if os.path.exists(output_path):
                    os.remove(output_path)  # Clean up empty file
                return None

        except Exception as e:
            import traceback
            logger.error(f"Voiceover generation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Clean up any partial/empty file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return None

    def select_music(self, mood):
        """Picks random music file from local library based on mood."""
        logger.info(f"Selecting music for mood: {mood}")
        music_dir = os.path.join(self.music_root, mood)

        if not os.path.exists(music_dir):
            logger.warning(f"Music directory {music_dir} not found")
            return None

        files = [f for f in os.listdir(music_dir) if f.endswith('.mp3')]

        if not files:
            logger.warning(f"No music files in {music_dir}")
            return None

        selected = random.choice(files)
        return os.path.join(music_dir, selected)

    def generate_card(self, text, mood):
        """Uses Nebius FLUX API to create title card image."""
        logger.info("Generating title card...")
        if not self.nebius_api_key:
            logger.warning("NEBIUS_API_KEY missing, skipping title card")
            return None

        try:
            style_prompts = {
                "hype": f"Energetic vibrant background with bold text '{text}', high energy, bright colors, dynamic composition",
                "suspense": f"Dark moody cinematic background with dramatic text '{text}', mysterious lighting, noir aesthetic",
                "chill": f"Calm aesthetic background with elegant text '{text}', soft pastel colors, minimalist design"
            }

            prompt = style_prompts.get(mood, style_prompts["hype"])

            response = requests.post(
                "https://api.studio.nebius.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.nebius_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "black-forest-labs/FLUX.1-schnell",  # Updated to available model
                    "prompt": prompt,
                    "width": 1080,
                    "height": 1920,
                    "num_inference_steps": 30,
                    "response_format": "b64_json"  # Request base64 format for reliability
                }
            )

            if response.status_code != 200:
                logger.error(f"Nebius API error: {response.text}")
                return None

            result = response.json()
            logger.info(f"FLUX API response structure: {list(result.keys())}")

            data = result.get('data', [])

            if not data:
                logger.error(f"Nebius API returned empty data: {result}")
                return None

            image_entry = data[0]
            logger.info(f"Image entry keys: {list(image_entry.keys())}")

            output_path = os.path.join(self.work_dir, "title_card.jpg")

            # Prefer b64_json if available (more reliable than URL)
            if 'b64_json' in image_entry and image_entry['b64_json'] is not None:
                logger.info("Using base64 image data from FLUX API")
                # FLUX response usually provides base64 data
                try:
                    image_bytes = base64.b64decode(image_entry['b64_json'])

                    # Detect image format from magic bytes
                    if image_bytes.startswith(b'\xff\xd8\xff'):
                        # JPEG
                        ext = '.jpg'
                    elif image_bytes.startswith(b'\x89PNG'):
                        # PNG
                        ext = '.png'
                    elif image_bytes.startswith(b'RIFF'):
                        # WebP
                        ext = '.webp'
                    elif image_bytes.startswith(b'GIF8'):
                        # GIF
                        ext = '.gif'
                    else:
                        # Default to jpg but log warning
                        logger.warning(
                            f"Unknown image format, defaulting to .jpg")
                        ext = '.jpg'

                    # Update output path with correct extension
                    output_path_with_ext = output_path.rsplit('.', 1)[0] + ext

                    with open(output_path_with_ext, "wb") as f:
                        f.write(image_bytes)
                    logger.info(
                        f"Title card (base64, {ext[1:]}) saved to: {output_path_with_ext} ({len(image_bytes)} bytes)")
                    return output_path_with_ext
                except Exception as decode_error:
                    logger.error(
                        f"Failed to decode base64 image: {decode_error}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return None

            # Only try URL if it exists and is not None
            if 'url' in image_entry and image_entry.get('url') is not None:
                image_url = image_entry['url']
                logger.info(f"Downloading title card from URL: {image_url}")
                img_response = requests.get(image_url, timeout=30)

                if img_response.status_code != 200:
                    logger.error(
                        f"Failed to download title card: {img_response.status_code}")
                    logger.error(f"Response body: {img_response.text[:500]}")
                    return None

                # Validate content type
                content_type = img_response.headers.get(
                    'Content-Type', '').lower()
                if not content_type.startswith('image/'):
                    logger.error(
                        f"Invalid content type: {content_type}. Expected image/*")
                    logger.error(
                        f"Response body (first 500 chars): {img_response.text[:500]}")
                    return None

                # Validate it's actually image data by checking magic bytes
                content = img_response.content
                if len(content) < 4:
                    logger.error(
                        f"Downloaded content too small ({len(content)} bytes), not a valid image")
                    return None

                # Check for common image magic bytes
                is_image = (
                    content.startswith(b'\xff\xd8\xff') or  # JPEG
                    content.startswith(b'\x89PNG') or      # PNG
                    content.startswith(b'GIF8') or          # GIF
                    content.startswith(b'RIFF') or          # WebP
                    content.startswith(b'<svg')             # SVG (text-based)
                )

                if not is_image:
                    logger.error(
                        f"Downloaded content is not a valid image format")
                    logger.error(f"First 100 bytes: {content[:100]}")
                    logger.error(f"Content-Type header: {content_type}")
                    # Check if it's an XML error
                    if content.startswith(b'<?xml') or content.startswith(b'<Error'):
                        logger.error(
                            "FLUX API returned an XML error instead of image")
                    return None

                with open(output_path, "wb") as f:
                    f.write(content)

                logger.info(
                    f"Title card downloaded to: {output_path} ({len(content)} bytes)")
                return output_path

            logger.error(
                f"Unsupported image payload from Nebius: {image_entry}")
            return None
        except Exception as e:
            logger.error(f"Title card generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def analyze_scene_changes(self, video_path):
        """
        Use Gemini Flash Lite to identify timestamps where subject or scene changes.
        Returns list of timestamps to analyze for crop positions.
        """
        logger.info(
            "Scene analysis: Using fallback mode (Gemini disabled to avoid timeout)")

        # TEMPORARY: Skip Gemini entirely due to timeout issues
        # TODO: Re-enable Gemini analysis once timeout issue is resolved

        try:
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # Sample every 2 seconds, max 10 samples for better face tracking
            if duration > 0:
                interval = min(2.0, duration / 6)
                timestamps = [0.0]
                current = interval
                while current < duration and len(timestamps) < 10:
                    timestamps.append(round(current, 2))
                    current += interval
                logger.info(
                    f"Fallback: {len(timestamps)} keyframes at {interval:.1f}s intervals")
                return timestamps
            else:
                logger.warning("Could not determine duration, using defaults")
                return [0.0]
        except Exception as e:
            logger.warning(f"Fallback failed: {e}, using minimal timestamps")
            return [0, 5, 10]

        except Exception as e:
            logger.error(f"Gemini scene analysis failed: {e}")
            logger.info("Using fallback: analyzing frames at fixed intervals")

            # Fallback: Get video duration and sample at intervals
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()

                # Sample every 5 seconds, max 4 samples
                if duration > 0:
                    interval = min(5.0, duration / 4)
                    timestamps = [0.0]
                    current = interval
                    while current < duration and len(timestamps) < 4:
                        timestamps.append(round(current, 2))
                        current += interval
                    logger.info(
                        f"Fallback: {len(timestamps)} keyframes at {interval:.1f}s intervals")
                    return timestamps
                else:
                    logger.warning(
                        "Could not determine duration, using defaults")
                    return [0.0]
            except:
                logger.warning("Fallback failed, using minimal timestamps")
                return [0, 5, 10]

    def extract_frames_at_timestamps(self, video_path, timestamps):
        """
        Extract frames from video at specific timestamps.
        Returns list of frame file paths.
        """
        logger.info(f"Extracting {len(timestamps)} frames...")
        video = VideoFileClip(video_path)

        frame_paths = []
        os.makedirs(self.frames_dir, exist_ok=True)

        for timestamp in timestamps:
            try:
                # Ensure timestamp is within video duration
                if timestamp >= video.duration:
                    timestamp = video.duration - 0.1

                # Get frame at timestamp
                frame = video.get_frame(timestamp)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Save frame
                frame_path = os.path.join(
                    self.frames_dir, f"frame_{timestamp:.2f}.jpg")
                cv2.imwrite(frame_path, frame_bgr)

                frame_paths.append({
                    'timestamp': timestamp,
                    'path': frame_path
                })

                logger.info(f"  Extracted frame at {timestamp:.2f}s")
            except Exception as e:
                logger.error(f"Failed to extract frame at {timestamp}s: {e}")

        video.close()
        return frame_paths

    def get_crop_position_from_frame(self, frame_path):
        """
        Use Qwen VL to determine where subject is in frame.
        Returns x-position as normalized value (0.0 = left, 0.5 = center, 1.0 = right)
        """
        logger.info(f"Analyzing frame with Qwen VL: {frame_path}")

        try:
            # Read and encode image
            with open(frame_path, 'rb') as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Make API request
            response = requests.post(
                "https://api.studio.nebius.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.nebius_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """You are analyzing a PODCAST/INTERVIEW video frame. Your goal is to find the horizontal position of the SPEAKER'S FACE so we can crop to keep their face visible.

CRITICAL: Focus on FACES, not bodies! In podcasts, people's faces must stay in frame.

STEP-BY-STEP:
1. Look for HUMAN FACES in the frame (there may be 1 or 2 people)
2. If there's ONE person speaking or prominent: locate the CENTER of their FACE (between the eyes)
3. If there are TWO people: find the MIDPOINT between both faces
4. Measure the horizontal position of that point as a value from 0.0 to 1.0

POSITION GUIDE (left edge = 0.0, right edge = 1.0):
- 0.0-0.2: Face is on the LEFT side of frame
- 0.2-0.4: Face is LEFT of center
- 0.4-0.6: Face is roughly CENTERED
- 0.6-0.8: Face is RIGHT of center
- 0.8-1.0: Face is on the RIGHT side of frame

COMMON PODCAST LAYOUTS:
- Single host centered: 0.5
- Single host slightly left: 0.3-0.4
- Single host slightly right: 0.6-0.7
- Two hosts side-by-side: 0.5 (midpoint)
- Guest on left, host on right: 0.5 (midpoint between them)
- One person talking on far left: 0.15-0.25
- One person talking on far right: 0.75-0.85

WHAT TO LOOK FOR:
- Human faces (eyes, nose, mouth)
- Head position, not body/torso
- If person is turned slightly, still use face center
- Ignore backgrounds, logos, text overlays

Return ONLY a single decimal number (e.g., 0.42), nothing else."""
                                }
                            ]
                        }
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(
                    f"Nebius API error: {response.status_code} - {response.text}")
                logger.warning("⚠️ Using fallback: center position (0.5)")
                return 0.5  # Default to center

            result = response.json()

            # Log the full response for debugging
            logger.info(
                f"  📝 Raw Qwen VL response: {result['choices'][0]['message']['content']}")

            # Extract position from response
            position_text = result['choices'][0]['message']['content'].strip()

            # Parse to float
            x_position = float(position_text)

            # Clamp to valid range
            x_position = max(0.0, min(1.0, x_position))

            logger.info(f"  ✅ Subject position: {x_position:.2f}")
            return x_position

        except Exception as e:
            logger.error(f"Qwen VL analysis failed: {e}")
            logger.warning("⚠️ Using fallback: center position (0.5)")

            logger.info("Falling back to center position")
            return 0.5  # Default to center if analysis fails

    def build_crop_position_map(self, video_path):
        """
        Complete pipeline: Gemini finds timestamps, Qwen finds positions.
        Returns list of {timestamp, x_position} for interpolation.
        """
        logger.info("=" * 60)
        logger.info("BUILDING CROP POSITION MAP")
        logger.info("=" * 60)

        # Step 1: Gemini analyzes video for scene changes
        logger.info("Step 1: Gemini analyzing video for scene changes...")
        timestamps = self.analyze_scene_changes(video_path)
        logger.info(f"Found {len(timestamps)} key timestamps: {timestamps}")

        # Step 2: Extract frames
        logger.info("Step 2: Extracting frames...")
        frames = self.extract_frames_at_timestamps(video_path, timestamps)

        # Step 3: Qwen analyzes each frame for subject position
        logger.info("Step 3: Qwen analyzing each frame for subject position...")
        crop_positions = []

        for frame_data in frames:
            timestamp = frame_data['timestamp']
            frame_path = frame_data['path']

            x_position = self.get_crop_position_from_frame(frame_path)

            crop_positions.append({
                'timestamp': timestamp,
                'x_position': x_position
            })

        # Step 4: Smooth out outliers to prevent sudden jumps that cut off faces
        # If a position jumps more than 0.25 from neighbors, it's likely an error
        if len(crop_positions) >= 3:
            logger.info("Step 4: Smoothing outliers in position data...")
            for i in range(1, len(crop_positions) - 1):
                prev_pos = crop_positions[i-1]['x_position']
                curr_pos = crop_positions[i]['x_position']
                next_pos = crop_positions[i+1]['x_position']

                # Check if current position is an outlier
                avg_neighbors = (prev_pos + next_pos) / 2
                if abs(curr_pos - avg_neighbors) > 0.25:
                    logger.warning(
                        f"  Smoothing outlier at {crop_positions[i]['timestamp']}s: {curr_pos:.2f} -> {avg_neighbors:.2f}")
                    crop_positions[i]['x_position'] = avg_neighbors

        logger.info(
            f"Crop position map complete: {len(crop_positions)} keyframes")
        for pos in crop_positions:
            logger.info(
                f"  {pos['timestamp']:.1f}s -> x={pos['x_position']:.2f}")

        return crop_positions

    def apply_dynamic_crop(self, video_path, crop_positions):
        """
        Apply smooth dynamic crop that follows subject.
        Interpolates between keyframe positions.
        """
        logger.info("Applying dynamic crop with interpolation...")

        video = VideoFileClip(video_path)

        # Target dimensions for 9:16
        target_width = int(video.h * 9 / 16)

        # Ensure target width doesn't exceed video width
        if target_width > video.w:
            logger.warning(
                f"Video width ({video.w}) is less than target crop width ({target_width})")
            logger.info(
                "Video is already portrait or too narrow, skipping crop")
            return video

        def get_crop_x_at_time(t):
            """
            Get crop x-position at specific time.
            Uses linear interpolation between keyframes.
            """
            # Find surrounding keyframes
            before = None
            after = None

            for i, pos in enumerate(crop_positions):
                if pos['timestamp'] <= t:
                    before = pos
                if pos['timestamp'] > t and after is None:
                    after = pos
                    break

            # Determine x_position based on keyframes
            if before is None:
                # Before first keyframe, use first position
                x_normalized = crop_positions[0]['x_position']
            elif after is None:
                # After last keyframe, use last position
                x_normalized = before['x_position']
            else:
                # Interpolate between keyframes
                time_progress = (t - before['timestamp']) / \
                    (after['timestamp'] - before['timestamp'])
                x_normalized = before['x_position'] + time_progress * \
                    (after['x_position'] - before['x_position'])

            # Convert normalized position (0-1) to pixel coordinates
            subject_x = int(x_normalized * video.w)

            # Add safety margin for faces - keep subject more centered in crop
            # This prevents faces from being right at the edge of the frame
            # We bias the crop to give 60% of space on the "outside" and 40% toward center
            center_of_video = video.w // 2

            if subject_x < center_of_video:
                # Subject is on left side - give more room on left
                # Shift crop slightly left to keep face away from right edge
                safety_offset = int(target_width * 0.08)  # 8% padding
                crop_x1 = subject_x - target_width // 2 - safety_offset
            else:
                # Subject is on right side - give more room on right
                # Shift crop slightly right to keep face away from left edge
                safety_offset = int(target_width * 0.08)  # 8% padding
                crop_x1 = subject_x - target_width // 2 + safety_offset

            # Keep within video bounds
            crop_x1 = max(0, min(crop_x1, video.w - target_width))
            crop_x2 = crop_x1 + target_width

            return crop_x1, crop_x2

        # Apply time-varying crop using MoviePy 2.x method
        # Use crop method with time-based lambda
        from moviepy import vfx

        # Create a function that returns crop parameters at each time
        def make_frame(t):
            """Generate cropped frame at time t"""
            frame = video.get_frame(t)
            x1, x2 = get_crop_x_at_time(t)
            return frame[:, x1:x2, :]

        # Create new clip with custom frame function
        from moviepy import VideoClip
        cropped_video = VideoClip(make_frame, duration=video.duration)
        cropped_video = cropped_video.with_fps(video.fps)

        # Copy audio from original
        if video.audio:
            cropped_video = cropped_video.with_audio(video.audio)

        logger.info(
            f"Dynamic crop applied: {video.w}x{video.h} -> {target_width}x{video.h}")
        return cropped_video

    def smart_crop_pipeline(self, video_path):
        """
        Full pipeline: analyze, crop, return cropped video path.
        """
        logger.info("=" * 60)
        logger.info("SMART CROP PIPELINE")
        logger.info("=" * 60)

        try:
            # Build crop position map
            crop_positions = self.build_crop_position_map(video_path)

            # Apply dynamic crop
            logger.info("Step 4: Applying dynamic crop...")
            cropped_video = self.apply_dynamic_crop(video_path, crop_positions)

            # Save cropped video
            output_path = os.path.join(self.work_dir, "smart_cropped.mp4")
            logger.info("Step 5: Exporting cropped video...")
            cropped_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=30,
                write_logfile=False  # Prevent MoviePy from hanging
            )

            cropped_video.close()
            # Force garbage collection to free memory
            gc.collect()

            logger.info("=" * 60)
            logger.info("SMART CROP COMPLETE")
            logger.info("=" * 60)

            return output_path

        except Exception as e:
            logger.error(f"Smart crop pipeline failed: {e}")
            logger.info("Falling back to original video")
            gc.collect()  # Clean up even on failure
            return video_path  # Fallback to original

    def generate_intro_screen(self, title_text, mood):
        """
        Generate intro image with FLUX via Nebius.
        Returns path to generated image.
        """
        logger.info("Generating intro screen with FLUX...")

        if not self.nebius_api_key:
            logger.warning("NEBIUS_API_KEY missing, skipping intro image")
            return None

        try:
            # Enhanced prompts for better text rendering and style
            style_prompts = {
                "hype": f"High-energy social media intro card, vertical 9:16, large bold typography reading '{title_text}', vibrant neon gradients (cyan, magenta, electric blue), motion blur effects, modern streetwear aesthetic, 4k resolution, trending on tiktok, clean background",
                "suspense": f"Cinematic movie poster style intro, vertical 9:16, dramatic lighting, large serif typography reading '{title_text}', dark moody atmosphere, fog and shadow effects, high contrast, 8k resolution, thriller aesthetic",
                "chill": f"Aesthetic minimal intro card, vertical 9:16, elegant thin typography reading '{title_text}', soft pastel color palette (sage green, cream, dusty pink), lo-fi vibes, clean composition, high quality, pinterest style"
            }

            prompt = style_prompts.get(mood, style_prompts["hype"])

            response = requests.post(
                "https://api.studio.nebius.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.nebius_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "black-forest-labs/flux-schnell",
                    "prompt": prompt,
                    "width": 1080,
                    "height": 1920,
                    "num_inference_steps": 4,  # Max allowed is 16 for schnell
                    "response_format": "b64_json"  # Request base64 format for reliability
                }
            )

            if response.status_code != 200:
                logger.error(f"Nebius API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

            result = response.json()
            logger.info(f"FLUX API response structure: {list(result.keys())}")

            data = result.get('data', [])

            if not data:
                logger.error(f"Nebius API returned empty data: {result}")
                return None

            image_entry = data[0]
            logger.info(f"Image entry keys: {list(image_entry.keys())}")

            # Debug: Log the actual values
            if 'b64_json' in image_entry:
                b64_value = image_entry['b64_json']
                logger.info(f"b64_json value type: {type(b64_value)}")
                if b64_value is not None:
                    logger.info(
                        f"b64_json length: {len(b64_value)} characters")
                else:
                    logger.error("b64_json is None!")

            if 'url' in image_entry:
                url_value = image_entry['url']
                logger.info(
                    f"url value type: {type(url_value)}, value: {url_value}")

            output_path = os.path.join(self.work_dir, "intro_image.jpg")

            # Prefer b64_json if available (more reliable than URL)
            # Match the test script approach: check if key exists, then access directly
            if 'b64_json' in image_entry:
                logger.info(
                    "Found b64_json in response, attempting to decode...")
                b64_data = image_entry['b64_json']

                # Check if it's None or empty
                if b64_data is None:
                    logger.error("b64_json is None in response!")
                elif not isinstance(b64_data, str):
                    logger.error(
                        f"b64_json is not a string: type={type(b64_data)}")
                elif len(b64_data) == 0:
                    logger.error("b64_json is empty string!")
                else:
                    logger.info(
                        f"Using base64 image data from FLUX API (length: {len(b64_data)} characters)")
                    try:
                        # Decode base64 (same as test script)
                        image_bytes = base64.b64decode(b64_data)
                        logger.info(
                            f"Decoded image size: {len(image_bytes)} bytes")

                        # Detect image format from magic bytes (same as test script)
                        if image_bytes.startswith(b'\xff\xd8\xff'):
                            # JPEG
                            ext = '.jpg'
                        elif image_bytes.startswith(b'\x89PNG'):
                            # PNG
                            ext = '.png'
                        elif image_bytes.startswith(b'RIFF'):
                            # WebP
                            ext = '.webp'
                        elif image_bytes.startswith(b'GIF8'):
                            # GIF
                            ext = '.gif'
                        else:
                            # Default to jpg but log warning
                            logger.warning(
                                f"Unknown image format, defaulting to .jpg")
                            ext = '.jpg'

                        # Update output path with correct extension
                        output_path_with_ext = output_path.rsplit('.', 1)[
                            0] + ext

                        with open(output_path_with_ext, "wb") as f:
                            f.write(image_bytes)
                        logger.info(
                            f"Intro image (base64, {ext[1:]}) saved to: {output_path_with_ext} ({len(image_bytes)} bytes)")
                        return output_path_with_ext
                    except Exception as decode_error:
                        logger.error(
                            f"Failed to decode base64 intro image: {decode_error}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        return None
                    # b64_json exists but is None or invalid, fall through to URL check
            else:
                logger.warning(
                    "b64_json key not found in response, trying URL fallback...")

            # Only try URL if it exists and is not None
            if 'url' in image_entry and image_entry.get('url') is not None:
                image_url = image_entry['url']
                logger.info(f"Downloading intro image from URL: {image_url}")
                img_response = requests.get(image_url, timeout=30)

                if img_response.status_code != 200:
                    logger.error(
                        f"Failed to download intro image: {img_response.status_code}")
                    logger.error(f"Response body: {img_response.text[:500]}")
                    return None

                # Validate content type
                content_type = img_response.headers.get(
                    'Content-Type', '').lower()
                if not content_type.startswith('image/'):
                    logger.error(
                        f"Invalid content type: {content_type}. Expected image/*")
                    logger.error(
                        f"Response body (first 500 chars): {img_response.text[:500]}")
                    return None

                # Validate it's actually image data by checking magic bytes
                content = img_response.content
                if len(content) < 4:
                    logger.error(
                        f"Downloaded content too small ({len(content)} bytes), not a valid image")
                    return None

                # Check for common image magic bytes
                is_image = (
                    content.startswith(b'\xff\xd8\xff') or  # JPEG
                    content.startswith(b'\x89PNG') or      # PNG
                    content.startswith(b'GIF8') or          # GIF
                    content.startswith(b'RIFF') or          # WebP
                    content.startswith(b'<svg')             # SVG (text-based)
                )

                if not is_image:
                    logger.error(
                        f"Downloaded content is not a valid image format")
                    logger.error(f"First 100 bytes: {content[:100]}")
                    logger.error(f"Content-Type header: {content_type}")
                    # Check if it's an XML error
                    if content.startswith(b'<?xml') or content.startswith(b'<Error'):
                        logger.error(
                            "FLUX API returned an XML error instead of image")
                    return None

                with open(output_path, "wb") as f:
                    f.write(content)

                logger.info(
                    f"Intro image downloaded to: {output_path} ({len(content)} bytes)")
                return output_path

            # If we get here, neither b64_json nor url worked
            logger.error(
                f"Unsupported image payload from Nebius: {image_entry}")
            logger.error(f"  b64_json present: {'b64_json' in image_entry}")
            logger.error(f"  b64_json value: {image_entry.get('b64_json')}")
            logger.error(f"  url present: {'url' in image_entry}")
            logger.error(f"  url value: {image_entry.get('url')}")
            gc.collect()  # Clean up response data
            return None

        except Exception as e:
            import traceback
            logger.error(f"Intro image generation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            gc.collect()  # Clean up on error
            return None

    def create_intro_video_ffmpeg(self, intro_image_path, intro_audio_path, width, height):
        """
        Uses ffmpeg to create intro video from image + audio.
        More reliable than MoviePy for this specific task.
        """
        try:
            if not os.path.exists(intro_image_path):
                logger.error(f"Intro image not found: {intro_image_path}")
                return None

            if not os.path.exists(intro_audio_path):
                logger.warning(f"Intro audio not found: {intro_audio_path}")
                return None

            output_path = os.path.join(self.work_dir, "intro_video.mp4")

            # Ensure dimensions are even (libx264 requirement)
            width = width if width % 2 == 0 else width + 1
            height = height if height % 2 == 0 else height + 1

            # FFmpeg command to create video from image with audio
            # -loop 1: loop the image
            # -i image: input image
            # -i audio: input audio
            # -c:v libx264: video codec
            # -c:a aac: audio codec
            # -shortest: match shortest stream (audio duration)
            # -pix_fmt yuv420p: pixel format for compatibility
            # -vf scale: resize to match video dimensions
            cmd = [
                'ffmpeg', '-y',  # overwrite output
                '-loop', '1',
                '-i', intro_image_path,
                '-i', intro_audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                '-vf', f'scale={width}:{height}',
                output_path
            ]

            logger.info(f"Creating intro video with ffmpeg: {width}x{height}")
            logger.info(f"  Image: {intro_image_path}")
            logger.info(f"  Audio: {intro_audio_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg error creating intro video:")
                logger.error(f"  Return code: {result.returncode}")
                logger.error(f"  stderr: {result.stderr}")
                logger.error(f"  stdout: {result.stdout}")
                # Clean up empty file if it was created
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    os.remove(output_path)
                return None

            # Validate output file was created and is not empty
            if not os.path.exists(output_path):
                logger.error(
                    f"Intro video file was not created: {output_path}")
                return None

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                logger.error(
                    f"Intro video file is empty (0 bytes): {output_path}")
                os.remove(output_path)
                return None

            logger.info(
                f"Intro video created: {output_path} ({file_size} bytes)")
            return output_path

        except Exception as e:
            logger.error(f"Error creating intro video with ffmpeg: {e}")
            return None

    def get_word_level_transcription(self, audio_path):
        """
        Uses faster-whisper to get word-level timestamps.
        Returns list of dicts: {'text': 'WORD', 'start': 0.0, 'end': 0.5}
        """
        try:
            from faster_whisper import WhisperModel
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"Using device: {device} for transcription")

            # Load faster-whisper model
            model = WhisperModel("base", device=device,
                                 compute_type=compute_type)

            # Transcribe with word-level timestamps
            segments, info = model.transcribe(audio_path, word_timestamps=True)
            logger.info(
                f"Detected language: {info.language} with probability {info.language_probability:.2f}")

            # Collect all words with timestamps
            full_words = []
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        full_words.append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end
                        })

            logger.info(f"Transcribed {len(full_words)} words")

            # Process into word chunks for TikTok style (4-6 words per chunk)
            chunks = []
            current_chunk = []
            chunk_start = 0

            for i, word in enumerate(full_words):
                if not current_chunk:
                    chunk_start = word["start"]

                current_chunk.append(word["word"])

                # Chunk logic: every 5 words or if pause > 0.5s
                is_pause = False
                if i < len(full_words) - 1:
                    next_word = full_words[i + 1]
                    if next_word["start"] - word["end"] > 0.5:
                        is_pause = True

                if len(current_chunk) >= 5 or is_pause:
                    chunks.append({
                        "text": " ".join(current_chunk).upper(),
                        "start": chunk_start,
                        "end": word["end"],
                        "duration": word["end"] - chunk_start
                    })
                    current_chunk = []

            # Add remaining words
            if current_chunk and full_words:
                chunks.append({
                    "text": " ".join(current_chunk).upper(),
                    "start": chunk_start,
                    "end": full_words[-1]["end"],
                    "duration": full_words[-1]["end"] - chunk_start
                })

            logger.info(f"Created {len(chunks)} subtitle chunks")
            return chunks

        except ImportError as e:
            logger.error(
                f"faster-whisper not installed. Cannot generate subtitles. Error: {e}")
            return []
        except Exception as e:
            import traceback
            logger.error(f"Transcription failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def add_subtitles_to_video(self, video_clip, subtitle_chunks):
        """
        Burns subtitles into the video using MoviePy.
        """
        try:
            subtitle_clips = []

            # Use full path to Arial Bold on macOS
            font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            # Fallback to Helvetica if Arial not found
            if not os.path.exists(font_path):
                font_path = "/System/Library/Fonts/HelveticaNeue.ttc"

            logger.info(f"Using font: {font_path}")

            # Create TextClip for each chunk
            for chunk in subtitle_chunks:
                # Create text clip with stroke for visibility
                # MoviePy 2.x uses 'text=' keyword argument
                # Using 'label' method for single-line centered text, or 'caption' with horizontal_align
                txt_clip = TextClip(
                    text=chunk["text"],
                    font=font_path,
                    font_size=50,  # Slightly smaller for better fit
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    size=(int(video_clip.w * 0.9), None),  # 90% width
                    method="caption",
                    text_align="center",  # Center align the text
                    horizontal_align="center"  # Center horizontally within the clip
                ).with_position(("center", 0.80), relative=True).with_start(chunk["start"]).with_duration(chunk["duration"])

                subtitle_clips.append(txt_clip)

            logger.info(f"Created {len(subtitle_clips)} subtitle clips")

            # Composite
            final = CompositeVideoClip([video_clip] + subtitle_clips)
            return final

        except Exception as e:
            import traceback
            logger.error(f"Error burning subtitles: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return video_clip

    def compose_final(self, original_video, intro_audio, bg_music, title_text=None, mood="hype",
                      enable_smart_crop=False, add_intro_image=True, add_subtitles=True):
        """
        Complete composition with all production features.
        Integrates: smart crop + intro image + subtitles + voiceover + music

        Note: Subtitles are generated at the END of the pipeline using WhisperX
        on the complete video, so transcription includes the ElevenLabs intro voiceover.
        """
        logger.info(
            f"Composing final video. Smart Crop: {enable_smart_crop}, Intro Image: {add_intro_image}, Subtitles: {add_subtitles}")

        # Step 1: Apply smart crop if enabled
        video_path = original_video
        if enable_smart_crop:
            logger.info("🎯 Applying AI-powered smart crop...")
            video_path = self.smart_crop_pipeline(original_video)
            # Free memory after heavy smart crop operation
            gc.collect()
            logger.info("🧹 Memory cleaned after smart crop")

        video = VideoFileClip(video_path)

        # Step 2: Create Intro Video (if enabled)
        intro_video_path = None

        # Validate intro audio file
        if intro_audio and os.path.exists(intro_audio):
            file_size = os.path.getsize(intro_audio)
            if file_size == 0:
                logger.warning("Intro audio file is empty, skipping")
                intro_audio = None

        if add_intro_image and title_text:
            logger.info("🖼️ Generating intro with FLUX + ffmpeg...")
            try:
                intro_image_path = self.generate_intro_screen(title_text, mood)

                if intro_image_path and os.path.exists(intro_image_path):
                    # Validate image file is not empty
                    if os.path.getsize(intro_image_path) == 0:
                        logger.error(
                            f"Intro image file is empty: {intro_image_path}")
                        intro_image_path = None
                    else:
                        logger.info(
                            f"✅ Intro image validated: {intro_image_path} ({os.path.getsize(intro_image_path)} bytes)")

                    # Free memory after FLUX image generation before ffmpeg
                    gc.collect()
                    logger.info("🧹 Memory cleaned before intro video creation")

                    # Check if we have intro audio
                    if intro_image_path and intro_audio and os.path.exists(intro_audio):
                        # Use ffmpeg to create intro video (image + audio)
                        intro_video_path = self.create_intro_video_ffmpeg(
                            intro_image_path,
                            intro_audio,
                            video.size[0],  # width
                            video.size[1]   # height
                        )

                        if intro_video_path and os.path.exists(intro_video_path):
                            # Validate intro video file is not empty
                            file_size = os.path.getsize(intro_video_path)
                            if file_size == 0:
                                logger.error(
                                    f"Intro video file is empty (0 bytes): {intro_video_path}")
                                intro_video_path = None
                            else:
                                logger.info(
                                    f"✅ Intro video created: {intro_video_path} ({file_size} bytes)")
                        else:
                            logger.warning(
                                "Failed to create intro video with ffmpeg")
                            intro_video_path = None
                    else:
                        logger.warning(
                            "No intro audio found - intro image will be skipped")
                        # TODO: Could create silent intro video here if needed
                else:
                    logger.warning("Intro image generation failed")
            except Exception as e:
                logger.error(f"Intro generation failed: {e}")

        # Step 3: Use the processed video path (cropped or original)
        processed_video_path = video_path
        video.close()

        # Step 4: Concatenate intro + main video using ffmpeg filter (if intro exists)
        if intro_video_path and os.path.exists(intro_video_path):
            # Validate intro video file is valid (non-zero size)
            intro_file_size = os.path.getsize(intro_video_path)
            if intro_file_size == 0:
                logger.error(
                    f"Intro video file is empty (0 bytes), skipping concatenation: {intro_video_path}")
                intro_video_path = None
            else:
                logger.info(
                    f"Concatenating intro + main video with ffmpeg filter...")
                logger.info(
                    f"Intro video: {intro_video_path} ({intro_file_size} bytes)")
                try:
                    # Get main video dimensions for consistent output
                    cap = cv2.VideoCapture(processed_video_path)
                    main_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    main_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    # Ensure even dimensions
                    main_width = main_width if main_width % 2 == 0 else main_width + 1
                    main_height = main_height if main_height % 2 == 0 else main_height + 1

                    concatenated_path = os.path.join(
                        self.work_dir, "concatenated.mp4")
                    # Use filter_complex to scale both videos to same size and concat
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', intro_video_path,
                        '-i', processed_video_path,
                        '-filter_complex',
                        f'[0:v]scale={main_width}:{main_height},fps=30,format=yuv420p[v0];'
                        f'[1:v]scale={main_width}:{main_height},fps=30,format=yuv420p[v1];'
                        f'[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[outv][outa]',
                        '-map', '[outv]',
                        '-map', '[outa]',
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'fast',
                        concatenated_path
                    ]

                    result = subprocess.run(
                        cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        # Validate concatenated file was created and is not empty
                        if os.path.exists(concatenated_path) and os.path.getsize(concatenated_path) > 0:
                            logger.info(
                                f"✅ Videos concatenated: {concatenated_path} ({os.path.getsize(concatenated_path)} bytes)")
                            processed_video_path = concatenated_path
                        else:
                            logger.error(
                                f"Concatenated file is empty or missing: {concatenated_path}")
                            logger.warning(
                                "Falling back to main video without intro")
                    else:
                        logger.error(f"FFmpeg concat error: {result.stderr}")
                        logger.warning(
                            "Falling back to main video without intro")

                except Exception as e:
                    logger.error(f"Concatenation failed: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.warning("Continuing with main video only")

        # Step 5: Add background music using ffmpeg
        video_with_music_path = os.path.join(
            self.work_dir, "video_with_music.mp4")

        if bg_music and os.path.exists(bg_music):
            logger.info("🎵 Adding background music with ffmpeg...")
            try:
                # Use ffmpeg to mix background music with video audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', processed_video_path,
                    '-stream_loop', '-1',  # loop music
                    '-i', bg_music,
                    '-filter_complex', '[1:a]volume=0.15[music];[0:a][music]amix=inputs=2:duration=first[aout]',
                    '-map', '0:v',
                    '-map', '[aout]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    video_with_music_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(
                        f"✅ Background music added: {video_with_music_path}")
                else:
                    logger.error(f"FFmpeg music error: {result.stderr}")
                    logger.warning("Saving video without background music")
                    import shutil
                    shutil.copy(processed_video_path, video_with_music_path)

            except Exception as e:
                logger.error(f"Background music addition failed: {e}")
                logger.warning("Saving video without background music")
                import shutil
                shutil.copy(processed_video_path, video_with_music_path)
        else:
            # No background music, just copy the processed video
            logger.info("No background music, finalizing video...")
            import shutil
            shutil.copy(processed_video_path, video_with_music_path)

        # Step 6: Add subtitles using WhisperX on the COMPLETE video (including ElevenLabs intro)
        output_path = os.path.join(self.work_dir, "polished_output.mp4")

        if add_subtitles:
            logger.info(
                "📝 Generating subtitles with WhisperX on complete video...")
            try:
                # Extract audio from complete video for transcription
                audio_path = os.path.join(self.work_dir, "audio_for_subs.mp3")
                complete_video = VideoFileClip(video_with_music_path)
                complete_video.audio.write_audiofile(audio_path)

                # Get word-level transcription (includes ElevenLabs intro voiceover)
                subtitle_chunks = self.get_word_level_transcription(audio_path)

                if subtitle_chunks:
                    logger.info(
                        f"Generated {len(subtitle_chunks)} subtitle chunks")
                    # Burn subtitles into video
                    video_with_subs = self.add_subtitles_to_video(
                        complete_video, subtitle_chunks)

                    # Save final video with subtitles
                    video_with_subs.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        fps=30,
                        write_logfile=False
                    )
                    video_with_subs.close()
                    complete_video.close()
                else:
                    logger.warning(
                        "No subtitles generated, continuing without subtitles")
                    complete_video.close()
                    import shutil
                    shutil.copy(video_with_music_path, output_path)

            except Exception as e:
                logger.error(
                    f"Subtitle generation failed: {e}, continuing without subtitles")
                import shutil
                shutil.copy(video_with_music_path, output_path)
        else:
            # No subtitles requested, use video with music as final
            import shutil
            shutil.copy(video_with_music_path, output_path)

        logger.info(f"Final video exported to: {output_path}")
        return output_path
