import gradio as gr
import os
import tempfile
import shutil
import logging
import json
import time
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from dotenv import load_dotenv

from src.paths import OUTPUT_DIR, ensure_runtime_dirs

# Patch Gradio's hostname whitelist to allow localhost for MCP
# This fixes "Hostname localhost failed validation" error when MCP calls local endpoints
try:
    from gradio import processing_utils
    if hasattr(processing_utils, 'PUBLIC_HOSTNAME_WHITELIST'):
        if 'localhost' not in processing_utils.PUBLIC_HOSTNAME_WHITELIST:
            processing_utils.PUBLIC_HOSTNAME_WHITELIST.extend(
                ['localhost', '127.0.0.1', '0.0.0.0'])
            print(
                "✅ Added localhost to Gradio's hostname whitelist for MCP compatibility")
except Exception as e:
    print(f"⚠️ Could not patch hostname whitelist: {e}")

# Import local modules
from src.scout import SignalScout
from src.verifier import Verifier
from src.director import Director
from src.hands import Hands
from src.utils import download_audio, download_video_segment, get_video_info, extract_video_id
from src.server import get_transcript
from src.showrunner import Showrunner

# Load environment variables
load_dotenv()
ensure_runtime_dirs()

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Components
scout = SignalScout()
verifier = Verifier()
director = Director()
hands = Hands()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VIDEO_API_KEY = os.getenv("VIDEO_API_KEY")

# Initial configuration with default key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def overlaps_bad_audio(start: float, end: float, bad_segments: List[Dict]) -> bool:
    """Check if a time range overlaps significantly with bad audio segments."""
    for bad in bad_segments:
        # Check for intersection
        bad_start = bad['start']
        bad_end = bad['end']

        overlap_start = max(start, bad_start)
        overlap_end = min(end, bad_end)

        if overlap_end > overlap_start:
            overlap_duration = overlap_end - overlap_start
            segment_duration = end - start

            # If overlap is more than 20% of the segment or > 2 seconds, reject it
            if overlap_duration > 2.0 or (overlap_duration / segment_duration) > 0.2:
                return True
    return False


def classify_video(video_info: Dict) -> str:
    """
    Classify video as 'podcast' or 'generic' based on metadata.
    Uses uploader, title, description, duration, and tags.
    """
    title = video_info.get('title', '').lower()
    description = video_info.get('description', '').lower()
    uploader = video_info.get('uploader', '').lower()
    channel = video_info.get('channel', '').lower()
    duration = video_info.get('duration', 0)
    tags = [tag.lower() for tag in video_info.get('tags', [])]

    # Known podcast channels/uploaders (high confidence)
    podcast_channels = [
        'joe rogan', 'powerfuljre', 'jre clips',
        'lex fridman', 'lex clips',
        'huberman lab', 'andrew huberman',
        'all-in podcast', 'all-in pod',
        'diary of a ceo',
        'impact theory',
        'tim ferriss',
        'smartless',
        'ted talks', 'ted',
        'flagrant', 'flagrant 2'
    ]

    # Check uploader/channel first (most reliable)
    for pc in podcast_channels:
        if pc in uploader or pc in channel:
            logger.info(f"Classified as PODCAST via channel match: {uploader}")
            return "podcast"

    # Podcast keywords - check title, description, tags
    podcast_keywords = [
        'podcast', 'interview', 'talk show', 'conversation',
        'episode', 'ep ', '#ep', 'hosted by', 'discussion',
        'with guest', 'full episode', 'guest on', 'sits down with',
        'in conversation', 'chat with'
    ]

    # Generic/Tutorial keywords that override podcast classification
    generic_keywords = [
        'tutorial', 'how to', 'guide', 'demo', 'review',
        'unboxing', 'gameplay', 'walkthrough', 'diy',
        'explained', 'breakdown', 'analysis of'
    ]

    # Check for generic keywords first (but only in title - avoid false positives)
    for keyword in generic_keywords:
        if keyword in title:
            logger.info(f"Classified as GENERIC via keyword: {keyword}")
            return "generic"

    # Score podcast signals
    podcast_score = 0
    matched_keywords = []

    for keyword in podcast_keywords:
        if keyword in title:
            podcast_score += 3
            matched_keywords.append(f"{keyword}(title)")
        if keyword in description[:500]:
            podcast_score += 1
            matched_keywords.append(f"{keyword}(desc)")
        if keyword in ' '.join(tags):
            podcast_score += 1
            matched_keywords.append(f"{keyword}(tag)")

    # Long videos (>15 min) with podcast indicators
    if duration > 900:  # 15+ minutes
        if podcast_score >= 2:
            logger.info(
                f"Classified as PODCAST via long+keywords: {matched_keywords}")
            return "podcast"

    # Strong podcast signals in title alone
    if podcast_score >= 3:
        logger.info(
            f"Classified as PODCAST via strong signals: {matched_keywords}")
        return "podcast"

    # Default to generic
    logger.info(
        f"Classified as GENERIC (score={podcast_score}, duration={duration}s)")
    return "generic"


def extract_transcript_window(transcript_text: str, start: float, end: float, buffer: float = 10) -> str:
    """Extract a window of transcript around given timestamps."""
    if not transcript_text:
        return "[No transcript available]"

    lines = transcript_text.strip().split('\n')
    relevant_lines = []

    for line in lines:
        match = re.match(r'\[(\d+):(\d+)\]', line)
        if match:
            mins, secs = int(match.group(1)), int(match.group(2))
            line_time = mins * 60 + secs
            if (start - buffer) <= line_time <= (end + buffer):
                relevant_lines.append(line)

    return '\n'.join(relevant_lines) if relevant_lines else "[No transcript available for this segment]"


def analyze_transcript_with_llm(transcript_text: str, video_info: Dict) -> List[Dict]:
    """
    Use Gemini to analyze transcript for viral-worthy moments.
    Returns list of hotspots with semantic scores and reasoning.
    """
    if not transcript_text or not GEMINI_API_KEY:
        return []

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Split transcript into chunks (60 sec windows with 10 sec overlap)
        lines = transcript_text.strip().split('\n')
        chunks = []
        current_chunk = []
        chunk_start = 0

        for line in lines:
            match = re.match(r'\[(\d+):(\d+)\]', line)
            if match:
                mins, secs = int(match.group(1)), int(match.group(2))
                timestamp = mins * 60 + secs

                if not current_chunk:
                    chunk_start = timestamp

                current_chunk.append(line)

                # Create chunk every 60 seconds
                if timestamp - chunk_start >= 60:
                    chunks.append({
                        'start': chunk_start,
                        'end': timestamp,
                        'text': '\n'.join(current_chunk)
                    })
                    # Overlap: keep last 10 seconds
                    overlap_lines = [
                        l for l in current_chunk if re.match(r'\[(\d+):(\d+)\]', l)]
                    current_chunk = overlap_lines[-3:] if len(
                        overlap_lines) > 3 else []
                    chunk_start = timestamp - 10

        # Add remaining chunk
        if current_chunk:
            last_match = re.match(r'\[(\d+):(\d+)\]', current_chunk[-1])
            if last_match:
                end_time = int(last_match.group(1)) * 60 + \
                    int(last_match.group(2))
                chunks.append({
                    'start': chunk_start,
                    'end': end_time,
                    'text': '\n'.join(current_chunk)
                })

        semantic_hotspots = []

        for chunk in chunks[:20]:  # Limit to first 20 chunks
            prompt = f"""You are analyzing a transcript excerpt from "{video_info.get('title', 'a video')}" to find VIRAL-WORTHY moments.

TRANSCRIPT EXCERPT (timestamps in [MM:SS] format):
{chunk['text']}

TASK: Rate this excerpt's VIRAL POTENTIAL on a scale of 1-10 based on:

HIGH VIRAL POTENTIAL (8-10):
- Controversial or debate-worthy statements
- Surprising revelations or confessions  
- Emotional peaks (anger, vulnerability, excitement)
- Quotable one-liners or memorable phrases
- Strong opinions stated with conviction

MEDIUM POTENTIAL (5-7):
- Interesting insights or unique perspectives
- Mild humor or wit
- Engaging storytelling moments

LOW POTENTIAL (1-4):
- Generic conversation, filler, or transitions
- Repeated information
- Context-setting without payoff

Respond in JSON format:
{{
    "viral_score": <1-10>,
    "reasoning": "<why this score>",
    "peak_moment": "<most shareable quote or moment if score >= 6>",
    "emotion": "<primary emotion: humor/controversy/revelation/passion/vulnerability/none>"
}}"""

            try:
                response = model.generate_content(prompt)
                text = response.text.strip()

                # Parse JSON from response
                json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    score = result.get('viral_score', 0)

                    if score >= 6:
                        semantic_hotspots.append({
                            'start': chunk['start'],
                            'end': chunk['end'],
                            'score': score / 10.0,  # Normalize to 0-1
                            'semantic_score': score,
                            'reasoning': result.get('reasoning', ''),
                            'peak_moment': result.get('peak_moment', ''),
                            'emotion': result.get('emotion', 'none'),
                            'source': 'semantic'
                        })
            except Exception as e:
                logger.warning(f"Chunk analysis failed: {e}")
                continue

            time.sleep(0.3)  # Rate limiting

        return semantic_hotspots

    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        return []


def deep_verify_clip(clip_path: str) -> float:
    """
    Uploads a clip to Gemini 2.0 Flash Exp and asks for a viral potential score.
    Uses VIDEO_API_KEY.
    """
    if not VIDEO_API_KEY:
        logger.warning("VIDEO_API_KEY not set. Skipping deep verification.")
        return 5.0

    try:
        # Switch to VIDEO_API_KEY
        genai.configure(api_key=VIDEO_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')

        logger.info(f"Uploading {clip_path} for deep verification...")
        video_file = genai.upload_file(path=clip_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            logger.error("Video processing failed.")
            return 0.0

        prompt = "Rate viral potential 1-10 based on visual motion and interest. Return ONLY the number."
        response = model.generate_content([video_file, prompt])

        try:
            score = float(response.text.strip())
            return score
        except ValueError:
            logger.warning(f"Could not parse score from: {response.text}")
            return 5.0

    except Exception as e:
        logger.error(f"Deep verification failed: {e}")
        return 5.0
    finally:
        # Restore GEMINI_API_KEY for other components
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)


# ============================================================================
# ENHANCED VERIFICATION (Video Upload - Parallel & Thread-Safe)
# ============================================================================

def verify_clip_video_upload(clip_path: str, hotspot: Dict) -> Dict:
    """
    Upload video clip directly to Gemini for holistic analysis.
    Falls back to multi-frame if upload fails.
    """
    if not VIDEO_API_KEY:
        logger.warning(
            "No VIDEO_API_KEY, falling back to frame-based verification")
        return verify_clip_multiframe_fallback(clip_path, hotspot)

    try:
        genai.configure(api_key=VIDEO_API_KEY)

        # Upload video file
        video_file = genai.upload_file(path=clip_path)

        # Wait for processing (with timeout)
        timeout = 60
        start_time = time.time()
        while video_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                logger.warning(
                    "Video processing timeout, falling back to frames")
                return verify_clip_multiframe_fallback(clip_path, hotspot)
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            return verify_clip_multiframe_fallback(clip_path, hotspot)

        # Analyze with Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """Analyze this video clip for viral/engagement potential.

Rate on a scale of 1-10:
1. VISUAL QUALITY: Lighting, framing, focus, production value
2. ENGAGEMENT: Does this hold attention? Is there movement, emotion, or visual interest?
3. AUDIO-VISUAL SYNC: Does the energy of visuals match what's being said?
4. SHAREABILITY: Would someone share this clip?

Respond in JSON:
{
    "visual_quality": <1-10>,
    "engagement": <1-10>,
    "audio_visual_sync": <1-10>,
    "shareability": <1-10>,
    "overall_score": <1-10>,
    "reasoning": "<brief explanation>",
    "visual_interest": "<what makes this visually interesting or not>"
}"""

        response = model.generate_content([video_file, prompt])
        text = response.text.strip()

        # Restore global API key
        genai.configure(api_key=GEMINI_API_KEY)

        # Clean up uploaded file
        try:
            genai.delete_file(video_file.name)
        except:
            pass

        # Parse response
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'verified': result.get('overall_score', 5) >= 5,
                'score': result.get('overall_score', 5),
                'visual_quality': result.get('visual_quality', 5),
                'engagement': result.get('engagement', 5),
                'reasoning': result.get('reasoning', ''),
                'visual_interest': result.get('visual_interest', '')
            }

    except Exception as e:
        logger.error(f"Video upload verification failed: {e}")
        genai.configure(api_key=GEMINI_API_KEY)

    return verify_clip_multiframe_fallback(clip_path, hotspot)


def verify_clip_multiframe_fallback(clip_path: str, hotspot: Dict) -> Dict:
    """Fallback: Extract frames and verify with vision AI."""
    try:
        # Use the existing verifier
        verification = verifier.verify(clip_path, hotspot)
        return {
            'verified': verification.get('keep', False),
            'score': verification.get('visual_score', 5),
            'reasoning': verification.get('analysis', ''),
            'visual_interest': verification.get('visual_interest', '')
        }
    except Exception as e:
        logger.error(f"Frame verification failed: {e}")
        return {'verified': True, 'score': 5, 'reasoning': 'Verification failed, keeping clip'}


def verify_single_clip_thread_safe(clip_path: str, hotspot: Dict, api_key: str) -> Tuple[int, Dict]:
    """Thread-safe verification - creates its own API configuration."""
    try:
        genai.configure(api_key=api_key)
        result = verify_clip_video_upload(clip_path, hotspot)
        genai.configure(api_key=GEMINI_API_KEY)  # Restore
        return (hotspot.get('index', 0), result)
    except Exception as e:
        logger.error(f"Thread-safe verification failed: {e}")
        genai.configure(api_key=GEMINI_API_KEY)
        return (hotspot.get('index', 0), {'verified': True, 'score': 5})


def verify_clips_parallel(clips_to_verify: List[Dict], api_key: str, max_workers: int = 4) -> List[Dict]:
    """Verify multiple clips in parallel."""
    if not clips_to_verify:
        return []

    logger.info(
        f"🔍 Starting parallel verification of {len(clips_to_verify)} clips...")
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, clip in enumerate(clips_to_verify):
            if i > 0:
                time.sleep(0.5)  # Rate limit protection
            clip['hotspot']['index'] = clip['index']
            future = executor.submit(
                verify_single_clip_thread_safe,
                clip['path'],
                clip['hotspot'],
                api_key
            )
            futures[future] = clip

        for future in as_completed(futures):
            clip = futures[future]
            try:
                idx, verification = future.result()
                results.append({
                    'index': clip['index'],
                    'path': clip['path'],
                    'hotspot': clip['hotspot'],
                    'verification': verification,
                    'start': clip['start'],
                    'end': clip['end']
                })
            except Exception as e:
                logger.error(
                    f"Verification failed for clip {clip['index']}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"🚀 {len(results)} clips verified in {elapsed:.1f}s")
    return results


# ============================================================================
# PARALLEL DOWNLOAD
# ============================================================================

def download_clips_parallel(download_tasks: List[Dict], max_workers: int = 4) -> List[Dict]:
    """Download multiple clips in parallel using ThreadPoolExecutor."""
    if not download_tasks:
        return []

    logger.info(
        f"📥 Starting parallel download of {len(download_tasks)} clips with {max_workers} workers...")
    start_time = time.time()

    def download_single(task):
        """Download a single clip."""
        try:
            clip_path = download_video_segment(
                task['url'],
                task['start'],
                task['end'],
                task['path'],
                format_spec='best'
            )
            return {
                'success': True,
                'path': clip_path,
                'hotspot': task['hotspot'],
                'start': task['start'],
                'end': task['end'],
                'index': task['index']
            }
        except Exception as e:
            logger.error(f"  Download failed for clip {task['index']}: {e}")
            return {
                'success': False,
                'error': str(e),
                'index': task['index']
            }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single, task)                   : task for task in download_tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Download task failed: {e}")

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.get('success'))
    logger.info(
        f"📥 {success_count}/{len(download_tasks)} clips downloaded in {elapsed:.1f}s")

    return results


def map_plan_to_clips(plan: List[Dict], clips_metadata: List[Dict]) -> List[Dict]:
    """
    Maps Director's plan (start/end times) to actual downloaded clip files.

    FLEXIBLE MATCHING: If the plan's START time falls within a clip, we use that clip
    and adjust the END time to fit within clip boundaries. This handles cases where
    the Director requests longer segments than the downloaded clips allow.
    """
    mapped_plan = []
    for item in plan:
        item_start = item.get('start')
        item_end = item.get('end')

        if item_start is None or item_end is None:
            logger.warning(f"Plan item missing start/end: {item}")
            continue

        best_match = None
        best_overlap = 0

        for clip in clips_metadata:
            clip_start = clip['start']
            clip_end = clip['end']

            # Check if plan's START time falls within this clip (with 1s tolerance)
            if clip_start - 1.0 <= item_start <= clip_end + 1.0:
                # Calculate how much of the requested segment this clip can provide
                usable_start = max(item_start, clip_start)
                usable_end = min(item_end, clip_end)
                overlap = usable_end - usable_start

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = clip

        if best_match and best_overlap >= 3.0:  # Minimum 3 seconds usable
            clip_start = best_match['start']
            clip_end = best_match['end']

            # Adjust plan times to fit within clip boundaries
            adjusted_start = max(item_start, clip_start)
            adjusted_end = min(item_end, clip_end)

            if adjusted_end != item_end or adjusted_start != item_start:
                logger.info(
                    f"Adjusted plan [{item_start:.1f}-{item_end:.1f}] to [{adjusted_start:.1f}-{adjusted_end:.1f}] (clip: [{clip_start:.1f}-{clip_end:.1f}])")
            else:
                logger.info(
                    f"Mapped plan [{item_start:.1f}-{item_end:.1f}] to clip [{clip_start:.1f}-{clip_end:.1f}]")

            item['start'] = adjusted_start
            item['end'] = adjusted_end
            item['clip_path'] = best_match['path']
            item['source_start'] = best_match['start']
            mapped_plan.append(item)
        else:
            logger.warning(
                f"⚠️ Could not map plan segment [{item_start:.1f}s-{item_end:.1f}s] - no clip with enough overlap")
            logger.warning(
                f"Available clips: {[(c['start'], c['end']) for c in clips_metadata]}")

    return mapped_plan


def pipeline_podcast(video_url: str, video_info: Dict, temp_dir: str, progress=gr.Progress()) -> str:
    """
    Podcast Pipeline: Semantic-heavy, audio used as filter.
    """
    progress(0.1, desc="Fetching transcript...")

    # 1. Get Transcript & Semantic Analysis (PRIMARY)
    transcript_text = get_transcript(video_url, include_timestamps=True)
    semantic_hotspots = []

    triggers = [
        "unpopular opinion", "controversial take", "here's the secret",
        "nobody talks about", "breakthrough moment", "funniest thing",
        "best advice", "biggest mistake", "game changer", "mind blowing",
        "never told anyone", "listen to this", "key takeaway",
        "bottom line is", "reason why", "truth is", "i realized"
    ]

    if "Error" not in transcript_text:
        lines = transcript_text.split('\n')
        for i, line in enumerate(lines):
            for trigger in triggers:
                if trigger in line.lower():
                    try:
                        time_part = line.split(']')[0].strip('[')
                        parts = list(map(int, time_part.split(':')))
                        if len(parts) == 2:
                            seconds = parts[0] * 60 + parts[1]
                        elif len(parts) == 3:
                            seconds = parts[0] * 3600 + \
                                parts[1] * 60 + parts[2]
                        else:
                            continue

                        # Extract context (2 lines before, 2 lines after)
                        start_idx = max(0, i - 2)
                        end_idx = min(len(lines), i + 4)
                        context_lines = lines[start_idx:end_idx]
                        clean_context = " ".join(
                            [l.split('] ', 1)[1] if '] ' in l else l for l in context_lines])

                        semantic_hotspots.append({
                            'start': float(seconds),
                            'end': float(seconds + 15),
                            'score': 0.95,
                            'type': 'semantic_trigger',
                            'trigger': trigger,
                            'context': clean_context
                        })
                    except:
                        pass

    # 2. Scout Audio & Filter (SECONDARY)
    progress(0.2, desc="Scouting audio quality...")
    audio_path = os.path.join(temp_dir, "audio")
    actual_audio_path = download_audio(video_url, audio_path)

    bad_segments = scout.detect_bad_audio(actual_audio_path)

    # Filter
    filtered_hotspots = [
        h for h in semantic_hotspots
        if not overlaps_bad_audio(h['start'], h['end'], bad_segments)
    ]

    # If no semantic, fallback to audio peaks
    if not filtered_hotspots:
        logger.warning(
            "No semantic hotspots found for podcast pipeline. Fallback to audio peaks.")
        filtered_hotspots = scout._analyze_audio(actual_audio_path, top_n=8)

    filtered_hotspots.sort(key=lambda x: x['score'], reverse=True)
    candidates = filtered_hotspots[:8]

    # 4. Verify (Download clips -> Verifier)
    progress(0.4, desc="Verifying clips...")
    verified_hotspots = []
    clips_metadata = []

    for i, h in enumerate(candidates):
        # Download short clip for verification (and potential use)
        start = max(0, h['start'] - 2)
        end = h['end'] + 2
        path = os.path.join(temp_dir, f"pod_clip_{i}")

        try:
            clip_path = download_video_segment(
                video_url, start, end, path, format_spec='best')  # Best for final use

            # Verify using local file
            clip_offset = h['start'] - start
            res = verifier.verify(clip_path, clip_offset)

            if res.get('score', 0) > 4:
                h['score'] = (h['score'] + (res.get('score', 0)/10.0)) / 2
                verified_hotspots.append(h)
                clips_metadata.append({
                    'start': start,
                    'end': end,
                    'path': clip_path,
                    'hotspot': h
                })
        except Exception as e:
            logger.error(f"Verification failed for candidate {i}: {e}")

    # 5. Director
    progress(0.6, desc="Director planning...")
    plan = director.create_edit_plan(verified_hotspots, video_info)

    # 6. Path Mapping
    final_plan = map_plan_to_clips(plan, clips_metadata)

    # 7. Hands
    progress(0.8, desc="Rendering...")
    if not final_plan:
        return None

    output_path = hands.execute(
        clips_metadata[0]['path'], final_plan, output_filename="podcast_edit.mp4")
    return output_path


def pipeline_generic(video_url: str, video_info: Dict, temp_dir: str, progress=gr.Progress()) -> str:
    """
    Generic Pipeline: Full multimodal analysis with deep verification.
    """
    # 1. Scout & Transcript (if available)
    progress(0.1, desc="Scouting...")
    audio_path = os.path.join(temp_dir, "audio")
    actual_audio_path = download_audio(video_url, audio_path)
    hotspots = scout.analyze(actual_audio_path)

    # Check for transcript triggers too
    transcript_text = get_transcript(video_url, include_timestamps=True)
    triggers = ["let me show you", "check this out", "watch this",
                "before and after", "transformation", "can't believe"]

    if "Error" not in transcript_text:
        lines = transcript_text.split('\n')
        for line in lines:
            for trigger in triggers:
                if trigger in line.lower():
                    # Add high score hotspot
                    try:
                        time_part = line.split(']')[0].strip('[')
                        parts = list(map(int, time_part.split(':')))
                        if len(parts) == 2:
                            sec = parts[0]*60 + parts[1]
                        elif len(parts) == 3:
                            sec = parts[0]*3600 + parts[1]*60 + parts[2]
                        else:
                            continue

                        hotspots.append({
                            'start': float(sec),
                            'end': float(sec + 10),
                            'score': 0.9,
                            'type': 'visual_trigger'
                        })
                    except:
                        pass

    # 2. Select Candidates (Top 5)
    hotspots.sort(key=lambda x: x['score'], reverse=True)
    top_hotspots = hotspots[:5]

    # 3. Download Clips
    progress(0.3, desc="Downloading candidates...")
    clips_metadata = []
    for i, h in enumerate(top_hotspots):
        start = max(0, h['start'] - 5)
        end = h['end'] + 5
        path = os.path.join(temp_dir, f"gen_clip_{i}")
        try:
            clip_path = download_video_segment(
                video_url, start, end, path, format_spec='best')
            clips_metadata.append({
                'start': start,
                'end': end,
                'path': clip_path,
                'hotspot': h
            })
        except Exception as e:
            logger.error(f"Failed to download candidate {i}: {e}")

    # 4. Deep Verification
    progress(0.5, desc="Deep verification (Gemini)...")
    high_quality_clips = []
    for clip in clips_metadata:
        score = deep_verify_clip(clip['path'])
        logger.info(f"Clip {clip['path']} score: {score}")
        if score > 7:
            clip['hotspot']['score'] = score / 10.0
            high_quality_clips.append(clip)

    if not high_quality_clips:
        logger.warning("No high quality clips found. Using top 2.")
        high_quality_clips = clips_metadata[:2]

    # 5. Director
    progress(0.7, desc="Director planning...")
    hq_hotspots = [c['hotspot'] for c in high_quality_clips]
    plan = director.create_edit_plan(hq_hotspots, video_info)

    # 6. Path Mapping
    final_plan = map_plan_to_clips(plan, high_quality_clips)

    # 7. Hands
    progress(0.9, desc="Rendering...")
    if not final_plan:
        return None

    output_path = hands.execute(
        high_quality_clips[0]['path'], final_plan, output_filename="generic_edit.mp4")
    return output_path


@gr.mcp.tool()
def process_video(url: str) -> str:
    """
    Process a YouTube video through the full autonomous editing pipeline.

    This is the main entry point for Director's Cut - an AI-powered autonomous video editor
    that transforms long-form YouTube videos into viral short-form content (30-60 seconds).

    **Pipeline Overview:**
    1. **Classification**: Automatically determines if video is "podcast" or "generic"
       - Podcast: Uses transcript semantic analysis to find interesting quotes/moments
       - Generic: Uses audio peaks + visual triggers + deep AI verification

    2. **Scout Phase**: Finds "hotspots" (interesting moments) using:
       - Audio analysis (loudness, energy peaks)
       - Transcript analysis (semantic triggers like "unpopular opinion", "game changer")
       - Bad audio filtering (removes silence/noise segments)

    3. **Verification Phase**: Downloads short clips and verifies quality:
       - Podcast: Standard vision AI verification
       - Generic: Deep verification using Gemini 2.0 Flash (viral potential scoring)

    4. **Director Phase**: AI creates an edit plan selecting 3-5 best clips (8-15s each)

    5. **Hands Phase**: Renders final video by concatenating selected clips

    **Architecture Components:**
    - Scout (SignalScout): Finds interesting moments via signal processing
    - Verifier: Validates clip quality with vision AI
    - Director: Creates edit plan using Gemini 2.0 Flash Lite
    - Hands: Executes the edit plan with MoviePy

    **When to Use:**
    - For creating viral short-form content from long videos
    - When you want fully autonomous editing (no manual selection)
    - For both podcast-style interviews and generic/tutorial videos

    Args:
        url: Full YouTube video URL (e.g., "https://youtube.com/watch?v=...")
            Must be a valid, accessible YouTube video URL.

    Returns:
        String containing success message with output file path, or error message if failed.
        Output video is saved to the runtime output directory
        (default `/tmp/directors-cut/output/final_{timestamp}.mp4`)
        Video duration: 30-60 seconds, optimized for TikTok/Instagram Reels/YouTube Shorts.

    Example:
        process_video("https://youtube.com/watch?v=dQw4w9WgXcQ")
        Returns: "Success! Processed as podcast. Output: /tmp/directors-cut/output/final_1234567890.mp4"
    """
    temp_dir = tempfile.mkdtemp()
    try:
        logger.info(f"Processing {url} in {temp_dir}")

        video_info = get_video_info(url)
        category = classify_video(video_info)
        logger.info(f"Classified as: {category}")

        # Create a dummy progress function for MCP context
        class DummyProgress:
            def __call__(self, *args, **kwargs):
                pass

        progress = DummyProgress()

        if category == "podcast":
            result = pipeline_podcast(url, video_info, temp_dir, progress)
        else:
            result = pipeline_generic(url, video_info, temp_dir, progress)

        if result and os.path.exists(result):
            final_path = os.path.join(
                OUTPUT_DIR, f"final_{int(time.time())}.mp4")
            shutil.copy(result, final_path)
            workflow_state['final_video_path'] = final_path
            return f"Success! Processed as {category}. Output: {final_path}"
        else:
            workflow_state['final_video_path'] = None
            return "Failed to generate video."

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        workflow_state['final_video_path'] = None
        return f"Error: {e}"
    finally:
        shutil.rmtree(temp_dir)

# ========================================
# STEP-BY-STEP WORKFLOW FUNCTIONS
# ========================================


# Global state management
workflow_state = {
    'video_url': None,
    'video_info': None,
    'category': None,
    'temp_dir': None,
    'audio_path': None,
    'transcript_text': None,
    'hotspots': [],
    'verified_hotspots': [],
    'clips_metadata': [],
    'edit_plan': [],
    'final_plan': [],
    'final_video_path': None,
    'num_hotspots': 5  # Default number of hotspots
}

# Manual mode state
manual_state = {
    'video_url': None,
    'video_info': None,
    'temp_dir': None,
    'transcript_text': None,
    'topics': [],
    'selected_indices': [],
    'clips_metadata': [],
    'verified_clips': [],
    'final_video_path': None
}


def step1_analyze_video(url: str):
    """
    Step 1: Analyze video metadata and classify.

    Downloads video metadata from YouTube and classifies it as either "podcast" or "generic"
    to determine which editing pipeline to use. This is the first step in the step-by-step workflow.

    Args:
        url: Full YouTube video URL to analyze (e.g., "https://youtube.com/watch?v=...")
            Must be a valid, accessible YouTube video URL.

    Returns:
        Tuple of (info_text, button_update):
        - info_text: Formatted markdown string with video info, duration, uploader, channel, and classification
        - button_update: Gradio update object to enable/disable next step button
    """
    try:
        workflow_state['temp_dir'] = tempfile.mkdtemp()
        workflow_state['video_url'] = url

        logger.info(f"Analyzing: {url}")
        video_info = get_video_info(url)
        workflow_state['video_info'] = video_info

        category = classify_video(video_info)
        workflow_state['category'] = category

        duration = video_info.get('duration', 0)
        uploader = video_info.get('uploader', 'Unknown')
        channel = video_info.get('channel', 'Unknown')

        info_text = f"""
**Video Info:**
- Title: {video_info.get('title', 'Unknown')}
- Duration: {duration:.0f}s ({duration/60:.1f} min)
- Uploader: {uploader}
- Channel: {channel}
- Classification: **{category.upper()}**

**Pipeline Selected:** {'🎙️ Podcast (Transcript + Audio)' if category == 'podcast' else '🎬 Generic (Visual + Audio)'}

**Classification Reasoning:** Check console logs for details.
"""
        return info_text, gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return f"❌ Error: {e}", gr.update(interactive=False)


def step2_scout_hotspots(url: str, num_hotspots: int = 5):
    """
    Step 2: Scout for hotspots using LLM semantic analysis + audio analysis.

    Finds interesting moments (hotspots) using multi-modal analysis:
    - LLM-based semantic analysis of transcript for viral-worthy moments
    - Audio analysis for energy peaks and bad segment filtering
    - Deduplication and scoring of combined hotspots

    This step must be run after step1_analyze_video.

    Args:
        url: Full YouTube video URL to scout (e.g., "https://youtube.com/watch?v=...")
            Should be the same URL used in step 1. Downloads audio and transcript for analysis.
        num_hotspots: Number of hotspots to find (default 5, range 3-10)

    Returns:
        Tuple of (result_text, button_update):
        - result_text: Formatted markdown string listing found hotspots with timestamps, scores, types, and context
        - button_update: Gradio update object to enable/disable next step button
    """
    try:
        if not workflow_state['video_info']:
            return "❌ Please run Step 1 first!", gr.update(interactive=False)

        workflow_state['num_hotspots'] = int(num_hotspots)
        video_info = workflow_state['video_info']
        temp_dir = workflow_state['temp_dir']

        logger.info(f"🔍 Scouting hotspots (finding top {num_hotspots})...")

        # Download audio
        logger.info("📥 Downloading audio...")
        audio_path = os.path.join(temp_dir, "audio")
        actual_audio_path = download_audio(url, audio_path)
        workflow_state['audio_path'] = actual_audio_path

        # Get transcript
        transcript_text = get_transcript(url, include_timestamps=True)
        workflow_state['transcript_text'] = transcript_text

        # Audio analysis
        logger.info("🎵 Analyzing audio signals...")
        audio_hotspots = scout.analyze(actual_audio_path)

        # Semantic analysis with LLM
        semantic_hotspots = []
        if transcript_text and 'Error' not in transcript_text:
            logger.info("🧠 Running LLM semantic analysis...")
            semantic_hotspots = analyze_transcript_with_llm(
                transcript_text, video_info)

        # Combine and deduplicate
        all_hotspots = []

        for h in audio_hotspots:
            h['source'] = 'audio'
            h['type'] = h.get('type', 'audio')
            all_hotspots.append(h)

        for h in semantic_hotspots:
            # Check for overlap with existing
            overlaps = False
            for existing in all_hotspots:
                if abs(existing['start'] - h['start']) < 15:
                    # Boost score and add semantic info to existing
                    existing['score'] = max(existing['score'], h['score'])
                    existing['semantic_score'] = h.get('semantic_score', 0)
                    existing['reasoning'] = h.get('reasoning', '')
                    overlaps = True
                    break
            if not overlaps:
                h['type'] = 'semantic'
                all_hotspots.append(h)

        # Sort by score
        all_hotspots.sort(key=lambda x: x['score'], reverse=True)
        candidates = all_hotspots[:num_hotspots + 2]  # Get a few extra

        workflow_state['hotspots'] = candidates

        # Build output
        result_text = f"""## 🎯 Hotspots Found

**Audio hotspots:** {len(audio_hotspots)}
**Semantic hotspots:** {len(semantic_hotspots)}
**Combined (deduplicated):** {len(all_hotspots)}

### Top {num_hotspots} Candidates:
"""
        for i, h in enumerate(candidates[:num_hotspots], 1):
            start_fmt = f"{int(h['start'] // 60)}:{int(h['start'] % 60):02d}"
            end_fmt = f"{int(h['end'] // 60)}:{int(h['end'] % 60):02d}"
            source = h.get('source', h.get('type', 'unknown'))
            result_text += f"\n{i}. **{start_fmt} - {end_fmt}** | Score: {h['score']:.2f} | Source: {source}"
            if h.get('reasoning'):
                result_text += f"\n   _{h['reasoning'][:80]}..._"

        return result_text, gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {e}", gr.update(interactive=False)


def step3_verify_hotspots(url: str):
    """
    Step 3: Download clips and verify with Vision AI (Parallel Processing).

    Downloads and verifies video clips in parallel for 3-4x speedup:
    - Phase 1: Parallel download of all clips (4 workers)
    - Phase 2: Parallel verification with Gemini Vision AI (4 workers)

    This step must be run after step2_scout_hotspots.

    Args:
        url: Full YouTube video URL to download clips from (e.g., "https://youtube.com/watch?v=...")
            Should be the same URL used in previous steps. Downloads video segments around hotspots.

    Returns:
        Tuple of (result_text, button_update):
        - result_text: Formatted markdown string showing verification results for each clip with scores and pass/fail status
        - button_update: Gradio update object to enable/disable next step button
    """
    try:
        logger.info("🔍 Step 3: Parallel download + verification...")

        if not workflow_state['hotspots']:
            return "❌ Please run Step 2 first!", gr.update(interactive=False)

        temp_dir = workflow_state['temp_dir']
        candidates = workflow_state['hotspots']
        num_hotspots = workflow_state.get('num_hotspots', 5)

        # Phase 1: Build download tasks
        download_tasks = []
        for i, h in enumerate(candidates[:num_hotspots]):
            start = max(0, h['start'] - 3)
            end = h['end'] + 3
            path = os.path.join(temp_dir, f"clip_{i}")
            download_tasks.append({
                'url': url,
                'start': start,
                'end': end,
                'path': path,
                'hotspot': h,
                'index': i
            })

        result_text = "## 🔍 Verification Results\n\n"
        result_text += f"**Phase 1: Downloading {len(download_tasks)} clips (parallel)...**\n"

        # Parallel download
        download_start = time.time()
        download_results = download_clips_parallel(
            download_tasks, max_workers=4)
        download_time = time.time() - download_start

        successful_downloads = [
            r for r in download_results if r.get('success')]
        result_text += f"✅ Downloaded {len(successful_downloads)}/{len(download_tasks)} clips in {download_time:.1f}s\n\n"

        if not successful_downloads:
            return "❌ No clips were downloaded successfully.", gr.update(interactive=False)

        # Phase 2: Build verification tasks
        clips_to_verify = []
        for r in successful_downloads:
            clips_to_verify.append({
                'path': r['path'],
                'hotspot': r['hotspot'],
                'index': r['index'],
                'start': r['start'],
                'end': r['end']
            })

        result_text += f"**Phase 2: Verifying {len(clips_to_verify)} clips (parallel)...**\n"

        # Parallel verification
        verify_start = time.time()
        api_key = VIDEO_API_KEY or GEMINI_API_KEY
        verification_results = verify_clips_parallel(
            clips_to_verify, api_key=api_key, max_workers=4)
        verify_time = time.time() - verify_start

        result_text += f"✅ Verified in {verify_time:.1f}s\n\n"

        # Process results
        verified_hotspots = []
        clips_metadata = []

        result_text += "### Results:\n"
        for vr in verification_results:
            v = vr['verification']
            h = vr['hotspot']
            score = v.get('score', 5)
            passed = score >= 5

            start_fmt = f"{int(h['start'] // 60)}:{int(h['start'] % 60):02d}"
            status = "✅ PASS" if passed else "❌ FAIL"
            result_text += f"- **{start_fmt}**: {status} (Score: {score}/10)\n"

            if passed:
                h['verified_score'] = score
                verified_hotspots.append(h)
                clips_metadata.append({
                    'start': vr['start'],
                    'end': vr['end'],
                    'path': vr['path'],
                    'hotspot': h
                })

        workflow_state['verified_hotspots'] = verified_hotspots
        workflow_state['clips_metadata'] = clips_metadata

        result_text += f"\n**{len(verified_hotspots)} clips passed verification**"

        if len(verified_hotspots) == 0:
            return result_text + "\n\n⚠️ No clips passed verification!", gr.update(interactive=False)

        return result_text, gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {e}", gr.update(interactive=False)


def _step4_create_plan_internal():
    """
    Internal function that creates the edit plan without Gradio UI dependencies.
    Returns plain data structures suitable for both UI and MCP.
    """
    verified_hotspots = workflow_state['verified_hotspots']
    video_info = workflow_state['video_info']
    clips_metadata = workflow_state['clips_metadata']

    logger.info(
        f"Director creating plan from {len(verified_hotspots)} verified hotspots...")
    logger.info(
        f"Video: {video_info.get('title', 'Unknown')} ({video_info.get('duration', 0):.0f}s)")
    logger.info(f"Available clips: {len(clips_metadata)}")

    # Check if Director is initialized
    if not director.api_key:
        error_msg = "❌ Director not initialized! GEMINI_API_KEY not set."
        logger.error(error_msg)
        return None, error_msg

    plan = director.create_edit_plan(verified_hotspots, video_info)

    if not plan:
        error_msg = "❌ Director failed to create plan! Check logs for details."
        logger.error(error_msg)
        logger.error(f"Verified hotspots: {len(verified_hotspots)}")
        logger.error(f"Video info: {video_info.get('title', 'Unknown')}")
        return None, error_msg

    logger.info(f"Director created plan with {len(plan)} items: {plan}")

    # Map to clips
    final_plan = map_plan_to_clips(plan, clips_metadata)

    if not final_plan:
        error_msg = f"❌ Failed to map plan to clips! Plan had {len(plan)} items but none mapped successfully."
        logger.error(error_msg)
        logger.error(f"Plan items: {plan}")
        logger.error(
            f"Available clips: {[(c['start'], c['end']) for c in clips_metadata]}")
        return None, error_msg

    workflow_state['edit_plan'] = plan
    workflow_state['final_plan'] = final_plan

    result_text = f"**Edit Plan ({len(final_plan)} clips):**\n\n"
    total_duration = 0

    for i, item in enumerate(final_plan, 1):
        duration = item['end'] - item['start']
        total_duration += duration
        result_text += f"{i}. **{item['start']:.1f}s - {item['end']:.1f}s** ({duration:.1f}s)\n"
        result_text += f"   Description: {item.get('description', 'N/A')}\n"
        result_text += f"   Source: {os.path.basename(item.get('clip_path', 'N/A'))}\n\n"

    result_text += f"**Total Duration: {total_duration:.1f}s**"

    logger.info(
        f"✅ Plan created successfully: {len(final_plan)} clips, {total_duration:.1f}s total")

    return final_plan, result_text


def step4_create_plan():
    """
    Step 4: Director creates edit plan.

    Uses AI (Gemini 2.0 Flash Lite) to create an edit plan from verified hotspots.
    Selects 3-5 best clips (8-15 seconds each) for a total duration of 30-60 seconds.
    Maps the plan to actual downloaded clip files.

    This step must be run after step3_verify_hotspots.

    Returns:
        Tuple of (result_text, button_update, plan_json):
        - result_text: Formatted markdown string showing the edit plan with clip timestamps, durations, descriptions, and total duration
        - button_update: Gradio update object to enable/disable next step button
        - plan_json: JSON string representation of the edit plan for inspection
    """
    try:
        # Check prerequisites with detailed logging
        if not workflow_state.get('verified_hotspots'):
            error_msg = "❌ Please run Step 3 first! No verified hotspots found."
            logger.warning(f"Step 4 failed: {error_msg}")
            logger.warning(
                f"Workflow state keys: {list(workflow_state.keys())}")
            return error_msg, gr.update(interactive=False), ""

        if not workflow_state.get('video_info'):
            error_msg = "❌ Please run Step 1 first! No video info found."
            logger.warning(f"Step 4 failed: {error_msg}")
            return error_msg, gr.update(interactive=False), ""

        if not workflow_state.get('clips_metadata'):
            error_msg = "❌ Please run Step 3 first! No clips metadata found."
            logger.warning(f"Step 4 failed: {error_msg}")
            return error_msg, gr.update(interactive=False), ""

        final_plan, result_text = _step4_create_plan_internal()

        if final_plan is None:
            logger.error(
                f"Step 4 internal function returned None. Result text: {result_text}")
            return result_text, gr.update(interactive=False), ""

        plan_json = json.dumps(final_plan, indent=2)
        logger.info(
            f"✅ Step 4 completed successfully: {len(final_plan)} clips in plan")
        return result_text, gr.update(interactive=True), plan_json

    except Exception as e:
        error_msg = f"❌ Step 4 failed: {e}"
        logger.error(error_msg)
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        return error_msg, gr.update(interactive=False), ""


def step5_render_video():
    """
    Step 5: Hands renders the final video.

    Executes the edit plan by concatenating selected clips into a final video.
    Uses the Hands component to handle timestamp mapping and video composition.

    This step must be run after step4_create_plan.

    Returns:
        Tuple of (result_text, video_path):
        - result_text: Success message with output file path, or error message if failed
        - video_path: Path to the rendered video file, or None if rendering failed
    """
    try:
        if not workflow_state['final_plan']:
            return "❌ Please run Step 4 first!", None

        final_plan = workflow_state['final_plan']
        clips_metadata = workflow_state['clips_metadata']
        category = workflow_state['category']

        if not clips_metadata:
            return "❌ No clips available!", None

        logger.info("Rendering final video...")
        output_filename = f"{category}_edit_{int(time.time())}.mp4"
        output_path = hands.execute(
            clips_metadata[0]['path'], final_plan, output_filename=output_filename)

        if output_path and os.path.exists(output_path):
            workflow_state['final_video_path'] = output_path
            success_msg = (
                "✅ **Success!** Video saved to:"
                f" `{output_path}`\n\nYou can now load this render in the Production Studio tab."
            )
            return success_msg, output_path
        else:
            workflow_state['final_video_path'] = None
            return "❌ Rendering failed!", None

    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        workflow_state['final_video_path'] = None
        return f"❌ Error: {e}", None


def get_most_recent_render() -> tuple:
    """
    Get the most recent rendered video path from either Auto Mode or Manual Mode.
    Returns: (path, source) where source is 'Auto Mode' or 'Manual Mode', or (None, None) if not found.
    """
    auto_path = workflow_state.get('final_video_path')
    manual_path = manual_state.get('final_video_path')

    candidates = []
    if auto_path and os.path.exists(auto_path):
        candidates.append(
            ('Auto Mode', auto_path, os.path.getmtime(auto_path)))
    if manual_path and os.path.exists(manual_path):
        candidates.append(('Manual Mode', manual_path,
                          os.path.getmtime(manual_path)))

    if not candidates:
        return None, None

    # Sort by modification time (most recent first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    source, path, _ = candidates[0]
    return path, source


def load_last_render_into_production():
    """
    Load the most recent rendered video (if available) into the Production Studio tab.
    Checks both Auto Mode (workflow_state) and Manual Mode (manual_state).
    """
    path, source = get_most_recent_render()

    if not path:
        message = "❌ No rendered video available. Please run Step 5 (Auto Mode) or Step 3 (Manual Mode) first."
        return gr.update(value=None), message

    message = f"✅ Loaded `{os.path.basename(path)}` from {source} into Production Studio."
    return gr.update(value=path), message


# ============================================================================
# TOPIC EXTRACTION FOR MANUAL MODE
# ============================================================================

def extract_topics_from_transcript(transcript_text: str, video_info: Dict) -> List[Dict]:
    """
    Use Gemini to analyze the full transcript and extract distinct topics/segments
    that users can choose from.
    """
    if not transcript_text or not GEMINI_API_KEY:
        return []

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Get video duration from info
        duration = video_info.get('duration', 600)

        prompt = f"""Analyze this transcript from "{video_info.get('title', 'a video')}" and identify 5-12 distinct TOPICS or SEGMENTS that could each make a standalone short clip.

TRANSCRIPT:
{transcript_text[:15000]}  

VIDEO DURATION: {duration // 60} minutes {duration % 60} seconds

For each topic, provide:
1. A catchy title (what would make someone click)
2. Start and end timestamps [MM:SS]
3. Brief description (1-2 sentences)
4. Viral potential score (1-10)
5. A notable quote from that segment

Respond in JSON format:
{{
    "topics": [
        {{
            "title": "Topic Title Here",
            "start_time": "MM:SS",
            "end_time": "MM:SS", 
            "description": "Brief description",
            "viral_score": 8,
            "notable_quote": "Exact quote from transcript"
        }}
    ]
}}

Focus on moments with:
- Strong opinions or controversial takes
- Emotional peaks
- Funny moments
- Surprising revelations
- Quotable statements

Order by appearance in video (chronologically)."""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            topics = result.get('topics', [])

            # Convert timestamps to seconds
            for topic in topics:
                start_parts = topic.get('start_time', '0:00').split(':')
                end_parts = topic.get('end_time', '0:00').split(':')

                topic['start_seconds'] = int(
                    start_parts[0]) * 60 + int(start_parts[1])
                topic['end_seconds'] = int(
                    end_parts[0]) * 60 + int(end_parts[1])

                # Ensure minimum clip length
                if topic['end_seconds'] - topic['start_seconds'] < 15:
                    topic['end_seconds'] = topic['start_seconds'] + 30

            return topics

    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        return []

    return []


# ============================================================================
# MANUAL MODE WORKFLOW FUNCTIONS
# ============================================================================

def manual_step1_analyze(url: str):
    """Analyze video and extract topics for user selection."""
    global manual_state

    if not url or not url.strip():
        return "❌ Please enter a YouTube URL.", gr.update(choices=[], interactive=False), gr.update(interactive=False)

    try:
        logger.info(f"📺 [Manual] Analyzing: {url}")

        # Get video info
        video_info = get_video_info(url)
        if not video_info:
            return "❌ Could not fetch video info.", gr.update(choices=[], interactive=False), gr.update(interactive=False)

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Get transcript
        video_id = extract_video_id(url)
        transcript_text = get_transcript(video_id)

        if "Error" in transcript_text or not transcript_text.strip():
            return f"❌ Could not get transcript: {transcript_text}", gr.update(choices=[], interactive=False), gr.update(interactive=False)

        # Extract topics
        logger.info("🧠 Extracting topics with LLM...")
        topics = extract_topics_from_transcript(transcript_text, video_info)

        if not topics:
            return "❌ Could not extract topics from transcript.", gr.update(choices=[], interactive=False), gr.update(interactive=False)

        # Store state
        manual_state['video_url'] = url
        manual_state['video_info'] = video_info
        manual_state['temp_dir'] = temp_dir
        manual_state['transcript_text'] = transcript_text
        manual_state['topics'] = topics

        # Format for checkboxes
        checkbox_choices = []
        for i, topic in enumerate(topics, 1):
            label = f"✅ {i}. {topic['title']} [{topic.get('start_time', '?')} - {topic.get('end_time', '?')}] (Score: {topic.get('viral_score', '?')}/10)"
            checkbox_choices.append(label)

        # Build output
        output = f"""## ✅ Video Analyzed

**Title:** {video_info.get('title', 'Unknown')}
**Duration:** {video_info.get('duration', 0) // 60}:{video_info.get('duration', 0) % 60:02d}

### 📋 Found {len(topics)} Topics

Select the topics you want to include in your video below:
"""
        for i, topic in enumerate(topics, 1):
            output += f"\n**{i}. {topic['title']}** ({topic.get('start_time', '?')} - {topic.get('end_time', '?')})\n"
            output += f"   Score: {topic.get('viral_score', '?')}/10 | {topic.get('description', '')}\n"
            if topic.get('notable_quote'):
                output += f"   💬 \"{topic['notable_quote'][:100]}...\"\n"

        return output, gr.update(choices=checkbox_choices, value=[], interactive=True), gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Manual Step 1 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(choices=[], interactive=False), gr.update(interactive=False)


def manual_step2_extract(selected_topics: List[str]):
    """Extract clips from selected topics."""
    global manual_state

    if not selected_topics:
        return "❌ Please select at least one topic.", gr.update(interactive=False)

    try:
        # Parse selected indices from checkbox values like "✅ 1. Topic Title [time] (Score: X/10)"
        selected_indices = []
        for topic_str in selected_topics:
            # Extract the number after the emoji
            match = re.search(r'(\d+)\.', topic_str)
            if match:
                idx = int(match.group(1)) - 1
                selected_indices.append(idx)

        manual_state['selected_indices'] = selected_indices

        # Get selected topics
        topics = manual_state['topics']
        url = manual_state['video_url']
        temp_dir = manual_state['temp_dir']

        # Build download tasks
        download_tasks = []
        for i, idx in enumerate(selected_indices):
            if idx < len(topics):
                topic = topics[idx]
                clip_path = os.path.join(temp_dir, f"manual_clip_{i}.mp4")
                start = max(0, topic['start_seconds'] - 2)
                end = topic['end_seconds'] + 2

                download_tasks.append({
                    'url': url,
                    'start': start,
                    'end': end,
                    'path': clip_path,
                    'format_spec': None,
                    'index': i,
                    'hotspot': topic  # Use 'hotspot' key for compatibility
                })

        # Parallel download
        download_start = time.time()
        download_results = download_clips_parallel(
            download_tasks, max_workers=4)
        download_time = time.time() - download_start

        # Process results
        clips_metadata = []
        for result in download_results:
            if result.get('success'):
                clips_metadata.append({
                    'path': result['path'],
                    'topic': download_tasks[result['index']]['hotspot'],
                    'start': download_tasks[result['index']]['start'],
                    'end': download_tasks[result['index']]['end'],
                    'index': result['index']
                })

        manual_state['clips_metadata'] = clips_metadata

        # Build output
        output_lines = [f"## ✅ Clips Extracted\n"]
        output_lines.append(
            f"Downloaded **{len(clips_metadata)}/{len(selected_indices)}** clips in **{download_time:.1f}s**\n")

        for clip in clips_metadata:
            topic = clip['topic']
            output_lines.append(
                f"✓ **{topic['title']}** ({topic.get('start_time', '?')} - {topic.get('end_time', '?')})")

        if clips_metadata:
            return '\n'.join(output_lines), gr.update(interactive=True)
        else:
            return "❌ No clips were successfully downloaded.", gr.update(interactive=False)

    except Exception as e:
        logger.error(f"Manual Step 2 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def manual_step3_render():
    """Render final video from selected clips."""
    global manual_state

    clips_metadata = manual_state.get('clips_metadata', [])

    if not clips_metadata:
        return "❌ No clips available to render.", None

    try:
        logger.info(f"🎬 [Manual] Rendering {len(clips_metadata)} clips...")

        temp_dir = manual_state['temp_dir']
        video_info = manual_state['video_info']

        # Create edit plan from clips - include clip_path for pre-downloaded clips
        edit_plan = []
        for clip in clips_metadata:
            topic = clip['topic']
            edit_plan.append({
                'start': clip['start'],
                'end': clip['end'],
                'clip_path': clip['path'],  # Use the downloaded clip
                # For calculating local timestamps
                'source_start': clip['start'],
                'description': topic['title'],
                'reason': topic.get('description', '')
            })

        # Sort by start time
        edit_plan.sort(key=lambda x: x['start'])

        # Render using hands component
        output_filename = f"manual_edit_{int(time.time())}.mp4"

        # Use execute method - pass a dummy video path since we have clip_paths
        output_path = hands.execute(
            # Dummy, won't be used since all clips have clip_path
            video_path=clips_metadata[0]['path'],
            edit_plan=edit_plan,
            output_filename=output_filename
        )

        if output_path and os.path.exists(output_path):
            # Calculate duration
            total_duration = sum(clip['end'] - clip['start']
                                 for clip in clips_metadata)

            manual_state['final_video_path'] = output_path

            return f"""## ✅ Video Rendered Successfully!

**Clips included:** {len(clips_metadata)}
**Total duration:** ~{int(total_duration)} seconds
**Output:** {os.path.basename(output_path)}

### Topics in your video:
""" + '\n'.join([f"- {clip['topic']['title']}" for clip in clips_metadata]), output_path
        else:
            return "❌ Rendering failed.", None

    except Exception as e:
        logger.error(f"Manual Step 3 error: {e}")
        return f"❌ Error rendering video: {str(e)}", None


def reset_manual_workflow():
    """Reset manual workflow state."""
    global manual_state

    # Clean up temp directory
    if manual_state.get('temp_dir') and os.path.exists(manual_state['temp_dir']):
        try:
            shutil.rmtree(manual_state['temp_dir'])
        except:
            pass

    manual_state = {
        'video_url': None,
        'video_info': None,
        'temp_dir': None,
        'transcript_text': None,
        'topics': [],
        'selected_indices': [],
        'clips_metadata': [],
        'verified_clips': [],
        'final_video_path': None
    }

    return (
        "",  # output1
        gr.update(choices=["⏳ Run Step 1 first to see available topics..."], value=[
        ], interactive=False),
        "",  # output2
        "",  # output3
        None,  # video
        gr.update(interactive=False),  # btn2
        gr.update(interactive=False)   # btn3
    )


# ============================================================================
# MCP TOOLS - Exposed to MCP clients
# ============================================================================

@gr.mcp.tool()
def step1_analyze_video_mcp(youtube_url: str) -> str:
    """
    Step 1: Analyze and classify a YouTube video for the editing pipeline.
    Downloads metadata and classifies as podcast or generic.
    Must run this before step2, step3, step4, and step5.
    """
    try:
        workflow_state['temp_dir'] = tempfile.mkdtemp()
        workflow_state['video_url'] = youtube_url

        logger.info(f"[MCP] Step 1: Analyzing {youtube_url}")
        video_info = get_video_info(youtube_url)
        workflow_state['video_info'] = video_info

        category = classify_video(video_info)
        workflow_state['category'] = category

        title = video_info.get('title', 'Unknown')
        duration = video_info.get('duration', 0)
        uploader = video_info.get('uploader', 'Unknown')

        pipeline = "Podcast Pipeline (transcript + audio)" if category == 'podcast' else "Generic Pipeline (audio + visual)"

        return f"Step 1 Complete!\n\nVideo: {title}\nDuration: {duration/60:.1f} min\nUploader: {uploader}\nClassification: {category.upper()}\nPipeline: {pipeline}\n\nNext: Call step2_scout_hotspots_mcp with the same URL."

    except Exception as e:
        logger.error(f"[MCP] Step 1 failed: {e}")
        return f"Error in step 1: {str(e)}"


@gr.mcp.tool()
def step2_scout_hotspots_mcp(youtube_url: str, num_hotspots: int = 5) -> str:
    """
    Step 2: Scout for interesting moments (hotspots) in the video using LLM analysis.
    Uses Gemini to analyze transcript for viral-worthy moments + audio analysis.
    Must run step1_analyze_video_mcp first.

    Args:
        youtube_url: The YouTube URL to analyze
        num_hotspots: Number of hotspots to find (default 5, range 3-10)
    """
    try:
        if not workflow_state.get('video_info'):
            return "Error: Please run step1_analyze_video_mcp first!"

        video_info = workflow_state['video_info']
        temp_dir = workflow_state['temp_dir']
        workflow_state['num_hotspots'] = int(num_hotspots)

        logger.info(
            f"[MCP] Step 2: Scouting {num_hotspots} hotspots with LLM analysis...")

        # Download audio
        audio_path = os.path.join(temp_dir, "audio")
        actual_audio_path = download_audio(youtube_url, audio_path)
        workflow_state['audio_path'] = actual_audio_path

        # Get transcript
        transcript_text = get_transcript(youtube_url, include_timestamps=True)
        workflow_state['transcript_text'] = transcript_text

        # Audio analysis
        logger.info("  Analyzing audio signals...")
        audio_hotspots = scout.analyze(actual_audio_path)

        # Semantic analysis with LLM
        semantic_hotspots = []
        if transcript_text and 'Error' not in transcript_text:
            logger.info("  Running LLM semantic analysis...")
            semantic_hotspots = analyze_transcript_with_llm(
                transcript_text, video_info)

        # Combine and deduplicate
        all_hotspots = []

        for h in audio_hotspots:
            h['source'] = 'audio'
            h['type'] = h.get('type', 'audio')
            all_hotspots.append(h)

        for h in semantic_hotspots:
            # Check for overlap with existing
            overlaps = False
            for existing in all_hotspots:
                if abs(existing['start'] - h['start']) < 15:
                    existing['score'] = max(existing['score'], h['score'])
                    existing['semantic_score'] = h.get('semantic_score', 0)
                    existing['reasoning'] = h.get('reasoning', '')
                    overlaps = True
                    break
            if not overlaps:
                h['type'] = 'semantic'
                all_hotspots.append(h)

        # Sort by score
        all_hotspots.sort(key=lambda x: x['score'], reverse=True)
        candidates = all_hotspots[:num_hotspots + 2]

        workflow_state['hotspots'] = candidates

        hotspots_info = []
        for i, h in enumerate(candidates[:num_hotspots]):
            source = h.get('source', h.get('type', 'unknown'))
            info = f"  {i+1}. {h['start']:.1f}s-{h['end']:.1f}s (score: {h['score']:.2f}, source: {source})"
            if h.get('reasoning'):
                info += f"\n      {h['reasoning'][:60]}..."
            hotspots_info.append(info)

        return f"Step 2 Complete!\n\nAudio hotspots: {len(audio_hotspots)}\nSemantic hotspots: {len(semantic_hotspots)}\nCombined: {len(all_hotspots)}\n\nTop {num_hotspots} candidates:\n" + "\n".join(hotspots_info) + "\n\nNext: Call step3_verify_hotspots_mcp with the same URL."

    except Exception as e:
        logger.error(f"[MCP] Step 2 failed: {e}")
        return f"Error in step 2: {str(e)}"


@gr.mcp.tool()
def step3_verify_hotspots_mcp(youtube_url: str) -> str:
    """
    Step 3: Download clips and verify quality with Vision AI (Parallel Processing).
    Uses parallel download (4 workers) + parallel verification (4 workers) for 3-4x speedup.
    Must run step2 first.
    """
    try:
        if not workflow_state.get('hotspots'):
            return "Error: Please run step2_scout_hotspots_mcp first!"

        candidates = workflow_state['hotspots']
        temp_dir = workflow_state['temp_dir']
        num_hotspots = workflow_state.get('num_hotspots', 5)

        logger.info(
            f"[MCP] Step 3: Parallel download + verification of {len(candidates[:num_hotspots])} hotspots...")

        # Phase 1: Build download tasks
        download_tasks = []
        for i, h in enumerate(candidates[:num_hotspots]):
            start = max(0, h['start'] - 3)
            end = h['end'] + 3
            path = os.path.join(temp_dir, f"clip_{i}")
            download_tasks.append({
                'url': youtube_url,
                'start': start,
                'end': end,
                'path': path,
                'hotspot': h,
                'index': i
            })

        # Parallel download
        download_start = time.time()
        download_results = download_clips_parallel(
            download_tasks, max_workers=4)
        download_time = time.time() - download_start

        successful_downloads = [
            r for r in download_results if r.get('success')]

        if not successful_downloads:
            return "Error: No clips were downloaded successfully."

        # Phase 2: Parallel verification
        clips_to_verify = []
        for r in successful_downloads:
            clips_to_verify.append({
                'path': r['path'],
                'hotspot': r['hotspot'],
                'index': r['index'],
                'start': r['start'],
                'end': r['end']
            })

        verify_start = time.time()
        api_key = VIDEO_API_KEY or GEMINI_API_KEY
        verification_results = verify_clips_parallel(
            clips_to_verify, api_key=api_key, max_workers=4)
        verify_time = time.time() - verify_start

        # Process results
        verified_hotspots = []
        clips_metadata = []
        results_info = []

        for vr in verification_results:
            v = vr['verification']
            h = vr['hotspot']
            score = v.get('score', 5)
            passed = score >= 5

            status = "PASSED" if passed else "FAILED"
            results_info.append(
                f"  {vr['index']+1}. {h['start']:.1f}s: score {score}/10 - {status}")

            if passed:
                h['verified_score'] = score
                verified_hotspots.append(h)
                clips_metadata.append({
                    'start': vr['start'],
                    'end': vr['end'],
                    'path': vr['path'],
                    'hotspot': h
                })

        workflow_state['verified_hotspots'] = verified_hotspots
        workflow_state['clips_metadata'] = clips_metadata

        next_step = "Next: Call step4_create_plan_mcp" if verified_hotspots else "No clips passed. Try a different video."
        timing = f"Download: {download_time:.1f}s, Verify: {verify_time:.1f}s (parallel)"

        return f"Step 3 Complete!\n\nTiming: {timing}\nVerified {len(verified_hotspots)}/{len(download_tasks)} clips:\n" + "\n".join(results_info) + f"\n\n{next_step}"

    except Exception as e:
        logger.error(f"[MCP] Step 3 failed: {e}")
        return f"Error in step 3: {str(e)}"


@gr.mcp.tool()
def step4_create_plan_mcp() -> str:
    """
    Step 4: Director creates edit plan from verified hotspots.
    Uses AI to select best clips. Must run step3 first.
    """
    try:
        if not workflow_state.get('verified_hotspots'):
            return "Error: Please run step3_verify_hotspots_mcp first!"

        final_plan, result_text = _step4_create_plan_internal()

        if final_plan is None:
            return f"Error: {result_text}"

        total_duration = sum(item['end'] - item['start']
                             for item in final_plan)

        clips_info = []
        for i, item in enumerate(final_plan):
            dur = round(item['end'] - item['start'], 1)
            clips_info.append(
                f"  {i+1}. {item['start']:.1f}s-{item['end']:.1f}s ({dur}s)")

        return f"Step 4 Complete!\n\nEdit plan: {len(final_plan)} clips, {total_duration:.1f}s total\n" + "\n".join(clips_info) + "\n\nNext: Call step5_render_video_mcp to render, or render_and_produce_mcp for full production."

    except Exception as e:
        logger.error(f"[MCP] Step 4 failed: {e}")
        return f"Error in step 4: {str(e)}"


@gr.mcp.tool()
def step5_render_video_mcp() -> str:
    """
    Step 5: Render the final edited video from the edit plan.
    Concatenates clips into final video. Must run step4 first.
    """
    try:
        if not workflow_state.get('final_plan'):
            return "Error: Please run step4_create_plan_mcp first!"

        final_plan = workflow_state['final_plan']
        clips_metadata = workflow_state['clips_metadata']
        category = workflow_state['category']

        if not clips_metadata:
            return "Error: No clips available to render!"

        logger.info("[MCP] Step 5: Rendering final video...")
        output_filename = f"{category}_edit_{int(time.time())}.mp4"
        output_path = hands.execute(
            clips_metadata[0]['path'], final_plan, output_filename=output_filename)

        if output_path and os.path.exists(output_path):
            workflow_state['final_video_path'] = output_path
            return f"Step 5 Complete!\n\nVideo rendered: {output_path}\n\nNext: Call add_production_value_mcp() to add smart crop, intro, subtitles, and music. It will auto-use this video!"
        else:
            workflow_state['final_video_path'] = None
            return "Error: Rendering failed - check logs for details"

    except Exception as e:
        logger.error(f"[MCP] Step 5 failed: {e}")
        workflow_state['final_video_path'] = None
        return f"Error in step 5: {str(e)}"


@gr.mcp.tool()
def get_workflow_state_mcp() -> str:
    """
    Get the current workflow state and rendered video path.
    Use to check progress and find paths to rendered videos.
    """
    try:
        steps_completed = []
        if workflow_state.get('video_info'):
            steps_completed.append("step1")
        if workflow_state.get('hotspots'):
            steps_completed.append("step2")
        if workflow_state.get('verified_hotspots'):
            steps_completed.append("step3")
        if workflow_state.get('final_plan'):
            steps_completed.append("step4")
        if workflow_state.get('final_video_path'):
            steps_completed.append("step5")

        video_path = workflow_state.get('final_video_path', 'None')
        ready = bool(video_path and video_path !=
                     'None' and os.path.exists(video_path))
        title = workflow_state.get('video_info', {}).get(
            'title', 'None') if workflow_state.get('video_info') else 'None'
        category = workflow_state.get('category', 'None')

        return f"Workflow State:\n\nSteps completed: {', '.join(steps_completed) if steps_completed else 'None'}\nVideo title: {title}\nCategory: {category}\nRendered video: {video_path}\nReady for production: {ready}"

    except Exception as e:
        return f"Error getting state: {str(e)}"


@gr.mcp.tool()
def add_production_value_mcp(
    video_path: str = "auto",
    mood: str = "auto",
    enable_smart_crop: bool = True,
    add_intro_image: bool = True,
    add_subtitles: bool = True
) -> str:
    """
    Add production value to a video with AI-powered enhancements.

    RECOMMENDED: Use video_path='auto' to automatically use the last rendered video from step5.
    This is the best approach after running the step-by-step workflow (steps 1-5).

    Features: smart vertical crop (9:16), intro image (FLUX), voiceover (ElevenLabs), 
    mood-matched music, and auto-generated subtitles (WhisperX).

    Args:
        video_path: 'auto' to use last rendered video (recommended), or absolute file path
        mood: 'auto' (recommended), 'hype', 'suspense', or 'chill'
        enable_smart_crop: Enable AI-powered 9:16 crop with subject tracking
        add_intro_image: Generate FLUX AI intro image
        add_subtitles: Add WhisperX word-level subtitles

    Returns:
        Success message with output file path, or error message
    """
    try:
        actual_video_path = video_path
        auto_detected = False

        # Auto-detect from workflow state or manual state (RECOMMENDED APPROACH)
        if video_path == "auto" or not video_path:
            detected_path, source = get_most_recent_render()
            if detected_path:
                actual_video_path = detected_path
                auto_detected = True
                logger.info(
                    f"[MCP] Auto-detected video from {source}: {actual_video_path}")
            else:
                return "Error: No video available. Run steps 1-5 (Auto Mode) or Manual Mode first, then call this tool with video_path='auto'."

        # Handle URL format (convert to local path if it's a Gradio URL)
        elif video_path.startswith('http://') or video_path.startswith('https://'):
            # Try to extract local path from Gradio URL format
            if '/file=' in video_path:
                # Extract path after /file=
                local_path = video_path.split('/file=')[-1].split('?')[0]
                if os.path.exists(local_path):
                    actual_video_path = local_path
                    logger.info(
                        f"[MCP] Extracted local path from URL: {actual_video_path}")
                else:
                    return f"Error: Could not find local file from URL. Use video_path='auto' instead."
            else:
                return f"Error: URL format not supported. Use video_path='auto' to use the last rendered video, or provide a local file path."

        # Verify file exists
        if not os.path.exists(actual_video_path):
            # One more fallback - check workflow state or manual state
            fallback_path, fallback_source = get_most_recent_render()
            if fallback_path:
                actual_video_path = fallback_path
                auto_detected = True
                logger.info(
                    f"[MCP] Fallback to {fallback_source} video: {actual_video_path}")
            else:
                return f"Error: Video file not found: {actual_video_path}. Use video_path='auto' to use the last rendered video."

        logger.info(f"[MCP] Adding production value to: {actual_video_path}")

        showrunner = Showrunner()

        if mood == "auto":
            direction = showrunner.analyze_video(actual_video_path)
            detected_mood = direction['mood']
            intro_script = direction['intro_script']
            title_text = direction['title_card']
        else:
            detected_mood = mood
            intro_script = "Get ready for something incredible..."
            title_text = "WATCH THIS"

        intro_audio = showrunner.generate_intro(intro_script)
        if not intro_audio or not os.path.exists(intro_audio) or os.path.getsize(intro_audio) == 0:
            intro_audio = None

        bg_music = showrunner.select_music(detected_mood)

        final_video = showrunner.compose_final(
            actual_video_path,
            intro_audio,
            bg_music,
            title_text=title_text,
            mood=detected_mood,
            enable_smart_crop=enable_smart_crop,
            add_intro_image=add_intro_image,
            add_subtitles=add_subtitles
        )

        if final_video and os.path.exists(final_video):
            output_filename = f"production_{int(time.time())}.mp4"
            final_output_path = os.path.join(OUTPUT_DIR, output_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.copy(final_video, final_output_path)

            source = "(auto-detected)" if auto_detected else actual_video_path
            features = f"mood={detected_mood}, crop={enable_smart_crop}, intro={add_intro_image}, subs={add_subtitles}"
            return f"Production Complete!\n\nOutput: {final_output_path}\nSource: {source}\nSettings: {features}\nTitle: {title_text}"
        else:
            return "Error: Production failed during composition. Check logs."

    except Exception as e:
        logger.error(f"[MCP] add_production_value_mcp failed: {e}")
        return f"Error in production: {str(e)}"


@gr.mcp.tool()
def render_and_produce_mcp(
    mood: str = "auto",
    enable_smart_crop: bool = True,
    add_intro_image: bool = True,
    add_subtitles: bool = True
) -> str:
    """
    Combined Step 5 + Production in one call. Renders video then adds production value.
    Must run steps 1-4 first.
    """
    try:
        # Step 5: Render (directly, not via MCP tool)
        logger.info("[MCP] render_and_produce: Starting render...")

        if not workflow_state.get('final_plan'):
            return "Error: Please run steps 1-4 first!"

        final_plan = workflow_state['final_plan']
        clips_metadata = workflow_state['clips_metadata']
        category = workflow_state['category']

        if not clips_metadata:
            return "Error: No clips available to render!"

        output_filename = f"{category}_edit_{int(time.time())}.mp4"
        rendered_video = hands.execute(
            clips_metadata[0]['path'], final_plan, output_filename=output_filename)

        if not rendered_video or not os.path.exists(rendered_video):
            return "Error: Rendering failed - check logs"

        workflow_state['final_video_path'] = rendered_video
        logger.info(f"[MCP] Rendered to {rendered_video}")

        # Production
        logger.info("[MCP] Adding production value...")
        showrunner = Showrunner()

        if mood == "auto":
            direction = showrunner.analyze_video(rendered_video)
            detected_mood = direction['mood']
            intro_script = direction['intro_script']
            title_text = direction['title_card']
        else:
            detected_mood = mood
            intro_script = "Get ready for something incredible..."
            title_text = "WATCH THIS"

        intro_audio = showrunner.generate_intro(intro_script)
        if not intro_audio or not os.path.exists(intro_audio) or os.path.getsize(intro_audio) == 0:
            intro_audio = None

        bg_music = showrunner.select_music(detected_mood)

        final_video = showrunner.compose_final(
            rendered_video,
            intro_audio,
            bg_music,
            title_text=title_text,
            mood=detected_mood,
            enable_smart_crop=enable_smart_crop,
            add_intro_image=add_intro_image,
            add_subtitles=add_subtitles
        )

        if final_video and os.path.exists(final_video):
            output_filename = f"production_{int(time.time())}.mp4"
            final_output_path = os.path.join(OUTPUT_DIR, output_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.copy(final_video, final_output_path)

            return f"Render + Production Complete!\n\nRendered: {rendered_video}\nFinal output: {final_output_path}\nMood: {detected_mood}\nFeatures: crop={enable_smart_crop}, intro={add_intro_image}, subs={add_subtitles}"
        else:
            return f"Rendering succeeded ({rendered_video}) but production failed. Check logs."

    except Exception as e:
        logger.error(f"[MCP] render_and_produce_mcp failed: {e}")
        return f"Error: {str(e)}"


@gr.mcp.tool()
def classify_video_type(youtube_url: str) -> str:
    """
    Classify a YouTube video as 'podcast' or 'generic' to determine editing pipeline.

    This tool analyzes video metadata (title, description, channel, tags, duration) to
    determine which editing pipeline will produce the best results.

    **Classification Logic:**

    **Podcast Detection (High Priority):**
    - Known podcast channels: Joe Rogan, Lex Fridman, Huberman Lab, All-In Podcast, etc.
    - Keywords in title/description: "podcast", "interview", "episode", "conversation"
    - Long videos (>15 min) with podcast indicators

    **Generic Detection:**
    - Tutorial keywords: "how to", "tutorial", "guide", "demo"
    - Visual content: "let me show you", "check this out", "watch this"
    - Shorter videos or videos without podcast signals

    **Pipeline Differences:**
    - **Podcast Pipeline**: Semantic-heavy, uses transcript triggers + audio filtering
    - **Generic Pipeline**: Multimodal analysis with deep AI verification (Gemini 2.0 Flash)

    **When to Use:**
    - Before processing to understand which pipeline will be used
    - To verify classification accuracy
    - For debugging misclassifications

    Args:
        youtube_url: Full YouTube video URL to classify
            Example: "https://youtube.com/watch?v=..."

    Returns:
        Formatted string with classification result, video title, and duration.
        Format: "Video classified as: {PODCAST|GENERIC}\nTitle: {title}\nDuration: {duration}s"

    Example:
        classify_video_type("https://youtube.com/watch?v=...")
        Returns: "Video classified as: PODCAST\nTitle: Joe Rogan Experience #1234\nDuration: 3600s"
    """
    try:
        video_info = get_video_info(youtube_url)
        category = classify_video(video_info)
        return f"Video classified as: {category.upper()}\nTitle: {video_info.get('title', 'Unknown')}\nDuration: {video_info.get('duration', 0):.0f}s"
    except Exception as e:
        return f"Error classifying video: {e}"


@gr.mcp.tool()
def smart_crop_video(video_path: str) -> str:
    """
    Apply AI-powered smart crop to convert landscape videos to 9:16 vertical format.

    This tool intelligently tracks the main subject in a video and dynamically crops
    to portrait format (9:16) while keeping the subject centered. Perfect for converting
    landscape videos for TikTok, Instagram Reels, and YouTube Shorts.

    **How It Works:**

    1. **Scene Analysis (Gemini 2.0 Flash Lite)**:
       - Analyzes video for scene changes and subject position shifts
       - Identifies key timestamps where crop position should be recalculated
       - Falls back to [0, duration/2, duration] if API fails

    2. **Subject Detection (Qwen 2.5-VL-72B)**:
       - Extracts frames at key timestamps
       - Analyzes each frame to detect main subject position (0.0-1.0 scale)
       - Prioritizes: person > face > text > main object
       - Returns normalized horizontal position (0.0 = far left, 0.5 = center, 1.0 = far right)

    3. **Dynamic Crop Application (MoviePy)**:
       - Builds position map from keyframe detections
       - Interpolates crop position between keyframes for smooth tracking
       - Calculates crop bounds to maintain 9:16 aspect ratio
       - Applies frame-by-frame cropping with smooth transitions

    **Technical Details:**
    - Input: Any video format (MP4, MOV, AVI, etc.)
    - Output: 9:16 portrait video (e.g., 607x1080 from 1920x1080 source)
    - Processing time: ~30-50 seconds for 15-second video
    - Audio: Preserved in output
    - Fallback: Returns center crop (0.5) if AI analysis fails

    **When to Use:**
    - Converting landscape videos for vertical platforms
    - Before adding production value (intro, subtitles, music)
    - Standalone crop testing without full production pipeline

    **Requirements:**
    - VIDEO_API_KEY (for Gemini scene analysis)
    - NEBIUS_API_KEY (for Qwen VL subject detection)

    Args:
        video_path: Absolute or relative path to the input video file.
            File must exist and be readable. Supports common video formats.

    Returns:
        String with success message and output file path, or error message if failed.
        Output saved to the runtime output directory

    Example:
        smart_crop_video("/path/to/landscape_video.mp4")
        Returns: "Success! Cropped video saved to: /tmp/directors-cut/output/smart_crop_1234567890.mp4"

    TIP: Use 'auto' as video_path to use the last rendered video from workflow.
    """
    actual_video_path = video_path

    # Handle 'auto' to use workflow state or manual state
    if video_path == "auto":
        detected_path, source = get_most_recent_render()
        if detected_path:
            actual_video_path = detected_path
            logger.info(
                f"[MCP] smart_crop_video auto-detected from {source}: {actual_video_path}")
        else:
            return "Error: No video available. Run steps 1-5 (Auto Mode) or Manual Mode first, or provide a video path."

    # Handle URL format
    elif video_path.startswith('http://') or video_path.startswith('https://'):
        if '/file=' in video_path:
            local_path = video_path.split('/file=')[-1].split('?')[0]
            if os.path.exists(local_path):
                actual_video_path = local_path
            else:
                return f"Error: Could not find local file from URL. Use 'auto' or a local path."
        else:
            return f"Error: URL format not supported. Use 'auto' or a local file path."

    if not os.path.exists(actual_video_path):
        return f"Error: Video file not found: {actual_video_path}"

    try:
        showrunner = Showrunner()
        cropped_video_path = showrunner.smart_crop_pipeline(actual_video_path)

        if cropped_video_path and os.path.exists(cropped_video_path):
            # Copy to output directory
            final_path = os.path.join(
                "temp", f"smart_crop_{int(time.time())}.mp4")
            shutil.copy(cropped_video_path, final_path)
            return f"Success! Cropped video saved to: {final_path}"
        else:
            return "Smart crop failed. Check logs for details."

    except Exception as e:
        import traceback
        error_msg = f"Error during smart crop: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


@gr.mcp.tool()
def add_production_value(
    video_path: str,
    mood: str = "auto",
    enable_smart_crop: bool = True,
    add_intro_image: bool = True,
    add_subtitles: bool = True,
) -> str:
    """
    Add professional production value to a video with AI-powered enhancements.

    This is the "Production Studio" pipeline that transforms raw videos into polished,
    social media-ready content with multiple AI-powered features.

    **Features Available:**

    1. **Smart Vertical Crop** (if enabled):
       - AI-powered 9:16 crop that tracks subjects
       - Uses Gemini + Qwen VL for intelligent subject tracking
       - Applied first, before other enhancements

    2. **AI-Generated Intro Image** (if enabled):
       - FLUX model generates custom intro screen
       - Based on video content analysis
       - Includes title card text

    3. **Professional Voiceover** (always generated):
       - ElevenLabs API generates high-quality intro voiceover
       - Script generated by AI based on video analysis
       - Falls back gracefully if API key missing

    4. **Background Music** (always added):
       - Mood-matched music selection
       - Options: "hype", "suspense", "chill"
       - Auto-detected from video content if mood="auto"
       - Music from local assets/music/{mood}/ directory

    5. **Auto-Generated Subtitles** (if enabled):
       - WhisperX for word-level transcription
       - TikTok-style captions burned into video
       - Includes intro voiceover in transcription

    **Mood Detection (when mood="auto"):**
    - Analyzes video content with Gemini 2.0 Flash
    - Detects emotional tone and pacing
    - Generates appropriate intro script and title
    - Selects matching background music

    **Processing Order:**
    1. Smart crop (if enabled)
    2. Intro image generation (if enabled)
    3. Voiceover generation
    4. Music selection
    5. Subtitle generation (if enabled)
    6. Final composition (combines all elements)

    **When to Use:**
    - After processing a video with `process_video()` to add polish
    - For standalone video enhancement without full editing pipeline
    - To create social media-ready content from any video

    **Requirements:**
    - GEMINI_API_KEY (for video analysis and mood detection)
    - ELEVENLABS_API_KEY (for voiceover - optional, continues without if missing)
    - NEBIUS_API_KEY (for FLUX intro images - optional)
    - Music files in `assets/music/{hype,suspense,chill}/` directories

    Args:
        video_path: Absolute or relative path to input video file.
            File must exist and be readable.

        mood: Video mood selection. Options:
            - "auto": AI analyzes video and detects mood automatically (recommended)
            - "hype": Upbeat, energetic music and style
            - "suspense": Tension-building music and style
            - "chill": Relaxed, calm music and style
            Default: "auto"

        enable_smart_crop: Enable AI-powered smart crop to 9:16 portrait format.
            Uses Gemini + Qwen VL for subject tracking. Recommended for landscape videos.
            Default: True

        add_intro_image: Generate AI intro image using FLUX model.
            Creates custom title card based on video content.
            Requires NEBIUS_API_KEY. Default: True

        add_subtitles: Add word-level subtitles using WhisperX.
            Creates TikTok-style captions burned into video.
            Includes intro voiceover in transcription. Default: True

    Returns:
        String with success message and output file path, or error message if failed.
        Output video includes all enabled features in polished final format.

    Example:
        add_production_value("/path/to/video.mp4", mood="auto", enable_smart_crop=True)
        Returns: "Success! Polished video saved to: /tmp/directors-cut/output/polished_1234567890.mp4"

    TIP: Use 'auto' as video_path to use the last rendered video from workflow.
    """
    actual_video_path = video_path

    # Handle 'auto' to use workflow state or manual state
    if video_path == "auto":
        detected_path, source = get_most_recent_render()
        if detected_path:
            actual_video_path = detected_path
            logger.info(
                f"[MCP] add_production_value auto-detected from {source}: {actual_video_path}")
        else:
            return "Error: No video available. Run steps 1-5 (Auto Mode) or Manual Mode first, or provide a video path."

    # Handle URL format
    elif video_path.startswith('http://') or video_path.startswith('https://'):
        if '/file=' in video_path:
            local_path = video_path.split('/file=')[-1].split('?')[0]
            if os.path.exists(local_path):
                actual_video_path = local_path
            else:
                return f"Error: Could not find local file from URL. Use 'auto' or a local path."
        else:
            return f"Error: URL format not supported. Use 'auto' or a local file path."

    if not os.path.exists(actual_video_path):
        return f"Error: Video file not found: {actual_video_path}"

    try:
        showrunner = Showrunner()

        # Get creative direction
        if mood == "auto":
            direction = showrunner.analyze_video(actual_video_path)
            detected_mood = direction['mood']
            intro_script = direction['intro_script']
            title_text = direction['title_card']
        else:
            detected_mood = mood
            intro_script = "Get ready for something incredible..."
            title_text = "WATCH THIS"

        # Voiceover
        intro_audio = showrunner.generate_intro(intro_script)
        if not intro_audio or not os.path.exists(intro_audio) or os.path.getsize(intro_audio) == 0:
            logger.warning(
                "Voiceover generation failed. Continuing without intro audio...")
            intro_audio = None

        # Music
        bg_music = showrunner.select_music(detected_mood)

        # Final composition
        final_video = showrunner.compose_final(
            actual_video_path,
            intro_audio,
            bg_music,
            title_text=title_text,
            mood=detected_mood,
            enable_smart_crop=enable_smart_crop,
            add_intro_image=add_intro_image,
            add_subtitles=add_subtitles
        )

        if final_video:
            return f"Success! Polished video saved to: {final_video}"
        else:
            return "Production failed during composition."

    except Exception as e:
        import traceback
        error_msg = f"Error during production: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def add_production_wrapper(video_file, mood_override, enable_smart_crop, add_intro_image, add_subtitles, progress=gr.Progress()):
    """
    Production Studio wrapper function for Gradio UI.

    Adds professional production value to videos with AI-powered features:
    - Smart vertical crop (9:16 with subject tracking)
    - AI-generated intro images (FLUX)
    - Professional voiceover (ElevenLabs)
    - Background music (mood-matched)
    - Auto-generated subtitles (WhisperX)

    RECOMMENDED FOR MCP: Use video_file='auto' to automatically use the last rendered 
    video from Auto Mode or Manual Mode. This is the smoothest approach after running 
    the step-by-step workflow.

    Args:
        video_file: 'auto' (recommended) to use last rendered video, or path to video file.
            - 'auto': Automatically uses most recent render from Auto/Manual Mode
            - Path: Direct file path to a video
            - Gradio URL: Will be converted to local path automatically
        mood_override: Video mood selection. Options:
            - "auto": AI analyzes video and detects mood automatically (recommended)
            - "hype": Upbeat, energetic music and style
            - "suspense": Tension-building music and style
            - "chill": Relaxed, calm music and style
        enable_smart_crop: Boolean flag to enable AI-powered smart crop to 9:16 portrait format.
            Uses Gemini + Qwen VL for subject tracking. Recommended for landscape videos.
        add_intro_image: Boolean flag to generate AI intro image using FLUX model.
            Creates custom title card based on video content. Requires NEBIUS_API_KEY.
        add_subtitles: Boolean flag to add word-level subtitles using WhisperX.
            Creates TikTok-style captions burned into video. Includes intro voiceover.
        progress: Gradio Progress object for tracking progress (automatically provided).

    Yields:
        Generator that yields tuples of (progress_text, video_path):
        - progress_text: Status messages showing production progress
        - video_path: Path to the polished video file, or None if failed
    """
    # Handle 'auto' or None - automatically use the last rendered video
    actual_video_path = video_file

    if video_file is None or video_file == 'auto' or (isinstance(video_file, str) and video_file.strip() == ''):
        # Try to get the most recent render from Auto or Manual mode
        detected_path, source = get_most_recent_render()
        if detected_path:
            actual_video_path = detected_path
            yield f"🔄 Auto-detected video from {source}: {os.path.basename(detected_path)}", None
        else:
            yield "❌ No video available. Please upload a video, use 'Load Last Render', or run Auto/Manual Mode first.", None
            return

    # Handle Gradio URL format - extract local path
    elif isinstance(video_file, str) and video_file.startswith('http://localhost'):
        # Try to extract local path from Gradio URL: http://localhost:7860/gradio_api/file=/path/to/file.mp4
        if '/gradio_api/file=' in video_file:
            local_path = video_file.split('/gradio_api/file=')[-1]
            if os.path.exists(local_path):
                actual_video_path = local_path
                yield f"🔄 Extracted local path from URL: {os.path.basename(local_path)}", None
            else:
                # Fallback to last render
                detected_path, source = get_most_recent_render()
                if detected_path:
                    actual_video_path = detected_path
                    yield f"🔄 URL path not found, using last render from {source}", None
                else:
                    yield f"❌ Could not find video at URL path: {local_path}", None
                    return
        else:
            # Unknown URL format, try last render
            detected_path, source = get_most_recent_render()
            if detected_path:
                actual_video_path = detected_path
                yield f"🔄 Using last render from {source}", None
            else:
                yield "❌ Could not resolve video URL. Please use 'auto' or upload a video.", None
                return

    # Verify the video file exists
    if not os.path.exists(actual_video_path):
        # One more fallback - try last render
        detected_path, source = get_most_recent_render()
        if detected_path:
            actual_video_path = detected_path
            yield f"🔄 File not found, using last render from {source}", None
        else:
            yield f"❌ Video file not found: {actual_video_path}", None
            return

    try:
        progress(0.05, desc="Initializing Showrunner...")
        yield "🎬 Showrunner is analyzing your video...", None

        showrunner = Showrunner()

        # Get creative direction
        if mood_override == "auto":
            yield "🧠 AI analyzing video mood and content...", None
            direction = showrunner.analyze_video(actual_video_path)
            mood = direction['mood']
            intro_script = direction['intro_script']
            title_text = direction['title_card']
        else:
            mood = mood_override
            intro_script = "Get ready for something incredible..."
            title_text = "WATCH THIS"

        yield f"📝 Creative Direction:\n  Mood: {mood}\n  Hook: {intro_script}\n  Title: {title_text}", None

        # Progress indicators for enabled features
        if enable_smart_crop:
            yield "🎯 Smart Crop: Gemini analyzing scene changes...", None
            yield "👁️ Smart Crop: Qwen analyzing subject positions...", None

        if add_intro_image:
            yield "🖼️ FLUX generating intro image...", None

        # Voiceover
        progress(0.3, desc="Generating Voiceover...")
        yield "🎤 ElevenLabs generating voiceover...", None
        intro_audio = showrunner.generate_intro(intro_script)

        if not intro_audio or not os.path.exists(intro_audio) or os.path.getsize(intro_audio) == 0:
            yield "⚠️ Warning: Voiceover generation failed. Check ELEVENLABS_API_KEY. Continuing without intro audio...", None
            intro_audio = None
        else:
            yield f"✅ Voiceover generated ({os.path.getsize(intro_audio)} bytes)", None

        # Music
        progress(0.5, desc="Selecting Music...")
        yield "🎵 Selecting background music...", None
        bg_music = showrunner.select_music(mood)

        # Final composition
        progress(0.7, desc="Composing Video...")
        yield "✂️ Composing final video with all features...", None

        if add_subtitles:
            yield "📝 WhisperX will transcribe complete video (including intro voiceover)...", None

        final_video = showrunner.compose_final(
            actual_video_path,
            intro_audio,
            bg_music,
            title_text=title_text,
            mood=mood,
            enable_smart_crop=enable_smart_crop,
            add_intro_image=add_intro_image,
            add_subtitles=add_subtitles
        )

        if final_video and os.path.exists(final_video):
            # Copy to output directory for persistence
            output_filename = f"production_{int(time.time())}.mp4"
            final_output_path = os.path.join(OUTPUT_DIR, output_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.copy(final_video, final_output_path)

            progress(1.0, desc="Complete!")
            yield f"✅ Production complete! Saved to: {final_output_path}", final_output_path
        else:
            yield "❌ Production failed during composition.", None

    except Exception as e:
        import traceback
        error_msg = f"❌ Error during production:\n{str(e)}\n\nDebug:\n{traceback.format_exc()}"
        yield error_msg, None


def reset_workflow():
    """
    Reset all workflow state.

    Clears all state from the step-by-step workflow, including:
    - Video URL and metadata
    - Classification
    - Hotspots and verified hotspots
    - Clips metadata
    - Edit plan and final plan
    - Temporary directories

    This function is used to start a new workflow from scratch.

    Returns:
        Tuple of reset values for all Gradio UI components:
        - Empty strings for all text outputs
        - None for video output
        - Disabled state for all step buttons (except step 1)
    """
    if workflow_state['temp_dir'] and os.path.exists(workflow_state['temp_dir']):
        try:
            shutil.rmtree(workflow_state['temp_dir'])
        except:
            pass

    for key in workflow_state:
        if key == 'temp_dir':
            workflow_state[key] = None
        elif isinstance(workflow_state[key], list):
            workflow_state[key] = []
        else:
            workflow_state[key] = None

    return (
        "",  # step1_output
        "",  # step2_output
        "",  # step3_output
        "",  # step4_output
        "",  # plan_json
        "",  # step5_output
        None,  # video_output
        gr.update(interactive=False),  # btn2
        gr.update(interactive=False),  # btn3
        gr.update(interactive=False),  # btn4
        gr.update(interactive=False),  # btn5
    )


# ============================================================================
# MCP RESOURCES - Read-only data exposed to MCP clients
# ============================================================================

@gr.mcp.resource("resource://video_classification_criteria")
def get_classification_criteria() -> dict:
    """
    Criteria and rules used by Director's Cut to classify videos as podcast vs generic.

    This resource provides transparency into the classification system, showing:
    - Known podcast channels that trigger podcast classification
    - Keywords that indicate podcast vs generic content
    - Semantic triggers used to find interesting moments in each pipeline

    **Classification Priority:**
    1. Channel/Uploader match (highest confidence)
    2. Title keywords
    3. Description keywords
    4. Tags
    5. Duration + keyword combination

    **Use Cases:**
    - Understanding why a video was classified a certain way
    - Debugging misclassifications
    - Extending classification criteria
    - Understanding semantic trigger system
    """
    return {
        "podcast_channels": [
            "joe rogan", "powerfuljre", "jre clips",
            "lex fridman", "lex clips",
            "huberman lab", "andrew huberman",
            "all-in podcast", "all-in pod",
            "diary of a ceo", "impact theory",
            "tim ferriss", "smartless", "ted talks", "ted",
            "flagrant", "flagrant 2"
        ],
        "podcast_keywords": [
            "podcast", "interview", "talk show", "conversation",
            "episode", "ep ", "#ep", "hosted by", "discussion",
            "with guest", "full episode", "guest on", "sits down with",
            "in conversation", "chat with"
        ],
        "generic_keywords": [
            "tutorial", "how to", "guide", "demo", "review",
            "unboxing", "gameplay", "walkthrough", "diy",
            "explained", "breakdown", "analysis of"
        ],
        "podcast_triggers": [
            "unpopular opinion", "controversial take", "here's the secret",
            "nobody talks about", "breakthrough moment", "funniest thing",
            "best advice", "biggest mistake", "game changer", "mind blowing",
            "never told anyone", "listen to this", "key takeaway",
            "bottom line is", "reason why", "truth is", "i realized"
        ],
        "generic_triggers": [
            "let me show you", "check this out", "watch this",
            "before and after", "transformation", "can't believe"
        ],
        "classification_notes": {
            "podcast_pipeline": "Semantic-heavy: Uses transcript triggers + audio filtering. Best for interviews, conversations, long-form discussions.",
            "generic_pipeline": "Multimodal: Uses audio peaks + visual triggers + deep AI verification. Best for tutorials, demos, visual content.",
            "duration_threshold": "Videos >15 minutes with podcast indicators are classified as podcast",
            "fallback": "Default classification is 'generic' if no strong signals found"
        }
    }


@gr.mcp.resource("resource://supported_features")
def get_supported_features() -> dict:
    """
    Comprehensive list of features and capabilities supported by Director's Cut.

    This resource provides an overview of all available features, pipelines, and options
    to help users understand what Director's Cut can do.

    **Architecture Components:**
    - Scout: Signal processing for hotspot detection
    - Verifier: Vision AI for clip quality validation
    - Director: AI edit planning with Gemini 2.0 Flash Lite
    - Hands: Video rendering and composition
    - Showrunner: Production enhancements (crop, intro, subtitles, music)
    """
    return {
        "pipelines": {
            "podcast": {
                "description": "Semantic-heavy pipeline optimized for interview/conversation content",
                "method": "Transcript triggers + audio filtering + standard verification",
                "best_for": "Long-form interviews, podcasts, conversations, talk shows",
                "hotspot_detection": "Semantic triggers in transcript (e.g., 'unpopular opinion', 'game changer')",
                "verification": "Standard vision AI verification",
                "output_duration": "30-60 seconds"
            },
            "generic": {
                "description": "Multimodal analysis pipeline for visual/tutorial content",
                "method": "Audio peaks + visual triggers + deep AI verification",
                "best_for": "Tutorials, demos, visual content, how-to videos",
                "hotspot_detection": "Audio energy peaks + transcript visual triggers",
                "verification": "Deep verification with Gemini 2.0 Flash (viral potential scoring)",
                "output_duration": "30-60 seconds"
            }
        },
        "production_features": {
            "smart_crop": {
                "description": "AI-powered 9:16 crop with intelligent subject tracking",
                "technology": "Gemini 2.0 Flash Lite (scene analysis) + Qwen 2.5-VL-72B (subject detection)",
                "output_format": "9:16 portrait (e.g., 607x1080 from 1920x1080 source)",
                "processing_time": "~30-50 seconds for 15-second video",
                "fallback": "Center crop if AI analysis fails"
            },
            "intro_image": {
                "description": "AI-generated intro screens with custom title cards",
                "technology": "FLUX model via Nebius AI Studio",
                "customization": "Title text generated from video analysis",
                "requirements": "NEBIUS_API_KEY"
            },
            "subtitles": {
                "description": "Word-level subtitles with TikTok-style formatting",
                "technology": "WhisperX for transcription",
                "features": "Burned-in captions, includes intro voiceover",
                "format": "Word-level timing for natural appearance"
            },
            "voiceover": {
                "description": "Professional AI-generated intro voiceover",
                "technology": "ElevenLabs API",
                "script_generation": "AI-generated based on video content analysis",
                "requirements": "ELEVENLABS_API_KEY (optional - continues without if missing)"
            },
            "background_music": {
                "description": "Mood-matched background music",
                "moods": ["hype", "suspense", "chill"],
                "selection": "Auto-detected from video or manually specified",
                "source": "Local music files in assets/music/{mood}/ directories"
            }
        },
        "moods": {
            "auto": "AI analyzes video content and detects mood automatically",
            "hype": "Upbeat, energetic music and production style",
            "suspense": "Tension-building music and production style",
            "chill": "Relaxed, calm music and production style"
        },
        "output_formats": {
            "aspect_ratio": "9:16 (portrait) for TikTok/Instagram Reels/YouTube Shorts",
            "duration": "30-60 seconds (optimized for short-form platforms)",
            "codec": "H.264 video, AAC audio",
            "resolution": "1080x1920 (9:16) or maintains source resolution"
        },
        "api_requirements": {
            "required": ["GEMINI_API_KEY"],
            "optional": ["VIDEO_API_KEY", "ELEVENLABS_API_KEY", "NEBIUS_API_KEY"],
            "notes": "Some features gracefully degrade if optional APIs are unavailable"
        }
    }


# ============================================================================
# MCP PROMPTS - Template prompts for common workflows
# ============================================================================

@gr.mcp.prompt()
def complete_video_editing_workflow() -> str:
    """
    Complete end-to-end workflow for creating polished viral clips from YouTube videos.

    This is the recommended workflow for Claude Desktop users. It provides a seamless
    5-step process followed by automatic production enhancement.

    **Use this prompt when:**
    - You want to create a polished viral clip from a YouTube video
    - You want the full experience: editing + production value
    - You're using Claude Desktop with Director's Cut as an MCP server
    """
    return """# Complete Video Editing Workflow (Recommended for Claude Desktop)

## Overview
Transform any YouTube video into a polished, social media-ready short clip (30-60 seconds) with professional production value.

## The 5-Step Process + Production

### Step 1: Analyze Video
```
step1_analyze_video_mcp(youtube_url="https://youtube.com/watch?v=...")
```
- Downloads video metadata
- Classifies as "podcast" or "generic"
- Determines which editing pipeline to use

### Step 2: Scout Hotspots
```
step2_scout_hotspots_mcp(youtube_url="https://youtube.com/watch?v=...")
```
- Finds interesting moments using AI analysis
- Podcast: Semantic transcript triggers
- Generic: Audio peaks + visual triggers

### Step 3: Verify Clips
```
step3_verify_hotspots_mcp(youtube_url="https://youtube.com/watch?v=...")
```
- Downloads video segments around hotspots
- Verifies quality with Vision AI
- Filters out low-quality clips

### Step 4: Create Edit Plan
```
step4_create_plan_mcp()
```
- AI creates optimal edit plan
- Selects 3-5 best clips
- Ensures 30-60 second duration

### Step 5 + Production (RECOMMENDED)
```
render_and_produce_mcp(mood="auto", enable_smart_crop=True, add_intro_image=True, add_subtitles=True)
```
This combines Step 5 (rendering) with full production value:
- 🎯 Smart vertical crop (9:16 with AI subject tracking)
- 🖼️ AI-generated intro image (FLUX)
- 🎤 Professional voiceover (ElevenLabs)
- 🎵 Mood-matched background music
- 📝 TikTok-style subtitles (WhisperX)

**OR** if you want just the rendered video without production:
```
step5_render_video_mcp()
```
Then optionally add production later:
```
add_production_value_mcp()  # Auto-uses the step 5 output!
```

## Quick Start Example

For a single YouTube video:
1. `step1_analyze_video_mcp(youtube_url="YOUR_URL")`
2. `step2_scout_hotspots_mcp(youtube_url="YOUR_URL")`
3. `step3_verify_hotspots_mcp(youtube_url="YOUR_URL")`
4. `step4_create_plan_mcp()`
5. `render_and_produce_mcp()` ← Gets polished final video!

## Checking Progress

Use `get_workflow_state_mcp()` at any time to:
- See which steps are completed
- Get the path to rendered videos
- Check if a video is ready for production

## Output Locations
- Rendered videos land in the runtime output dir
- Production videos also save to the runtime output dir

## Tips for Best Results
- Longer videos (>15 min) give more hotspot candidates
- "auto" mood detection works best for most videos
- Smart crop is highly recommended for landscape videos
- Subtitles significantly increase engagement"""


@gr.mcp.prompt()
def podcast_editing_workflow() -> str:
    """
    Complete workflow guide for editing podcast-style videos with Director's Cut.

    This prompt provides step-by-step instructions for processing long-form podcast/interview
    content into viral short-form clips optimized for social media platforms.

    **When to Use:**
    - Editing interview videos, podcasts, conversations, talk shows
    - Long-form content (>15 minutes) with dialogue/conversation
    - Content where interesting quotes/moments are more important than visual action
    """
    return """# Podcast Video Editing Workflow

## Overview
Transform long-form podcast/interview content into viral 30-60 second clips using Director's Cut's semantic-heavy pipeline.

## Step-by-Step Process

### Step 1: Classify the Video
Use `classify_video_type(youtube_url)` to verify the video will be processed as a podcast.
- This helps confirm the correct pipeline will be used
- Check the response to see classification reasoning
- Podcast classification triggers: known channels (Joe Rogan, Lex Fridman, etc.), keywords ("podcast", "interview", "episode"), long duration with conversation indicators

### Step 2: Process the Full Pipeline
Use `process_video(youtube_url)` to run the complete autonomous editing pipeline.

**What happens automatically:**
1. **Scout Phase**: 
   - Analyzes transcript for semantic triggers ("unpopular opinion", "game changer", "mind blowing", etc.)
   - Filters out bad audio segments (silence, noise)
   - Falls back to audio peaks if no semantic triggers found
   - Returns top 8 hotspot candidates

2. **Verification Phase**:
   - Downloads short clips around each hotspot (±2 seconds padding)
   - Verifies clip quality with standard vision AI
   - Filters clips with score > 4/10

3. **Director Phase**:
   - AI (Gemini 2.0 Flash Lite) creates edit plan
   - Selects 3-5 best clips (8-15 seconds each)
   - Ensures total duration 30-60 seconds
   - Arranges clips chronologically or for narrative flow

4. **Hands Phase**:
   - Renders final video by concatenating selected clips
   - Output saved to the runtime output directory

### Step 3: Add Production Value (Optional)
Use `add_production_value(video_path, mood="auto", enable_smart_crop=True, add_intro_image=True, add_subtitles=True)` to polish the video.

**Production features:**
- **Smart Crop**: Converts to 9:16 portrait format with AI subject tracking
- **Intro Image**: FLUX-generated title card based on video content
- **Voiceover**: ElevenLabs professional intro voiceover
- **Background Music**: Mood-matched music (auto-detected or specified)
- **Subtitles**: WhisperX word-level captions in TikTok style

## Pipeline Characteristics

**Podcast Pipeline Advantages:**
- Semantic analysis finds interesting quotes and moments
- Transcript-based detection is more reliable for dialogue-heavy content
- Filters out silence and bad audio automatically
- Optimized for finding "viral moments" in conversations

**Best For:**
- Joe Rogan Experience clips
- Lex Fridman interviews
- Huberman Lab episodes
- Any long-form interview/conversation content

**Output:**
- 30-60 second edited video
- Ready for TikTok, Instagram Reels, YouTube Shorts
- Can be further enhanced with production features

## Tips
- For best results, use videos with available transcripts
- Longer videos (>30 min) generate more hotspot candidates
- Production features add significant polish but increase processing time
- Smart crop is recommended for landscape source videos"""


@gr.mcp.prompt()
def generic_video_editing_workflow() -> str:
    """
    Complete workflow guide for editing generic/tutorial videos with Director's Cut.

    This prompt provides step-by-step instructions for processing visual content, tutorials,
    demos, and other non-podcast videos into viral short-form clips.

    **When to Use:**
    - Editing tutorial videos, how-to guides, demos
    - Visual content where action/movement is important
    - Short-form content or videos without strong podcast signals
    - Content where visual interest matters more than dialogue
    """
    return """# Generic Video Editing Workflow

## Overview
Transform tutorial/demo/visual content into viral 30-60 second clips using Director's Cut's multimodal analysis pipeline with deep AI verification.

## Step-by-Step Process

### Step 1: Classify the Video
Use `classify_video_type(youtube_url)` to verify the video will be processed as generic.
- Generic classification triggers: tutorial keywords ("how to", "tutorial", "guide"), visual triggers, or absence of podcast signals
- Check the response to see classification reasoning

### Step 2: Process the Full Pipeline
Use `process_video(youtube_url)` to run the complete autonomous editing pipeline.

**What happens automatically:**
1. **Scout Phase**: 
   - Analyzes audio for energy/loudness peaks (primary signal)
   - Checks transcript for visual triggers ("let me show you", "check this out", "watch this")
   - Combines audio and visual signals
   - Returns top 5 hotspot candidates

2. **Verification Phase**:
   - Downloads clips around each hotspot (±5 seconds padding for more context)
   - **Deep Verification**: Uses Gemini 2.0 Flash to score viral potential (1-10 scale)
   - Only clips with score > 7/10 pass verification
   - Falls back to top 2 clips if none pass deep verification

3. **Director Phase**:
   - AI (Gemini 2.0 Flash Lite) creates edit plan from verified clips
   - Selects 3-5 best moments (8-15 seconds each)
   - Ensures total duration 30-60 seconds
   - Optimizes for visual interest and engagement

4. **Hands Phase**:
   - Renders final video by concatenating selected clips
   - Output saved to the runtime output directory

### Step 3: Add Production Value (Optional)
Use `add_production_value(video_path, mood="auto", enable_smart_crop=True, add_intro_image=True, add_subtitles=True)` to polish the video.

**Production features:**
- **Smart Crop**: Converts to 9:16 portrait format with AI subject tracking
- **Intro Image**: FLUX-generated title card based on video content
- **Voiceover**: ElevenLabs professional intro voiceover
- **Background Music**: Mood-matched music (auto-detected or specified)
- **Subtitles**: WhisperX word-level captions in TikTok style

## Pipeline Characteristics

**Generic Pipeline Advantages:**
- Multimodal analysis (audio + visual + transcript)
- Deep AI verification ensures high-quality clips
- Better for visual content and action-heavy videos
- Optimized for finding engaging visual moments

**Deep Verification Process:**
- Uploads clips to Gemini 2.0 Flash
- Scores viral potential based on visual motion and interest
- Only high-scoring clips (>7/10) are included
- Ensures output quality is maximized

**Best For:**
- Tutorial videos ("How to...", "Guide to...")
- Product demos and reviews
- Visual transformations ("before and after")
- Gameplay and walkthroughs
- Any content where visual action is key

**Output:**
- 30-60 second edited video
- High-quality clips verified by AI
- Ready for TikTok, Instagram Reels, YouTube Shorts
- Can be further enhanced with production features

## Tips
- Deep verification adds processing time but improves quality
- Visual triggers in transcript help find key moments
- Smart crop is highly recommended for landscape tutorials
- Production features add significant polish for social media
- Works best with videos that have clear visual highlights"""


# Gradio Interface
# Gradio Interface
with gr.Blocks(title="Director's Cut") as app:
    gr.Markdown("# 🎬 Director's Cut - Autonomous Video Editor")

    with gr.Tabs():
        with gr.Tab("📹 Create Clip"):
            gr.Markdown(
                "**Step-by-Step Autonomous Video Editor** powered by Gemini 2.0")

            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtube.com/watch?v=...",
                    scale=4
                )
                reset_btn = gr.Button("🔄 Reset", scale=1, variant="secondary")

            # Step 1
            with gr.Group():
                gr.Markdown("### Step 1: Analyze Video")
                with gr.Row():
                    step1_btn = gr.Button(
                        "1️⃣ Analyze & Classify", variant="primary")
                step1_output = gr.Markdown()

            # Step 2
            with gr.Group():
                gr.Markdown("### Step 2: Scout Hotspots")
                num_hotspots_slider = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Hotspots to Find",
                    info="More hotspots = longer processing but more options"
                )
                with gr.Row():
                    step2_btn = gr.Button(
                        "2️⃣ Scout (LLM + Audio)", interactive=False)
                step2_output = gr.Markdown()

            # Step 3
            with gr.Group():
                gr.Markdown("### Step 3: Verify with Vision AI")
                with gr.Row():
                    step3_btn = gr.Button(
                        "3️⃣ Download & Verify Clips", interactive=False)
                step3_output = gr.Markdown()

            # Step 4
            with gr.Group():
                gr.Markdown("### Step 4: Director Creates Plan")
                with gr.Row():
                    step4_btn = gr.Button(
                        "4️⃣ Generate Edit Plan", interactive=False)
                step4_output = gr.Markdown()
                plan_json = gr.Code(label="Edit Plan (JSON)", language="json")

            # Step 5
            with gr.Group():
                gr.Markdown("### Step 5: Render Final Video")
                with gr.Row():
                    step5_btn = gr.Button("5️⃣ Render Video",
                                          interactive=False, variant="primary")
                step5_output = gr.Markdown()
                video_output = gr.Video(label="Final Edit")

            # Event handlers for Tab 1
            step1_btn.click(
                fn=step1_analyze_video,
                inputs=[url_input],
                outputs=[step1_output, step2_btn]
            )

            step2_btn.click(
                fn=step2_scout_hotspots,
                inputs=[url_input, num_hotspots_slider],
                outputs=[step2_output, step3_btn]
            )

            step3_btn.click(
                fn=step3_verify_hotspots,
                inputs=[url_input],
                outputs=[step3_output, step4_btn]
            )

            step4_btn.click(
                fn=step4_create_plan,
                inputs=[],
                outputs=[step4_output, step5_btn, plan_json]
            )

            step5_btn.click(
                fn=step5_render_video,
                inputs=[],
                outputs=[step5_output, video_output]
            )

            reset_btn.click(
                fn=reset_workflow,
                inputs=[],
                outputs=[
                    step1_output, step2_output, step3_output,
                    step4_output, plan_json, step5_output, video_output,
                    step2_btn, step3_btn, step4_btn, step5_btn
                ]
            )

        with gr.Tab("🎯 Smart Crop"):
            gr.Markdown("""
            ## AI-Powered Smart Crop to 9:16
            Test the smart crop feature independently:
            - 🧠 Gemini analyzes scene changes
            - 👁️ Qwen VL detects subject positions
            - ✂️ Smooth interpolated crop tracking
            
            **Use this to test cropping before adding production value!**
            """)

            crop_video_input = gr.Video(label="Upload Video to Crop")
            crop_btn = gr.Button("🎯 Apply Smart Crop",
                                 variant="primary", size="lg")

            crop_progress = gr.Textbox(
                label="Crop Progress", lines=8, interactive=False)
            crop_video_output = gr.Video(
                label="Cropped Video (9:16)", height=400)

            def smart_crop_only(video_file, progress=gr.Progress()):
                """
                Standalone smart crop function for testing AI-powered vertical crop.

                Applies AI-powered smart crop to convert landscape videos to 9:16 portrait format.
                Uses Gemini 2.0 Flash Lite for scene analysis and Qwen 2.5-VL-72B for subject
                position detection, then applies smooth interpolated crop tracking.

                This is a standalone function for testing the crop feature independently before
                adding full production value. If a user passes a file as an input, use the
                upload_file_to_gradio tool, if present, to upload the file to the gradio app
                and create a Gradio File Input. Then use the returned path as the input to the tool.

                Args:
                    video_file: Path to the input video file or Gradio File object.
                        Must be a valid video file (MP4, MOV, AVI, etc.). If None, returns error.
                    progress: Gradio Progress object for tracking progress (automatically provided).

                Returns:
                    Generator that yields tuples of (progress_text, video_path):
                    - progress_text: Status messages showing crop progress
                    - video_path: Path to the cropped video file (9:16 format), or None if failed

                Yields:
                    Progress updates during processing, then final result with cropped video path.
                """
                if video_file is None:
                    return "❌ Please upload a video first", None

                try:
                    progress(0.1, desc="Initializing Smart Crop...")
                    yield "🎯 Starting smart crop pipeline...", None

                    showrunner = Showrunner()

                    progress(0.2, desc="Gemini analyzing scene changes...")
                    yield "🧠 Gemini Flash Lite analyzing video for scene changes...", None

                    progress(0.5, desc="Qwen VL analyzing subject positions...")
                    yield "👁️ Qwen VL analyzing subject positions at keyframes...", None

                    progress(0.7, desc="Applying dynamic crop...")
                    yield "✂️ Applying smooth interpolated crop...", None

                    # Run smart crop pipeline
                    cropped_video_path = showrunner.smart_crop_pipeline(
                        video_file)

                    if cropped_video_path and os.path.exists(cropped_video_path):
                        progress(1.0, desc="Complete!")

                        # Copy to output directory for persistence
                        output_filename = f"smart_crop_{int(time.time())}.mp4"
                        final_path = os.path.join(
                            OUTPUT_DIR, output_filename)
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        shutil.copy(cropped_video_path, final_path)

                        yield f"✅ Smart crop complete! Saved to: {final_path}", final_path
                    else:
                        yield "❌ Smart crop failed. Check logs for details.", None

                except Exception as e:
                    import traceback
                    error_msg = f"❌ Error during smart crop: {str(e)}\n\nDebug info:\n{traceback.format_exc()}"
                    yield error_msg, None

            crop_btn.click(
                fn=smart_crop_only,
                inputs=[crop_video_input],
                outputs=[crop_progress, crop_video_output]
            )

        with gr.Tab("🎙️ Production Studio"):
            gr.Markdown("""
            ## Professional Video Production
            Transform your video with AI-powered features:
            - 🧠 Smart vertical crop that follows subjects
            - 🖼️ AI-generated intro screen
            - 📝 Auto-generated subtitles
            - 🎤 Professional voiceover
            - 🎵 Background music
            - 🔁 Load your latest render from Tab 1 with **Load Last Render**
            """)

            video_input_2 = gr.Video(
                label="Upload Video (from Tab 1 or any video)")

            with gr.Row():
                load_render_btn = gr.Button(
                    "⬇️ Load Last Render", variant="secondary")
            load_render_status = gr.Markdown()

            with gr.Row():
                mood_override = gr.Dropdown(
                    choices=["auto", "hype", "suspense", "chill"],
                    value="auto",
                    label="Mood (auto-detect or choose)"
                )

            with gr.Row():
                enable_smart_crop = gr.Checkbox(
                    label="🎯 Smart Vertical Crop",
                    value=True,
                    info="AI tracks subjects and crops to 9:16"
                )
                add_intro_image = gr.Checkbox(
                    label="🖼️ Generate Intro Image",
                    value=True,
                    info="FLUX creates custom intro screen"
                )
                add_subtitles = gr.Checkbox(
                    label="📝 Add Subtitles",
                    value=True,
                    info="TikTok-style captions"
                )

            produce_btn = gr.Button(
                "✨ Add Production Value", variant="primary", size="lg")

            progress_2 = gr.Textbox(
                label="Production Progress", lines=8, interactive=False)
            video_output_2 = gr.Video(label="Polished Video", height=500)

            produce_btn.click(
                fn=add_production_wrapper,
                inputs=[video_input_2, mood_override,
                        enable_smart_crop, add_intro_image, add_subtitles],
                outputs=[progress_2, video_output_2]
            )

            load_render_btn.click(
                fn=load_last_render_into_production,
                inputs=[],
                outputs=[video_input_2, load_render_status]
            )

        # ==================== MANUAL MODE TAB ====================
        with gr.Tab("🎯 Manual Mode"):
            gr.Markdown("""### You Choose the Topics
            
1. Enter a video URL and analyze it
2. Select which topics you want to include
3. Render your custom edit
""")

            with gr.Row():
                manual_url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtube.com/watch?v=...",
                    scale=4
                )
                manual_reset_btn = gr.Button("🔄 Reset", scale=1)

            # Step 1: Analyze
            with gr.Group():
                gr.Markdown("### Step 1: Analyze & Find Topics")
                manual_step1_btn = gr.Button(
                    "1️⃣ Analyze Video & Extract Topics")
                manual_step1_output = gr.Markdown()

            # Topic Selection
            with gr.Group():
                gr.Markdown("### Step 2: Select Topics")
                manual_topic_checkboxes = gr.CheckboxGroup(
                    choices=["⏳ Run Step 1 first to see available topics..."],
                    label="☑️ Select topics to include in your video",
                    info="Check the topics you want, then click Extract",
                    interactive=False
                )
                manual_step2_btn = gr.Button(
                    "2️⃣ Extract Selected Clips", interactive=False)
                manual_step2_output = gr.Markdown()

            # Step 3: Render
            with gr.Group():
                gr.Markdown("### Step 3: Render Final Video")
                manual_step3_btn = gr.Button(
                    "3️⃣ Render Video", variant="primary", interactive=False)
                manual_step3_output = gr.Markdown()
                manual_video_output = gr.Video(label="Your Custom Edit")

            # Manual Mode Event handlers
            manual_step1_btn.click(
                fn=manual_step1_analyze,
                inputs=[manual_url_input],
                outputs=[manual_step1_output,
                         manual_topic_checkboxes, manual_step2_btn]
            )

            manual_step2_btn.click(
                fn=manual_step2_extract,
                inputs=[manual_topic_checkboxes],
                outputs=[manual_step2_output, manual_step3_btn]
            )

            manual_step3_btn.click(
                fn=manual_step3_render,
                inputs=[],
                outputs=[manual_step3_output, manual_video_output]
            )

            manual_reset_btn.click(
                fn=reset_manual_workflow,
                inputs=[],
                outputs=[
                    manual_step1_output, manual_topic_checkboxes,
                    manual_step2_output, manual_step3_output, manual_video_output,
                    manual_step2_btn, manual_step3_btn
                ]
            )

if __name__ == "__main__":
    # Launch with MCP server enabled
    try:
        import sys
        import re
        from io import StringIO

        # Custom stdout wrapper to capture share URL
        class ShareURLCapture:
            def __init__(self, original_stdout):
                self.original_stdout = original_stdout
                self.share_url = None

            def write(self, text):
                # Write to original stdout
                self.original_stdout.write(text)
                # Try to extract share URL
                if "Running on public URL:" in text or "gradio.live" in text or "gradio.app" in text:
                    # Extract URL from the text
                    url_match = re.search(
                        r'https?://[^\s]+(?:gradio\.live|gradio\.app)[^\s]*', text)
                    if url_match:
                        self.share_url = url_match.group(0)
                        # Save to file
                        try:
                            with open("gradio_share_url.txt", "w") as f:
                                f.write(self.share_url)
                            self.original_stdout.write(
                                f"\n✅ Public share URL saved to: gradio_share_url.txt\n")
                            self.original_stdout.write(
                                f"🔗 Share this URL with collaborators: {self.share_url}\n\n")
                        except Exception as e:
                            pass
                return len(text)

            def flush(self):
                self.original_stdout.flush()

            def isatty(self):
                return self.original_stdout.isatty()

            def __getattr__(self, name):
                # Delegate any other attributes to original stdout
                return getattr(self.original_stdout, name)

        # Wrap stdout to capture share URL
        url_capture = ShareURLCapture(sys.stdout)
        sys.stdout = url_capture

        print("🚀 Starting Gradio app...")
        print("📡 Local URL: http://0.0.0.0:7860")
        print("⏳ Creating public share link (this may take a few seconds)...")

        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            mcp_server=True,
            share=False
        )
    except OSError as e:
        if "address already in use" in str(e) or "port" in str(e).lower():
            logger.error(f"Port 7860 is already in use. Please:")
            logger.error(
                "1. Stop the existing process: lsof -ti:7860 | xargs kill")
            logger.error(
                "2. Or use a different port by setting GRADIO_SERVER_PORT environment variable")
            logger.error(f"   Example: GRADIO_SERVER_PORT=7861 python app.py")
            raise
        else:
            raise
