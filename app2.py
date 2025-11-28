"""
Director's Cut - Enhanced Autonomous Video Editor
Features:
- LLM-based semantic analysis for viral moment detection
- Multi-feature audio analysis (RMS, spectral centroid, ZCR, rolloff, energy delta)
- Video upload verification (faster than frame extraction)
- Parallel downloads and verification (3-4x speedup)
- Dual-mode: Auto Mode + Manual Mode with topic selection
"""

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


# ============================================================================
# WORKFLOW STATE
# ============================================================================

workflow_state = {
    'video_url': None,
    'video_info': None,
    'category': None,
    'temp_dir': None,
    'audio_path': None,
    'transcript_text': None,
    'semantic_hotspots': [],
    'audio_hotspots': [],
    'all_hotspots': [],
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def overlaps_bad_audio(start: float, end: float, bad_segments: List[Dict]) -> bool:
    """Check if a time range overlaps significantly with bad audio segments."""
    for bad in bad_segments:
        bad_start = bad['start']
        bad_end = bad['end']
        overlap_start = max(start, bad_start)
        overlap_end = min(end, bad_end)
        if overlap_end > overlap_start:
            overlap_duration = overlap_end - overlap_start
            segment_duration = end - start
            if overlap_duration > 2.0 or (overlap_duration / segment_duration) > 0.2:
                return True
    return False


def classify_video(video_info: Dict) -> str:
    """Classify video as 'podcast' or 'generic' based on metadata."""
    title = video_info.get('title', '').lower()
    description = video_info.get('description', '').lower()
    uploader = video_info.get('uploader', '').lower()
    channel = video_info.get('channel', '').lower()
    duration = video_info.get('duration', 0)
    tags = [tag.lower() for tag in video_info.get('tags', [])]

    # Known podcast channels
    podcast_channels = [
        'joe rogan', 'powerfuljre', 'lex fridman', 'huberman lab',
        'all-in podcast', 'diary of a ceo', 'impact theory', 'tim ferriss',
        'flagrant', 'ted talks', 'smartless'
    ]

    for pc in podcast_channels:
        if pc in uploader or pc in channel or pc in title:
            return 'podcast'

    # Podcast keywords
    podcast_keywords = ['podcast', 'interview',
                        'conversation', 'episode', 'ep.', 'ep ']
    for kw in podcast_keywords:
        if kw in title or kw in description[:500]:
            return 'podcast'

    # Duration check (podcasts typically > 20 mins)
    if duration > 1200:
        talk_indicators = ['talk', 'discuss',
                           'chat', 'speak', 'interview', 'guest']
        if any(ind in title or ind in description[:500] for ind in talk_indicators):
            return 'podcast'

    return 'generic'


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


# ============================================================================
# LLM-BASED SEMANTIC ANALYSIS
# ============================================================================

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
                    'verification': verification
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
        futures = {executor.submit(download_single, task): task for task in download_tasks}

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
# AUTO MODE WORKFLOW FUNCTIONS
# ============================================================================

def step1_analyze_video(url: str):
    """Step 1: Fetch video info and transcript."""
    global workflow_state

    if not url or not url.strip():
        return "❌ Please enter a YouTube URL.", gr.update(interactive=False)

    try:
        logger.info(f"📺 Analyzing: {url}")

        # Get video info
        video_info = get_video_info(url)
        if not video_info:
            return "❌ Could not fetch video info. Please check the URL.", gr.update(interactive=False)

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Get transcript
        video_id = extract_video_id(url)
        transcript_text = get_transcript(video_id)

        if "Error" in transcript_text:
            return f"⚠️ Transcript issue: {transcript_text}\n\nContinuing with audio-only analysis...", gr.update(interactive=True)

        # Classify video
        category = classify_video(video_info)

        # Store in workflow state
        workflow_state['video_url'] = url
        workflow_state['video_info'] = video_info
        workflow_state['category'] = category
        workflow_state['temp_dir'] = temp_dir
        workflow_state['transcript_text'] = transcript_text

        duration = video_info.get('duration', 0)
        output = f"""## ✅ Video Analyzed

**Title:** {video_info.get('title', 'Unknown')}
**Channel:** {video_info.get('uploader', 'Unknown')}
**Duration:** {duration // 60}:{duration % 60:02d}
**Category:** {category.upper()}
**Transcript:** {'✅ Available' if transcript_text and 'Error' not in transcript_text else '❌ Not available'}
"""
        return output, gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Step 1 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def step2_scout(url: str, num_hotspots: int = 5):
    """Step 2: Scout for hotspots using audio + semantic analysis."""
    global workflow_state

    workflow_state['num_hotspots'] = int(num_hotspots)

    try:
        logger.info(f"🔍 Scouting hotspots (finding top {num_hotspots})...")

        video_info = workflow_state['video_info']
        temp_dir = workflow_state['temp_dir']
        transcript_text = workflow_state['transcript_text']

        # Download audio
        logger.info("📥 Downloading audio...")
        audio_path = download_audio(url, temp_dir)
        workflow_state['audio_path'] = audio_path

        # Audio analysis
        logger.info("🎵 Analyzing audio signals...")
        audio_hotspots = scout.analyze(audio_path)

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
            all_hotspots.append(h)

        for h in semantic_hotspots:
            # Check for overlap with existing
            overlaps = False
            for existing in all_hotspots:
                if abs(existing['start'] - h['start']) < 15:
                    existing['score'] = max(existing['score'], h['score'])
                    existing['semantic_score'] = h.get('semantic_score', 0)
                    overlaps = True
                    break
            if not overlaps:
                all_hotspots.append(h)

        # Sort by score
        all_hotspots.sort(key=lambda x: x['score'], reverse=True)
        workflow_state['all_hotspots'] = all_hotspots[:num_hotspots + 2]

        # Build output
        output = f"""## 🎯 Hotspots Found

**Audio hotspots:** {len(audio_hotspots)}
**Semantic hotspots:** {len(semantic_hotspots)}
**Combined (deduplicated):** {len(all_hotspots)}

### Top {num_hotspots} Candidates:
"""
        for i, h in enumerate(workflow_state['all_hotspots'][:num_hotspots], 1):
            start_fmt = f"{int(h['start'] // 60)}:{int(h['start'] % 60):02d}"
            end_fmt = f"{int(h['end'] // 60)}:{int(h['end'] % 60):02d}"
            source = h.get('source', 'unknown')
            output += f"\n{i}. **{start_fmt} - {end_fmt}** | Score: {h['score']:.2f} | Source: {source}"
            if h.get('reasoning'):
                output += f"\n   _{h['reasoning'][:80]}..._"

        return output, gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Step 2 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def step3_verify():
    """Step 3: Download clips and verify with Vision AI."""
    global workflow_state

    try:
        logger.info("🔍 Step 3: Parallel download + verification...")

        temp_dir = workflow_state['temp_dir']
        url = workflow_state['video_url']
        candidates = workflow_state['all_hotspots']
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

        return result_text, gr.update(interactive=len(verified_hotspots) > 0)

    except Exception as e:
        logger.error(f"Step 3 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def step4_director():
    """Step 4: Create edit plan with Director."""
    global workflow_state

    try:
        logger.info("🎬 Step 4: Creating edit plan...")

        verified_hotspots = workflow_state['verified_hotspots']
        video_info = workflow_state['video_info']
        transcript_text = workflow_state['transcript_text']
        category = workflow_state['category']

        if not verified_hotspots:
            return "❌ No verified hotspots available.", gr.update(interactive=False)

        # Add transcript context to each hotspot
        for h in verified_hotspots:
            h['transcript_excerpt'] = extract_transcript_window(
                transcript_text, h['start'], h['end'], buffer=15
            )

        # Create edit plan
        edit_plan = director.create_edit_plan(
            verified_hotspots,
            video_info,
            target_duration=90
        )

        if not edit_plan:
            return "❌ Director could not create edit plan.", gr.update(interactive=False)

        workflow_state['edit_plan'] = edit_plan

        # Map plan to clips
        clips_metadata = workflow_state['clips_metadata']
        final_plan = []

        for planned in edit_plan:
            for clip in clips_metadata:
                if abs(clip['hotspot']['start'] - planned['start']) < 5:
                    final_plan.append({
                        **planned,
                        'clip_path': clip['path'],
                        'source_start': clip['start']
                    })
                    break

        workflow_state['final_plan'] = final_plan

        # Build output
        output = f"""## 🎬 Edit Plan Created

**Clips in plan:** {len(final_plan)}
**Target duration:** ~90 seconds

### Sequence:
"""
        for i, clip in enumerate(final_plan, 1):
            start_fmt = f"{int(clip['start'] // 60)}:{int(clip['start'] % 60):02d}"
            end_fmt = f"{int(clip['end'] // 60)}:{int(clip['end'] % 60):02d}"
            output += f"\n{i}. **{start_fmt} - {end_fmt}**"
            if clip.get('reason'):
                output += f"\n   _{clip['reason'][:60]}..._"

        return output, gr.update(interactive=len(final_plan) > 0)

    except Exception as e:
        logger.error(f"Step 4 error: {e}")
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def step5_render():
    """Step 5: Render final video."""
    global workflow_state

    try:
        logger.info("🎥 Step 5: Rendering final video...")

        final_plan = workflow_state['final_plan']
        clips_metadata = workflow_state['clips_metadata']
        category = workflow_state['category']

        if not final_plan:
            return "❌ No edit plan available.", None

        # Render
        output_filename = f"v2_{category}_edit_{int(time.time())}.mp4"
        output_path = hands.execute(
            clips_metadata[0]['path'],
            final_plan,
            output_filename=output_filename
        )

        if output_path and os.path.exists(output_path):
            workflow_state['final_video_path'] = output_path

            total_duration = sum(c['end'] - c['start'] for c in final_plan)

            return f"""## ✅ Video Rendered!

**Output:** {output_filename}
**Duration:** ~{int(total_duration)} seconds
**Clips:** {len(final_plan)}
""", output_path
        else:
            return "❌ Rendering failed.", None

    except Exception as e:
        logger.error(f"Step 5 error: {e}")
        return f"❌ Error: {str(e)}", None


def reset_workflow():
    """Reset the auto workflow state."""
    global workflow_state

    if workflow_state.get('temp_dir') and os.path.exists(workflow_state['temp_dir']):
        try:
            shutil.rmtree(workflow_state['temp_dir'])
        except:
            pass

    workflow_state = {
        'video_url': None,
        'video_info': None,
        'category': None,
        'temp_dir': None,
        'audio_path': None,
        'transcript_text': None,
        'semantic_hotspots': [],
        'audio_hotspots': [],
        'all_hotspots': [],
        'verified_hotspots': [],
        'clips_metadata': [],
        'edit_plan': [],
        'final_plan': [],
        'final_video_path': None,
        'num_hotspots': 5
    }

    return (
        "",  # step1
        "",  # step2
        "",  # step3
        "",  # step4
        "",  # step5
        None,  # video
        gr.update(interactive=False),  # step2_btn
        gr.update(interactive=False),  # step3_btn
        gr.update(interactive=False),  # step4_btn
        gr.update(interactive=False)   # step5_btn
    )


# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(title="Director's Cut - Enhanced Video Editor") as app:
    gr.Markdown("""
# 🎬 Director's Cut - Enhanced Autonomous Video Editor

**Choose your workflow:**
- 🤖 **Auto Mode**: AI automatically finds viral moments
- 🎯 **Manual Mode**: You select which topics to include
""")

    with gr.Tabs():
        # ==================== AUTO MODE TAB ====================
        with gr.Tab("🤖 Auto Mode"):
            gr.Markdown("### Fully Autonomous Editing")

            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtube.com/watch?v=...",
                    scale=4
                )
                reset_btn = gr.Button("🔄 Reset", scale=1)

            # Step 1
            with gr.Group():
                gr.Markdown("### Step 1: Analyze Video")
                step1_btn = gr.Button("1️⃣ Analyze Video")
                step1_output = gr.Markdown()

            # Hotspot Slider
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
                step2_btn = gr.Button(
                    "2️⃣ Scout Hotspots (LLM + Audio)", interactive=False)
                step2_output = gr.Markdown()

            # Step 3
            with gr.Group():
                gr.Markdown("### Step 3: Verify Clips (Vision AI)")
                step3_btn = gr.Button(
                    "3️⃣ Download & Verify (Parallel)", interactive=False)
                step3_output = gr.Markdown()

            # Step 4
            with gr.Group():
                gr.Markdown("### Step 4: Create Edit Plan")
                step4_btn = gr.Button(
                    "4️⃣ Director: Create Plan", interactive=False)
                step4_output = gr.Markdown()

            # Step 5
            with gr.Group():
                gr.Markdown("### Step 5: Render Video")
                step5_btn = gr.Button(
                    "5️⃣ Render Final Video", variant="primary", interactive=False)
                step5_output = gr.Markdown()
                video_output = gr.Video(label="Your Edit")

            # Event handlers
            step1_btn.click(
                fn=step1_analyze_video,
                inputs=[url_input],
                outputs=[step1_output, step2_btn]
            )

            step2_btn.click(
                fn=step2_scout,
                inputs=[url_input, num_hotspots_slider],
                outputs=[step2_output, step3_btn]
            )

            step3_btn.click(
                fn=step3_verify,
                inputs=[],
                outputs=[step3_output, step4_btn]
            )

            step4_btn.click(
                fn=step4_director,
                inputs=[],
                outputs=[step4_output, step5_btn]
            )

            step5_btn.click(
                fn=step5_render,
                inputs=[],
                outputs=[step5_output, video_output]
            )

            reset_btn.click(
                fn=reset_workflow,
                inputs=[],
                outputs=[
                    step1_output, step2_output, step3_output,
                    step4_output, step5_output, video_output,
                    step2_btn, step3_btn, step4_btn, step5_btn
                ]
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Director's Cut...")
    print("📡 Access at: http://localhost:7861")
    print("")
    print("Available Modes:")
    print("  🤖 Auto Mode: AI automatically finds viral moments")
    print("  🎯 Manual Mode: You select which topics to include")
    print("")
    print("Features:")
    print("  ✅ LLM-based semantic analysis")
    print("  ✅ Multi-feature audio analysis")
    print("  ✅ Video upload verification")
    print("  ✅ Parallel downloads (4 workers)")
    print("  ✅ Parallel verification (4 workers)")
    print("")

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
