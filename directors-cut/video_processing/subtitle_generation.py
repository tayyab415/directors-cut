"""
Automatic Subtitle Generation for Director's Cut

This module provides functions for transcribing speech and
generating/burning subtitles into videos.
"""

import os
import tempfile
from typing import Optional
from dataclasses import dataclass

from .ffmpeg_utils import extract_audio, burn_subtitles


@dataclass
class SubtitleSegment:
    """A single subtitle segment."""
    index: int
    start_time: float
    end_time: float
    text: str


def format_srt_time(seconds: float) -> str:
    """
    Format time in SRT format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        SRT formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[SubtitleSegment]) -> str:
    """
    Convert subtitle segments to SRT format.

    Args:
        segments: List of SubtitleSegment objects

    Returns:
        SRT formatted string
    """
    srt_lines = []

    for seg in segments:
        srt_lines.append(str(seg.index))
        srt_lines.append(f"{format_srt_time(seg.start_time)} --> {format_srt_time(seg.end_time)}")
        srt_lines.append(seg.text)
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def transcribe_audio(
    audio_path: str,
    language: str = "en",
    model_size: str = "base",
) -> tuple[bool, list[SubtitleSegment]]:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code (en, es, fr, etc.)
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Tuple of (success: bool, segments: list[SubtitleSegment])
    """
    try:
        import whisper
    except ImportError:
        return False, []

    try:
        # Load model
        model = whisper.load_model(model_size)

        # Transcribe
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,
        )

        # Convert to SubtitleSegment objects
        segments = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            segments.append(SubtitleSegment(
                index=i,
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip(),
            ))

        return True, segments

    except Exception:
        return False, []


def transcribe_with_api(
    audio_path: str,
    language: str = "en",
    api_key: Optional[str] = None,
) -> tuple[bool, list[SubtitleSegment]]:
    """
    Transcribe audio using OpenAI's Whisper API.

    Args:
        audio_path: Path to audio file
        language: Language code
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

    Returns:
        Tuple of (success: bool, segments: list[SubtitleSegment])
    """
    try:
        from openai import OpenAI
    except ImportError:
        return False, []

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, []

    try:
        client = OpenAI(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for i, seg in enumerate(response.segments or [], start=1):
            segments.append(SubtitleSegment(
                index=i,
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text.strip(),
            ))

        return True, segments

    except Exception:
        return False, []


def generate_srt(
    video_path: str,
    output_path: str,
    language: str = "en",
    use_api: bool = False,
) -> tuple[bool, str]:
    """
    Generate SRT subtitle file from video.

    Args:
        video_path: Path to video file
        output_path: Path for output SRT file
        language: Language code
        use_api: If True, use OpenAI API; otherwise use local Whisper

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    # Extract audio from video
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")

        success, message = extract_audio(video_path, audio_path, format="wav")
        if not success:
            return False, f"Failed to extract audio: {message}"

        # Transcribe
        if use_api:
            success, segments = transcribe_with_api(audio_path, language)
        else:
            success, segments = transcribe_audio(audio_path, language)

        if not success or not segments:
            return False, "Failed to transcribe audio"

        # Generate SRT content
        srt_content = segments_to_srt(segments)

        # Write to file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            return True, output_path
        except Exception as e:
            return False, str(e)


def add_subtitles(
    video_path: str,
    output_path: str,
    srt_path: Optional[str] = None,
    language: str = "en",
    font_size: int = 24,
    use_api: bool = False,
) -> tuple[bool, str]:
    """
    Add burned-in subtitles to a video.

    If srt_path is not provided, subtitles will be auto-generated.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        srt_path: Path to existing SRT file (optional)
        language: Language for transcription (if generating)
        font_size: Subtitle font size
        use_api: Use OpenAI API for transcription

    Returns:
        Tuple of (success: bool, path_or_error: str)
    """
    # Generate SRT if not provided
    temp_srt = None
    if srt_path is None:
        temp_srt = tempfile.mktemp(suffix=".srt")
        success, result = generate_srt(video_path, temp_srt, language, use_api)
        if not success:
            return False, result
        srt_path = temp_srt

    try:
        # Burn subtitles into video
        success, message = burn_subtitles(
            video_path,
            srt_path,
            output_path,
            font_size=font_size,
        )

        if success:
            return True, output_path
        return False, message

    finally:
        # Clean up temp SRT
        if temp_srt and os.path.exists(temp_srt):
            os.remove(temp_srt)


def preview_subtitles(
    video_path: str,
    language: str = "en",
    max_segments: int = 10,
) -> tuple[bool, list[dict]]:
    """
    Preview generated subtitles without burning them.

    Args:
        video_path: Path to video file
        language: Language code
        max_segments: Maximum number of segments to return

    Returns:
        Tuple of (success: bool, segments as list of dicts)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")

        success, message = extract_audio(video_path, audio_path, format="wav")
        if not success:
            return False, []

        success, segments = transcribe_audio(audio_path, language, model_size="tiny")

        if not success:
            return False, []

        # Convert to dicts and limit
        result = []
        for seg in segments[:max_segments]:
            result.append({
                "start": round(seg.start_time, 2),
                "end": round(seg.end_time, 2),
                "text": seg.text,
            })

        return True, result
