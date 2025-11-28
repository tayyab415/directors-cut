"""
Viral Content Detection for Director's Cut

This module provides AI-powered analysis to detect the most engaging
segments of a video for viral content creation.
"""

import os
import json
import tempfile
from typing import Optional
from dataclasses import dataclass, asdict

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from .ffmpeg_utils import get_video_info, extract_frames, extract_audio


@dataclass
class SegmentScore:
    """Score for a video segment."""
    start: float
    end: float
    score: float
    motion_score: float
    face_score: float
    scene_change_score: float
    audio_score: float
    color_score: float
    reason: str


@dataclass
class ViralAnalysisResult:
    """Result of viral segment analysis."""
    segments: list[SegmentScore]
    video_duration: float
    analysis_summary: str


# Scoring weights for engagement detection
VIRAL_WEIGHTS = {
    "motion_intensity": 0.30,
    "face_detection": 0.25,
    "scene_changes": 0.20,
    "audio_peaks": 0.15,
    "color_variance": 0.10,
}


def calculate_motion(
    frame1: "np.ndarray",
    frame2: "np.ndarray",
) -> float:
    """
    Calculate motion intensity between two frames.

    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)

    Returns:
        Motion score between 0 and 1
    """
    if not HAS_CV2:
        return 0.5

    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Calculate mean of difference
        motion = np.mean(diff) / 255.0

        # Normalize to 0-1 range (typical motion values are 0.01-0.15)
        normalized = min(1.0, motion * 10)

        return normalized

    except Exception:
        return 0.5


def detect_faces(frame: "np.ndarray") -> tuple[int, float]:
    """
    Detect faces in a frame.

    Args:
        frame: Video frame (numpy array)

    Returns:
        Tuple of (face_count, face_area_ratio)
    """
    if not HAS_CV2:
        return 0, 0.0

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return 0, 0.0

        # Calculate total face area relative to frame
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = sum(w * h for (x, y, w, h) in faces)
        area_ratio = face_area / frame_area

        return len(faces), min(1.0, area_ratio * 5)

    except Exception:
        return 0, 0.0


def calculate_color_variance(frame: "np.ndarray") -> float:
    """
    Calculate color variance/interest in a frame.

    Args:
        frame: Video frame (numpy array)

    Returns:
        Color variance score between 0 and 1
    """
    if not HAS_CV2:
        return 0.5

    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate saturation variance (more saturated = more visually interesting)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation) / 255.0

        # Calculate hue diversity
        hue = hsv[:, :, 0]
        hue_std = np.std(hue) / 180.0

        # Combined score
        color_score = (sat_std + hue_std) / 2
        return min(1.0, color_score * 3)

    except Exception:
        return 0.5


def detect_scene_change(
    frame1: "np.ndarray",
    frame2: "np.ndarray",
    threshold: float = 0.4,
) -> bool:
    """
    Detect if there's a scene change between two frames.

    Args:
        frame1: First frame
        frame2: Second frame
        threshold: Threshold for scene change detection

    Returns:
        True if scene change detected
    """
    if not HAS_CV2:
        return False

    try:
        # Calculate histogram difference
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Compare histograms
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Lower correlation = more different = scene change
        return diff < (1 - threshold)

    except Exception:
        return False


def analyze_audio_energy(
    audio_path: str,
    segment_duration: float = 1.0,
) -> list[float]:
    """
    Analyze audio energy levels over time.

    Args:
        audio_path: Path to audio file
        segment_duration: Duration of each segment in seconds

    Returns:
        List of energy values for each segment
    """
    if not HAS_LIBROSA:
        return []

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Calculate RMS energy
        hop_length = int(sr * segment_duration)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Normalize to 0-1
        if len(rms) > 0 and np.max(rms) > 0:
            normalized = rms / np.max(rms)
            return normalized.tolist()

        return []

    except Exception:
        return []


def score_segment(
    frames: list["np.ndarray"],
    audio_energy: Optional[list[float]] = None,
    segment_index: int = 0,
) -> dict:
    """
    Score a video segment based on engagement metrics.

    Args:
        frames: List of frames in the segment
        audio_energy: Audio energy values for the segment
        segment_index: Index of this segment

    Returns:
        Dictionary with individual scores and combined score
    """
    if not frames or not HAS_CV2:
        return {
            "motion": 0.5,
            "faces": 0.5,
            "scene_changes": 0.5,
            "audio": 0.5,
            "color": 0.5,
            "combined": 0.5,
        }

    # Calculate motion between consecutive frames
    motion_scores = []
    for i in range(len(frames) - 1):
        motion_scores.append(calculate_motion(frames[i], frames[i + 1]))
    avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.5

    # Detect faces across frames
    face_scores = []
    for frame in frames:
        _, area_ratio = detect_faces(frame)
        face_scores.append(area_ratio)
    avg_faces = sum(face_scores) / len(face_scores) if face_scores else 0.0

    # Count scene changes
    scene_changes = 0
    for i in range(len(frames) - 1):
        if detect_scene_change(frames[i], frames[i + 1]):
            scene_changes += 1
    scene_score = min(1.0, scene_changes / 3)  # Normalize, 3+ changes = max

    # Audio score
    if audio_energy and segment_index < len(audio_energy):
        audio_score = audio_energy[segment_index]
    else:
        audio_score = 0.5

    # Color variance
    color_scores = [calculate_color_variance(f) for f in frames]
    avg_color = sum(color_scores) / len(color_scores) if color_scores else 0.5

    # Combined weighted score
    combined = (
        avg_motion * VIRAL_WEIGHTS["motion_intensity"] +
        avg_faces * VIRAL_WEIGHTS["face_detection"] +
        scene_score * VIRAL_WEIGHTS["scene_changes"] +
        audio_score * VIRAL_WEIGHTS["audio_peaks"] +
        avg_color * VIRAL_WEIGHTS["color_variance"]
    )

    return {
        "motion": round(avg_motion, 3),
        "faces": round(avg_faces, 3),
        "scene_changes": round(scene_score, 3),
        "audio": round(audio_score, 3),
        "color": round(avg_color, 3),
        "combined": round(combined, 3),
    }


def analyze_video_segments(
    video_path: str,
    segment_length: int = 30,
    top_n: int = 3,
    fps_sample: float = 1.0,
) -> ViralAnalysisResult:
    """
    Analyze video and return top engaging segments.

    Args:
        video_path: Path to video file
        segment_length: Length of each segment in seconds
        top_n: Number of top segments to return
        fps_sample: Frames per second to sample for analysis

    Returns:
        ViralAnalysisResult with top segments and scores
    """
    info = get_video_info(video_path)
    if not info:
        return ViralAnalysisResult(
            segments=[],
            video_duration=0,
            analysis_summary="Failed to get video info",
        )

    duration = info.duration
    num_segments = max(1, int(duration / segment_length))

    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames
        frames_dir = os.path.join(temp_dir, "frames")
        success, frame_paths = extract_frames(
            video_path, frames_dir, fps=fps_sample
        )

        if not success or not frame_paths:
            return ViralAnalysisResult(
                segments=[],
                video_duration=duration,
                analysis_summary="Failed to extract frames",
            )

        # Load frames into memory
        frames = []
        if HAS_CV2:
            for path in frame_paths:
                frame = cv2.imread(path)
                if frame is not None:
                    frames.append(frame)

        # Extract and analyze audio
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(video_path, audio_path, format="wav")
        audio_energy = analyze_audio_energy(audio_path, segment_length)

        # Analyze each segment
        segments = []
        frames_per_segment = int(segment_length * fps_sample)

        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, duration)

            # Get frames for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i + 1) * frames_per_segment, len(frames))
            segment_frames = frames[start_frame:end_frame] if frames else []

            # Score the segment
            scores = score_segment(segment_frames, audio_energy, i)

            # Generate reason based on top scores
            reasons = []
            if scores["motion"] > 0.6:
                reasons.append("high motion")
            if scores["faces"] > 0.3:
                reasons.append("facial expressions")
            if scores["scene_changes"] > 0.5:
                reasons.append("dynamic scene changes")
            if scores["audio"] > 0.7:
                reasons.append("audio peaks")
            if scores["color"] > 0.6:
                reasons.append("vibrant colors")

            reason = ", ".join(reasons) if reasons else "balanced content"

            segments.append(SegmentScore(
                start=start_time,
                end=end_time,
                score=scores["combined"],
                motion_score=scores["motion"],
                face_score=scores["faces"],
                scene_change_score=scores["scene_changes"],
                audio_score=scores["audio"],
                color_score=scores["color"],
                reason=reason,
            ))

    # Sort by score and get top N
    segments.sort(key=lambda x: x.score, reverse=True)
    top_segments = segments[:top_n]

    # Generate summary
    if top_segments:
        best = top_segments[0]
        summary = (
            f"Analyzed {num_segments} segments of {segment_length}s each. "
            f"Best segment: {best.start:.0f}s-{best.end:.0f}s "
            f"(score: {best.score:.2f}, reason: {best.reason})"
        )
    else:
        summary = "No segments analyzed"

    return ViralAnalysisResult(
        segments=top_segments,
        video_duration=duration,
        analysis_summary=summary,
    )


def get_best_segment(
    video_path: str,
    target_duration: int = 30,
) -> Optional[tuple[float, float]]:
    """
    Get the best segment of specified duration from a video.

    Args:
        video_path: Path to video file
        target_duration: Desired segment duration in seconds

    Returns:
        Tuple of (start_time, end_time) or None if analysis fails
    """
    result = analyze_video_segments(
        video_path,
        segment_length=target_duration,
        top_n=1,
    )

    if result.segments:
        best = result.segments[0]
        return (best.start, best.end)

    return None


def to_dict(result: ViralAnalysisResult) -> dict:
    """Convert ViralAnalysisResult to dictionary for JSON serialization."""
    return {
        "segments": [asdict(s) for s in result.segments],
        "video_duration": result.video_duration,
        "analysis_summary": result.analysis_summary,
    }
