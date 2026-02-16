"""
Video processor for extracting frames with TAB-AWARE scene change detection.

Key challenge: Guitar tab videos often have:
1. Background video of someone playing guitar
2. Moving highlight/cursor showing the current note
3. The actual TAB CONTENT (lines + numbers) only changes every few seconds

This module uses techniques to detect when the TAB CONTENT changes,
ignoring background video and moving highlights:
- Binarization: Convert to pure black/white to remove color highlights
- Edge detection: Compare structural edges, not pixel colors
- Region focus: Compare only the tab region, not the whole frame
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import (
    SCENE_CHANGE_THRESHOLD,
    MIN_FRAME_INTERVAL,
    FRAME_SAMPLE_RATE,
    MAX_FRAMES_PER_VIDEO,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    """Represents an extracted video frame."""
    image: Image.Image
    timestamp: float  # Seconds from start
    frame_number: int


# ============================================================================
# TAB-AWARE Frame Comparison
# ============================================================================

def binarize_frame(frame: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Convert frame to binary (pure black/white) using adaptive thresholding.
    
    This removes color highlights and subtle shading while preserving
    high-contrast elements like tab lines and numbers.
    
    Args:
        frame: BGR or grayscale frame
        block_size: Size of pixel neighborhood for adaptive threshold
        c: Constant subtracted from mean
        
    Returns:
        Binary image (0 or 255 values only)
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold - works well for text/lines on varying backgrounds
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    return binary


def extract_edges(frame: np.ndarray) -> np.ndarray:
    """
    Extract edges from a frame using Canny edge detection.
    
    Edges represent structural content (lines, numbers) and are
    stable regardless of color changes or highlights.
    
    Args:
        frame: BGR or grayscale frame
        
    Returns:
        Edge image (binary)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges


def detect_tab_region(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Attempt to detect the tab region in a frame.
    
    Looks for area with multiple horizontal parallel lines.
    
    Args:
        frame: BGR frame
        
    Returns:
        Tuple of (top, bottom, left, right) or None if not detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect horizontal lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=frame.shape[1] * 0.3,
        maxLineGap=10
    )
    
    if lines is None or len(lines) < 4:
        return None
    
    # Filter for horizontal lines
    horizontal_y = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:
            horizontal_y.append((y1 + y2) // 2)
    
    if len(horizontal_y) < 4:
        return None
    
    # Find the region containing most horizontal lines
    horizontal_y.sort()
    top = max(0, min(horizontal_y) - 20)
    bottom = min(frame.shape[0], max(horizontal_y) + 20)
    
    return (top, bottom, 0, frame.shape[1])


def extract_comparison_region(frame: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Extract the region of interest for comparison.
    
    If a tab region is detected, use that. Otherwise, use the center
    portion of the frame (where tabs are usually displayed).
    
    Args:
        frame: BGR frame
        region: Optional (top, bottom, left, right) tuple
        
    Returns:
        Cropped frame region
    """
    if region:
        top, bottom, left, right = region
        return frame[top:bottom, left:right]
    
    # Default: use center 60% of frame height (tabs usually in center)
    h, w = frame.shape[:2]
    top = int(h * 0.2)
    bottom = int(h * 0.8)
    return frame[top:bottom, :]


def calculate_tab_content_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate difference between frames focusing on TAB CONTENT.
    
    This method is designed to ignore:
    - Background video/animation
    - Moving highlights showing current note
    - Color changes
    
    And detect when:
    - New tab bars appear
    - Numbers/notes change
    - The actual musical content updates
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
        
    Returns:
        Difference score between 0 and 1 (higher = more different)
    """
    # Try to detect tab region in both frames
    region1 = detect_tab_region(frame1)
    region2 = detect_tab_region(frame2)
    
    # Use detected region if available, otherwise use center
    roi1 = extract_comparison_region(frame1, region1)
    roi2 = extract_comparison_region(frame2, region2)
    
    # Resize to same dimensions for comparison
    target_size = (400, 200)
    roi1_resized = cv2.resize(roi1, target_size)
    roi2_resized = cv2.resize(roi2, target_size)
    
    # Method 1: Binarized comparison (removes color highlights)
    binary1 = binarize_frame(roi1_resized)
    binary2 = binarize_frame(roi2_resized)
    binary_diff = np.mean(cv2.absdiff(binary1, binary2)) / 255.0
    
    # Method 2: Edge comparison (structural content only)
    edges1 = extract_edges(roi1_resized)
    edges2 = extract_edges(roi2_resized)
    edge_diff = np.mean(cv2.absdiff(edges1, edges2)) / 255.0
    
    # Combine methods - binarized is more sensitive to content changes,
    # edges are more robust to highlight overlays
    combined_diff = 0.6 * binary_diff + 0.4 * edge_diff
    
    return combined_diff


# ============================================================================
# Legacy comparison methods (kept for compatibility)
# ============================================================================

def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate the difference between two frames using histogram comparison.
    
    NOTE: This method is sensitive to ALL visual changes including
    background video and highlights. Use calculate_tab_content_difference()
    for tab-specific comparison.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    difference = 1.0 - max(0.0, correlation)
    
    return difference


def calculate_structural_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate structural difference using absolute pixel difference.
    
    NOTE: This method is sensitive to ALL visual changes. 
    Use calculate_tab_content_difference() for tab-specific comparison.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff) / 255.0
    
    return mean_diff


def combined_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Combine histogram and structural difference.
    
    NOTE: This is the LEGACY method. For tab extraction, use
    calculate_tab_content_difference() which ignores highlights
    and background video.
    """
    hist_diff = calculate_frame_difference(frame1, frame2)
    struct_diff = calculate_structural_difference(frame1, frame2)
    return 0.4 * hist_diff + 0.6 * struct_diff


def frame_generator(video_path: Path, sample_rate: int = 1) -> Generator[tuple, None, None]:
    """
    Generator that yields frames from a video.
    
    Args:
        video_path: Path to video file
        sample_rate: Only yield every Nth frame
        
    Yields:
        Tuple of (frame, frame_number, timestamp)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_rate == 0:
                timestamp = frame_number / fps
                yield frame, frame_number, timestamp
            
            frame_number += 1
    finally:
        cap.release()


def extract_scene_changes(
    video_path: Path,
    threshold: Optional[float] = None,
    min_interval: Optional[float] = None,
    sample_rate: Optional[int] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    tab_aware: bool = True,
) -> List[ExtractedFrame]:
    """
    Extract frames from video where tab content changes.
    
    Args:
        video_path: Path to video file
        threshold: Scene change sensitivity (0-1, lower = more frames)
        min_interval: Minimum seconds between captured frames
        sample_rate: Check every Nth frame
        max_frames: Maximum number of frames to extract
        progress_callback: Optional callback(current, total) for progress updates
        tab_aware: If True, use tab-aware comparison that ignores background
                   video and moving highlights. If False, use legacy comparison
                   that triggers on any visual change.
        
    Returns:
        List of ExtractedFrame objects
    """
    threshold = threshold if threshold is not None else SCENE_CHANGE_THRESHOLD
    min_interval = min_interval if min_interval is not None else MIN_FRAME_INTERVAL
    sample_rate = sample_rate if sample_rate is not None else FRAME_SAMPLE_RATE
    max_frames = max_frames if max_frames is not None else MAX_FRAMES_PER_VIDEO
    
    # Select comparison method
    if tab_aware:
        compare_func = calculate_tab_content_difference
        logger.info("Using TAB-AWARE comparison (ignores highlights/background)")
    else:
        compare_func = combined_frame_difference
        logger.info("Using LEGACY comparison (sensitive to all changes)")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    logger.info(f"Processing video: {total_frames} frames, {duration:.1f}s duration")
    logger.info(f"Settings: threshold={threshold}, min_interval={min_interval}s")
    
    extracted_frames: List[ExtractedFrame] = []
    prev_frame = None
    last_capture_time = -min_interval  # Allow capture at t=0
    
    # Track detected tab region for consistency
    stable_tab_region = None
    
    for frame, frame_num, timestamp in frame_generator(video_path, sample_rate):
        # Progress callback
        if progress_callback and frame_num % 100 == 0:
            progress_callback(frame_num, total_frames)
        
        # Check max frames limit
        if len(extracted_frames) >= max_frames:
            logger.warning(f"Reached max frames limit ({max_frames})")
            break
        
        # First frame is always captured
        if prev_frame is None:
            # Try to detect tab region in first frame
            stable_tab_region = detect_tab_region(frame)
            if stable_tab_region:
                logger.info(f"Detected tab region: rows {stable_tab_region[0]}-{stable_tab_region[1]}")
            
            extracted_frames.append(ExtractedFrame(
                image=cv2_to_pil(frame),
                timestamp=timestamp,
                frame_number=frame_num,
            ))
            prev_frame = frame.copy()
            last_capture_time = timestamp
            logger.debug(f"Captured first frame at {timestamp:.2f}s")
            continue
        
        # Check minimum interval
        if timestamp - last_capture_time < min_interval:
            continue
        
        # Calculate difference using selected method
        diff = compare_func(prev_frame, frame)
        
        # Capture if difference exceeds threshold
        if diff > threshold:
            extracted_frames.append(ExtractedFrame(
                image=cv2_to_pil(frame),
                timestamp=timestamp,
                frame_number=frame_num,
            ))
            prev_frame = frame.copy()
            last_capture_time = timestamp
            logger.debug(f"Captured frame at {timestamp:.2f}s (diff={diff:.3f})")
    
    logger.info(f"Extracted {len(extracted_frames)} frames from video")
    return extracted_frames


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB Image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def get_video_info(video_path: Path) -> dict:
    """Get basic video information."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        }
        return info
    finally:
        cap.release()
