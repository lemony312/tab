#!/usr/bin/env python3
"""
Quick script to extract tabs from a YouTube video.

Usage:
    python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode full_screen
    python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode numbers
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.downloader import download_video, get_video_info, cleanup_video
from backend.processor import extract_scene_changes
from backend.tab_detector import filter_tab_frames, remove_duplicate_tabs
from backend.pdf_builder import build_pdf
from backend.config import CLEANUP_AFTER_PROCESSING, OUTPUT_DIR


def extract_tabs_from_video(
    url: str,
    output_name: str = None,
    detection_mode: str = "auto",
    min_interval: float = 2.0,
) -> Path:
    """
    Extract guitar tabs from a YouTube video and generate a PDF.
    
    Args:
        url: YouTube video URL
        output_name: Optional name for the output PDF
        detection_mode: Detection mode (auto, full_screen, embedded, numbers)
        min_interval: Minimum interval between frame captures
        
    Returns:
        Path to the generated PDF
    """
    video_path = None
    
    try:
        # Step 1: Get video info
        logger.info("Fetching video information...")
        try:
            info = get_video_info(url)
            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            logger.info(f"Video: {title} ({duration}s)")
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            title = "Unknown"
        
        # Step 2: Download video
        logger.info("Downloading video (lowest quality for tab extraction)...")
        video_path = download_video(url)
        logger.info(f"Downloaded: {video_path}")
        
        # Adjust settings based on detection mode
        threshold = 0.3
        if detection_mode == "full_screen":
            # Full screen tabs change abruptly, use lower threshold
            threshold = 0.05
            logger.info(f"Using full_screen mode with lower threshold ({threshold})")
        
        # Step 3: Extract frames with tab-aware comparison
        logger.info(f"Extracting frames (mode: {detection_mode}, interval: {min_interval}s)...")
        frames = extract_scene_changes(
            video_path,
            threshold=threshold,
            min_interval=min_interval,
            tab_aware=True,
        )
        logger.info(f"Extracted {len(frames)} scene change frames")
        
        if not frames:
            raise RuntimeError("No frames extracted from video")
        
        # Step 4: Filter for tab frames
        logger.info(f"Detecting guitar tabs in frames (mode: {detection_mode})...")
        tab_frames = filter_tab_frames(frames, min_confidence=0.3, mode=detection_mode)
        logger.info(f"Found {len(tab_frames)} frames with tabs")
        
        # Step 5: Remove duplicates
        # - Compare only top tab system (avoids duplicates from scrolling)
        # - Use binarized comparison (ignores colors/highlights)
        # - Detect scroll patterns (content that moved up/down)
        if detection_mode == "full_screen":
            # Only remove frames that are 98%+ identical
            dup_threshold = 0.98
        else:
            dup_threshold = 0.90
        logger.info(f"Deduplication: threshold={dup_threshold}, top_system_only=True, detect_scroll=True")
        tab_frames = remove_duplicate_tabs(
            tab_frames,
            similarity_threshold=dup_threshold,
            use_top_system_only=True,   # Only compare top tab system
            detect_scrolling=True,       # Detect scrolled duplicates
        )
        logger.info(f"After deduplication: {len(tab_frames)} unique tab frames")
        
        if not tab_frames:
            raise RuntimeError(
                "No guitar tabs detected in this video. "
                f"Tried mode: {detection_mode}. "
                "Try a different detection mode (--mode full_screen or --mode numbers)."
            )
        
        # Step 6: Generate PDF
        logger.info("Generating PDF...")
        pdf_name = output_name or title
        pdf_path = build_pdf(tab_frames, title=pdf_name)
        logger.info(f"PDF saved to: {pdf_path}")
        
        return pdf_path
        
    finally:
        # Cleanup
        if video_path and CLEANUP_AFTER_PROCESSING:
            cleanup_video(video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract guitar tabs from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect tab layout
  python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Full-screen tabs (tabs occupy entire screen, change every few seconds)
  python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode full_screen
  
  # Detect tabs by finding fret numbers
  python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode numbers
  
  # Full-screen tabs with 5 second interval (for tabs that change every ~5 seconds)
  python extract_tabs.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode full_screen --interval 4
        """
    )
    
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", help="Output PDF name")
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "full_screen", "embedded", "numbers"],
        default="auto",
        help="Detection mode: auto (default), full_screen (tabs fill screen), embedded (tabs in portion), numbers (detect fret numbers)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Minimum interval between frame captures in seconds (default: 2.0, use 4-5 for full_screen tabs)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YouTube Tab Extractor")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Mode: {args.mode}")
    print(f"Interval: {args.interval}s")
    print("")
    
    try:
        pdf_path = extract_tabs_from_video(
            args.url,
            output_name=args.output,
            detection_mode=args.mode,
            min_interval=args.interval,
        )
        print("")
        print("=" * 60)
        print("SUCCESS!")
        print(f"PDF saved to: {pdf_path}")
        print("=" * 60)
    except Exception as e:
        print("")
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
