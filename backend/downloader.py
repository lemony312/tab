"""
YouTube video downloader using yt-dlp.
Handles video download with optional cookie-based authentication for premium content.

Key considerations for YouTube downloads (as of 2025):
- YouTube is enforcing PO Tokens for some content; we use clients that don't require them
- Cookies should be exported from a private/incognito window to avoid rotation issues
- Rate limiting: ~300 videos/hour for guests, ~2000/hour for accounts
- For tab extraction, we use the LOWEST quality video to save bandwidth

Features:
- Video caching: Re-use previously downloaded videos
- Rate limiting: Automatic sleep between downloads
- Abuse prevention: Track download counts and enforce limits
"""

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import yt_dlp

from .config import (
    DOWNLOADS_DIR,
    COOKIES_FILE,
    VIDEO_QUALITY,
    VIDEO_FORMAT,
    COOKIES_FROM_BROWSER,
    ENABLE_VIDEO_CACHE,
    VIDEO_CACHE_DIR,
    VIDEO_CACHE_MAX_SIZE_GB,
    VIDEO_CACHE_MAX_AGE_HOURS,
    DOWNLOAD_SLEEP_INTERVAL,
    MAX_DOWNLOADS_PER_HOUR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Track downloads and enforce rate limits to prevent YouTube abuse.
    """
    
    def __init__(self, max_per_hour: int = MAX_DOWNLOADS_PER_HOUR):
        self.max_per_hour = max_per_hour
        self.download_times: List[datetime] = []
        self.state_file = VIDEO_CACHE_DIR / "rate_limit_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load rate limit state from disk."""
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text())
                self.download_times = [
                    datetime.fromisoformat(ts) for ts in data.get("download_times", [])
                ]
                # Clean old entries
                self._cleanup_old_entries()
        except Exception as e:
            logger.warning(f"Could not load rate limit state: {e}")
            self.download_times = []
    
    def _save_state(self):
        """Save rate limit state to disk."""
        try:
            self._cleanup_old_entries()
            data = {
                "download_times": [ts.isoformat() for ts in self.download_times]
            }
            self.state_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Could not save rate limit state: {e}")
    
    def _cleanup_old_entries(self):
        """Remove entries older than 1 hour."""
        cutoff = datetime.now() - timedelta(hours=1)
        self.download_times = [ts for ts in self.download_times if ts > cutoff]
    
    def can_download(self) -> bool:
        """Check if we can download without exceeding rate limit."""
        self._cleanup_old_entries()
        return len(self.download_times) < self.max_per_hour
    
    def get_wait_time(self) -> float:
        """Get seconds to wait before next download is allowed."""
        if self.can_download():
            return 0
        
        # Find the oldest entry and calculate when it expires
        oldest = min(self.download_times)
        expires = oldest + timedelta(hours=1)
        wait = (expires - datetime.now()).total_seconds()
        return max(0, wait)
    
    def record_download(self):
        """Record a download."""
        self.download_times.append(datetime.now())
        self._save_state()
    
    def get_downloads_this_hour(self) -> int:
        """Get number of downloads in the last hour."""
        self._cleanup_old_entries()
        return len(self.download_times)


# Global rate limiter
_rate_limiter = RateLimiter()


def check_rate_limit() -> None:
    """
    Check rate limit and wait if necessary.
    
    Raises:
        RuntimeError: If rate limit is exceeded and wait time is too long
    """
    if not _rate_limiter.can_download():
        wait_time = _rate_limiter.get_wait_time()
        if wait_time > 300:  # More than 5 minutes
            raise RuntimeError(
                f"Rate limit exceeded. {_rate_limiter.get_downloads_this_hour()} downloads "
                f"in the last hour. Please wait {int(wait_time)}s or try again later."
            )
        logger.warning(f"Rate limit approaching, waiting {wait_time:.0f}s...")
        time.sleep(wait_time)


# ============================================================================
# Video Caching
# ============================================================================

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL.
    
    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r'(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # Just the ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def get_cache_path(video_id: str, quality: str = "lowest") -> Path:
    """Get the cache file path for a video."""
    # Include quality in filename to cache different qualities separately
    cache_name = f"{video_id}_{quality}.mp4"
    return VIDEO_CACHE_DIR / cache_name


def get_cached_video(url: str, quality: str = "lowest") -> Optional[Path]:
    """
    Check if video is already cached.
    
    Args:
        url: YouTube video URL
        quality: Video quality setting
        
    Returns:
        Path to cached video if exists and valid, None otherwise
    """
    if not ENABLE_VIDEO_CACHE:
        return None
    
    video_id = extract_video_id(url)
    if not video_id:
        return None
    
    cache_path = get_cache_path(video_id, quality)
    
    if cache_path.exists():
        # Check if cache is still valid (not too old)
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < VIDEO_CACHE_MAX_AGE_HOURS:
            logger.info(f"Using cached video: {cache_path} (age: {age_hours:.1f}h)")
            return cache_path
        else:
            logger.info(f"Cache expired for {video_id}, will re-download")
            cache_path.unlink()
    
    return None


def cache_video(source_path: Path, url: str, quality: str = "lowest") -> Path:
    """
    Cache a downloaded video.
    
    Args:
        source_path: Path to the downloaded video
        url: YouTube video URL
        quality: Video quality setting
        
    Returns:
        Path to the cached video
    """
    if not ENABLE_VIDEO_CACHE:
        return source_path
    
    video_id = extract_video_id(url)
    if not video_id:
        return source_path
    
    cache_path = get_cache_path(video_id, quality)
    
    # Move file to cache
    try:
        import shutil
        shutil.copy2(source_path, cache_path)
        source_path.unlink()  # Remove original
        logger.info(f"Cached video: {cache_path}")
        
        # Cleanup old cache files if needed
        cleanup_cache()
        
        return cache_path
    except Exception as e:
        logger.warning(f"Could not cache video: {e}")
        return source_path


def cleanup_cache() -> None:
    """
    Clean up old cache files to stay under size limit.
    """
    if not ENABLE_VIDEO_CACHE or not VIDEO_CACHE_DIR.exists():
        return
    
    try:
        # Get all cached videos
        cache_files = list(VIDEO_CACHE_DIR.glob("*.mp4"))
        
        # Calculate total size
        total_size_gb = sum(f.stat().st_size for f in cache_files) / (1024**3)
        
        if total_size_gb <= VIDEO_CACHE_MAX_SIZE_GB:
            return
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        # Delete oldest files until under limit
        for cache_file in cache_files:
            if total_size_gb <= VIDEO_CACHE_MAX_SIZE_GB * 0.8:  # Target 80% of limit
                break
            
            file_size_gb = cache_file.stat().st_size / (1024**3)
            cache_file.unlink()
            total_size_gb -= file_size_gb
            logger.info(f"Removed old cache file: {cache_file}")
            
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")


def get_format_string(quality: str) -> str:
    """
    Convert quality string to yt-dlp format selector.
    
    For tab extraction, we want the LOWEST quality that's still readable.
    Guitar tabs are text/lines, so even 240p-360p is usually sufficient.
    """
    quality_map = {
        # Lowest quality - prioritize smallest file size
        # wv* = worst video, wa = worst audio, w = worst combined
        "lowest": "wv*[height>=144][height<=360]+wa/w[height<=360]/wv*+wa/w",
        
        # Specific resolutions (still prefer smaller within range)
        "240p": "wv*[height<=240]+wa/w[height<=240]/wv*[height<=360]+wa/w",
        "360p": "wv*[height<=360]+wa/w[height<=360]/wv*[height<=480]+wa/w",
        "480p": "wv*[height<=480]+wa/w[height<=480]",
        "720p": "bv*[height<=720]+ba/b[height<=720]",
        "1080p": "bv*[height<=1080]+ba/b[height<=1080]",
    }
    return quality_map.get(quality, quality_map["lowest"])


def get_format_sort(quality: str) -> List[str]:
    """
    Get format sorting options for yt-dlp.
    
    This is used with the -S flag to prioritize smaller files.
    For tab extraction, we want: smallest size, lowest resolution, lowest bitrate.
    """
    if quality in ["lowest", "240p", "360p"]:
        # Prioritize smallest file: +size (ascending), +res (ascending), +br (ascending)
        return ["+size", "+res", "+br", "ext:mp4:m4a"]
    else:
        # Standard quality: best quality within resolution limit
        return ["res", "ext:mp4:m4a"]


def download_video(
    url: str,
    output_dir: Optional[Path] = None,
    quality: Optional[str] = None,
    use_cookies: bool = True,
    cookies_from_browser: Optional[str] = None,
    use_cache: bool = True,
) -> Path:
    """
    Download a YouTube video with caching and rate limiting.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video (defaults to DOWNLOADS_DIR)
        quality: Video quality ("lowest", "360p", "480p", "720p", "1080p")
        use_cookies: Whether to use cookies for authentication
        cookies_from_browser: Browser to extract cookies from (e.g., "chrome", "firefox")
                             If None, uses cookies.txt file if available
        use_cache: Whether to use/update video cache
        
    Returns:
        Path to the downloaded video file
        
    Raises:
        ValueError: If the URL is invalid
        RuntimeError: If download fails or rate limit exceeded
    """
    output_dir = output_dir or DOWNLOADS_DIR
    quality = quality or VIDEO_QUALITY
    cookies_from_browser = cookies_from_browser or COOKIES_FROM_BROWSER
    
    # Check cache first
    if use_cache:
        cached_path = get_cached_video(url, quality)
        if cached_path:
            return cached_path
    
    # Check rate limit before downloading
    check_rate_limit()
    
    # Log rate limit status
    downloads_this_hour = _rate_limiter.get_downloads_this_hour()
    logger.info(f"Downloads this hour: {downloads_this_hour}/{MAX_DOWNLOADS_PER_HOUR}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    output_template = str(output_dir / f"{file_id}.%(ext)s")
    
    ydl_opts = {
        "format": get_format_string(quality),
        "format_sort": get_format_sort(quality),
        "outtmpl": output_template,
        "merge_output_format": VIDEO_FORMAT,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        # Use clients that don't require PO Token
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],  # Avoid clients requiring PO tokens
            }
        },
        # Rate limiting: sleep between requests
        "sleep_interval": 1,  # Sleep 1 second between requests
        "max_sleep_interval": 5,
    }
    
    # Cookie authentication options
    if use_cookies:
        if cookies_from_browser:
            # Extract cookies directly from browser
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
            logger.info(f"Using cookies from browser: {cookies_from_browser}")
        elif COOKIES_FILE.exists():
            # Use cookies.txt file
            ydl_opts["cookiefile"] = str(COOKIES_FILE)
            logger.info("Using cookies from cookies.txt")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to validate URL and log details
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise ValueError(f"Could not extract info from URL: {url}")
            
            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            
            # Log available formats for debugging
            formats = info.get("formats", [])
            if formats:
                heights = sorted(set(f.get("height") for f in formats if f.get("height")))
                logger.info(f"Available resolutions: {heights}")
            
            logger.info(f"Downloading: {title} ({duration}s) at {quality} quality")
            
            # Download the video
            ydl.download([url])
            
            # Record the download for rate limiting
            _rate_limiter.record_download()
            
            # Find the downloaded file
            video_path = output_dir / f"{file_id}.{VIDEO_FORMAT}"
            if not video_path.exists():
                # Try to find file with different extension
                for ext in ["mp4", "mkv", "webm", "m4a"]:
                    alt_path = output_dir / f"{file_id}.{ext}"
                    if alt_path.exists():
                        video_path = alt_path
                        break
            
            if not video_path.exists():
                # List what files were created
                created_files = list(output_dir.glob(f"{file_id}.*"))
                if created_files:
                    video_path = created_files[0]
                    logger.info(f"Found downloaded file: {video_path}")
                else:
                    raise RuntimeError(f"Downloaded file not found at {video_path}")
            
            # Log file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded to: {video_path} ({file_size_mb:.1f} MB)")
            
            # Cache the video for future use
            if use_cache:
                video_path = cache_video(video_path, url, quality)
            
            # Sleep after download to be nice to YouTube
            if DOWNLOAD_SLEEP_INTERVAL > 0:
                logger.debug(f"Sleeping {DOWNLOAD_SLEEP_INTERVAL}s after download...")
                time.sleep(DOWNLOAD_SLEEP_INTERVAL)
            
            return video_path
            
    except yt_dlp.DownloadError as e:
        error_msg = str(e)
        
        # Provide helpful error messages for common issues
        if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
            raise RuntimeError(
                "YouTube requires authentication. Please provide cookies.txt or "
                "set COOKIES_FROM_BROWSER in config.py. "
                "See README.md for instructions."
            )
        elif "Private video" in error_msg:
            raise RuntimeError("This video is private. Ensure your cookies have access.")
        elif "Video unavailable" in error_msg:
            raise RuntimeError("Video is unavailable (may be deleted or geo-restricted).")
        elif "HTTP Error 429" in error_msg or "Too Many Requests" in error_msg:
            raise RuntimeError(
                "YouTube rate limit exceeded (HTTP 429). Please wait 10-30 minutes "
                "before trying again, or use authenticated cookies."
            )
        else:
            raise RuntimeError(f"Download failed: {e}")


def get_video_info(url: str, use_cookies: bool = True) -> dict:
    """
    Get video metadata without downloading.
    
    Args:
        url: YouTube video URL
        use_cookies: Whether to use cookies for authentication
        
    Returns:
        Dictionary with video metadata (title, duration, formats, etc.)
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }
    
    # Cookie authentication
    if use_cookies:
        if COOKIES_FROM_BROWSER:
            ydl_opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
        elif COOKIES_FILE.exists():
            ydl_opts["cookiefile"] = str(COOKIES_FILE)
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            raise ValueError(f"Could not extract info from URL: {url}")
        
        # Get available resolutions
        formats = info.get("formats", [])
        resolutions = sorted(set(
            f.get("height") for f in formats 
            if f.get("height") and f.get("vcodec") != "none"
        ))
        
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "uploader": info.get("uploader"),
            "available_resolutions": resolutions,
            "is_live": info.get("is_live", False),
        }


def cleanup_video(video_path: Path, force: bool = False) -> None:
    """
    Delete a downloaded video file.
    
    Args:
        video_path: Path to the video file
        force: If True, delete even if in cache directory
    """
    try:
        if not video_path.exists():
            return
        
        # Don't delete cached videos unless forced
        if ENABLE_VIDEO_CACHE and VIDEO_CACHE_DIR in video_path.parents:
            if not force:
                logger.debug(f"Keeping cached video: {video_path}")
                return
        
        video_path.unlink()
        logger.info(f"Cleaned up: {video_path}")
    except OSError as e:
        logger.warning(f"Failed to cleanup {video_path}: {e}")


def get_cache_stats() -> Dict:
    """
    Get statistics about the video cache.
    
    Returns:
        Dictionary with cache statistics
    """
    if not ENABLE_VIDEO_CACHE or not VIDEO_CACHE_DIR.exists():
        return {"enabled": False}
    
    cache_files = list(VIDEO_CACHE_DIR.glob("*.mp4"))
    total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
    
    return {
        "enabled": True,
        "file_count": len(cache_files),
        "total_size_mb": round(total_size_mb, 2),
        "max_size_gb": VIDEO_CACHE_MAX_SIZE_GB,
        "max_age_hours": VIDEO_CACHE_MAX_AGE_HOURS,
    }


def clear_cache() -> int:
    """
    Clear all cached videos.
    
    Returns:
        Number of files deleted
    """
    if not VIDEO_CACHE_DIR.exists():
        return 0
    
    cache_files = list(VIDEO_CACHE_DIR.glob("*.mp4"))
    for f in cache_files:
        try:
            f.unlink()
        except OSError:
            pass
    
    # Also clear rate limit state
    state_file = VIDEO_CACHE_DIR / "rate_limit_state.json"
    if state_file.exists():
        state_file.unlink()
    
    logger.info(f"Cleared {len(cache_files)} cached videos")
    return len(cache_files)
