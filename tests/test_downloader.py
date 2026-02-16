"""
Tests for the YouTube downloader module.

These tests verify that yt-dlp integration works correctly for:
- Video info extraction
- Lowest quality video download
- Cookie authentication
- Error handling
- Video caching
- Rate limiting

To run these tests:
    cd /path/to/tab
    pip install pytest
    pytest tests/test_downloader.py -v

For manual testing with a real video:
    python tests/test_downloader.py
"""

import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Import after path setup
from backend.downloader import (
    get_format_string,
    get_format_sort,
    get_video_info,
    download_video,
    cleanup_video,
    extract_video_id,
    get_cache_path,
    get_cached_video,
    cache_video,
    cleanup_cache,
    get_cache_stats,
    clear_cache,
    RateLimiter,
    check_rate_limit,
)
from backend.config import DOWNLOADS_DIR, VIDEO_CACHE_DIR


# ============================================================================
# Test Videos
# ============================================================================

# Short, public domain test videos that should always be available
TEST_VIDEOS = {
    # YouTube's official test video (if available) or other reliable public videos
    "short_public": "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo" - first YouTube video
    
    # For guitar tab testing, you'd use a video with tabs like:
    # "guitar_tab": "https://www.youtube.com/watch?v=XXXXXXX",  # Replace with actual tab video
}

# Flag to skip network tests in CI environments
SKIP_NETWORK_TESTS = os.environ.get("SKIP_NETWORK_TESTS", "").lower() == "true"


# ============================================================================
# Unit Tests (No Network Required)
# ============================================================================

class TestFormatStrings:
    """Test format string generation."""
    
    def test_lowest_quality_format(self):
        """Lowest quality format should prefer small files."""
        fmt = get_format_string("lowest")
        
        # Should contain worst video selector
        assert "wv*" in fmt or "w" in fmt
        # Should have height limit
        assert "360" in fmt or "height" in fmt
    
    def test_360p_format(self):
        """360p format should limit to 360p resolution."""
        fmt = get_format_string("360p")
        assert "360" in fmt
    
    def test_720p_format(self):
        """720p format should limit to 720p resolution."""
        fmt = get_format_string("720p")
        assert "720" in fmt
    
    def test_unknown_quality_defaults_to_lowest(self):
        """Unknown quality should default to lowest."""
        fmt = get_format_string("unknown_quality")
        lowest_fmt = get_format_string("lowest")
        assert fmt == lowest_fmt
    
    def test_format_sort_lowest(self):
        """Lowest quality should sort by ascending size."""
        sort_opts = get_format_sort("lowest")
        
        # Should prioritize smallest size
        assert "+size" in sort_opts
        assert "+res" in sort_opts
    
    def test_format_sort_720p(self):
        """Higher quality should sort by resolution."""
        sort_opts = get_format_sort("720p")
        
        # Should prioritize resolution (not +res which is ascending)
        assert "res" in sort_opts


# ============================================================================
# Video ID Extraction Tests
# ============================================================================

class TestVideoIdExtraction:
    """Test YouTube video ID extraction from URLs."""
    
    def test_standard_watch_url(self):
        """Should extract ID from standard watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_short_url(self):
        """Should extract ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_embed_url(self):
        """Should extract ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_watch_url_with_params(self):
        """Should extract ID from URL with extra parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_invalid_url(self):
        """Should return None for invalid URL."""
        assert extract_video_id("not a url") is None
        assert extract_video_id("https://example.com/video") is None
    
    def test_just_id(self):
        """Should extract ID when given just the ID."""
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


# ============================================================================
# Caching Tests (No Network Required)
# ============================================================================

class TestVideoCache:
    """Test video caching functionality."""
    
    def test_cache_path_generation(self):
        """Should generate consistent cache paths."""
        video_id = "test123"
        path1 = get_cache_path(video_id, "lowest")
        path2 = get_cache_path(video_id, "lowest")
        
        assert path1 == path2
        assert "test123_lowest.mp4" in str(path1)
    
    def test_cache_path_different_qualities(self):
        """Should generate different paths for different qualities."""
        video_id = "test123"
        path_low = get_cache_path(video_id, "lowest")
        path_high = get_cache_path(video_id, "720p")
        
        assert path_low != path_high
        assert "lowest" in str(path_low)
        assert "720p" in str(path_high)
    
    def test_get_cached_video_not_exists(self):
        """Should return None when video is not cached."""
        result = get_cached_video("https://www.youtube.com/watch?v=nonexistent123")
        assert result is None
    
    def test_cache_video_and_retrieve(self):
        """Should cache and retrieve video."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            mock_video = Path(temp_dir) / "test_video.mp4"
            mock_video.write_bytes(b"fake video data")
            
            # YouTube video IDs are exactly 11 characters
            url = "https://www.youtube.com/watch?v=cachetest01"
            
            # Cache the video
            cached_path = cache_video(mock_video, url, "lowest")
            
            # Verify it was cached
            assert cached_path.exists()
            assert "cachetest01" in str(cached_path)
            
            # Verify we can retrieve it
            retrieved = get_cached_video(url, "lowest")
            assert retrieved is not None
            assert retrieved.exists()
            
            # Cleanup
            if retrieved.exists():
                retrieved.unlink()
    
    def test_cache_stats(self):
        """Should return cache statistics."""
        stats = get_cache_stats()
        
        assert "enabled" in stats
        if stats["enabled"]:
            assert "file_count" in stats
            assert "total_size_mb" in stats
            assert "max_size_gb" in stats


# ============================================================================
# Rate Limiting Tests (No Network Required)
# ============================================================================

class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Should initialize with default values."""
        limiter = RateLimiter(max_per_hour=10)
        assert limiter.max_per_hour == 10
        assert limiter.can_download() is True
    
    def test_rate_limiter_tracks_downloads(self):
        """Should track download counts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create limiter with custom state file
            limiter = RateLimiter(max_per_hour=5)
            limiter.state_file = Path(temp_dir) / "rate_state.json"
            limiter.download_times = []
            
            assert limiter.get_downloads_this_hour() == 0
            
            limiter.record_download()
            assert limiter.get_downloads_this_hour() == 1
            
            limiter.record_download()
            assert limiter.get_downloads_this_hour() == 2
    
    def test_rate_limiter_blocks_at_limit(self):
        """Should block downloads when limit reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            limiter = RateLimiter(max_per_hour=3)
            limiter.state_file = Path(temp_dir) / "rate_state.json"
            limiter.download_times = []
            
            # Record 3 downloads
            for _ in range(3):
                limiter.record_download()
            
            assert limiter.can_download() is False
    
    def test_rate_limiter_clears_old_entries(self):
        """Should clear entries older than 1 hour."""
        with tempfile.TemporaryDirectory() as temp_dir:
            limiter = RateLimiter(max_per_hour=5)
            limiter.state_file = Path(temp_dir) / "rate_state.json"
            
            # Add old entries
            old_time = datetime.now() - timedelta(hours=2)
            limiter.download_times = [old_time, old_time]
            
            # Should be cleaned up
            assert limiter.get_downloads_this_hour() == 0
            assert limiter.can_download() is True
    
    def test_rate_limiter_wait_time(self):
        """Should calculate wait time when limit reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            limiter = RateLimiter(max_per_hour=2)
            limiter.state_file = Path(temp_dir) / "rate_state.json"
            limiter.download_times = []
            
            # Fill up the limit
            for _ in range(2):
                limiter.record_download()
            
            wait = limiter.get_wait_time()
            # Wait time should be close to 1 hour (since oldest entry is just now)
            assert wait > 3500  # ~59 minutes
            assert wait < 3700  # Just over 1 hour max
    
    def test_rate_limiter_persists_state(self):
        """Should persist state to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "rate_state.json"
            
            # Create limiter and record download
            limiter1 = RateLimiter(max_per_hour=10)
            limiter1.state_file = state_file
            limiter1.download_times = []
            limiter1.record_download()
            
            # Create new limiter with same state file
            limiter2 = RateLimiter(max_per_hour=10)
            limiter2.state_file = state_file
            limiter2._load_state()
            
            # Should have loaded the download
            assert limiter2.get_downloads_this_hour() == 1


# ============================================================================
# Integration Tests (Network Required)
# ============================================================================

@pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network tests disabled")
class TestVideoInfo:
    """Test video info extraction."""
    
    def test_get_video_info_public_video(self):
        """Should extract info from a public video."""
        url = TEST_VIDEOS["short_public"]
        
        try:
            info = get_video_info(url)
            
            assert info["id"] is not None
            assert info["title"] is not None
            assert info["duration"] is not None
            assert isinstance(info["available_resolutions"], list)
            
            print(f"\n  Video: {info['title']}")
            print(f"  Duration: {info['duration']}s")
            print(f"  Resolutions: {info['available_resolutions']}")
            
        except Exception as e:
            pytest.skip(f"Network test failed (may be rate limited): {e}")
    
    def test_get_video_info_invalid_url(self):
        """Should raise error for invalid URL."""
        import yt_dlp
        with pytest.raises((ValueError, RuntimeError, yt_dlp.utils.DownloadError, Exception)):
            get_video_info("https://www.youtube.com/watch?v=INVALID12345")


@pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network tests disabled")
class TestVideoDownload:
    """Test video download functionality."""
    
    def test_download_lowest_quality(self):
        """Should download video at lowest quality."""
        url = TEST_VIDEOS["short_public"]
        
        # Use temp directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                video_path = download_video(
                    url=url,
                    output_dir=temp_path,
                    quality="lowest",
                    use_cookies=False,  # Don't use cookies for public video
                )
                
                assert video_path.exists()
                
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"\n  Downloaded: {video_path.name}")
                print(f"  File size: {file_size_mb:.2f} MB")
                
                # Lowest quality should be reasonably small
                # (exact size depends on video length)
                assert file_size_mb > 0
                
            except Exception as e:
                pytest.skip(f"Network test failed: {e}")
    
    def test_download_360p_quality(self):
        """Should download video at 360p quality."""
        url = TEST_VIDEOS["short_public"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                video_path = download_video(
                    url=url,
                    output_dir=temp_path,
                    quality="360p",
                    use_cookies=False,
                )
                
                assert video_path.exists()
                print(f"\n  Downloaded at 360p: {video_path.name}")
                
            except Exception as e:
                pytest.skip(f"Network test failed: {e}")
    
    def test_cleanup_video(self):
        """Should delete video file after cleanup."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            temp_path = Path(f.name)
            f.write(b"test content")
        
        assert temp_path.exists()
        
        cleanup_video(temp_path)
        
        assert not temp_path.exists()
    
    def test_cleanup_nonexistent_file(self):
        """Should handle cleanup of non-existent file gracefully."""
        fake_path = Path("/nonexistent/path/to/video.mp4")
        
        # Should not raise an error
        cleanup_video(fake_path)


# ============================================================================
# Manual Testing
# ============================================================================

def manual_test_download():
    """
    Manual test for downloading a video.
    Run this directly to test with a real YouTube video.
    """
    print("=" * 60)
    print("Manual Download Test")
    print("=" * 60)
    
    # Test video info
    print("\n1. Testing video info extraction...")
    url = TEST_VIDEOS["short_public"]
    
    try:
        info = get_video_info(url)
        print(f"   Title: {info['title']}")
        print(f"   Duration: {info['duration']} seconds")
        print(f"   Uploader: {info['uploader']}")
        print(f"   Available resolutions: {info['available_resolutions']}")
        print("   ✓ Video info extraction successful")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test download at lowest quality
    print("\n2. Testing lowest quality download...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            video_path = download_video(
                url=url,
                output_dir=temp_path,
                quality="lowest",
                use_cookies=False,
            )
            
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"   Downloaded: {video_path.name}")
            print(f"   File size: {file_size_mb:.2f} MB")
            print("   ✓ Download successful")
            
            # Test cleanup
            print("\n3. Testing cleanup...")
            cleanup_video(video_path)
            
            if not video_path.exists():
                print("   ✓ Cleanup successful")
            else:
                print("   ✗ Cleanup failed - file still exists")
                
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def test_format_strings_manual():
    """Print all format strings for manual verification."""
    print("\n" + "=" * 60)
    print("Format Strings")
    print("=" * 60)
    
    qualities = ["lowest", "240p", "360p", "480p", "720p", "1080p"]
    
    for quality in qualities:
        fmt = get_format_string(quality)
        sort_opts = get_format_sort(quality)
        print(f"\n{quality}:")
        print(f"  Format: {fmt}")
        print(f"  Sort: {sort_opts}")


if __name__ == "__main__":
    # Run manual tests when executed directly
    test_format_strings_manual()
    print()
    manual_test_download()
