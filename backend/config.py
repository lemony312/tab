"""
Configuration settings for the YouTube Tab Extractor.
All magic numbers and configurable values are centralized here.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
OUTPUT_DIR = DATA_DIR / "output"
COOKIES_FILE = BASE_DIR / "cookies.txt"

# Ensure directories exist
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Video download settings
# For tab extraction, we use the LOWEST quality possible since we only need
# to read guitar tab notation. This saves bandwidth and processing time.
VIDEO_QUALITY = "lowest"  # Options: "lowest", "360p", "480p", "720p"
VIDEO_FORMAT = "mp4"      # Output format

# YouTube-specific settings
# Browser to extract cookies from (None = use cookies.txt file instead)
# Options: "chrome", "firefox", "edge", "safari", "brave", "opera", None
COOKIES_FROM_BROWSER = None

# Rate limiting and abuse prevention
# YouTube rate limits: ~300 videos/hour for guests, ~2000/hour for accounts
DOWNLOAD_SLEEP_INTERVAL = 5  # Seconds to wait between downloads
MAX_DOWNLOADS_PER_HOUR = 50  # Safety limit (well under YouTube's limits)
REQUEST_TIMEOUT = 30  # Timeout for requests in seconds

# Video caching
# Cache downloaded videos to avoid re-downloading the same video
ENABLE_VIDEO_CACHE = True
VIDEO_CACHE_DIR = DATA_DIR / "cache"
VIDEO_CACHE_MAX_SIZE_GB = 5  # Maximum cache size in GB
VIDEO_CACHE_MAX_AGE_HOURS = 24  # Delete cached videos older than this

# Ensure cache directory exists
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Scene change detection settings
SCENE_CHANGE_THRESHOLD = 0.3  # Sensitivity (0-1, lower = more frames captured)
MIN_FRAME_INTERVAL = 2.0       # Minimum seconds between frame captures
FRAME_SAMPLE_RATE = 2          # Check every N frames for scene changes

# Tab-aware comparison (recommended for guitar tab videos)
# When True: Uses binarization and edge detection to compare ONLY the tab content,
#            ignoring background video and moving note highlights
# When False: Uses legacy comparison that triggers on any visual change
TAB_AWARE_COMPARISON = True

# Intro skip settings
# Many tab videos have intros (channel logo, title, guitarist talking) before
# showing actual tabs. This setting determines how long to wait before
# concluding that a video has no tabs.
INTRO_SKIP_SECONDS = 30  # Skip first N seconds when checking for tabs

# Tab detection settings
TAB_LINE_COUNT = 6             # Expected horizontal lines for guitar tablature
TAB_LINE_TOLERANCE = 2         # Allow +/- this many lines for detection
MIN_LINE_LENGTH_RATIO = 0.3    # Minimum line length as ratio of frame width
LINE_DETECTION_THRESHOLD = 50  # Hough line detection threshold

# Tab detection mode
# Options:
#   "auto"        - Auto-detect between full screen and embedded tabs
#   "full_screen" - Tabs occupy entire screen (no region detection needed)
#   "embedded"    - Tabs in portion of screen (detect region)
#   "numbers"     - Detect tabs by finding numbers/digits (fret numbers)
TAB_DETECTION_MODE = "auto"

# Full-screen tab settings
# When tabs occupy the entire screen, we capture the whole frame
# and detect changes more aggressively
FULL_SCREEN_TAB_CHANGE_INTERVAL = 5.0  # Expected seconds between tab changes
FULL_SCREEN_MIN_CHANGE_THRESHOLD = 0.05  # Lower threshold for detecting changes

# Multiple tab systems
# Some videos show multiple lines/systems of tabs at once
DETECT_MULTIPLE_TAB_SYSTEMS = True
MAX_TAB_SYSTEMS_PER_FRAME = 4  # Maximum tab systems to detect per frame

# PDF generation settings
OUTPUT_DPI = 150               # Image quality in PDF
PDF_PAGE_WIDTH = 210           # A4 width in mm
PDF_PAGE_HEIGHT = 297          # A4 height in mm
PDF_MARGIN = 10                # Margin in mm

# Processing settings
MAX_FRAMES_PER_VIDEO = 500     # Safety limit on extracted frames
CLEANUP_AFTER_PROCESSING = True  # Delete downloaded video after processing

# API settings
JOB_EXPIRY_HOURS = 24          # How long to keep completed jobs
