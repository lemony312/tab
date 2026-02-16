# YouTube Tab Extractor

Extract guitar tablature from YouTube videos and compile them into a single PDF document.

## Features

- **YouTube Video Download**: Download videos with optional premium authentication
- **Scene Change Detection**: Automatically detect when tab content changes
- **Tab Recognition**: Identify frames containing guitar tablature using edge detection
- **PDF Generation**: Compile all tab sections into a single, printable PDF
- **Simple Web UI**: Easy-to-use browser interface

## Requirements

- Python 3.10+
- pip (Python package manager)

## Installation

1. Clone or download this project:

```bash
cd /path/to/tab
```

2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

## Running the Application

Start the server:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Then open your browser to: **http://localhost:8000**

## YouTube Authentication (Optional)

For premium content, private videos, or to avoid rate limiting, you can provide authentication.

### Option 1: Cookies from Browser (Recommended)

Edit `backend/config.py` and set:

```python
COOKIES_FROM_BROWSER = "chrome"  # or "firefox", "edge", "safari", "brave"
```

The application will automatically extract cookies from your browser.

### Option 2: Cookies File

For more control, export cookies manually:

1. Open a **private/incognito** browser window
2. Log into YouTube
3. Navigate to `https://www.youtube.com/robots.txt` (keep this as the only tab)
4. Install extension "Get cookies.txt LOCALLY" and export cookies
5. **Close the private window immediately** (prevents cookie rotation)
6. Save as `cookies.txt` in the project root

### Important Notes

- YouTube rotates cookies frequently on open tabs - always export from a private window
- Using your account with automated tools risks temporary or permanent bans
- Consider using a secondary/throwaway account
- Rate limits: ~300 videos/hour (guests), ~2000/hour (authenticated)

## Usage

1. Open http://localhost:8000 in your browser
2. Paste a YouTube video URL containing guitar tabs
3. (Optional) Adjust advanced options:
   - **Scene Change Sensitivity**: Lower values capture more frames
   - **Minimum Interval**: Minimum seconds between frame captures
4. Click "Extract Tabs"
5. Wait for processing to complete
6. Download your PDF

## Project Structure

```
tab/
├── backend/
│   ├── main.py              # FastAPI app and routes
│   ├── downloader.py        # YouTube video download
│   ├── processor.py         # Video frame extraction
│   ├── tab_detector.py      # Tab detection logic
│   ├── pdf_builder.py       # PDF generation
│   ├── config.py            # Configuration settings
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Web interface
│   ├── style.css            # Styles
│   └── app.js               # Frontend logic
├── data/
│   ├── downloads/           # Temporary video storage
│   └── output/              # Generated PDFs
├── cookies.txt              # YouTube auth (you create this)
└── README.md
```

## Configuration

Edit `backend/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `VIDEO_QUALITY` | lowest | Download quality (lowest, 360p, 480p, 720p) |
| `COOKIES_FROM_BROWSER` | None | Browser to extract cookies from |
| `TAB_AWARE_COMPARISON` | True | Ignore background video and highlights |
| `INTRO_SKIP_SECONDS` | 30 | Wait before concluding "no tabs" (for intros) |
| `SCENE_CHANGE_THRESHOLD` | 0.3 | Scene detection sensitivity (0-1) |
| `MIN_FRAME_INTERVAL` | 2.0 | Minimum seconds between captures |
| `TAB_LINE_COUNT` | 6 | Expected lines in guitar tab |
| `OUTPUT_DPI` | 150 | PDF image quality |

### Video Caching and Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_VIDEO_CACHE` | True | Cache downloaded videos to avoid re-downloading |
| `VIDEO_CACHE_MAX_SIZE_GB` | 5 | Maximum cache size before cleanup |
| `VIDEO_CACHE_MAX_AGE_HOURS` | 24 | Delete cached videos older than this |
| `DOWNLOAD_SLEEP_INTERVAL` | 5 | Seconds to wait between downloads |
| `MAX_DOWNLOADS_PER_HOUR` | 50 | Safety limit (well under YouTube's limits) |

**Rate Limiting Protection:**
- Automatic sleep between downloads to be respectful to YouTube
- Tracks downloads per hour to stay under limits
- Cached videos are reused, reducing total downloads
- Clear error messages when limits are approached

### Video Quality

For tab extraction, **lowest quality is recommended** (and is the default). Guitar tabs are text/lines that are readable even at 240p-360p. Using lowest quality:
- Saves bandwidth and storage
- Downloads 5-10x faster
- Reduces processing time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/extract` | POST | Start extraction job |
| `/status/{job_id}` | GET | Check job status |
| `/download/{job_id}` | GET | Download completed PDF |
| `/health` | GET | Health check |

## How It Works

1. **Download**: The video is downloaded using yt-dlp at lowest quality (sufficient for text)
2. **Tab-Aware Frame Extraction**: OpenCV scans the video for actual tab content changes
3. **Tab Detection**: Each frame is analyzed for horizontal parallel lines (characteristic of guitar tabs)
4. **Deduplication**: Similar/duplicate frames are removed
5. **PDF Generation**: Unique tab frames are compiled into a PDF

### Tab-Aware Comparison

Many guitar tab videos have:
- A background video of someone playing guitar
- A moving highlight/cursor showing the current note
- The actual tab content only changes every few seconds

**Tab-Aware Mode** (enabled by default) uses advanced techniques to detect when the actual tab notation changes, while ignoring:
- Background video movement
- Moving note highlights/cursors
- Color changes and animations

This is done by:
- **Binarization**: Converting frames to pure black/white to remove color highlights
- **Edge Detection**: Comparing structural content (lines and numbers) only
- **Region Focus**: Analyzing only the tab area, not the entire frame

If you're having issues, try disabling Tab-Aware Mode in Advanced Options.

## Troubleshooting

### "No guitar tabs detected"

- The video may not contain recognizable tab notation
- Try adjusting the scene change sensitivity (lower = more frames)
- Ensure the video actually shows guitar tablature (6 horizontal lines)
- **Increase Intro Skip**: If the video has a long intro (logo, talking, etc.), increase the "Intro Skip" value in Advanced Options. The default is 30 seconds.

### Download fails

- Check your internet connection
- If using cookies, ensure they are valid and not expired
- Some videos may be geo-restricted

### Slow processing

- Processing time depends on video length
- Longer videos may take several minutes
- The progress bar shows current status

## Development

The codebase is designed to be easy to modify:

- Each module has a single responsibility
- Configuration is centralized in `config.py`
- Type hints and docstrings throughout
- No complex build steps required

### Running Tests

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run all tests
pytest tests/ -v

# Run just the downloader tests
pytest tests/test_downloader.py -v

# Run manual download test
python tests/test_downloader.py
```

### Project Structure for Agents

The code follows agent-friendly patterns:
- Flat file structure (max 2 levels deep)
- Single-file modules with clear purposes
- No complex inheritance hierarchies
- All magic numbers in config.py
- Comprehensive docstrings

## License

MIT License
