"""
FastAPI application for YouTube Tab Extractor.
Provides REST API endpoints for video processing and PDF generation.
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from .config import (
    OUTPUT_DIR,
    CLEANUP_AFTER_PROCESSING,
    SCENE_CHANGE_THRESHOLD,
    MIN_FRAME_INTERVAL,
    TAB_AWARE_COMPARISON,
    INTRO_SKIP_SECONDS,
    TAB_DETECTION_MODE,
)
from .downloader import download_video, get_video_info, cleanup_video, get_cache_stats, clear_cache
from .processor import extract_scene_changes
from .tab_detector import filter_tab_frames, remove_duplicate_tabs
from .pdf_builder import build_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YouTube Tab Extractor",
    description="Extract guitar tablature from YouTube videos and generate PDFs",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=2)


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    DETECTING = "detecting"
    GENERATING_PDF = "generating_pdf"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a video processing job."""
    id: str
    url: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    video_title: Optional[str] = None
    pdf_path: Optional[Path] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    frames_extracted: int = 0
    tabs_detected: int = 0


# In-memory job storage
jobs: Dict[str, Job] = {}


class ExtractRequest(BaseModel):
    """Request body for extract endpoint."""
    url: str
    threshold: Optional[float] = None
    min_interval: Optional[float] = None
    tab_aware: Optional[bool] = None  # Use tab-aware comparison (ignores highlights)
    intro_skip: Optional[float] = None  # Seconds to skip before checking for tabs
    detection_mode: Optional[str] = None  # "auto", "full_screen", "embedded", "numbers"


class JobResponse(BaseModel):
    """Response for job status."""
    id: str
    status: str
    progress: int
    message: str
    video_title: Optional[str] = None
    frames_extracted: int = 0
    tabs_detected: int = 0
    error: Optional[str] = None
    download_url: Optional[str] = None


def process_video(
    job: Job,
    threshold: float,
    min_interval: float,
    tab_aware: bool,
    intro_skip: float,
    detection_mode: str = "auto",
) -> None:
    """
    Process a video in a background thread.
    
    Args:
        job: Job object to update
        threshold: Scene change detection threshold
        min_interval: Minimum interval between captures
        tab_aware: Use tab-aware comparison that ignores highlights/background
        intro_skip: Seconds to wait before determining video has no tabs
        detection_mode: Tab detection mode ("auto", "full_screen", "embedded", "numbers")
    """
    video_path = None
    
    try:
        # Step 1: Get video info
        job.status = JobStatus.DOWNLOADING
        job.message = "Fetching video information..."
        job.progress = 5
        
        try:
            info = get_video_info(job.url)
            job.video_title = info.get("title", "Unknown")
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            job.video_title = "Unknown"
        
        # Step 2: Download video
        job.message = f"Downloading: {job.video_title}"
        job.progress = 10
        
        video_path = download_video(job.url)
        job.progress = 30
        
        # Step 3: Extract frames using tab-aware comparison
        job.status = JobStatus.PROCESSING
        comparison_mode = "tab-aware" if tab_aware else "legacy"
        job.message = f"Extracting frames ({comparison_mode} mode)..."
        
        def update_progress(current: int, total: int):
            # Progress from 30 to 60 during frame extraction
            if total > 0:
                job.progress = 30 + int(30 * current / total)
        
        # For full_screen mode, use lower threshold to catch abrupt changes
        effective_threshold = 0.05 if detection_mode == "full_screen" else threshold
        
        frames = extract_scene_changes(
            video_path,
            threshold=effective_threshold,
            min_interval=min_interval,
            progress_callback=update_progress,
            tab_aware=tab_aware,
        )
        job.frames_extracted = len(frames)
        job.progress = 60
        
        # Step 4: Detect tabs
        job.status = JobStatus.DETECTING
        job.message = f"Analyzing {len(frames)} frames for tabs (mode: {detection_mode})..."
        
        tab_frames = filter_tab_frames(frames, mode=detection_mode)
        
        # For full_screen mode, tabs look similar (same lines, different numbers)
        # Use very high threshold to only remove nearly identical frames
        # Also enable scroll detection and use top system only for comparison
        dup_threshold = 0.98 if detection_mode == "full_screen" else 0.90
        tab_frames = remove_duplicate_tabs(
            tab_frames,
            similarity_threshold=dup_threshold,
            use_top_system_only=True,  # Compare only top tab system
            detect_scrolling=True,      # Detect scrolled duplicates
        )
        job.tabs_detected = len(tab_frames)
        job.progress = 80
        
        if not tab_frames:
            # Check if we have frames past the intro skip period
            frames_after_intro = [f for f in frames if f.timestamp >= intro_skip]
            
            if frames_after_intro:
                # We have frames after intro but still no tabs - video likely has no tabs
                job.status = JobStatus.FAILED
                job.error = f"No guitar tabs detected after {intro_skip}s intro skip"
                job.message = "No tabs found"
            else:
                # Video is shorter than intro_skip, check all frames
                job.status = JobStatus.FAILED
                job.error = "No guitar tabs detected in this video (video may be too short)"
                job.message = "No tabs found"
            return
        
        # Step 5: Generate PDF
        job.status = JobStatus.GENERATING_PDF
        job.message = f"Generating PDF with {len(tab_frames)} tab sections..."
        
        pdf_path = build_pdf(
            tab_frames,
            title=job.video_title or "Guitar Tabs",
        )
        job.pdf_path = pdf_path
        job.progress = 100
        
        # Done!
        job.status = JobStatus.COMPLETED
        job.message = "PDF ready for download"
        
    except Exception as e:
        logger.exception(f"Error processing job {job.id}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.message = "Processing failed"
        
    finally:
        # Cleanup downloaded video
        if video_path and CLEANUP_AFTER_PROCESSING:
            cleanup_video(video_path)


@app.post("/extract", response_model=JobResponse)
async def start_extraction(request: ExtractRequest, background_tasks: BackgroundTasks):
    """
    Start a new extraction job.
    
    Returns a job ID that can be used to check status.
    """
    # Create new job
    job_id = str(uuid.uuid4())[:8]
    job = Job(id=job_id, url=request.url)
    jobs[job_id] = job
    
    # Get parameters with defaults from config
    threshold = request.threshold if request.threshold is not None else SCENE_CHANGE_THRESHOLD
    min_interval = request.min_interval if request.min_interval is not None else MIN_FRAME_INTERVAL
    tab_aware = request.tab_aware if request.tab_aware is not None else TAB_AWARE_COMPARISON
    intro_skip = request.intro_skip if request.intro_skip is not None else INTRO_SKIP_SECONDS
    detection_mode = request.detection_mode if request.detection_mode else TAB_DETECTION_MODE
    
    # Validate detection mode
    valid_modes = ["auto", "full_screen", "embedded", "numbers"]
    if detection_mode not in valid_modes:
        detection_mode = "auto"
    
    # Start processing in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        process_video,
        job,
        threshold,
        min_interval,
        tab_aware,
        intro_skip,
        detection_mode,
    )
    
    logger.info(f"Started job {job_id} for URL: {request.url} (tab_aware={tab_aware}, intro_skip={intro_skip}s, mode={detection_mode})")
    
    return JobResponse(
        id=job.id,
        status=job.status.value,
        progress=job.progress,
        message="Job started",
    )


@app.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    download_url = None
    if job.status == JobStatus.COMPLETED and job.pdf_path:
        download_url = f"/download/{job_id}"
    
    return JobResponse(
        id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        video_title=job.video_title,
        frames_extracted=job.frames_extracted,
        tabs_detected=job.tabs_detected,
        error=job.error,
        download_url=download_url,
    )


@app.get("/download/{job_id}")
async def download_pdf(job_id: str):
    """Download the generated PDF for a completed job."""
    job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job.pdf_path or not job.pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Create safe filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in (job.video_title or "tabs"))
    filename = f"{safe_title}.pdf"
    
    return FileResponse(
        path=job.pdf_path,
        filename=filename,
        media_type="application/pdf",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text())
    
    # Fallback simple HTML if frontend not found
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>YouTube Tab Extractor</title></head>
    <body>
        <h1>YouTube Tab Extractor</h1>
        <p>Frontend not found. Please create frontend/index.html</p>
    </body>
    </html>
    """)


# Mount static files for CSS/JS
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "jobs_count": len(jobs)}


@app.get("/cache")
async def cache_status():
    """Get video cache statistics."""
    stats = get_cache_stats()
    return stats


@app.delete("/cache")
async def clear_video_cache():
    """Clear the video cache."""
    deleted = clear_cache()
    return {"message": f"Cleared {deleted} cached videos"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
