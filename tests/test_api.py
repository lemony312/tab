"""
Tests for the FastAPI endpoints.

Tests API routes using FastAPI TestClient.
No external network access required (mocked where needed).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from backend.main import app, jobs, Job, JobStatus


# ============================================================================
# Test Client Setup
# ============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    # Clear jobs before each test
    jobs.clear()
    return TestClient(app)


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    job = Job(
        id="test123",
        url="https://www.youtube.com/watch?v=test",
        status=JobStatus.COMPLETED,
        progress=100,
        message="PDF ready",
        video_title="Test Video",
        frames_extracted=10,
        tabs_detected=5,
    )
    jobs["test123"] = job
    return job


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_ok(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "jobs_count" in data
    
    def test_health_check_includes_job_count(self, client, sample_job):
        """Health check should include job count."""
        response = client.get("/health")
        
        data = response.json()
        assert data["jobs_count"] == 1


# ============================================================================
# Extract Endpoint Tests
# ============================================================================

class TestExtractEndpoint:
    """Tests for POST /extract endpoint."""
    
    def test_extract_requires_url(self, client):
        """Should require URL in request body."""
        response = client.post("/extract", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_extract_accepts_valid_request(self, client):
        """Should accept valid request and return job ID."""
        response = client.post("/extract", json={
            "url": "https://www.youtube.com/watch?v=test123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        # Status may be "pending" or "downloading" depending on timing
        assert data["status"] in ["pending", "downloading"]
    
    def test_extract_accepts_optional_params(self, client):
        """Should accept optional parameters."""
        response = client.post("/extract", json={
            "url": "https://www.youtube.com/watch?v=test123",
            "threshold": 0.5,
            "min_interval": 3.0,
            "tab_aware": True,
            "intro_skip": 60.0,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
    
    def test_extract_creates_job(self, client):
        """Should create job in jobs dict."""
        response = client.post("/extract", json={
            "url": "https://www.youtube.com/watch?v=test123"
        })
        
        data = response.json()
        job_id = data["id"]
        
        assert job_id in jobs
        assert jobs[job_id].url == "https://www.youtube.com/watch?v=test123"


# ============================================================================
# Status Endpoint Tests
# ============================================================================

class TestStatusEndpoint:
    """Tests for GET /status/{job_id} endpoint."""
    
    def test_status_returns_job_info(self, client, sample_job):
        """Should return job status info."""
        response = client.get(f"/status/{sample_job.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_job.id
        assert data["status"] == "completed"
        assert data["progress"] == 100
    
    def test_status_returns_404_for_unknown_job(self, client):
        """Should return 404 for unknown job ID."""
        response = client.get("/status/unknown_id")
        
        assert response.status_code == 404
    
    def test_status_includes_video_title(self, client, sample_job):
        """Should include video title in response."""
        response = client.get(f"/status/{sample_job.id}")
        
        data = response.json()
        assert data["video_title"] == "Test Video"
    
    def test_status_includes_frame_counts(self, client, sample_job):
        """Should include frame counts."""
        response = client.get(f"/status/{sample_job.id}")
        
        data = response.json()
        assert data["frames_extracted"] == 10
        assert data["tabs_detected"] == 5
    
    def test_status_includes_download_url_when_complete(self, client, sample_job):
        """Should include download URL when job is complete."""
        # Set pdf_path for completed job
        sample_job.pdf_path = Path("/tmp/test.pdf")
        
        response = client.get(f"/status/{sample_job.id}")
        
        data = response.json()
        assert data["download_url"] == f"/download/{sample_job.id}"
    
    def test_status_no_download_url_when_pending(self, client):
        """Should not include download URL when job is pending."""
        job = Job(id="pending1", url="test", status=JobStatus.PENDING)
        jobs["pending1"] = job
        
        response = client.get("/status/pending1")
        
        data = response.json()
        assert data["download_url"] is None


# ============================================================================
# Download Endpoint Tests
# ============================================================================

class TestDownloadEndpoint:
    """Tests for GET /download/{job_id} endpoint."""
    
    def test_download_returns_404_for_unknown_job(self, client):
        """Should return 404 for unknown job ID."""
        response = client.get("/download/unknown_id")
        
        assert response.status_code == 404
    
    def test_download_returns_400_for_incomplete_job(self, client):
        """Should return 400 for incomplete job."""
        job = Job(id="incomplete", url="test", status=JobStatus.PROCESSING)
        jobs["incomplete"] = job
        
        response = client.get("/download/incomplete")
        
        assert response.status_code == 400
    
    def test_download_returns_404_for_missing_file(self, client, sample_job):
        """Should return 404 if PDF file doesn't exist."""
        sample_job.pdf_path = Path("/nonexistent/path.pdf")
        
        response = client.get(f"/download/{sample_job.id}")
        
        assert response.status_code == 404


# ============================================================================
# Frontend Endpoint Tests
# ============================================================================

class TestFrontendEndpoint:
    """Tests for GET / endpoint."""
    
    def test_serves_html(self, client):
        """Should serve HTML content."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_html_contains_form(self, client):
        """HTML should contain the extract form."""
        response = client.get("/")
        
        # Check for key elements
        assert "extract-form" in response.text or "YouTube" in response.text


# ============================================================================
# Job Status Transitions Tests
# ============================================================================

class TestJobStatusTransitions:
    """Tests for job status values."""
    
    def test_all_statuses_are_valid(self, client):
        """All JobStatus values should be serializable."""
        for status in JobStatus:
            job = Job(
                id=f"test_{status.value}",
                url="test",
                status=status,
            )
            jobs[job.id] = job
            
            response = client.get(f"/status/{job.id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == status.value


# ============================================================================
# Request Validation Tests
# ============================================================================

class TestRequestValidation:
    """Tests for request validation."""
    
    def test_threshold_can_be_float(self, client):
        """Threshold should accept float values."""
        response = client.post("/extract", json={
            "url": "https://youtube.com/watch?v=test",
            "threshold": 0.25,
        })
        
        assert response.status_code == 200
    
    def test_min_interval_can_be_float(self, client):
        """Min interval should accept float values."""
        response = client.post("/extract", json={
            "url": "https://youtube.com/watch?v=test",
            "min_interval": 1.5,
        })
        
        assert response.status_code == 200
    
    def test_tab_aware_can_be_boolean(self, client):
        """Tab aware should accept boolean values."""
        for value in [True, False]:
            response = client.post("/extract", json={
                "url": "https://youtube.com/watch?v=test",
                "tab_aware": value,
            })
            
            assert response.status_code == 200
    
    def test_intro_skip_can_be_number(self, client):
        """Intro skip should accept numeric values."""
        response = client.post("/extract", json={
            "url": "https://youtube.com/watch?v=test",
            "intro_skip": 45.0,
        })
        
        assert response.status_code == 200


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
