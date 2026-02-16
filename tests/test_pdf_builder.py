"""
Tests for the PDF builder module.

Tests PDF generation using programmatically generated test images.
No external files or network access required.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pdf_builder import (
    calculate_image_dimensions,
    format_timestamp,
    build_pdf,
    build_simple_pdf,
    TabPDF,
)
from backend.tab_detector import TabFrame, TabRegion


# ============================================================================
# Test Data Generators
# ============================================================================

def create_test_image(width=800, height=200, color=(255, 255, 255)):
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color)
    return img


def create_tab_frame(width=800, height=200, timestamp=0.0, frame_number=0):
    """Create a TabFrame with a test image."""
    img = create_test_image(width, height)
    
    return TabFrame(
        image=img,
        timestamp=timestamp,
        frame_number=frame_number,
        tab_region=TabRegion(top=10, bottom=190, left=50, right=750, confidence=0.8),
    )


# ============================================================================
# Unit Tests - Pure Functions
# ============================================================================

class TestFormatTimestamp:
    """Tests for format_timestamp function."""
    
    def test_formats_seconds(self):
        """Should format seconds correctly."""
        assert format_timestamp(0) == "00:00"
        assert format_timestamp(30) == "00:30"
        assert format_timestamp(59) == "00:59"
    
    def test_formats_minutes(self):
        """Should format minutes correctly."""
        assert format_timestamp(60) == "01:00"
        assert format_timestamp(90) == "01:30"
        assert format_timestamp(125) == "02:05"
    
    def test_formats_large_values(self):
        """Should handle large values."""
        assert format_timestamp(3600) == "60:00"  # 1 hour
        assert format_timestamp(3661) == "61:01"
    
    def test_handles_floats(self):
        """Should handle float input."""
        assert format_timestamp(30.5) == "00:30"
        assert format_timestamp(90.9) == "01:30"


class TestCalculateImageDimensions:
    """Tests for calculate_image_dimensions function."""
    
    def test_fits_within_bounds(self):
        """Calculated dimensions should fit within max bounds."""
        img = create_test_image(800, 600)
        
        width, height = calculate_image_dimensions(img, max_width=100, max_height=100)
        
        assert width <= 100
        assert height <= 100
    
    def test_preserves_aspect_ratio(self):
        """Should preserve aspect ratio."""
        img = create_test_image(800, 400)  # 2:1 aspect ratio
        
        width, height = calculate_image_dimensions(img, max_width=200, max_height=200)
        
        # Aspect ratio should be approximately 2:1
        ratio = width / height
        assert 1.9 < ratio < 2.1
    
    def test_width_constrained(self):
        """When width is the constraint, should fit width exactly."""
        img = create_test_image(1000, 100)  # Very wide image
        
        width, height = calculate_image_dimensions(img, max_width=100, max_height=100)
        
        # Width should be the constraint
        assert abs(width - 100) < 1
    
    def test_height_constrained(self):
        """When height is the constraint, should fit height."""
        img = create_test_image(100, 1000)  # Very tall image
        
        width, height = calculate_image_dimensions(img, max_width=100, max_height=100)
        
        # Height should be the constraint
        assert abs(height - 100) < 1


# ============================================================================
# Integration Tests - PDF Generation
# ============================================================================

class TestBuildPdf:
    """Tests for build_pdf function."""
    
    def test_creates_pdf_file(self):
        """Should create a PDF file."""
        frames = [
            create_tab_frame(timestamp=0.0),
            create_tab_frame(timestamp=5.0),
            create_tab_frame(timestamp=10.0),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            result_path = build_pdf(frames, output_path=output_path, title="Test Tabs")
            
            assert result_path.exists()
            assert result_path.suffix == ".pdf"
    
    def test_pdf_has_content(self):
        """Generated PDF should have non-zero size."""
        frames = [create_tab_frame()]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            build_pdf(frames, output_path=output_path)
            
            file_size = output_path.stat().st_size
            assert file_size > 0
    
    def test_handles_single_frame(self):
        """Should handle single frame input."""
        frames = [create_tab_frame()]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            result = build_pdf(frames, output_path=output_path)
            
            assert result.exists()
    
    def test_handles_many_frames(self):
        """Should handle many frames (multiple pages)."""
        frames = [
            create_tab_frame(timestamp=i * 5.0, frame_number=i * 30)
            for i in range(20)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            result = build_pdf(frames, output_path=output_path)
            
            assert result.exists()
            # Should be larger due to more images
            assert output_path.stat().st_size > 1000
    
    def test_raises_on_empty_frames(self):
        """Should raise error for empty frames list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            with pytest.raises(ValueError):
                build_pdf([], output_path=output_path)
    
    def test_auto_generates_filename(self):
        """Should auto-generate filename if not provided."""
        frames = [create_tab_frame()]
        
        # Don't provide output_path
        result = build_pdf(frames, title="Auto Name Test")
        
        try:
            assert result.exists()
            assert result.suffix == ".pdf"
            assert "Auto_Name_Test" in result.name
        finally:
            # Cleanup
            if result.exists():
                result.unlink()
    
    def test_handles_rgba_images(self):
        """Should handle RGBA images (with alpha channel)."""
        img = Image.new("RGBA", (800, 200), (255, 255, 255, 128))
        frame = TabFrame(
            image=img,
            timestamp=0.0,
            frame_number=0,
            tab_region=None,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            result = build_pdf([frame], output_path=output_path)
            
            assert result.exists()


class TestBuildSimplePdf:
    """Tests for build_simple_pdf function."""
    
    def test_creates_pdf_from_images(self):
        """Should create PDF from list of PIL images."""
        images = [
            create_test_image(800, 200),
            create_test_image(800, 300),
            create_test_image(600, 200),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "simple.pdf"
            
            result = build_simple_pdf(images, output_path, title="Simple Test")
            
            assert result.exists()
            assert result.stat().st_size > 0
    
    def test_raises_on_empty_images(self):
        """Should raise error for empty image list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "simple.pdf"
            
            with pytest.raises(ValueError):
                build_simple_pdf([], output_path)


class TestTabPDF:
    """Tests for TabPDF class."""
    
    def test_creates_pdf_instance(self):
        """Should create PDF instance with title."""
        pdf = TabPDF(title="Test Title")
        
        assert pdf.title == "Test Title"
    
    def test_adds_page(self):
        """Should be able to add pages."""
        pdf = TabPDF(title="Test")
        pdf.add_page()
        
        assert pdf.page_no() == 1
    
    def test_supports_alias_nb_pages(self):
        """Should support page number aliasing."""
        pdf = TabPDF(title="Test")
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Should not raise an error
        assert True


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
