"""
PDF builder for compiling extracted tab frames into a single document.
Uses fpdf2 for PDF generation.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fpdf import FPDF
from PIL import Image

from .config import (
    OUTPUT_DIR,
    OUTPUT_DPI,
    PDF_PAGE_WIDTH,
    PDF_PAGE_HEIGHT,
    PDF_MARGIN,
)
from .tab_detector import (
    TabFrame,
    crop_to_tab_region,
    extract_top_tab_system,
    extract_top_notation_tab_pair,
)

logger = logging.getLogger(__name__)


class TabPDF(FPDF):
    """Custom PDF class for tab documents."""
    
    def __init__(self, title: str = "Guitar Tabs"):
        super().__init__()
        # Sanitize title to ASCII-only (Helvetica doesn't support Unicode)
        self.title = sanitize_title(title)
        self.set_auto_page_break(auto=True, margin=PDF_MARGIN)


def sanitize_title(title: str) -> str:
    """
    Sanitize title to contain only ASCII characters.
    
    Helvetica font doesn't support Unicode, so we strip non-ASCII characters.
    """
    # Keep only ASCII printable characters
    sanitized = "".join(c if ord(c) < 128 and c.isprintable() else "" for c in title)
    # Clean up multiple spaces
    sanitized = " ".join(sanitized.split())
    return sanitized.strip() or "Guitar Tabs"
    
    def header(self):
        """Add header to each page."""
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5, self.title, align="C")
        self.ln(8)
    
    def footer(self):
        """Add footer with page numbers."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def calculate_image_dimensions(
    image: Image.Image,
    max_width: float,
    max_height: float,
) -> tuple[float, float]:
    """
    Calculate dimensions to fit image within bounds while preserving aspect ratio.
    
    Args:
        image: PIL Image
        max_width: Maximum width in mm
        max_height: Maximum height in mm
        
    Returns:
        Tuple of (width, height) in mm
    """
    # Get image dimensions in pixels
    img_width, img_height = image.size
    
    # Calculate aspect ratio
    aspect = img_width / img_height
    
    # Calculate dimensions that fit within bounds
    width = max_width
    height = width / aspect
    
    if height > max_height:
        height = max_height
        width = height * aspect
    
    return width, height


def build_pdf(
    frames: List[TabFrame],
    output_path: Optional[Path] = None,
    title: str = "Guitar Tabs",
    crop_to_tabs: bool = True,
    use_top_system_only: bool = True,
    images_per_page: int = 0,  # 0 = auto
) -> Path:
    """
    Build a PDF from extracted tab frames.
    
    When use_top_system_only=True (default), each frame is cropped to show
    the TOP notation+tab pair as a single unit. This means:
    - Notation (5 lines) + Tabs (6 lines) are captured TOGETHER
    - Only the top pair is used (avoiding duplicates when content scrolls)
    - Each PDF page shows complete, readable notation+tab units
    
    Args:
        frames: List of TabFrame objects
        output_path: Path for output PDF (auto-generated if None)
        title: Title for the PDF document
        crop_to_tabs: Whether to crop images to tab region only
        use_top_system_only: If True, extract top notation+tab pair only
        images_per_page: Number of images per page (0 = auto-fit)
        
    Returns:
        Path to generated PDF
    """
    if not frames:
        raise ValueError("No frames provided for PDF generation")
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title[:30])
        output_path = OUTPUT_DIR / f"{safe_title}_{timestamp}.pdf"
    
    # Create PDF
    pdf = TabPDF(title=title)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Calculate available space
    usable_width = PDF_PAGE_WIDTH - 2 * PDF_MARGIN
    usable_height = PDF_PAGE_HEIGHT - 2 * PDF_MARGIN - 15  # Account for header/footer
    
    current_y = pdf.get_y()
    
    for i, frame in enumerate(frames):
        # Get image - extract top notation+tab pair for consistent layout
        if use_top_system_only and frame.tab_region is not None:
            image = extract_top_notation_tab_pair(frame)
        elif crop_to_tabs and frame.tab_region is not None:
            image = crop_to_tab_region(frame)
        else:
            image = frame.image
        
        # Convert to RGB if necessary (PDF doesn't support RGBA)
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        # Calculate image dimensions
        img_width, img_height = calculate_image_dimensions(
            image,
            max_width=usable_width,
            max_height=usable_height / 2,  # Allow at least 2 images per page
        )
        
        # Check if we need a new page
        if current_y + img_height > PDF_PAGE_HEIGHT - PDF_MARGIN - 10:
            pdf.add_page()
            current_y = pdf.get_y()
        
        # Save image to bytes buffer
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)
        
        # Add image to PDF
        pdf.image(
            img_buffer,
            x=PDF_MARGIN,
            y=current_y,
            w=img_width,
        )
        
        # Add timestamp label
        pdf.set_y(current_y + img_height)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 3, f"[{format_timestamp(frame.timestamp)}]", align="L")
        pdf.set_text_color(0, 0, 0)
        
        current_y = pdf.get_y() + 5
        
        logger.debug(f"Added frame {i+1}/{len(frames)} to PDF")
    
    # Save PDF
    pdf.output(str(output_path))
    logger.info(f"Generated PDF: {output_path}")
    
    return output_path


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_simple_pdf(
    images: List[Image.Image],
    output_path: Path,
    title: str = "Extracted Tabs",
) -> Path:
    """
    Build a simple PDF from a list of PIL Images.
    Simplified version without TabFrame metadata.
    
    Args:
        images: List of PIL Images
        output_path: Output file path
        title: Document title
        
    Returns:
        Path to generated PDF
    """
    if not images:
        raise ValueError("No images provided")
    
    pdf = TabPDF(title=title)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    usable_width = PDF_PAGE_WIDTH - 2 * PDF_MARGIN
    usable_height = PDF_PAGE_HEIGHT - 2 * PDF_MARGIN - 15
    
    current_y = pdf.get_y()
    
    for image in images:
        # Convert to RGB
        if image.mode != "RGB":
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert("RGB")
        
        img_width, img_height = calculate_image_dimensions(
            image, usable_width, usable_height / 2
        )
        
        if current_y + img_height > PDF_PAGE_HEIGHT - PDF_MARGIN - 10:
            pdf.add_page()
            current_y = pdf.get_y()
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)
        
        pdf.image(img_buffer, x=PDF_MARGIN, y=current_y, w=img_width)
        current_y = pdf.get_y() + img_height + 5
    
    pdf.output(str(output_path))
    return output_path
