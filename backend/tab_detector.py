"""
Tab detection module for identifying guitar tablature in video frames.

Supports multiple detection modes:
- "auto": Auto-detect between full screen and embedded tabs
- "full_screen": Tabs occupy entire screen
- "embedded": Tabs in portion of screen (detect region)
- "numbers": Detect tabs by finding numbers/digits (fret numbers)

Key differences from music notation:
- Guitar tabs have 6 lines (not 5 like music staff)
- Tabs have NUMBERS on lines (fret positions: 0-24)
- Music notation has notes (circles) on/between lines
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import cv2
import numpy as np
from PIL import Image

from .config import (
    TAB_LINE_COUNT,
    TAB_LINE_TOLERANCE,
    MIN_LINE_LENGTH_RATIO,
    LINE_DETECTION_THRESHOLD,
    TAB_DETECTION_MODE,
    DETECT_MULTIPLE_TAB_SYSTEMS,
    MAX_TAB_SYSTEMS_PER_FRAME,
)
from .processor import ExtractedFrame

logger = logging.getLogger(__name__)


class TabDetectionMode(Enum):
    """Detection modes for guitar tablature."""
    AUTO = "auto"
    FULL_SCREEN = "full_screen"
    EMBEDDED = "embedded"
    NUMBERS = "numbers"


class SystemType(Enum):
    """Type of line system detected."""
    TAB = "tab"              # Guitar tablature (6 lines with numbers)
    NOTATION = "notation"    # Standard music notation (5 lines)
    UNKNOWN = "unknown"


@dataclass
class TabSystem:
    """Represents a single tab system (6 lines) in a frame."""
    top: int
    bottom: int
    left: int
    right: int
    line_count: int
    system_type: SystemType = SystemType.UNKNOWN
    has_numbers: bool = False  # True if fret numbers detected
    number_density: float = 0.0  # How many numbers per unit area
    paired_with: Optional[int] = None  # Index of paired notation/tab system


@dataclass
class NotationTabPair:
    """
    Represents a notation+tab PAIR that should be captured together.
    
    In many guitar videos, music notation (5 lines) appears directly above
    guitar tabs (6 lines). These should be treated as ONE logical unit.
    """
    notation: Optional[TabSystem]  # The notation system (5 lines), may be None
    tab: TabSystem  # The tab system (6 lines with fret numbers)
    top: int  # Top of the entire pair (notation top or tab top if no notation)
    bottom: int  # Bottom of the entire pair (tab bottom)
    left: int
    right: int
    
    @property
    def is_paired(self) -> bool:
        """True if this has both notation and tab."""
        return self.notation is not None


@dataclass
class TabRegion:
    """Represents a detected tab region in a frame."""
    top: int
    bottom: int
    left: int
    right: int
    confidence: float  # 0-1, how confident we are this is a tab
    detection_mode: str = "embedded"  # Mode used for detection
    systems: List[TabSystem] = field(default_factory=list)  # Individual tab systems
    is_full_screen: bool = False  # True if tabs occupy entire screen


@dataclass
class TabFrame:
    """A frame confirmed to contain guitar tablature."""
    image: Image.Image
    timestamp: float
    frame_number: int
    tab_region: Optional[TabRegion]
    is_full_screen: bool = False


def detect_horizontal_lines(
    image: np.ndarray,
    min_length_ratio: float = MIN_LINE_LENGTH_RATIO,
    threshold: int = LINE_DETECTION_THRESHOLD,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect horizontal lines in an image using edge detection and Hough transform.
    
    Args:
        image: Grayscale or BGR image
        min_length_ratio: Minimum line length as ratio of image width
        threshold: Hough line detection threshold
        
    Returns:
        List of lines as (x1, y1, x2, y2) tuples
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Calculate minimum line length
    min_line_length = int(image.shape[1] * min_length_ratio)
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=10,
    )
    
    if lines is None:
        return []
    
    # Filter for horizontal lines (slope close to 0)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 == 0:
            continue  # Vertical line
        
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Keep lines that are nearly horizontal (within 5 degrees)
        if angle < 5 or angle > 175:
            horizontal_lines.append((x1, y1, x2, y2))
    
    # Merge nearby lines (edge detection often finds both edges of a line)
    # Use threshold of 3 to merge duplicate detections but not separate notation lines
    horizontal_lines = merge_nearby_lines(horizontal_lines, y_threshold=3)
    
    return horizontal_lines


def merge_nearby_lines(
    lines: List[Tuple[int, int, int, int]],
    y_threshold: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Merge horizontal lines that are very close together (within y_threshold pixels).
    
    Edge detection often detects both edges of a single line, resulting in
    duplicate detections. This merges them into a single line.
    
    Args:
        lines: List of horizontal lines as (x1, y1, x2, y2)
        y_threshold: Maximum y-distance to consider lines as duplicates
        
    Returns:
        List of merged lines
    """
    if not lines:
        return []
    
    # Sort by y-center
    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    
    merged = []
    current_group = [sorted_lines[0]]
    
    for line in sorted_lines[1:]:
        current_y = (line[1] + line[3]) / 2
        group_y = (current_group[-1][1] + current_group[-1][3]) / 2
        
        if abs(current_y - group_y) <= y_threshold:
            # Same group, merge
            current_group.append(line)
        else:
            # Different group, output the merged line and start new group
            merged.append(merge_line_group(current_group))
            current_group = [line]
    
    # Don't forget the last group
    merged.append(merge_line_group(current_group))
    
    return merged


def merge_line_group(
    lines: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """
    Merge a group of nearby lines into a single line.
    
    Takes the average y-position and extends x to cover all lines.
    """
    if len(lines) == 1:
        return lines[0]
    
    # Average y position
    avg_y1 = int(sum(l[1] for l in lines) / len(lines))
    avg_y2 = int(sum(l[3] for l in lines) / len(lines))
    
    # Extend x to cover all lines
    min_x = min(l[0] for l in lines)
    max_x = max(l[2] for l in lines)
    
    return (min_x, avg_y1, max_x, avg_y2)


def find_parallel_line_groups(
    lines: List[Tuple[int, int, int, int]],
    expected_count: int = TAB_LINE_COUNT,
    tolerance: int = TAB_LINE_TOLERANCE,
    spacing_tolerance: float = 0.3,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Find groups of parallel lines with roughly equal spacing.
    Guitar tabs have 6 equally-spaced horizontal lines.
    
    Args:
        lines: List of horizontal lines
        expected_count: Expected number of lines in a tab (6 for guitar)
        tolerance: Allow +/- this many lines
        spacing_tolerance: How much spacing variation to allow (0-1)
        
    Returns:
        List of line groups that could be guitar tabs
    """
    if len(lines) < expected_count - tolerance:
        return []
    
    # Sort lines by y-coordinate (vertical position)
    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    
    # Get y-centers of each line
    y_centers = [(l[1] + l[3]) / 2 for l in sorted_lines]
    
    # Find groups of equally-spaced lines
    groups = []
    min_count = expected_count - tolerance
    max_count = expected_count + tolerance
    
    for i in range(len(sorted_lines)):
        # Try to build a group starting from this line
        group = [sorted_lines[i]]
        group_indices = [i]
        
        for j in range(i + 1, len(sorted_lines)):
            if len(group) >= max_count:
                break
            
            # Calculate expected spacing based on first two lines
            if len(group) == 1:
                group.append(sorted_lines[j])
                group_indices.append(j)
            else:
                # Check if this line fits the spacing pattern
                # Calculate spacing from the first two lines in the group
                first_y = y_centers[group_indices[0]]
                second_y = y_centers[group_indices[1]]
                expected_spacing = second_y - first_y
                
                if expected_spacing <= 0:
                    continue
                
                actual_y = y_centers[j]
                expected_y = first_y + expected_spacing * len(group)
                
                if abs(actual_y - expected_y) < expected_spacing * spacing_tolerance:
                    group.append(sorted_lines[j])
                    group_indices.append(j)
        
        if min_count <= len(group) <= max_count:
            groups.append(group)
    
    return groups


def find_all_line_systems(
    image: np.ndarray,
    lines: List[Tuple[int, int, int, int]],
) -> List[TabSystem]:
    """
    Find all line systems in an image - both notation (5 lines) and tabs (6 lines).
    
    This detects:
    - Music notation: 5 equally-spaced lines (staff)
    - Guitar tabs: 6 equally-spaced lines with numbers
    
    Then pairs notation+tab systems that appear together.
    
    Args:
        image: BGR image
        lines: Detected horizontal lines
        
    Returns:
        List of TabSystem objects, sorted by vertical position
    """
    systems = []
    
    # FIRST: Find 6-line TAB groups (most reliable)
    tab_groups = find_parallel_line_groups(lines, expected_count=6, tolerance=1)
    
    # Track which lines are used by TAB groups
    used_line_indices = set()
    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    
    for group in tab_groups:
        all_y = [l[1] for l in group] + [l[3] for l in group]
        all_x = [l[0] for l in group] + [l[2] for l in group]
        
        top = max(0, min(all_y) - 5)
        bottom = min(image.shape[0], max(all_y) + 5)
        left = max(0, min(all_x))
        right = min(image.shape[1], max(all_x))
        
        # Mark lines as used
        for g_line in group:
            for idx, line in enumerate(sorted_lines):
                if line == g_line:
                    used_line_indices.add(idx)
        
        # Check for numbers
        has_nums, density, _ = detect_numbers_in_region(
            image, (top, bottom, left, right)
        )
        
        systems.append(TabSystem(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            line_count=len(group),
            system_type=SystemType.TAB,
            has_numbers=has_nums,
            number_density=density,
        ))
    
    # SECOND: Find 5-line NOTATION groups from REMAINING lines only
    remaining_lines = [l for idx, l in enumerate(sorted_lines) if idx not in used_line_indices]
    
    if remaining_lines:
        notation_groups = find_parallel_line_groups(remaining_lines, expected_count=5, tolerance=1)
        
        for group in notation_groups:
            all_y = [l[1] for l in group] + [l[3] for l in group]
            all_x = [l[0] for l in group] + [l[2] for l in group]
            
            top = max(0, min(all_y) - 5)
            bottom = min(image.shape[0], max(all_y) + 5)
            left = max(0, min(all_x))
            right = min(image.shape[1], max(all_x))
            
            systems.append(TabSystem(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                line_count=len(group),
                system_type=SystemType.NOTATION,
                has_numbers=False,
                number_density=0.0,
            ))
    
    # Sort by vertical position
    systems.sort(key=lambda s: s.top)
    
    # Pair notation+tab systems
    # Look for a 5-line system immediately followed by a 6-line system
    for i, sys in enumerate(systems):
        if sys.system_type == SystemType.NOTATION:
            # Look for a tab system right below this
            for j in range(i + 1, min(i + 3, len(systems))):
                if systems[j].system_type == SystemType.TAB:
                    # Check if they're close together (within reasonable distance)
                    gap = systems[j].top - sys.bottom
                    system_height = sys.bottom - sys.top
                    
                    if 0 < gap < system_height * 1.5:
                        # They're paired!
                        sys.paired_with = j
                        systems[j].paired_with = i
                        break
    
    return systems


def extract_tab_systems_only(
    systems: List[TabSystem],
) -> List[TabSystem]:
    """
    Extract only the TAB systems (6 lines), ignoring notation.
    
    When notation+tab pairs are detected, returns only the tab portion.
    Removes duplicate tabs that are part of the same pair.
    
    Args:
        systems: List of all detected systems
        
    Returns:
        List of TAB systems only
    """
    tab_systems = []
    
    for sys in systems:
        if sys.system_type == SystemType.TAB:
            tab_systems.append(sys)
        elif sys.system_type == SystemType.NOTATION:
            # Skip notation - we'll get the paired tab instead
            pass
    
    return tab_systems


def find_notation_tab_pairs(
    systems: List[TabSystem],
) -> List[NotationTabPair]:
    """
    Find notation+tab PAIRS from a list of detected systems.
    
    When notation (5 lines) appears immediately above tabs (6 lines),
    they form a PAIR and should be captured together as a single frame.
    
    The pairing logic:
    1. For each notation system, look for a tab system directly below it
    2. If found within reasonable distance, they form a pair
    3. Unpaired tab systems become standalone pairs (notation=None)
    4. Standalone notation systems (no tab below) are ignored
    
    Args:
        systems: List of TabSystem objects (from find_all_line_systems)
        
    Returns:
        List of NotationTabPair objects, sorted by vertical position (top first)
    """
    pairs = []
    used_indices = set()  # Track which systems have been paired
    
    # Sort systems by vertical position
    sorted_systems = sorted(enumerate(systems), key=lambda x: x[1].top)
    
    # First pass: find notation+tab pairs
    for idx, sys in sorted_systems:
        if sys.system_type != SystemType.NOTATION:
            continue
        if idx in used_indices:
            continue
            
        # Look for a tab system below this notation
        notation_bottom = sys.bottom
        best_tab_idx = None
        best_tab_gap = float('inf')
        
        for other_idx, other_sys in sorted_systems:
            if other_sys.system_type != SystemType.TAB:
                continue
            if other_idx in used_indices:
                continue
            
            # Tab must be BELOW the notation
            if other_sys.top < notation_bottom:
                continue
            
            # Calculate gap between notation bottom and tab top
            gap = other_sys.top - notation_bottom
            notation_height = sys.bottom - sys.top
            
            # Gap should be reasonable (within 1.5x the notation height)
            # This allows for some spacing but not too much
            if gap < notation_height * 2.0 and gap < best_tab_gap:
                best_tab_idx = other_idx
                best_tab_gap = gap
        
        if best_tab_idx is not None:
            # Found a pair!
            tab_sys = systems[best_tab_idx]
            
            # Calculate bounds of the entire pair
            pair_top = sys.top
            pair_bottom = tab_sys.bottom
            pair_left = min(sys.left, tab_sys.left)
            pair_right = max(sys.right, tab_sys.right)
            
            pairs.append(NotationTabPair(
                notation=sys,
                tab=tab_sys,
                top=pair_top,
                bottom=pair_bottom,
                left=pair_left,
                right=pair_right,
            ))
            
            used_indices.add(idx)
            used_indices.add(best_tab_idx)
            logger.debug(f"Found notation+tab pair at y={pair_top}-{pair_bottom}")
    
    # Second pass: create standalone pairs for unpaired tab systems
    for idx, sys in sorted_systems:
        if sys.system_type != SystemType.TAB:
            continue
        if idx in used_indices:
            continue
        
        # This tab has no paired notation - create standalone pair
        pairs.append(NotationTabPair(
            notation=None,
            tab=sys,
            top=sys.top,
            bottom=sys.bottom,
            left=sys.left,
            right=sys.right,
        ))
        used_indices.add(idx)
        logger.debug(f"Found standalone tab at y={sys.top}-{sys.bottom}")
    
    # Sort pairs by vertical position (top first)
    pairs.sort(key=lambda p: p.top)
    
    return pairs


# ============================================================================
# Number/Digit Detection (for distinguishing tabs from music notation)
# ============================================================================

def detect_numbers_in_region(
    image: np.ndarray,
    region: Tuple[int, int, int, int] = None,
) -> Tuple[bool, float, int]:
    """
    Detect if a region contains numbers/digits (fret numbers).
    
    Guitar tabs have numbers (0-24) on the lines representing fret positions.
    Music notation has circles (notes) on/between lines.
    
    Args:
        image: BGR or grayscale image
        region: (top, bottom, left, right) or None for whole image
        
    Returns:
        Tuple of (has_numbers, density, count)
        - has_numbers: True if numbers detected
        - density: Number of digits per 1000 pixels
        - count: Total digit count
    """
    # Extract region if specified
    if region:
        top, bottom, left, right = region
        roi = image[top:bottom, left:right]
    else:
        roi = image
    
    if roi.size == 0:
        return False, 0.0, 0
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Apply thresholding to isolate text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (potential digits)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that look like digits
    digit_count = 0
    h, w = roi.shape[:2]
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Digits have specific aspect ratio and size
        aspect_ratio = ch / max(cw, 1)
        area = cw * ch
        
        # Digit characteristics:
        # - Aspect ratio typically 1.0-2.5 (taller than wide)
        # - Not too small (noise) or too large (other elements)
        min_size = max(5, min(h, w) * 0.01)
        max_size = min(h, w) * 0.15
        
        if (0.5 <= aspect_ratio <= 3.0 and 
            min_size <= cw <= max_size and 
            min_size <= ch <= max_size * 2):
            digit_count += 1
    
    # Calculate density (digits per 1000 pixels)
    total_pixels = max(1, h * w)
    density = (digit_count * 1000) / total_pixels
    
    # Need at least some digits and reasonable density
    has_numbers = digit_count >= 5 and density > 0.05
    
    return has_numbers, density, digit_count


def is_full_screen_tab(image: np.ndarray) -> bool:
    """
    Detect if tabs occupy the entire screen.
    
    Full-screen tabs typically have:
    - Multiple groups of 6 horizontal lines
    - Lines spanning most of the screen width
    - High density of numbers
    
    Args:
        image: BGR image
        
    Returns:
        True if full-screen tab layout detected
    """
    h, w = image.shape[:2]
    
    # Detect horizontal lines
    lines = detect_horizontal_lines(image, min_length_ratio=0.5)  # Need longer lines
    
    # For full-screen tabs, we expect many long horizontal lines
    if len(lines) < TAB_LINE_COUNT:
        return False
    
    # Check if lines span most of the width
    line_lengths = [(abs(l[2] - l[0])) / w for l in lines]
    long_lines = sum(1 for length in line_lengths if length > 0.6)
    
    # Need multiple long lines (at least one tab system)
    if long_lines < TAB_LINE_COUNT - TAB_LINE_TOLERANCE:
        return False
    
    # Check for numbers
    has_nums, density, count = detect_numbers_in_region(image)
    
    # Full-screen tabs should have numbers
    if has_nums and density > 0.1:
        return True
    
    # Also check if multiple tab systems are present
    line_groups = find_parallel_line_groups(lines)
    if len(line_groups) >= 2:  # Multiple tab systems = full screen
        return True
    
    return False


def detect_multiple_tab_systems(
    image: np.ndarray,
    lines: List[Tuple[int, int, int, int]],
) -> List[TabSystem]:
    """
    Detect multiple tab systems in an image.
    
    This function:
    1. Finds both notation (5 lines) and tab (6 lines) systems
    2. Identifies notation+tab pairs (common in sheet music/tab videos)
    3. Returns ONLY the TAB portions, not notation
    
    Args:
        image: BGR image
        lines: Detected horizontal lines
        
    Returns:
        List of TabSystem objects (only TABS, not notation)
    """
    # Use the new comprehensive system finder
    all_systems = find_all_line_systems(image, lines)
    
    if not all_systems:
        # Fallback to old method
        line_groups = find_parallel_line_groups(lines)
        if not line_groups:
            return []
        
        systems = []
        for group in line_groups[:MAX_TAB_SYSTEMS_PER_FRAME]:
            all_y = [l[1] for l in group] + [l[3] for l in group]
            all_x = [l[0] for l in group] + [l[2] for l in group]
            
            top = max(0, min(all_y) - 10)
            bottom = min(image.shape[0], max(all_y) + 10)
            left = max(0, min(all_x))
            right = min(image.shape[1], max(all_x))
            
            has_nums, density, _ = detect_numbers_in_region(
                image, (top, bottom, left, right)
            )
            
            systems.append(TabSystem(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                line_count=len(group),
                system_type=SystemType.TAB,
                has_numbers=has_nums,
                number_density=density,
            ))
        
        systems.sort(key=lambda s: s.top)
        return systems[:MAX_TAB_SYSTEMS_PER_FRAME]
    
    # Filter to only TAB systems (not notation)
    tab_systems = extract_tab_systems_only(all_systems)
    
    # Log what we found for debugging
    notation_count = sum(1 for s in all_systems if s.system_type == SystemType.NOTATION)
    tab_count = len(tab_systems)
    if notation_count > 0:
        logger.debug(f"Found {notation_count} notation + {tab_count} tab systems, using tabs only")
    
    return tab_systems[:MAX_TAB_SYSTEMS_PER_FRAME]


# ============================================================================
# Main Detection Functions
# ============================================================================

def detect_tab_region(
    image: np.ndarray,
    mode: str = None,
) -> Optional[TabRegion]:
    """
    Detect if an image contains guitar tablature and find its region.
    
    Args:
        image: BGR image
        mode: Detection mode ("auto", "full_screen", "embedded", "numbers")
              If None, uses TAB_DETECTION_MODE from config
        
    Returns:
        TabRegion if tablature is detected, None otherwise
    """
    mode = mode or TAB_DETECTION_MODE
    h, w = image.shape[:2]
    
    # Detect horizontal lines
    lines = detect_horizontal_lines(image)
    
    # Auto-detect mode
    if mode == "auto":
        if is_full_screen_tab(image):
            mode = "full_screen"
        else:
            mode = "embedded"
    
    # Full-screen mode: use entire frame
    if mode == "full_screen":
        # Detect multiple systems
        systems = []
        if DETECT_MULTIPLE_TAB_SYSTEMS:
            systems = detect_multiple_tab_systems(image, lines)
        
        # Check for numbers
        has_nums, density, _ = detect_numbers_in_region(image)
        
        # For full-screen, we're more lenient
        if len(lines) >= TAB_LINE_COUNT - TAB_LINE_TOLERANCE or has_nums:
            return TabRegion(
                top=0,
                bottom=h,
                left=0,
                right=w,
                confidence=0.8 if has_nums else 0.6,
                detection_mode=mode,
                systems=systems,
                is_full_screen=True,
            )
        return None
    
    # Numbers mode: detect by finding digit-like features
    if mode == "numbers":
        has_nums, density, count = detect_numbers_in_region(image)
        
        if has_nums:
            # Find where the numbers are concentrated
            # For now, use entire frame if numbers detected
            return TabRegion(
                top=0,
                bottom=h,
                left=0,
                right=w,
                confidence=min(1.0, density * 2),
                detection_mode=mode,
                is_full_screen=True,
            )
        return None
    
    # Embedded mode: detect region containing tabs (original behavior)
    if len(lines) < TAB_LINE_COUNT - TAB_LINE_TOLERANCE:
        return None
    
    # Find groups of parallel lines that could be tabs
    line_groups = find_parallel_line_groups(lines)
    
    if not line_groups:
        return None
    
    # Detect multiple systems if enabled
    systems = []
    if DETECT_MULTIPLE_TAB_SYSTEMS:
        systems = detect_multiple_tab_systems(image, lines)
    
    # Use the largest/best group
    best_group = max(line_groups, key=len)
    
    # Calculate bounding region
    all_y = [l[1] for l in best_group] + [l[3] for l in best_group]
    all_x = [l[0] for l in best_group] + [l[2] for l in best_group]
    
    top = min(all_y) - 20  # Add padding
    bottom = max(all_y) + 20
    left = min(all_x)
    right = max(all_x)
    
    # Calculate confidence based on how close to 6 lines we found
    line_count_score = 1.0 - abs(len(best_group) - TAB_LINE_COUNT) / TAB_LINE_COUNT
    
    # Also consider how many total horizontal lines we found
    # (tabs should have clear, prominent lines)
    line_clarity_score = min(1.0, len(lines) / 20)
    
    # Boost confidence if numbers detected
    has_nums, density, _ = detect_numbers_in_region(
        image, (top, bottom, left, right)
    )
    number_score = 0.2 if has_nums else 0.0
    
    confidence = 0.5 * line_count_score + 0.3 * line_clarity_score + number_score
    
    return TabRegion(
        top=max(0, top),
        bottom=min(h, bottom),
        left=max(0, left),
        right=min(w, right),
        confidence=confidence,
        detection_mode=mode,
        systems=systems,
    )


def is_tab_frame(
    frame: ExtractedFrame,
    min_confidence: float = 0.5,
    mode: str = None,
) -> bool:
    """
    Check if a frame contains guitar tablature.
    
    Args:
        frame: ExtractedFrame to check
        min_confidence: Minimum confidence threshold
        mode: Detection mode (None uses config default)
        
    Returns:
        True if frame likely contains tabs
    """
    # Convert PIL image to OpenCV format
    cv_image = np.array(frame.image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    region = detect_tab_region(cv_image, mode=mode)
    
    return region is not None and region.confidence >= min_confidence


def filter_tab_frames(
    frames: List[ExtractedFrame],
    min_confidence: float = 0.5,
    mode: str = None,
) -> List[TabFrame]:
    """
    Filter frames to keep only those containing guitar tablature.
    
    Args:
        frames: List of extracted frames
        min_confidence: Minimum detection confidence
        mode: Detection mode ("auto", "full_screen", "embedded", "numbers")
        
    Returns:
        List of TabFrame objects (frames with tabs)
    """
    tab_frames = []
    
    for frame in frames:
        # Convert PIL to OpenCV
        cv_image = np.array(frame.image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Detect tab region
        region = detect_tab_region(cv_image, mode=mode)
        
        if region is not None and region.confidence >= min_confidence:
            tab_frames.append(TabFrame(
                image=frame.image,
                timestamp=frame.timestamp,
                frame_number=frame.frame_number,
                tab_region=region,
                is_full_screen=region.is_full_screen,
            ))
            logger.debug(
                f"Tab detected at {frame.timestamp:.2f}s "
                f"(confidence={region.confidence:.2f}, mode={region.detection_mode})"
            )
    
    logger.info(f"Filtered {len(tab_frames)} tab frames from {len(frames)} total")
    return tab_frames


def crop_to_tab_region(frame: TabFrame) -> Image.Image:
    """
    Crop a frame to show only the tab region.
    
    Args:
        frame: TabFrame with detected tab region
        
    Returns:
        Cropped PIL Image
    """
    if frame.tab_region is None:
        return frame.image
    
    region = frame.tab_region
    return frame.image.crop((region.left, region.top, region.right, region.bottom))


def extract_top_notation_tab_pair(frame: TabFrame) -> Optional[Image.Image]:
    """
    Extract the TOP notation+tab PAIR from a frame.
    
    When videos show notation (5 lines) paired with tabs (6 lines) below,
    this extracts BOTH together as a single unit. This is the preferred
    extraction method for videos that show notation+tab pairs.
    
    When multiple pairs are on screen (e.g., 2 notation+tab pairs),
    only the TOP pair is returned to avoid duplicates when scrolling.
    
    This function uses a robust approach:
    1. Find TAB systems (6 equally-spaced lines) - these are most reliable
    2. For the top TAB, look for ANY lines above it within reasonable distance
    3. Include those lines (notation) together with the TAB
    
    Args:
        frame: TabFrame with detected tab region and systems
        
    Returns:
        Cropped PIL Image containing the top notation+tab pair together
    """
    if frame.tab_region is None:
        return frame.image
    
    region = frame.tab_region
    img_array = np.array(frame.image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Find all TAB systems (6 lines with numbers - most reliable)
    tab_systems = [s for s in region.systems if s.system_type == SystemType.TAB and s.line_count >= 5]
    
    if not tab_systems:
        # No TAB systems, use full region
        return frame.image.crop((region.left, region.top, region.right, region.bottom))
    
    # Sort by vertical position, take the TOP tab system
    tab_systems.sort(key=lambda s: s.top)
    top_tab = tab_systems[0]
    
    # Now look for ALL horizontal lines above this tab
    # This will catch notation lines even if we can't form a complete 5-line group
    all_lines = detect_horizontal_lines(img_array)
    
    # Find lines above the top_tab
    notation_top = top_tab.top
    tab_top_y = top_tab.top
    
    # Look for lines in the region above the tab (within 2x tab height)
    tab_height = top_tab.bottom - top_tab.top
    search_region_top = max(0, tab_top_y - tab_height * 2)
    
    lines_above = []
    for line in all_lines:
        y_center = (line[1] + line[3]) / 2
        if search_region_top <= y_center < tab_top_y - 5:  # At least 5px gap
            lines_above.append(line)
    
    if lines_above:
        # There are lines above - likely notation. Extend crop to include them.
        top_y_values = [line[1] for line in lines_above] + [line[3] for line in lines_above]
        notation_top = min(top_y_values)
        logger.debug(f"Found {len(lines_above)} notation lines above tab, extending crop from {tab_top_y} to {notation_top}")
    
    # Build the crop region
    # Use larger top padding to capture notes/symbols above the notation staff
    padding = 15
    top_padding = 25  # Extra padding at top for notes above staff
    crop_top = max(0, notation_top - top_padding)
    crop_bottom = min(frame.image.height, top_tab.bottom + padding)
    crop_left = max(0, top_tab.left - padding)
    crop_right = min(frame.image.width, top_tab.right + padding)
    
    # Also extend horizontally if notation is wider
    if lines_above:
        notation_left = min(line[0] for line in lines_above)
        notation_right = max(line[2] for line in lines_above)
        crop_left = max(0, min(crop_left, notation_left - padding))
        crop_right = min(frame.image.width, max(crop_right, notation_right + padding))
    
    logger.debug(f"Extracting notation+tab: y={crop_top}-{crop_bottom}")
    
    return frame.image.crop((crop_left, crop_top, crop_right, crop_bottom))


def extract_top_tab_system(frame: TabFrame) -> Optional[Image.Image]:
    """
    Extract the TOP notation+tab pair from a frame.
    
    NOTE: This function now extracts notation+tab TOGETHER (not just tabs).
    When notation (5 lines) appears above tabs (6 lines), both are included
    as a single unit. This prevents splitting related content.
    
    For backwards compatibility, this function name is kept, but it now
    delegates to extract_top_notation_tab_pair().
    
    Args:
        frame: TabFrame with detected tab region and systems
        
    Returns:
        Cropped PIL Image containing the top notation+tab pair
    """
    return extract_top_notation_tab_pair(frame)


def binarize_for_comparison(image: Image.Image, size: Tuple[int, int] = (200, 100)) -> np.ndarray:
    """
    Convert image to binary (black/white) for content comparison.
    
    This ignores colors and highlights, focusing only on the actual tab content
    (lines and numbers).
    
    Args:
        image: PIL Image
        size: Size to resize to for comparison
        
    Returns:
        Binary numpy array (0s and 255s only)
    """
    # Convert to grayscale
    gray = image.convert("L")
    
    # Resize for comparison (larger size preserves more detail)
    resized = gray.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy
    arr = np.array(resized)
    
    # Use OpenCV's Otsu thresholding for better results
    # This automatically finds the optimal threshold
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def compare_tab_content(
    img1: Image.Image,
    img2: Image.Image,
    check_scroll: bool = True,
) -> Tuple[float, bool]:
    """
    Compare two tab images for content similarity, ignoring colors/highlights.
    
    Also checks if one image might be a scrolled version of the other
    (e.g., bottom half of img1 matches top half of img2).
    
    Args:
        img1: First image
        img2: Second image
        check_scroll: Whether to check for scroll patterns
        
    Returns:
        Tuple of (similarity 0-1, is_scrolled_duplicate bool)
    """
    # Binarize both images
    bin1 = binarize_for_comparison(img1)
    bin2 = binarize_for_comparison(img2)
    
    # Direct comparison
    diff = np.abs(bin1.astype(float) - bin2.astype(float))
    direct_similarity = 1.0 - (np.mean(diff) / 255.0)
    
    is_scrolled = False
    
    if check_scroll and direct_similarity < 0.95:
        # Check if bottom half of img1 matches top half of img2 (scroll down)
        h = bin1.shape[0]
        half = h // 2
        
        bottom1 = bin1[half:, :]
        top2 = bin2[:half, :]
        
        if bottom1.shape == top2.shape:
            scroll_diff = np.abs(bottom1.astype(float) - top2.astype(float))
            scroll_similarity = 1.0 - (np.mean(scroll_diff) / 255.0)
            
            if scroll_similarity > 0.92:  # High match = likely scrolled (strict threshold)
                is_scrolled = True
        
        # Also check reverse (scroll up)
        top1 = bin1[:half, :]
        bottom2 = bin2[half:, :]
        
        if top1.shape == bottom2.shape:
            scroll_diff = np.abs(top1.astype(float) - bottom2.astype(float))
            scroll_similarity = 1.0 - (np.mean(scroll_diff) / 255.0)
            
            if scroll_similarity > 0.92:
                is_scrolled = True
    
    return direct_similarity, is_scrolled


def remove_duplicate_tabs(
    frames: List[TabFrame],
    similarity_threshold: float = 0.95,
    use_top_system_only: bool = True,
    detect_scrolling: bool = True,
) -> List[TabFrame]:
    """
    Remove duplicate or very similar tab frames.
    
    Uses binarized comparison to ignore colors/highlights.
    Optionally detects scrolling patterns (content that scrolled up/down).
    
    When use_top_system_only=True (default), compares the TOP notation+tab PAIR
    from each frame. This is the recommended mode for videos that show:
    - Notation (5 lines) paired with tabs (6 lines) below
    - Two such pairs visible at once (content scrolls every ~5 seconds)
    
    By comparing only the top pair, we avoid capturing the same content twice
    as it scrolls from top position to bottom position.
    
    Args:
        frames: List of tab frames
        similarity_threshold: How similar frames must be to be considered duplicates
        use_top_system_only: If True, compare only the top notation+tab pair
        detect_scrolling: If True, detect when one frame is scrolled version of another
        
    Returns:
        Deduplicated list of tab frames
    """
    if len(frames) <= 1:
        return frames
    
    unique_frames = [frames[0]]
    scroll_duplicates = 0
    direct_duplicates = 0
    
    for frame in frames[1:]:
        is_duplicate = False
        
        # Get the image to compare (top system only if multiple systems present)
        if use_top_system_only:
            current_img = extract_top_tab_system(frame)
        else:
            current_img = frame.image
        
        # Compare with recent unique frames
        for unique_frame in unique_frames[-5:]:  # Compare with last 5 unique frames
            if use_top_system_only:
                prev_img = extract_top_tab_system(unique_frame)
            else:
                prev_img = unique_frame.image
            
            # Use binarized comparison (ignores colors/highlights)
            similarity, is_scrolled = compare_tab_content(
                current_img,
                prev_img,
                check_scroll=detect_scrolling,
            )
            
            if similarity > similarity_threshold:
                is_duplicate = True
                direct_duplicates += 1
                break
            
            if is_scrolled:
                is_duplicate = True
                scroll_duplicates += 1
                logger.debug(f"Detected scroll duplicate at {frame.timestamp:.1f}s")
                break
        
        if not is_duplicate:
            unique_frames.append(frame)
    
    removed = len(frames) - len(unique_frames)
    logger.info(
        f"Removed {removed} duplicate frames "
        f"({direct_duplicates} direct, {scroll_duplicates} scroll)"
    )
    return unique_frames
