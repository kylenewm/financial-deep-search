"""
Section Locator: Deterministic extraction of SEC filing sections.

This module provides layered, deterministic extraction of Item 1A (Risk Factors)
and other sections from SEC HTML filings. No LLM. No network calls.

Extraction Strategy (in priority order):
A) TOC anchor jump - Most reliable, uses Table of Contents links
B) DOM scan headers - Medium confidence, scans for header-like elements
C) Fuzzy fallback - Low confidence, bounded text extraction

Key Design Decisions:
- Always return explicit method + confidence + reason
- If extracted text < MIN_CHARS, treat as failure and try next method
- If all fail, return None (caller MUST fail closed)
- Deterministic: same input → same output
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Tuple
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag

from open_deep_research.models import RiskFactorsExtract

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Minimum chars for valid extraction (< this = retry next method)
MIN_CHARS = 2000

# Maximum chars to extract in fuzzy mode (prevent runaway)
MAX_CHARS_FUZZY = 30000

# Maximum paragraphs in fuzzy mode
MAX_PARAGRAPHS_FUZZY = 50

# Header patterns for Item 1A / Risk Factors
ITEM_1A_PATTERNS = [
    re.compile(r"item\s*1a\.?\s*[-–—]?\s*risk\s+factors?", re.IGNORECASE),
    re.compile(r"item\s*1a\.?", re.IGNORECASE),
    re.compile(r"risk\s+factors?", re.IGNORECASE),
]

# Patterns to identify next section (stop extraction)
# NOTE: Use word boundaries and exclude "continued" variants to avoid false positives
# e.g., "Item 1A (Continued)" should NOT trigger section end
NEXT_ITEM_PATTERNS = [
    re.compile(r"\bitem\s*1b\.?(?!\s*\(?\s*continued)", re.IGNORECASE),
    re.compile(r"\bitem\s*2\.?(?!\s*\(?\s*continued)", re.IGNORECASE),
    re.compile(r"\bitem\s*3\.?(?!\s*\(?\s*continued)", re.IGNORECASE),
    re.compile(r"\bunresolved\s+staff\s+comments\b", re.IGNORECASE),
    re.compile(r"\bproperties\b", re.IGNORECASE),
    re.compile(r"\blegal\s+proceedings\b", re.IGNORECASE),
]

# Patterns that indicate continuation (should NOT trigger section end)
# These are checked before NEXT_ITEM_PATTERNS to prevent false positives
CONTINUATION_PATTERNS = [
    re.compile(r"continued", re.IGNORECASE),
    re.compile(r"cont['']?d", re.IGNORECASE),
]

# Header-like tag names
HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "b", "strong"}

# Tags that indicate we're in a TOC region
TOC_INDICATORS = [
    "table of contents",
    "index",
    "part i",
    "part ii",
]


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode for comparison.
    
    Handles:
    - Unicode normalization (NFKC) to unify equivalent characters
    - Whitespace collapse
    - Common SEC HTML artifacts like &nbsp;
    """
    # Unicode normalization first to handle things like:
    # - \u2019 (RIGHT SINGLE QUOTATION MARK) → ' (APOSTROPHE)
    # - \xa0 (NO-BREAK SPACE) → ' ' (SPACE)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).strip()


def _get_text_length(element: Tag) -> int:
    """Get total text length of an element."""
    return len(_normalize_text(element.get_text()))


def _is_header_like(element: Tag) -> bool:
    """Check if an element looks like a section header."""
    if element.name in HEADER_TAGS:
        return True
    
    # Check for bold/font styling that indicates header
    style = element.get("style", "")
    if "font-weight" in style.lower() and ("bold" in style.lower() or "700" in style):
        return True
    
    # Check for small text that looks like a header (< 200 chars)
    text = _normalize_text(element.get_text())
    if len(text) < 200 and any(p.search(text) for p in ITEM_1A_PATTERNS + NEXT_ITEM_PATTERNS):
        return True
    
    return False


def _is_in_toc_region(element: Tag) -> bool:
    """Check if element appears to be inside a Table of Contents."""
    # Walk up the tree looking for TOC indicators
    current = element
    depth = 0
    max_depth = 10  # Don't go too far up
    
    while current and depth < max_depth:
        if hasattr(current, "get_text"):
            text = _normalize_text(current.get_text()).lower()
            # If we're in a small container with TOC-like text, skip it
            if _get_text_length(current) < 5000:
                for indicator in TOC_INDICATORS:
                    if indicator in text[:500]:  # Check near start
                        return True
        current = current.parent
        depth += 1
    
    return False


def _matches_item_1a(text: str) -> bool:
    """Check if text matches Item 1A or Risk Factors patterns."""
    normalized = _normalize_text(text)
    return any(p.search(normalized) for p in ITEM_1A_PATTERNS)


def _matches_next_item(text: str) -> bool:
    """Check if text matches a section after Item 1A.
    
    Excludes "continued" variants to prevent false positives like
    "Item 1A (Continued)" from triggering section end.
    """
    normalized = _normalize_text(text)
    
    # First check if this is a continuation - if so, it's NOT a next item
    if any(p.search(normalized) for p in CONTINUATION_PATTERNS):
        return False
    
    return any(p.search(normalized) for p in NEXT_ITEM_PATTERNS)


def _collect_paragraphs_until_next_item(
    start_element: Tag,
    max_chars: int = MAX_CHARS_FUZZY,
    max_paragraphs: int = MAX_PARAGRAPHS_FUZZY,
) -> Tuple[str, int]:
    """
    Collect text from start_element until we hit the next Item header.
    
    Returns: (collected_text, paragraph_count)
    """
    paragraphs: List[str] = []
    total_chars = 0
    paragraph_count = 0
    
    # Start from the element and iterate through siblings
    current = start_element
    
    while current and paragraph_count < max_paragraphs and total_chars < max_chars:
        if isinstance(current, NavigableString):
            text = str(current).strip()
            if text:
                paragraphs.append(text)
                total_chars += len(text)
                paragraph_count += 1
        elif isinstance(current, Tag):
            # Check if this is the next section header
            text = _normalize_text(current.get_text())
            if _is_header_like(current) and _matches_next_item(text):
                break
            
            # Collect text from this element
            text = current.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
                total_chars += len(text)
                paragraph_count += 1
        
        # Move to next sibling, or to parent's next sibling
        current = current.next_sibling
        if current is None and start_element.parent:
            # Try parent's siblings
            parent = start_element.parent
            while parent and current is None:
                current = parent.next_sibling
                parent = parent.parent
    
    return "\n\n".join(paragraphs), paragraph_count


# =============================================================================
# Method A: TOC Anchor Jump
# =============================================================================


def _extract_via_toc_anchor(soup: BeautifulSoup) -> Optional[RiskFactorsExtract]:
    """
    Method A: Extract Item 1A via Table of Contents anchor.
    
    Strategy:
    1. Find a region that looks like a TOC (contains "Table of Contents")
    2. Find an <a> tag with text matching "Item 1A" or "Risk Factors"
    3. Follow the href to the actual content
    4. Collect text until the next section
    
    Returns: RiskFactorsExtract or None if method fails
    """
    logger.debug("Trying TOC anchor extraction...")
    
    # Find all anchor links
    anchors = soup.find_all("a", href=True)
    
    for anchor in anchors:
        href = anchor.get("href", "")
        text = _normalize_text(anchor.get_text())
        
        # Check if this anchor text matches Item 1A patterns
        if not _matches_item_1a(text):
            continue
        
        # Skip if text is too long (not a TOC link)
        if len(text) > 120:
            continue
        
        # Check that href points to an internal anchor
        if not href.startswith("#"):
            continue
        
        anchor_id = unquote(href[1:])  # Remove # and URL-decode
        
        # Find the target element
        target = soup.find(id=anchor_id)
        if target is None:
            # Try finding by name attribute
            target = soup.find(attrs={"name": anchor_id})
        
        if target is None:
            logger.debug(f"Could not find target for anchor #{anchor_id}")
            continue
        
        # Skip if target is in TOC region (self-referential)
        if _is_in_toc_region(target):
            continue
        
        # Collect text from target
        collected_text, para_count = _collect_paragraphs_until_next_item(target)
        
        if len(collected_text) >= MIN_CHARS:
            logger.info(f"TOC anchor extraction successful: {len(collected_text)} chars, {para_count} paragraphs")
            return RiskFactorsExtract(
                text=collected_text,
                method="toc_anchor",
                confidence="HIGH",
                reason=f"Found via TOC link to anchor #{anchor_id}",
                char_count=len(collected_text),
                anchor_id=anchor_id,
            )
    
    logger.debug("TOC anchor extraction failed: no valid anchors found")
    return None


# =============================================================================
# Method B: DOM Scan Headers
# =============================================================================


def _extract_via_dom_scan(soup: BeautifulSoup) -> Optional[RiskFactorsExtract]:
    """
    Method B: Extract Item 1A by scanning for header elements.
    
    Strategy:
    1. Iterate through all elements in document order
    2. Find header-like elements matching Item 1A patterns
    3. Skip elements inside TOC regions
    4. Collect text until next Item header
    
    Returns: RiskFactorsExtract or None if method fails
    """
    logger.debug("Trying DOM scan extraction...")
    
    # Find all potential header elements
    candidates = []
    
    for tag_name in HEADER_TAGS:
        for element in soup.find_all(tag_name):
            text = _normalize_text(element.get_text())
            if len(text) < 200 and _matches_item_1a(text):
                candidates.append((element, text))
    
    # Also check div/p/span with bold styling or short matching text
    for element in soup.find_all(["div", "p", "span", "font"]):
        text = _normalize_text(element.get_text())
        if len(text) < 200 and _matches_item_1a(text):
            # Check if this looks like a header (styling or structure)
            if _is_header_like(element) or len(text) < 50:
                candidates.append((element, text))
    
    # Try each candidate
    for element, header_text in candidates:
        # Skip if in TOC
        if _is_in_toc_region(element):
            continue
        
        # Collect text from after this header
        collected_text, para_count = _collect_paragraphs_until_next_item(element)
        
        if len(collected_text) >= MIN_CHARS:
            logger.info(f"DOM scan extraction successful: {len(collected_text)} chars from header '{header_text[:50]}...'")
            return RiskFactorsExtract(
                text=collected_text,
                method="dom_scan",
                confidence="MED",
                reason=f"Found header matching Item 1A pattern",
                char_count=len(collected_text),
                header_text=header_text,
            )
    
    logger.debug("DOM scan extraction failed: no valid headers found")
    return None


# =============================================================================
# Method C: Fuzzy Fallback
# =============================================================================


def _extract_via_fuzzy(soup: BeautifulSoup) -> Optional[RiskFactorsExtract]:
    """
    Method C: Fuzzy fallback extraction.
    
    Strategy:
    1. Find ANY occurrence of "risk factors" (case-insensitive)
    2. Take the next N paragraphs or max chars
    3. Label as LOW confidence
    
    This is the last resort - should only fire when A and B fail.
    
    Returns: RiskFactorsExtract or None if method fails
    """
    logger.debug("Trying fuzzy extraction...")
    
    # Get all text and find "risk factors"
    full_text = soup.get_text()
    
    # Find the position of "risk factors" (case-insensitive)
    match = re.search(r"risk\s+factors?", full_text, re.IGNORECASE)
    
    if not match:
        logger.debug("Fuzzy extraction failed: no 'risk factors' text found")
        return None
    
    # Take text starting from the match, limited by max chars
    start_pos = match.start()
    extracted = full_text[start_pos : start_pos + MAX_CHARS_FUZZY]
    
    # Try to cut off at a paragraph boundary
    if len(extracted) == MAX_CHARS_FUZZY:
        # Find last double newline and cut there
        last_para = extracted.rfind("\n\n")
        if last_para > MIN_CHARS:
            extracted = extracted[:last_para]
    
    if len(extracted) >= MIN_CHARS:
        logger.info(f"Fuzzy extraction: {len(extracted)} chars (low confidence)")
        return RiskFactorsExtract(
            text=extracted,
            method="fuzzy",
            confidence="LOW",
            reason="Strict locator failed; fuzzy fallback used - extracted text after 'Risk Factors' keyword",
            char_count=len(extracted),
        )
    
    logger.debug("Fuzzy extraction failed: insufficient text found")
    return None


# =============================================================================
# Main Entry Point
# =============================================================================


def extract_item_1a(html: str) -> Optional[RiskFactorsExtract]:
    """
    Extract Item 1A (Risk Factors) from an SEC filing HTML.
    
    Layered deterministic strategy:
    A) TOC anchor jump (HIGH confidence)
    B) DOM scan headers (MED confidence)
    C) Fuzzy fallback (LOW confidence)
    
    Args:
        html: Raw HTML content of SEC filing
        
    Returns:
        RiskFactorsExtract with text + method + confidence, or None if all fail.
        
    ⚠️ If None is returned, caller MUST fail closed (no fake results).
    """
    if not html or len(html) < 1000:
        logger.warning("HTML too short for extraction")
        return None
    
    soup = BeautifulSoup(html, "lxml")
    
    # Try Method A: TOC Anchor
    result = _extract_via_toc_anchor(soup)
    if result:
        return result
    
    # Try Method B: DOM Scan
    result = _extract_via_dom_scan(soup)
    if result:
        return result
    
    # Try Method C: Fuzzy Fallback
    result = _extract_via_fuzzy(soup)
    if result:
        return result
    
    # All methods failed
    logger.warning("All extraction methods failed for Item 1A")
    return None


def extract_risk_factors_from_html(html: str) -> Optional[RiskFactorsExtract]:
    """
    Convenience wrapper for extract_item_1a.
    
    This is the function to use from orchestrator signal detection.
    Same behavior as extract_item_1a but with clearer name.
    """
    return extract_item_1a(html)

