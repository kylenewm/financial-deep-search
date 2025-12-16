"""
Tests for section_locator.py - Deterministic SEC section extraction.

Tests verify:
1. TOC anchor extraction (Method A)
2. DOM scan extraction (Method B)
3. Fuzzy fallback (Method C)
4. Method selection priority
5. Fail-closed behavior
6. Bounds and determinism
"""
from __future__ import annotations

import pytest

from open_deep_research.section_locator import (
    extract_item_1a,
    extract_risk_factors_from_html,
    MIN_CHARS,
    MAX_CHARS_FUZZY,
    _normalize_text,
    _matches_item_1a,
    _matches_next_item,
    _is_header_like,
)
from open_deep_research.models import RiskFactorsExtract


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def html_with_toc_anchor():
    """HTML with working TOC anchor link to Item 1A."""
    # Generate enough text to meet MIN_CHARS
    risk_text = " ".join(["This is a risk factor paragraph."] * 200)  # ~7000 chars
    
    return f"""
    <html>
    <body>
        <div id="toc">
            <h2>Table of Contents</h2>
            <p><a href="#item1a">Item 1A. Risk Factors</a></p>
            <p><a href="#item2">Item 2. Properties</a></p>
        </div>
        
        <div id="item1a">
            <h2>Item 1A. Risk Factors</h2>
            <p>{risk_text}</p>
        </div>
        
        <div id="item2">
            <h2>Item 2. Properties</h2>
            <p>We own several properties.</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def html_with_header_no_toc():
    """HTML with Item 1A header but no TOC."""
    risk_text = " ".join(["This is a risk factor about market conditions."] * 200)
    
    return f"""
    <html>
    <body>
        <h2>Item 1. Business</h2>
        <p>We are a technology company.</p>
        
        <h2>Item 1A. Risk Factors</h2>
        <p>{risk_text}</p>
        
        <h2>Item 1B. Unresolved Staff Comments</h2>
        <p>None.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_risk_factors_only():
    """HTML with only "Risk Factors" heading (no Item 1A)."""
    risk_text = " ".join(["We face significant competitive risks."] * 200)
    
    return f"""
    <html>
    <body>
        <h3>Risk Factors</h3>
        <p>{risk_text}</p>
        
        <h3>Properties</h3>
        <p>We lease our facilities.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_minimal():
    """HTML with minimal content for fuzzy extraction."""
    risk_text = " ".join(["Various risks could affect operations."] * 200)
    
    return f"""
    <html>
    <body>
        <p>Some preamble text.</p>
        <p>Risk Factors</p>
        <p>{risk_text}</p>
    </body>
    </html>
    """


@pytest.fixture
def html_no_risk_factors():
    """HTML with no risk factors section."""
    return """
    <html>
    <body>
        <h2>Item 1. Business</h2>
        <p>We are a company.</p>
        
        <h2>Item 2. Properties</h2>
        <p>We own buildings.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_too_short():
    """HTML with text too short for valid extraction."""
    return """
    <html>
    <body>
        <h2>Item 1A. Risk Factors</h2>
        <p>Short text.</p>
    </body>
    </html>
    """


@pytest.fixture
def html_10q_part_collision():
    """10-Q with both Part I and Part II containing Item 1."""
    part1_text = " ".join(["Part I financial discussion."] * 200)
    part2_text = " ".join(["Part II risk factors disclosure."] * 200)
    
    return f"""
    <html>
    <body>
        <div id="toc">
            <h2>Table of Contents</h2>
            <p><a href="#part1item1">Part I - Item 1. Financial Statements</a></p>
            <p><a href="#part1item1a">Part I - Item 1A. Risk Factors</a></p>
            <p><a href="#part2item1">Part II - Item 1. Legal Proceedings</a></p>
        </div>
        
        <div id="part1item1">
            <h2>Part I - Item 1. Financial Statements</h2>
            <p>{part1_text}</p>
        </div>
        
        <div id="part1item1a">
            <h2>Part I - Item 1A. Risk Factors</h2>
            <p>{part2_text}</p>
        </div>
        
        <div id="part2item1">
            <h2>Part II - Item 1. Legal Proceedings</h2>
            <p>No material legal proceedings.</p>
        </div>
    </body>
    </html>
    """


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_normalize_text(self):
        """Test whitespace normalization."""
        assert _normalize_text("  hello   world  ") == "hello world"
        assert _normalize_text("\n\nhello\n\nworld\n\n") == "hello world"
        assert _normalize_text("no change") == "no change"
    
    def test_matches_item_1a_patterns(self):
        """Test Item 1A pattern matching."""
        assert _matches_item_1a("Item 1A. Risk Factors")
        assert _matches_item_1a("ITEM 1A - RISK FACTORS")
        assert _matches_item_1a("Item 1A")
        assert _matches_item_1a("Risk Factors")
        assert _matches_item_1a("risk factors")
        
        assert not _matches_item_1a("Item 2. Properties")
        assert not _matches_item_1a("Business Overview")
    
    def test_matches_next_item(self):
        """Test next section pattern matching."""
        assert _matches_next_item("Item 1B. Unresolved Staff Comments")
        assert _matches_next_item("Item 2. Properties")
        assert _matches_next_item("ITEM 2 - PROPERTIES")
        assert _matches_next_item("Unresolved Staff Comments")
        assert _matches_next_item("Legal Proceedings")
        
        assert not _matches_next_item("Item 1A. Risk Factors")
        assert not _matches_next_item("Random text")


# =============================================================================
# TOC Anchor Extraction Tests (Method A)
# =============================================================================


class TestTocAnchorExtraction:
    """Test Method A: TOC anchor jump."""
    
    def test_extracts_via_toc_anchor(self, html_with_toc_anchor):
        """Verify TOC anchor extraction works."""
        result = extract_item_1a(html_with_toc_anchor)
        
        assert result is not None
        assert result.method == "toc_anchor"
        assert result.confidence == "HIGH"
        assert "item1a" in result.anchor_id
        assert result.char_count >= MIN_CHARS
        assert "risk factor paragraph" in result.text.lower()
    
    def test_toc_anchor_is_preferred(self, html_with_toc_anchor):
        """Verify TOC anchor is chosen over other methods when available."""
        result = extract_item_1a(html_with_toc_anchor)
        
        # Should use TOC anchor, not DOM scan
        assert result.method == "toc_anchor"
        assert result.confidence == "HIGH"


# =============================================================================
# DOM Scan Extraction Tests (Method B)
# =============================================================================


class TestDomScanExtraction:
    """Test Method B: DOM scan headers."""
    
    def test_extracts_via_dom_scan(self, html_with_header_no_toc):
        """Verify DOM scan extraction works when no TOC."""
        result = extract_item_1a(html_with_header_no_toc)
        
        assert result is not None
        assert result.method == "dom_scan"
        assert result.confidence == "MED"
        assert result.char_count >= MIN_CHARS
        assert "market conditions" in result.text.lower()
    
    def test_handles_risk_factors_header(self, html_risk_factors_only):
        """Verify extraction works with just 'Risk Factors' heading."""
        result = extract_item_1a(html_risk_factors_only)
        
        assert result is not None
        assert result.method in ("dom_scan", "fuzzy")  # Either is acceptable
        assert "competitive risks" in result.text.lower()
    
    def test_stops_at_next_section(self, html_with_header_no_toc):
        """Verify extraction stops before Item 1B."""
        result = extract_item_1a(html_with_header_no_toc)
        
        assert result is not None
        # Should not include Item 1B content
        assert "unresolved staff comments" not in result.text.lower()


# =============================================================================
# Fuzzy Extraction Tests (Method C)
# =============================================================================


class TestFuzzyExtraction:
    """Test Method C: Fuzzy fallback."""
    
    def test_falls_back_to_fuzzy(self, html_minimal):
        """Verify fuzzy extraction as last resort."""
        result = extract_item_1a(html_minimal)
        
        assert result is not None
        # Could be dom_scan or fuzzy depending on HTML structure
        assert result.method in ("dom_scan", "fuzzy")
        assert "risks" in result.text.lower()
    
    def test_fuzzy_confidence_is_low(self, html_minimal):
        """Verify fuzzy extraction has LOW confidence."""
        result = extract_item_1a(html_minimal)
        
        if result and result.method == "fuzzy":
            assert result.confidence == "LOW"
            assert "fallback" in result.reason.lower() or "fuzzy" in result.reason.lower()


# =============================================================================
# Failure Cases
# =============================================================================


class TestFailureCases:
    """Test fail-closed behavior."""
    
    def test_returns_none_when_no_risk_factors(self, html_no_risk_factors):
        """Verify None returned when no risk factors section."""
        result = extract_item_1a(html_no_risk_factors)
        
        assert result is None
    
    def test_returns_none_for_short_html(self, html_too_short):
        """Verify None returned when extracted text too short."""
        result = extract_item_1a(html_too_short)
        
        # Should be None because text doesn't meet MIN_CHARS
        assert result is None
    
    def test_returns_none_for_empty_html(self):
        """Verify None returned for empty HTML."""
        assert extract_item_1a("") is None
        assert extract_item_1a("<html></html>") is None
    
    def test_returns_none_for_very_short_html(self):
        """Verify None returned for HTML < 1000 chars."""
        result = extract_item_1a("<html><body><p>Short</p></body></html>")
        assert result is None


# =============================================================================
# Determinism and Bounds
# =============================================================================


class TestDeterminismAndBounds:
    """Test deterministic behavior and extraction bounds."""
    
    def test_extraction_is_deterministic(self, html_with_toc_anchor):
        """Verify same input produces same output."""
        result1 = extract_item_1a(html_with_toc_anchor)
        result2 = extract_item_1a(html_with_toc_anchor)
        
        assert result1 is not None
        assert result2 is not None
        assert result1.text == result2.text
        assert result1.method == result2.method
        assert result1.confidence == result2.confidence
        assert result1.char_count == result2.char_count
    
    def test_extraction_bounded_by_max_chars(self):
        """Verify fuzzy extraction respects MAX_CHARS_FUZZY."""
        # Create extremely long HTML
        huge_text = "Risk Factors " + "x" * (MAX_CHARS_FUZZY * 2)
        html = f"<html><body><p>{huge_text}</p></body></html>"
        
        result = extract_item_1a(html)
        
        if result:
            assert result.char_count <= MAX_CHARS_FUZZY + 1000  # Some tolerance


# =============================================================================
# Model Field Tests
# =============================================================================


class TestRiskFactorsExtractModel:
    """Test RiskFactorsExtract model fields."""
    
    def test_all_fields_populated(self, html_with_toc_anchor):
        """Verify all model fields are populated."""
        result = extract_item_1a(html_with_toc_anchor)
        
        assert result is not None
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.method in ("toc_anchor", "dom_scan", "fuzzy")
        assert result.confidence in ("HIGH", "MED", "LOW")
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        assert isinstance(result.char_count, int)
        assert result.char_count > 0
    
    def test_is_reliable_for_high_confidence(self, html_with_toc_anchor):
        """Verify is_reliable() returns True for HIGH confidence."""
        result = extract_item_1a(html_with_toc_anchor)
        
        assert result is not None
        assert result.confidence == "HIGH"
        assert result.is_reliable() is True
    
    def test_is_reliable_for_low_confidence(self, html_minimal):
        """Verify is_reliable() for non-HIGH confidence."""
        result = extract_item_1a(html_minimal)
        
        if result and result.confidence != "HIGH":
            assert result.is_reliable() is False
    
    def test_confidence_icon(self, html_with_toc_anchor):
        """Verify confidence icon mapping."""
        result = extract_item_1a(html_with_toc_anchor)
        
        assert result is not None
        icon = result.get_confidence_icon()
        assert icon in ("🟢", "🟡", "🔴", "⚪")


# =============================================================================
# Part I/II Collision Handling
# =============================================================================


class TestPartCollision:
    """Test handling of 10-Q Part I/II collisions."""
    
    def test_extracts_item_1a_from_10q(self, html_10q_part_collision):
        """Verify Item 1A extraction works in 10-Q with Part I/II."""
        result = extract_item_1a(html_10q_part_collision)
        
        assert result is not None
        # Should extract the Item 1A content (Part I - Item 1A. Risk Factors)
        assert "risk factors disclosure" in result.text.lower()
    
    def test_distinguishes_item_1a_from_item_1(self, html_10q_part_collision):
        """Verify Item 1A is extracted, not Item 1."""
        result = extract_item_1a(html_10q_part_collision)
        
        assert result is not None
        # Should get Risk Factors (Item 1A), not Financial Statements (Item 1)
        # The anchor should be part1item1a, not part1item1
        if result.anchor_id:
            assert "1a" in result.anchor_id.lower()


# =============================================================================
# Convenience Wrapper Tests
# =============================================================================


class TestConvenienceWrapper:
    """Test extract_risk_factors_from_html wrapper."""
    
    def test_wrapper_matches_main_function(self, html_with_toc_anchor):
        """Verify wrapper produces same result as main function."""
        result1 = extract_item_1a(html_with_toc_anchor)
        result2 = extract_risk_factors_from_html(html_with_toc_anchor)
        
        assert result1 is not None
        assert result2 is not None
        assert result1.text == result2.text
        assert result1.method == result2.method

