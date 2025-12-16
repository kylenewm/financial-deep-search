"""
Signal Pipeline Regression Tests (Phase 5).

Tests the end-to-end signal detection pipeline including:
- Text normalization for drift scoring
- Drift calculation stability
- Boilerplate suppression in EVENT mode
- Sentence-level diff accuracy
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_deep_research.signals import (
    analyze_risk_drift,
    calculate_drift,
    detect_boilerplate,
    normalize_text_for_diff,
    get_normalized_tokens,
    extract_sentences,
    SignalMode,
    BOILERPLATE_PATTERNS,
)
from open_deep_research.section_locator import (
    extract_item_1a,
    extract_risk_factors_from_html,
    MIN_CHARS,
)
from open_deep_research.models import DriftResult, SignalAlert


# =============================================================================
# Test Fixtures
# =============================================================================


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "filings"


@pytest.fixture
def html_10k_with_toc():
    """Load 10-K fixture with TOC anchor."""
    path = FIXTURES_DIR / "10k_with_toc.html"
    if path.exists():
        return path.read_text()
    pytest.skip("Fixture file not found")


@pytest.fixture
def html_10k_no_toc():
    """Load 10-K fixture without TOC."""
    path = FIXTURES_DIR / "10k_no_toc.html"
    if path.exists():
        return path.read_text()
    pytest.skip("Fixture file not found")


@pytest.fixture
def html_10q_risk_factors():
    """Load 10-Q fixture with Risk Factors heading only."""
    path = FIXTURES_DIR / "10q_risk_factors_only.html"
    if path.exists():
        return path.read_text()
    pytest.skip("Fixture file not found")


@pytest.fixture
def html_10q_collision():
    """Load 10-Q fixture with Part I/II collision."""
    path = FIXTURES_DIR / "10q_part_collision.html"
    if path.exists():
        return path.read_text()
    pytest.skip("Fixture file not found")


@pytest.fixture
def risk_text_v1():
    """Risk text version 1 (baseline)."""
    return """
    We face significant competition from established technology companies and emerging startups.
    Our market share could decline if competitors introduce superior products at lower prices.
    The technology industry is characterized by rapid innovation.
    Failure to keep pace could render our offerings obsolete.
    We depend on our intellectual property rights to protect our competitive advantages.
    Unauthorized use of our patents could harm our market position.
    Global economic conditions could reduce customer spending on our products.
    Currency exchange rate fluctuations could affect our international revenues.
    """


@pytest.fixture
def risk_text_v2_minor_change():
    """Risk text version 2 with minor changes (pronoun swap, formatting)."""
    return """
    The Company faces significant competition from established technology companies and emerging startups.
    The Company's market share could decline if competitors introduce superior products at lower prices.
    The technology industry is characterized by rapid innovation.
    Failure to keep pace could render the Company's offerings obsolete.
    The Company depends on intellectual property rights to protect competitive advantages.
    Unauthorized use of patents could harm market position.
    Global economic conditions could reduce customer spending on products.
    Currency exchange rate fluctuations could affect international revenues.
    """


@pytest.fixture
def risk_text_v3_major_change():
    """Risk text version 3 with major substantive changes."""
    return """
    We face significant competition from established technology companies and emerging startups.
    Our market share could decline if competitors introduce superior products at lower prices.
    The technology industry is characterized by rapid innovation.
    Failure to keep pace could render our offerings obsolete.
    We depend on our intellectual property rights to protect our competitive advantages.
    Unauthorized use of our patents could harm our market position.
    IMPORTANT: We are facing a new regulatory investigation by the SEC regarding our accounting practices.
    A class action lawsuit has been filed against us alleging securities fraud.
    Export restrictions on our products to China may significantly reduce international revenues.
    Cybersecurity breach in Q3 resulted in exposure of customer data affecting 10 million users.
    """


@pytest.fixture
def boilerplate_10q_text():
    """10-Q text that is mostly boilerplate."""
    return """
    There have been no material changes to the risk factors disclosed in our Annual Report on Form 10-K.
    As previously reported in our 10-K, we face various competitive and regulatory risks.
    The risk factors set forth in our annual report remain unchanged from prior disclosures.
    Except as previously disclosed, there are no new risks to report this quarter.
    As described in our annual report, we continue to face cybersecurity challenges.
    The discussion below should be read in conjunction with our Annual Report.
    No material changes have been made to our risk disclosures since year-end.
    As included in our Form 10-K, the risk factors remain substantially the same.
    Incorporated herein by reference are the risk factors in our 10-K filing.
    Updated only where necessary to reflect current period information.
    """


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Test text normalization for drift scoring."""
    
    def test_normalize_removes_stopwords(self):
        """Verify stopwords are removed from normalized text."""
        text = "The company faces significant risk factors."
        normalized = normalize_text_for_diff(text)
        
        # Should not contain common stopwords
        assert "the" not in normalized.split()
        assert "faces" in normalized.lower()  # Non-stopword kept
    
    def test_normalize_lowercases(self):
        """Verify text is lowercased."""
        text = "RISK FACTORS are IMPORTANT for investors."
        normalized = normalize_text_for_diff(text)
        
        assert normalized == normalized.lower()
    
    def test_normalize_collapses_whitespace(self):
        """Verify whitespace is collapsed."""
        text = "Multiple    spaces\n\nand    newlines\tand tabs."
        normalized = normalize_text_for_diff(text)
        
        assert "  " not in normalized
        assert "\n" not in normalized
        assert "\t" not in normalized
    
    def test_normalize_sorts_sentences(self):
        """Verify sentences are sorted (order-independent comparison)."""
        text1 = "First sentence. Second sentence."
        text2 = "Second sentence. First sentence."
        
        norm1 = normalize_text_for_diff(text1)
        norm2 = normalize_text_for_diff(text2)
        
        # After normalization and sorting, should be identical
        assert norm1 == norm2
    
    def test_normalize_skips_short_sentences(self):
        """Verify short sentences are skipped."""
        text = "Short. This is a much longer sentence with many tokens."
        normalized = normalize_text_for_diff(text, min_sentence_tokens=6)
        
        # "Short" has only 1 token, should be skipped
        assert "short" not in normalized.lower()
    
    def test_pronoun_swap_reduces_drift(self, risk_text_v1, risk_text_v2_minor_change):
        """Verify pronoun swaps don't spike drift with normalization."""
        # With normalization (default)
        drift_normalized = calculate_drift(
            risk_text_v1, 
            risk_text_v2_minor_change,
            use_normalization=True
        )
        
        # Without normalization
        drift_raw = calculate_drift(
            risk_text_v1, 
            risk_text_v2_minor_change,
            use_normalization=False
        )
        
        # Normalized drift should be LOWER than raw drift
        # Because pronoun swaps are noise, not signal
        assert drift_normalized.drift_score < drift_raw.drift_score
    
    def test_get_normalized_tokens(self):
        """Verify normalized token extraction works."""
        text = "The company faces significant risks from competition."
        tokens = get_normalized_tokens(text)
        
        assert isinstance(tokens, set)
        assert len(tokens) > 0
        # Common stopwords should be gone
        assert "the" not in tokens


# =============================================================================
# Drift Calculation Tests
# =============================================================================


class TestDriftCalculation:
    """Test drift scoring and stability."""
    
    def test_drift_is_deterministic(self, risk_text_v1, risk_text_v3_major_change):
        """Verify same inputs produce same outputs."""
        result1 = calculate_drift(risk_text_v1, risk_text_v3_major_change)
        result2 = calculate_drift(risk_text_v1, risk_text_v3_major_change)
        
        assert result1.drift_score == result2.drift_score
        assert result1.similarity == result2.similarity
        assert result1.added_sentences == result2.added_sentences
        assert result1.removed_sentences == result2.removed_sentences
    
    def test_drift_captures_major_changes(self, risk_text_v1, risk_text_v3_major_change):
        """Verify major changes are detected with high drift."""
        result = calculate_drift(risk_text_v1, risk_text_v3_major_change)
        
        # Major changes should have higher drift
        assert result.drift_score > 15  # Substantive change threshold
        
        # Should capture new risk sentences
        assert len(result.added_sentences) > 0
        
        # New risk keywords should be detected
        assert len(result.new_risk_keywords) > 0
    
    def test_drift_minimal_for_same_text(self, risk_text_v1):
        """Verify identical texts have zero drift."""
        result = calculate_drift(risk_text_v1, risk_text_v1)
        
        assert result.drift_score < 1.0  # Essentially zero
        assert result.similarity > 0.99
        assert len(result.added_sentences) == 0
        assert len(result.removed_sentences) == 0
    
    def test_drift_severity_classification(self):
        """Verify severity is correctly classified."""
        # Low drift
        result_low = calculate_drift(
            "Minor text about risks.",
            "Minor text about risks and opportunities."
        )
        
        # The actual severity depends on thresholds, but let's verify it's set
        assert result_low.severity in ("low", "moderate", "critical")
    
    def test_sentence_diff_shows_actual_text(self, risk_text_v1, risk_text_v3_major_change):
        """Verify sentence diff shows original (not normalized) text."""
        result = calculate_drift(risk_text_v1, risk_text_v3_major_change)
        
        # Added sentences should be actual text from v3
        for sentence in result.added_sentences:
            assert sentence in risk_text_v3_major_change


# =============================================================================
# Boilerplate Detection Tests
# =============================================================================


class TestBoilerplateDetection:
    """Test boilerplate detection for 10-Q noise suppression."""
    
    def test_detects_boilerplate_heavy_text(self, boilerplate_10q_text):
        """Verify boilerplate-heavy text is flagged."""
        result = detect_boilerplate(boilerplate_10q_text)
        
        assert result.is_boilerplate_heavy is True
        assert result.boilerplate_ratio > 0.6
        assert len(result.matched_patterns) > 0
    
    def test_boilerplate_low_for_substantive_text(self, risk_text_v3_major_change):
        """Verify substantive text has low boilerplate ratio."""
        result = detect_boilerplate(risk_text_v3_major_change)
        
        assert result.is_boilerplate_heavy is False
        assert result.boilerplate_ratio < 0.3
    
    def test_boilerplate_patterns_match_expected(self):
        """Verify specific boilerplate patterns are detected."""
        test_cases = [
            "No material changes have occurred since our annual report.",
            "As previously disclosed in our 10-K filing.",
            "Incorporated herein by reference to our annual report.",
            "There have been no material changes to risk factors.",
            "As described in our annual report on Form 10-K.",
        ]
        
        for text in test_cases:
            result = detect_boilerplate(text)
            # At least one sentence should match
            assert result.boilerplate_sentence_count >= 1, f"Failed on: {text}"


# =============================================================================
# Multi-Period Analysis Tests
# =============================================================================


class TestMultiPeriodAnalysis:
    """Test analyze_risk_drift across multiple periods."""
    
    def test_analyze_returns_signal_alert(self, risk_text_v1, risk_text_v3_major_change):
        """Verify analyze_risk_drift returns proper SignalAlert."""
        period_texts = {
            "Q1 FY2025": risk_text_v1,
            "Q2 FY2025": risk_text_v3_major_change,
        }
        
        alert = analyze_risk_drift("NVDA", period_texts)
        
        assert isinstance(alert, SignalAlert)
        assert alert.ticker == "NVDA"
        assert alert.signal_type == "risk_drift"
        assert len(alert.drift_results) > 0
    
    def test_analyze_compares_consecutive_periods(self):
        """Verify periods are compared in order."""
        period_texts = {
            "Q1 FY2025": "Risk text for Q1.",
            "Q2 FY2025": "Risk text for Q2 with new risks.",
            "Q3 FY2025": "Risk text for Q3 with even more risks.",
        }
        
        alert = analyze_risk_drift("AAPL", period_texts)
        
        # Should have comparisons between consecutive periods
        assert len(alert.drift_results) == 2  # Q1->Q2 and Q2->Q3
    
    def test_analyze_with_single_period(self, risk_text_v1):
        """Verify graceful handling of single period."""
        period_texts = {"Q1 FY2025": risk_text_v1}
        
        alert = analyze_risk_drift("MSFT", period_texts)
        
        # Should return alert with no drift results
        assert alert.ticker == "MSFT"
        assert len(alert.drift_results) == 0


# =============================================================================
# Fixture File Tests
# =============================================================================


class TestFixtureExtraction:
    """Test extraction from fixture files."""
    
    def test_10k_toc_extraction(self, html_10k_with_toc):
        """Verify TOC anchor extraction from 10-K fixture."""
        result = extract_item_1a(html_10k_with_toc)
        
        assert result is not None
        assert result.method == "toc_anchor"
        assert result.confidence == "HIGH"
        assert result.char_count >= MIN_CHARS
        
        # Should contain risk factor content
        assert "competition" in result.text.lower()
        assert "cybersecurity" in result.text.lower()
    
    def test_10k_no_toc_extraction(self, html_10k_no_toc):
        """Verify DOM scan extraction from 10-K without TOC."""
        result = extract_item_1a(html_10k_no_toc)
        
        assert result is not None
        assert result.method in ("dom_scan", "toc_anchor", "fuzzy")
        assert result.char_count >= MIN_CHARS
        
        # Should contain risk factor content
        assert "fraud" in result.text.lower() or "payment" in result.text.lower()
    
    def test_10q_risk_factors_extraction(self, html_10q_risk_factors):
        """Verify extraction from 10-Q with only 'Risk Factors' heading."""
        result = extract_item_1a(html_10q_risk_factors)
        
        assert result is not None
        # May be toc_anchor, dom_scan, or fuzzy depending on structure
        assert result.method in ("toc_anchor", "dom_scan", "fuzzy")
        
        # Should contain risk content
        assert "ai" in result.text.lower() or "artificial" in result.text.lower() or "cloud" in result.text.lower()
    
    def test_10q_collision_extraction(self, html_10q_collision):
        """Verify correct Item 1A extraction from 10-Q with Part I/II collision."""
        result = extract_item_1a(html_10q_collision)
        
        assert result is not None
        
        # Should get Item 1A Risk Factors, NOT Item 1 Legal Proceedings
        assert "geopolitical" in result.text.lower() or "tariff" in result.text.lower()
        
        # Should NOT contain Part II Item 1 content
        assert "legal proceedings" not in result.text.lower()


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestEndToEndPipeline:
    """Test complete extraction + drift pipeline."""
    
    def test_extract_and_drift_from_fixtures(self, html_10k_with_toc, html_10k_no_toc):
        """Test extraction and drift calculation across two fixtures."""
        # Extract from both fixtures
        rf1 = extract_item_1a(html_10k_with_toc)
        rf2 = extract_item_1a(html_10k_no_toc)
        
        assert rf1 is not None
        assert rf2 is not None
        
        # Calculate drift between them
        drift = calculate_drift(
            rf1.text,
            rf2.text,
            period_from="10-K v1",
            period_to="10-K v2",
        )
        
        # Should detect differences (different companies/risks)
        assert drift.drift_score > 0
        
        # Should have sentence-level changes
        assert len(drift.added_sentences) > 0 or len(drift.removed_sentences) > 0
    
    def test_pipeline_with_boilerplate_check(self, html_10q_collision):
        """Test extraction with boilerplate detection."""
        rf = extract_item_1a(html_10q_collision)
        assert rf is not None
        
        # Check for boilerplate
        bp_result = detect_boilerplate(rf.text)
        
        # This fixture has substantive updates, should not be boilerplate-heavy
        assert bp_result.boilerplate_ratio < 0.5


# =============================================================================
# Signal Mode Tests
# =============================================================================


class TestSignalMode:
    """Test signal mode enum and behavior."""
    
    def test_signal_mode_values(self):
        """Verify SignalMode enum values."""
        assert SignalMode.REGIME.value == "regime"
        assert SignalMode.EVENT.value == "event"
        assert SignalMode.QUARTERLY.value == "quarterly"
    
    def test_all_modes_defined(self):
        """Verify all expected modes exist."""
        modes = list(SignalMode)
        assert len(modes) == 3
        
        mode_values = {m.value for m in modes}
        assert "regime" in mode_values
        assert "event" in mode_values
        assert "quarterly" in mode_values

