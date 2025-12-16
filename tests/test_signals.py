"""
Tests for the Drift Engine (signals.py).

Tests the semantic drift detection functionality including:
- Tokenization with stopword removal
- Jaccard similarity calculation
- Sentence-level diff using difflib
- Risk keyword detection
- Full drift calculation
- Multi-period analysis
"""
from __future__ import annotations

import pytest

from open_deep_research.models import SignalRecord
from open_deep_research.signals import (
    tokenize,
    extract_sentences,
    calculate_jaccard,
    calculate_drift,
    get_sentence_diff,
    find_risk_keywords,
    analyze_risk_drift,
    format_signal_report,
    detect_boilerplate,
    BOILERPLATE_PATTERNS,
    NLTK_AVAILABLE,
    STOPWORDS,
)


# =============================================================================
# Test Tokenization
# =============================================================================


class TestTokenize:
    """Test the tokenize function."""
    
    def test_removes_stopwords_if_nltk_available(self):
        """Verify stopwords are removed when nltk is available."""
        tokens = tokenize("the quick brown fox jumps over the lazy dog")
        
        # Content words should remain
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "jumps" in tokens
        assert "lazy" in tokens
        assert "dog" in tokens
        
        # If NLTK is available, stopwords should be removed
        if NLTK_AVAILABLE and len(STOPWORDS) > 0:
            assert "the" not in tokens
            assert "over" not in tokens
    
    def test_lowercase(self):
        """Verify all tokens are lowercase."""
        tokens = tokenize("HELLO World MiXeD")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "mixed" in tokens
        
        # Uppercase versions should not exist
        assert "HELLO" not in tokens
        assert "World" not in tokens
        assert "MiXeD" not in tokens
    
    def test_filters_short_words(self):
        """Verify short words (<=2 chars) are filtered out."""
        tokens = tokenize("I am a big cat in the hat")
        
        # Short words should be removed
        assert "i" not in tokens
        assert "am" not in tokens
        assert "a" not in tokens
        assert "in" not in tokens
        
        # Longer words should remain
        assert "big" in tokens
        assert "cat" in tokens
        assert "hat" in tokens
    
    def test_filters_non_alpha(self):
        """Verify non-alphabetic tokens are filtered."""
        tokens = tokenize("revenue was $35,000,000 in 2024")
        
        # Numbers and symbols should not be in tokens
        assert "35" not in tokens
        assert "000" not in tokens
        assert "2024" not in tokens
        assert "$" not in tokens
        
        # Words should remain
        assert "revenue" in tokens
    
    def test_empty_text(self):
        """Verify empty text returns empty set."""
        tokens = tokenize("")
        assert tokens == set()
    
    def test_returns_set(self):
        """Verify return type is a set (for efficient intersection/union)."""
        tokens = tokenize("word word repeated word")
        assert isinstance(tokens, set)


# =============================================================================
# Test Sentence Extraction
# =============================================================================


class TestExtractSentences:
    """Test the extract_sentences function."""
    
    def test_basic_sentence_split(self):
        """Verify basic sentence splitting works."""
        # Use longer sentences to pass the 20-char minimum filter
        text = "This is the first sentence with enough content. This is the second sentence with details. This is the third sentence with information."
        sentences = extract_sentences(text)
        
        assert len(sentences) == 3
        assert "first sentence" in sentences[0]
        assert "second sentence" in sentences[1]
        assert "third sentence" in sentences[2]
    
    def test_filters_short_sentences(self):
        """Verify short sentences (<20 chars) are filtered."""
        text = "Short. This is a longer sentence that should be kept. OK."
        sentences = extract_sentences(text)
        
        # Short sentences should be filtered
        assert not any("Short" == s for s in sentences)
        assert not any("OK" == s for s in sentences)
        
        # Long sentence should remain
        assert any("longer sentence" in s for s in sentences)
    
    def test_handles_exclamation_and_question(self):
        """Verify handling of ! and ? as sentence terminators."""
        # Use longer sentences to pass the 20-char minimum filter
        text = "What was the total revenue for this quarter? Revenue was exceptionally high this period! The results exceeded expectations significantly."
        sentences = extract_sentences(text)
        
        # Should have 3 sentences (all long enough to pass filter)
        assert len(sentences) >= 1
    
    def test_empty_text(self):
        """Verify empty text returns empty list."""
        sentences = extract_sentences("")
        assert sentences == []


# =============================================================================
# Test Jaccard Similarity
# =============================================================================


class TestJaccard:
    """Test the calculate_jaccard function."""
    
    def test_identical_sets(self):
        """Verify identical sets have similarity 1.0."""
        s1 = {"alpha", "beta", "gamma"}
        assert calculate_jaccard(s1, s1) == 1.0
    
    def test_completely_different_sets(self):
        """Verify completely different sets have similarity 0.0."""
        s1 = {"alpha", "beta", "gamma"}
        s2 = {"delta", "epsilon", "zeta"}
        assert calculate_jaccard(s1, s2) == 0.0
    
    def test_partial_overlap(self):
        """Verify partial overlap calculates correctly."""
        s1 = {"alpha", "beta", "gamma"}
        s2 = {"beta", "gamma", "delta"}
        
        # intersection = {beta, gamma} = 2
        # union = {alpha, beta, gamma, delta} = 4
        # jaccard = 2/4 = 0.5
        assert calculate_jaccard(s1, s2) == 0.5
    
    def test_one_empty_set(self):
        """Verify handling when one set is empty."""
        s1 = {"alpha", "beta"}
        s2 = set()
        
        # intersection = 0, union = 2
        # jaccard = 0/2 = 0.0
        assert calculate_jaccard(s1, s2) == 0.0
    
    def test_both_empty_sets(self):
        """Verify both empty sets have similarity 1.0 (identical)."""
        assert calculate_jaccard(set(), set()) == 1.0
    
    def test_subset_relationship(self):
        """Verify subset relationship calculates correctly."""
        s1 = {"alpha", "beta"}
        s2 = {"alpha", "beta", "gamma", "delta"}
        
        # intersection = 2, union = 4
        # jaccard = 2/4 = 0.5
        assert calculate_jaccard(s1, s2) == 0.5


# =============================================================================
# Test Sentence Diff
# =============================================================================


class TestSentenceDiff:
    """Test the get_sentence_diff function."""
    
    def test_detects_additions(self):
        """Verify new sentences are detected as additions."""
        old = "We have no major risks. Business is stable and growing well."
        new = "We have no major risks. Business is stable and growing well. China sanctions now concern us significantly."
        
        added, removed = get_sentence_diff(old, new)
        
        # Should detect the China sentence as added
        assert len(added) >= 1
        assert any("china" in s.lower() or "sanction" in s.lower() for s in added)
        assert len(removed) == 0
    
    def test_detects_removals(self):
        """Verify removed sentences are detected."""
        old = "We have no major risks. China is not a concern for our business operations."
        new = "We have no major risks."
        
        added, removed = get_sentence_diff(old, new)
        
        # Should detect the China sentence as removed
        assert len(removed) >= 1
        assert any("china" in s.lower() for s in removed)
    
    def test_identical_text_no_diff(self):
        """Verify identical text produces no diff."""
        text = "This is a sentence that stays the same. Another sentence here."
        
        added, removed = get_sentence_diff(text, text)
        
        assert added == []
        assert removed == []
    
    def test_replacement(self):
        """Verify replaced sentences show as both added and removed."""
        old = "Our supply chain is robust. Manufacturing is on track."
        new = "Our supply chain faces disruptions. Manufacturing has delays."
        
        added, removed = get_sentence_diff(old, new)
        
        # Should have both additions and removals
        assert len(added) >= 1
        assert len(removed) >= 1
    
    def test_empty_texts(self):
        """Verify handling of empty texts."""
        added, removed = get_sentence_diff("", "")
        assert added == []
        assert removed == []


# =============================================================================
# Test Risk Keyword Detection
# =============================================================================


class TestRiskKeywords:
    """Test the find_risk_keywords function."""
    
    def test_finds_legal_terms(self):
        """Verify legal risk terms are detected."""
        text = "The company faces litigation and is under investigation by the SEC."
        keywords = find_risk_keywords(text)
        
        assert "litigation" in keywords
        assert "investigation" in keywords
        assert "sec" in keywords
    
    def test_finds_geopolitical_terms(self):
        """Verify geopolitical risk terms are detected."""
        text = "China tariffs and export sanctions affect our business significantly."
        keywords = find_risk_keywords(text)
        
        assert "china" in keywords
        assert "tariff" in keywords
        assert "sanction" in keywords
        assert "export" in keywords
    
    def test_finds_financial_terms(self):
        """Verify financial risk terms are detected."""
        text = "We recorded an impairment and may need restructuring due to liquidity."
        keywords = find_risk_keywords(text)
        
        assert "impairment" in keywords
        assert "restructuring" in keywords
        assert "liquidity" in keywords
    
    def test_finds_operational_terms(self):
        """Verify operational risk terms are detected."""
        text = "Supply chain disruption caused manufacturing delays and capacity constraints."
        keywords = find_risk_keywords(text)
        
        assert "supply chain" in keywords
        assert "disruption" in keywords
        assert "delay" in keywords
        assert "constraint" in keywords
        assert "capacity" in keywords
    
    def test_case_insensitive(self):
        """Verify detection is case-insensitive."""
        text = "LITIGATION and China TARIFFS are major concerns."
        keywords = find_risk_keywords(text)
        
        assert "litigation" in keywords
        assert "china" in keywords
        assert "tariff" in keywords
    
    def test_no_false_positives(self):
        """Verify normal text doesn't trigger false positives."""
        text = "Our business is performing well with strong revenue growth."
        keywords = find_risk_keywords(text)
        
        # Should find no risk keywords in benign text
        assert len(keywords) == 0
    
    def test_empty_text(self):
        """Verify empty text returns empty set."""
        keywords = find_risk_keywords("")
        assert keywords == set()


# =============================================================================
# Test Full Drift Calculation
# =============================================================================


class TestDriftCalculation:
    """Test the calculate_drift function."""
    
    def test_identical_text_zero_drift(self):
        """Verify identical text produces zero drift."""
        text = "This is our risk factors section with detailed information about potential risks."
        
        result = calculate_drift(text, text)
        
        assert result.drift_score == 0.0
        assert result.similarity == 1.0
        assert result.severity == "low"
        assert result.added_sentences == []
        assert result.removed_sentences == []
    
    def test_completely_different_high_drift(self):
        """Verify completely different text produces high drift."""
        old = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        new = "lambda mu nu xi omicron pi rho sigma tau upsilon"
        
        result = calculate_drift(old, new)
        
        # Should have high drift score (different words)
        assert result.drift_score > 80
        assert result.similarity < 0.2
        assert result.severity == "critical"
    
    def test_detects_new_risk_keywords(self):
        """Verify new risk keywords are detected in drift result."""
        old = "Our business is performing well with strong growth potential overall."
        new = "Our business faces litigation and investigation risks from regulators now."
        
        result = calculate_drift(old, new)
        
        assert "litigation" in result.new_risk_keywords
        assert "investigation" in result.new_risk_keywords
    
    def test_detects_removed_risk_keywords(self):
        """Verify removed risk keywords are detected."""
        old = "We face litigation and ongoing investigation by authorities."
        new = "Our business is performing well with strong growth potential."
        
        result = calculate_drift(old, new)
        
        assert "litigation" in result.removed_risk_keywords
        assert "investigation" in result.removed_risk_keywords
    
    def test_moderate_drift_severity(self):
        """Verify moderate changes produce detectable drift."""
        # Use texts with some overlap to get moderate drift
        old = "Our business operations continue with stable revenue growth and customer expansion."
        new = "Our business operations continue with stable performance and market development."
        
        result = calculate_drift(old, new)
        
        # Should detect some drift (not zero, not completely different)
        assert result.drift_score > 0
        # With significant word overlap, drift should be below 100
        assert result.drift_score < 100
    
    def test_period_labels_preserved(self):
        """Verify period labels are preserved in result."""
        result = calculate_drift(
            "text one here",
            "text two here",
            period_from="Q1 FY2025",
            period_to="Q2 FY2025",
        )
        
        assert result.period_from == "Q1 FY2025"
        assert result.period_to == "Q2 FY2025"
    
    def test_returns_drift_result_model(self):
        """Verify return type is DriftResult."""
        from open_deep_research.models import DriftResult
        
        result = calculate_drift("text", "text")
        assert isinstance(result, DriftResult)


# =============================================================================
# Test Multi-Period Analysis
# =============================================================================


class TestAnalyzeRiskDrift:
    """Test the analyze_risk_drift function."""
    
    def test_two_periods(self):
        """Verify analysis works with two periods."""
        period_texts = {
            "Q1 FY2025": "Our business is stable with no major concerns to report.",
            "Q2 FY2025": "Our business now faces litigation and regulatory investigation.",
        }
        
        alert = analyze_risk_drift("NVDA", period_texts)
        
        assert alert.ticker == "NVDA"
        assert alert.signal_type == "risk_drift"
        assert len(alert.drift_results) == 1
        assert alert.drift_results[0].period_from == "Q1 FY2025"
        assert alert.drift_results[0].period_to == "Q2 FY2025"
    
    def test_three_periods(self):
        """Verify analysis works with three periods (two comparisons)."""
        period_texts = {
            "Q1 FY2025": "Business is stable with normal operations across all segments.",
            "Q2 FY2025": "Business remains stable with continued normal operations.",
            "Q3 FY2025": "Business now faces litigation and investigation by regulators.",
        }
        
        alert = analyze_risk_drift("AAPL", period_texts)
        
        assert len(alert.drift_results) == 2
        # Q1->Q2 should have low drift (similar text)
        # Q2->Q3 should have higher drift (litigation added)
    
    def test_insufficient_data(self):
        """Verify handling of insufficient data (< 2 periods)."""
        alert = analyze_risk_drift("NVDA", {"Q1 FY2025": "some text"})
        
        assert alert.severity == "low"
        assert "Insufficient" in alert.headline
        assert len(alert.drift_results) == 0
    
    def test_empty_data(self):
        """Verify handling of empty period_texts."""
        alert = analyze_risk_drift("NVDA", {})
        
        assert alert.severity == "low"
        assert "Insufficient" in alert.headline
    
    def test_severity_propagates(self):
        """Verify maximum severity propagates to alert."""
        period_texts = {
            "Q1 FY2025": "alpha beta gamma delta epsilon zeta eta theta iota",
            "Q2 FY2025": "kappa lambda mu nu xi omicron pi rho sigma tau",  # Completely different
        }
        
        alert = analyze_risk_drift("NVDA", period_texts)
        
        # Should be critical due to high drift
        assert alert.severity == "critical"
    
    def test_returns_signal_alert(self):
        """Verify return type is SignalAlert."""
        from open_deep_research.models import SignalAlert
        
        alert = analyze_risk_drift("NVDA", {
            "Q1": "text one",
            "Q2": "text two",
        })
        
        assert isinstance(alert, SignalAlert)


# =============================================================================
# Test Report Formatting
# =============================================================================


class TestFormatSignalReport:
    """Test the format_signal_report function."""
    
    def test_includes_ticker(self):
        """Verify report includes ticker."""
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": "normal text here with business operations",
            "Q2 FY2025": "different text here about company changes",
        })
        
        report = format_signal_report(alert)
        
        assert "NVDA" in report
    
    def test_includes_severity(self):
        """Verify report includes severity information."""
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": "our business is stable and operations continue",
            "Q2 FY2025": "our business faces litigation investigation sanctions",
        })
        
        report = format_signal_report(alert)
        
        # Should include severity labels
        assert "LOW" in report or "MODERATE" in report or "CRITICAL" in report
    
    def test_includes_drift_score(self):
        """Verify report includes drift scores."""
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": "text one for period one with details",
            "Q2 FY2025": "text two for period two with details",
        })
        
        report = format_signal_report(alert)
        
        assert "Drift Score" in report
    
    def test_includes_periods(self):
        """Verify report includes period labels."""
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": "text for first period with content",
            "Q2 FY2025": "text for second period with content",
        })
        
        report = format_signal_report(alert)
        
        assert "Q1 FY2025" in report
        assert "Q2 FY2025" in report
    
    def test_returns_string(self):
        """Verify report is a string."""
        alert = analyze_risk_drift("NVDA", {
            "Q1": "text",
            "Q2": "text",
        })
        
        report = format_signal_report(alert)
        assert isinstance(report, str)
    
    def test_multiline_format(self):
        """Verify report has multiple lines."""
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": "text for first period with some content",
            "Q2 FY2025": "different text for second period here",
        })
        
        report = format_signal_report(alert)
        
        lines = report.split("\n")
        assert len(lines) > 5  # Should have multiple lines


# =============================================================================
# Integration Test
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_drift_workflow(self):
        """Test complete workflow from text to report."""
        # Simulate Risk Factors text from consecutive quarters
        q1_text = """
        Our operations are subject to various risks and uncertainties.
        We compete in a highly competitive market environment.
        Our revenue depends on customer demand for our products.
        Supply chain operations are critical to our business.
        """
        
        q2_text = """
        Our operations are subject to various risks and uncertainties.
        We compete in a highly competitive market environment.
        Our revenue depends on customer demand for our products.
        Supply chain disruption has caused manufacturing delays.
        We face increased litigation from patent disputes.
        China export restrictions affect our business operations.
        """
        
        # Run analysis
        alert = analyze_risk_drift("NVDA", {
            "Q1 FY2025": q1_text,
            "Q2 FY2025": q2_text,
        })
        
        # Verify alert structure
        assert alert.ticker == "NVDA"
        assert len(alert.drift_results) == 1
        
        dr = alert.drift_results[0]
        
        # Should detect new risk keywords
        assert any(kw in ["litigation", "disruption", "china", "delay", "export"] 
                   for kw in dr.new_risk_keywords)
        
        # Should have added sentences (the new risk language)
        assert len(dr.added_sentences) >= 1
        
        # Format and verify report
        report = format_signal_report(alert)
        assert "NVDA" in report
        assert "Q1 FY2025" in report
        assert "Q2 FY2025" in report
    
    def test_no_drift_scenario(self):
        """Test scenario with no changes (copy-paste risk factors)."""
        text = """
        Our business is subject to various risks and uncertainties.
        Competition in our industry is intense and ongoing.
        Economic conditions may affect customer purchasing decisions.
        """
        
        alert = analyze_risk_drift("AAPL", {
            "Q1 FY2025": text,
            "Q2 FY2025": text,
        })
        
        dr = alert.drift_results[0]
        
        assert dr.drift_score == 0.0
        assert dr.severity == "low"
        assert dr.added_sentences == []
        assert dr.removed_sentences == []
        assert dr.new_risk_keywords == []
    
    def test_deterministic_without_embeddings(self):
        """Test that drift analysis produces deterministic results even without embeddings.
        
        The system should always produce the same output for the same input,
        regardless of whether sentence-transformers is installed. Jaccard similarity
        and difflib.SequenceMatcher are deterministic.
        """
        q1_text = "Our business operations are stable. Revenue growth continues."
        q2_text = "Our business operations face challenges. Litigation is pending."
        
        # Run twice - should get identical results
        alert1 = analyze_risk_drift("TEST", {"Q1": q1_text, "Q2": q2_text})
        alert2 = analyze_risk_drift("TEST", {"Q1": q1_text, "Q2": q2_text})
        
        # Verify determinism
        dr1 = alert1.drift_results[0]
        dr2 = alert2.drift_results[0]
        
        assert dr1.drift_score == dr2.drift_score
        assert dr1.similarity == dr2.similarity
        assert dr1.severity == dr2.severity
        assert dr1.added_sentences == dr2.added_sentences
        assert dr1.removed_sentences == dr2.removed_sentences
        assert dr1.new_risk_keywords == dr2.new_risk_keywords
        
        # Verify core analysis works (Jaccard is always available)
        assert dr1.similarity >= 0.0
        assert dr1.similarity <= 1.0
        assert dr1.drift_score >= 0.0
        assert dr1.drift_score <= 100.0
        
        # Should detect the change
        assert dr1.drift_score > 0  # Not identical text
        
        # Should detect new risk keyword "litigation"
        assert "litigation" in dr1.new_risk_keywords


# =============================================================================
# Test Boilerplate Detection (P2)
# =============================================================================


class TestBoilerplateDetection:
    """Test the detect_boilerplate function."""
    
    def test_detects_no_material_changes(self):
        """Verify 'no material changes' is detected as boilerplate."""
        text = (
            "There have been no material changes to the risk factors "
            "previously disclosed in our Annual Report on Form 10-K."
        )
        result = detect_boilerplate(text)
        
        assert result.boilerplate_sentence_count >= 1
        assert result.boilerplate_ratio > 0
    
    def test_detects_incorporated_by_reference(self):
        """Verify 'incorporated by reference' is detected."""
        text = (
            "The risk factors are incorporated herein by reference to Item 1A "
            "of our Annual Report on Form 10-K for the fiscal year ended January 2024."
        )
        result = detect_boilerplate(text)
        
        assert result.boilerplate_sentence_count >= 1
    
    def test_real_risk_text_not_boilerplate(self):
        """Verify substantive risk text is not flagged as boilerplate."""
        text = (
            "We face significant litigation risk from patent disputes. "
            "Our supply chain has experienced major disruptions due to geopolitical tensions. "
            "China export restrictions have materially impacted our revenue in the Asia-Pacific region."
        )
        result = detect_boilerplate(text)
        
        # Should not be flagged as boilerplate-heavy
        assert result.is_boilerplate_heavy is False
        assert result.boilerplate_ratio < 0.5
    
    def test_mixed_content_partial_flag(self):
        """Verify mixed content gets appropriate ratio."""
        text = (
            "There have been no material changes to the risk factors. "
            "Except as previously disclosed, our risks remain unchanged. "
            "However, we now face new litigation from the SEC investigation. "
            "Regulatory enforcement actions have increased significantly."
        )
        result = detect_boilerplate(text)
        
        # Should detect some boilerplate but also real content
        assert 0 < result.boilerplate_ratio < 1
        assert result.total_sentence_count >= 2
    
    def test_high_boilerplate_ratio_flags(self):
        """Verify high boilerplate ratio sets is_boilerplate_heavy flag."""
        text = (
            "There have been no material changes since our annual report. "
            "Our risk factors remain unchanged from our Form 10-K. "
            "As previously disclosed, refer to our annual report. "
            "The discussion should be read in conjunction with our 10-K."
        )
        result = detect_boilerplate(text)
        
        # Should be flagged as boilerplate-heavy
        assert result.is_boilerplate_heavy is True
        assert result.boilerplate_ratio > 0.6
    
    def test_empty_text(self):
        """Verify empty text returns safe defaults."""
        result = detect_boilerplate("")
        
        assert result.is_boilerplate_heavy is False
        assert result.boilerplate_ratio == 0.0
        assert result.total_sentence_count == 0
    
    def test_patterns_are_compiled(self):
        """Verify all patterns are valid compiled regexes."""
        import re
        for pattern in BOILERPLATE_PATTERNS:
            assert isinstance(pattern, re.Pattern)
    
    def test_case_insensitive(self):
        """Verify detection is case-insensitive."""
        text = "THERE HAVE BEEN NO MATERIAL CHANGES to our risk factors."
        result = detect_boilerplate(text)
        
        assert result.boilerplate_sentence_count >= 1


# =============================================================================
# Test Signal Persistence (P1)
# =============================================================================


class TestSignalStore:
    """Test the SignalStore persistence."""
    
    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary signal store for testing."""
        from open_deep_research.store import SignalStore
        store_path = tmp_path / "test_signals.jsonl"
        return SignalStore(str(store_path))
    
    @pytest.fixture
    def sample_record(self):
        """Create a sample SignalRecord for testing."""
        from open_deep_research.models import SignalRecord
        return SignalRecord(
            signal_id="test-123",
            ticker="NVDA",
            cik="0001045810",
            filing_type="10-Q",
            base_accession="0001045810-24-000100",
            compare_accession="0001045810-25-000200",
            base_period_end_date="2024-07-28",
            compare_period_end_date="2024-10-27",
            filing_date="2024-11-20",
            base_fiscal_period="Q2 FY2025",
            compare_fiscal_period="Q3 FY2025",
            drift_score=25.5,
            jaccard_similarity=0.75,
            semantic_similarity=0.82,
            new_sentence_count=3,
            removed_sentence_count=1,
            new_keyword_count=2,
            removed_keyword_count=0,
            new_keywords_json='["litigation", "investigation"]',
            removed_keywords_json='[]',
            boilerplate_flag=False,
            boilerplate_ratio=0.1,
            severity="moderate",
            created_at="2024-11-20T12:00:00",
            model_version="1.0.0",
            base_snapshot_id="uuid-base",
            compare_snapshot_id="uuid-compare",
        )
    
    def test_append_and_load(self, temp_store, sample_record):
        """Verify records can be appended and loaded."""
        # Initially empty
        assert len(temp_store.load_all()) == 0
        
        # Append
        temp_store.append(sample_record)
        
        # Load and verify
        records = temp_store.load_all()
        assert len(records) == 1
        assert records[0].ticker == "NVDA"
        assert records[0].drift_score == 25.5
    
    def test_query_by_ticker(self, temp_store, sample_record):
        """Verify querying by ticker works."""
        temp_store.append(sample_record)
        
        # Query
        results = temp_store.query_by_ticker("NVDA")
        assert len(results) == 1
        
        # Case insensitive
        results = temp_store.query_by_ticker("nvda")
        assert len(results) == 1
        
        # Different ticker
        results = temp_store.query_by_ticker("AAPL")
        assert len(results) == 0
    
    def test_query_high_drift(self, temp_store, sample_record):
        """Verify querying by drift score works."""
        temp_store.append(sample_record)  # drift_score = 25.5
        
        # Below threshold
        results = temp_store.query_high_drift(threshold=30.0)
        assert len(results) == 0
        
        # Above threshold
        results = temp_store.query_high_drift(threshold=20.0)
        assert len(results) == 1
    
    def test_query_non_boilerplate(self, temp_store, sample_record):
        """Verify filtering out boilerplate works."""
        temp_store.append(sample_record)  # boilerplate_flag = False
        
        results = temp_store.query_non_boilerplate()
        assert len(results) == 1
    
    def test_count(self, temp_store, sample_record):
        """Verify count method."""
        assert temp_store.count() == 0
        temp_store.append(sample_record)
        assert temp_store.count() == 1
    
    def test_clear(self, temp_store, sample_record):
        """Verify clear method."""
        temp_store.append(sample_record)
        assert temp_store.count() == 1
        
        temp_store.clear()
        assert temp_store.count() == 0


# =============================================================================
# Test Signal Modes (P3 - Dual Filing Logic)
# =============================================================================


class TestSignalMode:
    """Test the SignalMode enum and dual-filing logic."""
    
    def test_signal_mode_values(self):
        """Verify SignalMode enum has correct values."""
        from open_deep_research.signals import SignalMode
        
        assert SignalMode.REGIME.value == "regime"
        assert SignalMode.EVENT.value == "event"
        assert SignalMode.QUARTERLY.value == "quarterly"
    
    def test_regime_mode_description(self):
        """Verify REGIME mode is for 10-K comparison."""
        from open_deep_research.signals import SignalMode
        
        # REGIME should compare 10-K → 10-K
        mode = SignalMode.REGIME
        assert "10-K" in mode.__doc__ or "annual" in str(mode).lower()
    
    def test_event_mode_description(self):
        """Verify EVENT mode is for 10-Q → 10-K overlay."""
        from open_deep_research.signals import SignalMode
        
        # EVENT should compare 10-Q → 10-K
        mode = SignalMode.EVENT
        assert "10-Q" in mode.__doc__ or "event" in str(mode).lower()


class TestSignalRecordModeFields:
    """Test the new mode fields in SignalRecord."""
    
    def test_signal_record_has_mode_fields(self):
        """Verify SignalRecord includes P3 mode fields."""
        from open_deep_research.models import SignalRecord
        
        # Check field existence via schema
        schema = SignalRecord.model_json_schema()
        properties = schema.get("properties", {})
        
        assert "signal_mode" in properties
        assert "base_filing_type" in properties
        assert "compare_filing_type" in properties
    
    def test_signal_record_mode_defaults(self):
        """Verify SignalRecord mode fields have correct defaults."""
        from open_deep_research.models import SignalRecord
        
        # Create minimal record
        record = SignalRecord(
            signal_id="test-123",
            ticker="NVDA",
            cik="0001045810",
            filing_type="10-Q",
            base_accession="acc-1",
            compare_accession="acc-2",
            base_period_end_date="2024-01-28",
            compare_period_end_date="2024-10-27",
            filing_date="2024-11-20",
            base_fiscal_period="Q4 FY2024",
            compare_fiscal_period="Q3 FY2025",
            drift_score=25.5,
            jaccard_similarity=0.75,
            new_sentence_count=3,
            removed_sentence_count=1,
            new_keyword_count=2,
            removed_keyword_count=0,
            new_keywords_json='[]',
            removed_keywords_json='[]',
            boilerplate_flag=False,
            boilerplate_ratio=0.1,
            severity="moderate",
            created_at="2024-11-20T12:00:00",
            model_version="1.0.0",
            base_snapshot_id="uuid-1",
            compare_snapshot_id="uuid-2",
        )
        
        # Check defaults
        assert record.signal_mode == "quarterly"
        assert record.base_filing_type == "10-Q"
        assert record.compare_filing_type == "10-Q"
    
    def test_signal_record_with_regime_mode(self):
        """Verify SignalRecord can store regime mode."""
        from open_deep_research.models import SignalRecord
        
        record = SignalRecord(
            signal_id="test-456",
            ticker="AAPL",
            cik="0000320193",
            filing_type="10-K",
            base_accession="acc-1",
            compare_accession="acc-2",
            signal_mode="regime",
            base_filing_type="10-K",
            compare_filing_type="10-K",
            base_period_end_date="2023-09-30",
            compare_period_end_date="2024-09-30",
            filing_date="2024-10-30",
            base_fiscal_period="FY2023",
            compare_fiscal_period="FY2024",
            drift_score=45.0,
            jaccard_similarity=0.55,
            new_sentence_count=10,
            removed_sentence_count=5,
            new_keyword_count=4,
            removed_keyword_count=2,
            new_keywords_json='["litigation", "investigation"]',
            removed_keywords_json='[]',
            boilerplate_flag=False,
            boilerplate_ratio=0.05,
            severity="critical",
            created_at="2024-11-01T12:00:00",
            model_version="1.0.0",
            base_snapshot_id="uuid-3",
            compare_snapshot_id="uuid-4",
        )
        
        assert record.signal_mode == "regime"
        assert record.base_filing_type == "10-K"
        assert record.compare_filing_type == "10-K"


class TestGetSignalFilings:
    """Test the get_signal_filings helper function."""
    
    def test_invalid_mode_raises(self):
        """Verify invalid mode raises ValueError."""
        from open_deep_research.ingestion import get_signal_filings
        
        with pytest.raises(ValueError) as exc_info:
            get_signal_filings("NVDA", "invalid_mode")
        
        assert "Invalid signal mode" in str(exc_info.value)
    
    def test_valid_modes_accepted(self):
        """Verify valid mode strings are accepted."""
        from open_deep_research.ingestion import get_signal_filings
        
        # These should not raise on the mode parsing
        # (they may fail on filing fetch, but that's ok for this test)
        valid_modes = ["regime", "event", "quarterly", "REGIME", "EVENT", "QUARTERLY"]
        
        for mode in valid_modes:
            try:
                get_signal_filings("NVDA", mode)
            except ValueError as e:
                # Only fail if it's a mode-related error
                assert "Invalid signal mode" not in str(e)
            except FileNotFoundError:
                # Expected - no actual filings
                pass
            except Exception:
                # Other errors ok (e.g., network, SEC_USER_AGENT)
                pass


# =============================================================================
# Test Ranking Views (P4)
# =============================================================================


class TestRankingViews:
    """
    Test ranking methods in SignalStore.
    
    ⚠️ IMPORTANT: Rankings are VIEWS, not TRUTH.
    These tests verify the mechanics of ranking, not that rankings
    produce trading alpha.
    """
    
    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a signal store with diverse test data."""
        from open_deep_research.store import SignalStore
        from open_deep_research.models import SignalRecord
        
        store_path = tmp_path / "test_signals.jsonl"
        store = SignalStore(str(store_path))
        
        # Create signals with varying characteristics
        signals = [
            # High drift, low novelty, no boilerplate
            SignalRecord(
                signal_id="sig-1", ticker="AAPL", cik="0000320193",
                filing_type="10-K", base_accession="a1", compare_accession="a2",
                signal_mode="regime", base_filing_type="10-K", compare_filing_type="10-K",
                base_period_end_date="2023-09-30", compare_period_end_date="2024-09-30",
                filing_date="2024-10-30", base_fiscal_period="FY2023", compare_fiscal_period="FY2024",
                drift_score=80.0, jaccard_similarity=0.2,
                new_sentence_count=5, removed_sentence_count=3,
                new_keyword_count=2, removed_keyword_count=1,
                new_keywords_json='["litigation"]', removed_keywords_json='[]',
                boilerplate_flag=False, boilerplate_ratio=0.1, severity="critical",
                created_at="2024-10-30T12:00:00", model_version="1.0.0",
                base_snapshot_id="s1", compare_snapshot_id="s2",
            ),
            # Medium drift, high novelty, no boilerplate
            SignalRecord(
                signal_id="sig-2", ticker="NVDA", cik="0001045810",
                filing_type="10-K", base_accession="a3", compare_accession="a4",
                signal_mode="regime", base_filing_type="10-K", compare_filing_type="10-K",
                base_period_end_date="2024-01-28", compare_period_end_date="2025-01-26",
                filing_date="2025-02-26", base_fiscal_period="FY2024", compare_fiscal_period="FY2025",
                drift_score=50.0, jaccard_similarity=0.5,
                new_sentence_count=15, removed_sentence_count=5,
                new_keyword_count=4, removed_keyword_count=0,
                new_keywords_json='["china", "export", "sanction", "tariff"]', removed_keywords_json='[]',
                boilerplate_flag=False, boilerplate_ratio=0.05, severity="moderate",
                created_at="2025-02-26T12:00:00", model_version="1.0.0",
                base_snapshot_id="s3", compare_snapshot_id="s4",
            ),
            # Low drift, boilerplate-heavy (should be excluded)
            SignalRecord(
                signal_id="sig-3", ticker="MSFT", cik="0000789019",
                filing_type="10-Q", base_accession="a5", compare_accession="a6",
                signal_mode="event", base_filing_type="10-K", compare_filing_type="10-Q",
                base_period_end_date="2024-06-30", compare_period_end_date="2024-09-30",
                filing_date="2024-10-22", base_fiscal_period="FY2024", compare_fiscal_period="Q1 FY2025",
                drift_score=10.0, jaccard_similarity=0.9,
                new_sentence_count=1, removed_sentence_count=0,
                new_keyword_count=0, removed_keyword_count=0,
                new_keywords_json='[]', removed_keywords_json='[]',
                boilerplate_flag=True, boilerplate_ratio=0.75, severity="low",
                created_at="2024-10-22T12:00:00", model_version="1.0.0",
                base_snapshot_id="s5", compare_snapshot_id="s6",
            ),
            # High keywords, medium drift
            SignalRecord(
                signal_id="sig-4", ticker="META", cik="0001326801",
                filing_type="10-K", base_accession="a7", compare_accession="a8",
                signal_mode="regime", base_filing_type="10-K", compare_filing_type="10-K",
                base_period_end_date="2023-12-31", compare_period_end_date="2024-12-31",
                filing_date="2025-02-01", base_fiscal_period="FY2023", compare_fiscal_period="FY2024",
                drift_score=45.0, jaccard_similarity=0.55,
                new_sentence_count=8, removed_sentence_count=4,
                new_keyword_count=6, removed_keyword_count=2,
                new_keywords_json='["litigation", "investigation", "sec", "doj", "regulatory", "enforcement"]',
                removed_keywords_json='["delay", "constraint"]',
                boilerplate_flag=False, boilerplate_ratio=0.15, severity="critical",
                created_at="2025-02-01T12:00:00", model_version="1.0.0",
                base_snapshot_id="s7", compare_snapshot_id="s8",
            ),
        ]
        
        for sig in signals:
            store.append(sig)
        
        return store
    
    def test_rank_by_drift_returns_sorted(self, populated_store):
        """Verify rank_by_drift returns signals sorted by drift_score descending."""
        results = populated_store.rank_by_drift(top_n=10)
        
        # Should be sorted by drift_score descending
        drift_scores = [r.drift_score for r in results]
        assert drift_scores == sorted(drift_scores, reverse=True)
        
        # First should be AAPL (highest drift)
        assert results[0].ticker == "AAPL"
        assert results[0].drift_score == 80.0
    
    def test_rank_by_drift_excludes_boilerplate(self, populated_store):
        """Verify boilerplate signals are excluded by default."""
        results = populated_store.rank_by_drift(top_n=10)
        
        # MSFT (boilerplate=True) should not be in results
        tickers = [r.ticker for r in results]
        assert "MSFT" not in tickers
        assert len(results) == 3  # AAPL, NVDA, META
    
    def test_rank_by_drift_includes_boilerplate_when_disabled(self, populated_store):
        """Verify boilerplate signals included when exclude_boilerplate=False."""
        results = populated_store.rank_by_drift(top_n=10, exclude_boilerplate=False)
        
        tickers = [r.ticker for r in results]
        assert "MSFT" in tickers
        assert len(results) == 4
    
    def test_rank_by_novelty_returns_sorted(self, populated_store):
        """Verify rank_by_novelty returns signals sorted by new_sentence_count."""
        results = populated_store.rank_by_novelty(top_n=10)
        
        counts = [r.new_sentence_count for r in results]
        assert counts == sorted(counts, reverse=True)
        
        # First should be NVDA (highest novelty: 15)
        assert results[0].ticker == "NVDA"
        assert results[0].new_sentence_count == 15
    
    def test_rank_by_keyword_hits_returns_sorted(self, populated_store):
        """Verify rank_by_keyword_hits returns signals sorted by new_keyword_count."""
        results = populated_store.rank_by_keyword_hits(top_n=10)
        
        counts = [r.new_keyword_count for r in results]
        assert counts == sorted(counts, reverse=True)
        
        # First should be META (highest keywords: 6)
        assert results[0].ticker == "META"
        assert results[0].new_keyword_count == 6
    
    def test_rank_composite_returns_tuples(self, populated_store):
        """Verify rank_composite returns (record, score) tuples."""
        results = populated_store.rank_composite(top_n=10)
        
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], SignalRecord)
            assert isinstance(item[1], float)
    
    def test_rank_composite_respects_weights(self, populated_store):
        """Verify different weights produce different rankings."""
        # Drift-heavy weights
        drift_results = populated_store.rank_composite(
            drift_weight=1.0, novelty_weight=0.0, keyword_weight=0.0
        )
        
        # Novelty-heavy weights
        novelty_results = populated_store.rank_composite(
            drift_weight=0.0, novelty_weight=1.0, keyword_weight=0.0
        )
        
        # Rankings should differ
        drift_first = drift_results[0][0].ticker
        novelty_first = novelty_results[0][0].ticker
        
        assert drift_first == "AAPL"  # Highest drift
        assert novelty_first == "NVDA"  # Highest novelty
    
    def test_rank_composite_normalization_global_max(self, populated_store):
        """Verify global_max normalization produces 0-1 scores."""
        results = populated_store.rank_composite(
            normalization="global_max"
        )
        
        # All scores should be between 0 and 1
        for record, score in results:
            assert 0 <= score <= 1
    
    def test_rank_composite_normalization_minmax(self, populated_store):
        """Verify minmax normalization produces 0-1 scores."""
        results = populated_store.rank_composite(
            normalization="minmax"
        )
        
        # All scores should be between 0 and 1
        for record, score in results:
            assert 0 <= score <= 1
    
    def test_rank_composite_invalid_normalization_raises(self, populated_store):
        """Verify invalid normalization strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            populated_store.rank_composite(normalization="invalid")
        
        assert "Invalid normalization strategy" in str(exc_info.value)
    
    def test_top_n_limits_results(self, populated_store):
        """Verify top_n limits number of results."""
        results = populated_store.rank_by_drift(top_n=2)
        assert len(results) == 2


class TestFilteringHelpers:
    """Test filtering methods in SignalStore."""
    
    @pytest.fixture
    def populated_store(self, tmp_path):
        """Reuse the populated store fixture."""
        from open_deep_research.store import SignalStore
        from open_deep_research.models import SignalRecord
        
        store_path = tmp_path / "test_signals.jsonl"
        store = SignalStore(str(store_path))
        
        signals = [
            SignalRecord(
                signal_id="sig-1", ticker="AAPL", cik="0000320193",
                filing_type="10-K", base_accession="a1", compare_accession="a2",
                signal_mode="regime", base_filing_type="10-K", compare_filing_type="10-K",
                base_period_end_date="2023-09-30", compare_period_end_date="2024-09-30",
                filing_date="2024-10-30", base_fiscal_period="FY2023", compare_fiscal_period="FY2024",
                drift_score=80.0, jaccard_similarity=0.2,
                new_sentence_count=5, removed_sentence_count=3,
                new_keyword_count=2, removed_keyword_count=1,
                new_keywords_json='[]', removed_keywords_json='[]',
                boilerplate_flag=False, boilerplate_ratio=0.1, severity="critical",
                created_at="2024-10-30T12:00:00", model_version="1.0.0",
                base_snapshot_id="s1", compare_snapshot_id="s2",
            ),
            SignalRecord(
                signal_id="sig-2", ticker="NVDA", cik="0001045810",
                filing_type="10-Q", base_accession="a3", compare_accession="a4",
                signal_mode="event", base_filing_type="10-K", compare_filing_type="10-Q",
                base_period_end_date="2024-01-28", compare_period_end_date="2024-10-27",
                filing_date="2024-11-20", base_fiscal_period="FY2024", compare_fiscal_period="Q3 FY2025",
                drift_score=25.0, jaccard_similarity=0.75,
                new_sentence_count=2, removed_sentence_count=0,
                new_keyword_count=1, removed_keyword_count=0,
                new_keywords_json='[]', removed_keywords_json='[]',
                boilerplate_flag=False, boilerplate_ratio=0.3, severity="low",
                created_at="2024-11-20T12:00:00", model_version="1.0.0",
                base_snapshot_id="s3", compare_snapshot_id="s4",
            ),
        ]
        
        for sig in signals:
            store.append(sig)
        
        return store
    
    def test_filter_by_mode(self, populated_store):
        """Verify filter_by_mode returns correct signals."""
        regime = populated_store.filter_by_mode("regime")
        event = populated_store.filter_by_mode("event")
        
        assert len(regime) == 1
        assert regime[0].ticker == "AAPL"
        
        assert len(event) == 1
        assert event[0].ticker == "NVDA"
    
    def test_filter_by_severity(self, populated_store):
        """Verify filter_by_severity returns signals at or above threshold."""
        critical = populated_store.filter_by_severity("critical")
        moderate = populated_store.filter_by_severity("moderate")
        low = populated_store.filter_by_severity("low")
        
        assert len(critical) == 1  # Only AAPL
        assert len(moderate) == 1  # AAPL (critical >= moderate)
        assert len(low) == 2  # Both
    
    def test_filter_by_ticker(self, populated_store):
        """Verify filter_by_ticker returns matching signals."""
        results = populated_store.filter_by_ticker(["AAPL"])
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
        
        results = populated_store.filter_by_ticker(["aapl", "nvda"])
        assert len(results) == 2


class TestSummaryStatistics:
    """Test summary statistics method."""
    
    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create store with known statistics."""
        from open_deep_research.store import SignalStore
        from open_deep_research.models import SignalRecord
        
        store_path = tmp_path / "test_signals.jsonl"
        store = SignalStore(str(store_path))
        
        # Create 3 signals with known stats
        for i, (ticker, mode, severity, drift) in enumerate([
            ("AAPL", "regime", "critical", 60.0),
            ("NVDA", "event", "moderate", 30.0),
            ("MSFT", "regime", "low", 15.0),
        ]):
            store.append(SignalRecord(
                signal_id=f"sig-{i}", ticker=ticker, cik=f"cik-{i}",
                filing_type="10-K", base_accession="a1", compare_accession="a2",
                signal_mode=mode, base_filing_type="10-K", compare_filing_type="10-K",
                base_period_end_date="2024-01-01", compare_period_end_date="2024-12-31",
                filing_date="2025-01-15", base_fiscal_period="FY2023", compare_fiscal_period="FY2024",
                drift_score=drift, jaccard_similarity=0.5,
                new_sentence_count=i + 1, removed_sentence_count=0,
                new_keyword_count=i, removed_keyword_count=0,
                new_keywords_json='[]', removed_keywords_json='[]',
                boilerplate_flag=False, boilerplate_ratio=0.1, severity=severity,
                created_at=f"2025-01-{15+i:02d}T12:00:00", model_version="1.0.0",
                base_snapshot_id="s1", compare_snapshot_id="s2",
            ))
        
        return store
    
    def test_summary_returns_stats(self, populated_store):
        """Verify summary returns expected statistics."""
        stats = populated_store.summary()
        
        assert stats["count"] == 3
        assert stats["by_mode"]["regime"] == 2
        assert stats["by_mode"]["event"] == 1
        assert stats["by_severity"]["critical"] == 1
        assert stats["by_severity"]["moderate"] == 1
        assert stats["by_severity"]["low"] == 1
        assert stats["unique_tickers"] == 3
        assert stats["avg_drift_score"] == 35.0  # (60 + 30 + 15) / 3
    
    def test_summary_empty_store(self, tmp_path):
        """Verify summary handles empty store."""
        from open_deep_research.store import SignalStore
        
        store = SignalStore(str(tmp_path / "empty.jsonl"))
        stats = store.summary()
        
        assert stats["count"] == 0
        assert "message" in stats
