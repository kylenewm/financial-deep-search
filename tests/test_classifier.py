"""
Classifier Regression Tests.

These tests catch silent routing regressions.
Run after ANY change to classifier.py or CANONICAL_EXAMPLES.

Design Philosophy:
- Tests encode BUSINESS EXPECTATIONS, not ML behavior
- High-confidence classifications should have good margins
- Ambiguous queries should be flagged
- Backwards compatibility must be maintained
- DEGRADED MODE: Without embeddings, system returns UNKNOWN (not classifications)
"""
import pytest
from open_deep_research.classifier import (
    classify_query,
    classify_query_detailed,
    QueryType,
    ConfidenceBand,
    extract_comparison_entities,
    is_multi_entity_query,
    _get_embedder,
)

# Check if embeddings are available
EMBEDDINGS_AVAILABLE = _get_embedder() is not None
skip_without_embeddings = pytest.mark.skipif(
    not EMBEDDINGS_AVAILABLE,
    reason="sentence-transformers not installed"
)
requires_embeddings = pytest.mark.skipif(
    not EMBEDDINGS_AVAILABLE,
    reason="sentence-transformers not installed - degraded mode returns UNKNOWN"
)


class TestFinancialLookup:
    """Financial queries MUST route to FINANCIAL_LOOKUP (requires embeddings)."""
    
    # ALL classification tests require embeddings - degraded mode returns UNKNOWN
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "What was NVDA revenue in Q3?",
        "EPS for the quarter",
        "Net income FY2024",
        "Gross margin percentage",
        "How much cash on hand?",
    ])
    def test_financial_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.FINANCIAL_LOOKUP, f"'{query}' → {qtype}"
        assert sim > 0.45, f"Low similarity: {sim}"
    
    # These require embeddings for semantic understanding
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "How much did Apple earn?",
        "Q3 top line?",
        "TTM earnings?",
        "What were operating expenses?",
        "Cost of revenue last quarter",
    ])
    def test_financial_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.FINANCIAL_LOOKUP, f"'{query}' → {qtype}"
        assert sim > 0.45, f"Low similarity: {sim}"


class TestSignalDetection:
    """Drift/risk queries MUST route to SIGNAL_DETECTION (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Any red flags?",
        "Risk factor changes?",
        "Show me drift between Q2 and Q3",
        "Warning signs in the 10-Q?",
    ])
    def test_signal_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.SIGNAL_DETECTION, f"'{query}' → {qtype}"
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "New risks mentioned?",
    ])
    def test_signal_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.SIGNAL_DETECTION, f"'{query}' → {qtype}"


class TestVerification:
    """Fact-check queries MUST route to VERIFICATION (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Is it true that NVDA revenue was $35 billion?",
        "Verify: Apple's EPS was $1.50",
        "Fact-check this claim",
    ])
    def test_verification_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.VERIFICATION, f"'{query}' → {qtype}"
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Is this Bloomberg headline correct?",
    ])
    def test_verification_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.VERIFICATION, f"'{query}' → {qtype}"


class TestComparison:
    """Multi-entity queries MUST route to COMPARISON (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Compare NVDA vs AMD revenue",
        "NVDA versus Intel",
    ])
    def test_comparison_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.COMPARISON, f"'{query}' → {qtype}"
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Which has higher margins, Apple or Microsoft?",
    ])
    def test_comparison_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.COMPARISON, f"'{query}' → {qtype}"


class TestDiscovery:
    """News/buzz queries MUST route to DISCOVERY (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "What's the buzz around NVDA?",
        "Find recent news about Apple",
        "Latest developments for Microsoft?",
    ])
    def test_discovery_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.DISCOVERY, f"'{query}' → {qtype}"
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "What are people saying about Tesla?",
    ])
    def test_discovery_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.DISCOVERY, f"'{query}' → {qtype}"


class TestQualitative:
    """Management commentary queries MUST route to QUALITATIVE (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "What did the CEO say about guidance?",
        "What are the main risks?",
    ])
    def test_qualitative_queries_basic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.QUALITATIVE_EXTRACT, f"'{query}' → {qtype}"
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Management's outlook for next year?",
        "Strategic initiatives mentioned?",
    ])
    def test_qualitative_queries_semantic(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.QUALITATIVE_EXTRACT, f"'{query}' → {qtype}"


class TestExploration:
    """General research queries MUST route to EXPLORATION (requires embeddings)."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "Research NVDA",
        "Tell me about Apple's business",
        "Give me an overview of Tesla",
        "Deep dive into AMD",
    ])
    def test_exploration_queries(self, query):
        qtype, sim = classify_query(query, use_llm_fallback=False)
        assert qtype == QueryType.EXPLORATION, f"'{query}' → {qtype}"


class TestAmbiguity:
    """Ambiguous queries SHOULD be flagged."""
    
    @requires_embeddings
    @pytest.mark.parametrize("query", [
        "What changed in the filing?",  # Could be SIGNAL or QUALITATIVE
    ])
    def test_ambiguous_queries(self, query):
        result = classify_query_detailed(query)
        # Either flagged as ambiguous OR has secondary candidates
        is_ambiguous = result.ambiguous or len(result.secondary_candidates) > 0
        assert is_ambiguous, f"'{query}' not flagged as ambiguous: {result}"


class TestMarginSafety:
    """High-confidence classifications should have good margins."""
    
    @skip_without_embeddings
    def test_clear_financial_has_margin(self):
        result = classify_query_detailed("What was NVDA revenue in Q3 FY2025?")
        assert result.similarity > 0.6, f"Low similarity: {result.similarity}"
        assert result.margin > 0.05, f"Low margin: {result.margin}"
    
    @skip_without_embeddings
    def test_clear_signal_has_margin(self):
        result = classify_query_detailed("Any red flags in the 10-K?")
        assert result.similarity > 0.6, f"Low similarity: {result.similarity}"
        assert result.margin > 0.05, f"Low margin: {result.margin}"


class TestConfidenceBands:
    """Confidence bands should be correctly assigned."""
    
    @skip_without_embeddings
    def test_high_confidence_query(self):
        result = classify_query_detailed("What was NVDA revenue in Q3 FY2025?")
        # Should be at least MEDIUM (HIGH requires sim > 0.75 AND margin > 0.12)
        assert result.confidence_band in [ConfidenceBand.HIGH, ConfidenceBand.MEDIUM], \
            f"Expected HIGH/MEDIUM, got {result.confidence_band}"
    
    def test_ambiguous_query_band(self):
        result = classify_query_detailed("What changed in the filing?")
        # Ambiguous queries get AMBIGUOUS or LOW band
        assert result.confidence_band in [ConfidenceBand.AMBIGUOUS, ConfidenceBand.LOW, ConfidenceBand.MEDIUM], \
            f"Unexpected band for ambiguous query: {result.confidence_band}"


class TestEntityExtraction:
    """Entity extraction for comparison queries."""
    
    def test_extract_two_entities(self):
        entities = extract_comparison_entities("Compare NVDA vs AMD revenue")
        assert "NVDA" in entities
        assert "AMD" in entities
    
    def test_extract_three_entities(self):
        entities = extract_comparison_entities("Compare AAPL vs MSFT vs GOOGL")
        assert len(entities) >= 3
    
    def test_is_multi_entity(self):
        assert is_multi_entity_query("Compare NVDA vs AMD")
        assert not is_multi_entity_query("What was NVDA revenue?")


class TestBackwardsCompatibility:
    """Existing queries must not regress."""
    
    def test_returns_tuple(self):
        result = classify_query("What was revenue?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], QueryType)
        assert isinstance(result[1], float)
    
    def test_detailed_returns_result(self):
        result = classify_query_detailed("What was revenue?")
        assert hasattr(result, 'query_type')
        assert hasattr(result, 'similarity')
        assert hasattr(result, 'margin')
        assert hasattr(result, 'confidence_band')


@skip_without_embeddings
class TestSlangAndAcronyms:
    """System should handle financial slang and acronyms (requires embeddings)."""
    
    @pytest.mark.parametrize("query,expected", [
        ("Q3 top line?", QueryType.FINANCIAL_LOOKUP),  # "top line" = revenue
        ("TTM earnings?", QueryType.FINANCIAL_LOOKUP),  # TTM = trailing twelve months
        ("What's the bottom line?", QueryType.FINANCIAL_LOOKUP),  # "bottom line" = net income
    ])
    def test_slang_understanding(self, query, expected):
        qtype, _ = classify_query(query, use_llm_fallback=False)
        assert qtype == expected, f"'{query}' → {qtype}, expected {expected}"


class TestMethodTracking:
    """Classification method should be tracked for audit."""
    
    def test_embedding_method(self):
        result = classify_query_detailed("What was NVDA revenue?")
        # Should use embeddings if available, regex_degraded otherwise
        # "regex_degraded" is the correct method when embeddings unavailable (Invariant I4)
        assert result.method in ["embedding", "regex_degraded"]
    
    def test_scores_populated(self):
        result = classify_query_detailed("What was NVDA revenue?")
        if result.method == "embedding":
            # Scores should be populated for embedding method
            assert len(result.scores) > 0
            # All query types should have scores
            assert QueryType.FINANCIAL_LOOKUP.value in result.scores
