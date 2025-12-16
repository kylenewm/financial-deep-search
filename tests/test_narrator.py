"""
Tests for the narrator module.

These tests verify:
1. Fact formatting for prompts
2. Citation parsing from LLM output
3. Relevant fact retrieval
4. Full report generation (mocked LLM)
5. Insufficient data handling
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.models import Citation, Fact, Location, NarratedReport
from open_deep_research.narrator import (
    format_fact_for_prompt,
    format_facts_for_prompt,
    generate_report,
    parse_citations_from_answer,
    retrieve_relevant_facts,
)
from open_deep_research.store import FactStore


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_facts() -> list[Fact]:
    """Create sample facts for testing."""
    return [
        Fact(
            fact_id="fact-1",
            entity="NVDA",
            metric="Total Revenue",
            value=35082000000.0,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-10-27",
                doc_type="10-Q",
                section_id="Item2",
            ),
            source_format="xbrl",
            doc_hash="abc123",
            snapshot_id="snap-1",
            verification_status="exact_match",
        ),
        Fact(
            fact_id="fact-2",
            entity="NVDA",
            metric="Net Income",
            value=19309000000.0,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-10-27",
                doc_type="10-Q",
                section_id="Item2",
            ),
            source_format="xbrl",
            doc_hash="abc123",
            snapshot_id="snap-1",
            verification_status="exact_match",
        ),
        Fact(
            fact_id="fact-3",
            entity="AAPL",
            metric="Total Revenue",
            value=124300000000.0,
            unit="USD",
            period="Q1 FY2025",
            period_end_date="2024-12-28",
            location=Location(
                cik="0000320193",
                doc_date="2024-12-28",
                doc_type="10-Q",
                section_id="Item1",
                paragraph_index=5,
            ),
            source_format="html_text",
            doc_hash="def456",
            snapshot_id="snap-2",
            verification_status="exact_match",
        ),
    ]


@pytest.fixture
def populated_store(sample_facts: list[Fact]) -> FactStore:
    """Create a FactStore with sample facts."""
    store = FactStore()
    for fact in sample_facts:
        store.add_fact(fact)
    return store


# =============================================================================
# Test: format_fact_for_prompt
# =============================================================================


class TestFormatFactForPrompt:
    """Tests for format_fact_for_prompt function."""
    
    def test_formats_billions(self, sample_facts: list[Fact]) -> None:
        """Test formatting of billion-dollar values."""
        fact = sample_facts[0]  # NVDA revenue $35.08B
        result = format_fact_for_prompt(fact, 1)
        
        assert "[1]" in result
        assert "NVDA" in result
        assert "Total Revenue" in result
        assert "Q3 FY2025" in result
        assert "$35.08B" in result
        assert "xbrl" in result.lower()
    
    def test_formats_millions(self) -> None:
        """Test formatting of million-dollar values."""
        fact = Fact(
            fact_id="test",
            entity="TEST",
            metric="Revenue",
            value=500000000.0,
            unit="USD",
            period="Q1 FY2025",
            period_end_date="2024-01-01",
            location=Location(
                cik="0000000000",
                doc_date="2024-01-01",
                doc_type="10-Q",
                section_id="Item1",
            ),
            source_format="xbrl",
            doc_hash="hash",
            snapshot_id="snap",
            verification_status="exact_match",
        )
        result = format_fact_for_prompt(fact, 1)
        assert "$500.00M" in result
    
    def test_includes_citation_index(self, sample_facts: list[Fact]) -> None:
        """Test that citation index is included."""
        result = format_fact_for_prompt(sample_facts[1], 5)
        assert "[5]" in result
    
    def test_includes_source_format(self, sample_facts: list[Fact]) -> None:
        """Test that source format is included."""
        result = format_fact_for_prompt(sample_facts[2], 1)  # html_text
        assert "html_text" in result
    
    def test_handles_paragraph_index(self, sample_facts: list[Fact]) -> None:
        """Test that paragraph index is included when present."""
        result = format_fact_for_prompt(sample_facts[2], 1)
        assert "Item1" in result
        assert "Para" in result


# =============================================================================
# Test: format_facts_for_prompt
# =============================================================================


class TestFormatFactsForPrompt:
    """Tests for format_facts_for_prompt function."""
    
    def test_formats_multiple_facts(self, sample_facts: list[Fact]) -> None:
        """Test formatting multiple facts."""
        result = format_facts_for_prompt(sample_facts)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "NVDA" in result
        assert "AAPL" in result
    
    def test_empty_list(self) -> None:
        """Test formatting empty list."""
        result = format_facts_for_prompt([])
        assert "No verified facts available" in result
    
    def test_single_fact(self, sample_facts: list[Fact]) -> None:
        """Test formatting single fact."""
        result = format_facts_for_prompt([sample_facts[0]])
        assert "[1]" in result
        assert "[2]" not in result


# =============================================================================
# Test: parse_citations_from_answer
# =============================================================================


class TestParseCitationsFromAnswer:
    """Tests for parse_citations_from_answer function."""
    
    def test_parses_single_citation(self, sample_facts: list[Fact]) -> None:
        """Test parsing a single citation."""
        answer = "NVIDIA's total revenue was $35.08B [1]."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert len(citations) == 1
        assert citations[0].citation_index == 1
        assert citations[0].fact_id == "fact-1"
    
    def test_parses_multiple_citations(self, sample_facts: list[Fact]) -> None:
        """Test parsing multiple citations."""
        answer = "NVIDIA reported revenue of $35.08B [1] and net income of $19.31B [2]."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert len(citations) == 2
        assert citations[0].citation_index == 1
        assert citations[1].citation_index == 2
    
    def test_handles_repeated_citations(self, sample_facts: list[Fact]) -> None:
        """Test that repeated citations are deduplicated."""
        answer = "Revenue was $35.08B [1]. This revenue [1] was strong."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert len(citations) == 1
    
    def test_ignores_invalid_indices(self, sample_facts: list[Fact]) -> None:
        """Test that invalid citation indices are ignored."""
        answer = "Revenue [1] and some invalid [99] and [0]."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert len(citations) == 1
        assert citations[0].citation_index == 1
    
    def test_no_citations(self, sample_facts: list[Fact]) -> None:
        """Test handling of text with no citations."""
        answer = "The company performed well."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert len(citations) == 0
    
    def test_includes_source_format(self, sample_facts: list[Fact]) -> None:
        """Test that citation includes source format."""
        answer = "Revenue was $35.08B [1]."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert citations[0].source_format == "xbrl"
    
    def test_includes_location(self, sample_facts: list[Fact]) -> None:
        """Test that citation includes location string."""
        answer = "Revenue was $35.08B [1]."
        citations = parse_citations_from_answer(answer, sample_facts)
        
        assert "XBRL" in citations[0].location
        assert "Q3 FY2025" in citations[0].location


# =============================================================================
# Test: retrieve_relevant_facts
# =============================================================================


class TestRetrieveRelevantFacts:
    """Tests for retrieve_relevant_facts function."""
    
    def test_filter_by_entity(self, populated_store: FactStore) -> None:
        """Test filtering facts by entity."""
        facts = retrieve_relevant_facts(
            query="What is the revenue?",
            fact_store=populated_store,
            entity="NVDA",
        )
        
        assert all(f.entity == "NVDA" for f in facts)
    
    def test_filter_by_metric(self, populated_store: FactStore) -> None:
        """Test filtering facts by metric."""
        facts = retrieve_relevant_facts(
            query="Tell me about revenue",
            fact_store=populated_store,
            metric="Revenue",
        )
        
        assert all("revenue" in f.metric.lower() for f in facts)
    
    def test_filter_by_period(self, populated_store: FactStore) -> None:
        """Test filtering facts by period."""
        facts = retrieve_relevant_facts(
            query="Q3 results",
            fact_store=populated_store,
            period="Q3 FY2025",
        )
        
        assert all("Q3" in f.period for f in facts)
    
    def test_combined_filters(self, populated_store: FactStore) -> None:
        """Test combining multiple filters."""
        facts = retrieve_relevant_facts(
            query="NVIDIA revenue",
            fact_store=populated_store,
            entity="NVDA",
            metric="Revenue",
        )
        
        assert len(facts) == 1
        assert facts[0].metric == "Total Revenue"
        assert facts[0].entity == "NVDA"
    
    def test_empty_store(self) -> None:
        """Test with empty store."""
        store = FactStore()
        facts = retrieve_relevant_facts("What is revenue?", store)
        
        assert len(facts) == 0


# =============================================================================
# Test: generate_report (mocked LLM)
# =============================================================================


class TestGenerateReport:
    """Tests for generate_report function with mocked LLM."""
    
    @patch("open_deep_research.narrator.Anthropic")
    def test_generates_report_with_citations(
        self,
        mock_anthropic: MagicMock,
        populated_store: FactStore,
    ) -> None:
        """Test that report generation includes citations."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="NVIDIA's total revenue was $35.08 billion [1] in Q3 FY2025.")]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            report = generate_report(
                query="What was NVIDIA's revenue in Q3 FY2025?",
                fact_store=populated_store,
                entity="NVDA",
            )
        
        assert isinstance(report, NarratedReport)
        assert "35.08" in report.answer
        assert len(report.citations) >= 1
        assert report.insufficient_data is False
    
    @patch("open_deep_research.narrator.Anthropic")
    def test_handles_insufficient_data(
        self,
        mock_anthropic: MagicMock,
        populated_store: FactStore,
    ) -> None:
        """Test handling when LLM says insufficient data."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Insufficient data to answer this question.")]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            report = generate_report(
                query="What is Tesla's revenue?",
                fact_store=populated_store,
                entity="TSLA",  # Not in store
            )
        
        assert report.insufficient_data is True
    
    def test_no_facts_returns_insufficient_data(self) -> None:
        """Test that empty store returns insufficient data."""
        store = FactStore()
        
        # No API call needed when no facts
        report = generate_report(
            query="What is the revenue?",
            fact_store=store,
        )
        
        assert report.insufficient_data is True
        assert "Insufficient data" in report.answer
        assert len(report.citations) == 0
    
    @patch("open_deep_research.narrator.Anthropic")
    def test_facts_used_matches_citations(
        self,
        mock_anthropic: MagicMock,
        populated_store: FactStore,
    ) -> None:
        """Test that facts_used contains only cited facts."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Revenue was $35.08B [1]. Net income was $19.31B [2].")]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            report = generate_report(
                query="What were NVIDIA's Q3 results?",
                fact_store=populated_store,
                entity="NVDA",
            )
        
        # Should have 2 facts used (revenue and net income)
        assert len(report.facts_used) == 2
        cited_ids = {c.fact_id for c in report.citations}
        used_ids = {f.fact_id for f in report.facts_used}
        assert cited_ids == used_ids
    
    def test_raises_without_api_key(self, populated_store: FactStore) -> None:
        """Test that missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing key
            import os
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                generate_report(
                    query="What is revenue?",
                    fact_store=populated_store,
                )


# =============================================================================
# Test: NarratedReport Model
# =============================================================================


class TestNarratedReportModel:
    """Tests for NarratedReport model."""
    
    def test_create_report(self, sample_facts: list[Fact]) -> None:
        """Test creating a NarratedReport."""
        report = NarratedReport(
            query="What was revenue?",
            answer="Revenue was $35.08B [1].",
            citations=[
                Citation(
                    fact_id="fact-1",
                    citation_index=1,
                    source_format="xbrl",
                    location="XBRL: Total Revenue, Q3 FY2025",
                )
            ],
            facts_used=[sample_facts[0]],
            generated_at=datetime.now(),
        )
        
        assert report.query == "What was revenue?"
        assert len(report.citations) == 1
        assert len(report.facts_used) == 1
        assert report.insufficient_data is False
    
    def test_insufficient_data_flag(self) -> None:
        """Test insufficient_data flag."""
        report = NarratedReport(
            query="Unknown question",
            answer="Insufficient data.",
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
        
        assert report.insufficient_data is True


# =============================================================================
# Test: Citation Model
# =============================================================================


class TestCitationModel:
    """Tests for Citation model."""
    
    def test_create_citation(self) -> None:
        """Test creating a Citation."""
        citation = Citation(
            fact_id="fact-123",
            citation_index=1,
            source_format="xbrl",
            location="us-gaap:Revenues, Q3 FY2025",
        )
        
        assert citation.fact_id == "fact-123"
        assert citation.citation_index == 1
        assert citation.source_format == "xbrl"
        assert "Revenues" in citation.location

