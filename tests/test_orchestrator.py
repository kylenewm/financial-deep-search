"""
Tests for the Orchestrator module.

Tests the simple user-facing API that wraps all pipeline components.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from open_deep_research.orchestrator import (
    Orchestrator,
    parse_fiscal_period,
    extract_entity_from_question,
    extract_period_from_question,
    quick_answer,
)
from open_deep_research.models import Fact, Location, NarratedReport, DiscoveryReport
from open_deep_research.store import FactStore


# =============================================================================
# Period Parsing Tests
# =============================================================================


class TestParseFiscalPeriod:
    """Tests for parse_fiscal_period function."""
    
    def test_q3_fy2025(self):
        year, period = parse_fiscal_period("Q3 FY2025")
        assert year == 2025
        assert period == "Q3"
    
    def test_q3fy2025_no_space(self):
        year, period = parse_fiscal_period("Q3FY2025")
        assert year == 2025
        assert period == "Q3"
    
    def test_fy2025_full_year(self):
        year, period = parse_fiscal_period("FY2025")
        assert year == 2025
        assert period == "FY"
    
    def test_q3_2025(self):
        year, period = parse_fiscal_period("Q3 2025")
        assert year == 2025
        assert period == "Q3"
    
    def test_2025_q3(self):
        year, period = parse_fiscal_period("2025 Q3")
        assert year == 2025
        assert period == "Q3"
    
    def test_just_year(self):
        year, period = parse_fiscal_period("2025")
        assert year == 2025
        assert period == "FY"
    
    def test_case_insensitive(self):
        year, period = parse_fiscal_period("q3 fy2025")
        assert year == 2025
        assert period == "Q3"
    
    def test_invalid_period(self):
        year, period = parse_fiscal_period("invalid")
        assert year is None
        assert period is None


# =============================================================================
# Entity Extraction Tests
# =============================================================================


class TestExtractEntityFromQuestion:
    """Tests for extract_entity_from_question function."""
    
    def test_nvidia(self):
        entity = extract_entity_from_question("What was NVIDIA's revenue?")
        assert entity == "NVDA"
    
    def test_nvda(self):
        entity = extract_entity_from_question("What was NVDA's revenue?")
        assert entity == "NVDA"
    
    def test_apple(self):
        entity = extract_entity_from_question("What was Apple's revenue?")
        assert entity == "AAPL"
    
    def test_walmart(self):
        entity = extract_entity_from_question("What was Walmart's revenue?")
        assert entity == "WMT"
    
    def test_case_insensitive(self):
        entity = extract_entity_from_question("what was nvidia's revenue?")
        assert entity == "NVDA"


# =============================================================================
# Period Extraction Tests
# =============================================================================


class TestExtractPeriodFromQuestion:
    """Tests for extract_period_from_question function."""
    
    def test_q3_fy2025(self):
        period = extract_period_from_question("What was revenue in Q3 FY2025?")
        assert period == "Q3 FY2025"
    
    def test_q3_2025(self):
        period = extract_period_from_question("What was revenue in Q3 2025?")
        assert period == "Q3 FY2025"
    
    def test_fy2024(self):
        period = extract_period_from_question("What was revenue in FY2024?")
        assert period == "FY2024"
    
    def test_no_period(self):
        period = extract_period_from_question("What was revenue?")
        assert period is None


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestOrchestrator:
    """Tests for Orchestrator class."""
    
    def test_init(self):
        orc = Orchestrator()
        assert len(orc.fact_store) == 0
        assert len(orc._loaded_periods) == 0
    
    def test_repr(self):
        orc = Orchestrator()
        assert "facts=0" in repr(orc)
    
    def test_clear(self):
        orc = Orchestrator()
        orc._loaded_periods.add(("NVDA", "Q3 FY2025"))
        orc.clear()
        assert len(orc._loaded_periods) == 0
        assert len(orc.fact_store) == 0
    
    def test_get_available_metrics(self):
        orc = Orchestrator()
        metrics = orc.get_available_metrics()
        assert len(metrics) > 0
        assert "total revenue" in metrics
        assert "net income" in metrics
    
    @patch("open_deep_research.orchestrator.extract_xbrl_fact")
    @patch("open_deep_research.orchestrator.resolve_entity")
    def test_load_facts_for_entity(self, mock_resolve, mock_extract):
        """Test loading facts with mocked XBRL extraction."""
        # Setup mocks
        mock_resolve.return_value = MagicMock(cik="0001045810", ticker="NVDA")
        
        mock_fact = Fact(
            fact_id="test-123",
            entity="NVDA",
            metric="Total Revenue",
            value=35082000000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="xbrl",
                paragraph_index=None,
                sentence_string="XBRL data",
            ),
            source_format="xbrl",
            verification_status="exact_match",
            doc_hash="abc123",
            snapshot_id="snap-123",
        )
        mock_extract.return_value = mock_fact
        
        orc = Orchestrator()
        count = orc.load_facts_for_entity("NVDA", "Q3 FY2025")
        
        assert count > 0
        assert len(orc.fact_store) > 0
        assert ("NVDA", "Q3 FY2025") in orc._loaded_periods
    
    @patch("open_deep_research.orchestrator.resolve_entity")
    def test_load_facts_unknown_ticker(self, mock_resolve):
        """Test loading facts for unknown ticker."""
        mock_resolve.return_value = None
        
        orc = Orchestrator()
        count = orc.load_facts_for_entity("UNKNOWN", "Q3 FY2025")
        
        assert count == 0
    
    def test_load_facts_invalid_period(self):
        """Test loading facts with invalid period."""
        orc = Orchestrator()
        count = orc.load_facts_for_entity("NVDA", "invalid period xyz")
        
        assert count == 0
    
    @patch("open_deep_research.orchestrator.extract_xbrl_fact")
    @patch("open_deep_research.orchestrator.resolve_entity")
    def test_load_facts_already_loaded(self, mock_resolve, mock_extract):
        """Test that already loaded periods are not reloaded."""
        mock_resolve.return_value = MagicMock(cik="0001045810", ticker="NVDA")
        mock_extract.return_value = None  # No facts
        
        orc = Orchestrator()
        orc._loaded_periods.add(("NVDA", "Q3 FY2025"))
        
        count = orc.load_facts_for_entity("NVDA", "Q3 FY2025")
        
        assert count == 0
        mock_extract.assert_not_called()


# =============================================================================
# Integration Tests (require ANTHROPIC_API_KEY)
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests that require live API calls."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("os").environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_ask_with_mocked_facts(self):
        """Test ask function with pre-populated facts."""
        orc = Orchestrator()
        
        # Manually add a fact
        fact = Fact(
            fact_id="test-123",
            entity="NVDA",
            metric="Total Revenue",
            value=35082000000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="xbrl",
                paragraph_index=None,
                sentence_string="XBRL data",
            ),
            source_format="xbrl",
            verification_status="exact_match",
            doc_hash="abc123",
            snapshot_id="snap-123",
        )
        orc.fact_store.add_fact(fact)
        
        # Ask question
        report = orc.ask("What was NVIDIA's revenue in Q3 FY2025?", auto_load=False)
        
        assert isinstance(report, NarratedReport)
        assert report.answer is not None
        assert len(report.answer) > 0


# =============================================================================
# Discovery Integration Tests
# =============================================================================


class TestOrchestratorDiscovery:
    """Test discovery integration."""
    
    @pytest.fixture
    def orc_with_facts(self):
        """Orchestrator with NVDA facts loaded."""
        orc = Orchestrator()
        # Add a fact directly for testing
        fact = Fact(
            fact_id="test-nvda-revenue",
            entity="NVDA",
            metric="Total Revenue",
            value=35_082_000_000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="Item2",
            ),
            source_format="xbrl",
            doc_hash="test",
            snapshot_id="test",
            verification_status="exact_match",
        )
        orc.fact_store._facts[fact.fact_id] = fact
        return orc
    
    def test_discover_from_article(self, orc_with_facts):
        """Should extract and verify claims from article."""
        article = "NVIDIA reported $35 billion in revenue for Q3."
        
        report = orc_with_facts.discover_from_article(
            text=article,
            source_name="Reuters",
            ticker="NVDA",
        )
        
        assert isinstance(report, DiscoveryReport)
        assert report.total_leads >= 1
        # The $35B claim should be confirmed (close to $35.08B)
        confirmed = [l for l in report.leads if l.verification_status == "confirmed"]
        assert len(confirmed) >= 1
    
    def test_discover_from_article_finds_contradiction(self, orc_with_facts):
        """Should detect contradicted claims."""
        article = "NVIDIA reported $20 billion in revenue."  # Wrong!
        
        report = orc_with_facts.discover_from_article(
            text=article,
            source_name="BadSource",
            ticker="NVDA",
        )
        
        assert report.total_leads >= 1
        contradicted = [l for l in report.leads if l.verification_status == "contradicted"]
        assert len(contradicted) >= 1
    
    def test_discover_no_verify(self, orc_with_facts):
        """Should return unverified leads when auto_verify=False."""
        article = "NVIDIA reported $35 billion in revenue."
        
        report = orc_with_facts.discover_from_article(
            text=article,
            source_name="Test",
            ticker="NVDA",
            auto_verify=False,
        )
        
        assert report.total_leads >= 1
        # Should be pending since not verified
        pending = [l for l in report.leads if l.verification_status == "pending"]
        assert len(pending) >= 1
    
    def test_discover_returns_report(self, orc_with_facts):
        """discover() should return DiscoveryReport."""
        report = orc_with_facts.discover(
            query="NVDA news",
            ticker="NVDA",
            auto_verify=True,
        )
        
        assert isinstance(report, DiscoveryReport)
        assert report.query == "NVDA news"
        assert report.ticker == "NVDA"
    
    def test_format_report_output(self, orc_with_facts):
        """format_report should return readable output."""
        article = "NVIDIA reported $35 billion in revenue."
        
        report = orc_with_facts.discover_from_article(
            text=article,
            source_name="Reuters",
            ticker="NVDA",
        )
        
        formatted = report.format_report()
        assert "DISCOVERY REPORT" in formatted
        assert "NVDA" in formatted

