"""
Tests for the XBRL fetcher and extractor module.

These tests verify:
1. XBRL concept mappings are correct
2. Fact extraction works for known companies
3. Caching behavior is correct
4. Error handling for invalid CIKs
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.xbrl import (
    METRIC_TO_CONCEPTS,
    CONCEPT_TO_METRIC,
    fetch_company_facts,
    get_concept_data,
    find_fact_by_period,
    find_facts_by_fiscal_period,
    extract_xbrl_fact,
    get_latest_value,
    list_available_concepts,
)
from open_deep_research.entities import pad_cik


# =============================================================================
# Test Data
# =============================================================================


# Sample XBRL response structure (minimal)
SAMPLE_XBRL_DATA = {
    "cik": 1045810,
    "entityName": "NVIDIA CORP",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {
                            "end": "2024-10-27",
                            "val": 35082000000,
                            "form": "10-Q",
                            "filed": "2024-11-20",
                            "fy": 2025,
                            "fp": "Q3",
                        },
                        {
                            "end": "2024-07-28",
                            "val": 30040000000,
                            "form": "10-Q",
                            "filed": "2024-08-28",
                            "fy": 2025,
                            "fp": "Q2",
                        },
                        {
                            "end": "2024-01-28",
                            "val": 60922000000,
                            "form": "10-K",
                            "filed": "2024-02-21",
                            "fy": 2024,
                            "fp": "FY",
                        },
                    ]
                }
            },
            "NetIncomeLoss": {
                "label": "Net Income (Loss)",
                "units": {
                    "USD": [
                        {
                            "end": "2024-10-27",
                            "val": 19309000000,
                            "form": "10-Q",
                            "filed": "2024-11-20",
                            "fy": 2025,
                            "fp": "Q3",
                        },
                    ]
                }
            },
            "GrossProfit": {
                "label": "Gross Profit",
                "units": {
                    "USD": [
                        {
                            "end": "2024-10-27",
                            "val": 26156000000,
                            "form": "10-Q",
                            "filed": "2024-11-20",
                            "fy": 2025,
                            "fp": "Q3",
                        },
                    ]
                }
            },
            "EarningsPerShareDiluted": {
                "label": "Earnings Per Share, Diluted",
                "units": {
                    "USD/shares": [
                        {
                            "end": "2024-10-27",
                            "val": 0.78,
                            "form": "10-Q",
                            "filed": "2024-11-20",
                            "fy": 2025,
                            "fp": "Q3",
                        },
                    ]
                }
            },
        }
    }
}


# =============================================================================
# Concept Mapping Tests
# =============================================================================


class TestConceptMappings:
    """Tests for XBRL concept mappings."""
    
    def test_revenue_mapping_exists(self):
        """Revenue metrics should have concept mappings."""
        assert "total revenue" in METRIC_TO_CONCEPTS
        assert "revenue" in METRIC_TO_CONCEPTS
        assert "Revenues" in METRIC_TO_CONCEPTS["total revenue"]
    
    def test_net_income_mapping_exists(self):
        """Net income metrics should have concept mappings."""
        assert "net income" in METRIC_TO_CONCEPTS
        assert "NetIncomeLoss" in METRIC_TO_CONCEPTS["net income"]
    
    def test_eps_mapping_exists(self):
        """EPS metrics should have concept mappings."""
        assert "eps" in METRIC_TO_CONCEPTS
        assert "EarningsPerShareDiluted" in METRIC_TO_CONCEPTS["eps"]
    
    def test_reverse_mapping_populated(self):
        """CONCEPT_TO_METRIC should have reverse mappings."""
        assert "Revenues" in CONCEPT_TO_METRIC
        assert "NetIncomeLoss" in CONCEPT_TO_METRIC


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetConceptData:
    """Tests for get_concept_data function."""
    
    def test_returns_concept_data(self):
        """Should return concept data when it exists."""
        result = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        assert result is not None
        assert "label" in result
        assert "units" in result
    
    def test_returns_none_for_missing(self):
        """Should return None for missing concepts."""
        result = get_concept_data(SAMPLE_XBRL_DATA, "NonexistentConcept")
        assert result is None


class TestFindFactByPeriod:
    """Tests for find_fact_by_period function."""
    
    def test_finds_exact_period(self):
        """Should find fact with exact period match."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        result = find_fact_by_period(concept_data, "2024-10-27")
        
        assert result is not None
        assert result["val"] == 35082000000
        assert result["fy"] == 2025
        assert result["fp"] == "Q3"
    
    def test_filters_by_form(self):
        """Should filter by form type when specified."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        
        # With 10-Q filter
        result = find_fact_by_period(concept_data, "2024-01-28", form="10-Q")
        assert result is None  # 2024-01-28 is 10-K, not 10-Q
        
        # With 10-K filter
        result = find_fact_by_period(concept_data, "2024-01-28", form="10-K")
        assert result is not None
        assert result["val"] == 60922000000
    
    def test_returns_none_for_missing_period(self):
        """Should return None for non-existent period."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        result = find_fact_by_period(concept_data, "1999-01-01")
        assert result is None


class TestFindFactsByFiscalPeriod:
    """Tests for find_facts_by_fiscal_period function."""
    
    def test_finds_fiscal_period(self):
        """Should find facts by fiscal year and period."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        results = find_facts_by_fiscal_period(concept_data, 2025, "Q3")
        
        assert len(results) == 1
        assert results[0]["val"] == 35082000000
    
    def test_finds_annual(self):
        """Should find annual filing."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        results = find_facts_by_fiscal_period(concept_data, 2024, "FY")
        
        assert len(results) == 1
        assert results[0]["val"] == 60922000000
    
    def test_returns_empty_for_missing(self):
        """Should return empty list for missing period."""
        concept_data = get_concept_data(SAMPLE_XBRL_DATA, "Revenues")
        results = find_facts_by_fiscal_period(concept_data, 2030, "Q1")
        assert len(results) == 0


# =============================================================================
# Extract XBRL Fact Tests
# =============================================================================


class TestExtractXbrlFact:
    """Tests for extract_xbrl_fact function."""
    
    @patch('open_deep_research.xbrl.fetch_company_facts')
    def test_extracts_revenue(self, mock_fetch):
        """Should extract revenue fact correctly."""
        mock_fetch.return_value = SAMPLE_XBRL_DATA
        
        fact = extract_xbrl_fact(
            cik="0001045810",
            metric="total revenue",
            fiscal_year=2025,
            fiscal_period="Q3",
        )
        
        assert fact is not None
        assert fact.value == 35082000000
        assert fact.unit == "USD"
        assert fact.period == "Q3 FY2025"
        assert fact.verification_status == "exact_match"
        assert fact.source_format == "xbrl"
    
    @patch('open_deep_research.xbrl.fetch_company_facts')
    def test_extracts_net_income(self, mock_fetch):
        """Should extract net income fact correctly."""
        mock_fetch.return_value = SAMPLE_XBRL_DATA
        
        fact = extract_xbrl_fact(
            cik="0001045810",
            metric="net income",
            fiscal_year=2025,
            fiscal_period="Q3",
        )
        
        assert fact is not None
        assert fact.value == 19309000000
    
    @patch('open_deep_research.xbrl.fetch_company_facts')
    def test_extracts_by_period_end(self, mock_fetch):
        """Should extract fact by period end date."""
        mock_fetch.return_value = SAMPLE_XBRL_DATA
        
        fact = extract_xbrl_fact(
            cik="0001045810",
            metric="gross profit",
            period_end="2024-10-27",
        )
        
        assert fact is not None
        assert fact.value == 26156000000
    
    @patch('open_deep_research.xbrl.fetch_company_facts')
    def test_returns_none_for_missing(self, mock_fetch):
        """Should return None for non-existent metric."""
        mock_fetch.return_value = SAMPLE_XBRL_DATA
        
        fact = extract_xbrl_fact(
            cik="0001045810",
            metric="fake metric that does not exist",
            fiscal_year=2025,
            fiscal_period="Q3",
        )
        
        assert fact is None
    
    @patch('open_deep_research.xbrl.fetch_company_facts')
    def test_normalizes_metric_name(self, mock_fetch):
        """Should normalize metric names (case, spacing)."""
        mock_fetch.return_value = SAMPLE_XBRL_DATA
        
        # Test various casings
        for metric_name in ["Total Revenue", "TOTAL REVENUE", "total revenue", " total revenue "]:
            fact = extract_xbrl_fact(
                cik="0001045810",
                metric=metric_name,
                fiscal_year=2025,
                fiscal_period="Q3",
            )
            assert fact is not None, f"Failed for metric: {metric_name}"
            assert fact.value == 35082000000


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for XBRL data caching."""
    
    @patch('open_deep_research.xbrl.get_xbrl_session')
    def test_caches_response(self, mock_session, tmp_path):
        """Should cache XBRL response to disk."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_XBRL_DATA
        mock_response.raise_for_status = MagicMock()
        mock_session.return_value.get.return_value = mock_response
        
        cik = "0001045810"
        
        # First call - should hit API
        result1 = fetch_company_facts(cik, cache_dir=str(tmp_path))
        assert mock_session.return_value.get.call_count == 1
        
        # Check cache file exists
        cache_file = tmp_path / f"CIK{cik}_facts.json"
        assert cache_file.exists()
        
        # Second call - should use cache (reset mock to verify)
        mock_session.reset_mock()
        result2 = fetch_company_facts(cik, cache_dir=str(tmp_path))
        
        # Should not have called API again
        assert mock_session.return_value.get.call_count == 0
        
        # Results should be equal
        assert result1["entityName"] == result2["entityName"]


# =============================================================================
# Integration Tests (requires network)
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("SEC_USER_AGENT"),
    reason="SEC_USER_AGENT not set - skipping integration tests"
)
class TestIntegration:
    """Integration tests that hit the real SEC API.
    
    These are skipped if SEC_USER_AGENT is not set.
    """
    
    def test_fetch_nvidia_facts(self, tmp_path):
        """Should fetch real NVIDIA XBRL data."""
        cik = "0001045810"  # NVIDIA
        
        data = fetch_company_facts(cik, cache_dir=str(tmp_path))
        
        assert data["entityName"] == "NVIDIA CORP"
        assert "us-gaap" in data["facts"]
    
    def test_extract_nvidia_revenue(self, tmp_path):
        """Should extract real NVIDIA revenue."""
        fact = extract_xbrl_fact(
            cik="0001045810",
            metric="total revenue",
            fiscal_year=2025,
            fiscal_period="Q3",
            cache_dir=str(tmp_path),
        )
        
        assert fact is not None
        assert fact.value > 30_000_000_000  # Q3 FY2025 was $35B+
        assert fact.unit == "USD"
    
    def test_list_nvidia_concepts(self, tmp_path):
        """Should list available concepts for NVIDIA."""
        concepts = list_available_concepts(
            cik="0001045810",
            cache_dir=str(tmp_path),
        )
        
        assert len(concepts) > 100  # NVIDIA has many concepts
        assert "Revenues" in concepts
        assert "NetIncomeLoss" in concepts

