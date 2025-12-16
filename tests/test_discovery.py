"""Tests for discovery system."""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from open_deep_research.config import ResearchConfig
from open_deep_research.discovery import (
    discover,
    discover_from_text,
    format_leads,
    verify_lead,
    verify_leads,
)
from open_deep_research.models import Lead, Fact, Location, DiscoveryReport
from open_deep_research.store import FactStore


class TestDiscoverFromText:
    """Test extracting leads from text."""
    
    def test_extracts_revenue_claim(self):
        """Should extract revenue claim as lead."""
        text = "NVIDIA reported $35 billion in revenue for Q3."
        leads = discover_from_text(text, "Reuters", entity="NVDA")
        
        assert len(leads) >= 1
        revenue_lead = next((l for l in leads if l.metric == "revenue"), None)
        assert revenue_lead is not None
        assert revenue_lead.value_raw == 35_000_000_000
    
    def test_sets_source_info(self):
        """Should set source information correctly."""
        text = "$50 billion in sales"
        leads = discover_from_text(
            text, 
            source_name="Bloomberg",
            source_url="https://bloomberg.com/article",
            source_tier=1,
        )
        
        if leads:
            assert leads[0].source_name == "Bloomberg"
            assert leads[0].source_url == "https://bloomberg.com/article"
            assert leads[0].source_tier == 1
    
    def test_sets_entity_hint(self):
        """Should use entity hint."""
        text = "$35B revenue"
        leads = discover_from_text(text, "Test", entity="NVDA")
        
        if leads:
            assert leads[0].entity == "NVDA"
    
    def test_generates_lead_id(self):
        """Each lead should have unique ID."""
        text = "$35B revenue. $20B profit."
        leads = discover_from_text(text, "Test")
        
        if len(leads) >= 2:
            assert leads[0].lead_id != leads[1].lead_id
    
    def test_sets_pending_status(self):
        """New leads should have pending verification status."""
        text = "$35 billion revenue"
        leads = discover_from_text(text, "Test")
        
        if leads:
            assert leads[0].verification_status == "pending"


class TestDiscover:
    """Test main discover function."""
    
    def test_disabled_returns_empty(self):
        """Should return empty when discovery disabled."""
        config = ResearchConfig()
        config.discovery_enabled = False
        
        leads = discover("test query", config=config)
        assert leads == []
    
    @patch('open_deep_research.discovery._discover_via_news')
    def test_uses_news_backend(self, mock_news):
        """Should use news backend when configured."""
        mock_news.return_value = []
        config = ResearchConfig()
        config.discovery_enabled = True
        config.discovery_backend = "news"
        
        discover("test query", config=config)
        mock_news.assert_called_once()


class TestFormatLeads:
    """Test lead formatting."""
    
    def test_empty_leads(self):
        """Should handle empty list."""
        result = format_leads([])
        assert "No leads found" in result
    
    def test_formats_lead_count(self):
        """Should show lead count."""
        leads = [
            Lead(
                lead_id="test1",
                text="Test claim",
                source_name="Reuters",
                source_tier=1,
                found_at=datetime.now(),
            )
        ]
        result = format_leads(leads)
        assert "1 LEADS" in result
    
    def test_shows_source_tier(self):
        """Should show source tier."""
        leads = [
            Lead(
                lead_id="test1",
                text="Test claim",
                source_name="Reuters",
                source_tier=1,
                found_at=datetime.now(),
            )
        ]
        result = format_leads(leads)
        assert "Tier 1" in result


class TestLeadModel:
    """Test Lead model."""
    
    def test_status_icon_pending(self):
        """Pending should show search icon."""
        lead = Lead(
            lead_id="test",
            text="Test",
            source_name="Test",
            found_at=datetime.now(),
            verification_status="pending",
        )
        assert lead.get_status_icon() == "🔍"
    
    def test_status_icon_confirmed(self):
        """Confirmed should show checkmark."""
        lead = Lead(
            lead_id="test",
            text="Test",
            source_name="Test",
            found_at=datetime.now(),
            verification_status="confirmed",
        )
        assert lead.get_status_icon() == "✅"
    
    def test_status_icon_contradicted(self):
        """Contradicted should show red circle."""
        lead = Lead(
            lead_id="test",
            text="Test",
            source_name="Test",
            found_at=datetime.now(),
            verification_status="contradicted",
        )
        assert lead.get_status_icon() == "🔴"


# =============================================================================
# Verification Tests
# =============================================================================


def make_test_fact(
    entity: str = "NVDA",
    metric: str = "Total Revenue",
    value: float = 35_082_000_000,
) -> Fact:
    """Create test fact."""
    return Fact(
        fact_id="test-fact",
        entity=entity,
        metric=metric,
        value=value,
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


class TestVerifyLead:
    """Test lead verification."""
    
    @pytest.fixture
    def fact_store(self):
        store = FactStore()
        store._facts["test"] = make_test_fact()
        return store
    
    def test_confirms_matching_value(self, fact_store):
        """Lead with matching value should be confirmed."""
        lead = Lead(
            lead_id="test",
            text="Revenue of $35B",
            source_name="Reuters",
            found_at=datetime.now(),
            entity="NVDA",
            metric="revenue",
            value=35.0,
            value_raw=35_000_000_000,
            lead_type="quantitative",
        )
        
        result = verify_lead(lead, fact_store)
        
        assert result.verification_status == "confirmed"
    
    def test_contradicts_wrong_value(self, fact_store):
        """Lead with wrong value should be contradicted."""
        lead = Lead(
            lead_id="test",
            text="Revenue of $20B",
            source_name="SeekingAlpha",
            found_at=datetime.now(),
            entity="NVDA",
            metric="revenue",
            value=20.0,
            value_raw=20_000_000_000,
            lead_type="quantitative",
        )
        
        result = verify_lead(lead, fact_store)
        
        assert result.verification_status == "contradicted"
        assert "difference" in result.verification_details.lower()
    
    def test_unverifiable_no_matching_fact(self):
        """Lead with no matching fact should be unverifiable."""
        empty_store = FactStore()
        lead = Lead(
            lead_id="test",
            text="Revenue of $35B",
            source_name="Test",
            found_at=datetime.now(),
            entity="NVDA",
            metric="revenue",
            value_raw=35_000_000_000,
            lead_type="quantitative",
        )
        
        result = verify_lead(lead, empty_store)
        
        assert result.verification_status == "unverifiable"
    
    def test_qualitative_is_unverifiable(self, fact_store):
        """Qualitative leads can't be verified against numbers."""
        lead = Lead(
            lead_id="test",
            text="Blackwell delays expected",
            source_name="Twitter",
            found_at=datetime.now(),
            lead_type="qualitative",
        )
        
        result = verify_lead(lead, fact_store)
        
        assert result.verification_status == "unverifiable"
    
    def test_close_value_is_confirmed(self, fact_store):
        """Lead with value within 5% should be confirmed as close."""
        # 3% difference should still be confirmed
        lead = Lead(
            lead_id="test",
            text="Revenue of $34B",
            source_name="Reuters",
            found_at=datetime.now(),
            entity="NVDA",
            metric="revenue",
            value=34.0,
            value_raw=34_000_000_000,
            lead_type="quantitative",
        )
        
        result = verify_lead(lead, fact_store)
        
        assert result.verification_status == "confirmed"
        assert "rounding" in result.verification_details.lower()


class TestVerifyLeads:
    """Test batch verification."""
    
    def test_verifies_all_leads(self):
        """Should verify all leads in list."""
        store = FactStore()
        store._facts["test"] = make_test_fact()
        
        leads = [
            Lead(
                lead_id="1",
                text="Revenue $35B",
                source_name="A",
                found_at=datetime.now(),
                entity="NVDA",
                metric="revenue",
                value_raw=35_000_000_000,
                lead_type="quantitative",
            ),
            Lead(
                lead_id="2",
                text="Something qualitative",
                source_name="B",
                found_at=datetime.now(),
                lead_type="qualitative",
            ),
        ]
        
        results = verify_leads(leads, store)
        
        assert len(results) == 2
        assert results[0].verification_status == "confirmed"
        assert results[1].verification_status == "unverifiable"


class TestDiscoveryReport:
    """Test DiscoveryReport model."""
    
    def test_computes_summary(self):
        """Should compute summary statistics."""
        leads = [
            Lead(lead_id="1", text="Test", source_name="A", found_at=datetime.now(), verification_status="confirmed"),
            Lead(lead_id="2", text="Test", source_name="B", found_at=datetime.now(), verification_status="contradicted"),
            Lead(lead_id="3", text="Test", source_name="C", found_at=datetime.now(), verification_status="unverifiable"),
        ]
        
        report = DiscoveryReport(
            query="test",
            leads=leads,
            generated_at=datetime.now(),
        )
        
        assert report.total_leads == 3
        assert report.confirmed_count == 1
        assert report.contradicted_count == 1
        assert report.unverifiable_count == 1
    
    def test_format_shows_contradicted_first(self):
        """Contradicted leads should appear first in report."""
        leads = [
            Lead(lead_id="1", text="Confirmed claim", source_name="A", found_at=datetime.now(), verification_status="confirmed"),
            Lead(lead_id="2", text="Contradicted claim", source_name="B", found_at=datetime.now(), verification_status="contradicted", verification_details="Wrong!"),
        ]
        
        report = DiscoveryReport(
            query="test",
            leads=leads,
            generated_at=datetime.now(),
        )
        
        formatted = report.format_report()
        contradicted_pos = formatted.find("CONTRADICTED")
        confirmed_pos = formatted.find("CONFIRMED")
        
        assert contradicted_pos < confirmed_pos
    
    def test_pending_count(self):
        """Should count pending leads."""
        leads = [
            Lead(lead_id="1", text="Test", source_name="A", found_at=datetime.now(), verification_status="pending"),
            Lead(lead_id="2", text="Test", source_name="B", found_at=datetime.now(), verification_status="pending"),
        ]
        
        report = DiscoveryReport(
            query="test",
            leads=leads,
            generated_at=datetime.now(),
        )
        
        assert report.pending_count == 2

