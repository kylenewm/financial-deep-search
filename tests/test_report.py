"""
Tests for report generation module.
"""
import re
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.models import (
    Analysis,
    Conflict,
    ConflictingValue,
    Fact,
    Location,
    NotFoundMetric,
)
from open_deep_research.report import (
    format_value_with_unit,
    generate_citations_section,
    generate_conflicts_section,
    generate_facts_section,
    generate_full_report,
    generate_not_found_section,
    generate_thesis_section,
)
from open_deep_research.store import FactStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_location() -> Location:
    """Create a sample location for testing."""
    return Location(
        cik="0001045810",
        doc_date="2024-11-20",
        doc_type="10-Q",
        section_id="Item7",
        paragraph_index=5,
        sentence_string="Revenue for the third quarter of fiscal 2025 was $35,082 million",
    )


@pytest.fixture
def sample_table_location() -> Location:
    """Create a sample table location for testing."""
    return Location(
        cik="0001045810",
        doc_date="2024-11-20",
        doc_type="10-Q",
        section_id="Item7",
        table_index=3,
        row_index=2,
        column_index=1,
        row_label="Data Center",
        column_label="Oct 27, 2024",
    )


@pytest.fixture
def revenue_fact(sample_location: Location) -> Fact:
    """Create a sample revenue fact."""
    return Fact(
        fact_id="fact_001",
        entity="NVDA",
        metric="Revenue",
        value=35082.0,
        unit="USD millions",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=sample_location,
        source_format="html_text",
        doc_hash="abc123",
        snapshot_id="snap_001",
        verification_status="exact_match",
    )


@pytest.fixture
def datacenter_fact(sample_table_location: Location) -> Fact:
    """Create a sample data center revenue fact from table."""
    return Fact(
        fact_id="fact_002",
        entity="NVDA",
        metric="Data Center Revenue",
        value=14514.0,
        unit="USD millions",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=sample_table_location,
        source_format="html_table",
        extracted_scale="millions",
        doc_hash="abc123",
        snapshot_id="snap_001",
        verification_status="exact_match",
    )


@pytest.fixture
def net_income_fact() -> Fact:
    """Create a sample net income fact."""
    return Fact(
        fact_id="fact_003",
        entity="NVDA",
        metric="Net Income",
        value=9243.0,
        unit="USD millions",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=Location(
            cik="0001045810",
            doc_date="2024-11-20",
            doc_type="10-Q",
            section_id="Item7",
            paragraph_index=10,
            sentence_string="Net income was $9,243 million",
        ),
        source_format="html_text",
        doc_hash="abc123",
        snapshot_id="snap_001",
        verification_status="exact_match",
    )


@pytest.fixture
def eps_fact() -> Fact:
    """Create a sample EPS fact."""
    return Fact(
        fact_id="fact_004",
        entity="NVDA",
        metric="Earnings Per Share",
        value=0.81,
        unit="USD per share",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=Location(
            cik="0001045810",
            doc_date="2024-11-20",
            doc_type="10-Q",
            section_id="Item7",
            table_index=5,
            row_index=3,
            column_index=1,
            row_label="Diluted EPS",
            column_label="Oct 27, 2024",
        ),
        source_format="html_table",
        doc_hash="abc123",
        snapshot_id="snap_001",
        verification_status="exact_match",
    )


@pytest.fixture
def populated_store(revenue_fact: Fact, datacenter_fact: Fact, net_income_fact: Fact) -> FactStore:
    """Create a FactStore populated with sample facts."""
    store = FactStore()
    store.add_fact(revenue_fact)
    store.add_fact(datacenter_fact)
    store.add_fact(net_income_fact)
    return store


@pytest.fixture
def sample_conflict() -> Conflict:
    """Create a sample conflict."""
    return Conflict(
        entity="NVDA",
        metric="Revenue",
        period="Q3 FY2025",
        values=[
            ConflictingValue(value=35082.0, fact_id="fact_001", source_description="10-Q 2024-11-20"),
            ConflictingValue(value=35100.0, fact_id="fact_010", source_description="8-K 2024-11-21"),
        ],
    )


# =============================================================================
# format_value_with_unit Tests
# =============================================================================


class TestFormatValueWithUnit:
    """Tests for value formatting with units."""

    def test_format_usd_millions(self):
        """Test formatting USD millions."""
        result = format_value_with_unit(35082.0, "USD millions")
        assert result == "$35,082 million"

    def test_format_usd_billions(self):
        """Test formatting USD billions."""
        result = format_value_with_unit(35.0, "USD billions")
        assert result == "$35 billion"

    def test_format_usd_thousands(self):
        """Test formatting USD thousands."""
        result = format_value_with_unit(500.0, "USD thousands")
        assert result == "$500 thousand"

    def test_format_percentage(self):
        """Test formatting percentage."""
        result = format_value_with_unit(42.5, "percentage")
        assert result == "42.50%"

    def test_format_percentage_symbol(self):
        """Test formatting with % symbol unit."""
        result = format_value_with_unit(15.0, "%")
        assert result == "15.00%"

    def test_format_per_share(self):
        """Test formatting per share values."""
        result = format_value_with_unit(0.81, "USD per share")
        assert result == "$0.81"

    def test_format_ratio(self):
        """Test formatting ratio."""
        result = format_value_with_unit(1.25, "ratio")
        assert result == "1.25"

    def test_format_none_value(self):
        """Test formatting None value."""
        result = format_value_with_unit(None, "USD millions")
        assert result == "N/A"

    def test_format_default(self):
        """Test default formatting for unknown units."""
        result = format_value_with_unit(123.456, "widgets")
        assert result == "123.46 widgets"


# =============================================================================
# generate_facts_section Tests
# =============================================================================


class TestGenerateFactsSection:
    """Tests for facts section generation."""

    def test_all_facts_included(self, revenue_fact: Fact, datacenter_fact: Fact):
        """Test that all facts are included in output."""
        facts = [revenue_fact, datacenter_fact]
        result = generate_facts_section(facts)
        
        assert "Revenue" in result
        assert "Data Center Revenue" in result
        assert "$35,082 million" in result
        assert "$14,514 million" in result

    def test_citation_numbers_sequential(self, revenue_fact: Fact, datacenter_fact: Fact, net_income_fact: Fact):
        """Test that citation numbers are sequential [1], [2], [3]."""
        facts = [revenue_fact, datacenter_fact, net_income_fact]
        result = generate_facts_section(facts)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_values_formatted_correctly(self, revenue_fact: Fact):
        """Test that values are formatted with correct units."""
        facts = [revenue_fact]
        result = generate_facts_section(facts)
        
        # Should have dollar sign and "million"
        assert "$35,082 million" in result

    def test_period_included(self, revenue_fact: Fact):
        """Test that period is included."""
        facts = [revenue_fact]
        result = generate_facts_section(facts)
        
        assert "Q3 FY2025" in result

    def test_grouping_by_metric(self, revenue_fact: Fact, datacenter_fact: Fact, net_income_fact: Fact):
        """Test that facts are grouped by metric type."""
        facts = [revenue_fact, datacenter_fact, net_income_fact]
        result = generate_facts_section(facts)
        
        # Should have metric headers
        assert "**Revenue**" in result
        assert "**Data Center Revenue**" in result
        assert "**Net Income**" in result

    def test_empty_facts_list(self):
        """Test handling empty facts list."""
        result = generate_facts_section([])
        assert result == "No verified facts available."


# =============================================================================
# generate_thesis_section Tests
# =============================================================================


class TestGenerateThesisSection:
    """Tests for thesis/analysis section generation."""

    def test_returns_analysis_object(self, revenue_fact: Fact):
        """Test that Analysis object is returned."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "This is the analysis."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_thesis_section([revenue_fact], "What is NVDA's revenue?")
            
            assert isinstance(result, Analysis)

    def test_classification_is_thesis(self, revenue_fact: Fact):
        """Test that classification is always 'thesis'."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "This is the analysis."
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_thesis_section([revenue_fact], "Query")
            
            assert result.classification == "thesis"

    def test_supporting_facts_references_valid_ids(self, revenue_fact: Fact, datacenter_fact: Fact):
        """Test that supporting_facts references valid fact_ids."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis text"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            facts = [revenue_fact, datacenter_fact]
            result = generate_thesis_section(facts, "Query")
            
            assert "fact_001" in result.supporting_facts
            assert "fact_002" in result.supporting_facts

    def test_llm_called_with_proper_prompt(self, revenue_fact: Fact):
        """Test that LLM is called with proper prompt containing key elements."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis text"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            generate_thesis_section([revenue_fact], "What is NVDA's revenue?")
            
            # Check the prompt contains key elements
            call_args = mock_llm.invoke.call_args[0][0]
            prompt_content = call_args[0].content
            
            assert "interpretation" in prompt_content.lower()
            assert "verified facts" in prompt_content.lower()
            assert "What is NVDA's revenue?" in prompt_content
            assert "Do not introduce new factual claims" in prompt_content

    def test_empty_facts_returns_insufficient_data(self):
        """Test that empty facts list returns appropriate analysis."""
        result = generate_thesis_section([], "Query")
        
        assert result.summary == "Insufficient data for analysis."
        assert result.classification == "thesis"
        assert result.supporting_facts == []

    def test_uses_provided_llm(self, revenue_fact: Fact):
        """Test that provided LLM instance is used."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis from provided LLM"
        mock_llm.invoke.return_value = mock_response
        
        result = generate_thesis_section([revenue_fact], "Query", llm=mock_llm)
        
        mock_llm.invoke.assert_called_once()
        assert result.summary == "Analysis from provided LLM"


# =============================================================================
# generate_citations_section Tests
# =============================================================================


class TestGenerateCitationsSection:
    """Tests for citations section generation."""

    def test_each_fact_has_citation(self, revenue_fact: Fact, datacenter_fact: Fact):
        """Test that each fact gets a citation."""
        facts = [revenue_fact, datacenter_fact]
        result = generate_citations_section(facts)
        
        assert "[1]" in result
        assert "[2]" in result

    def test_citation_includes_doc_type(self, revenue_fact: Fact):
        """Test that citation includes doc_type."""
        result = generate_citations_section([revenue_fact])
        
        assert "10-Q" in result

    def test_citation_includes_doc_date(self, revenue_fact: Fact):
        """Test that citation includes doc_date."""
        result = generate_citations_section([revenue_fact])
        
        assert "2024-11-20" in result

    def test_citation_includes_section_id(self, revenue_fact: Fact):
        """Test that citation includes section_id."""
        result = generate_citations_section([revenue_fact])
        
        assert "Item7" in result

    def test_citation_includes_exact_quote_for_text(self, revenue_fact: Fact):
        """Test that citation includes exact quote for text facts."""
        result = generate_citations_section([revenue_fact])
        
        assert "Revenue for the third quarter of fiscal 2025 was $35,082 million" in result

    def test_citation_includes_table_location_for_table(self, datacenter_fact: Fact):
        """Test that citation includes table location for table facts."""
        result = generate_citations_section([datacenter_fact])
        
        assert "Table 3" in result
        assert "Data Center" in result

    def test_empty_facts_list(self):
        """Test handling empty facts list."""
        result = generate_citations_section([])
        assert result == "No sources to cite."


# =============================================================================
# generate_conflicts_section Tests
# =============================================================================


class TestGenerateConflictsSection:
    """Tests for conflicts section generation."""

    def test_no_conflicts(self):
        """Test output when no conflicts exist."""
        result = generate_conflicts_section(None)
        assert result == "No conflicts detected."
        
        result = generate_conflicts_section([])
        assert result == "No conflicts detected."

    def test_conflict_details_shown(self, sample_conflict: Conflict):
        """Test that conflict details are shown."""
        result = generate_conflicts_section([sample_conflict])
        
        assert "NVDA" in result
        assert "Revenue" in result
        assert "Q3 FY2025" in result
        assert "35,082" in result
        assert "35,100" in result
        assert "10-Q 2024-11-20" in result
        assert "8-K 2024-11-21" in result


# =============================================================================
# generate_not_found_section Tests
# =============================================================================


class TestGenerateNotFoundSection:
    """Tests for not found section generation."""

    def test_all_found(self):
        """Test output when all metrics were found."""
        result = generate_not_found_section(None)
        assert result == "All requested metrics were found."
        
        result = generate_not_found_section([])
        assert result == "All requested metrics were found."

    def test_missing_metrics_listed(self):
        """Test that missing metrics are listed."""
        result = generate_not_found_section(["Gross Margin", "Free Cash Flow"])
        
        assert "Gross Margin" in result
        assert "Free Cash Flow" in result
        assert "Not found in retrieved Tier 1/2 sources" in result


class TestNotFoundMetricModel:
    """Tests for NotFoundMetric model."""

    def test_default_status(self):
        """Test that default status is set correctly."""
        nf = NotFoundMetric(metric="Operating Expenses")
        
        assert nf.metric == "Operating Expenses"
        assert nf.status == "Not found in retrieved Tier 1/2 sources"

    def test_custom_status(self):
        """Test that custom status can be set."""
        nf = NotFoundMetric(metric="EBITDA", status="Custom status message")
        
        assert nf.status == "Custom status message"


# =============================================================================
# generate_full_report Tests
# =============================================================================


class TestGenerateFullReport:
    """Tests for full report generation."""

    def test_report_includes_all_sections_in_order(self, populated_store: FactStore):
        """Test report includes all sections in correct order."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis content"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "What is NVDA's financial performance?")
            
            # Check sections exist in order
            sections = [
                "# Research Report",
                "## Verified Facts",
                "## Analysis",
                "## Data Conflicts",
                "## Not Found",
                "## Sources",
            ]
            
            positions = [result.find(s) for s in sections]
            assert all(p >= 0 for p in positions), "All sections should be present"
            assert positions == sorted(positions), "Sections should be in order"

    def test_report_includes_query(self, populated_store: FactStore):
        """Test report includes query."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            query = "What is NVDA's Q3 FY2025 revenue?"
            result = generate_full_report(populated_store, query)
            
            assert f"**Query:** {query}" in result

    def test_report_includes_timestamp(self, populated_store: FactStore):
        """Test report includes timestamp."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query")
            
            assert "**Generated:**" in result
            # Check timestamp format (YYYY-MM-DD HH:MM:SS)
            assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_conflicts_section_appears_when_conflicts_exist(self, populated_store: FactStore, sample_conflict: Conflict):
        """Test conflicts section shows conflict details when conflicts exist."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query", conflicts=[sample_conflict])
            
            # Should show conflict details, not "No conflicts"
            assert "35,082" in result
            assert "35,100" in result
            # "No conflicts detected." should not appear in the conflicts section

    def test_conflicts_section_shows_no_conflicts_when_none(self, populated_store: FactStore):
        """Test conflicts section shows 'No conflicts' when none exist."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query", conflicts=None)
            
            assert "No conflicts detected." in result

    def test_not_found_section_appears_when_metrics_missing(self, populated_store: FactStore):
        """Test not found section appears when metrics are missing."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query", not_found=["Gross Margin", "EBITDA"])
            
            assert "Gross Margin" in result
            assert "EBITDA" in result

    def test_not_found_section_shows_all_found_when_none_missing(self, populated_store: FactStore):
        """Test not found section shows 'All found' when none missing."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query", not_found=None)
            
            assert "All requested metrics were found." in result

    def test_analysis_disclaimer_included(self, populated_store: FactStore):
        """Test that analysis section includes interpretation disclaimer."""
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(populated_store, "Query")
            
            assert "interpretation based on the verified facts" in result
            assert "not verified factual content" in result


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for report generation."""

    def test_report_with_single_fact(self, revenue_fact: Fact):
        """Test report with single fact."""
        store = FactStore()
        store.add_fact(revenue_fact)
        
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Single fact analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(store, "Query")
            
            assert "Revenue" in result
            assert "[1]" in result
            assert "Single fact analysis" in result

    def test_report_with_no_facts(self):
        """Test report with no facts (should still generate valid report)."""
        store = FactStore()
        
        result = generate_full_report(store, "What is NVDA's revenue?")
        
        # Should have structure even with no facts
        assert "# Research Report" in result
        assert "## Verified Facts" in result
        assert "No verified facts available." in result
        assert "## Analysis" in result
        assert "Insufficient data for analysis." in result
        assert "## Sources" in result
        assert "No sources to cite." in result

    def test_report_with_conflicts(self, revenue_fact: Fact, sample_conflict: Conflict):
        """Test report with conflicts."""
        store = FactStore()
        store.add_fact(revenue_fact)
        
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis with conflict"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(store, "Query", conflicts=[sample_conflict])
            
            assert "## Data Conflicts" in result
            assert "NVDA" in result
            assert "35,082" in result

    def test_report_with_not_found_metrics(self, revenue_fact: Fact):
        """Test report with not_found metrics."""
        store = FactStore()
        store.add_fact(revenue_fact)
        
        with patch("open_deep_research.report.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            result = generate_full_report(
                store, 
                "Query", 
                not_found=["Gross Margin", "Operating Expenses"]
            )
            
            assert "## Not Found" in result
            assert "Gross Margin" in result
            assert "Operating Expenses" in result
            assert "Not found in retrieved Tier 1/2 sources" in result


# =============================================================================
# Analysis Model Tests
# =============================================================================


class TestAnalysisModel:
    """Tests for Analysis Pydantic model."""

    def test_analysis_default_classification(self):
        """Test that Analysis has default classification of 'thesis'."""
        analysis = Analysis(
            summary="Test summary",
            supporting_facts=["fact_001"]
        )
        
        assert analysis.classification == "thesis"

    def test_analysis_all_fields(self):
        """Test Analysis with all fields."""
        analysis = Analysis(
            summary="Detailed analysis text",
            classification="thesis",
            supporting_facts=["fact_001", "fact_002", "fact_003"]
        )
        
        assert analysis.summary == "Detailed analysis text"
        assert analysis.classification == "thesis"
        assert len(analysis.supporting_facts) == 3

    def test_analysis_empty_supporting_facts(self):
        """Test Analysis with empty supporting facts."""
        analysis = Analysis(
            summary="No supporting data",
            supporting_facts=[]
        )
        
        assert analysis.supporting_facts == []

