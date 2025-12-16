"""
Tests for structured output module.
"""
import csv
import io
import json
from datetime import datetime

import pytest

from open_deep_research.models import (
    Analysis,
    Conflict,
    ConflictingValue,
    Fact,
    FactContext,
    Location,
    NotFoundMetric,
    ResearchOutput,
)
from open_deep_research.output import (
    generate_research_output,
    output_to_csv,
    output_to_dict,
    output_to_json,
    output_to_markdown,
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
        paragraph_index=3,
        sentence_string="Data Center revenue was $14,514 million",
    )


@pytest.fixture
def sample_fact(sample_location: Location) -> Fact:
    """Create a sample verified fact for testing."""
    return Fact(
        fact_id="abc-123",
        entity="NVDA",
        metric="datacenter_revenue",
        value=14514000000.0,
        unit="USD",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=sample_location,
        source_format="html_text",
        doc_hash="sha256abc",
        snapshot_id="snap-456",
        verification_status="exact_match",
        context=FactContext(yoy_change="+112%"),
    )


@pytest.fixture
def sample_fact_2(sample_location: Location) -> Fact:
    """Create a second sample verified fact for testing."""
    loc2 = sample_location.model_copy(
        update={"sentence_string": "Total revenue was $35,082 million"}
    )
    return Fact(
        fact_id="def-456",
        entity="NVDA",
        metric="total_revenue",
        value=35082000000.0,
        unit="USD",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=loc2,
        source_format="html_text",
        doc_hash="sha256abc",
        snapshot_id="snap-456",
        verification_status="exact_match",
    )


@pytest.fixture
def sample_fact_different_source(sample_location: Location) -> Fact:
    """Create a fact from a different source for testing deduplication."""
    loc3 = sample_location.model_copy(
        update={
            "doc_date": "2024-08-28",
            "sentence_string": "Gaming revenue was $2,880 million",
        }
    )
    return Fact(
        fact_id="ghi-789",
        entity="NVDA",
        metric="gaming_revenue",
        value=2880000000.0,
        unit="USD",
        period="Q2 FY2025",
        period_end_date="2024-07-28",
        location=loc3,
        source_format="html_text",
        doc_hash="sha256def",
        snapshot_id="snap-789",
        verification_status="approximate_match",
    )


@pytest.fixture
def populated_store(
    sample_fact: Fact, sample_fact_2: Fact, sample_fact_different_source: Fact
) -> FactStore:
    """Create a FactStore with multiple facts."""
    store = FactStore()
    store.add_fact(sample_fact)
    store.add_fact(sample_fact_2)
    store.add_fact(sample_fact_different_source)
    return store


@pytest.fixture
def sample_analysis() -> Analysis:
    """Create a sample analysis for testing."""
    return Analysis(
        summary="NVIDIA's datacenter segment showed strong growth driven by AI infrastructure demand.",
        classification="thesis",
        supporting_facts=["abc-123", "def-456"],
    )


@pytest.fixture
def sample_conflicts() -> list[Conflict]:
    """Create sample conflicts for testing."""
    return [
        Conflict(
            entity="NVDA",
            metric="gross_margin",
            period="Q3 FY2025",
            values=[
                ConflictingValue(
                    value=75.0, fact_id="conflict-1", source_description="10-Q 2024-11-20"
                ),
                ConflictingValue(
                    value=72.5, fact_id="conflict-2", source_description="8-K 2024-11-15"
                ),
            ],
        )
    ]


# =============================================================================
# ResearchOutput Creation Tests
# =============================================================================


class TestResearchOutputCreation:
    """Tests for ResearchOutput model creation and computed fields."""

    def test_creation_with_minimal_data(self):
        """Test creation with just query and empty facts."""
        output = ResearchOutput(
            query="What was NVIDIA's revenue?",
            generated_at=datetime.now(),
            facts=[],
        )
        
        assert output.query == "What was NVIDIA's revenue?"
        assert output.facts == []
        assert output.total_facts == 0
        assert output.verified_facts == 0
        assert output.sources_used == []

    def test_creation_with_full_data(
        self,
        sample_fact: Fact,
        sample_analysis: Analysis,
        sample_conflicts: list[Conflict],
    ):
        """Test creation with full data including analysis and conflicts."""
        not_found = [NotFoundMetric(metric="operating_income")]
        
        output = ResearchOutput(
            query="What was NVIDIA's datacenter revenue?",
            generated_at=datetime.now(),
            as_of_date="2024-12-01",
            facts=[sample_fact],
            analysis=sample_analysis,
            conflicts=sample_conflicts,
            not_found=not_found,
        )
        
        assert output.query == "What was NVIDIA's datacenter revenue?"
        assert output.as_of_date == "2024-12-01"
        assert len(output.facts) == 1
        assert output.analysis == sample_analysis
        assert len(output.conflicts) == 1
        assert len(output.not_found) == 1

    def test_total_facts_calculated_correctly(
        self, sample_fact: Fact, sample_fact_2: Fact
    ):
        """Test that total_facts is calculated from facts list."""
        output = ResearchOutput(
            query="Test query",
            generated_at=datetime.now(),
            facts=[sample_fact, sample_fact_2],
        )
        
        assert output.total_facts == 2

    def test_verified_facts_count_is_correct(
        self, sample_fact: Fact, sample_fact_different_source: Fact, sample_location: Location
    ):
        """Test that verified_facts counts only exact_match and approximate_match."""
        # Create an unverified fact
        unverified_fact = Fact(
            fact_id="unverified-1",
            entity="NVDA",
            metric="some_metric",
            value=1000.0,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_text",
            doc_hash="sha256xyz",
            snapshot_id="snap-999",
            verification_status="mismatch",
        )
        
        output = ResearchOutput(
            query="Test query",
            generated_at=datetime.now(),
            facts=[sample_fact, sample_fact_different_source, unverified_fact],
        )
        
        # exact_match + approximate_match = 2, mismatch doesn't count
        assert output.verified_facts == 2
        assert output.total_facts == 3

    def test_sources_used_is_deduplicated(
        self, sample_fact: Fact, sample_fact_2: Fact, sample_fact_different_source: Fact
    ):
        """Test that sources_used contains unique snapshot_ids only."""
        # sample_fact and sample_fact_2 share snap-456
        # sample_fact_different_source uses snap-789
        output = ResearchOutput(
            query="Test query",
            generated_at=datetime.now(),
            facts=[sample_fact, sample_fact_2, sample_fact_different_source],
        )
        
        assert len(output.sources_used) == 2
        assert set(output.sources_used) == {"snap-456", "snap-789"}


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJsonSerialization:
    """Tests for output_to_json function."""

    def test_output_to_json_produces_valid_json(self, populated_store: FactStore):
        """Test that output_to_json produces valid JSON."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output)
        
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_datetime_serialized_correctly(self, populated_store: FactStore):
        """Test datetime is serialized in ISO format."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output)
        parsed = json.loads(json_str)
        
        # generated_at should be ISO format string
        generated_at = parsed["generated_at"]
        assert isinstance(generated_at, str)
        # Should be parseable as ISO format
        datetime.fromisoformat(generated_at)

    def test_pretty_printing_works(self, populated_store: FactStore):
        """Test pretty printing produces indented output."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output, pretty=True)
        
        # Should have newlines and indentation
        assert "\n" in json_str
        assert "  " in json_str  # 2-space indent

    def test_compact_mode_works(self, populated_store: FactStore):
        """Test non-pretty mode produces compact output."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output, pretty=False)
        
        # Should not have newlines (except maybe within string values)
        # At minimum, no indentation spaces after colons
        assert ": " not in json_str or '": "' not in json_str.split("\n")[0]
        # Compact format uses ": " so we just verify it's more compact
        pretty_str = output_to_json(output, pretty=True)
        assert len(json_str) < len(pretty_str)

    def test_can_be_parsed_back_with_json_loads(self, populated_store: FactStore):
        """Test JSON can be parsed back to dict structure."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output)
        parsed = json.loads(json_str)
        
        assert parsed["query"] == "Test query"
        assert len(parsed["facts"]) == 3
        assert parsed["total_facts"] == 3
        assert parsed["verified_facts"] == 3


# =============================================================================
# Markdown Generation Tests
# =============================================================================


class TestMarkdownGeneration:
    """Tests for output_to_markdown function."""

    def test_output_to_markdown_produces_valid_markdown(
        self, populated_store: FactStore
    ):
        """Test that output_to_markdown produces valid markdown."""
        output = generate_research_output(populated_store, "Test query")
        md = output_to_markdown(output)
        
        # Should be a string with markdown headers
        assert isinstance(md, str)
        assert "# Research Report" in md

    def test_all_sections_included(self, populated_store: FactStore):
        """Test all sections are present in markdown output."""
        output = generate_research_output(populated_store, "Test query")
        md = output_to_markdown(output)
        
        assert "## Verified Facts" in md
        assert "## Analysis" in md
        assert "## Data Conflicts" in md
        assert "## Not Found" in md
        assert "## Sources" in md

    def test_facts_formatted_correctly(self, populated_store: FactStore):
        """Test facts appear in markdown output."""
        output = generate_research_output(populated_store, "Test query")
        md = output_to_markdown(output)
        
        # Facts should be included
        assert "datacenter_revenue" in md
        assert "total_revenue" in md
        assert "gaming_revenue" in md

    def test_query_included(self, populated_store: FactStore):
        """Test query is shown in markdown output."""
        output = generate_research_output(populated_store, "What was NVIDIA's revenue?")
        md = output_to_markdown(output)
        
        assert "What was NVIDIA's revenue?" in md

    def test_as_of_date_shown_when_present(self, sample_fact: Fact):
        """Test as_of_date appears in markdown when provided."""
        store = FactStore()
        store.add_fact(sample_fact)
        output = generate_research_output(store, "Test query", as_of_date="2024-12-01")
        md = output_to_markdown(output)
        
        assert "As Of Date:" in md
        assert "2024-12-01" in md


# =============================================================================
# CSV Export Tests
# =============================================================================


class TestCsvExport:
    """Tests for output_to_csv function."""

    def test_output_to_csv_produces_valid_csv(self, populated_store: FactStore):
        """Test that output_to_csv produces valid CSV."""
        output = generate_research_output(populated_store, "Test query")
        csv_str = output_to_csv(output)
        
        # Should be parseable by csv.reader
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        
        # Header + 3 data rows
        assert len(rows) == 4

    def test_all_facts_included_as_rows(self, populated_store: FactStore):
        """Test all facts appear as rows in CSV."""
        output = generate_research_output(populated_store, "Test query")
        csv_str = output_to_csv(output)
        
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        
        # Skip header, should have 3 fact rows
        fact_rows = rows[1:]
        assert len(fact_rows) == 3

    def test_columns_are_correct(self, populated_store: FactStore):
        """Test CSV has correct column headers."""
        output = generate_research_output(populated_store, "Test query")
        csv_str = output_to_csv(output)
        
        reader = csv.reader(io.StringIO(csv_str))
        headers = next(reader)
        
        expected_headers = [
            "entity",
            "metric",
            "value",
            "unit",
            "period",
            "period_end_date",
            "source_doc",
            "source_date",
        ]
        assert headers == expected_headers

    def test_can_be_parsed_by_csv_reader(self, populated_store: FactStore):
        """Test CSV can be fully parsed and values are accessible."""
        output = generate_research_output(populated_store, "Test query")
        csv_str = output_to_csv(output)
        
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        
        # Find the datacenter_revenue row
        dc_row = next(r for r in rows if r["metric"] == "datacenter_revenue")
        
        assert dc_row["entity"] == "NVDA"
        assert float(dc_row["value"]) == 14514000000.0
        assert dc_row["unit"] == "USD"
        assert dc_row["period"] == "Q3 FY2025"
        assert dc_row["period_end_date"] == "2024-10-27"
        assert dc_row["source_doc"] == "10-Q"
        assert dc_row["source_date"] == "2024-11-20"


# =============================================================================
# Dict Conversion Tests
# =============================================================================


class TestDictConversion:
    """Tests for output_to_dict function."""

    def test_output_to_dict_returns_dict(self, populated_store: FactStore):
        """Test output_to_dict returns a dict."""
        output = generate_research_output(populated_store, "Test query")
        result = output_to_dict(output)
        
        assert isinstance(result, dict)

    def test_structure_matches_research_output_fields(
        self, populated_store: FactStore, sample_analysis: Analysis
    ):
        """Test dict structure matches ResearchOutput fields."""
        output = generate_research_output(
            populated_store, "Test query", analysis=sample_analysis
        )
        result = output_to_dict(output)
        
        # All fields should be present
        assert "query" in result
        assert "generated_at" in result
        assert "as_of_date" in result
        assert "facts" in result
        assert "analysis" in result
        assert "conflicts" in result
        assert "not_found" in result
        assert "total_facts" in result
        assert "verified_facts" in result
        assert "sources_used" in result

    def test_nested_structures_preserved(
        self, populated_store: FactStore, sample_analysis: Analysis
    ):
        """Test nested structures like facts and analysis are preserved."""
        output = generate_research_output(
            populated_store, "Test query", analysis=sample_analysis
        )
        result = output_to_dict(output)
        
        # Facts should be list of dicts
        assert isinstance(result["facts"], list)
        assert len(result["facts"]) == 3
        assert isinstance(result["facts"][0], dict)
        
        # Analysis should be a dict
        assert isinstance(result["analysis"], dict)
        assert result["analysis"]["summary"] == sample_analysis.summary


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for serialization/deserialization round-trips."""

    def test_json_can_be_parsed_back_to_dict(self, populated_store: FactStore):
        """Test JSON round-trip preserves data."""
        output = generate_research_output(populated_store, "Test query")
        json_str = output_to_json(output)
        parsed = json.loads(json_str)
        
        assert parsed["query"] == output.query
        assert len(parsed["facts"]) == len(output.facts)
        assert parsed["total_facts"] == output.total_facts

    def test_structure_matches_original(
        self, populated_store: FactStore, sample_analysis: Analysis
    ):
        """Test round-trip structure matches original output."""
        output = generate_research_output(
            populated_store, "Test query", analysis=sample_analysis
        )
        
        # Convert to dict directly
        direct_dict = output_to_dict(output)
        
        # Convert via JSON
        json_str = output_to_json(output)
        json_dict = json.loads(json_str)
        
        # Key structure should match (datetime format may differ)
        assert direct_dict["query"] == json_dict["query"]
        assert len(direct_dict["facts"]) == len(json_dict["facts"])
        assert direct_dict["total_facts"] == json_dict["total_facts"]
        assert direct_dict["verified_facts"] == json_dict["verified_facts"]


# =============================================================================
# generate_research_output Tests
# =============================================================================


class TestGenerateResearchOutput:
    """Tests for the generate_research_output function."""

    def test_creates_output_from_store(self, populated_store: FactStore):
        """Test basic output creation from store."""
        output = generate_research_output(populated_store, "Test query")
        
        assert output.query == "Test query"
        assert len(output.facts) == 3
        assert output.total_facts == 3

    def test_includes_analysis_when_provided(
        self, populated_store: FactStore, sample_analysis: Analysis
    ):
        """Test analysis is included when provided."""
        output = generate_research_output(
            populated_store, "Test query", analysis=sample_analysis
        )
        
        assert output.analysis is not None
        assert output.analysis.summary == sample_analysis.summary

    def test_includes_conflicts_when_provided(
        self, populated_store: FactStore, sample_conflicts: list[Conflict]
    ):
        """Test conflicts are included when provided."""
        output = generate_research_output(
            populated_store, "Test query", conflicts=sample_conflicts
        )
        
        assert len(output.conflicts) == 1
        assert output.conflicts[0].entity == "NVDA"

    def test_converts_not_found_strings_to_objects(self, populated_store: FactStore):
        """Test not_found strings are converted to NotFoundMetric objects."""
        output = generate_research_output(
            populated_store, "Test query", not_found=["operating_income", "net_income"]
        )
        
        assert len(output.not_found) == 2
        assert all(isinstance(nf, NotFoundMetric) for nf in output.not_found)
        assert output.not_found[0].metric == "operating_income"
        assert output.not_found[1].metric == "net_income"

    def test_as_of_date_passed_through(self, populated_store: FactStore):
        """Test as_of_date is passed through to output."""
        output = generate_research_output(
            populated_store, "Test query", as_of_date="2024-12-01"
        )
        
        assert output.as_of_date == "2024-12-01"

    def test_empty_store_produces_valid_output(self):
        """Test empty store produces valid output with zero facts."""
        store = FactStore()
        output = generate_research_output(store, "Test query")
        
        assert output.facts == []
        assert output.total_facts == 0
        assert output.verified_facts == 0
        assert output.sources_used == []

