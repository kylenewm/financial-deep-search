"""
Tests for FactStore - the single source of truth for verified facts.
"""
import json
import pytest

from open_deep_research.models import Conflict, ConflictingValue, Fact, Location
from open_deep_research.store import FactStore


# =============================================================================
# Fixtures
# =============================================================================


def make_location(
    doc_type: str = "10-K",
    doc_date: str = "2024-01-28",
    section_id: str = "Item7",
    paragraph_index: int = 0,
    sentence_string: str = "Revenue was $10 billion.",
) -> Location:
    """Create a Location for testing."""
    return Location(
        cik="0001045810",
        doc_date=doc_date,
        doc_type=doc_type,
        section_id=section_id,
        paragraph_index=paragraph_index,
        sentence_string=sentence_string,
    )


def make_fact(
    fact_id: str = "fact_001",
    entity: str = "NVDA",
    metric: str = "Revenue",
    value: float = 10_000_000_000.0,
    unit: str = "USD",
    period: str = "Q3 FY2025",
    period_end_date: str = "2024-10-27",
    verification_status: str = "exact_match",
    doc_type: str = "10-K",
    doc_date: str = "2024-01-28",
) -> Fact:
    """Create a Fact for testing."""
    return Fact(
        fact_id=fact_id,
        entity=entity,
        metric=metric,
        value=value,
        unit=unit,
        period=period,
        period_end_date=period_end_date,
        location=make_location(doc_type=doc_type, doc_date=doc_date),
        source_format="html_text",
        doc_hash="abc123",
        snapshot_id="snap_001",
        verification_status=verification_status,
    )


@pytest.fixture
def store() -> FactStore:
    """Create an empty FactStore."""
    return FactStore()


@pytest.fixture
def verified_fact() -> Fact:
    """Create a verified fact with exact_match status."""
    return make_fact(verification_status="exact_match")


@pytest.fixture
def approximate_fact() -> Fact:
    """Create a verified fact with approximate_match status."""
    return make_fact(fact_id="fact_002", verification_status="approximate_match")


# =============================================================================
# Add Fact Tests
# =============================================================================


class TestAddFact:
    """Tests for adding facts to the store."""
    
    def test_add_exact_match_succeeds(self, store: FactStore) -> None:
        """Test adding fact with exact_match status succeeds."""
        fact = make_fact(verification_status="exact_match")
        store.add_fact(fact)
        assert len(store) == 1
        assert store.get_fact(fact.fact_id) == fact
    
    def test_add_approximate_match_succeeds(self, store: FactStore) -> None:
        """Test adding fact with approximate_match status succeeds."""
        fact = make_fact(verification_status="approximate_match")
        store.add_fact(fact)
        assert len(store) == 1
        assert store.get_fact(fact.fact_id) == fact
    
    def test_add_mismatch_raises_error(self, store: FactStore) -> None:
        """Test adding fact with mismatch status raises ValueError."""
        fact = make_fact(verification_status="mismatch")
        with pytest.raises(ValueError) as exc_info:
            store.add_fact(fact)
        assert "mismatch" in str(exc_info.value)
        assert "Only verified facts" in str(exc_info.value)
    
    def test_add_unverified_raises_error(self, store: FactStore) -> None:
        """Test adding fact with unverified status raises ValueError."""
        fact = make_fact(verification_status="unverified")
        with pytest.raises(ValueError) as exc_info:
            store.add_fact(fact)
        assert "unverified" in str(exc_info.value)
        assert "Only verified facts" in str(exc_info.value)
    
    def test_error_message_is_helpful(self, store: FactStore) -> None:
        """Test that error message explains what statuses are allowed."""
        fact = make_fact(verification_status="pending")
        with pytest.raises(ValueError) as exc_info:
            store.add_fact(fact)
        error_msg = str(exc_info.value)
        assert "pending" in error_msg
        assert "exact_match" in error_msg or "approximate_match" in error_msg
    
    def test_add_multiple_facts(self, store: FactStore) -> None:
        """Test adding multiple facts to the store."""
        fact1 = make_fact(fact_id="fact_001", verification_status="exact_match")
        fact2 = make_fact(fact_id="fact_002", verification_status="approximate_match")
        store.add_fact(fact1)
        store.add_fact(fact2)
        assert len(store) == 2
    
    def test_add_overwrites_existing_fact_with_same_id(self, store: FactStore) -> None:
        """Test that adding a fact with same ID overwrites the existing one."""
        fact1 = make_fact(fact_id="fact_001", value=100.0, verification_status="exact_match")
        fact2 = make_fact(fact_id="fact_001", value=200.0, verification_status="exact_match")
        store.add_fact(fact1)
        store.add_fact(fact2)
        assert len(store) == 1
        assert store.get_fact("fact_001").value == 200.0


# =============================================================================
# Retrieval Tests
# =============================================================================


class TestRetrieval:
    """Tests for retrieving facts from the store."""
    
    def test_get_fact_by_id(self, store: FactStore) -> None:
        """Test get_fact by ID returns correct fact."""
        fact = make_fact(fact_id="unique_id_123")
        store.add_fact(fact)
        retrieved = store.get_fact("unique_id_123")
        assert retrieved == fact
    
    def test_get_fact_unknown_id_returns_none(self, store: FactStore) -> None:
        """Test get_fact with unknown ID returns None."""
        fact = make_fact(fact_id="exists")
        store.add_fact(fact)
        assert store.get_fact("does_not_exist") is None
    
    def test_get_facts_by_entity(self, store: FactStore) -> None:
        """Test get_facts_by_entity returns all matching facts."""
        fact1 = make_fact(fact_id="f1", entity="NVDA")
        fact2 = make_fact(fact_id="f2", entity="NVDA", metric="Net Income")
        fact3 = make_fact(fact_id="f3", entity="AAPL")
        store.add_fact(fact1)
        store.add_fact(fact2)
        store.add_fact(fact3)
        
        nvda_facts = store.get_facts_by_entity("NVDA")
        assert len(nvda_facts) == 2
        assert all(f.entity == "NVDA" for f in nvda_facts)
    
    def test_get_facts_by_entity_case_insensitive(self, store: FactStore) -> None:
        """Test get_facts_by_entity is case-insensitive."""
        fact = make_fact(fact_id="f1", entity="NVDA")
        store.add_fact(fact)
        
        # Should match regardless of case
        assert len(store.get_facts_by_entity("nvda")) == 1
        assert len(store.get_facts_by_entity("Nvda")) == 1
        assert len(store.get_facts_by_entity("NVDA")) == 1
    
    def test_get_facts_by_metric(self, store: FactStore) -> None:
        """Test get_facts_by_metric returns correct facts."""
        fact1 = make_fact(fact_id="f1", metric="Revenue")
        fact2 = make_fact(fact_id="f2", metric="Revenue", entity="AAPL")
        fact3 = make_fact(fact_id="f3", metric="Net Income")
        store.add_fact(fact1)
        store.add_fact(fact2)
        store.add_fact(fact3)
        
        revenue_facts = store.get_facts_by_metric("Revenue")
        assert len(revenue_facts) == 2
        assert all(f.metric == "Revenue" for f in revenue_facts)
    
    def test_get_facts_by_metric_case_insensitive(self, store: FactStore) -> None:
        """Test get_facts_by_metric is case-insensitive."""
        fact = make_fact(fact_id="f1", metric="Net Income")
        store.add_fact(fact)
        
        # Should match regardless of case
        assert len(store.get_facts_by_metric("net income")) == 1
        assert len(store.get_facts_by_metric("NET INCOME")) == 1
        assert len(store.get_facts_by_metric("Net Income")) == 1
    
    def test_get_facts_by_period(self, store: FactStore) -> None:
        """Test get_facts_by_period returns correct facts."""
        fact1 = make_fact(fact_id="f1", period="Q3 FY2025")
        fact2 = make_fact(fact_id="f2", period="Q3 FY2025", metric="Net Income")
        fact3 = make_fact(fact_id="f3", period="Q4 FY2025")
        store.add_fact(fact1)
        store.add_fact(fact2)
        store.add_fact(fact3)
        
        q3_facts = store.get_facts_by_period("Q3 FY2025")
        assert len(q3_facts) == 2
        assert all(f.period == "Q3 FY2025" for f in q3_facts)
    
    def test_get_all_facts(self, store: FactStore) -> None:
        """Test get_all_facts returns all facts."""
        facts = [
            make_fact(fact_id=f"f{i}", metric=f"Metric{i}")
            for i in range(5)
        ]
        for fact in facts:
            store.add_fact(fact)
        
        all_facts = store.get_all_facts()
        assert len(all_facts) == 5
        assert set(f.fact_id for f in all_facts) == {f"f{i}" for i in range(5)}


# =============================================================================
# Conflict Detection Tests
# =============================================================================


class TestConflictDetection:
    """Tests for conflict detection between facts."""
    
    def test_no_conflicts_when_values_match(self, store: FactStore) -> None:
        """Test no conflicts when all values match (within 1%)."""
        # Two facts with same entity/metric/period and identical values
        fact1 = make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0, doc_type="10-K", doc_date="2024-01-28"
        )
        fact2 = make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0, doc_type="10-Q", doc_date="2024-11-20"
        )
        store.add_fact(fact1)
        store.add_fact(fact2)
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 0
    
    def test_conflict_detected_when_values_differ(self, store: FactStore) -> None:
        """Test conflict detected when same entity+metric+period has values differing by >1%."""
        # Two facts with same entity/metric/period but 10% different values
        fact1 = make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0, doc_type="10-K", doc_date="2024-01-28"
        )
        fact2 = make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=11_000_000_000.0, doc_type="10-Q", doc_date="2024-11-20"
        )
        store.add_fact(fact1)
        store.add_fact(fact2)
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].entity == "NVDA"
        assert conflicts[0].metric == "revenue"  # normalized to lowercase
        assert conflicts[0].period == "Q3 FY2025"
    
    def test_no_conflict_within_tolerance(self, store: FactStore) -> None:
        """Test conflict not detected when difference is within 1%."""
        # Two facts with same entity/metric/period, values within 1%
        fact1 = make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        )
        fact2 = make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_050_000_000.0  # 0.5% difference
        )
        store.add_fact(fact1)
        store.add_fact(fact2)
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 0
    
    def test_multiple_conflicts_detected(self, store: FactStore) -> None:
        """Test multiple conflicts can be detected."""
        # First conflict: Revenue
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=15_000_000_000.0  # 50% different
        ))
        
        # Second conflict: Net Income
        store.add_fact(make_fact(
            fact_id="f3", entity="NVDA", metric="Net Income", period="Q3 FY2025",
            value=5_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f4", entity="NVDA", metric="Net Income", period="Q3 FY2025",
            value=6_000_000_000.0  # 20% different
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 2
    
    def test_conflict_includes_all_conflicting_values(self, store: FactStore) -> None:
        """Test conflict includes all conflicting values."""
        # Three facts with same key but different values
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0, doc_type="10-K", doc_date="2024-01-28"
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=11_000_000_000.0, doc_type="10-Q", doc_date="2024-04-28"
        ))
        store.add_fact(make_fact(
            fact_id="f3", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=12_000_000_000.0, doc_type="8-K", doc_date="2024-07-28"
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 1
        assert len(conflicts[0].values) == 3
        
        # Check all values are included
        values = {v.value for v in conflicts[0].values}
        assert values == {10_000_000_000.0, 11_000_000_000.0, 12_000_000_000.0}
        
        # Check source descriptions are included
        descriptions = {v.source_description for v in conflicts[0].values}
        assert "10-K 2024-01-28" in descriptions
        assert "10-Q 2024-04-28" in descriptions
        assert "8-K 2024-07-28" in descriptions
    
    def test_no_conflict_different_entities(self, store: FactStore) -> None:
        """Test no conflict when entities differ."""
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="AAPL", metric="Revenue", period="Q3 FY2025",
            value=50_000_000_000.0
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 0
    
    def test_no_conflict_different_periods(self, store: FactStore) -> None:
        """Test no conflict when periods differ."""
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q4 FY2025",
            value=15_000_000_000.0
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 0
    
    def test_conflict_entity_case_insensitive(self, store: FactStore) -> None:
        """Test conflict detection is case-insensitive for entity."""
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="nvda", metric="Revenue", period="Q3 FY2025",
            value=15_000_000_000.0
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 1
    
    def test_conflict_metric_case_insensitive(self, store: FactStore) -> None:
        """Test conflict detection is case-insensitive for metric."""
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        store.add_fact(make_fact(
            fact_id="f2", entity="NVDA", metric="revenue", period="Q3 FY2025",
            value=15_000_000_000.0
        ))
        
        conflicts = store.find_conflicts()
        assert len(conflicts) == 1


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for JSON serialization/deserialization."""
    
    def test_to_json_produces_valid_json(self, store: FactStore) -> None:
        """Test to_json produces valid JSON."""
        store.add_fact(make_fact(fact_id="f1"))
        store.add_fact(make_fact(fact_id="f2", metric="Net Income"))
        
        json_str = store.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
    
    def test_from_json_reconstructs_store(self, store: FactStore) -> None:
        """Test from_json reconstructs store correctly."""
        original_fact = make_fact(fact_id="f1", value=12345.0)
        store.add_fact(original_fact)
        
        json_str = store.to_json()
        new_store = FactStore.from_json(json_str)
        
        assert len(new_store) == 1
        retrieved = new_store.get_fact("f1")
        assert retrieved is not None
        assert retrieved.value == 12345.0
    
    def test_round_trip_preserves_facts(self, store: FactStore) -> None:
        """Test round-trip: to_json -> from_json preserves all facts."""
        facts = [
            make_fact(fact_id="f1", entity="NVDA", metric="Revenue"),
            make_fact(fact_id="f2", entity="AAPL", metric="Net Income"),
            make_fact(fact_id="f3", entity="MSFT", metric="EPS", verification_status="approximate_match"),
        ]
        for fact in facts:
            store.add_fact(fact)
        
        json_str = store.to_json()
        restored = FactStore.from_json(json_str)
        
        assert len(restored) == 3
        
        for original in facts:
            retrieved = restored.get_fact(original.fact_id)
            assert retrieved is not None
            assert retrieved.entity == original.entity
            assert retrieved.metric == original.metric
            assert retrieved.value == original.value
            assert retrieved.verification_status == original.verification_status
    
    def test_len_works_correctly(self, store: FactStore) -> None:
        """Test len() works correctly."""
        assert len(store) == 0
        
        store.add_fact(make_fact(fact_id="f1"))
        assert len(store) == 1
        
        store.add_fact(make_fact(fact_id="f2"))
        assert len(store) == 2
    
    def test_from_json_bypasses_validation(self) -> None:
        """Test from_json bypasses verification status check (for loading)."""
        # Create JSON with a fact that has mismatch status
        # (would normally be rejected by add_fact)
        fact_dict = make_fact(
            fact_id="f1",
            verification_status="mismatch"  # Would fail add_fact
        ).model_dump()
        
        json_str = json.dumps([fact_dict], default=str)
        
        # Should load without error
        store = FactStore.from_json(json_str)
        assert len(store) == 1
        assert store.get_fact("f1").verification_status == "mismatch"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_store_get_fact_returns_none(self, store: FactStore) -> None:
        """Test empty store get_fact returns None."""
        assert store.get_fact("any_id") is None
    
    def test_empty_store_get_all_facts_returns_empty(self, store: FactStore) -> None:
        """Test empty store get_all_facts returns empty list."""
        assert store.get_all_facts() == []
    
    def test_empty_store_find_conflicts_returns_empty(self, store: FactStore) -> None:
        """Test empty store find_conflicts returns empty list."""
        assert store.find_conflicts() == []
    
    def test_empty_store_to_json_returns_empty_array(self, store: FactStore) -> None:
        """Test empty store to_json returns empty JSON array."""
        json_str = store.to_json()
        assert json.loads(json_str) == []
    
    def test_empty_store_len_is_zero(self, store: FactStore) -> None:
        """Test empty store len is zero."""
        assert len(store) == 0
    
    def test_adding_multiple_facts_for_same_entity(self, store: FactStore) -> None:
        """Test adding multiple facts for same entity."""
        facts = [
            make_fact(fact_id="f1", entity="NVDA", metric="Revenue"),
            make_fact(fact_id="f2", entity="NVDA", metric="Net Income"),
            make_fact(fact_id="f3", entity="NVDA", metric="EPS"),
        ]
        for fact in facts:
            store.add_fact(fact)
        
        nvda_facts = store.get_facts_by_entity("NVDA")
        assert len(nvda_facts) == 3
    
    def test_adding_facts_for_different_entities(self, store: FactStore) -> None:
        """Test adding facts for different entities."""
        entities = ["NVDA", "AAPL", "MSFT", "GOOGL"]
        for i, entity in enumerate(entities):
            store.add_fact(make_fact(fact_id=f"f{i}", entity=entity))
        
        assert len(store) == 4
        for entity in entities:
            facts = store.get_facts_by_entity(entity)
            assert len(facts) == 1
    
    def test_fact_with_none_value_not_counted_in_conflict(self, store: FactStore) -> None:
        """Test that facts with None values are excluded from conflict detection."""
        store.add_fact(make_fact(
            fact_id="f1", entity="NVDA", metric="Revenue", period="Q3 FY2025",
            value=10_000_000_000.0
        ))
        # Fact with None value
        fact_with_none = make_fact(
            fact_id="f2", entity="NVDA", metric="Revenue", period="Q3 FY2025",
        )
        fact_with_none = fact_with_none.model_copy(update={"value": None})
        store._facts[fact_with_none.fact_id] = fact_with_none  # bypass add_fact
        
        # Should not report conflict (only one fact has a numeric value)
        conflicts = store.find_conflicts()
        assert len(conflicts) == 0
    
    def test_repr(self, store: FactStore) -> None:
        """Test __repr__ method."""
        assert "FactStore" in repr(store)
        assert "facts=0" in repr(store)
        
        store.add_fact(make_fact(fact_id="f1"))
        assert "facts=1" in repr(store)

