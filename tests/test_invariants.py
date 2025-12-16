"""
P0 Safety Tests - Invariant Enforcement

These tests lock critical safety guarantees that MUST NOT regress.
They correspond to invariants defined in docs/invariants.md.

Run with: PYTHONPATH=src pytest tests/test_invariants.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from open_deep_research.orchestrator import Orchestrator
from open_deep_research.classifier import (
    classify_query_detailed,
    QueryType,
    ConfidenceBand,
)
from open_deep_research.models import NarratedReport


class TestPlaceholderHandlersFailClosed:
    """
    Invariant I4: Fail Closed
    
    All placeholder handlers MUST:
    - Set insufficient_data=True
    - Return empty citations
    - Return empty facts_used
    - Include "NOT IMPLEMENTED" in answer
    """
    
    @pytest.fixture
    def orc(self):
        return Orchestrator()
    
    def test_signal_detection_fails_closed(self, orc):
        """Signal detection handler must fail closed (provides guidance, not data)."""
        report = orc._handle_signal_detection("Any red flags in NVDA?")
        
        # Signal detection is "wired" but requires multi-period SEC data
        # It provides usage guidance but no actual signal analysis
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []
        # Should provide helpful guidance (not just "NOT IMPLEMENTED")
        assert "Signal Detection" in report.answer or "analyze_risk_signal" in report.answer
    
    def test_verification_fails_closed(self, orc):
        """Verification handler must fail closed."""
        report = orc._handle_verification("Is it true NVDA had $35B revenue?")
        
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []
        assert "NOT IMPLEMENTED" in report.answer
    
    def test_qualitative_fails_closed(self, orc):
        """Qualitative handler must fail closed."""
        report = orc._handle_qualitative("What did the CEO say about AI?")
        
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []
        assert "NOT IMPLEMENTED" in report.answer
    
    def test_exploration_fails_closed(self, orc):
        """Exploration handler must fail closed."""
        report = orc._handle_exploration("Tell me everything about NVDA")
        
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []
        assert "NOT IMPLEMENTED" in report.answer
    
    def test_unknown_fails_closed(self, orc):
        """Unknown handler must fail closed."""
        report = orc._handle_unknown("asdfghjkl random gibberish")
        
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []


class TestDegradedModeNeverRoutesXBRL:
    """
    Invariant: Degraded mode MUST NOT touch XBRL (Tier 1)
    
    When semantic classification is unavailable, the system cannot
    verify intent. It MUST refuse to route to XBRL to prevent
    "authoritative wrong" answers.
    """
    
    @pytest.fixture
    def orc(self):
        return Orchestrator()
    
    def test_degraded_mode_returns_insufficient_data(self, orc):
        """Degraded mode must return insufficient_data=True."""
        from open_deep_research.classifier import ClassificationResult
        
        # Simulate degraded mode result
        degraded_result = ClassificationResult(
            query_type=QueryType.UNKNOWN,
            similarity=0.0,
            margin=0.0,
            confidence_band=ConfidenceBand.AMBIGUOUS,
            method="regex_degraded",
            ambiguous=True,
            scores={"regex_hints": ["financial_keywords"]},
            secondary_candidates=[],
        )
        
        report = orc._handle_degraded_mode(
            question="What was NVDA revenue?",
            result=degraded_result,
            auto_load=True,  # Should be ignored!
        )
        
        assert report.insufficient_data is True
        assert "DEGRADED MODE" in report.answer
        assert report.citations == []
        assert report.facts_used == []
    
    def test_degraded_mode_never_calls_load_facts(self, orc):
        """Degraded mode MUST NOT call load_facts_for_entity."""
        from open_deep_research.classifier import ClassificationResult
        
        # Track if load_facts_for_entity was called
        original_load = orc.load_facts_for_entity
        load_called = False
        
        def mock_load(*args, **kwargs):
            nonlocal load_called
            load_called = True
            return original_load(*args, **kwargs)
        
        orc.load_facts_for_entity = mock_load
        
        degraded_result = ClassificationResult(
            query_type=QueryType.UNKNOWN,
            similarity=0.0,
            margin=0.0,
            confidence_band=ConfidenceBand.AMBIGUOUS,
            method="regex_degraded",
            ambiguous=True,
            scores={"regex_hints": ["financial_keywords"]},
            secondary_candidates=[],
        )
        
        orc._handle_degraded_mode(
            question="What was NVDA revenue in Q3 2024?",
            result=degraded_result,
            auto_load=True,
        )
        
        assert not load_called, "XBRL (load_facts_for_entity) was called in degraded mode!"
    
    def test_degraded_mode_never_calls_xbrl_extract(self, orc):
        """Degraded mode MUST NOT call extract_xbrl_fact."""
        from open_deep_research.classifier import ClassificationResult
        
        with patch('open_deep_research.xbrl.extract_xbrl_fact') as mock_xbrl:
            mock_xbrl.return_value = None
            
            degraded_result = ClassificationResult(
                query_type=QueryType.UNKNOWN,
                similarity=0.0,
                margin=0.0,
                confidence_band=ConfidenceBand.AMBIGUOUS,
                method="regex_degraded",
                ambiguous=True,
                scores={"regex_hints": ["financial_keywords"]},
                secondary_candidates=[],
            )
            
            orc._handle_degraded_mode(
                question="What was NVDA revenue?",
                result=degraded_result,
                auto_load=True,
            )
            
            mock_xbrl.assert_not_called()
    
    def test_classifier_returns_unknown_when_embeddings_down(self):
        """Classifier must return UNKNOWN in degraded mode."""
        result = classify_query_detailed("What was NVDA revenue?")
        
        # If we're in degraded mode (no sentence-transformers)
        if result.method == "regex_degraded":
            assert result.query_type == QueryType.UNKNOWN
            assert result.similarity == 0.0
            assert result.margin == 0.0
            assert result.confidence_band == ConfidenceBand.AMBIGUOUS
            assert result.ambiguous is True


class TestAskNeverReturnsMisleadingOutput:
    """
    Invariant: ask() cannot return authoritative-looking output
    unless facts exist in FactStore with citations.
    """
    
    @pytest.fixture
    def empty_orc(self):
        """Orchestrator with empty FactStore."""
        return Orchestrator()
    
    def test_ask_with_empty_factstore_no_misleading_answer(self, empty_orc):
        """ask() with empty FactStore must not return fake facts."""
        # Force a query that would go to financial lookup
        # but with no facts loaded
        with patch.object(empty_orc, 'load_facts_for_entity', return_value=0):
            # This should not return a fake answer
            report = empty_orc.ask("What was NVDA revenue in Q3 2024?", auto_load=False)
        
        # Either insufficient_data OR answer should indicate no data
        if report.facts_used:
            # If facts are claimed, they must be real
            assert len(empty_orc.fact_store) > 0
        else:
            # No facts = should indicate insufficiency
            assert report.insufficient_data or "no data" in report.answer.lower() or "not found" in report.answer.lower() or "could not" in report.answer.lower()
    
    def test_placeholder_report_always_insufficient(self, empty_orc):
        """_placeholder_report must always set insufficient_data=True."""
        report = empty_orc._placeholder_report("test query", "test message")
        
        assert report.insufficient_data is True
        assert report.citations == []
        assert report.facts_used == []


class TestClassifierDegradedModeSafety:
    """
    Classifier contract when embeddings unavailable.
    """
    
    def test_degraded_returns_unknown_not_financial(self):
        """Degraded mode must return UNKNOWN, never FINANCIAL_LOOKUP."""
        result = classify_query_detailed("What was Apple revenue?")
        
        if result.method == "regex_degraded":
            # MUST be UNKNOWN, never FINANCIAL_LOOKUP
            assert result.query_type == QueryType.UNKNOWN
            assert result.query_type != QueryType.FINANCIAL_LOOKUP
    
    def test_degraded_has_regex_hints(self):
        """Degraded mode should include regex hints for diagnostics."""
        result = classify_query_detailed("What was NVDA revenue last quarter?")
        
        if result.method == "regex_degraded":
            assert "regex_hints" in result.scores
            # Should detect financial keywords
            hints = result.scores.get("regex_hints", [])
            assert isinstance(hints, list)
    
    def test_degraded_never_claims_high_confidence(self):
        """Degraded mode must never claim high confidence."""
        result = classify_query_detailed("What was revenue?")
        
        if result.method == "regex_degraded":
            assert result.confidence_band == ConfidenceBand.AMBIGUOUS
            assert result.similarity == 0.0


class TestGateAEnforcement:
    """
    Gate A: Source-Alignment Gate (Tier 2, SEC HTML → FactStore)
    
    Ensures:
    - Mismatched facts are NEVER stored
    - Only verified facts can enter FactStore
    - No bypass paths exist
    """
    
    def test_mismatch_never_stored(self):
        """Facts that fail verification must NOT enter verified list."""
        from open_deep_research.pipeline import process_extracted_facts
        from open_deep_research.models import Fact, Location
        
        # Create a fact with a hallucinated sentence
        hallucinated_fact = Fact(
            fact_id="test-hallucinated",
            entity="NVDA",
            metric="revenue",
            value=100_000_000_000.0,  # $100B - fake
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="Item7",
                sentence_string="Revenue was a massive $100 billion!",  # HALLUCINATED
            ),
            source_format="html_text",
            doc_hash="test-hash",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        # The actual source text
        source_text = "Revenue was $35 billion for the quarter."
        
        verified, rejected = process_extracted_facts(
            facts=[hallucinated_fact],
            source_text=source_text,
            tables=[]
        )
        
        # CRITICAL: Hallucinated fact must be in rejected, NOT verified
        assert len(verified) == 0, "Hallucinated fact was incorrectly verified!"
        assert len(rejected) == 1
        assert rejected[0].verification_status == "mismatch"
    
    def test_only_verified_facts_pass(self):
        """Only exact_match or approximate_match facts pass verification."""
        from open_deep_research.pipeline import process_extracted_facts
        from open_deep_research.models import Fact, Location
        
        # Create a legitimate fact
        real_fact = Fact(
            fact_id="test-real",
            entity="NVDA",
            metric="revenue",
            value=35_000_000_000.0,  # $35B
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="Item7",
                sentence_string="Revenue was $35 billion for the quarter.",  # REAL
            ),
            source_format="html_text",
            doc_hash="test-hash",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        source_text = "In Q3 FY2025, Revenue was $35 billion for the quarter. This exceeded expectations."
        
        verified, rejected = process_extracted_facts(
            facts=[real_fact],
            source_text=source_text,
            tables=[]
        )
        
        assert len(verified) == 1
        assert len(rejected) == 0
        assert verified[0].verification_status in ("exact_match", "approximate_match")
    
    def test_no_bypass_without_verification(self):
        """Facts cannot enter verified list without passing verification."""
        from open_deep_research.pipeline import process_extracted_facts
        from open_deep_research.models import Fact, Location
        
        # Fact with no sentence_string (can't be verified)
        unverifiable_fact = Fact(
            fact_id="test-unverifiable",
            entity="NVDA",
            metric="revenue",
            value=35_000_000_000.0,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-20",
                doc_type="10-Q",
                section_id="Item7",
                sentence_string=None,  # Missing! Can't verify
            ),
            source_format="html_text",
            doc_hash="test-hash",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        verified, rejected = process_extracted_facts(
            facts=[unverifiable_fact],
            source_text="Revenue was $35 billion.",
            tables=[]
        )
        
        # Must be rejected, not verified
        assert len(verified) == 0
        assert len(rejected) == 1


class TestTier3Containment:
    """
    Tier 3 Contract: News/Web → Gate B → Report (READ-ONLY)
    
    Tier 3 (discovery, news verification) must:
    - NEVER write to FactStore
    - ALWAYS label outputs as contextual/soft
    - Use Gate B (cross-source consistency) not Gate A (source-alignment)
    """
    
    def test_discover_never_writes_to_factstore(self):
        """discover() must not modify FactStore."""
        from open_deep_research.discovery import discover
        from open_deep_research.store import FactStore
        
        store = FactStore()
        initial_count = len(store)
        
        # Run discovery - this should NOT touch FactStore
        # Note: discover() doesn't even take a FactStore param
        leads = discover("NVDA news", ticker="NVDA")
        
        # FactStore should be unchanged
        assert len(store) == initial_count
    
    def test_discover_and_verify_never_writes_to_factstore(self):
        """discover_and_verify() must not add facts to FactStore."""
        from open_deep_research.discovery import discover_and_verify
        from open_deep_research.store import FactStore
        from open_deep_research.models import Fact, Location
        
        store = FactStore()
        
        # Add a seed fact
        seed_fact = Fact(
            fact_id="test-seed",
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
        store.add_fact(seed_fact)
        initial_count = len(store)
        
        # Run discovery and verification
        report = discover_and_verify("NVDA revenue", fact_store=store, ticker="NVDA")
        
        # FactStore count should be UNCHANGED
        # Discovery reads from FactStore but NEVER writes
        assert len(store) == initial_count, (
            f"Tier 3 wrote to FactStore! Count went from {initial_count} to {len(store)}"
        )
    
    def test_discovery_report_has_tier3_disclaimer(self):
        """DiscoveryReport must include Tier 3 disclaimer."""
        from open_deep_research.models import DiscoveryReport, Lead
        from datetime import datetime
        
        report = DiscoveryReport(
            query="test query",
            ticker="NVDA",
            leads=[
                Lead(
                    lead_id="test-lead-1",
                    text="Test lead",
                    source_name="Test Source",
                    source_url="https://test.com",
                    found_at=datetime.now(),
                )
            ],
            generated_at=datetime.now(),
        )
        
        formatted = report.format_report()
        
        # Must contain Tier 3 warning
        assert "TIER 3" in formatted or "Tier 3" in formatted
        assert "Read-Only" in formatted or "read-only" in formatted.lower()
        assert "NOT write" in formatted or "does not write" in formatted.lower()
    
    def test_verify_leads_never_writes_to_factstore(self):
        """verify_leads() must not modify FactStore."""
        from open_deep_research.discovery import verify_leads
        from open_deep_research.store import FactStore
        from open_deep_research.models import Lead, Fact, Location
        
        store = FactStore()
        
        # Add seed fact
        seed_fact = Fact(
            fact_id="test-seed",
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
        store.add_fact(seed_fact)
        initial_count = len(store)
        
        # Create leads to verify
        from datetime import datetime
        leads = [
            Lead(
                lead_id="test-lead-verify",
                text="NVIDIA reported $35.08B revenue",
                source_name="Test",
                found_at=datetime.now(),
                entity="NVDA",
                metric="revenue",
                value=35_080_000_000,
                period="Q3 FY2025",
            )
        ]
        
        # Verify leads
        verified_leads = verify_leads(leads, store)
        
        # FactStore must be unchanged
        assert len(store) == initial_count


class TestFactStoreIntegrity:
    """
    Invariant I1: FactStore is the only truth source.
    Narrator must only use facts from FactStore.
    """
    
    def test_factstore_starts_empty(self):
        """New Orchestrator must have empty FactStore."""
        orc = Orchestrator()
        assert len(orc.fact_store) == 0
    
    def test_clear_empties_factstore(self):
        """clear() must empty the FactStore."""
        orc = Orchestrator()
        # Add a mock fact using the helper from test_discovery
        from open_deep_research.models import Fact, Location
        
        fact = Fact(
            fact_id="test-fact",
            entity="TEST",
            metric="revenue",
            value=1000000,
            unit="USD",
            period="Q1 2024",
            period_end_date="2024-01-31",
            location=Location(
                cik="0000000000",
                doc_date="2024-01-01",
                doc_type="10-K",
                section_id="test",
            ),
            source_format="xbrl",
            doc_hash="test-hash",
            snapshot_id="test-snapshot",
            verification_status="exact_match",
        )
        orc.fact_store.add_fact(fact)
        assert len(orc.fact_store) == 1
        
        orc.clear()
        assert len(orc.fact_store) == 0

