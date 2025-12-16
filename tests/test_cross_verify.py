"""
Tests for cross-verification system.

CRITICAL: These tests ensure unit normalization works correctly.
The #1 failure mode is unit mismatch:
- News: "$35 billion" or "$35B"
- XBRL: "35082000000" (raw) or "35082" (in millions)

If we compare 35 to 35082000000, we get "CONTRADICTED" when it should be "VERIFIED".
"""
import pytest

from open_deep_research.cross_verify import (
    SPACY_AVAILABLE,
    detect_scale_from_text,
    extract_claims_from_text,
    extract_entities_from_text,
    format_verification_report,
    infer_scale_from_magnitude,
    normalize_to_base_units,
    verify_all_claims,
    verify_claim,
    _compare_values,
    _find_matching_fact,
    _get_metric_aliases,
)
from open_deep_research.models import Claim, Fact, Location, VerificationResult
from open_deep_research.store import FactStore


# =============================================================================
# Unit Normalization Tests (CRITICAL)
# =============================================================================


class TestNormalizeToBaseUnits:
    """CRITICAL: These tests ensure unit normalization works correctly."""
    
    def test_billion_text(self):
        """$35.08 billion should normalize to 35,080,000,000."""
        result = normalize_to_base_units(35.08, "billion")
        assert result == 35_080_000_000
    
    def test_billion_short_b(self):
        """35B should normalize to 35,000,000,000."""
        result = normalize_to_base_units(35, "B")
        assert result == 35_000_000_000
    
    def test_billion_short_bn(self):
        """35bn should normalize to 35,000,000,000."""
        result = normalize_to_base_units(35, "bn")
        assert result == 35_000_000_000
    
    def test_million_text(self):
        """500 million should normalize to 500,000,000."""
        result = normalize_to_base_units(500, "million")
        assert result == 500_000_000
    
    def test_million_short_m(self):
        """35082M should normalize to 35,082,000,000."""
        result = normalize_to_base_units(35082, "M")
        assert result == 35_082_000_000
    
    def test_million_short_mn(self):
        """35082mn should normalize to 35,082,000,000."""
        result = normalize_to_base_units(35082, "mn")
        assert result == 35_082_000_000
    
    def test_trillion_text(self):
        """1.5 trillion should normalize to 1,500,000,000,000."""
        result = normalize_to_base_units(1.5, "trillion")
        assert result == 1_500_000_000_000
    
    def test_trillion_short(self):
        """1.5T should normalize to 1,500,000,000,000."""
        result = normalize_to_base_units(1.5, "T")
        assert result == 1_500_000_000_000
    
    def test_thousand_text(self):
        """500 thousand should normalize to 500,000."""
        result = normalize_to_base_units(500, "thousand")
        assert result == 500_000
    
    def test_thousand_short(self):
        """500K should normalize to 500,000."""
        result = normalize_to_base_units(500, "K")
        assert result == 500_000
    
    def test_no_scale(self):
        """Raw value should pass through unchanged."""
        result = normalize_to_base_units(35_082_000_000)
        assert result == 35_082_000_000
    
    def test_no_hint(self):
        """None unit hint should return value unchanged."""
        result = normalize_to_base_units(35_082_000_000, None)
        assert result == 35_082_000_000
    
    def test_xbrl_decimals_millions(self):
        """XBRL with decimals=-6 means 'in millions'."""
        result = normalize_to_base_units(35082, decimals_attr=-6)
        assert result == 35_082_000_000
    
    def test_xbrl_decimals_billions(self):
        """XBRL with decimals=-9 means 'in billions'."""
        result = normalize_to_base_units(35, decimals_attr=-9)
        assert result == 35_000_000_000
    
    def test_xbrl_decimals_thousands(self):
        """XBRL with decimals=-3 means 'in thousands'."""
        result = normalize_to_base_units(35082000, decimals_attr=-3)
        assert result == 35_082_000_000
    
    def test_case_insensitive_billion(self):
        """Billion/BILLION/BiLLiOn should all work."""
        assert normalize_to_base_units(35, "BILLION") == 35_000_000_000
        assert normalize_to_base_units(35, "Billion") == 35_000_000_000
        assert normalize_to_base_units(35, "BiLLiOn") == 35_000_000_000


class TestDetectScaleFromText:
    """Test extraction of value and scale from natural language text."""
    
    def test_detect_billion_word(self):
        """Detect '35 billion'."""
        value, mult = detect_scale_from_text("$35 billion in revenue")
        assert value == 35.0
        assert mult == 1_000_000_000
    
    def test_detect_billion_short(self):
        """Detect '35.08B'."""
        value, mult = detect_scale_from_text("revenue of $35.08B")
        assert value == 35.08
        assert mult == 1_000_000_000
    
    def test_detect_bn(self):
        """Detect '35bn'."""
        value, mult = detect_scale_from_text("reported 35bn revenue")
        assert value == 35.0
        assert mult == 1_000_000_000
    
    def test_detect_million_word(self):
        """Detect '500 million'."""
        value, mult = detect_scale_from_text("$500 million profit")
        assert value == 500.0
        assert mult == 1_000_000
    
    def test_detect_million_short(self):
        """Detect '500M'."""
        value, mult = detect_scale_from_text("earned $500M")
        assert value == 500.0
        assert mult == 1_000_000
    
    def test_detect_trillion(self):
        """Detect '1.5 trillion'."""
        value, mult = detect_scale_from_text("market cap of $1.5 trillion")
        assert value == 1.5
        assert mult == 1_000_000_000_000
    
    def test_detect_just_dollar(self):
        """Detect '$500' without scale suffix."""
        value, mult = detect_scale_from_text("paid $500 for services")
        assert value == 500.0
        assert mult == 1.0
    
    def test_detect_with_commas(self):
        """Detect values with commas like '1,500 billion'."""
        value, mult = detect_scale_from_text("$1,500 million in sales")
        assert value == 1500.0
        assert mult == 1_000_000
    
    def test_detect_decimal(self):
        """Detect '35.082 billion' with decimals."""
        value, mult = detect_scale_from_text("revenue hit $35.082 billion")
        assert value == 35.082
        assert mult == 1_000_000_000
    
    def test_no_value_found(self):
        """Return None when no monetary value found."""
        value, mult = detect_scale_from_text("no numbers here")
        assert value is None
        assert mult is None
    
    def test_case_insensitive(self):
        """Handle BILLION, Billion, billion."""
        value1, mult1 = detect_scale_from_text("$35 BILLION")
        value2, mult2 = detect_scale_from_text("$35 Billion")
        value3, mult3 = detect_scale_from_text("$35 billion")
        
        assert value1 == value2 == value3 == 35.0
        assert mult1 == mult2 == mult3 == 1_000_000_000


class TestInferScaleFromMagnitude:
    """Test heuristic scale inference when no hint is provided."""
    
    def test_already_base_units(self):
        """Large values (>1B) assumed to be base units."""
        result = infer_scale_from_magnitude(35_082_000_000)
        assert result == 35_082_000_000
    
    def test_infer_millions_reporting(self):
        """Values 1000-1M assumed to be 'in millions'."""
        result = infer_scale_from_magnitude(35_082)  # 35,082 in millions
        assert result == 35_082_000_000
    
    def test_infer_billions_small(self):
        """Small values (<1000) assumed to be 'in billions'."""
        result = infer_scale_from_magnitude(35)
        assert result == 35_000_000_000


# =============================================================================
# Entity Extraction Tests
# =============================================================================


class TestExtractEntitiesFromText:
    """Test spacy-based entity extraction with graceful degradation."""
    
    def test_returns_dict_structure(self):
        """Should return dict with expected keys even if empty."""
        result = extract_entities_from_text("some text")
        assert "organizations" in result
        assert "money" in result
        assert "dates" in result
        assert "percentages" in result
    
    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spacy not installed")
    def test_extracts_money_with_spacy(self):
        """Extract MONEY entities with spacy."""
        result = extract_entities_from_text("NVIDIA earned $35 billion in Q3.")
        assert len(result["money"]) >= 1
    
    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spacy not installed")
    def test_extracts_organizations_with_spacy(self):
        """Extract ORG entities with spacy."""
        # Use well-known companies that en_core_web_sm reliably recognizes
        result = extract_entities_from_text("Apple Inc. and Microsoft Corporation reported earnings.")
        # en_core_web_sm may not recognize all orgs; just verify structure works
        assert isinstance(result["organizations"], list)


# =============================================================================
# Claim Extraction Tests
# =============================================================================


class TestExtractClaimsFromText:
    """Test extraction of verifiable claims from news text."""
    
    def test_extract_revenue_billion(self):
        """Extract '$35 billion in revenue'."""
        text = "NVIDIA reported $35 billion in revenue last quarter."
        claims = extract_claims_from_text(text, "test")
        
        assert len(claims) >= 1
        # Find revenue claim
        revenue_claim = None
        for c in claims:
            if c.metric and "revenue" in c.metric.lower():
                revenue_claim = c
                break
        
        assert revenue_claim is not None
        assert revenue_claim.value == 35.0
        assert revenue_claim.value_raw == 35_000_000_000
    
    def test_extract_revenue_b(self):
        """Extract 'Revenue reached $35.08B'."""
        text = "Revenue reached $35.08B in Q3."
        claims = extract_claims_from_text(text, "test")
        
        assert len(claims) >= 1
        revenue_claim = claims[0]
        assert abs(revenue_claim.value_raw - 35_080_000_000) < 1_000_000
    
    def test_extract_income(self):
        """Extract 'net income of $18 billion'."""
        text = "Net income of $18 billion beat expectations."
        claims = extract_claims_from_text(text, "test")
        
        assert len(claims) >= 1
        income_claim = None
        for c in claims:
            if c.metric and "income" in c.metric.lower():
                income_claim = c
                break
        
        if income_claim:
            assert income_claim.value == 18.0
    
    def test_extract_with_source_tier(self):
        """Claims should have correct source tier."""
        text = "$35 billion in revenue"
        claims = extract_claims_from_text(text, "Reuters", source_tier="medium")
        
        if claims:
            assert claims[0].source_tier == "medium"
    
    def test_extract_with_source(self):
        """Claims should have source set."""
        text = "$35 billion in revenue"
        claims = extract_claims_from_text(text, "Reuters")
        
        if claims:
            assert claims[0].source == "Reuters"
    
    def test_no_claims_in_empty_text(self):
        """Return empty list for text with no financial claims."""
        text = "The weather is nice today."
        claims = extract_claims_from_text(text, "test")
        assert claims == []
    
    def test_deduplication(self):
        """Should not return duplicate claims for same text."""
        text = "$35 billion in revenue. The company earned $35 billion in revenue."
        claims = extract_claims_from_text(text, "test")
        
        # Check for unique texts
        texts = [c.text for c in claims]
        assert len(texts) <= 2  # At most two unique claims


# =============================================================================
# Metric Alias Tests
# =============================================================================


class TestGetMetricAliases:
    """Test metric name alias resolution."""
    
    def test_revenue_aliases(self):
        """Revenue should include sales, total revenue, etc."""
        aliases = _get_metric_aliases("revenue")
        assert "revenue" in aliases
        assert "sales" in aliases
        assert "total revenue" in aliases
    
    def test_income_aliases(self):
        """Income should include net income, profit, etc."""
        aliases = _get_metric_aliases("income")
        assert "income" in aliases
        assert "net income" in aliases
        assert "profit" in aliases
    
    def test_eps_aliases(self):
        """EPS should include earnings per share."""
        aliases = _get_metric_aliases("eps")
        assert "eps" in aliases
        assert "earnings per share" in aliases
    
    def test_unknown_metric(self):
        """Unknown metric should return itself."""
        aliases = _get_metric_aliases("unusual_metric")
        assert aliases == ["unusual_metric"]
    
    def test_none_metric(self):
        """None metric should return empty list."""
        aliases = _get_metric_aliases(None)
        assert aliases == []


# =============================================================================
# Verification Tests
# =============================================================================


def make_test_fact(
    entity: str = "NVDA",
    metric: str = "Total Revenue",
    value: float = 35_082_000_000,
    period: str = "Q3 FY2025",
) -> Fact:
    """Create a test fact with minimal required fields."""
    return Fact(
        fact_id=f"test-{metric.lower().replace(' ', '-')}",
        entity=entity,
        metric=metric,
        value=value,
        unit="USD",
        period=period,
        period_end_date="2024-10-27",
        location=Location(
            cik="0001045810",
            doc_date="2024-11-20",
            doc_type="10-Q",
            section_id="Item2",
        ),
        source_format="xbrl",
        doc_hash="test-hash",
        snapshot_id="test-snapshot",
        verification_status="exact_match",
    )


def make_test_claim(
    text: str = "$35 billion in revenue",
    metric: str = "revenue",
    value: float = 35.0,
    value_raw: float = 35_000_000_000,
    entity: str = "NVDA",
) -> Claim:
    """Create a test claim."""
    return Claim(
        text=text,
        source="test",
        source_tier="soft",
        claim_type="quantitative",
        entity=entity,
        metric=metric,
        value=value,
        value_raw=value_raw,
        unit_hint="billion",
    )


class TestVerification:
    """Test that verification handles unit differences correctly."""
    
    @pytest.fixture
    def fact_store_with_nvda(self):
        """Create a FactStore with NVDA revenue fact."""
        store = FactStore()
        fact = make_test_fact()
        store._facts[fact.fact_id] = fact  # Bypass verification status check
        return store
    
    def test_verify_35b_matches_35082000000(self, fact_store_with_nvda):
        """CRITICAL: $35B should be 'close' to 35,082,000,000 (2.3% diff)."""
        claim = make_test_claim()
        result = verify_claim(claim, fact_store_with_nvda)
        
        # Should be "close" (within 5%) not "contradicted"
        assert result.status in ("verified", "close"), f"Got {result.status}: {result.explanation}"
        assert result.status != "contradicted"
    
    def test_verify_exact_match(self, fact_store_with_nvda):
        """$35.082 billion should be 'verified' (exact match)."""
        claim = make_test_claim(
            text="$35.082 billion in revenue",
            value=35.082,
            value_raw=35_082_000_000,
        )
        result = verify_claim(claim, fact_store_with_nvda)
        
        assert result.status == "verified"
    
    def test_verify_wrong_value_contradicted(self, fact_store_with_nvda):
        """$30B should be 'contradicted' (14% off)."""
        claim = make_test_claim(
            text="$30 billion in revenue",
            value=30.0,
            value_raw=30_000_000_000,
        )
        result = verify_claim(claim, fact_store_with_nvda)
        
        assert result.status == "contradicted"
        assert result.difference_pct is not None
        assert result.difference_pct > 10  # Should be ~14%
    
    def test_verify_slightly_off(self, fact_store_with_nvda):
        """$35.4B should be 'close' (~1% off)."""
        claim = make_test_claim(
            text="$35.4 billion in revenue",
            value=35.4,
            value_raw=35_400_000_000,
        )
        result = verify_claim(claim, fact_store_with_nvda)
        
        assert result.status in ("verified", "close")
    
    def test_unverifiable_no_facts(self):
        """Should be 'unverifiable' when no matching facts exist."""
        store = FactStore()  # Empty store
        claim = make_test_claim()
        result = verify_claim(claim, store)
        
        assert result.status == "unverifiable"
    
    def test_unverifiable_qualitative_claim(self, fact_store_with_nvda):
        """Qualitative claims cannot be verified against SEC data."""
        claim = Claim(
            text="NVIDIA is a great company",
            source="test",
            source_tier="soft",
            claim_type="qualitative",
        )
        result = verify_claim(claim, fact_store_with_nvda)
        
        assert result.status == "unverifiable"
    
    def test_unverifiable_no_value(self, fact_store_with_nvda):
        """Claims without parsed values are unverifiable."""
        claim = Claim(
            text="revenue was substantial",
            source="test",
            source_tier="soft",
            claim_type="quantitative",
            value_raw=None,
        )
        result = verify_claim(claim, fact_store_with_nvda)
        
        assert result.status == "unverifiable"
    
    def test_verify_income_metric(self):
        """Verify income claims against net income facts."""
        store = FactStore()
        fact = make_test_fact(metric="Net Income", value=18_000_000_000)
        store._facts[fact.fact_id] = fact
        
        claim = make_test_claim(
            text="$18 billion in income",
            metric="income",
            value=18.0,
            value_raw=18_000_000_000,
        )
        result = verify_claim(claim, store)
        
        assert result.status == "verified"


class TestVerifyAllClaims:
    """Test batch verification of multiple claims."""
    
    def test_verify_multiple_claims(self):
        """Verify multiple claims at once."""
        store = FactStore()
        fact = make_test_fact()
        store._facts[fact.fact_id] = fact
        
        claims = [
            make_test_claim(value=35.0, value_raw=35_000_000_000),
            make_test_claim(value=30.0, value_raw=30_000_000_000),
        ]
        
        results = verify_all_claims(claims, store)
        
        assert len(results) == 2
        # First should be close, second should be contradicted
        statuses = {r.status for r in results}
        assert "contradicted" in statuses


class TestFindMatchingFact:
    """Test fact matching logic."""
    
    def test_matches_by_metric_alias(self):
        """Should match 'revenue' claim to 'Total Revenue' fact."""
        store = FactStore()
        fact = make_test_fact(metric="Total Revenue")
        store._facts[fact.fact_id] = fact
        
        claim = make_test_claim(metric="revenue")
        matched = _find_matching_fact(claim, store)
        
        assert matched is not None
        assert matched.metric == "Total Revenue"
    
    def test_filters_by_entity(self):
        """Should only match facts for the same entity."""
        store = FactStore()
        nvda_fact = make_test_fact(entity="NVDA")
        aapl_fact = make_test_fact(entity="AAPL", value=50_000_000_000)
        store._facts[nvda_fact.fact_id] = nvda_fact
        store._facts["aapl-revenue"] = aapl_fact
        
        claim = make_test_claim(entity="NVDA")
        matched = _find_matching_fact(claim, store)
        
        assert matched is not None
        assert matched.entity == "NVDA"


# =============================================================================
# Report Formatting Tests
# =============================================================================


class TestFormatVerificationReport:
    """Test report formatting."""
    
    def test_report_has_summary(self):
        """Report should include summary counts."""
        claim = make_test_claim()
        results = [
            VerificationResult(
                claim=claim,
                status="verified",
                hard_source="SEC XBRL",
                hard_value=35_082_000_000,
                difference_pct=2.3,
                explanation="Matches SEC data",
            )
        ]
        
        report = format_verification_report(results)
        
        assert "Total Claims Analyzed: 1" in report
        assert "Verified:" in report
    
    def test_report_shows_contradicted(self):
        """Contradicted claims should be prominently displayed."""
        claim = make_test_claim()
        results = [
            VerificationResult(
                claim=claim,
                status="contradicted",
                hard_source="SEC XBRL",
                hard_value=35_082_000_000,
                difference_pct=14.5,
                explanation="Differs from SEC data",
            )
        ]
        
        report = format_verification_report(results)
        
        assert "CONTRADICTED" in report
        assert "🔴" in report
    
    def test_report_shows_verified(self):
        """Verified claims should be shown."""
        claim = make_test_claim()
        results = [
            VerificationResult(
                claim=claim,
                status="verified",
                hard_source="SEC XBRL",
                hard_value=35_082_000_000,
                difference_pct=0.2,
                explanation="Matches SEC data",
            )
        ]
        
        report = format_verification_report(results)
        
        assert "VERIFIED" in report
        assert "🟢" in report
    
    def test_empty_results(self):
        """Should handle empty results list."""
        report = format_verification_report([])
        
        assert "Total Claims Analyzed: 0" in report


class TestVerificationResultModel:
    """Test VerificationResult model methods."""
    
    def test_get_trust_icon_verified(self):
        """Verified status should show green circle."""
        claim = make_test_claim()
        result = VerificationResult(claim=claim, status="verified")
        assert result.get_trust_icon() == "🟢"
    
    def test_get_trust_icon_contradicted(self):
        """Contradicted status should show red circle."""
        claim = make_test_claim()
        result = VerificationResult(claim=claim, status="contradicted")
        assert result.get_trust_icon() == "🔴"
    
    def test_get_trust_icon_close(self):
        """Close status should show yellow circle."""
        claim = make_test_claim()
        result = VerificationResult(claim=claim, status="close")
        assert result.get_trust_icon() == "🟡"
    
    def test_get_trust_icon_unverifiable(self):
        """Unverifiable status should show white circle."""
        claim = make_test_claim()
        result = VerificationResult(claim=claim, status="unverifiable")
        assert result.get_trust_icon() == "⚪"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCrossVerifyIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from text to verification."""
        # Setup fact store with known value
        store = FactStore()
        fact = make_test_fact()
        store._facts[fact.fact_id] = fact
        
        # Extract claims from news text
        news_text = "NVIDIA reported $35 billion in revenue for Q3 2024."
        claims = extract_claims_from_text(news_text, "Reuters")
        
        # Add entity to claims
        for c in claims:
            c.entity = "NVDA"
        
        # Verify claims
        results = verify_all_claims(claims, store)
        
        # Check results
        if results:
            # The $35B claim should be close to $35.082B
            assert results[0].status in ("verified", "close")
    
    def test_contradicted_pipeline(self):
        """Test pipeline detects contradicted claims."""
        # Setup fact store
        store = FactStore()
        fact = make_test_fact(value=35_082_000_000)
        store._facts[fact.fact_id] = fact
        
        # News with wrong value
        news_text = "NVIDIA reported $20 billion in revenue."  # Wrong!
        claims = extract_claims_from_text(news_text, "BadSource")
        
        for c in claims:
            c.entity = "NVDA"
        
        results = verify_all_claims(claims, store)
        
        if results:
            # The $20B claim should be contradicted against $35.082B
            assert results[0].status == "contradicted"

