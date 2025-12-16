"""
Cross-Verification System - Verify soft source claims against hard sources.

This module is THE key differentiator of the anti-hallucination system:
"News says NVDA revenue was $30B. SEC says $35B. News is wrong."

The #1 failure mode is unit mismatch:
- News: "$35 billion" or "$35B"
- XBRL: "35082000000" (raw) or "35082" (in millions)

Solution: Normalize EVERYTHING to base units (raw USD) before comparison.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

from open_deep_research.models import Claim, Fact, VerificationResult

if TYPE_CHECKING:
    from open_deep_research.store import FactStore


logger = logging.getLogger(__name__)


# =============================================================================
# NLP Library Setup (Graceful Degradation)
# P0 Fix: Use shared lazy-loading from signals module
# =============================================================================

from open_deep_research.signals import get_nlp, is_spacy_available

# Backwards compatibility: export SPACY_AVAILABLE for test skipif decorators
# We keep this as a callable check, but pytest decorators need a constant
# So we evaluate at import time for tests (acceptable tradeoff)
SPACY_AVAILABLE = is_spacy_available()


# =============================================================================
# Scale Detection Patterns
# =============================================================================

SCALE_PATTERNS = [
    (r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:trillion|T)\b', 1_000_000_000_000),
    (r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|B|bn)\b', 1_000_000_000),
    (r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|mn)\b', 1_000_000),
    (r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|K)\b', 1_000),
]


# Verification tolerances
EXACT_TOLERANCE = 0.01      # 1% = verified (exact match)
CLOSE_TOLERANCE = 0.05      # 5% = close (likely rounding)
# >5% = contradicted


# =============================================================================
# Unit Normalization (CRITICAL)
# =============================================================================


def detect_scale_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect value and scale from text like "$35 billion" or "35B".
    
    Args:
        text: Text containing a monetary value
        
    Returns:
        Tuple of (value, multiplier) or (None, None) if not found.
        
    Examples:
        >>> detect_scale_from_text("$35 billion in revenue")
        (35.0, 1000000000)
        >>> detect_scale_from_text("revenue of $35.08B")
        (35.08, 1000000000)
    """
    text = text.replace(',', '')  # Remove commas
    
    for pattern, multiplier in SCALE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            value = float(value_str)
            return value, multiplier
    
    # Try to find just a number with $ sign
    dollar_match = re.search(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if dollar_match:
        value_str = dollar_match.group(1).replace(',', '')
        return float(value_str), 1.0
    
    return None, None


def normalize_to_base_units(
    value: float, 
    unit_hint: Optional[str] = None,
    decimals_attr: Optional[int] = None
) -> float:
    """
    Normalize a value to base USD (raw number, no scale).
    
    This is the CRITICAL function that prevents unit mismatch failures.
    
    Args:
        value: The numeric value
        unit_hint: Text hint like "billion", "B", "million", "M"
        decimals_attr: XBRL decimals attribute (e.g., -6 means millions)
    
    Returns:
        Value in base units (raw USD)
    
    Examples:
        >>> normalize_to_base_units(35.08, "billion")
        35080000000.0
        >>> normalize_to_base_units(35082, decimals_attr=-6)
        35082000000
        >>> normalize_to_base_units(35082000000)
        35082000000
    """
    if unit_hint:
        hint_lower = str(unit_hint).lower()
        if "trillion" in hint_lower or hint_lower == "t":
            return value * 1_000_000_000_000
        elif "billion" in hint_lower or hint_lower in ("b", "bn"):
            return value * 1_000_000_000
        elif "million" in hint_lower or hint_lower in ("m", "mn"):
            return value * 1_000_000
        elif "thousand" in hint_lower or hint_lower == "k":
            return value * 1_000
    
    if decimals_attr is not None:
        # XBRL decimals: -9 = billions, -6 = millions, -3 = thousands
        return value * (10 ** abs(decimals_attr))
    
    return value


def infer_scale_from_magnitude(
    value: float, 
    expected_range: Optional[Tuple[float, float]] = None
) -> float:
    """
    Infer the correct scale when unit hint is missing.
    
    For revenue of a large company:
    - 35 → likely billions → 35,000,000,000
    - 35000 → likely millions → 35,000,000,000
    - 35000000000 → already base units
    
    This is a heuristic and should be used with caution.
    
    Args:
        value: The numeric value
        expected_range: Optional (min, max) expected range
        
    Returns:
        Value scaled to likely base units
    """
    # If value is already in billions range, assume base
    if value > 1_000_000_000:
        return value
    
    # If value is in millions range, might be "in millions" reporting
    if 1_000 < value < 1_000_000:
        # Likely "in millions"
        return value * 1_000_000
    
    # If value is small (< 1000), likely "in billions"  
    if value < 1000:
        return value * 1_000_000_000
    
    return value


# =============================================================================
# Entity Extraction
# =============================================================================


def extract_entities_from_text(text: str) -> dict:
    """
    Extract named entities using spacy.
    
    Returns dict with:
    - organizations: List of company names
    - money: List of money value strings
    - dates: List of date strings
    - percentages: List of percentage values
    
    Falls back to empty dict if spacy is not available.
    """
    result = {
        "organizations": [],
        "money": [],
        "dates": [],
        "percentages": [],
    }
    
    nlp = get_nlp()
    if nlp is None:
        return result
    
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            result["organizations"].append(ent.text)
        elif ent.label_ == "MONEY":
            result["money"].append(ent.text)
        elif ent.label_ == "DATE":
            result["dates"].append(ent.text)
        elif ent.label_ == "PERCENT":
            result["percentages"].append(ent.text)
    
    return result


# =============================================================================
# Claim Extraction
# =============================================================================


def _find_metric_near_money(text: str, money_text: str) -> Optional[str]:
    """
    Find the financial metric associated with a money value.
    
    Looks for keywords like 'revenue', 'income', 'profit' within
    ~50 characters of the money value.
    """
    idx = text.find(money_text)
    if idx == -1:
        return None
    
    # Get context window
    start = max(0, idx - 50)
    end = min(len(text), idx + len(money_text) + 50)
    context = text[start:end].lower()
    
    metrics = ["revenue", "income", "profit", "earnings", "sales", "eps"]
    for metric in metrics:
        if metric in context:
            return metric
    
    return None


def _get_unit_hint_str(scale: Optional[float]) -> Optional[str]:
    """Convert numeric scale to unit hint string."""
    if scale is None:
        return None
    if scale == 1_000_000_000_000:
        return "trillion"
    if scale == 1_000_000_000:
        return "billion"
    if scale == 1_000_000:
        return "million"
    if scale == 1_000:
        return "thousand"
    return None


def extract_claims_from_text(
    text: str, 
    source: str,
    source_tier: Literal["hard", "medium", "soft"] = "soft"
) -> List[Claim]:
    """
    Extract verifiable claims from text (news article, web page, etc).
    
    Uses spacy NER for entity detection when available,
    falls back to regex patterns otherwise.
    
    Args:
        text: The text to extract claims from
        source: Description of the source (e.g., "Reuters article")
        source_tier: Trust level of the source
        
    Returns:
        List of Claim objects extracted from the text
    """
    claims = []
    seen_texts = set()
    
    # Use spacy to find money entities if available
    if is_spacy_available():
        entities = extract_entities_from_text(text)
        
        # Process MONEY entities from spacy
        for money_text in entities["money"]:
            if money_text in seen_texts:
                continue
            seen_texts.add(money_text)
            
            value, scale = detect_scale_from_text(money_text)
            if value is not None:
                unit_hint = _get_unit_hint_str(scale)
                if scale:
                    value_raw = value * scale
                else:
                    value_raw = value
                
                # Try to find associated metric (look for nearby keywords)
                metric = _find_metric_near_money(text, money_text)
                
                claims.append(Claim(
                    text=money_text,
                    source=source,
                    source_tier=source_tier,
                    claim_type="quantitative",
                    metric=metric or "unspecified",
                    value=value,
                    value_raw=value_raw,
                    unit_hint=unit_hint,
                ))
    
    # Also use regex patterns (catches cases spacy misses)
    patterns = [
        # "$35 billion in revenue"
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|B|M|bn|mn)?\s*(?:in\s+)?(revenue|income|profit|earnings|sales)',
         lambda m: (float(m.group(1).replace(',', '')), m.group(2), m.group(3))),
        
        # "revenue of $35 billion"
        (r'(revenue|income|profit|earnings|sales)\s+(?:of|was|were|is|are|reached|hit)\s+\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|B|M)?',
         lambda m: (float(m.group(2).replace(',', '')), m.group(3), m.group(1))),
        
        # "reported $35B revenue"
        (r'reported\s+\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(B|M|billion|million)?\s*(revenue|income|profit)?',
         lambda m: (float(m.group(1).replace(',', '')), m.group(2), m.group(3) or "revenue")),
    ]
    
    for pattern, extractor in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value, scale_str, metric = extractor(match)
                
                # Normalize to base units
                value_raw = normalize_to_base_units(value, scale_str)
                
                # Check if we already have this claim
                claim_text = match.group(0)
                if claim_text in seen_texts:
                    continue
                seen_texts.add(claim_text)
                
                claims.append(Claim(
                    text=claim_text,
                    source=source,
                    source_tier=source_tier,
                    claim_type="quantitative",
                    metric=metric.lower() if metric else "revenue",
                    value=value,
                    value_raw=value_raw,
                    unit_hint=scale_str.lower() if scale_str else None,
                ))
            except (ValueError, AttributeError):
                continue
    
    return claims


# =============================================================================
# Verification Logic
# =============================================================================


def _get_metric_aliases(metric: Optional[str]) -> List[str]:
    """Get all aliases for a metric name."""
    if not metric:
        return []
    
    metric_lower = metric.lower()
    
    aliases_map = {
        "revenue": ["revenue", "revenues", "sales", "total revenue", "net revenue"],
        "income": ["income", "net income", "net earnings", "profit"],
        "profit": ["profit", "net income", "gross profit", "operating profit"],
        "earnings": ["earnings", "net income", "net earnings"],
        "eps": ["eps", "earnings per share", "diluted eps"],
    }
    
    for key, alias_list in aliases_map.items():
        if key in metric_lower or metric_lower in alias_list:
            return alias_list
    
    return [metric_lower]


def _find_matching_fact(claim: Claim, fact_store: "FactStore") -> Optional[Fact]:
    """Find a fact in the store that matches the claim."""
    all_facts = fact_store.get_all_facts()
    
    # Filter by entity if specified
    if claim.entity:
        all_facts = [f for f in all_facts if f.entity.upper() == claim.entity.upper()]
    
    # Find matching metric
    metric_aliases = _get_metric_aliases(claim.metric)
    
    for fact in all_facts:
        fact_metric = fact.metric.lower() if fact.metric else ""
        if any(alias in fact_metric for alias in metric_aliases):
            return fact
    
    return None


def _compare_values(
    claim: Claim, 
    claim_value: float, 
    fact_value: float,
    fact: Fact
) -> VerificationResult:
    """Compare normalized values and determine verification status."""
    
    if fact_value == 0:
        return VerificationResult(
            claim=claim,
            status="unverifiable",
            hard_source=f"SEC {fact.source_format.upper()}",
            hard_value=fact_value,
            explanation="Cannot verify against zero value."
        )
    
    # Calculate percentage difference
    difference = abs(claim_value - fact_value)
    difference_pct = difference / fact_value
    
    # Determine status based on tolerance
    if difference_pct <= EXACT_TOLERANCE:
        status = "verified"
        explanation = f"Claim matches SEC data (within {difference_pct:.1%})."
    elif difference_pct <= CLOSE_TOLERANCE:
        status = "close"
        explanation = f"Claim is within {difference_pct:.1%} of SEC data (likely rounding)."
    else:
        status = "contradicted"
        explanation = f"Claim differs from SEC data by {difference_pct:.1%}."
    
    return VerificationResult(
        claim=claim,
        status=status,
        hard_source=f"SEC {fact.source_format.upper()}",
        hard_value=fact_value,
        hard_period=fact.period,
        difference_pct=difference_pct * 100,
        confidence=max(0, 1.0 - difference_pct),
        explanation=explanation
    )


def verify_claim(claim: Claim, fact_store: "FactStore") -> VerificationResult:
    """
    Verify a claim against hard sources in the FactStore.
    
    Args:
        claim: The claim to verify
        fact_store: Store containing verified facts
        
    Returns:
        VerificationResult with status and explanation
    """
    if claim.claim_type != "quantitative":
        return VerificationResult(
            claim=claim,
            status="unverifiable",
            explanation="Only quantitative claims can be verified against SEC data."
        )
    
    if claim.value_raw is None:
        return VerificationResult(
            claim=claim,
            status="unverifiable",
            explanation="Could not parse numeric value from claim."
        )
    
    # Find matching fact
    matching_fact = _find_matching_fact(claim, fact_store)
    
    if not matching_fact:
        return VerificationResult(
            claim=claim,
            status="unverifiable",
            confidence=0.0,
            explanation=f"No SEC data found for '{claim.metric}' to verify against."
        )
    
    # Get fact value (already in base units from XBRL)
    if matching_fact.value is None:
        return VerificationResult(
            claim=claim,
            status="unverifiable",
            explanation="SEC fact has no numeric value."
        )
    
    fact_value = matching_fact.value
    
    # Compare normalized values
    return _compare_values(claim, claim.value_raw, fact_value, matching_fact)


def verify_all_claims(
    claims: List[Claim], 
    fact_store: "FactStore"
) -> List[VerificationResult]:
    """
    Verify all claims against fact store.
    
    Args:
        claims: List of claims to verify
        fact_store: Store containing verified facts
        
    Returns:
        List of VerificationResult objects
    """
    return [verify_claim(claim, fact_store) for claim in claims]


# =============================================================================
# Report Formatting
# =============================================================================


def format_verification_report(results: List[VerificationResult]) -> str:
    """
    Format verification results as a professional report.
    
    Args:
        results: List of VerificationResult objects
        
    Returns:
        Formatted markdown/text report
    """
    lines = [
        "━" * 60,
        "📋 CLAIM VERIFICATION REPORT",
        "━" * 60,
        ""
    ]
    
    verified = [r for r in results if r.status == "verified"]
    close = [r for r in results if r.status == "close"]
    contradicted = [r for r in results if r.status == "contradicted"]
    unverifiable = [r for r in results if r.status == "unverifiable"]
    
    # Summary
    lines.append(f"Total Claims Analyzed: {len(results)}")
    lines.append(f"🟢 Verified:      {len(verified)}")
    lines.append(f"🟡 Close Match:   {len(close)}")
    lines.append(f"🔴 Contradicted:  {len(contradicted)}")
    lines.append(f"⚪ Unverifiable:  {len(unverifiable)}")
    lines.append("")
    
    # Contradicted claims (most important)
    if contradicted:
        lines.append("━" * 60)
        lines.append("🔴 CONTRADICTED CLAIMS")
        lines.append("━" * 60)
        for r in contradicted:
            lines.append(f"  Claim: \"{r.claim.text}\"")
            lines.append(f"  Source: {r.claim.source}")
            if r.claim.value_raw is not None:
                lines.append(f"  Claim Value: ${r.claim.value_raw:,.0f}")
            if r.hard_value is not None:
                lines.append(f"  SEC Value:   ${r.hard_value:,.0f}")
            if r.difference_pct is not None:
                lines.append(f"  Difference:  {r.difference_pct:.1f}%")
            lines.append("")
    
    # Close matches
    if close:
        lines.append("━" * 60)
        lines.append("🟡 CLOSE MATCHES (likely rounding)")
        lines.append("━" * 60)
        for r in close:
            lines.append(f"  ≈ \"{r.claim.text}\"")
            if r.difference_pct is not None:
                lines.append(f"    {r.difference_pct:.1f}% difference from {r.hard_source}")
        lines.append("")
    
    # Verified claims
    if verified:
        lines.append("━" * 60)
        lines.append("🟢 VERIFIED CLAIMS")
        lines.append("━" * 60)
        for r in verified:
            lines.append(f"  ✓ \"{r.claim.text}\"")
            if r.difference_pct is not None:
                lines.append(f"    Matches {r.hard_source} ({r.difference_pct:.1f}% diff)")
        lines.append("")
    
    lines.append("━" * 60)
    
    return "\n".join(lines)



