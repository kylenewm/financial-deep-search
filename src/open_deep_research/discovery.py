"""
Discovery System - Find leads from web/news for investigation.

Discovery generates hypotheses. Verification tests them.
These are separate steps.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from open_deep_research.config import ResearchConfig, default_config
from open_deep_research.cross_verify import (
    extract_claims_from_text,
    detect_scale_from_text,
    normalize_to_base_units,
)
from open_deep_research.models import Lead, Claim, Fact, get_domain_tier

if TYPE_CHECKING:
    from open_deep_research.store import FactStore
    from open_deep_research.models import DiscoveryReport

logger = logging.getLogger(__name__)


def discover(
    query: str,
    config: ResearchConfig = default_config,
    ticker: Optional[str] = None,
) -> List[Lead]:
    """
    Discover leads by searching web/news for a query.
    
    Returns list of Lead objects with extracted claims.
    Does NOT verify - that's a separate step.
    
    Args:
        query: Search query
        config: Research configuration
        ticker: Optional ticker to focus search
        
    Returns:
        List of Lead objects
    """
    if not config.discovery_enabled:
        logger.warning("Discovery is disabled. Set DISCOVERY_ENABLED=true to enable.")
        return []
    
    if config.discovery_backend == "news":
        return _discover_via_news(query, ticker)
    elif config.discovery_backend == "deep_research":
        return _discover_via_deep_research(query, ticker)
    else:
        logger.warning(f"Unknown discovery backend: {config.discovery_backend}")
        return []


def _discover_via_news(query: str, ticker: Optional[str] = None) -> List[Lead]:
    """Discover leads using news search (tier 1/2 sources)."""
    try:
        from open_deep_research.news_search import search_news
        
        # Search for news
        search_query = f"{ticker} {query}" if ticker else query
        news_results = search_news(search_query, num_results=10)
        
        # Extract leads from news snippets
        leads = []
        for result in news_results:
            article_leads = discover_from_text(
                text=result.snippet,
                source_name=result.domain,
                source_url=result.url,
                source_tier=result.tier,
                entity=ticker,
            )
            leads.extend(article_leads)
        
        return leads
        
    except ImportError:
        logger.warning("news_search not available")
        return []
    except Exception as e:
        logger.error(f"News discovery failed: {e}")
        return []


def _discover_via_deep_research(query: str, ticker: Optional[str] = None) -> List[Lead]:
    """Discover leads using deep research (full web search)."""
    try:
        # Import deep researcher lazily
        from src.legacy.graph import graph as deep_researcher
        
        # Run deep research
        result = deep_researcher.invoke({"topic": query})
        report = result.get("final_report", "")
        
        if not report:
            return []
        
        # Extract leads from the report
        leads = discover_from_text(
            text=report,
            source_name="Deep Research",
            source_url=None,
            source_tier=3,  # Web search is tier 3
            entity=ticker,
        )
        
        return leads
        
    except ImportError:
        logger.warning("deep_researcher not available")
        return []
    except Exception as e:
        logger.error(f"Deep research discovery failed: {e}")
        return []


def discover_from_text(
    text: str,
    source_name: str,
    source_url: Optional[str] = None,
    source_tier: int = 3,
    entity: Optional[str] = None,
) -> List[Lead]:
    """
    Extract leads from a block of text (e.g., news article).
    
    Useful when analyst has specific text to analyze.
    
    Args:
        text: Text to extract leads from
        source_name: Name of source (e.g., "Reuters")
        source_url: URL of source
        source_tier: Trust tier (1-3)
        entity: Optional ticker hint
        
    Returns:
        List of Lead objects
    """
    leads = []
    now = datetime.now()
    
    # Use existing claim extraction
    claims = extract_claims_from_text(text, source_name, source_tier="soft")
    
    for claim in claims:
        lead = Lead(
            lead_id=str(uuid.uuid4())[:8],
            text=claim.text,
            source_url=source_url,
            source_name=source_name,
            source_tier=source_tier,
            found_at=now,
            entity=entity or claim.entity,
            metric=claim.metric,
            value=claim.value,
            value_raw=claim.value_raw,
            period=claim.period,
            lead_type="quantitative" if claim.claim_type == "quantitative" else "qualitative",
            verification_status="pending",
        )
        leads.append(lead)
    
    return leads


def format_leads(leads: List[Lead]) -> str:
    """Format leads as a readable list."""
    if not leads:
        return "No leads found."
    
    lines = [
        "━" * 50,
        f"🔍 DISCOVERED {len(leads)} LEADS",
        "━" * 50,
        "",
    ]
    
    for i, lead in enumerate(leads, 1):
        icon = lead.get_status_icon()
        lines.append(f"{i}. {icon} \"{lead.text}\"")
        lines.append(f"   Source: {lead.source_name} (Tier {lead.source_tier})")
        if lead.metric and lead.value_raw:
            lines.append(f"   Extracted: {lead.metric} = ${lead.value_raw:,.0f}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Lead Verification
# =============================================================================


# Tolerance for matching
CONFIRM_TOLERANCE = 0.02   # 2% = confirmed
CONTRADICT_TOLERANCE = 0.05  # >5% = contradicted


def verify_lead(lead: Lead, fact_store: "FactStore") -> Lead:
    """
    Verify a single lead against FactStore.
    
    Updates lead.verification_status:
    - "confirmed": Lead value matches FactStore (within 2%)
    - "contradicted": Lead value differs from FactStore (>5%)
    - "unverifiable": No matching fact in FactStore, or non-quantitative
    
    Args:
        lead: The lead to verify
        fact_store: Store containing verified facts
        
    Returns:
        Lead with updated verification_status
    """
    # Non-quantitative leads can't be verified against numbers
    if lead.lead_type != "quantitative" or lead.value_raw is None:
        lead.verification_status = "unverifiable"
        lead.verification_details = "Non-quantitative claim"
        return lead
    
    # Find matching fact in store
    matching_fact = _find_matching_fact_for_lead(lead, fact_store)
    
    if not matching_fact or matching_fact.value is None:
        lead.verification_status = "unverifiable"
        lead.verification_details = f"No SEC data found for {lead.metric or 'this metric'}"
        return lead
    
    # Compare values
    fact_value = matching_fact.value
    lead_value = lead.value_raw
    
    if fact_value == 0:
        lead.verification_status = "unverifiable"
        lead.verification_details = "Cannot compare against zero value"
        return lead
    
    difference_pct = abs(lead_value - fact_value) / fact_value
    
    if difference_pct <= CONFIRM_TOLERANCE:
        lead.verification_status = "confirmed"
        lead.verification_details = f"Matches SEC: ${fact_value:,.0f} ({difference_pct:.1%} diff)"
        lead.verified_against_fact_id = matching_fact.fact_id
    elif difference_pct <= CONTRADICT_TOLERANCE:
        # Close but not exact - still mark as confirmed with note
        lead.verification_status = "confirmed"
        lead.verification_details = f"Close to SEC: ${fact_value:,.0f} ({difference_pct:.1%} diff, likely rounding)"
        lead.verified_against_fact_id = matching_fact.fact_id
    else:
        lead.verification_status = "contradicted"
        lead.verification_details = f"SEC shows ${fact_value:,.0f} ({difference_pct:.1%} difference)"
        lead.verified_against_fact_id = matching_fact.fact_id
    
    return lead


def _find_matching_fact_for_lead(lead: Lead, fact_store: "FactStore") -> Optional[Fact]:
    """Find a fact in the store that matches the lead."""
    all_facts = fact_store.get_all_facts()
    
    # Filter by entity if specified
    if lead.entity:
        all_facts = [f for f in all_facts if f.entity.upper() == lead.entity.upper()]
    
    if not all_facts:
        return None
    
    # Find matching metric
    if not lead.metric:
        return None
    
    metric_aliases = _get_metric_aliases(lead.metric)
    
    for fact in all_facts:
        fact_metric = fact.metric.lower() if fact.metric else ""
        if any(alias in fact_metric for alias in metric_aliases):
            return fact
    
    return None


def _get_metric_aliases(metric: str) -> List[str]:
    """Get aliases for a metric name."""
    if not metric:
        return []
    
    metric_lower = metric.lower()
    
    aliases_map = {
        "revenue": ["revenue", "revenues", "sales", "total revenue", "net revenue"],
        "income": ["income", "net income", "net earnings", "profit"],
        "profit": ["profit", "net income", "gross profit", "operating profit"],
        "earnings": ["earnings", "net income", "net earnings"],
        "eps": ["eps", "earnings per share", "diluted eps"],
        "margin": ["margin", "gross margin", "operating margin"],
    }
    
    for key, alias_list in aliases_map.items():
        if key in metric_lower or metric_lower in alias_list:
            return alias_list
    
    return [metric_lower]


def verify_leads(leads: List[Lead], fact_store: "FactStore") -> List[Lead]:
    """
    Verify all leads against FactStore.
    
    Args:
        leads: List of leads to verify
        fact_store: Store containing verified facts
        
    Returns:
        List of leads with updated verification_status
    """
    return [verify_lead(lead, fact_store) for lead in leads]


def discover_and_verify(
    query: str,
    fact_store: "FactStore",
    config: ResearchConfig = default_config,
    ticker: Optional[str] = None,
) -> "DiscoveryReport":
    """
    Full pipeline: discover leads, then verify against FactStore.
    
    Args:
        query: What to search for
        fact_store: Store containing verified facts
        config: Research configuration
        ticker: Optional ticker to focus search
        
    Returns:
        DiscoveryReport with verified leads
    """
    from open_deep_research.models import DiscoveryReport
    
    # Step 1: Discover leads
    leads = discover(query, config=config, ticker=ticker)
    
    # Step 2: Verify against FactStore
    verified_leads = verify_leads(leads, fact_store)
    
    # Step 3: Build report
    report = DiscoveryReport(
        query=query,
        ticker=ticker,
        leads=verified_leads,
        generated_at=datetime.now(),
    )
    
    return report

