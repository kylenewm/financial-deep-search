"""
XBRL data fetcher for SEC filings.

Fetches structured financial data from SEC's XBRL API.
This is deterministic - no LLM calls, no verification needed.
The data comes pre-structured from SEC.

API endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_deep_research.models import DocumentSnapshot, Fact, Location
from open_deep_research.entities import pad_cik

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# SEC XBRL API endpoint
XBRL_API_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# Default cache directory
DEFAULT_XBRL_CACHE_DIR = ".cache/xbrl"

# Rate limiting - SEC allows 10 req/sec, we use 5 for safety
SEC_RATE_LIMIT_DELAY = 0.2


# =============================================================================
# XBRL Concept Mappings
# =============================================================================

# Map human-readable metric names to us-gaap XBRL concepts
# Note: Companies may use different concepts for the same metric
METRIC_TO_CONCEPTS: Dict[str, List[str]] = {
    # Revenue
    "total revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "NetRevenues",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "NetRevenues",
    ],
    "net revenue": [
        "Revenues",
        "NetRevenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "sales": [
        "Revenues",
        "SalesRevenueNet",
    ],
    "net sales": [
        "Revenues",
        "SalesRevenueNet",
    ],
    
    # Income
    "net income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "net profit": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "gross profit": [
        "GrossProfit",
    ],
    "gross margin": [
        "GrossProfit",
    ],
    "operating income": [
        "OperatingIncomeLoss",
    ],
    "operating profit": [
        "OperatingIncomeLoss",
    ],
    "income from operations": [
        "OperatingIncomeLoss",
    ],
    "operating earnings": [
        "OperatingIncomeLoss",
    ],
    
    # Costs
    "cost of revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "cost of revenues": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "cost of sales": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "cost of goods sold": [
        "CostOfGoodsSold",
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ],
    "cogs": [
        "CostOfGoodsSold",
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ],
    "r&d expenses": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "research and development": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "selling, general and administrative": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    "operating expenses": [
        "OperatingExpenses",
    ],
    "opex": [
        "OperatingExpenses",
    ],
    
    # EPS
    "earnings per share": [
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
    ],
    "eps": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "diluted eps": [
        "EarningsPerShareDiluted",
    ],
    "basic eps": [
        "EarningsPerShareBasic",
    ],
    
    # Balance Sheet Items
    "total assets": [
        "Assets",
    ],
    "assets": [
        "Assets",
    ],
    "total liabilities": [
        "Liabilities",
    ],
    "liabilities": [
        "Liabilities",
    ],
    "stockholders equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "shareholders equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    
    # Cash & Liquidity
    "cash and cash equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    
    # Debt
    "long term debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
    ],
    "short term debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
    ],
    "total debt": [
        "DebtInstrumentCarryingAmount", # Often calculated, but check for explicit
    ],
}

# Reverse mapping: concept -> metric name (for display)
CONCEPT_TO_METRIC: Dict[str, str] = {}
for metric, concepts in METRIC_TO_CONCEPTS.items():
    for concept in concepts:
        if concept not in CONCEPT_TO_METRIC:
            CONCEPT_TO_METRIC[concept] = metric


# =============================================================================
# HTTP Session
# =============================================================================


def get_xbrl_session() -> requests.Session:
    """Create a requests session with retry logic for SEC API."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    # SEC requires User-Agent
    user_agent = os.environ.get("SEC_USER_AGENT", "")
    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    else:
        # Fallback - should be set properly
        session.headers.update({
            "User-Agent": "ResearchAgent research@example.com"
        })
    
    return session


# =============================================================================
# XBRL Fetching
# =============================================================================


def fetch_company_facts(
    cik: str,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch all XBRL facts for a company from SEC API.
    
    Args:
        cik: 10-digit zero-padded CIK
        cache_dir: Directory for caching (default: .cache/xbrl/)
        force_refresh: If True, bypass cache and fetch fresh data
        
    Returns:
        Complete company facts dictionary from SEC API
        
    Structure:
        {
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
                                    ...
                                }
                            ]
                        }
                    }
                }
            }
        }
    """
    # Ensure CIK is padded
    cik_padded = pad_cik(cik)
    
    # Setup cache
    if cache_dir is None:
        cache_dir = DEFAULT_XBRL_CACHE_DIR
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cached_file = cache_path / f"CIK{cik_padded}_facts.json"
    
    # Check cache (unless force refresh)
    if not force_refresh and cached_file.exists():
        # Check if cache is less than 24 hours old
        cache_age = time.time() - cached_file.stat().st_mtime
        if cache_age < 86400:  # 24 hours
            logger.info(f"Using cached XBRL data for CIK {cik_padded}")
            with open(cached_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # Fetch from SEC API
    url = f"{XBRL_API_BASE}/CIK{cik_padded}.json"
    logger.info(f"Fetching XBRL data from {url}")
    
    session = get_xbrl_session()
    time.sleep(SEC_RATE_LIMIT_DELAY)  # Rate limiting
    
    # P0 Fix: Add explicit timeout to prevent hanging on slow/unreachable SEC API
    response = session.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Cache the response
    with open(cached_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Cached XBRL data for {data.get('entityName', cik_padded)}")
    
    return data


def get_concept_data(
    facts_data: Dict[str, Any],
    concept: str,
    taxonomy: str = "us-gaap",
) -> Optional[Dict[str, Any]]:
    """
    Get data for a specific XBRL concept.
    
    Args:
        facts_data: The full company facts from fetch_company_facts()
        concept: XBRL concept name (e.g., "Revenues", "NetIncomeLoss")
        taxonomy: XBRL taxonomy (default: "us-gaap")
        
    Returns:
        Concept data dict with "label", "units", etc., or None if not found
    """
    facts = facts_data.get("facts", {})
    taxonomy_facts = facts.get(taxonomy, {})
    return taxonomy_facts.get(concept)


def find_fact_by_period(
    concept_data: Dict[str, Any],
    period_end: str,
    form: Optional[str] = None,
    unit: str = "USD",
) -> Optional[Dict[str, Any]]:
    """
    Find a specific fact value by period end date.
    
    Args:
        concept_data: Data for a specific concept from get_concept_data()
        period_end: Period end date in format "YYYY-MM-DD"
        form: Optional form type filter ("10-K", "10-Q")
        unit: Unit type (default: "USD")
        
    Returns:
        The fact entry dict, or None if not found
    """
    units_data = concept_data.get("units", {})
    values = units_data.get(unit, [])
    
    for entry in values:
        entry_end = entry.get("end", "")
        entry_form = entry.get("form", "")
        
        if entry_end == period_end:
            if form is None or entry_form == form:
                return entry
    
    return None


def _calculate_duration_months(start: str, end: str) -> int:
    """Calculate approximate duration in months between two dates."""
    from datetime import datetime
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        return round(days / 30)
    except Exception:
        return 0


def find_facts_by_fiscal_period(
    concept_data: Dict[str, Any],
    fiscal_year: int,
    fiscal_period: str,
    unit: str = "USD",
    quarterly_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find facts by fiscal year and period (FY, Q1, Q2, Q3).
    
    Args:
        concept_data: Data for a specific concept
        fiscal_year: Fiscal year (e.g., 2025)
        fiscal_period: "FY", "Q1", "Q2", "Q3"
        unit: Unit type (default: "USD")
        quarterly_only: If True and period is Q1/Q2/Q3, only return 3-month entries
        
    Returns:
        List of matching fact entries, sorted by end date (most recent first)
    """
    units_data = concept_data.get("units", {})
    values = units_data.get(unit, [])
    
    matches = []
    for entry in values:
        if entry.get("fy") == fiscal_year and entry.get("fp") == fiscal_period:
            # For quarterly periods, we may have cumulative (9-month) vs quarterly (3-month)
            # Filter to get just the quarterly value
            if quarterly_only and fiscal_period in ("Q1", "Q2", "Q3"):
                start = entry.get("start", "")
                end = entry.get("end", "")
                if start and end:
                    months = _calculate_duration_months(start, end)
                    # Quarterly entries should be ~3 months
                    if months > 5:  # Skip cumulative entries (6, 9 months)
                        continue
            matches.append(entry)
    
    # Sort by end date descending (most recent first) to get current quarter, not prior year comparison
    matches.sort(key=lambda x: x.get("end", ""), reverse=True)
    
    return matches


# =============================================================================
# High-Level Extraction
# =============================================================================


def extract_xbrl_fact(
    cik: str,
    metric: str,
    period_end: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    fiscal_period: Optional[str] = None,
    form: Optional[str] = None,
    cache_dir: Optional[str] = None,
    ticker: Optional[str] = None,
) -> Optional[Fact]:
    """
    Extract a financial fact from XBRL data.
    
    This is the main entry point for XBRL extraction. Specify EITHER:
    - period_end: Exact date like "2024-10-27"
    - fiscal_year + fiscal_period: Like (2025, "Q3")
    
    Args:
        cik: Company CIK (will be zero-padded)
        metric: Human-readable metric name (e.g., "total revenue", "net income")
        period_end: Period end date (YYYY-MM-DD)
        fiscal_year: Fiscal year number
        fiscal_period: "FY", "Q1", "Q2", "Q3"
        form: Filter by form type ("10-K", "10-Q")
        cache_dir: Cache directory for XBRL data
        
    Returns:
        Fact object if found, None otherwise
    """
    # Normalize metric name
    metric_lower = metric.lower().strip()
    
    # Get XBRL concepts to search
    concepts = METRIC_TO_CONCEPTS.get(metric_lower)
    if not concepts:
        logger.warning(f"No XBRL concept mapping for metric: {metric}")
        return None
    
    # Fetch company facts
    try:
        facts_data = fetch_company_facts(cik, cache_dir)
    except Exception as e:
        logger.error(f"Failed to fetch XBRL data for CIK {cik}: {e}")
        return None
    
    entity_name = facts_data.get("entityName", "")
    
    # Try each concept until we find data
    for concept in concepts:
        concept_data = get_concept_data(facts_data, concept)
        if not concept_data:
            continue
        
        # Try each available unit in the concept data
        units_data = concept_data.get("units", {})
        if not units_data:
            continue
            
        # Prioritize USD, then USD/shares, then others
        unit_priority = ["USD", "USD/shares", "shares"]
        available_units = list(units_data.keys())
        sorted_units = sorted(
            available_units,
            key=lambda x: unit_priority.index(x) if x in unit_priority else 999
        )
        
        entry = None
        used_unit = None
        
        for unit in sorted_units:
            # Find the specific fact
            if period_end:
                match = find_fact_by_period(concept_data, period_end, form, unit=unit)
                if match:
                    entry = match
                    used_unit = unit
                    break
            elif fiscal_year and fiscal_period:
                matches = find_facts_by_fiscal_period(
                    concept_data, fiscal_year, fiscal_period, unit=unit
                )
                # If form specified, filter
                if form:
                    matches = [m for m in matches if m.get("form") == form]
                
                if matches:
                    entry = matches[0]
                    used_unit = unit
                    break
        
        if entry and used_unit:
            # Build the Fact
            return _build_fact_from_xbrl(
                entry=entry,
                concept=concept,
                unit=used_unit,
                entity_name=entity_name,
                cik=cik,
                metric_name=metric,
                ticker=ticker,
            )
    
    logger.info(f"No XBRL data found for {metric} in CIK {cik}")
    return None


def extract_all_facts_for_period(
    cik: str,
    period_end: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    fiscal_period: Optional[str] = None,
    form: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Fact]:
    """
    Extract all available financial facts for a specific period.
    
    Args:
        cik: Company CIK
        period_end: Period end date (YYYY-MM-DD)
        fiscal_year: Fiscal year number
        fiscal_period: "FY", "Q1", "Q2", "Q3"
        form: Filter by form type
        cache_dir: Cache directory
        
    Returns:
        List of all Fact objects found for the period
    """
    facts = []
    
    for metric in METRIC_TO_CONCEPTS.keys():
        fact = extract_xbrl_fact(
            cik=cik,
            metric=metric,
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            form=form,
            cache_dir=cache_dir,
        )
        if fact:
            facts.append(fact)
    
    return facts


# =============================================================================
# Fact Building
# =============================================================================


def _build_fact_from_xbrl(
    entry: Dict[str, Any],
    concept: str,
    unit: str,
    entity_name: str,
    cik: str,
    metric_name: str,
    ticker: Optional[str] = None,
) -> Fact:
    """Build a Fact object from XBRL entry data."""
    # Generate IDs
    fact_id = str(uuid.uuid4())
    snapshot_id = str(uuid.uuid4())
    
    # Extract period info
    period_end = entry.get("end", "")
    fiscal_year = entry.get("fy")
    fiscal_period = entry.get("fp", "")
    form = entry.get("form", "")
    filed = entry.get("filed", "")
    
    # Build period string (e.g., "Q3 FY2025")
    if fiscal_period and fiscal_year:
        if fiscal_period == "FY":
            period = f"FY{fiscal_year}"
        else:
            period = f"{fiscal_period} FY{fiscal_year}"
    else:
        period = period_end
    
    # Value
    value = entry.get("val")
    if value is not None:
        value = float(value)
    
    # Normalize unit display
    unit_display = unit
    if unit == "USD":
        unit_display = "USD"
    elif unit == "USD/shares":
        unit_display = "USD/share"
    elif unit == "shares":
        unit_display = "shares"
    
    # Build location
    location = Location(
        cik=pad_cik(cik),
        doc_date=filed,
        doc_type=form,
        section_id="xbrl",  # Special section for XBRL facts
        paragraph_index=None,
        sentence_string=f"XBRL: {concept} = {value} {unit}",  # For traceability
    )
    
    # Compute doc hash from XBRL data
    hash_input = f"{cik}:{concept}:{period_end}:{value}"
    doc_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    # Use ticker if provided, otherwise fall back to entity name
    entity_value = ticker.upper() if ticker else entity_name
    
    return Fact(
        fact_id=fact_id,
        entity=entity_value,
        metric=metric_name.title(),  # Capitalize for display
        value=value,
        unit=unit_display,
        period=period,
        period_end_date=period_end,
        location=location,
        source_format="xbrl",  # New source format
        extracted_scale=None,  # XBRL values are already in base units
        doc_hash=doc_hash,
        snapshot_id=snapshot_id,
        verification_status="exact_match",  # XBRL is authoritative - no verification needed
    )


# =============================================================================
# Utility Functions
# =============================================================================


def list_available_concepts(
    cik: str,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    List all us-gaap concepts available for a company.
    
    Useful for debugging and discovering what data is available.
    
    Args:
        cik: Company CIK
        cache_dir: Cache directory
        
    Returns:
        List of concept names
    """
    facts_data = fetch_company_facts(cik, cache_dir)
    facts = facts_data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    return sorted(us_gaap.keys())


def extract_all_facts(
    ticker: str,
    period: str,
    cache_dir: Optional[str] = None,
) -> List[Fact]:
    """
    Extract all available XBRL facts for a ticker and period.
    
    This is a convenience wrapper around extract_all_facts_for_period
    that takes a ticker instead of CIK and handles period parsing.
    
    Args:
        ticker: Stock ticker (e.g., "NVDA", "AAPL")
        period: Fiscal period string (e.g., "Q3 FY2025", "FY2024")
        cache_dir: Cache directory for XBRL data
        
    Returns:
        List of Fact objects for all available metrics
    """
    from open_deep_research.entities import resolve_entity
    
    # Resolve ticker to CIK
    entity = resolve_entity(ticker)
    if not entity:
        logger.warning(f"Could not resolve ticker: {ticker}")
        return []
    
    # Parse period
    period_upper = period.upper().strip()
    fiscal_year = None
    fiscal_period = None
    
    # Pattern: "Q3 FY2025" or "Q3FY2025"
    import re
    match = re.match(r'(Q[1-4])?\s*FY(\d{4})', period_upper)
    if match:
        fiscal_period = match.group(1) if match.group(1) else "FY"
        fiscal_year = int(match.group(2))
    else:
        # Pattern: "Q3 2025"
        match = re.match(r'(Q[1-4])\s*(\d{4})', period_upper)
        if match:
            fiscal_period = match.group(1)
            fiscal_year = int(match.group(2))
    
    if not fiscal_year or not fiscal_period:
        logger.warning(f"Could not parse period: {period}")
        return []
    
    # Extract facts with ticker for entity field
    facts = []
    for metric in METRIC_TO_CONCEPTS.keys():
        fact = extract_xbrl_fact(
            cik=entity.cik,
            metric=metric,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            cache_dir=cache_dir,
            ticker=ticker,
        )
        if fact:
            facts.append(fact)
    
    logger.info(f"Extracted {len(facts)} facts for {ticker} {period}")
    return facts


def get_latest_value(
    cik: str,
    metric: str,
    cache_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent value for a metric.
    
    Args:
        cik: Company CIK
        metric: Metric name
        cache_dir: Cache directory
        
    Returns:
        Dict with "value", "period_end", "form", "filed" or None
    """
    metric_lower = metric.lower().strip()
    concepts = METRIC_TO_CONCEPTS.get(metric_lower, [])
    
    if not concepts:
        return None
    
    facts_data = fetch_company_facts(cik, cache_dir)
    
    for concept in concepts:
        concept_data = get_concept_data(facts_data, concept)
        if not concept_data:
            continue
        
        # Get USD values and find most recent by filed date
        units_data = concept_data.get("units", {})
        values = units_data.get("USD", [])
        
        if values:
            # Sort by filed date (most recent first)
            sorted_values = sorted(
                values,
                key=lambda x: x.get("filed", ""),
                reverse=True,
            )
            latest = sorted_values[0]
            return {
                "value": latest.get("val"),
                "period_end": latest.get("end"),
                "form": latest.get("form"),
                "filed": latest.get("filed"),
                "fiscal_year": latest.get("fy"),
                "fiscal_period": latest.get("fp"),
            }
    
    return None

