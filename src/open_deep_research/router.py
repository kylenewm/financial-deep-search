"""
Source Router for query routing.

Routes queries to the appropriate extraction path:
- XBRL path: Structured financial data from SEC (no LLM, no verification)
- LLM path: Text-based data requiring LLM extraction + verification gate

This is the key decision point for the anti-hallucination architecture.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class SourceType(Enum):
    """Type of source to use for extraction."""
    XBRL = "xbrl"          # SEC XBRL API (structured, deterministic)
    SEC_HTML = "sec_html"  # SEC filing HTML (LLM + verification)
    NEWS = "news"          # News articles (LLM + verification)
    WEBSITE = "website"    # General websites (LLM + verification)


class DataType(Enum):
    """Type of data being requested."""
    FINANCIAL_METRIC = "financial_metric"   # Numbers from financial statements
    SEGMENT_METRIC = "segment_metric"       # Segment-level breakdowns
    QUALITATIVE = "qualitative"             # Text/commentary/guidance
    EVENT = "event"                         # Discrete events (earnings calls, announcements)


@dataclass
class RouteResult:
    """Result of routing a query."""
    source_type: SourceType
    data_type: DataType
    metric_name: Optional[str]  # Normalized metric name if applicable
    requires_verification: bool
    confidence: float  # 0.0-1.0, how confident we are in this routing


from open_deep_research.xbrl import METRIC_TO_CONCEPTS


# =============================================================================
# Metric Keyword Sets
# =============================================================================


# Core financial metrics available in XBRL - dynamically loaded from xbrl.py
XBRL_METRICS: Set[str] = set(METRIC_TO_CONCEPTS.keys())

# Segment metrics - may be in XBRL or may need HTML extraction
SEGMENT_METRICS: Set[str] = {
    "datacenter revenue", "data center revenue",
    "gaming revenue",
    "professional visualization revenue",
    "automotive revenue",
    "oem revenue",
    "compute revenue",
    "networking revenue",
    "segment revenue",
}

# Qualitative terms that require LLM extraction
QUALITATIVE_KEYWORDS: Set[str] = {
    "guidance", "outlook", "forecast", "projection", "expectation",
    "commentary", "discussion", "analysis",
    "risk", "risks", "risk factor",
    "strategy", "strategic", "plan", "initiative",
    "competition", "competitive", "market position",
    "trend", "trends", "growth driver",
    "why", "how", "explain", "describe", "what caused",
    "management", "said", "stated", "announced", "mentioned",
}

# Event-related keywords
EVENT_KEYWORDS: Set[str] = {
    "earnings call", "conference call",
    "announcement", "announced",
    "dividend", "buyback", "repurchase",
    "acquisition", "merger", "deal",
    "lawsuit", "litigation", "settlement",
}


# =============================================================================
# Routing Logic
# =============================================================================


def normalize_metric_name(text: str) -> str:
    """Normalize a metric name for matching."""
    text = text.lower().strip()
    # Remove common suffixes
    text = re.sub(r"\s*(for|in|during|of|q[1-4]|fy\d+|20\d{2}).*$", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metric_from_question(question: str) -> Optional[str]:
    """
    Extract the metric name from a question.
    
    Args:
        question: The full question text
        
    Returns:
        The extracted metric name, or None if not found
    """
    question_lower = question.lower()
    
    # Check for exact XBRL metrics
    for metric in sorted(XBRL_METRICS, key=len, reverse=True):
        if metric in question_lower:
            return metric
    
    # Check for segment metrics
    for metric in sorted(SEGMENT_METRICS, key=len, reverse=True):
        if metric in question_lower:
            return metric
    
    # Try to extract from common question patterns
    patterns = [
        r"what (?:was|is|were) (?:the )?(?:company'?s?|nvidia'?s?|[\w]+'?s?)?\s*(.+?)(?:\s+in\s+|\s+for\s+|\s+during\s+|\?|$)",
        r"how much (?:was|is|did) (?:the )?(?:company'?s?|[\w]+'?s?)?\s*(.+?)(?:\s+in\s+|\?|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            extracted = normalize_metric_name(match.group(1))
            # Check if extracted text contains known metrics
            for metric in XBRL_METRICS | SEGMENT_METRICS:
                if metric in extracted:
                    return metric
    
    return None


def detect_data_type(question: str, metric: Optional[str]) -> DataType:
    """
    Detect the type of data being requested.
    
    Args:
        question: The full question text
        metric: Extracted metric name, if any
        
    Returns:
        DataType indicating what kind of data is needed
    """
    question_lower = question.lower()
    
    # Check for qualitative indicators
    for keyword in QUALITATIVE_KEYWORDS:
        if keyword in question_lower:
            return DataType.QUALITATIVE
    
    # Check for event indicators
    for keyword in EVENT_KEYWORDS:
        if keyword in question_lower:
            return DataType.EVENT
    
    # Check if metric is a segment metric
    if metric and metric in SEGMENT_METRICS:
        return DataType.SEGMENT_METRIC
    
    # Default to financial metric if we have a metric
    if metric and metric in XBRL_METRICS:
        return DataType.FINANCIAL_METRIC
    
    # No clear signal - assume qualitative
    return DataType.QUALITATIVE


def route_query(
    question: str,
    entity: Optional[str] = None,
    source_hint: Optional[str] = None,
) -> RouteResult:
    """
    Route a query to the appropriate extraction path.
    
    This is the main routing function. It analyzes the question and determines:
    1. What type of data is being requested
    2. What source should be used to get it
    3. Whether verification is needed
    
    Args:
        question: The research question
        entity: The company ticker (optional, for context)
        source_hint: Hint about the source (e.g., "10-Q", "news")
        
    Returns:
        RouteResult with routing decision
    """
    # Extract metric from question
    metric = extract_metric_from_question(question)
    
    # Detect data type
    data_type = detect_data_type(question, metric)
    
    # Handle source hints
    if source_hint:
        source_hint_lower = source_hint.lower()
        if "news" in source_hint_lower:
            return RouteResult(
                source_type=SourceType.NEWS,
                data_type=data_type,
                metric_name=metric,
                requires_verification=True,
                confidence=0.9,
            )
        if "web" in source_hint_lower or "site" in source_hint_lower:
            return RouteResult(
                source_type=SourceType.WEBSITE,
                data_type=data_type,
                metric_name=metric,
                requires_verification=True,
                confidence=0.9,
            )
    
    # Route based on data type
    if data_type == DataType.FINANCIAL_METRIC and metric in XBRL_METRICS:
        # Core financial metrics -> XBRL (high confidence, no verification)
        logger.info(f"Routing '{metric}' to XBRL path")
        return RouteResult(
            source_type=SourceType.XBRL,
            data_type=data_type,
            metric_name=metric,
            requires_verification=False,  # XBRL is authoritative
            confidence=0.95,
        )
    
    if data_type == DataType.SEGMENT_METRIC:
        # Segment metrics might be in XBRL or might need HTML
        # Try XBRL first (will fall back in extraction)
        logger.info(f"Routing '{metric}' to XBRL (segment), will fallback to HTML")
        return RouteResult(
            source_type=SourceType.XBRL,  # Try XBRL first
            data_type=data_type,
            metric_name=metric,
            requires_verification=False,  # If from XBRL, no verification
            confidence=0.7,  # Lower confidence - might need fallback
        )
    
    if data_type == DataType.QUALITATIVE:
        # Qualitative data -> SEC HTML with LLM + verification
        logger.info(f"Routing qualitative query to SEC HTML path")
        return RouteResult(
            source_type=SourceType.SEC_HTML,
            data_type=data_type,
            metric_name=metric,
            requires_verification=True,
            confidence=0.8,
        )
    
    if data_type == DataType.EVENT:
        # Events might be in SEC filings or news
        logger.info(f"Routing event query to SEC HTML path")
        return RouteResult(
            source_type=SourceType.SEC_HTML,
            data_type=data_type,
            metric_name=metric,
            requires_verification=True,
            confidence=0.7,
        )
    
    # Default: SEC HTML with verification
    return RouteResult(
        source_type=SourceType.SEC_HTML,
        data_type=data_type,
        metric_name=metric,
        requires_verification=True,
        confidence=0.5,  # Low confidence - unclear routing
    )


# =============================================================================
# Batch Routing
# =============================================================================


def route_questions(
    questions: List[str],
    entity: Optional[str] = None,
) -> List[Tuple[str, RouteResult]]:
    """
    Route a batch of questions.
    
    Args:
        questions: List of question strings
        entity: Company ticker
        
    Returns:
        List of (question, RouteResult) tuples
    """
    results = []
    for question in questions:
        result = route_query(question, entity)
        results.append((question, result))
    return results


def group_by_source(
    routed: List[Tuple[str, RouteResult]],
) -> dict[SourceType, List[Tuple[str, RouteResult]]]:
    """
    Group routed questions by source type.
    
    Useful for batching - process all XBRL questions together,
    all SEC HTML questions together, etc.
    
    Args:
        routed: List of (question, RouteResult) tuples
        
    Returns:
        Dict mapping SourceType to list of (question, RouteResult)
    """
    groups: dict[SourceType, List[Tuple[str, RouteResult]]] = {}
    
    for question, result in routed:
        if result.source_type not in groups:
            groups[result.source_type] = []
        groups[result.source_type].append((question, result))
    
    return groups

