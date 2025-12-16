"""
Query Classification - Semantic Embedding + Margin-Based Routing.

Design Principles:
1. Deterministic: same query → same result
2. Calibrated: explicit similarity, not LLM "vibes"
3. Testable: unit tests catch regressions
4. Defensible: explainable to compliance/risk committee
5. Stable: general embeddings, not financial fine-tuning

Architecture:
- Layer 1: General-purpose embeddings (bge-large-en-v1.5)
- Layer 2: Margin-based semantic routing
- Layer 3: LLM arbiter ONLY for ambiguous cases
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    FINANCIAL_LOOKUP = "financial_lookup"
    QUALITATIVE_EXTRACT = "qualitative"
    SIGNAL_DETECTION = "signal"
    VERIFICATION = "verification"
    EXPLORATION = "exploration"
    COMPARISON = "comparison"
    DISCOVERY = "discovery"
    UNKNOWN = "unknown"


class ConfidenceBand(Enum):
    """Human-readable confidence levels for auditing."""
    HIGH = "high"      # similarity > 0.75, margin > 0.12
    MEDIUM = "medium"  # similarity > 0.55, margin > 0.08
    LOW = "low"        # below thresholds
    AMBIGUOUS = "ambiguous"  # needs LLM fallback


# =============================================================================
# Canonical Examples (THE KNOWLEDGE BASE)
# Add more examples to improve classification - this is maintainable
# =============================================================================

CANONICAL_EXAMPLES: Dict[QueryType, List[str]] = {
    QueryType.FINANCIAL_LOOKUP: [
        "What was NVDA revenue in Q3 FY2025?",
        "How much did Apple earn last quarter?",
        "Show me Tesla's EPS for Q2",
        "Net income for Microsoft FY2024",
        "What's the gross margin?",
        "Q3 top line?",
        "TTM earnings?",
        "How much cash does NVDA have?",
        "Total assets for AAPL?",
        "What were operating expenses?",
        "Cost of revenue last quarter",
        "Diluted EPS for the fiscal year",
        "Free cash flow Q3",
        "R&D spending",
        "How much debt does the company have?",
        "What are the financials?",
        "Revenue breakdown",
        "Profit margins for the quarter",
    ],
    QueryType.SIGNAL_DETECTION: [
        "Any red flags in the 10-K?",
        "Did risk factors change between quarters?",
        "Show me drift between Q2 and Q3",
        "What's different in the new filing?",
        "Risk language changes?",
        "Warning signs in the latest 10-Q?",
        "Material changes in disclosures?",
        "New risks mentioned?",
        "Removed risk factors?",
        "Sentiment shift in filings?",
        "Alpha signals in the filing?",
        "Panic language detected?",
        # Intentional overlap - tests ambiguity handling:
        "What changed in the filing?",
    ],
    QueryType.VERIFICATION: [
        "Is it true that NVDA revenue was $35 billion?",
        "Verify: Apple's EPS was $1.50",
        "News says Tesla lost money, is that accurate?",
        "Fact-check this claim about Microsoft",
        "Confirm the revenue number from Reuters",
        "Is this Bloomberg headline correct?",
        "Check if the $10B figure is right",
        "Validate this earnings claim",
        "Is this rumor accurate?",
        "Correct that revenue was $50B?",
    ],
    QueryType.COMPARISON: [
        "Compare NVDA vs AMD revenue",
        "Which has higher margins, Apple or Microsoft?",
        "NVDA versus Intel EPS comparison",
        "Difference between Tesla and Ford profitability",
        "How does Google compare to Meta on R&D?",
        "Revenue comparison: AAPL vs MSFT vs GOOGL",
        "Who has more cash, Apple or Google?",
        "Which company is more profitable?",
        "Compare earnings across these tickers",
    ],
    QueryType.DISCOVERY: [
        "What's the buzz around NVDA?",
        "Find recent news about Apple",
        "What are people saying about Tesla?",
        "Latest developments for Microsoft?",
        "Search for NVDA news",
        "Market sentiment on AAPL?",
        "Any recent articles about the company?",
        "What's happening with the stock?",
        "Find news about earnings announcement",
        "What's going on with this company?",
    ],
    QueryType.QUALITATIVE_EXTRACT: [
        "What are the main risks?",
        "What did the CEO say about guidance?",
        "Management's outlook for next year?",
        "Strategic initiatives mentioned?",
        "Why did revenue decline?",
        "What caused the margin compression?",
        "CFO comments on cash position?",
        "Forward-looking statements?",
        "What challenges did they mention?",
        "What did management say about competition?",
        # Intentional overlap - tests ambiguity handling:
        "What changed in the filing?",
    ],
    QueryType.EXPLORATION: [
        "Research NVDA",
        "Tell me about Apple's business",
        "Give me an overview of Tesla",
        "Analyze Microsoft's position",
        "Deep dive into AMD",
        "Summarize the company",
        "Explain Google's revenue streams",
        "Investigate this company",
        "What should I know about NVDA?",
    ],
}


# =============================================================================
# Thresholds (TUNE THESE AFTER COLLECTING PRODUCTION DATA)
# =============================================================================

# Minimum similarity to consider a match
MIN_SIMILARITY = 0.45

# Minimum margin between top-1 and top-2 to be confident
MIN_MARGIN = 0.08

# Below this similarity, always use LLM fallback
LLM_FALLBACK_THRESHOLD = 0.55

# Thresholds for confidence bands
HIGH_SIMILARITY = 0.75
HIGH_MARGIN = 0.12
MEDIUM_SIMILARITY = 0.55
MEDIUM_MARGIN = 0.08

# Secondary candidates within this margin are flagged
MULTI_INTENT_MARGIN = 0.05


# =============================================================================
# Embedding Layer (Lazy Loading)
# =============================================================================

_EMBEDDER = None
_EXAMPLE_EMBEDDINGS: Dict[QueryType, np.ndarray] = {}


def _get_embedder():
    """
    Lazy load embedding model with timeout and offline guard.
    
    P0 Fix:
    - Respects EMBEDDINGS_DISABLED env var for offline mode
    - Adds timeout to prevent hangs during model download
    - Gracefully fails if model unavailable
    """
    global _EMBEDDER
    
    if _EMBEDDER is not None:
        return _EMBEDDER
    
    # Check for explicit disable via environment variable
    import os
    if os.environ.get("EMBEDDINGS_DISABLED", "").lower() in ("1", "true", "yes"):
        logger.warning("Embeddings disabled via EMBEDDINGS_DISABLED env var")
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
        import signal
        import sys
        
        # Only use signal-based timeout on Unix systems
        if sys.platform != "win32":
            def timeout_handler(signum, frame):
                raise TimeoutError("Embedding model load timed out after 30 seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
        
        try:
            _EMBEDDER = SentenceTransformer("BAAI/bge-large-en-v1.5")
            logger.info("Loaded embedding model: bge-large-en-v1.5")
        finally:
            if sys.platform != "win32":
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)
                
    except ImportError:
        logger.warning("sentence-transformers not installed, using regex fallback")
        _EMBEDDER = None
    except TimeoutError as e:
        logger.error(f"Embedding model load timed out: {e}")
        logger.warning("Running in degraded mode - classification will use regex fallback")
        _EMBEDDER = None
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        logger.warning("Running in degraded mode - classification will use regex fallback")
        _EMBEDDER = None
    
    return _EMBEDDER


def _get_example_embeddings() -> Dict[QueryType, np.ndarray]:
    """Get or compute cached example embeddings."""
    global _EXAMPLE_EMBEDDINGS
    if not _EXAMPLE_EMBEDDINGS:
        embedder = _get_embedder()
        if embedder is None:
            return {}
        
        for qtype, examples in CANONICAL_EXAMPLES.items():
            # bge instruction prefix for better separation
            prefixed = [f"Classify the intent of this query: {ex}" for ex in examples]
            embeddings = embedder.encode(prefixed, normalize_embeddings=True)
            _EXAMPLE_EMBEDDINGS[qtype] = embeddings
        logger.info(f"Computed embeddings for {len(_EXAMPLE_EMBEDDINGS)} query types")
    
    return _EXAMPLE_EMBEDDINGS


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between normalized vectors."""
    return float(np.dot(a, b))


# =============================================================================
# Classification Result
# =============================================================================

@dataclass
class ClassificationResult:
    """
    Structured classification result for auditability.
    
    Note: 'similarity' is the raw cosine score, NOT calibrated confidence.
    Use 'confidence_band' for human-readable certainty levels.
    """
    query_type: QueryType
    similarity: float  # Raw cosine score (NOT calibrated confidence)
    margin: float  # Gap to second-best (higher = more certain)
    confidence_band: ConfidenceBand  # Human-readable: HIGH/MEDIUM/LOW/AMBIGUOUS
    method: str  # "embedding", "llm_fallback", or "regex_fallback"
    ambiguous: bool  # True if margin < threshold
    
    # For debugging and audit
    scores: Dict[str, float] = field(default_factory=dict)
    secondary_candidates: List[str] = field(default_factory=list)
    
    # For audit trail (when LLM fallback used)
    embedding_result: Optional[str] = None
    llm_reasoning: Optional[str] = None


def _compute_confidence_band(similarity: float, margin: float, ambiguous: bool) -> ConfidenceBand:
    """Derive human-readable confidence band from metrics."""
    if ambiguous:
        return ConfidenceBand.AMBIGUOUS
    if similarity >= HIGH_SIMILARITY and margin >= HIGH_MARGIN:
        return ConfidenceBand.HIGH
    if similarity >= MEDIUM_SIMILARITY and margin >= MEDIUM_MARGIN:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


# =============================================================================
# Main Classification Function
# =============================================================================

def classify_query_semantic(query: str) -> ClassificationResult:
    """
    Classify query using semantic embeddings with margin-based confidence.
    
    Returns structured result with:
    - query_type: The classified type
    - similarity: Max cosine similarity score (NOT calibrated confidence)
    - margin: Gap to second-best type
    - confidence_band: HIGH / MEDIUM / LOW / AMBIGUOUS
    - secondary_candidates: Other types within MULTI_INTENT_MARGIN
    - ambiguous: True if should use LLM fallback
    """
    embedder = _get_embedder()
    example_embeddings = _get_example_embeddings()
    
    if embedder is None or not example_embeddings:
        return _regex_fallback(query)
    
    # Embed with instruction prefix (matches example encoding)
    query_embedding = embedder.encode(
        f"Classify the intent of this query: {query}",
        normalize_embeddings=True
    )
    
    # Score against each query type
    scores: Dict[QueryType, float] = {}
    for qtype, examples_emb in example_embeddings.items():
        similarities = [_cosine_similarity(query_embedding, ex) for ex in examples_emb]
        scores[qtype] = max(similarities)
    
    # Sort by score descending
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0
    margin = best_score - second_score
    
    # Find secondary candidates (potential multi-intent)
    secondary = [
        qt.value for qt, s in scores.items()
        if qt != best_type and best_score - s < MULTI_INTENT_MARGIN
    ]
    
    # Determine if ambiguous
    ambiguous = (
        best_score < LLM_FALLBACK_THRESHOLD or
        margin < MIN_MARGIN
    )
    
    confidence_band = _compute_confidence_band(best_score, margin, ambiguous)
    
    result = ClassificationResult(
        query_type=best_type if best_score >= MIN_SIMILARITY else QueryType.UNKNOWN,
        similarity=best_score,
        margin=margin,
        confidence_band=confidence_band,
        method="embedding",
        ambiguous=ambiguous,
        scores={qt.value: round(s, 4) for qt, s in scores.items()},
        secondary_candidates=secondary,
    )
    
    # Log for threshold tuning
    logger.info(
        f"[CLASSIFY] query='{query[:60]}' "
        f"type={result.query_type.value} "
        f"sim={result.similarity:.3f} "
        f"margin={result.margin:.3f} "
        f"band={result.confidence_band.value} "
        f"secondary={result.secondary_candidates}"
    )
    
    return result


# =============================================================================
# LLM Fallback (Only for Ambiguous Cases)
# =============================================================================

LLM_CLASSIFIER_PROMPT = """Classify this research query into exactly ONE category.

Categories:
- FINANCIAL_LOOKUP: Specific financial metrics (revenue, EPS, margins, cash, debt)
- SIGNAL_DETECTION: Risk changes, red flags, drift analysis between periods
- VERIFICATION: Fact-checking a specific claim against SEC data
- COMPARISON: Comparing metrics between multiple companies
- DISCOVERY: News search, market buzz, recent developments
- QUALITATIVE: Management commentary, guidance, strategic discussion
- EXPLORATION: General research request, company overview
- UNKNOWN: Cannot determine intent

Query: {query}

Respond with ONLY valid JSON:
{{"type": "FINANCIAL_LOOKUP", "reasoning": "Query asks for specific revenue metric"}}
"""


async def classify_query_llm(query: str) -> Tuple[QueryType, str]:
    """LLM fallback for ambiguous queries. Returns (type, reasoning)."""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast, cheap, good at classification
            max_tokens=150,
            temperature=0,  # Deterministic
            messages=[{"role": "user", "content": LLM_CLASSIFIER_PROMPT.format(query=query)}]
        )
        
        result = json.loads(response.content[0].text)
        type_str = result["type"].lower()
        
        # Map QUALITATIVE back to enum value
        if type_str == "qualitative":
            type_str = "qualitative"
        
        query_type = QueryType(type_str)
        reasoning = result.get("reasoning", "")
        
        return query_type, reasoning
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
        return QueryType.UNKNOWN, f"LLM error: {e}"


def _run_llm_fallback(query: str) -> Tuple[QueryType, str]:
    """Sync wrapper for LLM fallback."""
    import asyncio
    try:
        # Try to get running loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, classify_query_llm(query))
                return future.result(timeout=10.0)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(classify_query_llm(query))
    except Exception as e:
        logger.warning(f"LLM fallback execution failed: {e}")
        return QueryType.UNKNOWN, str(e)


# =============================================================================
# Main Entry Point
# =============================================================================

def classify_query(query: str, use_llm_fallback: bool = True) -> Tuple[QueryType, float]:
    """
    Main classification function. Returns (QueryType, similarity).
    
    Backwards-compatible with existing code.
    """
    result = classify_query_semantic(query)
    
    if result.ambiguous and use_llm_fallback:
        llm_type, reasoning = _run_llm_fallback(query)
        
        if llm_type != QueryType.UNKNOWN:
            # Log the override for audit
            logger.info(
                f"[CLASSIFY] LLM override: {result.query_type.value} → {llm_type.value} "
                f"reason='{reasoning}'"
            )
            return llm_type, 0.8  # LLM doesn't give calibrated score
    
    return result.query_type, result.similarity


def classify_query_detailed(query: str) -> ClassificationResult:
    """Full classification result for debugging/auditing."""
    return classify_query_semantic(query)


# =============================================================================
# Regex Fallback (Absolute Last Resort)
# =============================================================================

# Keep regex patterns for when embeddings unavailable
import re

# Financial metric keywords for regex fallback
FINANCIAL_KEYWORDS = {
    "revenue", "revenues", "sales", "income", "profit", "earnings",
    "eps", "margin", "margins", "cost", "costs", "expense", "expenses",
    "assets", "liabilities", "equity", "cash", "debt", "capex",
    "ebitda", "top line", "bottom line", "ttm",
}

FINANCIAL_PATTERNS = [
    r"what (was|were|is|are) .*(revenue|income|profit|eps|margin|sales|cash|assets|debt|expense)",
    r"how much .*(revenue|income|cost|profit|cash|expense|earn|debt)",
    r"(revenue|income|profit|eps|margin|sales|earnings) .*(in|for|during|\?)",
    r"(total|net|gross|operating)\s+(revenue|income|profit|margin|expense)",
    r"\b(ttm|trailing)\b.*\b(earnings|revenue|income)",
    r"top line|bottom line",
    r"cost of (revenue|goods|sales)",
    r"(diluted|basic)\s+eps",
    r"free cash flow",
    r"r&d\s+(spend|expense)",
]

SIGNAL_PATTERNS = [
    r"red flag", r"drift", r"risk.*(change|factor|differ|mention|new|add|remove)",
    r"(significant|major|material).*(change|shift)", r"warning sign",
    r"new risks", r"removed risk",
]

VERIFICATION_PATTERNS = [
    r"is (it|this|that) true", r"verify", r"confirm",
    r"news says", r"fact.?check", r"accurate", r"correct that",
    r"headline.*(correct|true|accurate)",
]

COMPARISON_PATTERNS = [
    r"compare", r"\bversus\b|\bvs\.?\b", r"(higher|lower|more|less) than",
    r"difference between", r"which.*(higher|lower|better|worse)",
]

DISCOVERY_PATTERNS = [
    r"what('s| is) (the )?(buzz|news|happening|going on)",
    r"find.*(news|articles)", r"(latest|recent).*(news|developments)",
    r"what are people saying", r"search.*(news|web)",
]

EXPLORATION_PATTERNS = [
    r"^research\s", r"^tell me about", r"overview",
    r"^analyze\s", r"^deep dive", r"^investigate",
]

QUALITATIVE_PATTERNS = [
    r"what (are|were) the .*(risk|challenge|opportunity)",
    r"(guidance|outlook|forecast)",
    r"(ceo|cfo|management).*(said|say|mentioned|comment)",
    r"strategic.*(initiative|plan)",
]


def _regex_fallback(query: str) -> ClassificationResult:
    """
    Degraded mode fallback when embeddings unavailable.
    
    INVARIANTS (production requirements):
    1. NEVER claims semantic understanding - returns UNKNOWN
    2. NEVER routes to XBRL (Tier 1) - downstream must use safe path
    3. ALWAYS requires downstream verification
    4. ALWAYS returns UNKNOWN with regex hints (not classifications)
    
    The regex hints are for LOGGING and ROUTING TIER SELECTION only.
    They are NOT intent classifications.
    
    Downstream behavior when method="regex_degraded":
    - Route to SEC_HTML (Tier 2), never XBRL
    - Always require verification gate
    - Mark output as degraded mode
    """
    logger.warning(
        "[DEGRADED MODE] Embeddings unavailable - system operating in degraded mode. "
        "Returning UNKNOWN with regex hints. Downstream must route to safe path."
    )
    query_lower = query.lower()
    
    # Collect regex hints (NOT classifications!)
    # These hints help downstream decide which SAFE path to take
    hints: List[str] = []
    
    pattern_checks = [
        (VERIFICATION_PATTERNS, "verification_keywords"),
        (SIGNAL_PATTERNS, "signal_keywords"),
        (COMPARISON_PATTERNS, "comparison_keywords"),
        (DISCOVERY_PATTERNS, "discovery_keywords"),
        (EXPLORATION_PATTERNS, "exploration_keywords"),
        (QUALITATIVE_PATTERNS, "qualitative_keywords"),
        (FINANCIAL_PATTERNS, "financial_keywords"),
    ]
    
    for patterns, hint_name in pattern_checks:
        for pattern in patterns:
            if re.search(pattern, query_lower):
                hints.append(hint_name)
                break  # Only one hint per category
    
    # Check for financial keywords
    words = set(query_lower.split())
    if words & FINANCIAL_KEYWORDS:
        if "financial_keywords" not in hints:
            hints.append("financial_keywords")
    
    # Log the hints for audit
    logger.info(f"[DEGRADED] Query: '{query[:60]}' | Hints: {hints}")
    
    return ClassificationResult(
        query_type=QueryType.UNKNOWN,  # NEVER claim to understand
        similarity=0.0,                 # No semantic similarity available
        margin=0.0,                     # No margin available
        confidence_band=ConfidenceBand.AMBIGUOUS,
        method="regex_degraded",        # Explicit degraded mode marker
        ambiguous=True,
        scores={"regex_hints": hints},  # Hints for routing, not classifications
        secondary_candidates=[],
    )


# =============================================================================
# Helper Functions
# =============================================================================

def get_route_description(query_type: QueryType) -> str:
    """Human-readable description for each route."""
    descriptions = {
        QueryType.FINANCIAL_LOOKUP: "📊 Financial Lookup (XBRL extraction)",
        QueryType.SIGNAL_DETECTION: "🚨 Signal Detection (drift analysis)",
        QueryType.VERIFICATION: "✅ Verification (cross-check claims)",
        QueryType.COMPARISON: "⚖️ Comparison (multi-entity lookup)",
        QueryType.DISCOVERY: "🔍 Discovery (news/web search)",
        QueryType.QUALITATIVE_EXTRACT: "💬 Qualitative (text extraction)",
        QueryType.EXPLORATION: "🔬 Exploration (deep research)",
        QueryType.UNKNOWN: "❓ Unknown (needs clarification)",
    }
    return descriptions.get(query_type, "Unknown")


def extract_comparison_entities(query: str) -> List[str]:
    """Extract entity tickers from comparison queries."""
    # Common patterns: "NVDA vs AMD", "compare Apple and Microsoft"
    query_upper = query.upper()
    
    # Find potential tickers (1-5 uppercase letters)
    import re
    potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query_upper)
    
    # Filter out common words
    stop_words = {
        'THE', 'AND', 'FOR', 'WITH', 'WHAT', 'HOW', 'WHICH', 'THAT', 'THIS',
        'ARE', 'WAS', 'WERE', 'HAS', 'HAVE', 'HAD', 'BEEN', 'WILL', 'WOULD',
        'VS', 'OR', 'NOT', 'BUT', 'FROM', 'THAN', 'MORE', 'LESS', 'EPS',
        'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'TTM', 'YOY', 'QOQ',
    }
    
    tickers = [t for t in potential_tickers if t not in stop_words]
    return tickers


def is_multi_entity_query(query: str) -> bool:
    """Check if query involves multiple entities."""
    entities = extract_comparison_entities(query)
    return len(entities) >= 2
