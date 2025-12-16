"""
Semantic Drift Engine for SEC Risk Factors.

Detects language changes in SEC Risk Factors (Item 1A) across quarters.
This creates a trading signal from unstructured text - the hypothesis is
that if management substantially rewrites the Risk section, they may be
pricing in material changes not yet reflected in structured financial data.

Key Features:
- Uses difflib.SequenceMatcher for sentence-level diff (shows WHAT changed)
- Professional NLP libraries with graceful fallbacks
- Returns added/removed sentences, not just similarity scores
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from open_deep_research.models import DriftResult, SignalAlert


logger = logging.getLogger(__name__)


# =============================================================================
# Signal Modes (P3 - Dual Filing Logic)
# =============================================================================


class SignalMode(Enum):
    """
    Signal detection mode determining which filings to compare.
    
    REGIME: 10-K → 10-K comparison (annual baseline)
        - Detects substantive risk regime changes
        - Where new risk categories are introduced
        - Best for quarterly investment review
    
    EVENT: 10-Q → 10-K comparison (fast overlay)
        - Detects novel risk language since last 10-K
        - Suppresses boilerplate ("no material changes")
        - Best for between-review monitoring
    
    QUARTERLY: 10-Q → 10-Q comparison (legacy)
        - High noise, low signal
        - Mostly boilerplate detection
        - Use for debugging/comparison only
    """
    
    REGIME = "regime"       # 10-K → 10-K (annual baseline)
    EVENT = "event"         # 10-Q → 10-K (fast overlay)
    QUARTERLY = "quarterly" # 10-Q → 10-Q (legacy behavior)


# =============================================================================
# NLP Library Imports (with graceful fallbacks)
# =============================================================================

# Try nltk for tokenization and stopwords - LAZY INITIALIZATION
# P0 Fix: No downloads at import time - call setup_nlp() explicitly
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    NLTK_AVAILABLE = True
    STOPWORDS: Set[str] = set()  # Populated by setup_nlp()
    _nltk_initialized = False
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = set()  # Empty - skip stopword removal if nltk unavailable
    _nltk_initialized = True  # No setup needed
    logger.warning("nltk not available - using basic tokenization")


def setup_nlp() -> bool:
    """
    Initialize NLP resources (NLTK data downloads).
    
    Call this ONCE at application startup, NOT at import time.
    This prevents hanging/crashing during import if network is unavailable.
    
    Returns:
        True if setup succeeded, False otherwise
    """
    global _nltk_initialized, STOPWORDS
    
    if _nltk_initialized:
        return NLTK_AVAILABLE
    
    if not NLTK_AVAILABLE:
        _nltk_initialized = True
        return False
    
    try:
        # Check and download punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        # Check and download punkt_tab
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt_tab...")
            nltk.download('punkt_tab', quiet=True)
        
        # Check and download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        # Now load stopwords
        STOPWORDS.update(stopwords.words('english'))
        logger.info(f"NLTK initialized: {len(STOPWORDS)} stopwords loaded")
        
    except Exception as e:
        logger.warning(f"NLTK setup failed: {e}. Using fallback tokenization.")
    
    _nltk_initialized = True
    return True


# Fallback stopwords if NLTK is not available (minimal common words)
FALLBACK_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "it", "its", "we", "our", "their",
    "they", "them", "you", "your", "he", "she", "him", "her", "his",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "here", "there", "about", "above", "after",
    "before", "below", "between", "into", "through", "during", "under",
    "again", "further", "then", "once", "any", "if", "because", "until",
}

# Try spacy for sentence segmentation (best quality) - LAZY INITIALIZATION
# P0 Fix: No model loading at import time
try:
    import spacy
    _SPACY_INSTALLED = True
except ImportError:
    _SPACY_INSTALLED = False
    logger.warning("spacy not installed")

_nlp = None
_nlp_initialized = False


def get_nlp():
    """
    Lazy-load spaCy model with graceful fallback.
    
    Call this instead of accessing _nlp directly. The model is loaded
    on first use, not at import time, to prevent import hangs.
    
    Returns:
        spaCy Language object, or None if unavailable
    """
    global _nlp, _nlp_initialized
    
    if _nlp_initialized:
        return _nlp
    
    if not _SPACY_INSTALLED:
        _nlp_initialized = True
        return None
    
    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )
        _nlp = None
    except Exception as e:
        logger.warning(f"spaCy load failed: {e}")
        _nlp = None
    
    _nlp_initialized = True
    return _nlp


def is_spacy_available() -> bool:
    """Check if spaCy is available (triggers lazy load if needed)."""
    return get_nlp() is not None

# Try sentence-transformers for semantic similarity (optional enhancement)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    _embedding_model: Optional[SentenceTransformer] = None
    
    def _get_embedding_model() -> SentenceTransformer:
        """Lazy-load the embedding model."""
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _embedding_model
    
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, skipping semantic similarity")


# =============================================================================
# Risk Keywords (Domain-Specific)
# =============================================================================

RISK_KEYWORDS: Set[str] = {
    # Legal
    "litigation", "lawsuit", "subpoena", "investigation", "settlement",
    "regulatory", "enforcement", "violation", "penalty", "fine",
    "indictment", "sec", "doj", "ftc",
    
    # Operational
    "shortage", "delay", "disruption", "constraint", "bottleneck",
    "supply chain", "manufacturing", "capacity", "backlog",
    
    # Financial
    "impairment", "restatement", "default", "covenant", "liquidity",
    "write-down", "goodwill", "restructuring", "bankruptcy",
    
    # Geopolitical
    "china", "russia", "tariff", "export", "sanction", "restriction",
    "geopolitical", "government", "regulation", "ban",
    
    # Competitive
    "competition", "market share", "pricing pressure", "commoditization",
    "obsolete", "disruption",
}


# =============================================================================
# Boilerplate Detection (P2 - Noise Suppression)
# =============================================================================

# Compiled patterns for boilerplate SEC language
# These are high-precision patterns that indicate non-substantive text
BOILERPLATE_PATTERNS: List[re.Pattern] = [
    # === Explicit "no changes" language ===
    re.compile(r"no material change", re.IGNORECASE),
    re.compile(r"have not changed materially", re.IGNORECASE),
    re.compile(r"has not changed materially", re.IGNORECASE),
    re.compile(r"there have been no material changes", re.IGNORECASE),
    re.compile(r"there has been no material change", re.IGNORECASE),
    re.compile(r"there were no changes", re.IGNORECASE),
    re.compile(r"no changes (have been|were) made", re.IGNORECASE),
    re.compile(r"no significant change", re.IGNORECASE),
    re.compile(r"remain(s)? unchanged", re.IGNORECASE),
    re.compile(r"unchanged from our annual report", re.IGNORECASE),
    
    # === Reference to annual report ===
    re.compile(r"except as (previously )?disclosed", re.IGNORECASE),
    re.compile(r"as set forth in our (annual|10-k)", re.IGNORECASE),
    re.compile(r"as described in (our )?annual report", re.IGNORECASE),
    re.compile(r"as discussed in (our )?annual report", re.IGNORECASE),
    re.compile(r"as included in our form 10-k", re.IGNORECASE),
    re.compile(r"as previously reported", re.IGNORECASE),
    re.compile(r"as previously filed", re.IGNORECASE),
    re.compile(r"refer(ence)? to (our )?(annual|10-k)", re.IGNORECASE),
    re.compile(r"incorporated (herein )?by reference", re.IGNORECASE),
    re.compile(r"set forth under the heading", re.IGNORECASE),
    
    # === Boilerplate qualifiers ===
    re.compile(r"does not differ materially", re.IGNORECASE),
    re.compile(r"do not differ materially", re.IGNORECASE),
    re.compile(r"updated (only )?where (necessary|appropriate)", re.IGNORECASE),
    re.compile(r"remain(s)? substantially (the same|unchanged)", re.IGNORECASE),
    
    # === Sentence-local patterns (fixed: no greedy .*) ===
    re.compile(r"(the )?risk factors[^.]{0,200}have not (materially )?changed", re.IGNORECASE),
    re.compile(r"(the )?discussion[^.]{0,200}should be read (together|in conjunction) with", re.IGNORECASE),
]


@dataclass
class BoilerplateResult:
    """
    Result of boilerplate detection.
    
    IMPORTANT: This is a FLAG and RATIO, not a binary decision.
    Use boilerplate_ratio as a penalty/dampener, not a kill-switch.
    A filing can contain BOTH boilerplate AND real changes.
    """
    
    is_boilerplate_heavy: bool      # True if ratio > 0.6
    boilerplate_ratio: float        # 0.0 - 1.0
    boilerplate_sentence_count: int
    total_sentence_count: int
    matched_patterns: List[str]     # For debugging/audit


def detect_boilerplate(text: str) -> BoilerplateResult:
    """
    Detect boilerplate in Risk Factors text.
    
    This is a first-order noise suppressor for 10-Q filings where
    ~60-70% of Item 1A sections are boilerplate-dominant.
    
    IMPORTANT: Returns a flag and ratio, not a binary decision.
    Use boilerplate_ratio as a penalty/dampener in ranking.
    
    Args:
        text: Risk Factors text to analyze
        
    Returns:
        BoilerplateResult with detection metrics
    """
    # Normalize whitespace
    text_normalized = " ".join(text.split())
    
    sentences = extract_sentences(text_normalized)
    if not sentences:
        return BoilerplateResult(
            is_boilerplate_heavy=False,
            boilerplate_ratio=0.0,
            boilerplate_sentence_count=0,
            total_sentence_count=0,
            matched_patterns=[],
        )
    
    # Track which patterns matched (for audit)
    matched_patterns: Set[str] = set()
    
    # Count boilerplate sentences
    boilerplate_sentences = 0
    for sentence in sentences:
        for pattern in BOILERPLATE_PATTERNS:
            if pattern.search(sentence):
                boilerplate_sentences += 1
                matched_patterns.add(pattern.pattern)
                break  # One match per sentence is enough
    
    ratio = boilerplate_sentences / len(sentences)
    
    return BoilerplateResult(
        is_boilerplate_heavy=(ratio > 0.6),
        boilerplate_ratio=round(ratio, 3),
        boilerplate_sentence_count=boilerplate_sentences,
        total_sentence_count=len(sentences),
        matched_patterns=sorted(matched_patterns),
    )


# =============================================================================
# Text Normalization (Phase 4 - Avoid Boilerplate Shuffle)
# =============================================================================


def normalize_text_for_diff(
    text: str,
    remove_stopwords: bool = True,
    min_sentence_tokens: int = 6,
) -> str:
    """
    Normalize text for drift scoring to reduce false positives.
    
    This function reduces noise from:
    - Pronoun swaps (we/the company, our/their)
    - Formatting changes (whitespace, punctuation)
    - Sentence reordering (by sorting normalized sentences)
    - Short boilerplate sentences
    
    IMPORTANT: Use normalized text for SCORING only.
    Keep original sentences for the DISPLAY diff (users need to see actual text).
    
    Args:
        text: Raw text to normalize
        remove_stopwords: Whether to remove stopwords (default True)
        min_sentence_tokens: Minimum tokens for a sentence to be kept (default 6)
        
    Returns:
        Normalized text string suitable for similarity scoring
    """
    if not text:
        return ""
    
    # Step 1: Lowercase
    text_lower = text.lower()
    
    # Step 2: Remove punctuation (except periods for sentence boundaries)
    # Keep periods temporarily for sentence splitting
    text_no_punct = re.sub(r"[^\w\s.]", "", text_lower)
    
    # Step 3: Collapse whitespace
    text_collapsed = " ".join(text_no_punct.split())
    
    # Step 4: Split into sentences
    sentences = extract_sentences(text_collapsed)
    
    # Step 5: Normalize each sentence
    normalized_sentences: List[str] = []
    # Use NLTK stopwords if loaded, otherwise use fallback set
    stopwords_to_use = STOPWORDS if len(STOPWORDS) > 0 else FALLBACK_STOPWORDS
    
    for sentence in sentences:
        # Tokenize
        if NLTK_AVAILABLE:
            tokens = word_tokenize(sentence)
        else:
            tokens = re.findall(r'\b[a-z]+\b', sentence)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stopwords_to_use]
        
        # Skip short sentences (likely boilerplate or headers)
        if len(tokens) < min_sentence_tokens:
            continue
        
        # Rejoin tokens
        normalized_sentences.append(" ".join(tokens))
    
    # Step 6: Sort sentences (order-independent comparison)
    # This reduces drift from paragraph reordering
    normalized_sentences.sort()
    
    return " ".join(normalized_sentences)


def get_normalized_tokens(text: str) -> Set[str]:
    """
    Get normalized tokens from text for similarity scoring.
    
    Combines normalize_text_for_diff with tokenization.
    Used for Jaccard calculation with reduced noise.
    
    Args:
        text: Raw text
        
    Returns:
        Set of normalized tokens
    """
    normalized = normalize_text_for_diff(text)
    return tokenize(normalized)


# =============================================================================
# Tokenization & Sentence Splitting
# =============================================================================


def tokenize(text: str) -> Set[str]:
    """Clean and tokenize text for comparison.
    
    Uses nltk.word_tokenize if available, falls back to regex.
    Removes stopwords and short words.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Set of cleaned, lowercase tokens
    """
    text_lower = text.lower()
    
    if NLTK_AVAILABLE:
        words = word_tokenize(text_lower)
    else:
        # Fallback: simple regex tokenization
        words = re.findall(r'\b[a-z]+\b', text_lower)
    
    # Use NLTK stopwords if loaded, otherwise use fallback set
    stopwords_to_use = STOPWORDS if len(STOPWORDS) > 0 else FALLBACK_STOPWORDS
    
    # Remove stopwords and short words (keep only meaningful content)
    return {w for w in words if w not in stopwords_to_use and len(w) > 2 and w.isalpha()}


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Uses spacy if available (best for handling abbreviations like "Dr." or "Inc."),
    nltk.sent_tokenize if available (good), falls back to regex (basic).
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentence strings, filtered for minimum length
    """
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    elif NLTK_AVAILABLE:
        sentences = sent_tokenize(text)
    else:
        # Fallback: basic regex split (less accurate for abbreviations)
        sentences = re.split(r'[.!?]+\s*', text)
    
    # Filter: must be real sentences (>20 chars, not just noise)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]


# =============================================================================
# Similarity Calculations
# =============================================================================


def calculate_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets.
    
    Jaccard = |intersection| / |union|
    Returns 1.0 for identical sets, 0.0 for completely different.
    
    Args:
        set1: First set of tokens
        set2: Second set of tokens
        
    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
    """
    if not set1 and not set2:
        return 1.0  # Both empty = identical
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_semantic_similarity(text1: str, text2: str) -> Optional[float]:
    """Calculate semantic similarity using sentence embeddings.
    
    Uses sentence-transformers if available.
    Returns cosine similarity (0-1), or None if embeddings not available.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Cosine similarity (0.0 to 1.0), or None if not available
    """
    if not EMBEDDINGS_AVAILABLE:
        return None
    
    try:
        model = _get_embedding_model()
        
        # Limit text length to avoid memory issues
        emb1 = model.encode(text1[:5000])
        emb2 = model.encode(text2[:5000])
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")
        return None


# =============================================================================
# Sentence-Level Diff (THE KEY FEATURE)
# =============================================================================


def get_sentence_diff(text_old: str, text_new: str) -> Tuple[List[str], List[str]]:
    """Get added and removed sentences using difflib.
    
    CRITICAL: This is what makes the drift analysis useful.
    A score alone is meaningless. Users need to SEE what changed.
    
    Uses difflib.SequenceMatcher to find:
    - Sentences that were added (new text)
    - Sentences that were removed (old text)
    
    Args:
        text_old: Original text
        text_new: New text
        
    Returns:
        Tuple of (added_sentences, removed_sentences)
    """
    sentences_old = extract_sentences(text_old)
    sentences_new = extract_sentences(text_new)
    
    matcher = difflib.SequenceMatcher(None, sentences_old, sentences_new)
    
    added: List[str] = []
    removed: List[str] = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Old sentences replaced with new ones
            removed.extend(sentences_old[i1:i2])
            added.extend(sentences_new[j1:j2])
        elif tag == 'delete':
            # Sentences removed
            removed.extend(sentences_old[i1:i2])
        elif tag == 'insert':
            # Sentences added
            added.extend(sentences_new[j1:j2])
        # 'equal' tag means no change - we don't track those
    
    return added, removed


# =============================================================================
# Risk Keyword Detection
# =============================================================================


def find_risk_keywords(text: str) -> Set[str]:
    """Find risk-related keywords in text.
    
    Searches for domain-specific risk terms that may indicate
    legal, operational, financial, or geopolitical concerns.
    
    Args:
        text: Text to search
        
    Returns:
        Set of found risk keywords
    """
    text_lower = text.lower()
    found: Set[str] = set()
    
    for keyword in RISK_KEYWORDS:
        if keyword in text_lower:
            found.add(keyword)
    
    return found


# =============================================================================
# Main Drift Calculation
# =============================================================================


def calculate_drift(
    text_from: str,
    text_to: str,
    period_from: str = "",
    period_to: str = "",
    use_normalization: bool = True,
) -> DriftResult:
    """Calculate semantic drift between two text blocks.
    
    Uses multiple similarity measures:
    - Jaccard similarity (word overlap)
    - Semantic similarity (embedding cosine, if available)
    
    Phase 4 Enhancement: Uses normalized text for SCORING to reduce false positives
    from pronoun swaps, formatting changes, and sentence reordering. BUT keeps
    original sentences for the DISPLAY diff (users need to see actual text).
    
    The drift_score is calculated as (1 - similarity) * 100, where
    similarity combines Jaccard and semantic (if available).
    
    Args:
        text_from: Original text (e.g., Q1 Risk Factors)
        text_to: New text (e.g., Q2 Risk Factors)
        period_from: Label for source period (e.g., "Q1 FY2025")
        period_to: Label for target period (e.g., "Q2 FY2025")
        use_normalization: Whether to normalize text for scoring (default True)
        
    Returns:
        DriftResult with scores, severity, and sentence-level changes
    """
    # Phase 4: Use normalized text for SCORING to reduce noise
    # But keep original sentences for DISPLAY diff
    if use_normalization:
        # Normalized tokens for Jaccard (reduces pronoun/formatting noise)
        words_from = get_normalized_tokens(text_from)
        words_to = get_normalized_tokens(text_to)
        
        # Normalized text for semantic similarity
        normalized_from = normalize_text_for_diff(text_from)
        normalized_to = normalize_text_for_diff(text_to)
        
        # FALLBACK: If normalization filters out all text (very short sentences),
        # fall back to raw tokenization to avoid false "identical" results
        if not words_from and not words_to:
            raw_from = tokenize(text_from)
            raw_to = tokenize(text_to)
            # Only fall back if raw tokens exist (texts weren't actually empty)
            if raw_from or raw_to:
                logger.debug(
                    "Normalization produced empty tokens, falling back to raw tokenization"
                )
                words_from = raw_from
                words_to = raw_to
                normalized_from = text_from
                normalized_to = text_to
    else:
        # Legacy behavior: use raw text
        words_from = tokenize(text_from)
        words_to = tokenize(text_to)
        normalized_from = text_from
        normalized_to = text_to
    
    # Calculate Jaccard similarity (on normalized tokens)
    jaccard_sim = calculate_jaccard(words_from, words_to)
    
    # Calculate semantic similarity (on normalized text if available)
    semantic_sim = calculate_semantic_similarity(normalized_from, normalized_to)
    
    # Combine similarities: weight semantic 70%, Jaccard 30% if available
    if semantic_sim is not None:
        combined_sim = (semantic_sim * 0.7) + (jaccard_sim * 0.3)
    else:
        combined_sim = jaccard_sim
    
    # Drift score: 0-100 (higher = more change)
    drift_score = (1 - combined_sim) * 100
    
    # Get sentence-level diff using difflib on ORIGINAL text (THE IMPORTANT PART)
    # Users need to see actual sentences, not normalized versions
    added_sentences, removed_sentences = get_sentence_diff(text_from, text_to)
    
    # Find risk keyword changes
    keywords_from = find_risk_keywords(text_from)
    keywords_to = find_risk_keywords(text_to)
    new_keywords = sorted(keywords_to - keywords_from)
    removed_keywords = sorted(keywords_from - keywords_to)
    
    # Determine severity based on drift score and new risk keywords
    if drift_score > 30 or len(new_keywords) >= 3:
        severity = "critical"
    elif drift_score > 12 or len(new_keywords) >= 1:
        severity = "moderate"
    else:
        severity = "low"
    
    return DriftResult(
        period_from=period_from,
        period_to=period_to,
        drift_score=round(drift_score, 2),
        similarity=round(jaccard_sim, 4),
        semantic_similarity=round(semantic_sim, 4) if semantic_sim is not None else None,
        severity=severity,
        added_sentences=added_sentences,
        removed_sentences=removed_sentences,
        new_risk_keywords=new_keywords,
        removed_risk_keywords=removed_keywords,
    )


# =============================================================================
# Multi-Period Analysis
# =============================================================================


def analyze_risk_drift(
    ticker: str,
    period_texts: Dict[str, str],
) -> SignalAlert:
    """Analyze drift across multiple periods.
    
    Compares consecutive periods and generates a signal alert
    summarizing the overall risk language changes.
    
    Args:
        ticker: Company ticker (e.g., "NVDA")
        period_texts: Dict mapping period labels to text content
                     e.g., {"Q1 FY2025": "risk text...", "Q2 FY2025": "risk text..."}
    
    Returns:
        SignalAlert with drift analysis across all period pairs
    """
    drift_results: List[DriftResult] = []
    max_severity = "low"
    severity_order = {"low": 0, "moderate": 1, "critical": 2}
    
    if len(period_texts) < 2:
        return SignalAlert(
            ticker=ticker,
            signal_type="risk_drift",
            severity="low",
            headline=f"Insufficient data for {ticker} drift analysis",
            details=f"Only {len(period_texts)} period(s) provided. Need at least 2.",
            drift_results=[],
            generated_at=datetime.now(),
        )
    
    # Sort periods and compare consecutive pairs
    sorted_periods = sorted(period_texts.keys())
    
    for i in range(len(sorted_periods) - 1):
        period_from = sorted_periods[i]
        period_to = sorted_periods[i + 1]
        
        result = calculate_drift(
            text_from=period_texts[period_from],
            text_to=period_texts[period_to],
            period_from=period_from,
            period_to=period_to,
        )
        
        drift_results.append(result)
        
        # Track maximum severity
        if severity_order[result.severity] > severity_order[max_severity]:
            max_severity = result.severity
    
    # Generate headline based on severity
    if max_severity == "critical":
        headline = f"CRITICAL: Major risk language changes detected for {ticker}"
    elif max_severity == "moderate":
        headline = f"MODERATE: Notable risk factor changes for {ticker}"
    else:
        headline = f"LOW: Routine risk factor updates for {ticker}"
    
    # Generate details summary
    details_lines = []
    for dr in drift_results:
        icon = "CRITICAL" if dr.severity == "critical" else "MODERATE" if dr.severity == "moderate" else "LOW"
        details_lines.append(f"[{icon}] {dr.period_from} -> {dr.period_to}: Score {dr.drift_score:.1f}")
        if dr.new_risk_keywords:
            details_lines.append(f"   New terms: {', '.join(dr.new_risk_keywords[:5])}")
    
    return SignalAlert(
        ticker=ticker,
        signal_type="risk_drift",
        severity=max_severity,
        headline=headline,
        details="\n".join(details_lines),
        drift_results=drift_results,
        generated_at=datetime.now(),
    )


# =============================================================================
# Report Formatting
# =============================================================================


def format_signal_report(alert: SignalAlert) -> str:
    """Format alert as professional markdown report with visual diff.
    
    CRITICAL: This must show the actual text changes, not just scores.
    The visual diff is key for understanding risk factor evolution.
    
    Args:
        alert: SignalAlert to format
        
    Returns:
        Formatted markdown string
    """
    lines = [
        "=" * 60,
        f"SEMANTIC SIGNAL ALERT: {alert.ticker}",
        "=" * 60,
        "",
        alert.headline,
        "",
        "Risk Factors Drift Analysis (Item 1A)",
        "-" * 40,
        "",
    ]
    
    for dr in alert.drift_results:
        severity_label = dr.severity.upper()
        lines.append(f"[{severity_label}] {dr.period_from} -> {dr.period_to}")
        lines.append(f"   Drift Score: {dr.drift_score:.1f}/100 ({dr.severity.upper()})")
        lines.append(f"   Jaccard Similarity: {dr.similarity:.1%}")
        
        if dr.semantic_similarity is not None:
            lines.append(f"   Semantic Similarity: {dr.semantic_similarity:.1%}")
        
        if dr.new_risk_keywords:
            lines.append(f"   NEW Risk Terms: {', '.join(dr.new_risk_keywords)}")
        
        if dr.removed_risk_keywords:
            lines.append(f"   REMOVED Risk Terms: {', '.join(dr.removed_risk_keywords)}")
        
        # Show added sentences (THE KEY PART - the visual diff)
        if dr.added_sentences:
            lines.append("")
            lines.append("   NEW LANGUAGE ADDED:")
            for sentence in dr.added_sentences[:3]:  # Limit to top 3
                # Truncate long sentences for readability
                display = sentence[:150] + "..." if len(sentence) > 150 else sentence
                lines.append(f"   + {display}")
        
        # Show removed sentences
        if dr.removed_sentences:
            lines.append("")
            lines.append("   LANGUAGE REMOVED:")
            for sentence in dr.removed_sentences[:3]:
                display = sentence[:150] + "..." if len(sentence) > 150 else sentence
                lines.append(f"   - {display}")
        
        lines.append("")
    
    # Quant interpretation section
    lines.append("=" * 60)
    lines.append("QUANT INTERPRETATION")
    lines.append("=" * 60)
    
    if alert.severity == "critical":
        lines.append("CRITICAL ANOMALY DETECTED")
        lines.append("Risk Factors section was substantially rewritten.")
        lines.append("Management may be pricing in material changes not yet")
        lines.append("reflected in structured financial data.")
    elif alert.severity == "moderate":
        lines.append("Notable changes in risk language.")
        lines.append("Worth monitoring but not yet alarming.")
    else:
        lines.append("Routine quarterly updates.")
        lines.append("No significant structural changes detected.")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


