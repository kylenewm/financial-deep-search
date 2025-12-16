"""
Pydantic models for financial fact extraction and verification.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class EntityInfo(BaseModel):
    """Information about a company/entity resolved from ticker or name."""
    
    ticker: str
    company_name: str
    cik: str  # 10-digit zero-padded string
    fiscal_year_end: Optional[str] = None  # Month name: "January", "December", etc.


class DocumentSnapshot(BaseModel):
    """Immutable snapshot of an SEC filing document."""
    
    snapshot_id: str  # UUID
    url: str
    cik: str
    doc_type: str  # 10-K, 10-Q, 8-K
    doc_date: str = ""  # Filing date or accession number for period identification
    retrieved_at: datetime
    content_hash: str  # SHA-256 of raw_html
    raw_html: str


class Location(BaseModel):
    """Identifies the exact location of a fact within an SEC filing or news article."""
    
    cik: str
    doc_date: str
    doc_type: str  # 10-K, 10-Q, 8-K, or "news"
    section_id: str  # Item7, Item1A, etc. (empty for news)
    
    # For TEXT facts (from paragraphs)
    paragraph_index: Optional[int] = None
    sentence_string: Optional[str] = None
    
    # For TABLE facts (from cells)
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    row_label: Optional[str] = None      # e.g., "Data Center"
    column_label: Optional[str] = None   # e.g., "Oct 27, 2024"
    
    # For NEWS facts
    article_url: Optional[str] = None
    article_title: Optional[str] = None
    article_domain: Optional[str] = None
    article_published_date: Optional[str] = None


class FactContext(BaseModel):
    """Additional context about a financial fact."""
    
    yoy_change: Optional[str] = None
    vs_guidance: Optional[str] = None


class Fact(BaseModel):
    """A single financial fact extracted from an SEC filing or news article."""
    
    fact_id: str
    entity: str  # ticker
    metric: str
    value: Optional[float] = None
    unit: str
    period: str
    period_end_date: str  # EXACT date from source, e.g., "2024-10-27"
    location: Location
    source_format: Literal["html_text", "html_table", "xbrl", "news"]
    
    # For table facts: the scale defined in the table header
    # e.g., "millions", "thousands", "billions"
    extracted_scale: Optional[str] = None
    
    doc_hash: str
    snapshot_id: str
    verification_status: str  # exact_match, approximate_match, mismatch, unverified
    negative_evidence: Optional[str] = None
    context: Optional[FactContext] = None
    
    # Trust level for ranking facts from different source types
    # "high" = XBRL (authoritative), "medium" = news (from Tier 1 sources), "low" = other
    trust_level: Literal["high", "medium", "low"] = "high"
    
    # P1 Enhancement: Extraction provenance for debugging and auditability
    # Tracks HOW the fact was extracted, enabling confidence-based filtering
    extraction_method: Optional[Literal[
        "toc_anchor",      # HIGH confidence: found via Table of Contents link
        "dom_scan",        # MED confidence: found via DOM header scanning
        "fuzzy",           # LOW confidence: fuzzy text matching fallback
        "xbrl",            # HIGH confidence: structured XBRL data
        "regex_fallback",  # LOW confidence: regex-only extraction
    ]] = None
    extraction_confidence: Optional[float] = None  # 0.0-1.0 confidence score
    source_hash: Optional[str] = None  # Content hash for reproducibility checks


# =============================================================================
# Parsing Models
# =============================================================================


class RiskFactorsExtract(BaseModel):
    """Result of extracting Item 1A (Risk Factors) from an SEC filing.
    
    This model captures both the extracted text AND metadata about how
    it was extracted. The metadata is crucial for:
    - Debugging extraction failures
    - Understanding confidence in drift results
    - Failing explicitly when extraction is uncertain
    
    Extraction methods (in priority order):
    - toc_anchor: Found via Table of Contents link (most reliable)
    - dom_scan: Found via header matching in document (medium confidence)
    - fuzzy: Best-effort match when others fail (low confidence)
    
    ⚠️ If all methods fail, this returns None. Callers MUST fail closed.
    """
    
    text: str                                   # Extracted Risk Factors text
    method: Literal["toc_anchor", "dom_scan", "fuzzy"]  # How it was extracted
    confidence: Literal["HIGH", "MED", "LOW"]   # Confidence level
    reason: str                                 # Why this method was chosen
    char_count: int                             # Length of extracted text
    
    # Optional metadata for debugging
    anchor_id: Optional[str] = None             # If toc_anchor, what anchor was used
    header_text: Optional[str] = None           # If dom_scan, what header text matched
    
    def is_reliable(self) -> bool:
        """Check if extraction is reliable enough for signal detection."""
        # Only HIGH confidence (toc_anchor) is fully reliable
        # MED is acceptable but should be flagged
        # LOW should trigger warnings
        return self.confidence == "HIGH"
    
    def get_confidence_icon(self) -> str:
        """Visual indicator for confidence level."""
        return {
            "HIGH": "🟢",
            "MED": "🟡", 
            "LOW": "🔴",
        }.get(self.confidence, "⚪")


class Paragraph(BaseModel):
    """A paragraph extracted from a filing section."""
    
    index: int  # 0-based index within section
    text: str   # Clean text content (whitespace normalized)
    html: str   # Original HTML for reference


class Section(BaseModel):
    """A section of an SEC filing (e.g., Item 7, Item 1A).
    
    NOTE: 10-Qs can have both Part I and Part II with same Item numbers.
    Use the `part` field to distinguish: "Part I, Item 1" vs "Part II, Item 1".
    The `section_id` is kept for backwards compatibility.
    """
    
    section_id: str       # Normalized: "Item7", "Item1A"
    title: str            # Full title text if available
    paragraphs: List[Paragraph]
    raw_html: str         # Original HTML of entire section
    
    # Part identity for disambiguation (Phase 1 fix)
    part: Optional[str] = None                  # "Part I", "Part II", or None
    full_section_id: Optional[str] = None       # "Part I, Item 1" for unambiguous lookups


class CoverPageMetadata(BaseModel):
    """Metadata extracted from the filing cover page."""
    
    fiscal_period_end_date: Optional[str] = None  # e.g., "January 28, 2024"
    fiscal_period_type: Optional[str] = None      # "annual" or "quarterly"
    company_name: Optional[str] = None


class ParsedFiling(BaseModel):
    """A fully parsed SEC filing."""
    
    cik: str
    company_name: Optional[str] = None
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    cover_page: CoverPageMetadata
    sections: List[Section]
    raw_html: str         # Original full document HTML


# =============================================================================
# Table Models
# =============================================================================


class TableCell(BaseModel):
    """A single cell extracted from a financial table."""
    
    value: str              # Raw string value from cell
    row_label: str          # Label from first column (e.g., "Data Center")
    column_label: str       # Header of the column (e.g., "Oct 27, 2024")
    row_index: int          # 0-based row index
    column_index: int       # 0-based column index
    effective_scale: Optional[str] = None  # Scale to apply for THIS cell (None for per-share)


class ExtractedTable(BaseModel):
    """A table extracted from an SEC filing."""
    
    table_index: int        # Index within document or section
    section_id: Optional[str] = None  # Which section it came from
    html: str               # Original table HTML
    headers: List[str]      # Column headers (flattened if MultiIndex)
    row_count: int          # Number of data rows
    column_count: int       # Number of columns
    scale: Optional[str] = None  # Base scale from header: "millions", "thousands", etc.
    has_per_share_exception: bool = False  # True if "except per share" in header
    
    # Store DataFrame as JSON for serialization
    dataframe_json: str
    
    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Conflict Detection Models
# =============================================================================


class ConflictingValue(BaseModel):
    """A single value involved in a fact conflict."""
    
    value: float
    fact_id: str
    source_description: str  # e.g., "10-Q 2024-11-20"


class Conflict(BaseModel):
    """A conflict where same entity+metric+period has different values."""
    
    entity: str
    metric: str
    period: str
    values: List[ConflictingValue]


# =============================================================================
# Report Generation Models
# =============================================================================


class Analysis(BaseModel):
    """LLM-generated analysis/interpretation of verified facts."""
    
    summary: str
    classification: Literal["thesis"] = "thesis"
    supporting_facts: List[str]  # List of fact_ids


class NotFoundMetric(BaseModel):
    """A metric that was requested but not found in sources."""
    
    metric: str
    status: str = "Not found in retrieved Tier 1/2 sources"


# =============================================================================
# Research Output Models
# =============================================================================


class ResearchOutput(BaseModel):
    """Complete structured output from a research query.
    
    This is the primary output format. The JSON representation
    can be pulled directly into Excel or Python for further analysis.
    """
    
    query: str
    generated_at: datetime
    as_of_date: Optional[str] = None  # For time-machine mode
    facts: List[Fact]
    analysis: Optional[Analysis] = None
    conflicts: List[Conflict] = []
    not_found: List[NotFoundMetric] = []
    
    # Metadata (computed)
    total_facts: int = 0
    verified_facts: int = 0
    sources_used: List[str] = []  # List of unique snapshot_ids
    
    def model_post_init(self, __context) -> None:
        """Compute metadata fields after initialization."""
        self.total_facts = len(self.facts)
        self.verified_facts = len([
            f for f in self.facts 
            if f.verification_status in ("exact_match", "approximate_match")
        ])
        self.sources_used = list(set(f.snapshot_id for f in self.facts))


# =============================================================================
# News Search Models
# =============================================================================


class NewsResult(BaseModel):
    """A news article result from search."""
    
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None  # May not always be available
    domain: str
    tier: int  # 1, 2, or 3
    retrieved_at: datetime


# Domain tier classification
NEWS_DOMAIN_TIERS: dict[str, int] = {
    # Tier 1 - Highest credibility
    "wsj.com": 1,
    "reuters.com": 1,
    "bloomberg.com": 1,
    "ft.com": 1,
    
    # Tier 2 - Good credibility
    "cnbc.com": 2,
    "marketwatch.com": 2,
    "barrons.com": 2,
    "economist.com": 2,
    
    # Tier 3 - Lower credibility (use with caution)
    "seekingalpha.com": 3,
    "yahoo.com": 3,
    "benzinga.com": 3,
}


def get_domain_tier(url: str) -> int:
    """Get tier for a domain, default to 3 if unknown."""
    url_lower = url.lower()
    for domain, tier in NEWS_DOMAIN_TIERS.items():
        if domain in url_lower:
            return tier
    return 3  # Unknown domains default to Tier 3


# =============================================================================
# Evaluation Models
# =============================================================================


class EvalQuestion(BaseModel):
    """A single evaluation question with known answer."""
    
    question_id: str
    question: str
    entity: str  # ticker
    metric: str
    expected_value: float
    expected_unit: str
    expected_period: str
    authoritative_source: str  # e.g., "NVIDIA 10-Q Q3 FY2025"
    notes: Optional[str] = None  # Any special notes about this question


class EvalResult(BaseModel):
    """Result of evaluating a single question."""
    
    question_id: str
    question: str
    
    # Expected
    expected_value: float
    expected_period: str
    
    # Actual
    actual_value: Optional[float] = None
    actual_period: Optional[str] = None
    
    # Scores
    value_correct: bool  # Within 1% tolerance
    period_correct: bool  # Exact period match
    source_tier_correct: bool  # Used Tier 1/2 source
    
    # Error classification
    # "retrieval_failure", "extraction_failure", "verification_failure", 
    # "hallucination", "period_mismatch", None
    error_type: Optional[str] = None
    
    # Details
    facts_returned: int
    verification_status: Optional[str] = None


class EvalSummary(BaseModel):
    """Summary of full evaluation run."""
    
    total_questions: int
    
    # Accuracy metrics
    value_accuracy: float  # % correct values (within 1%)
    period_accuracy: float  # % correct periods
    source_tier_accuracy: float  # % using correct source tier
    
    # Error breakdown
    retrieval_failures: int  # Didn't find the right document
    extraction_failures: int  # Found doc but didn't extract fact
    verification_failures: int  # Extracted but failed verification
    hallucinations: int  # Returned wrong value confidently
    
    # Overall
    pass_rate: float  # % of questions fully correct
    
    # Details
    results: List[EvalResult]


# =============================================================================
# Narrator Models
# =============================================================================


class Citation(BaseModel):
    """A citation linking narrated text to a verified fact."""
    
    fact_id: str
    citation_index: int  # 1-based index in the report (e.g., [1], [2])
    source_format: str  # "xbrl", "html_text", "html_table"
    location: str  # Human-readable location (e.g., "us-gaap:Revenues, Q3 FY2025")


class NarratedReport(BaseModel):
    """A report generated by the LLM narrator from verified facts.
    
    The narrator can ONLY include information from the facts_used list.
    Each statement must be cited with [1], [2], etc.
    """
    
    query: str
    answer: str
    citations: List[Citation]
    facts_used: List[Fact]
    
    # Metadata
    generated_at: datetime
    insufficient_data: bool = False  # True if facts didn't answer the query


# =============================================================================
# Drift Detection Models
# =============================================================================


class DriftResult(BaseModel):
    """Result of comparing two text blocks for semantic drift.
    
    Used to detect changes in SEC Risk Factors across quarters.
    The key insight: if the Risk section text changes significantly,
    management may be pricing in material changes not yet reflected
    in structured financial data.
    """
    
    period_from: str
    period_to: str
    drift_score: float  # 0-100, higher = more change
    similarity: float   # 0-1 Jaccard similarity
    semantic_similarity: Optional[float] = None  # 0-1 embedding cosine (if available)
    severity: Literal["low", "moderate", "critical"]
    
    # CRITICAL: These fields enable the visual diff - the "wow" moment
    added_sentences: List[str]    # Sentences that appeared
    removed_sentences: List[str]  # Sentences that disappeared
    new_risk_keywords: List[str]  # Risk terms that appeared
    removed_risk_keywords: List[str]  # Risk terms removed


class SignalAlert(BaseModel):
    """Alpha signal from text analysis.
    
    Represents a trading signal derived from unstructured text analysis.
    This is the differentiating feature - derived from unstructured filings,
    not available from standard data vendors.
    """
    
    ticker: str
    signal_type: str  # "risk_drift", "guidance_change", etc.
    severity: Literal["low", "moderate", "critical"]
    headline: str
    details: str
    drift_results: List[DriftResult]
    generated_at: datetime


class SignalRecord(BaseModel):
    """
    Persistent record of a signal detection run.
    
    Flat, table-like schema designed for:
    - Backtesting against returns (filing_date is the join key)
    - Cross-ticker comparison
    - Research iteration without schema changes
    - Easy migration to SQLite/PostgreSQL
    
    Key invariant: filing_date (announcement date) is the backtest join key,
    not period_end_date, to avoid lookahead bias.
    """
    
    # Identity
    signal_id: str  # UUID
    ticker: str
    cik: str
    
    # Filing context
    filing_type: str  # "10-K", "10-Q" (deprecated - use base/compare_filing_type)
    base_accession: str
    compare_accession: str
    
    # Signal mode (P3 - dual filing logic)
    signal_mode: str = "quarterly"  # "regime", "event", "quarterly"
    base_filing_type: str = "10-Q"  # "10-K" or "10-Q"
    compare_filing_type: str = "10-Q"  # "10-K" or "10-Q"
    
    # Dates - ALL THREE REQUIRED for valid backtests
    base_period_end_date: str       # Fiscal period end (e.g., "2024-01-28")
    compare_period_end_date: str    # Fiscal period end
    filing_date: str                # BACKTEST JOIN KEY - when filing went public
    
    # Fiscal labels (human-readable)
    base_fiscal_period: str         # "Q4 FY2024"
    compare_fiscal_period: str      # "Q3 FY2025"
    
    # Primitives (decomposed - don't collapse to single score)
    drift_score: float              # 0-100
    jaccard_similarity: float       # 0-1
    semantic_similarity: Optional[float] = None  # 0-1, None if embeddings unavailable
    
    # Sentence counts
    new_sentence_count: int
    removed_sentence_count: int
    
    # Keywords (flattened for SQL compatibility)
    new_keyword_count: int
    removed_keyword_count: int
    new_keywords_json: str          # JSON array as string
    removed_keywords_json: str      # JSON array as string
    
    # Boilerplate detection (P2)
    boilerplate_flag: bool          # True if ratio > 0.6
    boilerplate_ratio: float        # 0-1
    
    # Derived (for filtering, not ranking)
    severity: str                   # "low", "moderate", "critical"
    
    # Audit
    created_at: str                 # ISO timestamp
    model_version: str              # For reproducibility
    
    # Pointers
    base_snapshot_id: str
    compare_snapshot_id: str


# =============================================================================
# Discovery Models (Lead Generation)
# =============================================================================


class Lead(BaseModel):
    """A lead/claim found during discovery that can be investigated.
    
    Leads are hypotheses from web/news that may or may not be accurate.
    They should be verified against hard sources before acting on them.
    """
    
    lead_id: str                 # Unique identifier
    text: str                    # The claim text
    source_url: Optional[str] = None    # Where it came from
    source_name: str             # "Reuters", "Bloomberg", etc.
    source_tier: int = 3         # 1 (highest trust), 2, or 3
    found_at: datetime
    
    # Extracted info (if parseable)
    entity: Optional[str] = None      # Ticker if detected
    metric: Optional[str] = None      # "revenue", "margin", etc.
    value: Optional[float] = None     # Parsed numeric value
    value_raw: Optional[float] = None # Normalized to base units
    period: Optional[str] = None      # "Q3 FY2025", etc.
    
    # Classification
    lead_type: Literal["quantitative", "qualitative", "event"] = "qualitative"
    
    # Verification status (filled in by verify step)
    verification_status: Optional[Literal["pending", "confirmed", "contradicted", "unverifiable"]] = "pending"
    verification_details: Optional[str] = None
    verified_against_fact_id: Optional[str] = None  # If confirmed/contradicted
    
    def get_status_icon(self) -> str:
        """Get visual indicator for verification status."""
        icons = {
            "pending": "🔍",
            "confirmed": "✅",
            "contradicted": "🔴",
            "unverifiable": "⚪",
        }
        return icons.get(self.verification_status or "pending", "🔍")


class DiscoveryReport(BaseModel):
    """Result of discovery + verification pipeline.
    
    Contains all discovered leads and their verification status,
    plus summary statistics.
    """
    
    query: str
    ticker: Optional[str] = None
    leads: List[Lead]
    generated_at: datetime
    
    # Summary (computed)
    total_leads: int = 0
    confirmed_count: int = 0
    contradicted_count: int = 0
    unverifiable_count: int = 0
    pending_count: int = 0
    
    def model_post_init(self, __context) -> None:
        """Compute summary stats after init."""
        self.total_leads = len(self.leads)
        self.confirmed_count = sum(1 for l in self.leads if l.verification_status == "confirmed")
        self.contradicted_count = sum(1 for l in self.leads if l.verification_status == "contradicted")
        self.unverifiable_count = sum(1 for l in self.leads if l.verification_status == "unverifiable")
        self.pending_count = sum(1 for l in self.leads if l.verification_status == "pending")
    
    def format_report(self) -> str:
        """Format as human-readable report."""
        lines = [
            "━" * 60,
            f"🔍 DISCOVERY REPORT: {self.ticker or 'General'}",
            "━" * 60,
            "",
            "⚠️  TIER 3 CONTEXT (Read-Only)",
            "    This report shows news/web claims verified AGAINST FactStore.",
            "    It does NOT write to FactStore. Use for context, not authority.",
            "",
            f"Found {self.total_leads} leads:",
            f"  ✅ Confirmed:    {self.confirmed_count}",
            f"  🔴 Contradicted: {self.contradicted_count}",
            f"  ⚪ Unverifiable: {self.unverifiable_count}",
            "",
        ]
        
        # Show contradicted first (most important)
        contradicted = [l for l in self.leads if l.verification_status == "contradicted"]
        if contradicted:
            lines.append("━" * 60)
            lines.append("🔴 CONTRADICTED (potential mispricing)")
            lines.append("━" * 60)
            for lead in contradicted:
                lines.append(f"  \"{lead.text}\"")
                lines.append(f"  Source: {lead.source_name}")
                lines.append(f"  ⚠️ {lead.verification_details}")
                lines.append("")
        
        # Then confirmed
        confirmed = [l for l in self.leads if l.verification_status == "confirmed"]
        if confirmed:
            lines.append("━" * 60)
            lines.append("✅ CONFIRMED")
            lines.append("━" * 60)
            for lead in confirmed:
                lines.append(f"  \"{lead.text}\"")
                lines.append(f"  {lead.verification_details}")
            lines.append("")
        
        # Then unverifiable
        unverifiable = [l for l in self.leads if l.verification_status == "unverifiable"]
        if unverifiable:
            lines.append("━" * 60)
            lines.append("⚪ UNVERIFIABLE (no SEC data to compare)")
            lines.append("━" * 60)
            for lead in unverifiable:
                lines.append(f"  \"{lead.text}\" ({lead.source_name})")
            lines.append("")
        
        lines.append("━" * 60)
        return "\n".join(lines)


# =============================================================================
# Cross-Verification Models
# =============================================================================


class Claim(BaseModel):
    """A claim extracted from a source that can be verified.
    
    Used to extract quantitative claims from news articles and verify
    them against hard sources (SEC XBRL data).
    """
    
    text: str                                    # Original claim text
    source: str                                  # Where it came from
    source_url: Optional[str] = None
    source_tier: Literal["hard", "medium", "soft"]
    claim_type: Literal["quantitative", "qualitative", "quote"]
    entity: Optional[str] = None                 # Ticker if identifiable
    metric: Optional[str] = None                 # e.g., "revenue"
    value: Optional[float] = None                # e.g., 35.0 (before normalization)
    value_raw: Optional[float] = None            # After normalization to base units
    unit_hint: Optional[str] = None              # e.g., "billion", "million", "B"
    period: Optional[str] = None                 # e.g., "Q3 FY2025"


class VerificationResult(BaseModel):
    """Result of verifying a claim against hard sources.
    
    Status meanings:
    - verified: claim matches hard source (within 1%)
    - close: claim is within 5% (likely rounding differences)
    - contradicted: claim differs by >5% from hard source
    - unverifiable: no hard source to check against
    """
    
    claim: Claim
    status: Literal["verified", "contradicted", "close", "unverifiable"]
    
    hard_source: Optional[str] = None            # e.g., "SEC XBRL"
    hard_value: Optional[float] = None           # The verified value (normalized)
    hard_period: Optional[str] = None
    
    difference_pct: Optional[float] = None       # % difference
    confidence: float = 0.0                      # 0-1 confidence
    explanation: str = ""                        # Human-readable
    
    def get_trust_icon(self) -> str:
        """Get visual trust indicator for the verification status."""
        icons = {
            "verified": "🟢",
            "contradicted": "🔴",
            "close": "🟡",
            "unverifiable": "⚪",
        }
        return icons.get(self.status, "⚪")
