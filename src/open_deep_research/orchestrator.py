"""
Simple Orchestrator - User-facing API for the research pipeline.

This is the main entry point for using the anti-hallucination research system.
It wraps all the existing components into a simple, clean interface:

    Question -> Router -> XBRL/HTML Extraction -> FactStore -> Narrator -> Cited Answer

Example usage:
    from open_deep_research.orchestrator import Orchestrator
    
    orc = Orchestrator()
    orc.load_facts_for_entity("NVDA", "Q3 FY2025")
    
    report = orc.ask("What was NVIDIA's revenue in Q3 FY2025?")
    print(report.answer)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from open_deep_research.classifier import (
    classify_query,
    classify_query_detailed,
    ClassificationResult,
    QueryType,
    ConfidenceBand,
    get_route_description,
    extract_comparison_entities,
)
from open_deep_research.config import ResearchConfig, default_config
from open_deep_research.cross_verify import (
    extract_claims_from_text,
    verify_all_claims,
    format_verification_report,
)
from open_deep_research.discovery import (
    discover,
    discover_and_verify,
    discover_from_text,
    verify_leads,
)
from open_deep_research.entities import resolve_entity, FISCAL_YEAR_ENDS
from open_deep_research.models import Fact, NarratedReport, VerificationResult, DiscoveryReport, Lead, Citation
from open_deep_research.narrator import generate_report
from open_deep_research.router import (
    route_query,
    extract_metric_from_question,
    SourceType,
    XBRL_METRICS,
)
from open_deep_research.store import FactStore
from open_deep_research.xbrl import extract_xbrl_fact, METRIC_TO_CONCEPTS
from open_deep_research.signals import analyze_risk_drift, format_signal_report, detect_boilerplate, SignalMode
from open_deep_research.ingestion import get_recent_filings, get_filing_metadata, get_signal_filings
from open_deep_research.parsing import parse_filing_html, extract_risk_factors
from open_deep_research.section_locator import extract_risk_factors_from_html
from open_deep_research.store import SignalStore
from open_deep_research.models import SignalRecord, RiskFactorsExtract


logger = logging.getLogger(__name__)


# =============================================================================
# Period Parsing
# =============================================================================


def parse_fiscal_period(period_str: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a period string into fiscal year and period.
    
    Args:
        period_str: e.g., "Q3 FY2025", "FY2024", "2024 Q3", "Q3 2024"
        
    Returns:
        Tuple of (fiscal_year, fiscal_period) e.g., (2025, "Q3")
    """
    period_str = period_str.upper().strip()
    
    # Pattern 1: "Q3 FY2025", "Q3FY2025", "FY2025"
    match = re.match(r'(Q[1-4])?\s*FY(\d{4})', period_str)
    if match:
        quarter = match.group(1)  # "Q3" or None
        year = int(match.group(2))
        return year, quarter if quarter else "FY"
        
    # Pattern 2: "Q3 2025", "Q3 25" (assume 20xx)
    match = re.match(r'(Q[1-4])\s*(\d{4}|\d{2})', period_str)
    if match:
        quarter = match.group(1)
        year_str = match.group(2)
        year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
        return year, quarter

    # Pattern 3: "2025 Q3"
    match = re.match(r'(\d{4})\s*(Q[1-4])', period_str)
    if match:
        year = int(match.group(1))
        quarter = match.group(2)
        return year, quarter
        
    # Pattern 4: Just year "2025" -> FY2025
    match = re.match(r'^(\d{4})$', period_str)
    if match:
        return int(match.group(1)), "FY"
    
    return None, None


def extract_entity_from_question(question: str) -> Optional[str]:
    """
    Extract entity (ticker) from a question.
    
    Args:
        question: The full question text
        
    Returns:
        Ticker if found, None otherwise
    """
    question_lower = question.lower()
    
    # Check for known company names and their variations
    company_patterns = {
        "nvidia": "NVDA",
        "nvda": "NVDA",
        "apple": "AAPL",
        "aapl": "AAPL",
        "microsoft": "MSFT",
        "msft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "googl": "GOOGL",
        "amazon": "AMZN",
        "amzn": "AMZN",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "tsla": "TSLA",
        "amd": "AMD",
        "intel": "INTC",
        "intc": "INTC",
        "walmart": "WMT",
        "wmt": "WMT",
    }
    
    for pattern, ticker in company_patterns.items():
        if pattern in question_lower:
            return ticker
    
    # Try to find ticker pattern (uppercase 2-5 letters)
    ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
    if ticker_match:
        potential_ticker = ticker_match.group(1)
        # Verify it's a known ticker
        entity = resolve_entity(potential_ticker)
        if entity:
            return entity.ticker
    
    return None


def extract_period_from_question(question: str) -> Optional[str]:
    """
    Extract fiscal period from a question.
    
    Args:
        question: The full question text
        
    Returns:
        Period string like "Q3 FY2025" if found
    """
    question_upper = question.upper()
    
    # Pattern: "Q3 FY2025", "Q3FY2025"
    match = re.search(r'(Q[1-4])\s*FY(\d{4})', question_upper)
    if match:
        return f"{match.group(1)} FY{match.group(2)}"
    
    # Pattern: "Q3 2025"
    match = re.search(r'(Q[1-4])\s+(\d{4})', question_upper)
    if match:
        return f"{match.group(1)} FY{match.group(2)}"
    
    # Pattern: "FY2025"
    match = re.search(r'FY(\d{4})', question_upper)
    if match:
        return f"FY{match.group(1)}"
    
    return None


# =============================================================================
# Orchestrator Class
# =============================================================================


class Orchestrator:
    """
    Simple orchestrator for the anti-hallucination research pipeline.
    
    This class wraps all the existing components into a clean API:
    - FactStore: Holds verified facts
    - Router: Decides XBRL vs LLM path
    - XBRL extractor: Gets structured financial data
    - Narrator: Generates cited answers from facts
    
    Example:
        orc = Orchestrator()
        orc.load_facts_for_entity("NVDA", "Q3 FY2025")
        report = orc.ask("What was NVIDIA's revenue?")
        print(report.answer)
    """
    
    def __init__(self, config: ResearchConfig = None) -> None:
        """Initialize the orchestrator with an empty FactStore.
        
        Args:
            config: Research configuration (uses default if not provided)
        """
        self.fact_store = FactStore()
        self.config = config or default_config
        self._loaded_periods: set[Tuple[str, str]] = set()  # (ticker, period)
        self._start_time = datetime.now()
        logger.info("Orchestrator initialized")
    
    def health_check(self) -> Dict[str, Any]:
        """Return system health status for monitoring.
        
        Provides a snapshot of the orchestrator's current state,
        useful for health checks, monitoring dashboards, and debugging.
        
        Returns:
            Dict with health metrics:
            - status: "healthy" or "degraded"
            - fact_store_count: Number of facts in store
            - loaded_periods: List of (ticker, period) tuples loaded
            - embeddings_available: Whether semantic embeddings are available
            - nlp_available: Whether spaCy NLP is available
            - uptime_seconds: Time since initialization
        """
        from open_deep_research.signals import get_nlp
        from open_deep_research.classifier import _get_embedder
        
        # Check component availability
        embeddings_available = _get_embedder() is not None
        nlp_available = get_nlp() is not None
        
        # Calculate uptime
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        # Determine overall status
        # Degraded if missing key components but still functional
        status = "healthy"
        degraded_reasons = []
        
        if not embeddings_available:
            degraded_reasons.append("embeddings_unavailable")
        if not nlp_available:
            degraded_reasons.append("nlp_unavailable")
        
        if degraded_reasons:
            status = "degraded"
        
        return {
            "status": status,
            "fact_store_count": len(self.fact_store),
            "loaded_periods": list(self._loaded_periods),
            "embeddings_available": embeddings_available,
            "nlp_available": nlp_available,
            "uptime_seconds": round(uptime, 2),
            "degraded_reasons": degraded_reasons if degraded_reasons else None,
        }
    
    def load_facts_for_entity(
        self,
        ticker: str,
        period: str,
        metrics: Optional[List[str]] = None,
    ) -> int:
        """
        Pre-load facts from XBRL for an entity and period.
        
        This populates the FactStore with verified facts BEFORE
        answering questions. This is the "narrator over verified fact table"
        pattern - facts are loaded first, then narrated.
        
        Args:
            ticker: Stock ticker (e.g., "NVDA", "AAPL")
            period: Fiscal period (e.g., "Q3 FY2025", "FY2024")
            metrics: Optional list of specific metrics to load.
                     If None, loads all available metrics.
                     
        Returns:
            Number of facts loaded
        """
        # Check if already loaded
        key = (ticker.upper(), period.upper())
        if key in self._loaded_periods:
            logger.info(f"Facts for {ticker} {period} already loaded")
            return 0
        
        # Resolve entity
        entity = resolve_entity(ticker)
        if not entity:
            logger.warning(f"Could not resolve entity: {ticker}")
            return 0
        
        # Parse period
        fiscal_year, fiscal_period = parse_fiscal_period(period)
        if not fiscal_year or not fiscal_period:
            logger.warning(f"Could not parse period: {period}")
            return 0
        
        # Determine which metrics to load
        if metrics is None:
            # Load common financial metrics
            metrics = [
                "total revenue", "net income", "gross profit",
                "operating income", "cost of revenue",
                "earnings per share", "eps",
                "total assets", "total liabilities",
                "cash and cash equivalents",
                "r&d expenses", "operating expenses",
            ]
        
        logger.info(f"Loading XBRL facts for {ticker} {period}...")
        
        count = 0
        for metric in metrics:
            fact = extract_xbrl_fact(
                cik=entity.cik,
                metric=metric,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                ticker=ticker,
            )
            
            if fact:
                try:
                    self.fact_store.add_fact(fact)
                    count += 1
                    logger.debug(f"  Loaded: {fact.metric} = {fact.value}")
                except ValueError as e:
                    logger.warning(f"  Could not add fact: {e}")
        
        self._loaded_periods.add(key)
        logger.info(f"Loaded {count} facts for {ticker} {period}")
        
        return count
    
    def load_facts_for_periods(
        self,
        ticker: str,
        periods: List[str],
    ) -> int:
        """
        Load facts for multiple periods (for trend analysis).
        
        Args:
            ticker: Stock ticker
            periods: List of periods (e.g., ["Q1 FY2025", "Q2 FY2025", "Q3 FY2025"])
            
        Returns:
            Total number of facts loaded
        """
        total = 0
        for period in periods:
            count = self.load_facts_for_entity(ticker, period)
            total += count
        return total
    
    def ask(
        self,
        question: str,
        auto_load: bool = True,
        verbose: bool = False,
    ) -> NarratedReport:
        """
        Answer a question with intelligent routing.
        
        This is the main entry point. The process:
        1. Classify the query to determine route
        2. Handle DEGRADED MODE if embeddings unavailable
        3. Parse question to extract entity, metric, period
        4. Route to appropriate handler based on classification
        5. Return cited answer
        
        Args:
            question: The research question (e.g., "What was NVIDIA's revenue in Q3 FY2025?")
            auto_load: If True, automatically load facts if not already loaded
            verbose: If True, print routing information
            
        Returns:
            NarratedReport with answer, citations, and facts used
        """
        logger.info(f"Question: {question}")
        
        # Get detailed classification result
        result = classify_query_detailed(question)
        query_type = result.query_type
        confidence = result.similarity
        
        # =================================================================
        # DEGRADED MODE HANDLING
        # When embeddings unavailable, we CANNOT claim semantic understanding.
        # Route to SAFE path: SEC_HTML (Tier 2) + verification, NEVER XBRL.
        # =================================================================
        if result.method == "regex_degraded":
            logger.warning(
                f"[DEGRADED MODE] Routing to safe path. "
                f"Hints: {result.scores.get('regex_hints', [])}"
            )
            if verbose:
                print("⚠️ DEGRADED MODE: Embeddings unavailable, routing to safe verification path")
            
            return self._handle_degraded_mode(question, result, auto_load)
        
        if verbose:
            print(f"🧭 Route: {get_route_description(query_type)} (confidence: {confidence:.0%})")
        
        logger.debug(f"Classified as: {query_type.value} (confidence: {confidence:.2f})")
        
        # Route to appropriate handler
        if query_type == QueryType.FINANCIAL_LOOKUP:
            return self._handle_financial_lookup(question, auto_load)
        
        elif query_type == QueryType.SIGNAL_DETECTION:
            return self._handle_signal_detection(question)
        
        elif query_type == QueryType.VERIFICATION:
            return self._handle_verification(question)
        
        elif query_type == QueryType.COMPARISON:
            return self._handle_comparison(question, auto_load)
        
        elif query_type == QueryType.QUALITATIVE_EXTRACT:
            return self._handle_qualitative(question)
        
        elif query_type == QueryType.EXPLORATION:
            return self._handle_exploration(question)
        
        elif query_type == QueryType.DISCOVERY:
            return self._handle_discovery(question)
        
        elif query_type == QueryType.UNKNOWN:
            # Unknown with high-quality classification - ask for clarification
            return self._handle_unknown(question)
        
        else:
            # Fallback - should never reach here with proper enum handling
            logger.warning(f"Unhandled query type: {query_type}")
            return self._handle_unknown(question)
    
    def _handle_financial_lookup(
        self, 
        question: str, 
        auto_load: bool
    ) -> NarratedReport:
        """Handle financial metric lookups via XBRL."""
        # Extract entity, metric, period from question
        entity = extract_entity_from_question(question)
        metric = extract_metric_from_question(question)
        period = extract_period_from_question(question)
        
        logger.debug(f"Parsed: entity={entity}, metric={metric}, period={period}")
        
        # Auto-load facts if needed
        if auto_load and entity and period:
            key = (entity.upper(), period.upper())
            if key not in self._loaded_periods:
                logger.info(f"Auto-loading facts for {entity} {period}")
                self.load_facts_for_entity(entity, period)
        
        # Route the question to see what type of data is needed
        route_result = route_query(question, entity)
        
        # If XBRL metric but no facts loaded, try to load now
        if (
            auto_load 
            and route_result.source_type == SourceType.XBRL 
            and len(self.fact_store) == 0
            and entity
            and period
        ):
            self.load_facts_for_entity(entity, period)
        
        # Generate report using narrator
        report = generate_report(
            query=question,
            fact_store=self.fact_store,
            entity=entity,
            metric=metric,
            period=period,
        )
        
        return report
    
    def _handle_signal_detection(
        self, 
        question: str,
        mode: SignalMode = SignalMode.REGIME,
    ) -> NarratedReport:
        """Handle signal/drift detection requests.
        
        Signal detection analyzes changes in SEC Risk Factors (Item 1A) across
        periods to detect potential alpha signals before they appear in financials.
        
        Modes (P3 dual-filing logic):
        - REGIME: 10-K → 10-K comparison (annual baseline, substantive changes)
        - EVENT: 10-Q → 10-K comparison (fast overlay, novel risk detection)
        - QUARTERLY: 10-Q → 10-Q comparison (legacy, high noise)
        
        This handler:
        1. Fetches the correct filing pair based on mode
        2. Extracts Item 1A (Risk Factors) from each
        3. Runs semantic drift analysis
        4. In EVENT mode: suppresses if boilerplate-heavy
        5. Returns a formatted signal report
        
        Requires: Ticker extractable from question
        """
        from datetime import datetime
        
        # Extract ticker
        ticker = extract_entity_from_question(question)
        
        if not ticker:
            return NarratedReport(
                query=question,
                answer=(
                    "⚠️ **Signal Detection**: Could not extract company from query.\n\n"
                    "Please specify a ticker, e.g.:\n"
                    "- 'Any red flags in NVDA's latest filing?'\n"
                    "- 'Risk factors changes for Apple'\n"
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        
        try:
            # Fetch filings based on mode (P3)
            mode_str = mode.value if isinstance(mode, SignalMode) else str(mode)
            logger.info(f"Signal detection for {ticker} in {mode_str.upper()} mode...")
            
            base_filing, compare_filing = get_signal_filings(ticker, mode_str)
            filings = [compare_filing, base_filing]  # Order: newest first
            
            # Determine filing types based on mode
            if mode == SignalMode.REGIME:
                base_type, compare_type = "10-K", "10-K"
            elif mode == SignalMode.EVENT:
                base_type, compare_type = "10-K", "10-Q"
            else:  # QUARTERLY
                base_type, compare_type = "10-Q", "10-Q"
            
            # Extract Risk Factors from each filing with metadata
            # Now uses section_locator for deterministic, layered extraction
            period_texts: Dict[str, str] = {}
            citations: List[str] = []
            filing_metadata: List[Dict] = []
            extraction_results: List[Optional[RiskFactorsExtract]] = []
            
            for i, snapshot in enumerate([compare_filing, base_filing]):
                # Use new section_locator (Phase 2) - deterministic extraction
                rf_extract = extract_risk_factors_from_html(snapshot.raw_html)
                extraction_results.append(rf_extract)
                
                filing_type = compare_type if i == 0 else base_type
                
                if rf_extract and rf_extract.text:
                    # Get fiscal period metadata (P0)
                    metadata = get_filing_metadata(snapshot)
                    filing_metadata.append(metadata)
                    
                    # Use fiscal period as label (human-readable)
                    period_label = metadata.get("fiscal_period") or snapshot.doc_date or f"Filing_{i+1}"
                    period_texts[period_label] = rf_extract.text
                    
                    # Build citation pointer with extraction method
                    extraction_note = f"[{rf_extract.method}/{rf_extract.confidence}]"
                    citation = f"SEC {filing_type} | CIK {snapshot.cik} | {period_label} | Item 1A {extraction_note}"
                    citations.append(citation)
                    
                    logger.info(
                        f"Extracted {rf_extract.char_count} chars of Risk Factors from {period_label} "
                        f"(method={rf_extract.method}, confidence={rf_extract.confidence})"
                    )
                else:
                    logger.warning(f"No Item 1A found in filing {i+1} for {ticker}")
            
            if len(period_texts) < 2:
                return NarratedReport(
                    query=question,
                    answer=(
                        f"⚠️ **Extraction Issue**: Could not extract Risk Factors (Item 1A) from 2 filings.\n\n"
                        f"Extracted from {len(period_texts)} filing(s). Some filings may not have Item 1A "
                        "(e.g., 8-Ks or certain 10-Qs for smaller filers).\n\n"
                        "Try using 10-K filings instead: `orc.analyze_risk_signal(ticker, period_texts)`"
                    ),
                    citations=citations,
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,
                )
            
            # Run drift analysis
            logger.info(f"Running drift analysis for {ticker} across {len(period_texts)} periods...")
            alert = analyze_risk_drift(ticker, period_texts)
            
            # Detect boilerplate in the compare (most recent) filing (P2)
            compare_text = list(period_texts.values())[0]  # Most recent
            boilerplate_result = detect_boilerplate(compare_text)
            
            if boilerplate_result.is_boilerplate_heavy:
                logger.info(f"Boilerplate detected: ratio={boilerplate_result.boilerplate_ratio:.2f}")
            
            # P3: EVENT mode suppression - if boilerplate-heavy, return early
            if mode == SignalMode.EVENT and boilerplate_result.is_boilerplate_heavy:
                return NarratedReport(
                    query=question,
                    answer=(
                        f"📊 **No Material Signal for {ticker}** (EVENT Mode)\n\n"
                        f"The latest 10-Q Risk Factors section is **{boilerplate_result.boilerplate_ratio:.0%} boilerplate** "
                        "relative to the 10-K baseline.\n\n"
                        "**Common patterns detected:**\n"
                        "- 'No material changes'\n"
                        "- 'As previously disclosed'\n"
                        "- 'Incorporated by reference'\n\n"
                        "**What this means:** No novel risk language was detected in this 10-Q. "
                        "This is normal for most quarterly filings.\n\n"
                        "_For substantive risk changes, run REGIME mode to compare annual 10-Ks._"
                    ),
                    citations=citations,
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,  # No signal, not a failure
                )
            
            # Persist signal record (P1)
            import uuid
            import json
            
            if alert.drift_results:
                dr = alert.drift_results[0]
                
                # Get metadata for base and compare filings
                base_meta = filing_metadata[1] if len(filing_metadata) > 1 else {}
                compare_meta = filing_metadata[0] if len(filing_metadata) > 0 else {}
                
                signal_record = SignalRecord(
                    signal_id=str(uuid.uuid4()),
                    ticker=ticker.upper(),
                    cik=filings[0].cik if filings else "",
                    filing_type=compare_type,  # Use the actual compare filing type
                    base_accession=base_meta.get("accession", ""),
                    compare_accession=compare_meta.get("accession", ""),
                    # P3: New mode fields
                    signal_mode=mode_str,
                    base_filing_type=base_type,
                    compare_filing_type=compare_type,
                    # Dates
                    base_period_end_date=base_meta.get("period_end_date", ""),
                    compare_period_end_date=compare_meta.get("period_end_date", ""),
                    filing_date=compare_meta.get("filing_date", ""),
                    base_fiscal_period=base_meta.get("fiscal_period", ""),
                    compare_fiscal_period=compare_meta.get("fiscal_period", ""),
                    drift_score=dr.drift_score,
                    jaccard_similarity=dr.similarity,
                    semantic_similarity=dr.semantic_similarity,
                    new_sentence_count=len(dr.added_sentences),
                    removed_sentence_count=len(dr.removed_sentences),
                    new_keyword_count=len(dr.new_risk_keywords),
                    removed_keyword_count=len(dr.removed_risk_keywords),
                    new_keywords_json=json.dumps(dr.new_risk_keywords),
                    removed_keywords_json=json.dumps(dr.removed_risk_keywords),
                    boilerplate_flag=boilerplate_result.is_boilerplate_heavy,
                    boilerplate_ratio=boilerplate_result.boilerplate_ratio,
                    severity=dr.severity,
                    created_at=datetime.now().isoformat(),
                    model_version="1.0.0",
                    base_snapshot_id=filings[1].snapshot_id if len(filings) > 1 else "",
                    compare_snapshot_id=filings[0].snapshot_id if filings else "",
                )
                
                # Persist to JSONL
                signal_store = SignalStore()
                signal_store.append(signal_record)
                logger.info(f"Signal persisted: {signal_record.signal_id} (mode={mode_str})")
            
            # Format the signal report
            formatted_report = format_signal_report(alert)
            
            # Add mode header (P3)
            mode_descriptions = {
                "regime": "10-K → 10-K (Annual Risk Regime)",
                "event": "10-Q → 10-K (Event Detection)",
                "quarterly": "10-Q → 10-Q (Quarterly Comparison)",
            }
            mode_header = f"**Mode**: {mode_descriptions.get(mode_str, mode_str)}\n\n"
            
            # Add extraction confidence note if any extraction was LOW confidence
            extraction_warning = ""
            low_confidence_extracts = [
                r for r in extraction_results 
                if r and r.confidence == "LOW"
            ]
            if low_confidence_extracts:
                extraction_warning = (
                    "\n\n⚠️ **Low Confidence Extraction**: One or more Risk Factors sections were extracted "
                    "using fuzzy fallback (low confidence). Results may include extraneous content.\n"
                )
            
            # Add boilerplate warning if applicable
            boilerplate_warning = ""
            if boilerplate_result.is_boilerplate_heavy:
                boilerplate_warning = (
                    f"\n\n⚠️ **High Boilerplate Ratio**: {boilerplate_result.boilerplate_ratio:.0%} of sentences "
                    "matched boilerplate patterns (e.g., 'no material changes'). "
                    "Signal may be noise.\n"
                )
            
            # Add SIGNAL disclaimer (this is soft context, not a verified fact)
            disclaimer = (
                "\n\n---\n"
                "⚠️ **SIGNAL ARTIFACT** (Tier 3 Soft Context)\n"
                "This analysis surfaces textual changes in Risk Factors. "
                "It does not constitute verified facts and should be corroborated "
                "with financial data before making investment decisions."
            )
            
            return NarratedReport(
                query=question,
                answer=mode_header + formatted_report + extraction_warning + boilerplate_warning + disclaimer,
                citations=citations,
                facts_used=[],  # Signals are not facts
                generated_at=datetime.now(),
                insufficient_data=False,  # We have actual analysis
            )
            
        except FileNotFoundError as e:
            logger.error(f"Filing not found for {ticker}: {e}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Filing Not Found**: Could not retrieve SEC filings for {ticker}.\n\n"
                    f"Error: {e}\n\n"
                    "Ensure the ticker is valid and SEC_USER_AGENT is configured."
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        except ValueError as e:
            logger.error(f"Entity resolution failed for {ticker}: {e}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Entity Resolution Failed**: {e}\n\n"
                    "Could not resolve the ticker to a CIK. Ensure it's a valid SEC filer."
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in signal detection for {ticker}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Error**: An unexpected error occurred during signal detection.\n\n"
                    f"Error: {type(e).__name__}: {e}\n\n"
                    "Please try again or use the direct method: `orc.analyze_risk_signal(ticker, period_texts)`"
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
    
    def analyze_regime_and_events(
        self,
        ticker: str,
    ) -> Dict[str, Optional[NarratedReport]]:
        """
        Run both regime (10-K) and event (10-Q) analysis for a ticker.
        
        This is the recommended entry point for comprehensive signal analysis.
        It runs:
        1. REGIME mode: 10-K → 10-K (annual baseline, substantive changes)
        2. EVENT mode: 10-Q → 10-K (fast overlay, novel risk detection)
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            
        Returns:
            Dict with keys:
            - "regime": NarratedReport from 10-K comparison (or None if failed)
            - "event": NarratedReport from 10-Q overlay (or None if failed/suppressed)
            
        Example:
            results = orc.analyze_regime_and_events("NVDA")
            print(results["regime"].answer)  # Annual risk regime
            print(results["event"].answer)   # Quarterly event detection
        """
        results: Dict[str, Optional[NarratedReport]] = {
            "regime": None,
            "event": None,
        }
        
        # Run REGIME mode (10-K → 10-K)
        try:
            question = f"Risk factor changes for {ticker}"
            results["regime"] = self._handle_signal_detection(
                question=question,
                mode=SignalMode.REGIME,
            )
            logger.info(f"REGIME analysis complete for {ticker}")
        except Exception as e:
            logger.error(f"REGIME analysis failed for {ticker}: {e}")
            results["regime"] = None
        
        # Run EVENT mode (10-Q → 10-K)
        try:
            results["event"] = self._handle_signal_detection(
                question=question,
                mode=SignalMode.EVENT,
            )
            logger.info(f"EVENT analysis complete for {ticker}")
        except Exception as e:
            logger.error(f"EVENT analysis failed for {ticker}: {e}")
            results["event"] = None
        
        return results
    
    def _handle_verification(self, question: str) -> NarratedReport:
        """Handle verification requests - NOT IMPLEMENTED via ask()."""
        from datetime import datetime
        return NarratedReport(
            query=question,
            answer=(
                "⚠️ **NOT IMPLEMENTED**: Verification is not available via `ask()`.\n\n"
                "This handler is a placeholder. The functionality exists but requires direct method calls:\n"
                "```python\n"
                "result = orc.verify_news(news_text, ticker='NVDA')\n"
                "```\n\n"
                "_System returned insufficient_data=True per Invariant I4 (Fail Closed)._"
            ),
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    def _handle_comparison(self, question: str, auto_load: bool) -> NarratedReport:
        """Handle multi-entity comparisons."""
        # Extract tickers from the comparison query
        tickers = extract_comparison_entities(question)
        
        if len(tickers) < 2:
            return self._placeholder_report(
                question,
                "Comparison requires at least two tickers. "
                "Example: 'Compare NVDA vs AMD revenue in Q3 FY2025'"
            )
        
        # Extract period
        period = extract_period_from_question(question)
        
        # Load facts for each ticker
        if auto_load and period:
            for ticker in tickers:
                self.load_facts_for_entity(ticker, period)
        
        # Generate comparison report
        metric = extract_metric_from_question(question)
        report = generate_report(
            query=question,
            fact_store=self.fact_store,
            entity=None,  # Multiple entities
            metric=metric,
            period=period,
        )
        
        return report
    
    def _handle_qualitative(self, question: str) -> NarratedReport:
        """Handle qualitative extraction from SEC filings with Gate A verification.
        
        This handler implements the full anti-hallucination pipeline for qualitative data:
        1. Get filing (10-K or 10-Q based on period)
        2. Extract section (Item 1A Risk Factors for now)
        3. Extract facts via LLM
        4. Gate A verification (only verified facts pass)
        5. Store verified facts in FactStore
        6. Narrate answer from verified facts only
        
        Qualitative queries include:
        - Risk factor questions ("What are the main risks?")
        - Management commentary ("What did the CEO say about guidance?")
        - Strategic discussion ("Strategic initiatives mentioned?")
        """
        from datetime import datetime
        import uuid
        import hashlib
        
        # Import dependencies here to avoid circular imports
        from open_deep_research.pipeline import process_extracted_facts
        from open_deep_research.extraction import extract_facts_from_text
        from open_deep_research.models import DocumentSnapshot
        
        # Step 1: Extract entity (ticker) from question
        ticker = extract_entity_from_question(question)
        
        if not ticker:
            return NarratedReport(
                query=question,
                answer=(
                    "⚠️ **Entity Not Found**: Could not extract company from query.\n\n"
                    "Please specify a ticker, e.g.:\n"
                    "- 'What are the main risks for NVDA?'\n"
                    "- 'Management commentary on Apple guidance'\n"
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        
        # Step 2: Extract period if mentioned (default to most recent)
        period = extract_period_from_question(question)
        
        try:
            # Step 3: Get the most recent filing
            # Use 10-K for annual/comprehensive queries, 10-Q for quarterly
            filing_type = "10-K"  # Default to 10-K for more comprehensive content
            if period and "Q" in period.upper():
                filing_type = "10-Q"
            
            logger.info(f"Fetching {filing_type} filing for {ticker}...")
            filings = get_recent_filings(ticker, filing_type=filing_type, count=1)
            
            if not filings:
                return NarratedReport(
                    query=question,
                    answer=(
                        f"⚠️ **Filing Not Found**: Could not retrieve {filing_type} for {ticker}.\n\n"
                        "Ensure SEC_USER_AGENT is configured and the ticker is valid."
                    ),
                    citations=[],
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,
                )
            
            filing = filings[0]
            logger.info(f"Retrieved filing: {filing.doc_type} dated {filing.doc_date}")
            
            # Step 4: Extract section using section_locator
            # Currently only supports Item 1A (Risk Factors)
            rf_extract = extract_risk_factors_from_html(filing.raw_html)
            
            if not rf_extract or not rf_extract.text:
                return NarratedReport(
                    query=question,
                    answer=(
                        f"⚠️ **Section Not Found**: Could not extract Risk Factors (Item 1A) from {ticker}'s {filing_type}.\n\n"
                        "The filing may not contain Item 1A or the extraction failed.\n"
                        "Try specifying a 10-K filing for more comprehensive content."
                    ),
                    citations=[],
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,
                )
            
            logger.info(
                f"Extracted {rf_extract.char_count} chars from Item 1A "
                f"(method={rf_extract.method}, confidence={rf_extract.confidence})"
            )
            
            # Step 5: Create DocumentSnapshot for extraction
            doc_snapshot = DocumentSnapshot(
                snapshot_id=filing.snapshot_id,
                url=filing.url,
                cik=filing.cik,
                doc_type=filing.doc_type,
                doc_date=filing.doc_date,
                retrieved_at=filing.retrieved_at,
                content_hash=filing.content_hash,
                raw_html=filing.raw_html,
            )
            
            # Step 6: Split extracted text into paragraphs for fact extraction
            # Simple paragraph splitting on double newlines
            raw_paragraphs = [p.strip() for p in rf_extract.text.split("\n\n") if p.strip()]
            
            # Filter to substantive paragraphs (> 100 chars)
            substantive_paragraphs = [p for p in raw_paragraphs if len(p) > 100]
            
            if not substantive_paragraphs:
                return NarratedReport(
                    query=question,
                    answer=(
                        f"⚠️ **No Substantive Content**: Extracted Risk Factors for {ticker} "
                        "but found no substantive paragraphs after filtering.\n\n"
                        "This may indicate a parsing issue or boilerplate-only content."
                    ),
                    citations=[],
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,
                )
            
            # Limit to first N paragraphs to avoid excessive LLM calls
            max_paragraphs = 10
            paragraphs_to_process = substantive_paragraphs[:max_paragraphs]
            
            logger.info(f"Processing {len(paragraphs_to_process)} paragraphs for fact extraction...")
            
            # Step 7: Extract facts from each paragraph via LLM
            all_extracted_facts = []
            
            for i, para_text in enumerate(paragraphs_to_process):
                try:
                    facts = extract_facts_from_text(
                        text=para_text,
                        entity=ticker,
                        doc_snapshot=doc_snapshot,
                        section_id="Item1A",
                        paragraph_index=i
                    )
                    all_extracted_facts.extend(facts)
                    logger.debug(f"Paragraph {i}: extracted {len(facts)} facts")
                except Exception as e:
                    logger.warning(f"Fact extraction failed for paragraph {i}: {e}")
                    continue
            
            logger.info(f"Total extracted facts (before verification): {len(all_extracted_facts)}")
            
            # Step 8: Gate A verification - only verified facts pass
            # Note: For text facts, this verifies sentence_string exists in source
            source_text = "\n\n".join(paragraphs_to_process)
            verified_facts, rejected_facts = process_extracted_facts(
                facts=all_extracted_facts,
                source_text=source_text,
                tables=[]  # No tables for qualitative extraction
            )
            
            logger.info(
                f"Gate A results: {len(verified_facts)} verified, {len(rejected_facts)} rejected"
            )
            
            # Step 9: Store only verified facts in FactStore
            for fact in verified_facts:
                try:
                    self.fact_store.add_fact(fact)
                    logger.debug(f"Stored fact: {fact.metric} = {fact.value}")
                except ValueError as e:
                    logger.warning(f"Could not store fact: {e}")
            
            # Step 10: Generate report from verified facts
            if not verified_facts:
                # No verified facts extracted, but we have content
                # Create a qualitative summary without facts
                return NarratedReport(
                    query=question,
                    answer=(
                        f"📋 **Qualitative Analysis for {ticker}** ({filing.doc_type})\n\n"
                        f"Extracted {rf_extract.char_count} characters of Risk Factors (Item 1A), "
                        f"but no verifiable financial facts were found.\n\n"
                        f"**Extraction Details:**\n"
                        f"- Method: {rf_extract.method} ({rf_extract.confidence} confidence)\n"
                        f"- Paragraphs processed: {len(paragraphs_to_process)}\n"
                        f"- Source: SEC {filing.doc_type} | CIK {filing.cik}\n\n"
                        f"**Note:** Qualitative content like risk disclosures often contains "
                        "narrative text rather than specific financial metrics. For quantitative "
                        "data, try asking about specific metrics like revenue, EPS, or margins."
                    ),
                    citations=[Citation(
                        fact_id="",
                        citation_index=1,
                        source_format="html_text",
                        location=f"SEC {filing.doc_type} Item 1A for {ticker}",
                    )],
                    facts_used=[],
                    generated_at=datetime.now(),
                    insufficient_data=True,
                )
            
            # Build citations for verified facts
            citations = []
            for i, fact in enumerate(verified_facts, 1):
                citations.append(Citation(
                    fact_id=fact.fact_id,
                    citation_index=i,
                    source_format=fact.source_format,
                    location=(
                        f"{fact.metric}: {fact.value} {fact.unit} ({fact.period}) - "
                        f"SEC {filing.doc_type} Item 1A [{rf_extract.method}]"
                    ),
                ))
            
            # Generate narrative answer
            report = generate_report(
                query=question,
                fact_store=self.fact_store,
                entity=ticker,
                metric=None,  # No specific metric for qualitative
                period=period,
            )
            
            # Enhance answer with extraction metadata
            extraction_note = (
                f"\n\n---\n"
                f"**Extraction Metadata:**\n"
                f"- Source: SEC {filing.doc_type} Item 1A (Risk Factors)\n"
                f"- Extraction method: {rf_extract.method} ({rf_extract.confidence} confidence)\n"
                f"- Facts verified: {len(verified_facts)} / {len(all_extracted_facts)}\n"
                f"- Gate A rejection rate: {len(rejected_facts)}/{len(all_extracted_facts)}"
            )
            
            return NarratedReport(
                query=question,
                answer=report.answer + extraction_note,
                citations=citations,
                facts_used=verified_facts,
                generated_at=datetime.now(),
                insufficient_data=report.insufficient_data,
            )
            
        except FileNotFoundError as e:
            logger.error(f"Filing not found for {ticker}: {e}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Filing Not Found**: Could not retrieve SEC filings for {ticker}.\n\n"
                    f"Error: {e}\n\n"
                    "Ensure the ticker is valid and SEC_USER_AGENT is configured."
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        except ValueError as e:
            logger.error(f"Entity resolution failed for {ticker}: {e}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Entity Resolution Failed**: {e}\n\n"
                    "Could not resolve the ticker to a CIK. Ensure it's a valid SEC filer."
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in qualitative extraction for {ticker}")
            return NarratedReport(
                query=question,
                answer=(
                    f"⚠️ **Error**: An unexpected error occurred during qualitative extraction.\n\n"
                    f"Error: {type(e).__name__}: {e}\n\n"
                    "Please try again or use signal detection for risk analysis."
                ),
                citations=[],
                facts_used=[],
                generated_at=datetime.now(),
                insufficient_data=True,
            )
    
    def _handle_exploration(self, question: str) -> NarratedReport:
        """Handle open-ended exploration - NOT IMPLEMENTED via ask()."""
        from datetime import datetime
        return NarratedReport(
            query=question,
            answer=(
                "⚠️ **NOT IMPLEMENTED**: Deep research exploration is not available via `ask()`.\n\n"
                "This handler is a placeholder. The LangGraph deep research integration exists but "
                "has not been wired to the `ask()` entry point yet.\n\n"
                "_System returned insufficient_data=True per Invariant I4 (Fail Closed)._"
            ),
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    def _handle_degraded_mode(
        self,
        question: str,
        result: ClassificationResult,
        auto_load: bool,  # Ignored in degraded mode - no auto-loading allowed
    ) -> NarratedReport:
        """
        Handle queries when system is in degraded mode (embeddings unavailable).
        
        INVARIANTS (per $15B AUM requirements):
        - NEVER routes to XBRL (Tier 1) - can't verify intent
        - NEVER calls load_facts_for_entity() - that uses XBRL
        - ALWAYS fails closed with explicit limitation
        - Collects regex hints for logging only (NOT for classification)
        
        This handler is INTENTIONALLY restrictive. When semantic classification
        is unavailable, we cannot trust intent routing. The correct behavior
        is to refuse service and ask the user to wait or rephrase.
        
        Args:
            question: The user's question
            result: The ClassificationResult with regex hints
            auto_load: IGNORED - no auto-loading in degraded mode
            
        Returns:
            NarratedReport with insufficient_data=True
        """
        from datetime import datetime
        
        hints = result.scores.get("regex_hints", [])
        logger.warning(
            f"[DEGRADED MODE] Semantic routing offline. "
            f"Query: '{question[:100]}...' Hints: {hints}"
        )
        
        # FAIL CLOSED: Do not attempt to process the query
        # No XBRL, no fact loading, no report generation
        return NarratedReport(
            query=question,
            answer=(
                "⚠️ **DEGRADED MODE**: The classification system is operating without "
                "semantic embeddings.\n\n"
                "**What this means:**\n"
                "- The system cannot reliably determine the intent of your query\n"
                "- XBRL/financial data access is disabled to prevent incorrect routing\n"
                "- This is a safety measure per Invariant I4 (Fail Closed)\n\n"
                "**What you can do:**\n"
                "1. Wait for the embedding service to recover\n"
                "2. Use direct API methods if you know exactly what you need:\n"
                "   - `orc.load_facts_for_entity('NVDA', 'Q3 2024')` for XBRL data\n"
                "   - `orc.verify_news(text, ticker='NVDA')` for news verification\n"
                "   - `orc.discover('query', ticker='NVDA')` for news discovery\n\n"
                f"_Regex hints detected (for diagnostics only): {hints}_"
            ),
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    def _handle_unknown(self, question: str) -> NarratedReport:
        """
        Handle queries that couldn't be classified.
        
        This is the SAFE default - ask for clarification rather than guess.
        """
        from datetime import datetime
        
        ticker = extract_entity_from_question(question)
        
        return NarratedReport(
            query=question,
            answer=(
                "I couldn't determine the intent of your query. "
                "To help you better, please clarify:\n\n"
                "- **Financial lookup**: 'What was [COMPANY]'s revenue in Q3 FY2025?'\n"
                "- **Verification**: 'Is it true that [COMPANY] reported $X revenue?'\n"
                "- **Signal detection**: 'Any red flags in [COMPANY]'s latest 10-K?'\n"
                "- **Comparison**: 'Compare [COMPANY1] vs [COMPANY2] revenue'\n"
                "- **Discovery**: 'Find news about [COMPANY]'\n\n"
                f"Detected entity: {ticker or 'None'}\n\n"
                "_Please rephrase your question with more specificity._"
            ),
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    def _placeholder_report(self, question: str, message: str) -> NarratedReport:
        """Create a placeholder report for unimplemented features."""
        from datetime import datetime
        return NarratedReport(
            query=question,
            answer=message,
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of all metrics that can be loaded from XBRL.
        
        Returns:
            List of metric names
        """
        return sorted(METRIC_TO_CONCEPTS.keys())
    
    def get_loaded_facts(self) -> List[Fact]:
        """
        Get all facts currently in the store.
        
        Returns:
            List of Fact objects
        """
        return self.fact_store.get_all_facts()
    
    def get_facts_for_entity(self, ticker: str) -> List[Fact]:
        """
        Get all facts for a specific entity.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List of facts for that entity
        """
        return self.fact_store.get_facts_by_entity(ticker)
    
    def verify_news(
        self, 
        news_text: str, 
        source: str = "news article",
        ticker: Optional[str] = None
    ) -> str:
        """
        Verify claims in a news article against SEC data.
        
        This is the key differentiator: soft source claims are verified
        against hard sources (XBRL). If the news says "$35B revenue" but
        SEC data shows $35.08B, we can flag this as "close" (likely rounding).
        If the news says "$30B" but SEC shows $35B, we flag as "contradicted".
        
        Args:
            news_text: The full text of the news article
            source: Description of the source (e.g., "Reuters")
            ticker: Optional ticker to filter facts (e.g., "NVDA")
            
        Returns:
            Formatted verification report as string
            
        Example:
            >>> orc = Orchestrator()
            >>> orc.load_facts_for_entity("NVDA", "Q3 FY2025")
            >>> report = orc.verify_news(
            ...     "NVIDIA reported $35 billion in revenue.",
            ...     source="Reuters",
            ...     ticker="NVDA"
            ... )
            >>> print(report)
        """
        # Extract claims from the text
        claims = extract_claims_from_text(news_text, source, source_tier="soft")
        
        # Add ticker hint to claims if provided
        if ticker:
            for claim in claims:
                claim.entity = ticker
        
        if not claims:
            return "No verifiable financial claims found in the text."
        
        # Verify each claim against our fact store
        results = verify_all_claims(claims, self.fact_store)
        
        # Format and return report
        return format_verification_report(results)
    
    def verify_news_detailed(
        self, 
        news_text: str, 
        source: str = "news article",
        ticker: Optional[str] = None
    ) -> List[VerificationResult]:
        """
        Verify claims in a news article and return structured results.
        
        Same as verify_news() but returns VerificationResult objects
        instead of a formatted string, for programmatic access.
        
        Args:
            news_text: The full text of the news article
            source: Description of the source
            ticker: Optional ticker to filter facts
            
        Returns:
            List of VerificationResult objects
        """
        claims = extract_claims_from_text(news_text, source, source_tier="soft")
        
        if ticker:
            for claim in claims:
                claim.entity = ticker
        
        return verify_all_claims(claims, self.fact_store)

    def analyze_risk_signal(
        self,
        ticker: str,
        period_texts: Dict[str, str],
    ) -> str:
        """Run drift analysis on Risk Factors text and return formatted report.
        
        This is the alpha signal feature - detects semantic changes in SEC Risk
        Factors (Item 1A) across quarters. Substantial rewrites may indicate
        management is pricing in material changes not yet reflected in
        structured financial data.
        
        Args:
            ticker: Company ticker (e.g., "NVDA")
            period_texts: Dict mapping period labels to Risk Factors text
                         e.g., {"Q1 FY2025": "risk text...", "Q2 FY2025": "..."}
                         
        Returns:
            Formatted markdown report showing drift scores and visual diff
            of what text changed between periods.
            
        Example:
            >>> orc = Orchestrator()
            >>> report = orc.analyze_risk_signal("NVDA", {
            ...     "Q1 FY2025": q1_risk_text,
            ...     "Q2 FY2025": q2_risk_text,
            ... })
            >>> print(report)
        """
        alert = analyze_risk_drift(ticker, period_texts)
        return format_signal_report(alert)
    
    # =========================================================================
    # Discovery Methods
    # =========================================================================
    
    def discover(
        self, 
        query: str, 
        ticker: Optional[str] = None,
        auto_verify: bool = True,
    ) -> DiscoveryReport:
        """
        Discover leads from web/news and optionally verify.
        
        This searches for news/web mentions and extracts claims,
        then optionally verifies them against SEC data in the FactStore.
        
        Args:
            query: What to search for (e.g., "NVDA news")
            ticker: Filter to specific company
            auto_verify: If True, verify leads against FactStore
            
        Returns:
            DiscoveryReport with leads and verification status
            
        Example:
            >>> orc = Orchestrator()
            >>> orc.load_facts_for_entity("NVDA", "Q3 FY2025")
            >>> report = orc.discover("NVDA news", ticker="NVDA")
            >>> print(report.format_report())
        """
        if auto_verify:
            return discover_and_verify(
                query=query,
                fact_store=self.fact_store,
                config=self.config,
                ticker=ticker,
            )
        else:
            from datetime import datetime
            leads = discover(query, config=self.config, ticker=ticker)
            return DiscoveryReport(
                query=query,
                ticker=ticker,
                leads=leads,
                generated_at=datetime.now(),
            )
    
    def discover_from_article(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        ticker: Optional[str] = None,
        auto_verify: bool = True,
    ) -> DiscoveryReport:
        """
        Extract and verify claims from a specific article.
        
        Use this when an analyst has a specific article they want
        to fact-check against SEC data.
        
        Args:
            text: Article text
            source_name: Name of source (e.g., "Reuters")
            source_url: URL of article
            ticker: Company the article is about
            auto_verify: If True, verify against FactStore
            
        Returns:
            DiscoveryReport with leads and verification
            
        Example:
            >>> article = "NVIDIA reported $35 billion in revenue..."
            >>> report = orc.discover_from_article(article, "Reuters", ticker="NVDA")
        """
        from datetime import datetime
        
        leads = discover_from_text(
            text=text,
            source_name=source_name,
            source_url=source_url,
            entity=ticker,
        )
        
        if auto_verify:
            leads = verify_leads(leads, self.fact_store)
        
        return DiscoveryReport(
            query=f"Article from {source_name}",
            ticker=ticker,
            leads=leads,
            generated_at=datetime.now(),
        )
    
    def _handle_discovery(self, question: str) -> NarratedReport:
        """Handle discovery queries."""
        # Extract ticker if present
        ticker = extract_entity_from_question(question)
        
        # Run discovery
        report = self.discover(question, ticker=ticker)
        
        # Convert to NarratedReport format
        from datetime import datetime
        return NarratedReport(
            query=question,
            answer=report.format_report(),
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=report.total_leads == 0,
        )
    
    def clear(self) -> None:
        """Clear all loaded facts and reset the orchestrator."""
        self.fact_store = FactStore()
        self._loaded_periods.clear()
        logger.info("Orchestrator cleared")
    
    def __repr__(self) -> str:
        return f"Orchestrator(facts={len(self.fact_store)}, periods={len(self._loaded_periods)})"


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_answer(question: str) -> NarratedReport:
    """
    Quick one-liner to answer a question.
    
    Creates a temporary orchestrator, auto-loads facts, and returns answer.
    
    Args:
        question: The research question
        
    Returns:
        NarratedReport with answer
        
    Example:
        report = quick_answer("What was NVIDIA's revenue in Q3 FY2025?")
        print(report.answer)
    """
    orc = Orchestrator()
    return orc.ask(question)

