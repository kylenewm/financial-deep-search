"""
Fact Store - Single source of truth for verified facts.

The fact store holds verified facts that can be referenced during report generation.
Only facts with verification_status of "exact_match" or "approximate_match" can be added.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from open_deep_research.models import Conflict, ConflictingValue, Fact, SignalRecord
from open_deep_research.numeric_verification import verify_numeric_fact

if TYPE_CHECKING:
    pass


class FactStore:
    """
    In-memory store for verified financial facts.
    
    The store enforces that only verified facts (exact_match or approximate_match)
    can be added. This ensures report generation only uses facts that have been
    verified against source documents.
    """
    
    ALLOWED_STATUSES = {"exact_match", "approximate_match"}
    
    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}  # fact_id -> Fact
    
    def add_fact(self, fact: Fact) -> None:
        """Add a verified fact to the store.
        
        Args:
            fact: A Fact with verification_status of exact_match or approximate_match
        
        Raises:
            ValueError: If fact.verification_status is not exact_match or approximate_match
            ValueError: If fact is missing a durable location pointer
        """
        # HARD GATE 1: Verification status check
        if fact.verification_status not in self.ALLOWED_STATUSES:
            raise ValueError(
                f"Cannot add fact with status '{fact.verification_status}'. "
                "Only verified facts (exact_match, approximate_match) can be added. "
                "Facts must pass Gate A verification before storage."
            )
        
        # HARD GATE 2: Location pointer required (P1 enhancement)
        # Every fact must have a durable pointer back to its source
        if not fact.location:
            raise ValueError(
                "Fact must have a durable location pointer. "
                "Cannot add fact without provenance information."
            )
        
        # Location must have either CIK (for SEC filings) or article_url (for news)
        has_sec_pointer = bool(fact.location.cik)
        has_news_pointer = bool(fact.location.article_url)
        
        if not has_sec_pointer and not has_news_pointer:
            raise ValueError(
                "Fact location must have either a CIK (SEC filing) or article_url (news). "
                "Cannot add fact without traceable source."
            )
        
        self._facts[fact.fact_id] = fact
    
    def get_fact(self, fact_id: str) -> Fact | None:
        """Get a fact by ID.
        
        Args:
            fact_id: The unique identifier of the fact
        
        Returns:
            The Fact if found, None otherwise
        """
        return self._facts.get(fact_id)
    
    def get_facts_by_entity(self, entity: str) -> list[Fact]:
        """Get all facts for a given entity (ticker).
        
        Args:
            entity: Ticker symbol (case-insensitive)
        
        Returns:
            List of facts for the entity
        """
        entity_upper = entity.upper()
        return [f for f in self._facts.values() if f.entity.upper() == entity_upper]
    
    def get_facts_by_metric(self, metric: str) -> list[Fact]:
        """Get all facts for a given metric name (case-insensitive).
        
        Args:
            metric: Metric name (e.g., "Revenue", "Net Income")
        
        Returns:
            List of facts with matching metric name
        """
        metric_lower = metric.lower()
        return [f for f in self._facts.values() if f.metric.lower() == metric_lower]
    
    def get_facts_by_period(self, period: str) -> list[Fact]:
        """Get all facts for a given period.
        
        Args:
            period: Period string (e.g., "Q3 FY2025", "FY2024")
        
        Returns:
            List of facts for the period
        """
        return [f for f in self._facts.values() if f.period == period]
    
    def get_all_facts(self) -> list[Fact]:
        """Get all facts in the store.
        
        Returns:
            List of all facts
        """
        return list(self._facts.values())
    
    def find_conflicts(self) -> list[Conflict]:
        """Find facts where same entity+metric+period has different values.
        
        Two facts conflict if they have the same entity, metric, and period,
        but their values differ by more than 1%.
        
        Returns:
            List of Conflict objects describing conflicting facts
        """
        # Group by (entity, metric, period)
        groups: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
        for fact in self._facts.values():
            key = (fact.entity.upper(), fact.metric.lower(), fact.period)
            groups[key].append(fact)
        
        conflicts = []
        for (entity, metric, period), facts in groups.items():
            if len(facts) < 2:
                continue
            
            # Check if values conflict (differ by more than 1%)
            values = [f.value for f in facts if f.value is not None]
            if len(values) < 2:
                continue
            
            # Compare first value against all others
            base_value = values[0]
            has_conflict = False
            for v in values[1:]:
                if verify_numeric_fact(base_value, v) == "mismatch":
                    has_conflict = True
                    break
            
            if has_conflict:
                conflicting_values = [
                    ConflictingValue(
                        value=f.value,
                        fact_id=f.fact_id,
                        source_description=f"{f.location.doc_type} {f.location.doc_date}"
                    )
                    for f in facts if f.value is not None
                ]
                conflicts.append(Conflict(
                    entity=entity,
                    metric=metric,
                    period=period,
                    values=conflicting_values
                ))
        
        return conflicts
    
    def to_json(self) -> str:
        """Serialize store to JSON.
        
        Returns:
            JSON string representation of all facts
        """
        facts_list = [f.model_dump() for f in self._facts.values()]
        return json.dumps(facts_list, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "FactStore":
        """Deserialize store from JSON.
        
        Args:
            json_str: JSON string from to_json()
        
        Returns:
            FactStore populated with facts from JSON
        
        Note:
            Bypasses verification status check for loading
            (assumes facts were previously validated)
        """
        store = cls()
        facts_list = json.loads(json_str)
        for fact_dict in facts_list:
            fact = Fact(**fact_dict)
            # Bypass validation for loading (assume previously validated)
            store._facts[fact.fact_id] = fact
        return store
    
    def __len__(self) -> int:
        """Return the number of facts in the store."""
        return len(self._facts)
    
    def __repr__(self) -> str:
        return f"FactStore(facts={len(self)})"


# =============================================================================
# Signal Store (JSONL Persistence for Alpha Signals)
# =============================================================================


class SignalStore:
    """
    Append-only signal persistence using JSONL format.
    
    Designed for:
    - Easy append operations (no locking issues)
    - Trivial Pandas loading: pd.read_json(path, lines=True)
    - Simple SQLite migration later
    - Research iteration without schema changes
    
    JSONL format: one JSON object per line, enabling:
    - Streaming reads for large datasets
    - Atomic appends
    - Easy inspection with command-line tools
    """
    
    DEFAULT_PATH = ".cache/signals/signals.jsonl"
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize signal store.
        
        Args:
            path: Path to JSONL file (default: .cache/signals/signals.jsonl)
        """
        self.path = Path(path or self.DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, record: SignalRecord) -> None:
        """
        Append a signal record to the store.
        
        Args:
            record: SignalRecord to persist
        """
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
    
    def load_all(self) -> List[SignalRecord]:
        """
        Load all signal records from the store.
        
        Returns:
            List of all SignalRecords
        """
        if not self.path.exists():
            return []
        
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(SignalRecord.model_validate_json(line))
        return records
    
    def query_by_ticker(self, ticker: str) -> List[SignalRecord]:
        """
        Get all signals for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (case-insensitive)
            
        Returns:
            List of SignalRecords for the ticker
        """
        ticker_upper = ticker.upper()
        return [r for r in self.load_all() if r.ticker.upper() == ticker_upper]
    
    def query_by_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> List[SignalRecord]:
        """
        Get signals within a date range (for backtesting).
        
        Uses filing_date (announcement date) as the filter key,
        not period_end_date, to avoid lookahead bias.
        
        Args:
            start_date: ISO date string (inclusive)
            end_date: ISO date string (inclusive)
            
        Returns:
            List of SignalRecords within the date range
        """
        return [
            r for r in self.load_all()
            if start_date <= r.filing_date <= end_date
        ]
    
    def query_high_drift(self, threshold: float = 30.0) -> List[SignalRecord]:
        """
        Get signals with drift score above threshold.
        
        Args:
            threshold: Minimum drift score (default: 30.0 = moderate)
            
        Returns:
            List of SignalRecords with high drift
        """
        return [r for r in self.load_all() if r.drift_score >= threshold]
    
    def query_non_boilerplate(self) -> List[SignalRecord]:
        """
        Get signals that are NOT boilerplate-heavy.
        
        This filters out 10-Q "no material changes" noise.
        
        Returns:
            List of SignalRecords where boilerplate_flag is False
        """
        return [r for r in self.load_all() if not r.boilerplate_flag]
    
    def to_dataframe(self) -> Any:
        """
        Load all signals as a Pandas DataFrame for analysis.
        
        Requires pandas to be installed.
        
        Returns:
            DataFrame with one row per signal
        """
        import pandas as pd
        
        if not self.path.exists():
            return pd.DataFrame()
        
        return pd.read_json(self.path, lines=True)
    
    def count(self) -> int:
        """Return total number of signals in store."""
        return len(self.load_all())
    
    def clear(self) -> None:
        """Clear all signals (use with caution)."""
        if self.path.exists():
            self.path.unlink()
    
    def __len__(self) -> int:
        return self.count()
    
    def __repr__(self) -> str:
        return f"SignalStore(path={self.path}, count={self.count()})"
    
    # =========================================================================
    # Ranking Views (P4)
    # 
    # IMPORTANT: Rankings are VIEWS, not TRUTH.
    # These methods compute scores at query time for exploration and research.
    # They are NOT validated against returns and should NOT be used as
    # authoritative trading signals without backtesting and validation.
    # =========================================================================
    
    def rank_by_drift(
        self,
        top_n: int = 10,
        exclude_boilerplate: bool = True,
    ) -> List[SignalRecord]:
        """
        Rank signals by drift score (highest first).
        
        ⚠️ THIS IS A VIEW, NOT TRUTH.
        Drift score measures textual change but does NOT distinguish:
        - Rewording vs genuine new content
        - Formatting changes vs substantive changes
        - Boilerplate shuffling vs risk admission
        
        Use this as a starting point for research, not as a trading signal.
        
        Args:
            top_n: Number of top signals to return
            exclude_boilerplate: If True, filters out boilerplate-heavy signals
            
        Returns:
            List of SignalRecords sorted by drift_score descending
        """
        records = self.load_all()
        if exclude_boilerplate:
            records = [r for r in records if not r.boilerplate_flag]
        return sorted(records, key=lambda r: r.drift_score, reverse=True)[:top_n]
    
    def rank_by_novelty(
        self,
        top_n: int = 10,
        exclude_boilerplate: bool = True,
    ) -> List[SignalRecord]:
        """
        Rank by new sentence count (most novel first).
        
        ⚠️ THIS IS A VIEW, NOT TRUTH.
        New sentence count is a proxy for "genuinely new risk admission" but
        does NOT account for:
        - Sentence splitting/merging
        - Paraphrasing of existing content
        - Semantic novelty (only structural)
        
        Use this as a starting point for research, not as a trading signal.
        
        Args:
            top_n: Number of top signals to return
            exclude_boilerplate: If True, filters out boilerplate-heavy signals
            
        Returns:
            List of SignalRecords sorted by new_sentence_count descending
        """
        records = self.load_all()
        if exclude_boilerplate:
            records = [r for r in records if not r.boilerplate_flag]
        return sorted(records, key=lambda r: r.new_sentence_count, reverse=True)[:top_n]
    
    def rank_by_keyword_hits(
        self,
        top_n: int = 10,
        exclude_boilerplate: bool = True,
    ) -> List[SignalRecord]:
        """
        Rank by new risk keyword count (most risk terms first).
        
        ⚠️ THIS IS A VIEW, NOT TRUTH.
        Keyword hits use a predefined domain-specific word list but:
        - May miss novel/emerging risk terminology
        - May overcounts mentions that aren't risks
        - Does not understand context or negation
        
        Use this as a starting point for research, not as a trading signal.
        
        Args:
            top_n: Number of top signals to return
            exclude_boilerplate: If True, filters out boilerplate-heavy signals
            
        Returns:
            List of SignalRecords sorted by new_keyword_count descending
        """
        records = self.load_all()
        if exclude_boilerplate:
            records = [r for r in records if not r.boilerplate_flag]
        return sorted(records, key=lambda r: r.new_keyword_count, reverse=True)[:top_n]
    
    def rank_composite(
        self,
        top_n: int = 10,
        drift_weight: float = 0.5,
        novelty_weight: float = 0.3,
        keyword_weight: float = 0.2,
        exclude_boilerplate: bool = True,
        normalization: str = "global_max",
    ) -> List[Tuple[SignalRecord, float]]:
        """
        Composite ranking with user-defined weights.
        
        ⚠️ THIS IS A VIEW, NOT TRUTH.
        The composite score combines multiple signals with arbitrary weights.
        Default weights are NOT validated against returns and are purely
        heuristic starting points. You MUST backtest before using for trading.
        
        Returns (record, score) tuples for transparency - you can see exactly
        how each signal was scored.
        
        Args:
            top_n: Number of top signals to return
            drift_weight: Weight for drift score (default: 0.5)
            novelty_weight: Weight for new sentence count (default: 0.3)
            keyword_weight: Weight for new keyword count (default: 0.2)
            exclude_boilerplate: If True, filters out boilerplate-heavy signals
            normalization: Strategy for normalizing dimensions:
                - "global_max": Divide by max value across all signals (default)
                - "minmax": (value - min) / (max - min)
                - "none": Use raw values (not recommended, breaks weight semantics)
            
        Returns:
            List of (SignalRecord, composite_score) tuples sorted by score descending
            
        Raises:
            ValueError: If normalization strategy is invalid
        """
        if normalization not in ("global_max", "minmax", "none"):
            raise ValueError(
                f"Invalid normalization strategy: '{normalization}'. "
                "Valid options: 'global_max', 'minmax', 'none'"
            )
        
        records = self.load_all()
        if exclude_boilerplate:
            records = [r for r in records if not r.boilerplate_flag]
        
        if not records:
            return []
        
        # Compute normalization factors based on strategy
        drift_scores = [r.drift_score for r in records]
        novelty_scores = [r.new_sentence_count for r in records]
        keyword_scores = [r.new_keyword_count for r in records]
        
        if normalization == "global_max":
            # Divide by max value (0-1 scale, 0 if all zeros)
            drift_norm = max(drift_scores) if drift_scores else 1
            novelty_norm = max(novelty_scores) if novelty_scores else 1
            keyword_norm = max(keyword_scores) if keyword_scores else 1
            
            def normalize(value: float, norm: float) -> float:
                return value / norm if norm > 0 else 0
                
        elif normalization == "minmax":
            # (value - min) / (max - min)
            drift_min, drift_max = min(drift_scores), max(drift_scores)
            novelty_min, novelty_max = min(novelty_scores), max(novelty_scores)
            keyword_min, keyword_max = min(keyword_scores), max(keyword_scores)
            
            def normalize_minmax(value: float, vmin: float, vmax: float) -> float:
                return (value - vmin) / (vmax - vmin) if vmax > vmin else 0
            
            def normalize(value: float, _: float) -> float:
                # Not used in minmax, handled inline
                return value
                
        else:  # normalization == "none"
            def normalize(value: float, _: float) -> float:
                return value
            drift_norm = novelty_norm = keyword_norm = 1
        
        def compute_score(r: SignalRecord) -> float:
            if normalization == "minmax":
                norm_drift = normalize_minmax(r.drift_score, drift_min, drift_max)
                norm_novelty = normalize_minmax(r.new_sentence_count, novelty_min, novelty_max)
                norm_keywords = normalize_minmax(r.new_keyword_count, keyword_min, keyword_max)
            else:
                norm_drift = normalize(r.drift_score, drift_norm)
                norm_novelty = normalize(r.new_sentence_count, novelty_norm)
                norm_keywords = normalize(r.new_keyword_count, keyword_norm)
            
            return (
                drift_weight * norm_drift +
                novelty_weight * norm_novelty +
                keyword_weight * norm_keywords
            )
        
        scored = [(r, round(compute_score(r), 4)) for r in records]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    
    # =========================================================================
    # Filtering Helpers
    # =========================================================================
    
    def filter_by_mode(
        self,
        mode: str,  # "regime", "event", "quarterly"
    ) -> List[SignalRecord]:
        """
        Get signals from a specific mode only.
        
        Args:
            mode: Signal mode ("regime", "event", "quarterly")
            
        Returns:
            List of SignalRecords matching the mode
        """
        return [r for r in self.load_all() if r.signal_mode == mode.lower()]
    
    def filter_by_severity(
        self,
        min_severity: str = "moderate",
    ) -> List[SignalRecord]:
        """
        Get signals at or above a severity threshold.
        
        Args:
            min_severity: Minimum severity ("low", "moderate", "critical")
            
        Returns:
            List of SignalRecords at or above the threshold
        """
        severity_order = {"low": 0, "moderate": 1, "critical": 2}
        min_val = severity_order.get(min_severity.lower(), 0)
        return [
            r for r in self.load_all()
            if severity_order.get(r.severity, 0) >= min_val
        ]
    
    def filter_by_ticker(
        self,
        tickers: List[str],
    ) -> List[SignalRecord]:
        """
        Get signals for specific tickers.
        
        Args:
            tickers: List of ticker symbols (case-insensitive)
            
        Returns:
            List of SignalRecords for matching tickers
        """
        tickers_upper = {t.upper() for t in tickers}
        return [r for r in self.load_all() if r.ticker.upper() in tickers_upper]
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all signals.
        
        ⚠️ THESE ARE DESCRIPTIVE STATISTICS, NOT TRADING SIGNALS.
        Useful for dashboards and health checks, not for generating alpha.
        
        Returns:
            Dict with counts, distributions, and averages
        """
        records = self.load_all()
        if not records:
            return {"count": 0, "message": "No signals stored"}
        
        return {
            "count": len(records),
            "by_mode": {
                "regime": sum(1 for r in records if r.signal_mode == "regime"),
                "event": sum(1 for r in records if r.signal_mode == "event"),
                "quarterly": sum(1 for r in records if r.signal_mode == "quarterly"),
            },
            "by_severity": {
                "critical": sum(1 for r in records if r.severity == "critical"),
                "moderate": sum(1 for r in records if r.severity == "moderate"),
                "low": sum(1 for r in records if r.severity == "low"),
            },
            "boilerplate_ratio": round(
                sum(1 for r in records if r.boilerplate_flag) / len(records), 3
            ),
            "avg_drift_score": round(
                sum(r.drift_score for r in records) / len(records), 2
            ),
            "avg_new_sentences": round(
                sum(r.new_sentence_count for r in records) / len(records), 2
            ),
            "unique_tickers": len(set(r.ticker for r in records)),
            "date_range": {
                "earliest": min(r.created_at for r in records),
                "latest": max(r.created_at for r in records),
            },
        }

