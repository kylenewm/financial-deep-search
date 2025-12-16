"""
Structured Logging for Anti-Hallucination Pipeline.

Provides audit-ready structured logs for:
- Gate decisions (verification pass/fail)
- Extraction events
- Pipeline flow tracking

Uses Python's logging with structured output format.
For production, consider integrating with structlog or OpenTelemetry.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps


# =============================================================================
# Logger Setup
# =============================================================================

logger = logging.getLogger("open_deep_research.observability")


def setup_structured_logging(
    level: int = logging.INFO,
    json_format: bool = False,
) -> None:
    """Configure structured logging for the pipeline.
    
    Args:
        level: Logging level (default INFO)
        json_format: If True, output logs as JSON for machine parsing
    """
    handler = logging.StreamHandler()
    
    if json_format:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    
    # Configure the observability logger
    obs_logger = logging.getLogger("open_deep_research.observability")
    obs_logger.handlers = []
    obs_logger.addHandler(handler)
    obs_logger.setLevel(level)
    obs_logger.propagate = False


# =============================================================================
# Structured Log Events
# =============================================================================

@dataclass
class GateDecision:
    """Structured log for a verification gate decision."""
    gate: str  # "gate_a", "gate_b", etc.
    input_claim: str
    result: str  # "pass", "fail", "skip"
    evidence: str
    fact_id: Optional[str] = None
    entity: Optional[str] = None
    metric: Optional[str] = None
    verification_status: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass  
class ExtractionEvent:
    """Structured log for a fact extraction event."""
    event_type: str  # "start", "complete", "error"
    source_type: str  # "xbrl", "html_text", "html_table"
    entity: str
    section_id: Optional[str] = None
    facts_extracted: int = 0
    facts_verified: int = 0
    facts_rejected: int = 0
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PipelineEvent:
    """Structured log for pipeline flow events."""
    event: str  # "query_received", "routing", "extraction", "verification", "narration"
    query_type: Optional[str] = None
    entity: Optional[str] = None
    status: str = "started"  # "started", "completed", "failed"
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# =============================================================================
# Logging Functions
# =============================================================================

def log_gate_decision(
    gate: str,
    input_claim: str,
    result: str,
    evidence: str,
    **kwargs
) -> None:
    """Log a verification gate decision.
    
    Args:
        gate: Gate identifier (e.g., "gate_a", "text_verification")
        input_claim: The claim being verified
        result: "pass", "fail", or "skip"
        evidence: Supporting evidence or reason for decision
        **kwargs: Additional fields (fact_id, entity, metric, etc.)
    """
    decision = GateDecision(
        gate=gate,
        input_claim=input_claim[:200],  # Truncate for log readability
        result=result,
        evidence=evidence[:500],
        **kwargs
    )
    
    log_data = asdict(decision)
    
    if result == "pass":
        logger.info(f"GATE_DECISION | {json.dumps(log_data)}")
    else:
        logger.warning(f"GATE_DECISION | {json.dumps(log_data)}")


def log_extraction_event(
    event_type: str,
    source_type: str,
    entity: str,
    **kwargs
) -> None:
    """Log a fact extraction event.
    
    Args:
        event_type: "start", "complete", or "error"
        source_type: "xbrl", "html_text", "html_table"
        entity: Ticker symbol
        **kwargs: Additional fields
    """
    event = ExtractionEvent(
        event_type=event_type,
        source_type=source_type,
        entity=entity,
        **kwargs
    )
    
    log_data = asdict(event)
    
    if event_type == "error":
        logger.error(f"EXTRACTION | {json.dumps(log_data)}")
    else:
        logger.info(f"EXTRACTION | {json.dumps(log_data)}")


def log_pipeline_event(
    event: str,
    status: str = "started",
    **kwargs
) -> None:
    """Log a pipeline flow event.
    
    Args:
        event: Event name (e.g., "query_received", "verification")
        status: "started", "completed", or "failed"
        **kwargs: Additional fields
    """
    pipeline_event = PipelineEvent(
        event=event,
        status=status,
        **kwargs
    )
    
    log_data = asdict(pipeline_event)
    
    if status == "failed":
        logger.error(f"PIPELINE | {json.dumps(log_data)}")
    else:
        logger.info(f"PIPELINE | {json.dumps(log_data)}")


# =============================================================================
# Context Managers & Decorators
# =============================================================================

@contextmanager
def timed_operation(operation_name: str, **context):
    """Context manager for timing operations with structured logging.
    
    Usage:
        with timed_operation("extraction", entity="NVDA") as timer:
            # do work
            timer["facts_count"] = 10
    
    Args:
        operation_name: Name of the operation
        **context: Additional context fields
    """
    start_time = time.perf_counter()
    result_data: Dict[str, Any] = {}
    
    log_pipeline_event(operation_name, status="started", details=context)
    
    try:
        yield result_data
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_pipeline_event(
            operation_name, 
            status="completed", 
            duration_ms=round(duration_ms, 2),
            details={**context, **result_data}
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_pipeline_event(
            operation_name, 
            status="failed", 
            duration_ms=round(duration_ms, 2),
            details={**context, "error": str(e)}
        )
        raise


def log_gate(gate_name: str):
    """Decorator for wrapping verification gate functions with logging.
    
    Usage:
        @log_gate("text_verification")
        def verify_text_fact(fact, source_text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Try to extract fact info from args
                fact = args[0] if args else None
                if hasattr(fact, 'metric') and hasattr(fact, 'verification_status'):
                    log_gate_decision(
                        gate=gate_name,
                        input_claim=f"{fact.metric}: {fact.value}",
                        result="pass" if fact.verification_status in ("exact_match", "approximate_match") else "fail",
                        evidence=f"Verified in {duration_ms:.1f}ms",
                        fact_id=getattr(fact, 'fact_id', None),
                        entity=getattr(fact, 'entity', None),
                        metric=getattr(fact, 'metric', None),
                        verification_status=getattr(fact, 'verification_status', None),
                    )
                
                return result
            except Exception as e:
                log_gate_decision(
                    gate=gate_name,
                    input_claim=str(args[0])[:100] if args else "unknown",
                    result="fail",
                    evidence=f"Error: {str(e)}",
                )
                raise
        
        return wrapper
    return decorator


# =============================================================================
# Audit Summary
# =============================================================================

class AuditSummary:
    """Collects metrics during a pipeline run for final summary."""
    
    def __init__(self):
        self.start_time = time.perf_counter()
        self.facts_extracted = 0
        self.facts_verified = 0
        self.facts_rejected = 0
        self.gates_passed = 0
        self.gates_failed = 0
        self.errors: List[str] = []
    
    def record_extraction(self, extracted: int, verified: int, rejected: int):
        """Record extraction results."""
        self.facts_extracted += extracted
        self.facts_verified += verified
        self.facts_rejected += rejected
    
    def record_gate(self, passed: bool):
        """Record a gate decision."""
        if passed:
            self.gates_passed += 1
        else:
            self.gates_failed += 1
    
    def record_error(self, error: str):
        """Record an error."""
        self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return summary as dict."""
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        return {
            "duration_ms": round(duration_ms, 2),
            "facts_extracted": self.facts_extracted,
            "facts_verified": self.facts_verified,
            "facts_rejected": self.facts_rejected,
            "verification_rate": (
                round(self.facts_verified / self.facts_extracted, 3)
                if self.facts_extracted > 0 else 0.0
            ),
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "error_count": len(self.errors),
        }
    
    def log_summary(self):
        """Log the final summary."""
        summary = self.to_dict()
        logger.info(f"AUDIT_SUMMARY | {json.dumps(summary)}")

