"""
Structured Output - Primary output format for research results.

JSON is the primary output format. Institutional users want structured data they can
pull into Excel or Python. Markdown is a secondary human-readable wrapper.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from open_deep_research.models import (
    Analysis,
    Conflict,
    NotFoundMetric,
    ResearchOutput,
)
from open_deep_research.report import generate_full_report
from open_deep_research.store import FactStore

if TYPE_CHECKING:
    pass


def generate_research_output(
    store: FactStore,
    query: str,
    analysis: Analysis | None = None,
    conflicts: list[Conflict] | None = None,
    not_found: list[str] | None = None,
    as_of_date: str | None = None,
) -> ResearchOutput:
    """Create complete ResearchOutput object from a FactStore.
    
    Args:
        store: FactStore containing verified facts
        query: The original research query
        analysis: Optional LLM-generated analysis
        conflicts: Optional list of detected conflicts
        not_found: Optional list of metric names that weren't found
        as_of_date: Optional date string for time-machine mode
    
    Returns:
        Complete ResearchOutput object ready for serialization
    """
    facts = store.get_all_facts()
    
    # Convert not_found strings to NotFoundMetric objects
    not_found_metrics = []
    if not_found:
        not_found_metrics = [NotFoundMetric(metric=m) for m in not_found]
    
    return ResearchOutput(
        query=query,
        generated_at=datetime.now(),
        as_of_date=as_of_date,
        facts=facts,
        analysis=analysis,
        conflicts=conflicts or [],
        not_found=not_found_metrics,
    )


def output_to_json(output: ResearchOutput, pretty: bool = True) -> str:
    """Serialize ResearchOutput to JSON.
    
    Args:
        output: ResearchOutput object to serialize
        pretty: If True, format with indentation (default). 
                If False, compact single-line output.
    
    Returns:
        JSON string representation
    """
    # Use model_dump for Pydantic v2
    data = output.model_dump()
    
    if pretty:
        return json.dumps(data, indent=2, default=_json_serializer)
    else:
        return json.dumps(data, separators=(",", ":"), default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def output_to_markdown(output: ResearchOutput) -> str:
    """Generate markdown report from structured output.
    
    This wraps the report.py functionality for a human-readable format.
    
    Args:
        output: ResearchOutput object to convert
    
    Returns:
        Markdown string representation of the report
    """
    # Reconstruct FactStore from facts
    store = FactStore()
    for fact in output.facts:
        # Bypass verification for reconstruction
        store._facts[fact.fact_id] = fact
    
    # Extract not_found metric names
    not_found_names = [nf.metric for nf in output.not_found] if output.not_found else None
    
    # Generate report (without LLM - use existing analysis if available)
    facts = store.get_all_facts()
    timestamp = output.generated_at.strftime("%Y-%m-%d %H:%M:%S")
    
    # Import section generators from report module
    from open_deep_research.report import (
        generate_citations_section,
        generate_conflicts_section,
        generate_facts_section,
        generate_not_found_section,
    )
    
    facts_section = generate_facts_section(facts)
    conflicts_section = generate_conflicts_section(output.conflicts)
    not_found_section = generate_not_found_section(not_found_names)
    citations_section = generate_citations_section(facts)
    
    # Use existing analysis or create placeholder
    if output.analysis:
        analysis_summary = output.analysis.summary
        supporting_ref = ", ".join(output.analysis.supporting_facts) if output.analysis.supporting_facts else "None"
    else:
        analysis_summary = "No analysis generated."
        supporting_ref = "None"
    
    # Build as_of_date line if present
    as_of_line = f"\n**As Of Date:** {output.as_of_date}" if output.as_of_date else ""
    
    report = f"""# Research Report

**Query:** {output.query}
**Generated:** {timestamp}{as_of_line}
**Facts:** {output.total_facts} total, {output.verified_facts} verified

## Verified Facts

{facts_section}

## Analysis

*Note: The following is interpretation based on the verified facts above. This is not verified factual content.*

{analysis_summary}

**Supporting Evidence:** Facts [{supporting_ref}]

## Data Conflicts

{conflicts_section}

## Not Found

{not_found_section}

## Sources

{citations_section}
"""
    
    return report


def output_to_csv(output: ResearchOutput) -> str:
    """Export facts as CSV for Excel import.
    
    Columns: entity, metric, value, unit, period, period_end_date, source_doc, source_date
    
    Args:
        output: ResearchOutput object to convert
    
    Returns:
        CSV string that can be parsed by csv.reader or imported into Excel
    """
    string_buffer = io.StringIO()
    writer = csv.writer(string_buffer)
    
    # Write header
    writer.writerow([
        "entity",
        "metric", 
        "value",
        "unit",
        "period",
        "period_end_date",
        "source_doc",
        "source_date",
    ])
    
    # Write facts
    for fact in output.facts:
        writer.writerow([
            fact.entity,
            fact.metric,
            fact.value if fact.value is not None else "",
            fact.unit,
            fact.period,
            fact.period_end_date,
            fact.location.doc_type,
            fact.location.doc_date,
        ])
    
    return string_buffer.getvalue()


def output_to_dict(output: ResearchOutput) -> dict:
    """Convert ResearchOutput to plain dict (for API responses).
    
    Args:
        output: ResearchOutput object to convert
    
    Returns:
        Plain dictionary matching ResearchOutput structure
    """
    return output.model_dump()

