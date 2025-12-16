"""
Report Generation - Generates research reports from verified facts.

The report generator can ONLY use facts from the fact store. No free-form factual 
generation. Every factual claim must have a corresponding Fact object in the store.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from open_deep_research.models import Analysis, Conflict, Fact, NotFoundMetric
from open_deep_research.store import FactStore

if TYPE_CHECKING:
    pass


def format_value_with_unit(value: float | None, unit: str) -> str:
    """Format a numeric value with its unit.
    
    Args:
        value: The numeric value (may be None)
        unit: The unit string (e.g., "USD millions", "percentage")
    
    Returns:
        Formatted string like "$35,082 million" or "42.5%"
    """
    if value is None:
        return "N/A"
    
    # Handle percentage
    if unit.lower() == "percentage" or unit == "%":
        return f"{value:,.2f}%"
    
    # Handle USD units
    if "usd" in unit.lower() or "$" in unit.lower():
        # Determine scale from unit
        if "billion" in unit.lower():
            return f"${value:,.0f} billion"
        elif "million" in unit.lower():
            return f"${value:,.0f} million"
        elif "thousand" in unit.lower():
            return f"${value:,.0f} thousand"
        else:
            return f"${value:,.2f}"
    
    # Handle ratio
    if unit.lower() == "ratio":
        return f"{value:,.2f}"
    
    # Handle shares
    if "share" in unit.lower():
        return f"${value:,.2f}"
    
    # Default formatting
    return f"{value:,.2f} {unit}"


def generate_facts_section(facts: list[Fact]) -> str:
    """Generate markdown section for verified facts.
    
    Each fact includes inline citation number [1], [2], etc.
    Groups facts by metric type if multiple facts.
    
    Args:
        facts: List of verified Fact objects
    
    Returns:
        Markdown string with formatted facts section
    """
    if not facts:
        return "No verified facts available."
    
    # Group facts by metric
    by_metric: dict[str, list[tuple[int, Fact]]] = defaultdict(list)
    for i, fact in enumerate(facts, start=1):
        by_metric[fact.metric].append((i, fact))
    
    lines = []
    for metric, indexed_facts in by_metric.items():
        lines.append(f"**{metric}**")
        for citation_num, fact in indexed_facts:
            formatted_value = format_value_with_unit(fact.value, fact.unit)
            lines.append(f"- {fact.metric}: {formatted_value} ({fact.period}) [{citation_num}]")
        lines.append("")  # Blank line between groups
    
    return "\n".join(lines).rstrip()


def generate_thesis_section(facts: list[Fact], query: str, llm: ChatOpenAI | None = None) -> Analysis:
    """Use LLM to generate interpretation/analysis based on facts.
    
    The analysis is clearly labeled as thesis/interpretation and references
    supporting fact_ids. The LLM cannot introduce new factual claims.
    
    Args:
        facts: List of verified Fact objects
        query: The original research query
        llm: Optional LLM instance (creates default if not provided)
    
    Returns:
        Analysis object with summary and supporting fact references
    """
    if not facts:
        return Analysis(
            summary="Insufficient data for analysis.",
            classification="thesis",
            supporting_facts=[]
        )
    
    # Build facts context for LLM
    facts_context = []
    for fact in facts:
        formatted_value = format_value_with_unit(fact.value, fact.unit)
        facts_context.append(
            f"- [{fact.fact_id}] {fact.metric}: {formatted_value} ({fact.period})"
        )
    
    prompt = f"""You are generating an investment analysis based on verified financial facts.

IMPORTANT: This analysis is interpretation based on the verified facts below. Do not introduce new factual claims.

Research Query: {query}

Verified Facts:
{chr(10).join(facts_context)}

Generate a 2-3 paragraph analysis that:
1. Interprets the meaning of these facts in the context of the query
2. Identifies trends or patterns if multiple periods are present
3. Discusses investment implications

Your response should be the analysis text only, no headers or preamble."""

    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content if isinstance(response.content, str) else str(response.content)
    
    return Analysis(
        summary=summary.strip(),
        classification="thesis",
        supporting_facts=[f.fact_id for f in facts]
    )


def generate_citations_section(facts: list[Fact]) -> str:
    """Generate citations section with full source details.
    
    Each citation shows: doc_type, doc_date, section_id, exact quote.
    
    Args:
        facts: List of verified Fact objects
    
    Returns:
        Markdown string with formatted citations section
    """
    if not facts:
        return "No sources to cite."
    
    lines = []
    for i, fact in enumerate(facts, start=1):
        loc = fact.location
        
        # Build source description
        source_desc = f"{fact.entity} {loc.doc_type}"
        if loc.doc_date:
            source_desc += f", filed {loc.doc_date}"
        source_desc += f", {loc.section_id}"
        
        # Get the quote - use sentence_string if available, otherwise describe table location
        if loc.sentence_string:
            quote = loc.sentence_string
        elif loc.table_index is not None:
            quote = f"Table {loc.table_index}, Row: {loc.row_label or loc.row_index}, Column: {loc.column_label or loc.column_index}"
        else:
            quote = "Source location recorded"
        
        lines.append(f"[{i}] {source_desc}")
        lines.append(f'    "{quote}"')
        lines.append("")
    
    return "\n".join(lines).rstrip()


def generate_conflicts_section(conflicts: list[Conflict] | None) -> str:
    """Generate section describing data conflicts.
    
    Args:
        conflicts: List of Conflict objects, or None
    
    Returns:
        Markdown string describing conflicts or "No conflicts detected."
    """
    if not conflicts:
        return "No conflicts detected."
    
    lines = []
    for conflict in conflicts:
        lines.append(f"**{conflict.entity} - {conflict.metric} ({conflict.period})**")
        for cv in conflict.values:
            lines.append(f"- Value: {cv.value:,.2f} (from {cv.source_description}, fact: {cv.fact_id})")
        lines.append("")
    
    return "\n".join(lines).rstrip()


def generate_not_found_section(not_found: list[str] | None) -> str:
    """Generate section listing metrics that weren't found.
    
    Args:
        not_found: List of metric names that weren't found, or None
    
    Returns:
        Markdown string listing missing metrics or "All requested metrics were found."
    """
    if not not_found:
        return "All requested metrics were found."
    
    lines = []
    for metric in not_found:
        nf = NotFoundMetric(metric=metric)
        lines.append(f"- {nf.metric}: {nf.status}")
    
    return "\n".join(lines)


def generate_full_report(
    store: FactStore,
    query: str,
    conflicts: list[Conflict] | None = None,
    not_found: list[str] | None = None,
    llm: ChatOpenAI | None = None,
) -> str:
    """Combine all sections into full research report.
    
    Structure:
    1. Query echo
    2. Verified Facts section
    3. Analysis section (labeled as interpretation)
    4. Conflicts section (if any)
    5. Not Found section (if any metrics weren't found)
    6. Sources section
    
    Args:
        store: FactStore containing verified facts
        query: The original research query
        conflicts: Optional list of detected conflicts
        not_found: Optional list of metrics not found
        llm: Optional LLM instance for analysis generation
    
    Returns:
        Complete markdown report string
    """
    facts = store.get_all_facts()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate each section
    facts_section = generate_facts_section(facts)
    analysis = generate_thesis_section(facts, query, llm=llm)
    conflicts_section = generate_conflicts_section(conflicts)
    not_found_section = generate_not_found_section(not_found)
    citations_section = generate_citations_section(facts)
    
    # Format supporting facts reference
    if analysis.supporting_facts:
        supporting_ref = ", ".join(analysis.supporting_facts)
    else:
        supporting_ref = "None"
    
    report = f"""# Research Report

**Query:** {query}
**Generated:** {timestamp}

## Verified Facts

{facts_section}

## Analysis

*Note: The following is interpretation based on the verified facts above. This is not verified factual content.*

{analysis.summary}

**Supporting Evidence:** Facts [{supporting_ref}]

## Data Conflicts

{conflicts_section}

## Not Found

{not_found_section}

## Sources

{citations_section}
"""
    
    return report

