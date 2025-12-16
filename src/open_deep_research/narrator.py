"""
LLM Narrator - Generates reports from verified facts.

The narrator is the ONLY component that generates human-readable text.
It can ONLY use facts from the FactStore - it cannot generate new facts.

Core principle: LLM is a "narrator over a verified fact table," not an author.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import List, Optional

from anthropic import Anthropic

from open_deep_research.models import Citation, Fact, NarratedReport
from open_deep_research.store import FactStore


logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================


NARRATOR_PROMPT = """You are a financial research analyst writing a factual report.

CRITICAL RULES:
1. ONLY use information from the verified facts below - do NOT add any outside knowledge
2. If the facts don't answer the question, respond with "Insufficient data to answer this question."
3. Cite each fact using [1], [2], etc. at the end of the sentence that uses that fact
4. Be concise and precise - stick to what the facts say
5. Do NOT speculate, infer, or add context not in the facts
6. Numbers must match the facts EXACTLY (do not round or convert)

Question: {query}

Verified Facts:
{numbered_facts}

Answer:"""


# =============================================================================
# Fact Formatting
# =============================================================================


def format_fact_for_prompt(fact: Fact, index: int) -> str:
    """Format a fact for inclusion in the narrator prompt.
    
    Args:
        fact: The fact to format
        index: 1-based index for citation
        
    Returns:
        Formatted string like "[1] NVDA Total Revenue Q3 FY2025: $35.08B (source: xbrl)"
    """
    # Format value
    if fact.value is not None:
        if abs(fact.value) >= 1_000_000_000:
            value_str = f"${fact.value / 1_000_000_000:.2f}B"
        elif abs(fact.value) >= 1_000_000:
            value_str = f"${fact.value / 1_000_000:.2f}M"
        else:
            value_str = f"${fact.value:,.2f}"
    else:
        value_str = "N/A"
    
    # Format location
    if fact.source_format == "xbrl":
        location = f"XBRL {fact.period}"
    elif fact.location.section_id:
        location = f"{fact.location.section_id}"
        if fact.location.paragraph_index is not None:
            location += f", Para {fact.location.paragraph_index}"
    else:
        location = fact.source_format
    
    return f"[{index}] {fact.entity} {fact.metric} {fact.period}: {value_str} (source: {fact.source_format}, {location})"


def format_facts_for_prompt(facts: List[Fact]) -> str:
    """Format multiple facts for the narrator prompt.
    
    Args:
        facts: List of facts to format
        
    Returns:
        Multi-line string with numbered facts
    """
    if not facts:
        return "No verified facts available."
    
    lines = []
    for i, fact in enumerate(facts, 1):
        lines.append(format_fact_for_prompt(fact, i))
    
    return "\n".join(lines)


# =============================================================================
# Fact Retrieval
# =============================================================================


def retrieve_relevant_facts(
    query: str,
    fact_store: FactStore,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    period: Optional[str] = None,
) -> List[Fact]:
    """Retrieve facts relevant to a query.
    
    Uses simple keyword matching for now. Can be enhanced with
    semantic search if needed.
    
    Args:
        query: The research question
        fact_store: Store containing verified facts
        entity: Optional ticker to filter by
        metric: Optional metric name to filter by
        period: Optional period to filter by
        
    Returns:
        List of relevant facts
    """
    query_lower = query.lower()
    
    # Start with all facts
    candidates = fact_store.get_all_facts()
    
    # Filter by entity if specified
    if entity:
        candidates = [f for f in candidates if f.entity.upper() == entity.upper()]
    
    # Filter by metric if specified
    if metric:
        metric_lower = metric.lower()
        candidates = [f for f in candidates if metric_lower in f.metric.lower()]
    
    # Filter by period if specified
    if period:
        candidates = [f for f in candidates if period.lower() in f.period.lower()]
    
    # If no explicit filters, try to extract from query
    if not entity and not metric:
        filtered = []
        for fact in candidates:
            # Check if entity mentioned in query
            if fact.entity.lower() in query_lower:
                # Check if metric mentioned in query
                if fact.metric.lower() in query_lower:
                    filtered.append(fact)
                    continue
                # Check for common metric keywords
                metric_keywords = {
                    "revenue": ["revenue", "sales"],
                    "net income": ["net income", "profit", "earnings"],
                    "gross profit": ["gross profit", "gross margin"],
                    "operating income": ["operating income", "operating profit"],
                    "cost": ["cost", "expense"],
                    "eps": ["eps", "earnings per share"],
                }
                for metric_name, keywords in metric_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if metric_name.lower() in fact.metric.lower():
                            filtered.append(fact)
                            break
        candidates = filtered if filtered else candidates
    
    # Limit to reasonable number
    return candidates[:20]


# =============================================================================
# Citation Parsing
# =============================================================================


def parse_citations_from_answer(
    answer: str,
    facts: List[Fact],
) -> List[Citation]:
    """Parse citation markers [1], [2], etc. from the answer.
    
    Args:
        answer: The LLM-generated answer text
        facts: The list of facts that were provided (1-indexed)
        
    Returns:
        List of Citation objects for each citation found
    """
    citations = []
    seen_indices = set()
    
    # Find all [N] patterns
    pattern = r'\[(\d+)\]'
    matches = re.finditer(pattern, answer)
    
    for match in matches:
        index = int(match.group(1))
        
        # Skip if already seen or invalid index
        if index in seen_indices or index < 1 or index > len(facts):
            continue
        
        seen_indices.add(index)
        fact = facts[index - 1]  # Convert to 0-indexed
        
        # Build location string
        if fact.source_format == "xbrl":
            location = f"XBRL: {fact.metric}, {fact.period}"
        elif fact.location.section_id:
            location = f"{fact.location.section_id}"
            if fact.location.paragraph_index is not None:
                location += f", Para {fact.location.paragraph_index}"
        else:
            location = fact.source_format
        
        citations.append(Citation(
            fact_id=fact.fact_id,
            citation_index=index,
            source_format=fact.source_format,
            location=location,
        ))
    
    # Sort by citation index
    citations.sort(key=lambda c: c.citation_index)
    
    return citations


# =============================================================================
# Main Narrator Function
# =============================================================================


def generate_report(
    query: str,
    fact_store: FactStore,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    period: Optional[str] = None,
) -> NarratedReport:
    """Generate a report answering a query using only verified facts.
    
    This is the main entry point for the narrator. It:
    1. Retrieves relevant facts from the FactStore
    2. Formats them into a prompt
    3. Calls the LLM to generate an answer
    4. Parses citations from the answer
    5. Returns a NarratedReport
    
    Args:
        query: The research question to answer
        fact_store: Store containing verified facts
        entity: Optional ticker to focus on
        metric: Optional metric to focus on
        period: Optional period to focus on
        
    Returns:
        NarratedReport with answer, citations, and facts used
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
    """
    logger.info(f"Generating report for: {query}")
    
    # Retrieve relevant facts
    facts = retrieve_relevant_facts(
        query=query,
        fact_store=fact_store,
        entity=entity,
        metric=metric,
        period=period,
    )
    
    logger.info(f"Found {len(facts)} relevant facts")
    
    # Handle no facts case
    if not facts:
        logger.warning("No relevant facts found in store")
        return NarratedReport(
            query=query,
            answer="Insufficient data to answer this question. No verified facts are available.",
            citations=[],
            facts_used=[],
            generated_at=datetime.now(),
            insufficient_data=True,
        )
    
    # Format facts for prompt
    numbered_facts = format_facts_for_prompt(facts)
    
    # Build prompt
    prompt = NARRATOR_PROMPT.format(
        query=query,
        numbered_facts=numbered_facts,
    )
    
    # Call LLM
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable required for narrator"
        )
    
    client = Anthropic(api_key=api_key)
    
    logger.debug(f"Calling LLM with prompt:\n{prompt}")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    
    answer = response.content[0].text.strip()
    
    logger.info(f"LLM answer: {answer[:100]}...")
    
    # Check for insufficient data
    insufficient_data = "insufficient data" in answer.lower()
    
    # Parse citations
    citations = parse_citations_from_answer(answer, facts)
    
    # Get facts actually used (only those cited)
    cited_fact_ids = {c.fact_id for c in citations}
    facts_used = [f for f in facts if f.fact_id in cited_fact_ids]
    
    # If no citations found but facts were provided, include all facts
    # (LLM may have used them without explicit citation)
    if not facts_used and facts and not insufficient_data:
        facts_used = facts
    
    return NarratedReport(
        query=query,
        answer=answer,
        citations=citations,
        facts_used=facts_used,
        generated_at=datetime.now(),
        insufficient_data=insufficient_data,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_report_from_facts(
    query: str,
    facts: List[Fact],
) -> NarratedReport:
    """Generate a report from a specific list of facts.
    
    Convenience function when you already have the facts and don't
    need to query a FactStore.
    
    Args:
        query: The research question
        facts: List of verified facts to use
        
    Returns:
        NarratedReport
    """
    # Create temporary store
    store = FactStore()
    for fact in facts:
        store.add_fact(fact)
    
    return generate_report(query, store)

