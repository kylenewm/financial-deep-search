#!/usr/bin/env python3
"""
Evaluation Runner Script.

Runs the golden dataset evaluation using the new XBRL-based architecture:
1. Routes each question to XBRL or LLM path via Source Router
2. XBRL path: Direct extraction, no LLM, no verification needed
3. LLM path: Extraction + verification gate
4. All paths feed into FactStore

Usage:
    python scripts/run_eval.py --verbose
    python scripts/run_eval.py --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from open_deep_research.models import (
    EvalQuestion,
    EvalResult,
    EvalSummary,
    Fact,
    ResearchOutput,
)
from open_deep_research.eval import (
    load_golden_dataset,
    evaluate_single,
    print_eval_report,
)
from open_deep_research.router import (
    route_query,
    SourceType,
    RouteResult,
)
from open_deep_research.xbrl import (
    extract_xbrl_fact,
)
from open_deep_research.entities import resolve_entity
from open_deep_research.store import FactStore
from open_deep_research.narrator import generate_report


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the evaluation run."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Reduce noise from some loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# =============================================================================
# Period Parsing
# =============================================================================


def parse_fiscal_period(period_str: str) -> tuple[Optional[int], Optional[str]]:
    """
    Parse a period string into fiscal year and period.
    
    Args:
        period_str: e.g., "Q3 FY2025", "FY2024", "2024 Q3", "Q3 2024"
        
    Returns:
        Tuple of (fiscal_year, fiscal_period) e.g., (2025, "Q3")
    """
    import re
    
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


# =============================================================================
# XBRL-Based Question Handler
# =============================================================================


def handle_xbrl_question(
    question: EvalQuestion,
    route_result: RouteResult,
) -> ResearchOutput:
    """
    Handle a question using the XBRL path.
    
    This is deterministic - no LLM calls, no verification needed.
    
    Args:
        question: The evaluation question
        route_result: Routing result with metric name
        
    Returns:
        ResearchOutput with the extracted fact (or empty if not found)
    """
    logger = logging.getLogger(__name__)
    
    # Resolve entity to get CIK
    entity = resolve_entity(question.entity)
    if not entity:
        logger.warning(f"Could not resolve entity: {question.entity}")
        return ResearchOutput(
            query=question.question,
            generated_at=datetime.now(),
            facts=[],
            not_found=[],
        )
    
    # Parse the expected period
    fiscal_year, fiscal_period = parse_fiscal_period(question.expected_period)
    
    if not fiscal_year or not fiscal_period:
        logger.warning(f"Could not parse period: {question.expected_period}")
        return ResearchOutput(
            query=question.question,
            generated_at=datetime.now(),
            facts=[],
            not_found=[],
        )
    
    # Extract fact from XBRL
    logger.info(f"XBRL extraction: {question.metric} for {question.entity} {fiscal_period} FY{fiscal_year}")
    
    fact = extract_xbrl_fact(
        cik=entity.cik,
        metric=question.metric,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        ticker=question.entity,  # Pass ticker for entity field matching
    )
    
    if fact:
        logger.info(f"XBRL found: {fact.metric} = {fact.value:,.0f} {fact.unit}")
        return ResearchOutput(
            query=question.question,
            generated_at=datetime.now(),
            facts=[fact],
            not_found=[],
        )
    else:
        logger.warning(f"XBRL not found: {question.metric}")
        return ResearchOutput(
            query=question.question,
            generated_at=datetime.now(),
            facts=[],
            not_found=[],
        )


# =============================================================================
# LLM-Based Question Handler (Placeholder)
# =============================================================================


def handle_llm_question(
    question: EvalQuestion,
    route_result: RouteResult,
) -> ResearchOutput:
    """
    Handle a question using the LLM + verification path.
    
    This is a placeholder - for now we return empty results.
    The LLM path would:
    1. Download SEC filing HTML
    2. Parse into sections
    3. Extract facts with LLM
    4. Verify through verification gate
    5. Add verified facts to store
    
    Args:
        question: The evaluation question
        route_result: Routing result
        
    Returns:
        ResearchOutput (currently empty)
    """
    logger = logging.getLogger(__name__)
    logger.warning(f"LLM path not fully implemented - returning empty for: {question.question[:50]}...")
    
    return ResearchOutput(
        query=question.question,
        generated_at=datetime.now(),
        facts=[],
        not_found=[],
    )


# =============================================================================
# Main Question Handler
# =============================================================================


def handle_question(question: EvalQuestion) -> ResearchOutput:
    """
    Handle a single evaluation question using the Source Router.
    
    Routes to XBRL or LLM path based on the question type.
    
    Args:
        question: The evaluation question
        
    Returns:
        ResearchOutput with extracted facts
    """
    logger = logging.getLogger(__name__)
    
    # Route the question
    route_result = route_query(
        question=question.question,
        entity=question.entity,
    )
    
    logger.info(
        f"Routed '{question.metric}' to {route_result.source_type.value} "
        f"(confidence: {route_result.confidence:.0%})"
    )
    
    # Handle based on route
    if route_result.source_type == SourceType.XBRL:
        return handle_xbrl_question(question, route_result)
    else:
        return handle_llm_question(question, route_result)


# =============================================================================
# FactStore Builder
# =============================================================================


def build_fact_store(
    dataset: list[EvalQuestion],
    verbose: bool = False,
) -> FactStore:
    """
    Build a FactStore with all XBRL facts for the dataset.
    
    This populates the fact table BEFORE report generation.
    
    Args:
        dataset: List of evaluation questions
        verbose: If True, print progress
        
    Returns:
        FactStore populated with verified XBRL facts
    """
    logger = logging.getLogger(__name__)
    store = FactStore()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Building FactStore from XBRL data...")
        print("=" * 60)
    
    # Track unique (entity, metric, period) to avoid duplicates
    seen = set()
    
    for question in dataset:
        key = (question.entity, question.metric, question.expected_period)
        if key in seen:
            continue
        seen.add(key)
        
        # Route the question
        route_result = route_query(
            question=question.question,
            entity=question.entity,
        )
        
        # Only handle XBRL-routed questions
        if route_result.source_type != SourceType.XBRL:
            logger.debug(f"Skipping non-XBRL question: {question.metric}")
            continue
        
        # Extract from XBRL
        entity = resolve_entity(question.entity)
        if not entity:
            continue
        
        fiscal_year, fiscal_period = parse_fiscal_period(question.expected_period)
        if not fiscal_year or not fiscal_period:
            continue
        
        fact = extract_xbrl_fact(
            cik=entity.cik,
            metric=question.metric,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            ticker=question.entity,
        )
        
        if fact:
            store.add_fact(fact)
            if verbose:
                print(f"  ✓ {question.entity} {question.metric}: ${fact.value/1e9:.2f}B")
    
    if verbose:
        print(f"\nFactStore populated with {len(store)} verified facts")
    
    return store


# =============================================================================
# Evaluation Runner
# =============================================================================


def run_evaluation(
    dataset: list[EvalQuestion],
    verbose: bool = False,
    narrate: bool = True,
) -> EvalSummary:
    """
    Run the full evaluation on a dataset.
    
    Args:
        dataset: List of evaluation questions
        verbose: If True, print detailed progress
        narrate: If True, generate narrated reports for each question
        
    Returns:
        EvalSummary with all results
    """
    logger = logging.getLogger(__name__)
    results = []
    
    # Build FactStore first (narrator over verified fact table)
    fact_store = build_fact_store(dataset, verbose=verbose)
    
    for i, question in enumerate(dataset, 1):
        if verbose:
            print(f"\n[{i}/{len(dataset)}] {question.question}")
        
        try:
            output = handle_question(question)
            result = evaluate_single(question, output)
            
            if verbose:
                status = "PASS" if result.value_correct else "FAIL"
                if result.actual_value is not None:
                    print(f"  Expected: {question.expected_value:,.0f}")
                    print(f"  Actual:   {result.actual_value:,.0f}")
                    print(f"  Status:   {status}")
                else:
                    print(f"  Status:   {status} (no value extracted)")
                    print(f"  Error:    {result.error_type}")
            
            # Generate narrated report if enabled
            if narrate and verbose and len(fact_store) > 0:
                _print_narrated_report(question, fact_store)
                    
        except Exception as e:
            logger.error(f"Question {question.question_id} failed: {e}")
            result = EvalResult(
                question_id=question.question_id,
                question=question.question,
                expected_value=question.expected_value,
                expected_period=question.expected_period,
                actual_value=None,
                actual_period=None,
                value_correct=False,
                period_correct=False,
                source_tier_correct=False,
                error_type="retrieval_failure",
                facts_returned=0,
                verification_status=None,
            )
        
        results.append(result)
    
    # Build summary
    return _build_summary(results)


def _print_narrated_report(
    question: EvalQuestion,
    fact_store: FactStore,
) -> None:
    """
    Generate and print a narrated report for a question.
    
    Args:
        question: The evaluation question
        fact_store: Store containing verified facts
    """
    import os
    
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [Narrator skipped - no ANTHROPIC_API_KEY]")
        return
    
    try:
        report = generate_report(
            query=question.question,
            fact_store=fact_store,
            entity=question.entity,
        )
        
        print(f"\n  📝 Narrated Answer:")
        print(f"     {report.answer}")
        
        if report.citations:
            print(f"\n  📚 Citations ({len(report.citations)}):")
            for c in report.citations:
                # Verify citation matches a fact in store
                fact = fact_store.get_fact(c.fact_id)
                verified = "✓" if fact else "✗"
                print(f"     [{c.citation_index}] {verified} {c.source_format}: {c.location}")
        
        if report.insufficient_data:
            print("     ⚠️  Insufficient data to answer")
            
    except Exception as e:
        print(f"  [Narrator error: {e}]")


def _build_summary(results: list[EvalResult]) -> EvalSummary:
    """Build EvalSummary from list of results."""
    total = len(results)
    
    if total == 0:
        return EvalSummary(
            total_questions=0,
            value_accuracy=0.0,
            period_accuracy=0.0,
            source_tier_accuracy=0.0,
            retrieval_failures=0,
            extraction_failures=0,
            verification_failures=0,
            hallucinations=0,
            pass_rate=0.0,
            results=[],
        )
    
    value_correct = sum(1 for r in results if r.value_correct)
    period_correct = sum(1 for r in results if r.period_correct)
    source_correct = sum(1 for r in results if r.source_tier_correct)
    
    # Count error types
    retrieval_failures = sum(1 for r in results if r.error_type == "retrieval_failure")
    extraction_failures = sum(1 for r in results if r.error_type == "extraction_failure")
    verification_failures = sum(1 for r in results if r.error_type == "verification_failure")
    hallucinations = sum(1 for r in results if r.error_type == "hallucination")
    
    # Pass = value AND period correct
    passes = sum(1 for r in results if r.value_correct and r.period_correct)
    
    return EvalSummary(
        total_questions=total,
        value_accuracy=(value_correct / total) * 100,
        period_accuracy=(period_correct / total) * 100,
        source_tier_accuracy=(source_correct / total) * 100,
        retrieval_failures=retrieval_failures,
        extraction_failures=extraction_failures,
        verification_failures=verification_failures,
        hallucinations=hallucinations,
        pass_rate=(passes / total) * 100,
        results=results,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run the golden dataset evaluation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Generate narrated reports using LLM (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to golden dataset JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print header
    print("=" * 60)
    print("XBRL-Based Evaluation Runner")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.narrate:
        print("Narrator: ENABLED")
    print()
    
    # Load dataset
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.json"
    
    try:
        dataset = load_golden_dataset(str(dataset_path))
        print(f"Loaded {len(dataset)} questions from {dataset_path}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return 1
    
    # Apply limit if specified
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limited to {len(dataset)} questions")
    
    print("-" * 60)
    
    # Run evaluation
    summary = run_evaluation(
        dataset,
        verbose=args.verbose,
        narrate=args.narrate,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print_eval_report(summary)
    
    # Determine exit code
    # Pass if >= 80% pass rate (arbitrary threshold)
    if summary.pass_rate >= 80.0:
        print("\nEVALUATION PASSED")
        return 0
    else:
        print(f"\nEVALUATION FAILED (pass rate {summary.pass_rate:.1f}% < 80%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
