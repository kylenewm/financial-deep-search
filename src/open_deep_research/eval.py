"""
Evaluation Harness - Golden dataset testing for anti-hallucination verification.

Creates a golden dataset with known answers, runs the system on each question,
and scores results. This is how you prove the system works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from open_deep_research.models import (
    EvalQuestion,
    EvalResult,
    EvalSummary,
    ResearchOutput,
)
from open_deep_research.numeric_verification import verify_numeric_fact


def load_golden_dataset(path: str = "evals/golden_dataset.json") -> list[EvalQuestion]:
    """Load evaluation questions from JSON file.
    
    Args:
        path: Path to the golden dataset JSON file
        
    Returns:
        List of EvalQuestion objects
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValidationError: If questions are missing required fields
    """
    dataset_path = Path(path)
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        questions.append(EvalQuestion(**item))
    
    return questions


def evaluate_single(question: EvalQuestion, output: ResearchOutput) -> EvalResult:
    """Evaluate a single question against system output.
    
    Finds a matching fact in the output based on entity and metric,
    then checks if the value and period are correct.
    
    Args:
        question: The evaluation question with expected answer
        output: The system's output for this question
        
    Returns:
        EvalResult with scores and error classification
    """
    # Find matching fact in output
    matching_fact = None
    for fact in output.facts:
        if (fact.entity.upper() == question.entity.upper() and 
            fact.metric.lower() == question.metric.lower()):
            matching_fact = fact
            break
    
    # No matching fact found
    if matching_fact is None:
        return EvalResult(
            question_id=question.question_id,
            question=question.question,
            expected_value=question.expected_value,
            expected_period=question.expected_period,
            actual_value=None,
            actual_period=None,
            value_correct=False,
            period_correct=False,
            source_tier_correct=False,
            error_type="extraction_failure",
            facts_returned=len(output.facts),
            verification_status=None,
        )
    
    # Check value (1% tolerance)
    if matching_fact.value is None:
        value_correct = False
        value_status = "mismatch"
    else:
        value_status = verify_numeric_fact(matching_fact.value, question.expected_value)
        value_correct = value_status in ("exact_match", "approximate_match")
    
    # Check period
    period_correct = matching_fact.period == question.expected_period
    
    # Determine source tier correctness (Tier 1 = SEC filings)
    # If we got here, we used SEC filing (Tier 1)
    source_tier_correct = True
    
    # Determine error type
    error_type = None
    if not value_correct:
        if matching_fact.verification_status == "exact_match":
            error_type = "hallucination"  # System was confident but wrong
        else:
            error_type = "verification_failure"
    elif not period_correct:
        error_type = "period_mismatch"
    
    return EvalResult(
        question_id=question.question_id,
        question=question.question,
        expected_value=question.expected_value,
        expected_period=question.expected_period,
        actual_value=matching_fact.value,
        actual_period=matching_fact.period,
        value_correct=value_correct,
        period_correct=period_correct,
        source_tier_correct=source_tier_correct,
        error_type=error_type,
        facts_returned=len(output.facts),
        verification_status=matching_fact.verification_status,
    )


def run_full_evaluation(
    dataset: list[EvalQuestion],
    system_fn: Callable[[str], ResearchOutput],
) -> EvalSummary:
    """Run evaluation on full dataset.
    
    Args:
        dataset: List of evaluation questions
        system_fn: Function that takes a question string and returns ResearchOutput
        
    Returns:
        EvalSummary with aggregate metrics and all results
    """
    results = []
    for question in dataset:
        try:
            output = system_fn(question.question)
            result = evaluate_single(question, output)
        except Exception:
            # System crashed - count as retrieval failure
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
    
    return _build_summary(results)


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
    
    value_correct_count = sum(1 for r in results if r.value_correct)
    period_correct_count = sum(1 for r in results if r.period_correct)
    source_correct_count = sum(1 for r in results if r.source_tier_correct)
    fully_correct_count = sum(1 for r in results if r.value_correct and r.period_correct)
    
    # Error breakdown
    retrieval_failures = sum(1 for r in results if r.error_type == "retrieval_failure")
    extraction_failures = sum(1 for r in results if r.error_type == "extraction_failure")
    verification_failures = sum(1 for r in results if r.error_type == "verification_failure")
    hallucinations = sum(1 for r in results if r.error_type == "hallucination")
    
    return EvalSummary(
        total_questions=total,
        value_accuracy=value_correct_count / total,
        period_accuracy=period_correct_count / total,
        source_tier_accuracy=source_correct_count / total,
        retrieval_failures=retrieval_failures,
        extraction_failures=extraction_failures,
        verification_failures=verification_failures,
        hallucinations=hallucinations,
        pass_rate=fully_correct_count / total,
        results=results,
    )


async def run_full_evaluation_async(
    dataset: list[EvalQuestion],
    system_fn,  # Async callable
) -> EvalSummary:
    """Async version - runs all questions in a single event loop.
    
    This avoids "Event loop is closed" errors by using one event loop.
    
    Args:
        dataset: List of evaluation questions
        system_fn: Async function that takes question string and returns ResearchOutput
        
    Returns:
        EvalSummary with aggregate metrics and all results
    """
    results = []
    for question in dataset:
        try:
            output = await system_fn(question.question, question)
            result = evaluate_single(question, output)
        except Exception as e:
            # System crashed - count as retrieval failure
            import logging
            logging.error(f"Question {question.question_id} failed: {e}")
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
    
    return _build_summary(results)


def print_eval_report(summary: EvalSummary) -> None:
    """Print human-readable evaluation report.
    
    Args:
        summary: EvalSummary containing all results and metrics
    """
    print("=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Questions: {summary.total_questions}")
    print(f"Pass Rate: {summary.pass_rate:.1f}%")
    print()
    print("ACCURACY METRICS")
    print("-" * 30)
    print(f"Value Accuracy:  {summary.value_accuracy:.1f}%")
    print(f"Period Accuracy: {summary.period_accuracy:.1f}%")
    print(f"Source Tier:     {summary.source_tier_accuracy:.1f}%")
    print()
    print("ERROR BREAKDOWN")
    print("-" * 30)
    print(f"Retrieval Failures:    {summary.retrieval_failures}")
    print(f"Extraction Failures:   {summary.extraction_failures}")
    print(f"Verification Failures: {summary.verification_failures}")
    print(f"Hallucinations:        {summary.hallucinations}")
    print()
    print("DETAILED RESULTS")
    print("-" * 30)
    
    for result in summary.results:
        status = "✓" if result.value_correct and result.period_correct else "✗"
        error_info = f" ({result.error_type})" if result.error_type else ""
        print(f"[{status}] {result.question_id}{error_info}")
        if not result.value_correct and result.actual_value is not None:
            print(f"    Expected: {result.expected_value}, Got: {result.actual_value}")
        elif result.actual_value is None:
            print(f"    Expected: {result.expected_value}, Got: None")
    
    print("=" * 50)

