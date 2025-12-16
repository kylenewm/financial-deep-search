"""
Tests for the evaluation harness.

Tests cover:
- Dataset loading (valid, invalid, empty)
- Single evaluation (correct, wrong value, missing fact, period mismatch)
- Summary calculation (accuracy, pass rate, error counts)
- Report generation
- Edge cases (empty dataset, system crashes, all pass/fail)
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from open_deep_research.eval import (
    evaluate_single,
    load_golden_dataset,
    print_eval_report,
    run_full_evaluation,
)
from open_deep_research.models import (
    EvalQuestion,
    EvalResult,
    EvalSummary,
    Fact,
    Location,
    ResearchOutput,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_question() -> EvalQuestion:
    """Create a sample evaluation question."""
    return EvalQuestion(
        question_id="nvda-dc-rev-q3fy25",
        question="What was NVIDIA's datacenter revenue in Q3 FY2025?",
        entity="NVDA",
        metric="datacenter_revenue",
        expected_value=14514000000,
        expected_unit="USD",
        expected_period="Q3 FY2025",
        authoritative_source="NVIDIA 10-Q filed 2024-11-20",
        notes="From Item 7, segment revenue table",
    )


@pytest.fixture
def sample_location() -> Location:
    """Create a sample location."""
    return Location(
        cik="0001045810",
        doc_date="2024-11-20",
        doc_type="10-Q",
        section_id="Item7",
        table_index=0,
        row_index=1,
        column_index=1,
        row_label="Data Center",
        column_label="Oct 27, 2024",
    )


@pytest.fixture
def sample_fact(sample_location: Location) -> Fact:
    """Create a sample fact matching the sample question."""
    return Fact(
        fact_id="fact-001",
        entity="NVDA",
        metric="datacenter_revenue",
        value=14514000000,
        unit="USD",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=sample_location,
        source_format="html_table",
        extracted_scale="millions",
        doc_hash="abc123",
        snapshot_id="snap-001",
        verification_status="exact_match",
    )


@pytest.fixture
def sample_output(sample_fact: Fact) -> ResearchOutput:
    """Create a sample ResearchOutput with the sample fact."""
    return ResearchOutput(
        query="What was NVIDIA's datacenter revenue in Q3 FY2025?",
        generated_at=datetime.now(),
        facts=[sample_fact],
    )


@pytest.fixture
def golden_dataset_path() -> Path:
    """Return path to the actual golden dataset."""
    return Path("evals/golden_dataset.json")


# =============================================================================
# Dataset Loading Tests
# =============================================================================


class TestLoadGoldenDataset:
    """Tests for load_golden_dataset function."""
    
    def test_load_valid_dataset(self, golden_dataset_path: Path):
        """Test loading a valid dataset."""
        if not golden_dataset_path.exists():
            pytest.skip("Golden dataset not found")
        
        questions = load_golden_dataset(str(golden_dataset_path))
        
        assert len(questions) >= 5  # We have at least 5 questions
        assert all(isinstance(q, EvalQuestion) for q in questions)
        
        # Check first question (golden dataset was restored with real financial metrics)
        q = questions[0]
        assert q.question_id == "nvda-total-revenue-q3fy25"
        assert q.entity == "NVDA"
        assert q.expected_value == 35082000000
    
    def test_load_valid_dataset_from_temp_file(self):
        """Test loading from a temporary file."""
        data = [
            {
                "question_id": "test-q1",
                "question": "Test question?",
                "entity": "TEST",
                "metric": "test_metric",
                "expected_value": 1000.0,
                "expected_unit": "USD",
                "expected_period": "Q1 2024",
                "authoritative_source": "Test source",
                "notes": None,
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            questions = load_golden_dataset(temp_path)
            assert len(questions) == 1
            assert questions[0].question_id == "test-q1"
        finally:
            Path(temp_path).unlink()
    
    def test_handles_empty_dataset(self):
        """Test that empty dataset returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            temp_path = f.name
        
        try:
            questions = load_golden_dataset(temp_path)
            assert questions == []
        finally:
            Path(temp_path).unlink()
    
    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raises validation error."""
        data = [
            {
                "question_id": "test-q1",
                "question": "Test question?",
                # Missing: entity, metric, expected_value, etc.
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):  # Pydantic ValidationError
                load_golden_dataset(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_golden_dataset(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_golden_dataset("nonexistent/path/to/dataset.json")


# =============================================================================
# Single Evaluation Tests
# =============================================================================


class TestEvaluateSingle:
    """Tests for evaluate_single function."""
    
    def test_correct_answer_passes(
        self, sample_question: EvalQuestion, sample_output: ResearchOutput
    ):
        """Test that correct answer scores as pass."""
        result = evaluate_single(sample_question, sample_output)
        
        assert result.value_correct is True
        assert result.period_correct is True
        assert result.source_tier_correct is True
        assert result.error_type is None
        assert result.actual_value == 14514000000
        assert result.actual_period == "Q3 FY2025"
    
    def test_wrong_value_fails(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that wrong value (>1% off) scores as fail."""
        # Create fact with wrong value (20% off)
        wrong_fact = Fact(
            fact_id="fact-002",
            entity="NVDA",
            metric="datacenter_revenue",
            value=12000000000,  # Wrong value
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="mismatch",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[wrong_fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is False
        assert result.error_type == "verification_failure"
        assert result.actual_value == 12000000000
    
    def test_missing_fact_returns_extraction_failure(
        self, sample_question: EvalQuestion
    ):
        """Test that missing fact returns extraction_failure."""
        # Empty output
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is False
        assert result.period_correct is False
        assert result.error_type == "extraction_failure"
        assert result.actual_value is None
        assert result.facts_returned == 0
    
    def test_wrong_period_detected(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that wrong period is detected as period_mismatch."""
        # Create fact with correct value but wrong period
        wrong_period_fact = Fact(
            fact_id="fact-003",
            entity="NVDA",
            metric="datacenter_revenue",
            value=14514000000,  # Correct value
            unit="USD",
            period="Q2 FY2025",  # Wrong period
            period_end_date="2024-07-28",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="exact_match",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[wrong_period_fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is True
        assert result.period_correct is False
        assert result.error_type == "period_mismatch"
    
    def test_value_within_1_percent_passes(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that value within 1% tolerance passes."""
        # Create fact with value 0.5% off
        close_fact = Fact(
            fact_id="fact-004",
            entity="NVDA",
            metric="datacenter_revenue",
            value=14586570000,  # 0.5% higher
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="approximate_match",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[close_fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is True  # Within tolerance
        assert result.period_correct is True
        assert result.error_type is None
    
    def test_hallucination_detected(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that hallucination is detected when system is confident but wrong."""
        # Create fact with wrong value but exact_match verification status
        # This simulates a case where verification passed but answer is still wrong
        hallucination_fact = Fact(
            fact_id="fact-005",
            entity="NVDA",
            metric="datacenter_revenue",
            value=20000000000,  # Wrong value
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="exact_match",  # System thinks it's correct
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[hallucination_fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is False
        assert result.error_type == "hallucination"
    
    def test_case_insensitive_entity_matching(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that entity matching is case-insensitive."""
        # Create fact with lowercase entity
        fact = Fact(
            fact_id="fact-006",
            entity="nvda",  # lowercase
            metric="datacenter_revenue",
            value=14514000000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="exact_match",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is True
        assert result.period_correct is True
    
    def test_case_insensitive_metric_matching(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test that metric matching is case-insensitive."""
        # Create fact with different case metric
        fact = Fact(
            fact_id="fact-007",
            entity="NVDA",
            metric="Datacenter_Revenue",  # Mixed case
            value=14514000000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="exact_match",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is True
    
    def test_null_value_in_fact(
        self, sample_question: EvalQuestion, sample_location: Location
    ):
        """Test handling of fact with null value."""
        fact = Fact(
            fact_id="fact-008",
            entity="NVDA",
            metric="datacenter_revenue",
            value=None,  # Null value
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=sample_location,
            source_format="html_table",
            doc_hash="abc123",
            snapshot_id="snap-001",
            verification_status="unverified",
        )
        
        output = ResearchOutput(
            query="Test",
            generated_at=datetime.now(),
            facts=[fact],
        )
        
        result = evaluate_single(sample_question, output)
        
        assert result.value_correct is False
        assert result.actual_value is None


# =============================================================================
# Summary Calculation Tests
# =============================================================================


class TestRunFullEvaluation:
    """Tests for run_full_evaluation function."""
    
    def test_value_accuracy_calculated_correctly(self, sample_question: EvalQuestion):
        """Test that value_accuracy is calculated correctly."""
        questions = [sample_question, sample_question]  # 2 identical questions
        
        call_count = 0
        
        def mock_system_fn(question: str) -> ResearchOutput:
            nonlocal call_count
            call_count += 1
            
            # First call returns correct value, second returns wrong
            if call_count == 1:
                return ResearchOutput(
                    query=question,
                    generated_at=datetime.now(),
                    facts=[
                        Fact(
                            fact_id="f1",
                            entity="NVDA",
                            metric="datacenter_revenue",
                            value=14514000000,
                            unit="USD",
                            period="Q3 FY2025",
                            period_end_date="2024-10-27",
                            location=Location(
                                cik="0001045810",
                                doc_date="2024-11-20",
                                doc_type="10-Q",
                                section_id="Item7",
                            ),
                            source_format="html_table",
                            doc_hash="abc",
                            snapshot_id="snap",
                            verification_status="exact_match",
                        )
                    ],
                )
            else:
                return ResearchOutput(
                    query=question,
                    generated_at=datetime.now(),
                    facts=[],  # No facts = extraction failure
                )
        
        summary = run_full_evaluation(questions, mock_system_fn)
        
        assert summary.value_accuracy == 0.5  # 1 of 2 correct
    
    def test_period_accuracy_calculated_correctly(self):
        """Test that period_accuracy is calculated correctly."""
        questions = [
            EvalQuestion(
                question_id="q1",
                question="Q1",
                entity="TEST",
                metric="metric",
                expected_value=100,
                expected_unit="USD",
                expected_period="Q1 2024",
                authoritative_source="Test",
            ),
            EvalQuestion(
                question_id="q2",
                question="Q2",
                entity="TEST",
                metric="metric",
                expected_value=200,
                expected_unit="USD",
                expected_period="Q2 2024",
                authoritative_source="Test",
            ),
        ]
        
        def mock_system_fn(question: str) -> ResearchOutput:
            # Always return Q1 2024 period (correct for first, wrong for second)
            return ResearchOutput(
                query=question,
                generated_at=datetime.now(),
                facts=[
                    Fact(
                        fact_id="f",
                        entity="TEST",
                        metric="metric",
                        value=100,  # Matches first question
                        unit="USD",
                        period="Q1 2024",  # Only matches first question
                        period_end_date="2024-03-31",
                        location=Location(
                            cik="0000000001",
                            doc_date="2024-01-01",
                            doc_type="10-K",
                            section_id="Item7",
                        ),
                        source_format="html_text",
                        doc_hash="abc",
                        snapshot_id="snap",
                        verification_status="exact_match",
                    )
                ],
            )
        
        summary = run_full_evaluation(questions, mock_system_fn)
        
        # First: value correct, period correct
        # Second: value wrong (100 vs 200), period wrong (Q1 vs Q2)
        assert summary.period_accuracy == 0.5  # Only first has correct period
    
    def test_pass_rate_calculated_correctly(self):
        """Test that pass_rate (value AND period correct) is calculated correctly."""
        questions = [
            EvalQuestion(
                question_id="q1",
                question="Q1",
                entity="TEST",
                metric="metric",
                expected_value=100,
                expected_unit="USD",
                expected_period="Q1 2024",
                authoritative_source="Test",
            ),
        ]
        
        def mock_system_fn(question: str) -> ResearchOutput:
            return ResearchOutput(
                query=question,
                generated_at=datetime.now(),
                facts=[
                    Fact(
                        fact_id="f",
                        entity="TEST",
                        metric="metric",
                        value=100,
                        unit="USD",
                        period="Q1 2024",
                        period_end_date="2024-03-31",
                        location=Location(
                            cik="0000000001",
                            doc_date="2024-01-01",
                            doc_type="10-K",
                            section_id="Item7",
                        ),
                        source_format="html_text",
                        doc_hash="abc",
                        snapshot_id="snap",
                        verification_status="exact_match",
                    )
                ],
            )
        
        summary = run_full_evaluation(questions, mock_system_fn)
        
        assert summary.pass_rate == 1.0  # All passed
    
    def test_error_counts_are_correct(self):
        """Test that error type counts are calculated correctly."""
        questions = [
            EvalQuestion(
                question_id=f"q{i}",
                question=f"Q{i}",
                entity="TEST",
                metric=f"metric{i}",
                expected_value=100,
                expected_unit="USD",
                expected_period="Q1 2024",
                authoritative_source="Test",
            )
            for i in range(4)
        ]
        
        call_count = 0
        
        def mock_system_fn(question: str) -> ResearchOutput:
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # Extraction failure (no matching fact)
                return ResearchOutput(
                    query=question,
                    generated_at=datetime.now(),
                    facts=[],
                )
            elif call_count == 2:
                raise RuntimeError("System crash")  # Retrieval failure
            elif call_count == 3:
                # Verification failure (wrong value, not confident)
                return ResearchOutput(
                    query=question,
                    generated_at=datetime.now(),
                    facts=[
                        Fact(
                            fact_id="f",
                            entity="TEST",
                            metric="metric2",
                            value=999,  # Wrong value
                            unit="USD",
                            period="Q1 2024",
                            period_end_date="2024-03-31",
                            location=Location(
                                cik="0000000001",
                                doc_date="2024-01-01",
                                doc_type="10-K",
                                section_id="Item7",
                            ),
                            source_format="html_text",
                            doc_hash="abc",
                            snapshot_id="snap",
                            verification_status="mismatch",
                        )
                    ],
                )
            else:
                # Hallucination (wrong value but confident)
                return ResearchOutput(
                    query=question,
                    generated_at=datetime.now(),
                    facts=[
                        Fact(
                            fact_id="f",
                            entity="TEST",
                            metric="metric3",
                            value=999,  # Wrong value
                            unit="USD",
                            period="Q1 2024",
                            period_end_date="2024-03-31",
                            location=Location(
                                cik="0000000001",
                                doc_date="2024-01-01",
                                doc_type="10-K",
                                section_id="Item7",
                            ),
                            source_format="html_text",
                            doc_hash="abc",
                            snapshot_id="snap",
                            verification_status="exact_match",  # Confident
                        )
                    ],
                )
        
        summary = run_full_evaluation(questions, mock_system_fn)
        
        assert summary.extraction_failures == 1
        assert summary.retrieval_failures == 1
        assert summary.verification_failures == 1
        assert summary.hallucinations == 1
    
    def test_all_results_included_in_summary(self, sample_question: EvalQuestion):
        """Test that all results are included in summary.results."""
        questions = [sample_question] * 3
        
        def mock_system_fn(question: str) -> ResearchOutput:
            return ResearchOutput(
                query=question,
                generated_at=datetime.now(),
                facts=[],
            )
        
        summary = run_full_evaluation(questions, mock_system_fn)
        
        assert len(summary.results) == 3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataset(self):
        """Test evaluation with zero questions."""
        summary = run_full_evaluation([], lambda q: None)
        
        assert summary.total_questions == 0
        assert summary.pass_rate == 0.0
        assert summary.value_accuracy == 0.0
        assert summary.results == []
    
    def test_system_fn_raises_exception(self, sample_question: EvalQuestion):
        """Test when system_fn raises exception."""
        def crashing_system(question: str) -> ResearchOutput:
            raise RuntimeError("System crashed!")
        
        summary = run_full_evaluation([sample_question], crashing_system)
        
        assert summary.total_questions == 1
        assert summary.retrieval_failures == 1
        assert summary.results[0].error_type == "retrieval_failure"
        assert summary.pass_rate == 0.0
    
    def test_all_questions_pass(self):
        """Test when all questions pass."""
        questions = [
            EvalQuestion(
                question_id="q1",
                question="Q1",
                entity="TEST",
                metric="metric",
                expected_value=100,
                expected_unit="USD",
                expected_period="Q1 2024",
                authoritative_source="Test",
            )
        ] * 5
        
        def perfect_system(question: str) -> ResearchOutput:
            return ResearchOutput(
                query=question,
                generated_at=datetime.now(),
                facts=[
                    Fact(
                        fact_id="f",
                        entity="TEST",
                        metric="metric",
                        value=100,
                        unit="USD",
                        period="Q1 2024",
                        period_end_date="2024-03-31",
                        location=Location(
                            cik="0000000001",
                            doc_date="2024-01-01",
                            doc_type="10-K",
                            section_id="Item7",
                        ),
                        source_format="html_text",
                        doc_hash="abc",
                        snapshot_id="snap",
                        verification_status="exact_match",
                    )
                ],
            )
        
        summary = run_full_evaluation(questions, perfect_system)
        
        assert summary.pass_rate == 1.0
        assert summary.value_accuracy == 1.0
        assert summary.period_accuracy == 1.0
        assert summary.retrieval_failures == 0
        assert summary.extraction_failures == 0
        assert summary.verification_failures == 0
        assert summary.hallucinations == 0
    
    def test_all_questions_fail(self):
        """Test when all questions fail."""
        questions = [
            EvalQuestion(
                question_id="q1",
                question="Q1",
                entity="TEST",
                metric="metric",
                expected_value=100,
                expected_unit="USD",
                expected_period="Q1 2024",
                authoritative_source="Test",
            )
        ] * 5
        
        def failing_system(question: str) -> ResearchOutput:
            return ResearchOutput(
                query=question,
                generated_at=datetime.now(),
                facts=[],  # No facts found
            )
        
        summary = run_full_evaluation(questions, failing_system)
        
        assert summary.pass_rate == 0.0
        assert summary.value_accuracy == 0.0
        assert summary.period_accuracy == 0.0
        assert summary.extraction_failures == 5


# =============================================================================
# Report Tests
# =============================================================================


class TestPrintEvalReport:
    """Tests for print_eval_report function."""
    
    def test_print_report_runs_without_error(self, capsys):
        """Test that print_eval_report runs without error."""
        summary = EvalSummary(
            total_questions=5,
            value_accuracy=0.8,
            period_accuracy=1.0,
            source_tier_accuracy=1.0,
            retrieval_failures=0,
            extraction_failures=1,
            verification_failures=0,
            hallucinations=0,
            pass_rate=0.8,
            results=[
                EvalResult(
                    question_id="q1",
                    question="Q1",
                    expected_value=100,
                    expected_period="Q1 2024",
                    actual_value=100,
                    actual_period="Q1 2024",
                    value_correct=True,
                    period_correct=True,
                    source_tier_correct=True,
                    error_type=None,
                    facts_returned=1,
                    verification_status="exact_match",
                ),
            ],
        )
        
        # Should not raise
        print_eval_report(summary)
        
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
    
    def test_report_includes_all_sections(self, capsys):
        """Test that report includes all major sections."""
        # Note: accuracy values are stored as percentages (50.0 = 50%)
        summary = EvalSummary(
            total_questions=2,
            value_accuracy=50.0,
            period_accuracy=50.0,
            source_tier_accuracy=100.0,
            retrieval_failures=0,
            extraction_failures=1,
            verification_failures=0,
            hallucinations=0,
            pass_rate=50.0,
            results=[
                EvalResult(
                    question_id="q1",
                    question="Q1",
                    expected_value=100,
                    expected_period="Q1 2024",
                    actual_value=100,
                    actual_period="Q1 2024",
                    value_correct=True,
                    period_correct=True,
                    source_tier_correct=True,
                    error_type=None,
                    facts_returned=1,
                    verification_status="exact_match",
                ),
                EvalResult(
                    question_id="q2",
                    question="Q2",
                    expected_value=200,
                    expected_period="Q2 2024",
                    actual_value=None,
                    actual_period=None,
                    value_correct=False,
                    period_correct=False,
                    source_tier_correct=False,
                    error_type="extraction_failure",
                    facts_returned=0,
                    verification_status=None,
                ),
            ],
        )
        
        print_eval_report(summary)
        
        captured = capsys.readouterr()
        
        # Check all sections are present
        assert "EVALUATION REPORT" in captured.out
        assert "Total Questions: 2" in captured.out
        assert "Pass Rate: 50.0%" in captured.out
        assert "ACCURACY METRICS" in captured.out
        assert "Value Accuracy:" in captured.out
        assert "Period Accuracy:" in captured.out
        assert "ERROR BREAKDOWN" in captured.out
        assert "Retrieval Failures:" in captured.out
        assert "Extraction Failures:   1" in captured.out
        assert "DETAILED RESULTS" in captured.out
        assert "[✓] q1" in captured.out
        assert "[✗] q2 (extraction_failure)" in captured.out
    
    def test_report_shows_expected_vs_actual_for_failures(self, capsys):
        """Test that failed results show expected vs actual values."""
        summary = EvalSummary(
            total_questions=1,
            value_accuracy=0.0,
            period_accuracy=1.0,
            source_tier_accuracy=1.0,
            retrieval_failures=0,
            extraction_failures=0,
            verification_failures=1,
            hallucinations=0,
            pass_rate=0.0,
            results=[
                EvalResult(
                    question_id="q1",
                    question="Q1",
                    expected_value=100,
                    expected_period="Q1 2024",
                    actual_value=150,  # Wrong value
                    actual_period="Q1 2024",
                    value_correct=False,
                    period_correct=True,
                    source_tier_correct=True,
                    error_type="verification_failure",
                    facts_returned=1,
                    verification_status="mismatch",
                ),
            ],
        )
        
        print_eval_report(summary)
        
        captured = capsys.readouterr()
        # Values are floats so printed with decimal points
        assert "Expected: 100.0, Got: 150.0" in captured.out
    
    def test_report_shows_none_for_missing_values(self, capsys):
        """Test that missing values show as None."""
        summary = EvalSummary(
            total_questions=1,
            value_accuracy=0.0,
            period_accuracy=0.0,
            source_tier_accuracy=0.0,
            retrieval_failures=0,
            extraction_failures=1,
            verification_failures=0,
            hallucinations=0,
            pass_rate=0.0,
            results=[
                EvalResult(
                    question_id="q1",
                    question="Q1",
                    expected_value=100,
                    expected_period="Q1 2024",
                    actual_value=None,
                    actual_period=None,
                    value_correct=False,
                    period_correct=False,
                    source_tier_correct=False,
                    error_type="extraction_failure",
                    facts_returned=0,
                    verification_status=None,
                ),
            ],
        )
        
        print_eval_report(summary)
        
        captured = capsys.readouterr()
        # Values are floats so printed with decimal points
        assert "Expected: 100.0, Got: None" in captured.out

