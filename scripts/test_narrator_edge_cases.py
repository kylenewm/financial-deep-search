#!/usr/bin/env python3
"""
Narrator Edge Case Stress Test

This script tests the Narrator's behavior in complex and edge case scenarios
by manually populating the FactStore with specific test data (or lack thereof).

Scenarios:
1. Comparison (NVDA vs AAPL)
2. Irrelevant Data (Distractors)
3. Insufficient Data (Missing metric)
4. Multi-period synthesis
5. Hallucination attempt (Ask for something not there)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from open_deep_research.models import Fact, Location
from open_deep_research.narrator import generate_report
from open_deep_research.store import FactStore


def create_mock_fact(
    entity: str,
    metric: str,
    value: float,
    period: str,
    fact_id: str,
) -> Fact:
    """Helper to create a mock fact."""
    return Fact(
        fact_id=fact_id,
        entity=entity,
        metric=metric,
        value=value,
        unit="USD",
        period=period,
        period_end_date="2024-01-01",
        location=Location(
            cik="0000000000",
            doc_date="2024-01-01",
            doc_type="10-Q",
            section_id="mock",
        ),
        source_format="xbrl",
        doc_hash="mock",
        snapshot_id="mock",
        verification_status="exact_match",
    )


def run_test_case(name: str, query: str, facts: list[Fact]):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"TEST CASE: {name}")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Facts provided: {len(facts)}")
    for f in facts:
        print(f"  - {f.entity} {f.metric} {f.period}: ${f.value:,.0f}")
    
    # Create store
    store = FactStore()
    for f in facts:
        store.add_fact(f)
    
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[SKIPPED] ANTHROPIC_API_KEY not set")
        return

    try:
        report = generate_report(query, store)
        
        print(f"\n📝 Narrated Answer:\n{report.answer}")
        
        print(f"\n📚 Citations:")
        for c in report.citations:
            print(f"  [{c.citation_index}] {c.source_format}: {c.location}")
            
        print(f"\n⚠️ Insufficient Data: {report.insufficient_data}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    print("Starting Narrator Stress Test...")
    
    # 1. Comparison Test
    facts_comparison = [
        create_mock_fact("NVDA", "Total Revenue", 35082000000, "Q3 FY2025", "f1"),
        create_mock_fact("AAPL", "Total Revenue", 94930000000, "Q4 FY2024", "f2"),
    ]
    run_test_case(
        "Comparison (NVDA vs AAPL)",
        "Compare NVIDIA's Q3 FY2025 revenue with Apple's Q4 FY2024 revenue. Which is higher?",
        facts_comparison
    )
    
    # 2. Irrelevant Data (Distractors)
    facts_distractor = [
        create_mock_fact("NVDA", "Total Revenue", 35082000000, "Q3 FY2025", "f1"),
        create_mock_fact("NVDA", "Net Income", 19309000000, "Q3 FY2025", "f2"),
        create_mock_fact("INTC", "Revenue", 13000000000, "Q3 2024", "f3"),
    ]
    run_test_case(
        "Distractors (Specific Metric)",
        "What was NVIDIA's Net Income in Q3 FY2025? Ignore revenue.",
        facts_distractor
    )
    
    # 3. Insufficient Data
    facts_insufficient = [
        create_mock_fact("NVDA", "Total Revenue", 35082000000, "Q3 FY2025", "f1"),
    ]
    run_test_case(
        "Insufficient Data (Missing Metric)",
        "What was NVIDIA's Operating Income in Q3 FY2025?",
        facts_insufficient
    )
    
    # 4. Multi-Fact Synthesis (Margin)
    facts_margin = [
        create_mock_fact("NVDA", "Total Revenue", 35082000000, "Q3 FY2025", "f1"),
        create_mock_fact("NVDA", "Operating Income", 21869000000, "Q3 FY2025", "f2"),
    ]
    run_test_case(
        "Synthesis (Operating Margin)",
        "What was NVIDIA's operating margin in Q3 FY2025? (Calculate from income/revenue)",
        facts_margin
    )
    
    # 5. Hallucination Trap
    facts_hallucination = [
        create_mock_fact("NVDA", "Total Revenue", 35082000000, "Q3 FY2025", "f1"),
    ]
    run_test_case(
        "Hallucination Trap",
        "What guidance did NVIDIA provide for Q4?",
        facts_hallucination
    )


if __name__ == "__main__":
    main()

