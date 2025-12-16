#!/usr/bin/env python3
"""
E2E Test for Apple (AAPL) - Full Pipeline Validation

Run with:
    cd <project_root>
    source .venv/bin/activate
    PYTHONPATH=./src python scripts/e2e_apple_test.py
"""
import logging
import os

# Enable logging so you can see API calls and extraction steps
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv("open_deep_research/.env")

from open_deep_research.orchestrator import Orchestrator


def main():
    orc = Orchestrator()
    
    # ========================================
    # SETUP CHECK
    # ========================================
    print("=" * 60)
    print("SETUP CHECK")
    print("=" * 60)
    
    status = orc.setup_check()
    print(status["summary"])
    
    for cap, available in status["capabilities"].items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {cap}")
    
    if not status["ready"]:
        print("\n❌ BLOCKING ISSUES:")
        for b in status["blocking"]:
            print(f"  • {b}")
        print("\nFix blocking issues and re-run.")
        return
    
    # ========================================
    # TIER 1: XBRL FACTS
    # ========================================
    print("\n" + "=" * 60)
    print("TIER 1: XBRL FACTS")
    print("=" * 60)
    
    print("\nLoading AAPL FY2024 facts...")
    count = orc.load_facts_for_entity("AAPL", "FY2024")
    print(f"✅ Loaded {count} facts")
    
    print("\nSample facts:")
    for fact in orc.fact_store.get_all_facts()[:5]:
        print(f"  • {fact.metric}: ${fact.value:,.0f} (tier: {fact.source_tier})")
    
    print("\nAsking: 'What was Apple's revenue in FY2024?'")
    report = orc.ask("What was Apple's revenue in FY2024?")
    
    print(f"\nAnswer:\n{report.answer[:600]}")
    print(f"\n--- Metadata ---")
    print(f"  Facts used: {len(report.facts_used)}")
    print(f"  Citations: {len(report.citations)}")
    print(f"  insufficient_data: {report.insufficient_data}")
    
    # Invariant checks
    if not report.insufficient_data:
        assert len(report.facts_used) > 0, "FAIL: No facts used!"
        for f in report.facts_used:
            assert f.verification_status == "exact_match", f"FAIL: Unverified fact!"
            assert f.source_tier == "tier1", f"FAIL: Not tier1!"
        print("\n✅ TIER 1 INVARIANTS PASSED")
    else:
        print("\n⚠️ Returned insufficient_data (may need narrator)")
    
    # ========================================
    # TIER 2: QUALITATIVE EXTRACTION
    # ========================================
    print("\n" + "=" * 60)
    print("TIER 2: QUALITATIVE EXTRACTION")
    print("=" * 60)
    
    print("\nAsking: 'What are the main risks for Apple?'")
    report = orc.ask("What are the main risks for Apple?")
    
    print(f"\nAnswer:\n{report.answer[:600]}")
    print(f"\n--- Metadata ---")
    print(f"  Facts used: {len(report.facts_used)}")
    print(f"  insufficient_data: {report.insufficient_data}")
    
    # Valid outcomes: either has cited facts OR fails closed
    if report.insufficient_data:
        print("\n✅ Correctly failed closed (no verified qualitative facts)")
    else:
        assert len(report.citations) > 0, "FAIL: Answer without citations!"
        print("\n✅ TIER 2: Returned cited answer")
    
    # ========================================
    # TIER 3: SIGNAL DETECTION
    # ========================================
    print("\n" + "=" * 60)
    print("TIER 3: SIGNAL DETECTION")
    print("=" * 60)
    
    print("\nAsking: 'Any red flags in Apple's latest filing?'")
    report = orc.ask("Any red flags in Apple's latest filing?")
    
    print(f"\nAnswer:\n{report.answer[:600]}")
    print(f"\n--- Metadata ---")
    print(f"  insufficient_data: {report.insufficient_data}")
    
    # Check for transparency markers
    answer_lower = report.answer.lower()
    has_method = any(m in answer_lower for m in ["toc_anchor", "dom_scan", "fuzzy", "extraction"])
    has_disclaimer = "signal" in answer_lower or "artifact" in answer_lower
    
    if has_method or has_disclaimer:
        print("\n✅ TIER 3: Shows extraction method or disclaimer")
    else:
        print("\n⚠️ No extraction method shown (check if signal detection ran)")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("E2E TEST COMPLETE")
    print("=" * 60)
    print("\nAll tiers tested. Check output above for any failures.")


if __name__ == "__main__":
    main()

