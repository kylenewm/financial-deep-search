# Deep Research System: Architecture Part 3 - Gates, Storage, Verification

---

## 6. Gate System

### 6.1 Gate A: Tier 2 Verification

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             GATE A                                                               │
│                                    (Tier 2 HTML Extraction Verification)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Verify that LLM-extracted facts actually exist in the source HTML                                     │
│  FILE: pipeline.py → process_extracted_facts()                                                                  │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    THE CORE INVARIANT                                                       │ │
│  │                                                                                                             │ │
│  │  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ║   Every fact stored in FactStore from Tier 2 MUST have its sentence_string                            ║ │ │
│  │  ║   verified to exist in the original source HTML.                                                      ║ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ║   This is the ONLY barrier between LLM hallucination and stored "truth".                              ║ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝ │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    VERIFICATION FLOW                                                        │ │
│  │                                                                                                             │ │
│  │                       Fact (from LLM extraction)                                                            │ │
│  │                       verification_status = "unverified"                                                    │ │
│  │                                   │                                                                         │ │
│  │                                   ▼                                                                         │ │
│  │                      ┌─────────────────────────┐                                                            │ │
│  │                      │  Is fact numeric?       │                                                            │ │
│  │                      │  (has value field)      │                                                            │ │
│  │                      └───────────┬─────────────┘                                                            │ │
│  │                                  │                                                                          │ │
│  │            ┌─────────────────────┼─────────────────────┐                                                    │ │
│  │            │ YES                 │                     │ NO                                                 │ │
│  │            ▼                     │                     ▼                                                    │ │
│  │  ┌─────────────────────────┐     │          ┌─────────────────────────┐                                     │ │
│  │  │  NUMERIC VERIFICATION   │     │          │   TEXT VERIFICATION     │                                     │ │
│  │  │                         │     │          │                         │                                     │ │
│  │  │  1. Extract number from │     │          │  1. Normalize source:   │                                     │ │
│  │  │     sentence_string     │     │          │     - NFKC unicode      │                                     │ │
│  │  │     using regex         │     │          │     - collapse whitespace│                                    │ │
│  │  │                         │     │          │     - lowercase         │                                     │ │
│  │  │  2. Normalize both:     │     │          │                         │                                     │ │
│  │  │     - Handle "million", │     │          │  2. Normalize sentence: │                                     │ │
│  │  │       "billion", etc.   │     │          │     - Same as source    │                                     │ │
│  │  │     - Parse "$35.08B"   │     │          │                         │                                     │ │
│  │  │       as 35080000000    │     │          │  3. Check if normalized │                                     │ │
│  │  │                         │     │          │     sentence is         │                                     │ │
│  │  │  3. Compare:            │     │          │     substring of        │                                     │ │
│  │  │     extracted_value vs  │     │          │     normalized source   │                                     │ │
│  │  │     fact.value          │     │          │                         │                                     │ │
│  │  │                         │     │          │                         │                                     │ │
│  │  │  Tolerances:            │     │          │                         │                                     │ │
│  │  │  • ≤1% → exact_match    │     │          │  • Found → exact_match  │                                     │ │
│  │  │  • ≤5% → approx_match   │     │          │  • Not found → mismatch │                                     │ │
│  │  │  • >5% → mismatch       │     │          │                         │                                     │ │
│  │  └───────────┬─────────────┘     │          └───────────┬─────────────┘                                     │ │
│  │              │                   │                      │                                                   │ │
│  │              └───────────────────┼──────────────────────┘                                                   │ │
│  │                                  │                                                                          │ │
│  │                                  ▼                                                                          │ │
│  │                      ┌─────────────────────────┐                                                            │ │
│  │                      │  Update fact.           │                                                            │ │
│  │                      │  verification_status    │                                                            │ │
│  │                      └───────────┬─────────────┘                                                            │ │
│  │                                  │                                                                          │ │
│  │            ┌─────────────────────┼─────────────────────┐                                                    │ │
│  │            │                     │                     │                                                    │ │
│  │            ▼                     ▼                     ▼                                                    │ │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                                            │ │
│  │  │  exact_match    │   │ approx_match    │   │    mismatch     │                                            │ │
│  │  │                 │   │                 │   │                 │                                            │ │
│  │  │  → Can enter    │   │  → Can enter    │   │  → BLOCKED      │                                            │ │
│  │  │    FactStore    │   │    FactStore    │   │    from Store   │                                            │ │
│  │  │                 │   │    (flagged)    │   │                 │                                            │ │
│  │  │  Trust: HIGH    │   │  Trust: HIGH    │   │  → Logged as    │                                            │ │
│  │  │                 │   │  (with note)    │   │    rejection    │                                            │ │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────┘                                            │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    IMPLEMENTATION DETAILS                                                   │ │
│  │                                                                                                             │ │
│  │  Numeric extraction regex:                                                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ NUMBER_PATTERNS = [                                                                                 │   │ │
│  │  │     r'\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(trillion|billion|million|T|B|M|bn|mn)?',                  │   │ │
│  │  │     r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*%',  # percentages                                               │   │ │
│  │  │     r'(\d+(?:,\d{3})*(?:\.\d+)?)',      # plain numbers                                             │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  Scale normalization:                                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ SCALE_MAP = {                                                                                       │   │ │
│  │  │     'trillion': 1_000_000_000_000,                                                                  │   │ │
│  │  │     't': 1_000_000_000_000,                                                                         │   │ │
│  │  │     'billion': 1_000_000_000,                                                                       │   │ │
│  │  │     'b': 1_000_000_000,                                                                             │   │ │
│  │  │     'bn': 1_000_000_000,                                                                            │   │ │
│  │  │     'million': 1_000_000,                                                                           │   │ │
│  │  │     'm': 1_000_000,                                                                                 │   │ │
│  │  │     'mn': 1_000_000,                                                                                │   │ │
│  │  │ }                                                                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  Why 5% tolerance?                                                                                          │ │
│  │  • Rounding differences: "$35 billion" vs "35,082,000,000"                                                  │ │
│  │  • Unit ambiguity: "35" could be 35M or 35B depending on context                                            │ │
│  │  • LLM extraction may parse "approximately $35B" as exactly 35B                                             │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    REJECTION LOGGING                                                        │ │
│  │                                                                                                             │ │
│  │  Every rejected fact is logged with:                                                                        │ │
│  │  • fact_id                                                                                                  │ │
│  │  • metric                                                                                                   │ │
│  │  • extracted_value (from sentence_string)                                                                   │ │
│  │  • claimed_value (fact.value)                                                                               │ │
│  │  • difference_pct                                                                                           │ │
│  │  • sentence_string (for debugging)                                                                          │ │
│  │                                                                                                             │ │
│  │  Example log:                                                                                               │ │
│  │  [GATE_A_REJECT] fact_id=abc123 metric=revenue                                                              │ │
│  │                  extracted=35000000000 claimed=30000000000                                                  │ │
│  │                  diff=16.7% sentence="Revenue was $35 billion..."                                           │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE                                                                                                │
│  LIMITATIONS:                                                                                                   │
│  • Numeric extraction regex may miss complex formats                                                            │
│  • Cannot verify metric correctness (only value correctness)                                                    │
│  • 5% tolerance may be too loose for some use cases                                                             │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Gate B: Tier 3 Cross-Verification

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             GATE B                                                               │
│                                    (Tier 3 Web/News Verification)                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Verify web/news claims against authoritative SEC data                                                 │
│  FILE: cross_verify.py                                                                                          │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    VERIFICATION MODES                                                       │ │
│  │                                                                                                             │ │
│  │  MODE 1: TEXT PROVENANCE                                                                                    │ │
│  │  ─────────────────────────────                                                                              │ │
│  │  Same as Gate A: sentence_string must exist in source article.                                              │ │
│  │  Catches LLM hallucinations during news extraction.                                                         │ │
│  │                                                                                                             │ │
│  │  MODE 2: CROSS-VERIFICATION (Tier 3 vs Tier 1/2)                                                            │ │
│  │  ───────────────────────────────────────────────                                                            │ │
│  │  Compare news claims against SEC data in FactStore.                                                         │ │
│  │                                                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                                                                                                     │   │ │
│  │  │  News Claim: "NVDA reported $35B revenue"                                                           │   │ │
│  │  │       │                                                                                             │   │ │
│  │  │       ▼                                                                                             │   │ │
│  │  │  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐ │   │ │
│  │  │  │  Step 1: Parse claim                                                                          │ │   │ │
│  │  │  │  • Extract entity: "NVDA"                                                                     │ │   │ │
│  │  │  │  • Extract metric: "revenue"                                                                  │ │   │ │
│  │  │  │  • Extract value: 35,000,000,000                                                              │ │   │ │
│  │  │  └───────────────────────────────────────────────────────────────────────────────────────────────┘ │   │ │
│  │  │       │                                                                                             │   │ │
│  │  │       ▼                                                                                             │   │ │
│  │  │  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐ │   │ │
│  │  │  │  Step 2: Find matching SEC fact                                                               │ │   │ │
│  │  │  │  • Query FactStore for entity="NVDA", metric contains "revenue"                               │ │   │ │
│  │  │  │  • Handle metric aliases: revenue, revenues, sales, total revenue                             │ │   │ │
│  │  │  └───────────────────────────────────────────────────────────────────────────────────────────────┘ │   │ │
│  │  │       │                                                                                             │   │ │
│  │  │       ├──────────────────────────────────────────────────────────────────┐                          │   │ │
│  │  │       │ Found                                                            │ Not Found                │   │ │
│  │  │       ▼                                                                  ▼                          │   │ │
│  │  │  ┌────────────────────────────────────┐                   ┌────────────────────────────────────┐    │   │ │
│  │  │  │  Step 3: Compare values            │                   │  Return: "unverifiable"            │    │   │ │
│  │  │  │                                    │                   │                                    │    │   │ │
│  │  │  │  SEC fact: $30,082,000,000         │                   │  No SEC data to compare against.   │    │   │ │
│  │  │  │  News claim: $35,000,000,000       │                   │  News fact can still be stored     │    │   │ │
│  │  │  │  Difference: 16.3%                 │                   │  if it passes text provenance.     │    │   │ │
│  │  │  │                                    │                   │                                    │    │   │ │
│  │  │  │  16.3% > 5% → CONTRADICTED         │                   │  Trust level: MEDIUM               │    │   │ │
│  │  │  └────────────────────────────────────┘                   └────────────────────────────────────┘    │   │ │
│  │  │       │                                                                                             │   │ │
│  │  │       ▼                                                                                             │   │ │
│  │  │  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐ │   │ │
│  │  │  │  Verification Result:                                                                         │ │   │ │
│  │  │  │  {                                                                                            │ │   │ │
│  │  │  │    "status": "contradicted",                                                                  │ │   │ │
│  │  │  │    "hard_source": "SEC XBRL",                                                                 │ │   │ │
│  │  │  │    "hard_value": 30082000000,                                                                 │ │   │ │
│  │  │  │    "difference_pct": 16.3,                                                                    │ │   │ │
│  │  │  │    "confidence": 0.84,                                                                        │ │   │ │
│  │  │  │    "explanation": "Claim differs from SEC data by 16.3%"                                      │ │   │ │
│  │  │  │  }                                                                                            │ │   │ │
│  │  │  └───────────────────────────────────────────────────────────────────────────────────────────────┘ │   │ │
│  │  │                                                                                                     │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    VERIFICATION STATUS OUTCOMES                                             │ │
│  │                                                                                                             │ │
│  │  ┌────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ Status         │ Meaning                                                                             │  │ │
│  │  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤  │ │
│  │  │ verified       │ Claim matches SEC data within 1%. HIGH confidence.                                  │  │ │
│  │  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤  │ │
│  │  │ close          │ Claim matches SEC data within 5%. Likely rounding difference.                       │  │ │
│  │  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤  │ │
│  │  │ contradicted   │ Claim differs from SEC data by >5%. NEWS IS WRONG.                                  │  │ │
│  │  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤  │ │
│  │  │ unverifiable   │ No SEC data available to compare against. Store with caution.                       │  │ │
│  │  └────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    WHAT HAPPENS TO CONTRADICTED CLAIMS                                      │ │
│  │                                                                                                             │ │
│  │  Option 1 (Current): BLOCK from FactStore                                                                   │ │
│  │  • Contradicted claims are rejected                                                                         │ │
│  │  • Logged for debugging                                                                                     │ │
│  │  • Never shown to user as fact                                                                              │ │
│  │                                                                                                             │ │
│  │  Option 2 (Planned): SURFACE THE CONFLICT                                                                   │ │
│  │  • Store both claims                                                                                        │ │
│  │  • Narrator explicitly mentions disagreement:                                                               │ │
│  │    "Reuters reports $35B, but SEC filings show $30B"                                                        │ │
│  │  • User can investigate further                                                                             │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE (text provenance + cross-verification)                                                        │
│  LIMITATIONS:                                                                                                   │
│  • Multi-source corroboration NOT IMPLEMENTED                                                                   │
│  • Conflict surfacing NOT IMPLEMENTED (currently just blocks)                                                   │
│  • Domain trust ranking is soft, not hard gate                                                                  │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Storage Architecture

### 7.1 FactStore

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            FACTSTORE                                                             │
│                                    (Single Source of Truth for Facts)                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Store ONLY verified facts with full provenance                                                        │
│  FILE: store.py                                                                                                 │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    HARD INVARIANTS                                                          │ │
│  │                                                                                                             │ │
│  │  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ║  1. ONLY verification_status ∈ {"exact_match", "approximate_match"} can be stored                     ║ │ │
│  │  ║     → "unverified" → ValueError                                                                       ║ │ │
│  │  ║     → "mismatch" → ValueError                                                                         ║ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ║  2. EVERY fact MUST have a Location pointer                                                           ║ │ │
│  │  ║     → location=None → ValueError                                                                      ║ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ║  3. Location MUST have durable source reference                                                       ║ │ │
│  │  ║     → location.cik=None AND location.article_url=None → ValueError                                    ║ │ │
│  │  ║                                                                                                       ║ │ │
│  │  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝ │ │
│  │                                                                                                             │ │
│  │  These invariants are ENFORCED IN CODE. The FactStore is physically incapable of accepting                 │ │
│  │  unverified or untraceable facts.                                                                           │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    DATA MODEL                                                               │ │
│  │                                                                                                             │ │
│  │  class Fact(BaseModel):                                                                                     │ │
│  │      # Core identification                                                                                  │ │
│  │      fact_id: str                    # UUID, globally unique                                                │ │
│  │      entity: str                     # Ticker symbol (e.g., "AAPL")                                         │ │
│  │                                                                                                             │ │
│  │      # Fact content                                                                                         │ │
│  │      metric: str                     # e.g., "Revenue", "Net Income"                                        │ │
│  │      value: Optional[float]          # Numeric value (None for qualitative)                                 │ │
│  │      unit: str                       # "USD", "shares", "percent"                                           │ │
│  │      period: str                     # e.g., "Q3 FY2025", "FY2024"                                          │ │
│  │      period_end_date: Optional[str]  # ISO date, e.g., "2024-10-27"                                         │ │
│  │                                                                                                             │ │
│  │      # Provenance                                                                                           │ │
│  │      location: Location              # Full source pointer (see below)                                      │ │
│  │      source_format: str              # "xbrl", "html", "news"                                               │ │
│  │      source_tier: Literal["tier1", "tier2", "tier3"]                                                        │ │
│  │      trust_level: Literal["high", "medium", "low"]                                                          │ │
│  │                                                                                                             │ │
│  │      # Verification                                                                                         │ │
│  │      verification_status: str        # "exact_match", "approximate_match", "unverified", "mismatch"         │ │
│  │      extraction_method: Optional[str]  # "toc_anchor", "dom_scan", "xbrl_parse"                             │ │
│  │      extraction_confidence: Optional[str]  # "HIGH", "MEDIUM", "LOW"                                        │ │
│  │                                                                                                             │ │
│  │      # Hashes for integrity                                                                                 │ │
│  │      doc_hash: Optional[str]         # SHA256 of source document                                            │ │
│  │      snapshot_id: Optional[str]      # Unique ID for this extraction run                                    │ │
│  │                                                                                                             │ │
│  │                                                                                                             │ │
│  │  class Location(BaseModel):                                                                                 │ │
│  │      # SEC filing pointer                                                                                   │ │
│  │      cik: Optional[str]              # SEC CIK (e.g., "0000320193")                                         │ │
│  │      doc_date: Optional[str]         # Filing date                                                          │ │
│  │      doc_type: Optional[str]         # "10-K", "10-Q", "8-K", "news"                                        │ │
│  │      accession_number: Optional[str] # SEC accession number                                                 │ │
│  │                                                                                                             │ │
│  │      # Position within document                                                                             │ │
│  │      section_id: Optional[str]       # "Item1A", "Item7"                                                    │ │
│  │      paragraph_index: Optional[int]  # Which paragraph                                                      │ │
│  │      sentence_string: Optional[str]  # EXACT sentence containing fact                                       │ │
│  │                                                                                                             │ │
│  │      # Table position (for tabular data)                                                                    │ │
│  │      table_index: Optional[int]                                                                             │ │
│  │      row_index: Optional[int]                                                                               │ │
│  │      column_index: Optional[int]                                                                            │ │
│  │      row_label: Optional[str]                                                                               │ │
│  │      column_label: Optional[str]                                                                            │ │
│  │                                                                                                             │ │
│  │      # News article pointer                                                                                 │ │
│  │      article_url: Optional[str]      # Durable URL                                                          │ │
│  │      article_title: Optional[str]                                                                           │ │
│  │      article_domain: Optional[str]                                                                          │ │
│  │      article_published_date: Optional[str]                                                                  │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    QUERY METHODS                                                            │ │
│  │                                                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ Method                          │ Description                                                       │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ get_fact(fact_id)               │ Get single fact by ID                                             │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ get_facts_by_entity(ticker)     │ All facts for a company                                           │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ get_facts_by_metric(metric)     │ All facts for a metric (e.g., "revenue")                          │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ get_facts_by_period(period)     │ All facts for a time period                                       │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ get_all_facts()                 │ Everything in the store                                           │   │ │
│  │  ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────┤   │ │
│  │  │ find_conflicts()                │ Facts where same metric has different values                      │   │ │
│  │  └─────────────────────────────────┴───────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    CONFLICT DETECTION                                                       │ │
│  │                                                                                                             │ │
│  │  find_conflicts() identifies facts where:                                                                   │ │
│  │  • Same entity (ticker)                                                                                     │ │
│  │  • Same metric                                                                                              │ │
│  │  • Same period                                                                                              │ │
│  │  • BUT different values (>1% difference)                                                                    │ │
│  │                                                                                                             │ │
│  │  Example conflict:                                                                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ Conflict(                                                                                           │   │ │
│  │  │   entity="AAPL",                                                                                    │   │ │
│  │  │   metric="revenue",                                                                                 │   │ │
│  │  │   period="FY2024",                                                                                  │   │ │
│  │  │   values=[                                                                                          │   │ │
│  │  │     ConflictingValue(value=391000000000, fact_id="abc", source="XBRL FY2024"),                      │   │ │
│  │  │     ConflictingValue(value=395000000000, fact_id="def", source="10-K Item 7"),                      │   │ │
│  │  │   ]                                                                                                 │   │ │
│  │  │ )                                                                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  Narrator behavior on conflict:                                                                             │ │
│  │  • Prefer Tier 1 over Tier 2 over Tier 3                                                                    │ │
│  │  • If same tier conflicts, mention both values                                                              │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE (in-memory)                                                                                    │
│  LIMITATIONS:                                                                                                   │
│  • In-memory only (no persistence across restarts)                                                              │
│  • No indexing (O(n) queries)                                                                                   │
│  • No versioning/history                                                                                        │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 SignalStore

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           SIGNALSTORE                                                            │
│                                    (Append-Only Signal Persistence)                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Persist semantic drift signals for backtesting and analysis                                           │
│  FILE: store.py → SignalStore                                                                                   │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    DESIGN                                                                   │ │
│  │                                                                                                             │ │
│  │  Format: JSONL (one JSON object per line)                                                                   │ │
│  │  Location: .cache/signals/signals.jsonl                                                                     │ │
│  │                                                                                                             │ │
│  │  Why JSONL?                                                                                                 │ │
│  │  • Append-only (no locking issues)                                                                          │ │
│  │  • Trivial Pandas loading: pd.read_json(path, lines=True)                                                   │ │
│  │  • Streaming reads for large datasets                                                                       │ │
│  │  • Easy inspection with command-line tools                                                                  │ │
│  │  • Simple SQLite migration later                                                                            │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    SIGNAL RECORD MODEL                                                      │ │
│  │                                                                                                             │ │
│  │  class SignalRecord(BaseModel):                                                                             │ │
│  │      # Identification                                                                                       │ │
│  │      signal_id: str                  # UUID                                                                 │ │
│  │      ticker: str                     # e.g., "NVDA"                                                         │ │
│  │      signal_mode: str                # "regime", "event", "quarterly"                                       │ │
│  │                                                                                                             │ │
│  │      # Timing                                                                                               │ │
│  │      filing_date: str                # When filing was released                                             │ │
│  │      period_end_date: str            # Period the filing covers                                             │ │
│  │      created_at: str                 # When signal was generated                                            │ │
│  │                                                                                                             │ │
│  │      # Drift metrics                                                                                        │ │
│  │      drift_score: float              # 0-100 (higher = more change)                                         │ │
│  │      new_sentence_count: int         # Sentences not in previous filing                                     │ │
│  │      removed_sentence_count: int     # Sentences removed from previous                                      │ │
│  │      new_keyword_count: int          # New risk keywords                                                    │ │
│  │                                                                                                             │ │
│  │      # Quality flags                                                                                        │ │
│  │      boilerplate_flag: bool          # True if mostly boilerplate                                           │ │
│  │      extraction_confidence: str      # "HIGH", "MEDIUM", "LOW"                                              │ │
│  │                                                                                                             │ │
│  │      # Classification                                                                                       │ │
│  │      severity: str                   # "critical", "moderate", "low"                                        │ │
│  │                                                                                                             │ │
│  │      # Detailed results (optional)                                                                          │ │
│  │      drift_results: Optional[List[DriftResult]]  # Per-paragraph analysis                                   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    INVARIANT: LOW CONFIDENCE SIGNALS NOT PERSISTED                         │ │
│  │                                                                                                             │ │
│  │  If extraction_confidence == "LOW", the signal is NOT written to the store.                                 │ │
│  │                                                                                                             │ │
│  │  This prevents:                                                                                             │ │
│  │  • Storing signals from failed/incomplete extractions                                                       │ │
│  │  • Polluting the signal history with unreliable data                                                        │ │
│  │  • False positives in backtesting                                                                           │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    QUERY METHODS                                                            │ │
│  │                                                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ Method                              │ Description                                                   │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ append(record)                      │ Add new signal                                                │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ load_all()                          │ Get all signals                                               │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ query_by_ticker(ticker)             │ Signals for a company                                         │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ query_by_date_range(start, end)     │ Signals in date range (for backtesting)                       │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ query_high_drift(threshold)         │ Signals above drift threshold                                 │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ query_non_boilerplate()             │ Filter out boilerplate signals                                │   │ │
│  │  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤   │ │
│  │  │ to_dataframe()                      │ Load as Pandas DataFrame                                      │   │ │
│  │  └─────────────────────────────────────┴───────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  RANKING VIEWS (Exploratory, NOT Trading Signals):                                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ • rank_by_drift(top_n)              │ Highest drift scores                                          │   │ │
│  │  │ • rank_by_novelty(top_n)            │ Most new sentences                                            │   │ │
│  │  │ • rank_by_keyword_hits(top_n)       │ Most new risk keywords                                        │   │ │
│  │  │ • rank_composite(weights)           │ Weighted combination                                          │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  ⚠️ WARNING: Rankings are VIEWS, not TRUTH. Not validated against returns.                                 │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE                                                                                                │
│  LIMITATIONS:                                                                                                   │
│  • No real-time streaming                                                                                       │
│  • Rankings are heuristic, not validated                                                                        │
│  • No deduplication                                                                                             │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```


