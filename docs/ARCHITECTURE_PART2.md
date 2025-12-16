# Deep Research System: Architecture Part 2 - Component Deep Dives

---

## 4. Component Deep Dives

### 4.1 SEC EDGAR Integration Layer

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        SEC EDGAR INTEGRATION                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Fetch official SEC filings with proper rate limiting and caching                                      │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                         MODULES                                                             │ │
│  │                                                                                                             │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐  │ │
│  │  │    ingestion.py     │    │      xbrl.py        │    │    cik_lookup.py    │    │   sec_session.py    │  │ │
│  │  │                     │    │                     │    │                     │    │                     │  │ │
│  │  │  Filing downloads   │    │  XBRL fact parsing  │    │  Ticker → CIK       │    │  Rate-limited       │  │ │
│  │  │  via sec-edgar-     │    │  via edgartools     │    │  resolution         │    │  HTTP session       │  │ │
│  │  │  downloader         │    │                     │    │                     │    │                     │  │ │
│  │  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘  │ │
│  │             │                          │                          │                          │             │ │
│  │             └──────────────────────────┼──────────────────────────┼──────────────────────────┘             │ │
│  │                                        │                          │                                        │ │
│  │                                        ▼                          ▼                                        │ │
│  │                           ┌─────────────────────────────────────────────────────┐                          │ │
│  │                           │              SEC EDGAR API                          │                          │ │
│  │                           │                                                     │                          │ │
│  │                           │  Endpoints:                                         │                          │ │
│  │                           │  • /cgi-bin/browse-edgar (filing search)           │                          │ │
│  │                           │  • /Archives/edgar/data/{cik}/ (filing content)    │                          │ │
│  │                           │  • /cik-lookup-data.txt (ticker resolution)        │                          │ │
│  │                           │  • /submissions/CIK{cik}.json (company metadata)   │                          │ │
│  │                           │                                                     │                          │ │
│  │                           │  Rate Limits:                                       │                          │ │
│  │                           │  • 10 requests/second (enforced by SEC)            │                          │ │
│  │                           │  • User-Agent REQUIRED (legal requirement)         │                          │ │
│  │                           │                                                     │                          │ │
│  │                           └─────────────────────────────────────────────────────┘                          │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    CONFIGURATION                                                            │ │
│  │                                                                                                             │ │
│  │  REQUIRED ENV VARS:                                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ SEC_USER_AGENT="MyApp/1.0 (contact@example.com)"                                                    │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ REQUIRED BY SEC. Without this:                                                                      │   │ │
│  │  │ • Requests will be blocked (403)                                                                    │   │ │
│  │  │ • Tier 1 and Tier 2 are COMPLETELY UNAVAILABLE                                                      │   │ │
│  │  │ • System degrades to Tier 3 only (web search)                                                       │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  TIMEOUTS:                                                                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ • HTTP requests: 60 seconds (configurable)                                                          │   │ │
│  │  │ • XBRL parsing: 30 seconds per filing                                                               │   │ │
│  │  │ • Total pipeline: 120 seconds (fail closed after)                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    DATA FLOW                                                                │ │
│  │                                                                                                             │ │
│  │   ticker ("AAPL")                                                                                           │ │
│  │        │                                                                                                    │ │
│  │        ▼                                                                                                    │ │
│  │   ┌─────────────────┐                                                                                       │ │
│  │   │  cik_lookup.py  │ ─────────────────────────────────────────────────────────────────────────────────┐    │ │
│  │   │  resolve_cik()  │                                                                                  │    │ │
│  │   └────────┬────────┘                                                                                  │    │ │
│  │            │                                                                                           │    │ │
│  │            ▼ CIK ("0000320193")                                                                        │    │ │
│  │   ┌─────────────────┐                                                                                  │    │ │
│  │   │  ingestion.py   │                                                                                  │    │ │
│  │   │ get_filings()   │ ──► Returns list of Filing objects with:                                         │    │ │
│  │   └────────┬────────┘     • accession_number                                                           │    │ │
│  │            │              • filing_date                                                                │    │ │
│  │            │              • form_type (10-K, 10-Q, 8-K)                                                │    │ │
│  │            │              • raw_html (full filing HTML)                                                │    │ │
│  │            │              • xbrl_url (if available)                                                    │    │ │
│  │            │                                                                                           │    │ │
│  │            ├───────────────────────────────────────────────────────────────────────────────────────────┘    │ │
│  │            │                                                                                                │ │
│  │            ▼                                                                                                │ │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │   │                                TWO PARALLEL PATHS                                                   │   │ │
│  │   │                                                                                                     │   │ │
│  │   │   PATH A: XBRL (Tier 1)                      PATH B: HTML (Tier 2)                                 │   │ │
│  │   │   ─────────────────────                      ──────────────────────                                │   │ │
│  │   │                                                                                                     │   │ │
│  │   │   Filing.xbrl_url                            Filing.raw_html                                        │   │ │
│  │   │        │                                          │                                                 │   │ │
│  │   │        ▼                                          ▼                                                 │   │ │
│  │   │   ┌─────────────┐                            ┌─────────────┐                                        │   │ │
│  │   │   │  xbrl.py    │                            │ section_    │                                        │   │ │
│  │   │   │ parse_xbrl()│                            │ locator.py  │                                        │   │ │
│  │   │   └──────┬──────┘                            └──────┬──────┘                                        │   │ │
│  │   │          │                                          │                                               │   │ │
│  │   │          ▼                                          ▼                                               │   │ │
│  │   │   Structured XBRL facts:                      Extracted section:                                    │   │ │
│  │   │   • us-gaap:Revenues                          • Item 1A (Risk Factors)                              │   │ │
│  │   │   • us-gaap:NetIncomeLoss                     • Item 7 (MD&A)                                       │   │ │
│  │   │   • us-gaap:Assets                            • Clean text, no HTML                                 │   │ │
│  │   │   • Exact values, no ambiguity                • Preserves paragraph structure                       │   │ │
│  │   │          │                                          │                                               │   │ │
│  │   │          ▼                                          ▼                                               │   │ │
│  │   │   FactStore (tier1)                           LLM Extraction → Gate A → FactStore (tier2)           │   │ │
│  │   │                                                                                                     │   │ │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE                                                                                                │
│  LIMITATIONS:                                                                                                   │
│  • No caching layer (re-fetches on every call)                                                                  │
│  • No filing archive/persistence                                                                                │
│  • XBRL parsing can be slow for large filings                                                                   │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Section Locator (Deterministic HTML Extraction)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        SECTION LOCATOR                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Deterministically extract specific sections from SEC filing HTML                                      │
│  FILE: section_locator.py                                                                                       │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    WHY DETERMINISTIC?                                                       │ │
│  │                                                                                                             │ │
│  │  LLMs cannot reliably extract long sections from HTML:                                                      │ │
│  │  • Context window limits (100K tokens still not enough for full 10-K)                                      │ │
│  │  • Hallucination risk when asked to "find Item 1A"                                                         │ │
│  │  • Cost prohibitive to send full filing to LLM                                                             │ │
│  │                                                                                                             │ │
│  │  Solution: Rule-based extraction with multiple fallback strategies                                         │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    EXTRACTION STRATEGIES                                                    │ │
│  │                                                                                                             │ │
│  │  STRATEGY 1: Table of Contents Anchor                                                                       │ │
│  │  ────────────────────────────────────────────                                                               │ │
│  │                                                                                                             │ │
│  │  Many filings have a ToC with links to sections:                                                            │ │
│  │                                                                                                             │ │
│  │  <a href="#item1a">Item 1A. Risk Factors</a>                                                                │ │
│  │      ...                                                                                                    │ │
│  │  <a name="item1a"></a>                                                                                      │ │
│  │  <h2>Item 1A. Risk Factors</h2>                                                                             │ │
│  │  <p>Our business faces significant risks...</p>                                                             │ │
│  │                                                                                                             │ │
│  │  Algorithm:                                                                                                 │ │
│  │  1. Find <a> tags with href containing "item1a" (case-insensitive)                                         │ │
│  │  2. Extract anchor target (e.g., "#item1a")                                                                 │ │
│  │  3. Find <a name="item1a"> or <div id="item1a">                                                             │ │
│  │  4. Collect all text until next section header                                                              │ │
│  │                                                                                                             │ │
│  │  Success rate: ~70% of filings                                                                              │ │
│  │                                                                                                             │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │ │
│  │                                                                                                             │ │
│  │  STRATEGY 2: DOM Scan with Header Detection                                                                 │ │
│  │  ──────────────────────────────────────────────                                                             │ │
│  │                                                                                                             │ │
│  │  For filings without proper ToC anchors:                                                                    │ │
│  │                                                                                                             │ │
│  │  Algorithm:                                                                                                 │ │
│  │  1. Find all header-like elements (h1, h2, h3, strong, b, div with large font)                             │ │
│  │  2. Check each for Item 1A patterns:                                                                        │ │
│  │     • "Item 1A" or "Item1A"                                                                                 │ │
│  │     • "Risk Factors"                                                                                        │ │
│  │     • Variations: "ITEM 1A.", "Item 1A -", etc.                                                             │ │
│  │  3. Collect all sibling/following text                                                                      │ │
│  │  4. Stop when hitting next section header (Item 1B, Item 2, etc.)                                          │ │
│  │                                                                                                             │ │
│  │  Success rate: ~25% of filings (catches most ToC failures)                                                  │ │
│  │                                                                                                             │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │ │
│  │                                                                                                             │ │
│  │  STRATEGY 3: Full Text Regex (Last Resort)                                                                  │ │
│  │  ─────────────────────────────────────────────                                                              │ │
│  │                                                                                                             │ │
│  │  For edge cases (malformed HTML, unusual formatting):                                                       │ │
│  │                                                                                                             │ │
│  │  Algorithm:                                                                                                 │ │
│  │  1. Strip all HTML tags → plain text                                                                        │ │
│  │  2. Regex search for "Item 1A" header                                                                       │ │
│  │  3. Extract text until "Item 1B" or "Item 2"                                                                │ │
│  │  4. Risk: May include navigation/footer text                                                                │ │
│  │                                                                                                             │ │
│  │  Success rate: ~5% (rare fallback)                                                                          │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    PATTERN DEFINITIONS                                                      │ │
│  │                                                                                                             │ │
│  │  ITEM 1A (Risk Factors):                                                                                    │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ ITEM_1A_PATTERNS = [                                                                                │   │ │
│  │  │     re.compile(r"item\s*1a\.?\s*[-–—]?\s*risk\s*factors?", re.IGNORECASE),                          │   │ │
│  │  │     re.compile(r"item\s*1a\.?(?!\s*\(?\s*continued)", re.IGNORECASE),                               │   │ │
│  │  │     re.compile(r"risk\s*factors?", re.IGNORECASE),                                                  │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ NEXT_ITEM_PATTERNS = [  # Stop patterns                                                             │   │ │
│  │  │     re.compile(r"\bitem\s*1b\.?(?!\s*\(?\s*continued)", re.IGNORECASE),                             │   │ │
│  │  │     re.compile(r"\bitem\s*2\.?(?!\s*\(?\s*continued)", re.IGNORECASE),                              │   │ │
│  │  │     re.compile(r"\bunresolved\s*staff\s*comments", re.IGNORECASE),                                  │   │ │
│  │  │     re.compile(r"\bproperties\b", re.IGNORECASE),                                                   │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ CONTINUATION_PATTERNS = [  # Don't stop on these                                                    │   │ │
│  │  │     re.compile(r"continued", re.IGNORECASE),                                                        │   │ │
│  │  │     re.compile(r"\(cont(?:inued|'d)\)", re.IGNORECASE),                                             │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  ITEM 7 (MD&A):                                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ ITEM_7_PATTERNS = [                                                                                 │   │ │
│  │  │     re.compile(r"item\s*7\.?\s*[-–—]?\s*management", re.IGNORECASE),                                │   │ │
│  │  │     re.compile(r"item\s*7\.?(?!\s*a)", re.IGNORECASE),  # Item 7 but NOT 7A                         │   │ │
│  │  │     re.compile(r"management['']?s?\s+discussion\s+and\s+analysis", re.IGNORECASE),                  │   │ │
│  │  │     re.compile(r"\bmd&a\b", re.IGNORECASE),                                                         │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ ITEM_7_STOP_PATTERNS = [                                                                            │   │ │
│  │  │     re.compile(r"\bitem\s*7a\.?", re.IGNORECASE),                                                   │   │ │
│  │  │     re.compile(r"\bitem\s*8\.?", re.IGNORECASE),                                                    │   │ │
│  │  │     re.compile(r"financial\s+statements\s+and\s+supplementary", re.IGNORECASE),                     │   │ │
│  │  │     re.compile(r"quantitative\s+and\s+qualitative\s+disclosures", re.IGNORECASE),                   │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    OUTPUT FORMAT                                                            │ │
│  │                                                                                                             │ │
│  │  @dataclass                                                                                                 │ │
│  │  class RiskFactorsExtract:                                                                                  │ │
│  │      """Result of section extraction."""                                                                    │ │
│  │      text: str                      # Clean extracted text                                                  │ │
│  │      paragraphs: List[str]          # Split into paragraphs                                                 │ │
│  │      source_html_hash: str          # SHA256 of source for provenance                                       │ │
│  │      extraction_method: str         # "toc_anchor" | "dom_scan" | "regex"                                   │ │
│  │      confidence: str                # "HIGH" | "MEDIUM" | "LOW"                                             │ │
│  │      word_count: int                # For validation                                                        │ │
│  │      warnings: List[str]            # Any issues detected                                                   │ │
│  │                                                                                                             │ │
│  │  Confidence levels:                                                                                         │ │
│  │  • HIGH: ToC anchor found, clean extraction, >1000 words                                                    │ │
│  │  • MEDIUM: DOM scan worked, reasonable length                                                               │ │
│  │  • LOW: Regex fallback, or suspiciously short (<500 words)                                                  │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    VALIDATION RULES                                                         │ │
│  │                                                                                                             │ │
│  │  MIN_CHARS = 2000          # Reject extractions shorter than this                                           │ │
│  │  MAX_CHARS = 500000        # Reject if suspiciously long (likely whole filing)                              │ │
│  │  BOILERPLATE_THRESHOLD = 0.3  # If >30% boilerplate, flag as warning                                        │ │
│  │                                                                                                             │ │
│  │  Text normalization:                                                                                        │ │
│  │  • Unicode NFKC normalization (handles special characters)                                                  │ │
│  │  • Whitespace collapsing                                                                                    │ │
│  │  • Smart quote normalization                                                                                │ │
│  │  • HTML entity decoding                                                                                     │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE (Item 1A, Item 7)                                                                              │
│  LIMITATIONS:                                                                                                   │
│  • No Item 1, Item 2, Item 8 support yet                                                                        │
│  • Some unusual filing formats may fail                                                                         │
│  • No table extraction (tables are stripped)                                                                    │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Section Router (Semantic Classification)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        SECTION ROUTER                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Determine which SEC filing section to extract based on user query                                     │
│  FILE: classifier.py → classify_section()                                                                       │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    DESIGN PHILOSOPHY                                                        │ │
│  │                                                                                                             │ │
│  │  "Semantic suggestion, deterministic decision"                                                              │ │
│  │                                                                                                             │ │
│  │  • Use embeddings to SCORE compatibility with each section                                                  │ │
│  │  • Apply HARD GATES to prevent ambiguous or low-confidence routing                                         │ │
│  │  • FAIL CLOSED if conditions not met                                                                        │ │
│  │  • Log all decisions for auditability                                                                       │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    SECTION REFERENCE EMBEDDINGS                                             │ │
│  │                                                                                                             │ │
│  │  Each section has reference phrases that capture its semantic intent:                                       │ │
│  │                                                                                                             │ │
│  │  SECTION_REFS = {                                                                                           │ │
│  │      "Item1A": [                                                                                            │ │
│  │          "What are the risks facing the company?",                                                          │ │
│  │          "What uncertainties could affect the business?",                                                   │ │
│  │          "Describe potential threats or concerns",                                                          │ │
│  │          "What are the risk factors?",                                                                      │ │
│  │          "What could go wrong?",                                                                            │ │
│  │      ],                                                                                                     │ │
│  │      "Item7": [                                                                                             │ │
│  │          "How did revenue perform this year?",                                                              │ │
│  │          "Discuss financial results and margins",                                                           │ │
│  │          "What were operating expenses?",                                                                   │ │
│  │          "Analyze the company's financial performance",                                                     │ │
│  │          "What was net income?",                                                                            │ │
│  │          "How did sales compare to last year?",                                                             │ │
│  │      ],                                                                                                     │ │
│  │  }                                                                                                          │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    CLASSIFICATION ALGORITHM                                                 │ │
│  │                                                                                                             │ │
│  │                            User Query                                                                       │ │
│  │                                │                                                                            │ │
│  │                                ▼                                                                            │ │
│  │                    ┌───────────────────────┐                                                                │ │
│  │                    │  Embedder Available?  │                                                                │ │
│  │                    └───────────┬───────────┘                                                                │ │
│  │                                │                                                                            │ │
│  │              ┌─────────────────┼─────────────────┐                                                          │ │
│  │              │ NO              │                 │ YES                                                      │ │
│  │              ▼                 │                 ▼                                                          │ │
│  │    ┌─────────────────┐         │       ┌─────────────────┐                                                  │ │
│  │    │  FAIL CLOSED    │         │       │  Encode Query   │                                                  │ │
│  │    │                 │         │       │  q_emb = embed( │                                                  │ │
│  │    │  reason:        │         │       │    question     │                                                  │ │
│  │    │  "embeddings    │         │       │  )              │                                                  │ │
│  │    │   unavailable"  │         │       └────────┬────────┘                                                  │ │
│  │    │                 │         │                │                                                           │ │
│  │    │  confident:     │         │                ▼                                                           │ │
│  │    │  False          │         │       ┌─────────────────────────────────────┐                              │ │
│  │    └─────────────────┘         │       │  For each section in SECTION_REFS:  │                              │ │
│  │                                │       │                                     │                              │ │
│  │                                │       │  1. Encode all reference phrases    │                              │ │
│  │                                │       │  2. Compute cosine similarity:      │                              │ │
│  │                                │       │     score = max(q_emb @ ref_embs.T) │                              │ │
│  │                                │       │  3. Store {section: score}          │                              │ │
│  │                                │       └────────────────┬────────────────────┘                              │ │
│  │                                │                        │                                                   │ │
│  │                                │                        ▼                                                   │ │
│  │                                │       ┌─────────────────────────────────────┐                              │ │
│  │                                │       │        HARD GATE 1:                 │                              │ │
│  │                                │       │   max_score < MIN_CONFIDENCE (0.4)  │                              │ │
│  │                                │       └────────────────┬────────────────────┘                              │ │
│  │                                │                        │                                                   │ │
│  │                                │        YES ◄───────────┼───────────► NO                                    │ │
│  │                                │         │              │              │                                    │ │
│  │                                │         ▼              │              ▼                                    │ │
│  │                                │  ┌─────────────────┐   │   ┌─────────────────────────────────────┐         │ │
│  │                                │  │  FAIL CLOSED    │   │   │        HARD GATE 2:                 │         │ │
│  │                                │  │                 │   │   │   |score1 - score2| < AMBIGUITY     │         │ │
│  │                                │  │  reason:        │   │   │   MARGIN (0.08)                     │         │ │
│  │                                │  │  "below conf    │   │   └────────────────┬────────────────────┘         │ │
│  │                                │  │   threshold"    │   │                    │                              │ │
│  │                                │  │                 │   │    YES ◄───────────┼───────────► NO               │ │
│  │                                │  │  scores: {...}  │   │     │              │              │               │ │
│  │                                │  │  confident:     │   │     ▼              │              ▼               │ │
│  │                                │  │  False          │   │  ┌─────────────┐   │   ┌─────────────────┐        │ │
│  │                                │  └─────────────────┘   │  │ FAIL CLOSED │   │   │    SUCCESS      │        │ │
│  │                                │                        │  │             │   │   │                 │        │ │
│  │                                │                        │  │ reason:     │   │   │ section:        │        │ │
│  │                                │                        │  │ "ambiguous" │   │   │   "Item1A" or   │        │ │
│  │                                │                        │  │             │   │   │   "Item7"       │        │ │
│  │                                │                        │  │ scores:     │   │   │                 │        │ │
│  │                                │                        │  │ {...}       │   │   │ scores: {...}   │        │ │
│  │                                │                        │  │             │   │   │                 │        │ │
│  │                                │                        │  │ confident:  │   │   │ confident:      │        │ │
│  │                                │                        │  │ False       │   │   │ True            │        │ │
│  │                                │                        │  └─────────────┘   │   └─────────────────┘        │ │
│  │                                │                        │                    │                              │ │
│  └────────────────────────────────┴────────────────────────┴────────────────────┴──────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    OUTPUT FORMAT                                                            │ │
│  │                                                                                                             │ │
│  │  @dataclass                                                                                                 │ │
│  │  class SectionDecision:                                                                                     │ │
│  │      section: Optional[str]        # "Item1A", "Item7", or None                                             │ │
│  │      scores: Dict[str, float]      # {"Item1A": 0.72, "Item7": 0.41}                                        │ │
│  │      reason: str                   # Human-readable explanation                                             │ │
│  │      confident: bool               # True if safe to proceed                                                │ │
│  │                                                                                                             │ │
│  │  Example outputs:                                                                                           │ │
│  │                                                                                                             │ │
│  │  Query: "What risks does Apple face?"                                                                       │ │
│  │  → SectionDecision(                                                                                         │ │
│  │      section="Item1A",                                                                                      │ │
│  │      scores={"Item1A": 0.82, "Item7": 0.31},                                                                │ │
│  │      reason="Clear match to Item1A (0.82 vs 0.31, gap=0.51)",                                               │ │
│  │      confident=True                                                                                         │ │
│  │  )                                                                                                          │ │
│  │                                                                                                             │ │
│  │  Query: "Tell me about Apple"                                                                               │ │
│  │  → SectionDecision(                                                                                         │ │
│  │      section=None,                                                                                          │ │
│  │      scores={"Item1A": 0.35, "Item7": 0.32},                                                                │ │
│  │      reason="Below confidence threshold (max=0.35 < 0.4)",                                                  │ │
│  │      confident=False                                                                                        │ │
│  │  )                                                                                                          │ │
│  │                                                                                                             │ │
│  │  Query: "Discuss Apple's performance and risks"                                                             │ │
│  │  → SectionDecision(                                                                                         │ │
│  │      section=None,                                                                                          │ │
│  │      scores={"Item1A": 0.65, "Item7": 0.61},                                                                │ │
│  │      reason="Ambiguous (gap=0.04 < 0.08 margin)",                                                           │ │
│  │      confident=False                                                                                        │ │
│  │  )                                                                                                          │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    CONFIGURATION                                                            │ │
│  │                                                                                                             │ │
│  │  MIN_CONFIDENCE = 0.4      # Below this → fail closed                                                       │ │
│  │  AMBIGUITY_MARGIN = 0.08   # Gap between top 2 must exceed this                                             │ │
│  │                                                                                                             │ │
│  │  Tuning notes:                                                                                              │ │
│  │  • MIN_CONFIDENCE=0.4 catches most off-topic queries                                                        │ │
│  │  • AMBIGUITY_MARGIN=0.08 balances false positives vs. usability                                             │ │
│  │  • Both values determined empirically on test set                                                           │ │
│  │                                                                                                             │ │
│  │  Embedding model:                                                                                           │ │
│  │  • sentence-transformers/all-MiniLM-L6-v2                                                                   │ │
│  │  • 384-dimensional embeddings                                                                               │ │
│  │  • ~50ms per encoding                                                                                       │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE                                                                                                │
│  LIMITATIONS:                                                                                                   │
│  • Only supports Item 1A and Item 7 routing                                                                     │
│  • Requires sentence-transformers (graceful degradation if unavailable)                                        │ │
│  • Reference phrases may need tuning for edge cases                                                             │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 LLM Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        LLM EXTRACTION PIPELINE                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                  │
│  PURPOSE: Extract structured facts from unstructured SEC filing text                                            │
│  FILE: extraction.py                                                                                            │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    EXTRACTION PROMPT                                                        │ │
│  │                                                                                                             │ │
│  │  The prompt is carefully engineered to minimize hallucination:                                              │ │
│  │                                                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ EXTRACTION_PROMPT = '''                                                                             │   │ │
│  │  │ Extract financial facts from the following text. This is from a SEC filing.                        │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ CRITICAL RULES:                                                                                     │   │ │
│  │  │ 1. Only extract facts EXPLICITLY stated in the text                                                 │   │ │
│  │  │ 2. Do NOT infer, calculate, or guess any values                                                     │   │ │
│  │  │ 3. Each fact MUST include sentence_string - the EXACT sentence from the text                        │   │ │
│  │  │ 4. If no financial facts are present, return []                                                     │   │ │
│  │  │ 5. Prefer specific numbers over ranges                                                              │   │ │
│  │  │ 6. Include period information when available                                                        │   │ │
│  │  │                                                                                                     │   │ │
│  │  │ OUTPUT FORMAT (JSON array):                                                                         │   │ │
│  │  │ [                                                                                                   │   │ │
│  │  │   {                                                                                                 │   │ │
│  │  │     "metric": "string - name of the metric",                                                        │   │ │
│  │  │     "value": number or null,                                                                        │   │ │
│  │  │     "unit": "string - USD, shares, percent",                                                        │   │ │
│  │  │     "period": "string - e.g., 'Q3 FY2025'",                                                         │   │ │
│  │  │     "period_end_date": "string - e.g., '2024-10-27'",                                               │   │ │
│  │  │     "sentence_string": "string - EXACT quote from text"                                             │   │ │
│  │  │   }                                                                                                 │   │ │
│  │  │ ]                                                                                                   │   │ │
│  │  │ '''                                                                                                 │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                                             │ │
│  │  Key design decisions:                                                                                      │ │
│  │  • sentence_string is MANDATORY - enables Gate A verification                                              │ │
│  │  • JSON output for reliable parsing                                                                         │ │
│  │  • Explicit "do not infer" instruction                                                                      │ │
│  │  • Empty array is valid output (no facts is better than fake facts)                                        │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                    EXTRACTION FLOW                                                          │ │
│  │                                                                                                             │ │
│  │                        Section Text (from section_locator)                                                  │ │
│  │                                      │                                                                      │ │
│  │                                      ▼                                                                      │ │
│  │                        ┌─────────────────────────┐                                                          │ │
│  │                        │   Chunk if >10K tokens  │                                                          │ │
│  │                        │   (preserve paragraphs) │                                                          │ │
│  │                        └────────────┬────────────┘                                                          │ │
│  │                                     │                                                                       │ │
│  │                    ┌────────────────┼────────────────┐                                                      │ │
│  │                    │                │                │                                                      │ │
│  │                    ▼                ▼                ▼                                                      │ │
│  │            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                                │ │
│  │            │  Chunk 1    │  │  Chunk 2    │  │  Chunk N    │                                                │ │
│  │            └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                                │ │
│  │                   │                │                │                                                       │ │
│  │                   ▼                ▼                ▼                                                       │ │
│  │            ┌─────────────────────────────────────────────────┐                                              │ │
│  │            │          LLM (Claude claude-sonnet-4-20250514)           │                                              │ │
│  │            │                                                 │                                              │ │
│  │            │  • Parallel chunk processing                    │                                              │ │
│  │            │  • Each chunk → JSON array of facts             │                                              │ │
│  │            │  • Max 4096 output tokens per chunk             │                                              │ │
│  │            └─────────────────────┬───────────────────────────┘                                              │ │
│  │                                  │                                                                          │ │
│  │                                  ▼                                                                          │ │
│  │            ┌─────────────────────────────────────────────────┐                                              │ │
│  │            │           Response Parsing                      │                                              │ │
│  │            │                                                 │                                              │ │
│  │            │  • Extract JSON from markdown code blocks       │                                              │ │
│  │            │  • Validate JSON structure                      │                                              │ │
│  │            │  • Handle malformed responses gracefully        │                                              │ │
│  │            └─────────────────────┬───────────────────────────┘                                              │ │
│  │                                  │                                                                          │ │
│  │                                  ▼                                                                          │ │
│  │            ┌─────────────────────────────────────────────────┐                                              │ │
│  │            │           Fact Object Creation                  │                                              │ │
│  │            │                                                 │                                              │ │
│  │            │  For each raw extraction:                       │                                              │ │
│  │            │  • Generate unique fact_id (UUID)               │                                              │ │
│  │            │  • Create Location object with:                 │                                              │ │
│  │            │    - CIK, doc_date, doc_type                    │                                              │ │
│  │            │    - section_id, paragraph_index                │                                              │ │
│  │            │    - sentence_string (exact quote)              │                                              │ │
│  │            │  • Set source_format="html"                     │                                              │ │
│  │            │  • Set source_tier="tier2"                      │                                              │ │
│  │            │  • Set verification_status="unverified"         │                                              │ │
│  │            │                                                 │                                              │ │
│  │            └─────────────────────┬───────────────────────────┘                                              │ │
│  │                                  │                                                                          │ │
│  │                                  │  List[Fact] with status="unverified"                                     │ │
│  │                                  ▼                                                                          │ │
│  │                        ┌─────────────────────────┐                                                          │ │
│  │                        │        GATE A           │                                                          │ │
│  │                        │    (next section)       │                                                          │ │
│  │                        └─────────────────────────┘                                                          │ │
│  │                                                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                  │
│  STATUS: ✅ LIVE                                                                                                │
│  LIMITATIONS:                                                                                                   │
│  • Requires ANTHROPIC_API_KEY                                                                                   │
│  • LLM may miss some facts (recall < 100%)                                                                      │
│  • LLM may extract non-facts (precision < 100%, but Gate A catches these)                                      │
│  • Cost: ~$0.01-0.05 per section extraction                                                                     │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```


