## Call Path Contracts (Authoritative) — Whole System

### 0) Primary Entry Point Contract

**User-facing entry** must go through:

* `Orchestrator.ask()` → `classifier.py` → `router.py` → tier handler
* Any handler that produces text must end at: `narrator.py` (LLM) OR a safe non-LLM formatter.

**Hard ban:** No direct LLM “answer generation” from raw sources. Narrator only.

**Note:** `Orchestrator.ask()` handlers for QUALITATIVE / SIGNAL / VERIFICATION / EXPLORATION may be placeholders; do not assume these routes work unless explicitly tested in code or documented in this file.

---

### 1) Tier 1 Contract (HARD): XBRL → FactStore 🟢

Allowed authoritative numeric write path:

* `entities.py` (ticker/name → CIK + fiscal year alignment)
* `xbrl.py` (SEC companyfacts extraction)
* `FactStore.add_fact()` only for `exact_match/approximate_match` statuses

**Hard ban:** LLM involvement anywhere in Tier 1 numeric extraction.

---

### 2) Tier 2 Contract (MEDIUM): SEC HTML + LLM → Gate A → FactStore 🟡

(You already wrote this; it’s the critical write contract.)

* `ingestion.py` → `parsing.py` → `extraction.py` (LLM proposes candidate facts WITH pointers)
* `pipeline.py` (+ `tables.py`, `numeric_verification.py`) = **Gate A**
* `FactStore.add_fact()` **ONLY after Gate A passes**

**Hard ban:** Any Tier 2 write that bypasses Gate A.

---

### 3) Tier 3 Contract (SOFT): News/Web → Gate B → Report 🟠

(You already wrote this; it’s the “read-only” contract.)

* `discovery.py` / `news_search.py` → Claims/Leads (no facts)
* `cross_verify.py` = **Gate B** (compares vs FactStore; **reads only**)
* returns `DiscoveryReport` / verification output

**Hard ban:** Tier 3 writes to FactStore (unless an explicit `ContextStore` is introduced).

---

### 4) Narrator Contract (LLM Output)

Any natural-language answer must be:

* `FactStore` (retrieve relevant verified facts)
* `narrator.py` formats facts + citations → LLM narrates
* citations parsed back → `NarratedReport`

**Hard ban:** Narrator cannot cite anything not in FactStore (no raw HTML snippets, no news text, no “common knowledge”).

---

### 5) Drift / Signals Contract (Alpha Signal)

Signals are **compute + visualization**, not facts.

* `ingestion.py` (get two SEC snapshots)
* `parsing.py` (extract Risk Factors sections)
* `signals.py` (sentence diff via `difflib.SequenceMatcher`, optional semantic similarity)
* returns `SignalAlert` / `DriftResult`

**Hard ban:** Drift output must not be written to FactStore as “facts” (it’s an analytic artifact). If stored, store separately as “signals/artifacts.”

---

### 6) Output / Report Formatting Contract

Output generation must be downstream-only:

* `report.py` / `output.py` format what’s already in:

  * FactStore (facts)
  * verification reports (DiscoveryReport)
  * signal artifacts (DriftResult)
* No new extraction/verification occurs here

**Hard ban:** Output layer cannot “reach back” to sources to add facts.

---

### 7) Evaluation Harness Contract

Eval must run the system via a single stable callable:

* `eval.py` → `system_fn(question)` → returns structured output
* evaluation compares against golden dataset with numeric tolerances

**Hard ban:** Eval cannot “cheat” by reading golden answers inside extraction/routing.

---

### 8) Legacy Deep Research Contract (LangGraph)

This is **Tier 4 / Parallel Track** unless fully integrated.

* `deep_researcher.py` produces a report + sources
* If integrated with Scout & Judge, must route through:

  * claim extraction → Gate B (and/or Gate A if it touches SEC HTML) → **never straight to FactStore**

**Hard ban:** Deep Research findings cannot be treated as facts unless they pass the same gates.
