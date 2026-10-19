# PreFlight v2.0 Plan, Dry-Run, and Plan PDF Validation Report (Phase 4 / M3)

## 1. Executive Summary

This report documents the implementation and validation of the **PreFlight v2.0 Plan/Apply architecture, Dry-Run engine, Plain-Language Sentence Templates, and Plan PDF generation** (Phase 4 / M3).

Evaluation was conducted against:
- The entire 254-file benchmark corpus (`corpus/raw/`) and 2,850 columns.
- Dedicated unit test suites validating planning purity, Decision lifecycle status transitions, deterministic fingerprint stability, plan.json round-tripping, jargon linting, failure isolation, and plan/apply equivalence.

---

## 2. Corpus Validation Metrics

All measurements reflect actual evaluation across all 254 files in `corpus/raw/` executed via `preflight.dry_run`.

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Total Corpus Files** | **254** | Complete benchmark corpus |
| **Unhandled Exception Rate** | **0.00%** (0 / 254) | 254 files completed without unhandled exceptions |
| **Total Plan PDF Pages** | **553 pages** | Generated using ReportLab (`NumberedCanvas`) |
| **Median PDF Pages / File** | **2.0** | Typical plan is a concise 2-page specification |
| **Maximum PDF Pages / File** | **8** | Large tables (68+ columns with questions) |
| **Median `needs_attention` / File** | **1.0** | 50% of files require $\le 1$ human confirmation |
| **Mean `needs_attention` / File** | **3.30** | Driven by a small subset of wide survey tables |
| **Max `needs_attention` / File** | **67** | `elements-by-episode.csv` (68 columns) |
| **Files Exceeding 10 Questions** | **15 files** (5.9%) | Listed in full below without threshold loosening |

---

## 3. Analysis of Files Exceeding 10 Questions

Per project guidelines, no inference thresholds were loosened or manipulated to hide question volume. Exactly 15 files (5.9% of corpus) generated more than 10 `needs_attention` items:

| Corpus Filename | Columns | `needs_attention` Count | Root Cause / Pattern |
| :--- | :---: | :---: | :--- |
| `elements-by-episode.csv` | 68 | **67** | 68 indicator columns of elements in Bob Ross paintings; sparse binary occurrences cause close candidate margins |
| `thanksgiving_meals.csv` | 65 | **62** | Broad national survey with 65 text columns containing multiple text options and abstentions |
| `food-world-cup-data.csv` | 48 | **41** | 48 survey columns evaluating perceptions of international cuisine |
| `raw-responses.csv` | 40 | **38** | Multi-question survey dataset with ambiguous categorical responses |
| `big_epa_cars.csv` | 83 | **24** | 83 columns with mixed measurement units and embedded units |
| `bachelorette.csv` | 19 | **21** | Contestant and elimination flags with non-standard status markers |
| `police_killings.csv` | 34 | **19** | Demographic and county code columns with leading zeros and contextual codes |
| `COUNT_affairs.csv` | 19 | **18** | Discrete integer count metrics with ambiguous rating distributions |
| `full_trains.csv` | 27 | **14** | Transit metrics with mixed timestamps and punctuality stats |
| `week9_comic_characters.csv` | 16 | **14** | Comic character attributes with ambiguous ID codes and appearances |
| `nobel_winners.csv` | 18 | **12** | Mixed death/birth dates and ambiguous day/month formatting |
| `nyc_squirrels.csv` | 31 | **12** | Behavioral indicator columns with discrete binary values |
| `StarWars.csv` | 38 | **12** | Survey responses regarding movie rankings with ambiguous ordering |
| `week7_starwars.csv` | 38 | **12** | Duplicate survey extract matching StarWars.csv |
| `ipf_lifts.csv` | 16 | **11** | Weightlifting records with equipment strings and decimal comma formats |

### Finding on Question Volume
The files exceeding 10 questions are almost exclusively **wide survey questionnaires** (40 to 83 columns) or tables consisting almost entirely of binary survey indicators. Because each column's candidate margin is evaluated independently, survey datasets naturally generate a question per column when multiple text interpretations are viable. For subsequent milestones, stage-level batching (e.g. "Confirm all 60 survey indicators as categories with one click") will be beneficial for UX.

---

## 4. Plan PDF Section Analysis

The Plan PDF specification document renders the following sections in exact order:
1. **Summary box**: File name, row/column counts, designated target column, prediction objective, count of items needing review, and data safety guarantee.
2. **"Needs your decision"**: Each flagged decision presented as a clear question with options, a highlighted recommended safe default, and contextual explanation. If 0 items need review, confirms all automated checks passed.
3. **"What we will do"**: Grouped into five distinct stages:
   - **Reading**: Active (contains file text format, column separator, table title row, column type conversions, and missing marker handling).
   - **Fixing**: **Empty** ("Nothing to do at this stage.").
   - **Filling gaps**: **Empty** ("Nothing to do at this stage.").
   - **Removing**: **Empty** ("Nothing to do at this stage.").
   - **Creating**: **Empty** ("Nothing to do at this stage.").
4. **"What we won't touch"**: Explicitly guarantees that the source file is never modified or overwritten, and details preserved columns (identifiers, constants, raw text).
5. **Sign-off checklist**: Three verification check boxes with name, signature, and date lines.

### Empty Sections Rationale
As mandated by the milestone scope, clean and engineer logic from v1 are not ported in Phase 4. Rather than hiding these stages, the planner registry registers them as future extension points and the Plan PDF explicitly displays **"Nothing to do at this stage."** for:
- **Fixing** (data cleaning, value standardization, corrupt cell replacement)
- **Filling gaps** (missing data handling)
- **Removing** (constant/duplicate column dropping, outlier pruning)
- **Creating** (feature encoding, scaling, transformations)

---

## 5. Dependency Licences and Jargon Verification

### 5.1 Third-Party Dependencies
- **`reportlab` (pinned as `>=4.0.0`)**: Core dependency.
  - **Licence**: **BSD license** (ReportLab Open Source License based on BSD). Permissive and fully compatible with PreFlight's MIT license.
- **`pypdf` (pinned as `>=5.0.0`)**: Developer-only test dependency (`[project.optional-dependencies] dev`).
  - **Licence**: **BSD-3-Clause**. Permissive and fully compatible.
- No other dependencies were added or modified.

### 5.2 Banned Jargon Lint
All user-facing templates and PDF elements were verified against the strict banned jargon list:
`imputation`, `winsorize`, `cardinality`, `dtype`, `coercion`, `heuristic`, `parse`, `schema`, `delimiter`, and raw encoding identifiers (e.g. `utf-8`, `latin-1`).
- `tests/unit/test_jargon_lint.py` rendered Plan PDFs across 7 hand-built edge-case files (leading-zero IDs, European decimal comma, ambiguous dates, embedded units, semicolon separators, ambiguous targets).
- Full text extracted via `pypdf` across all pages confirmed **0 occurrences of any banned jargon**.

---

## 6. Architecture & Verification Summary

### 6.1 Plan / Apply Split (`src/preflight/plan.py`)
- **Purity**: `plan()` operates in read-only mode, never mutating source files or creating cleaned data files.
- **Stage Protocol**: Each stage exposes `plan(context) -> List[Decision]` and `apply(context, decisions) -> Any`.
- **Typing Apply Losslessness**: `TypingStage.apply` converts strings to target dtypes. Any cell failing conversion is logged to `context.anomalies` with row index, column name, raw value, and error description. No value is silently coerced or dropped.
- **Decision Status Transitions**: Supports `confirm()` (status $\to$ `"confirmed"`) and `change(new_action)` (status $\to$ `"changed"`). Timestamps update on transition while the 16-hex deterministic fingerprint remains invariant.
- **Dry-Run Failure Isolation**: `dry_run` writes `plan.json` *before* attempting PDF rendering. When PDF rendering fails, `plan.json` persists on disk and a clear error naming the saved file is raised.

### 6.2 Test Suite Execution Results
```
pytest tests/unit/
============================== 56 passed in 1.42s ==============================

pytest tests/corpus/test_plan_corpus.py
======================== 1 passed in 424.36s (0:07:04) =========================

pytest tests/corpus/test_ingest_corpus.py
============================== 3 passed in 12.42s ==============================

pytest tests/corpus/test_label_integrity.py
============================= 512 passed in 5.01s ==============================
```
Total: **572 automated tests passing**, confirming zero regressions across all Phase 1, Phase 2, Phase 3, and Phase 4 capabilities.
