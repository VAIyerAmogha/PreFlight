# PreFlight v2 — Project Guide

This guide assumes you know nothing about the project. Read Section 1 first (one page, the
whole picture). Section 2 goes through every file. Section 3 is a glossary: if a word is
unclear, look it up there.

- **Section 1** — the whole project in one page
- **Section 2** — file by file
  - 2.1 `src/preflight/` top-level files · 2.2 `ingest/` · 2.3 `infer/` · 2.4 `clean/` ·
    2.5 `engineer/` · 2.6 `interact/` · 2.7 `report/` · 2.8 `scripts/` · 2.9 `tests/` ·
    2.10 `corpus/` · 2.11 `docs/` · 2.12 root files (`pyproject.toml`, CI, nox)
- **Section 3** — glossary

Line numbers refer to the files as they are at the end of Phase 7c.

---

# Section 1 — The whole project in one page

## What it does

You give PreFlight **the path of a messy CSV file** and the name of the column you want to
predict (the **target**). PreFlight:

1. **Reads the file safely.** It works out the text format, the separator and the column-names
   row instead of assuming them.
2. **Decides each column's type** (number, date, category, yes/no, ID, free text...), and finds
   values that secretly mean "missing" (`NA`, `?`, `-999`).
3. **Writes every choice down as a Decision**, with the evidence and a confidence score, and
   **asks you** about the ones it is unsure of.
4. Shows you the whole plan in a **Plan PDF** *before* changing anything.
5. Produces a **typed, cleaned table**, plus an **unfitted scikit-learn pipeline** holding
   every step that learns from data (fills, clipping limits, encodings), so you can evaluate
   models honestly.
6. Ends every real run with a **Report PDF**: what was done, who decided it, what looked odd.

Your source file is never changed. Outputs go to `./preflight_out/`.

## The public functions

| Function | Does | Writes |
|---|---|---|
| `preflight.plan(path, target)` | Looks and proposes. Pure: writes nothing. | nothing |
| `preflight.dry_run(path, target)` | plan + answer questions + Plan PDF. No data output. | `plan.json`, `plan.pdf`, `answers.json` |
| `preflight.resolve(plan or path or review.json, mode, channel)` | Answers the open questions. | `plan.json`, `plan.pdf`, `answers.json`, `review.json` |
| `preflight.apply(plan, mode=...)` | Runs the plan. Returns an `ApplyResult`. | nothing by itself |
| `preflight.prepare(path, target)` | Everything, for real: plan → questions → apply → save → Report PDF. | everything below |

## The pipeline, as a picture

```
 your_file.csv  (never modified)
      │
      ▼
 ┌─────────────────────── READING (ingest/) ────────────────────────┐
 │  bytes ─► text format? ─► separator? ─► column-names row?        │
 │           (encoding)      (dialect)      (header, title/footer)  │
 │  ─► raw table: EVERY cell kept as text                           │
 └──────────────────────────────────────────────────────────────────┘
      │                                 each step = 1 Decision
      ▼
 ┌─────────────────────── TYPING (infer/) ──────────────────────────┐
 │  per column: disguised missing values? ─► score 8 types          │
 │  ─► clear winner, or ABSTAIN (keep as text and ask)               │
 │  ─► TableSchema (the file's "ID card") + 1 Decision per column    │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
 ┌───────────── PLANNING cleaning + engineering (clean/, engineer/) ┐
 │  simulated on a copy; nothing is changed yet                     │
 │  markers ► read numbers ► repeats ► drop columns ► tidy text ►   │
 │  fill blanks ► clip extremes ► leakage check ► encode/scale      │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
  Plan = list of Decisions ──► Plan PDF ("what we will do", "needs your decision")
      │
      ▼
 ┌─────────────────────── QUESTIONS (interact/) ────────────────────┐
 │  needs_attention Decisions ─► ONE batch of questions             │
 │  stored answers (same fingerprint) reused                         │
 │  mode: ask (terminal / notebook / review file / callback)        │
 │        defaults (safe default for each) · strict (error)         │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
 ┌─────────────────────── APPLY (plan.py, assemble.py) ─────────────┐
 │  typed table (ApplyResult.data)                                  │
 │  + steps that learn NOTHING ─► prepared table (ApplyResult.prepared)
 │  steps that LEARN ─► UNFITTED pipeline (ApplyResult.pipeline):   │
 │      FoldCleaner (rare values, fills, clipping)                  │
 │      ─► EngineerTransformer (dates, encodings, scaling, extras)  │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
  prepare() saves: cleaned_data.csv, schema.json, decisions.json,
                   pipeline.joblib, pipeline_fitted_on_all_rows.joblib
                   ─► THEN report.pdf
```

## The data flow, step by step

1. **CSV path → raw text table.** `ReadingStage` (`plan.py`) calls `detect_encoding`,
   `detect_dialect` and `detect_header` (`ingest/`), then `load_raw` reads every cell as a string.
   Example: `"02138"` stays `"02138"` and `"NA"` stays `"NA"`.
2. **Raw table → TableSchema + type Decisions.** `infer_table_types` scores each column and
   records `inferred_type`, `confidence`, `missing_markers`, `anomalies` and `evidence` in the
   schema.
3. **Plan.** `_ensure_prep_plan` builds the typed table in memory, computes the column
   **signals** (`signals.py`: missing rate, variety, correlation...), and runs
   `plan_clean_full` and `plan_engineer`. The result is a `Plan`: all Decisions in order,
   plus the schema.
4. **Questions.** `resolve` turns every `needs_attention` Decision into a question, gets
   answers, and marks each Decision `confirmed`, `changed` or `automatic`. If a reading or
   typing answer changed, the cleaning and engineering plan is rebuilt (`replan_prep`).
5. **Apply.** `TypingStage.apply` converts the text to real types. Values that fail are logged
   as **anomalies**, never dropped silently. This gives the **typed table**
   (`ApplyResult.data`). The steps that learn nothing run on it, giving the **prepared
   table** (`ApplyResult.prepared`). The steps that learn go into the **unfitted pipeline**
   (`ApplyResult.pipeline`).
6. **Outputs.** `prepare()` writes the files, then the Report PDF.

## How you are meant to use the pipeline (fit hygiene)

```python
from sklearn.base import clone
from sklearn.model_selection import KFold

out = preflight.prepare("data.csv", target="label", mode="defaults")
df = out["apply_result"].prepared
X, y = df.drop(columns=["label"]), df["label"]   # pipeline input = prepared WITHOUT the target
for train, test in KFold(5).split(X):
    pipe = clone(out["pipeline"])                 # fresh, unfitted copy per fold
    Xtr = pipe.fit_transform(X.iloc[train], y.iloc[train])
    Xte = pipe.transform(X.iloc[test])
    # fit your model on Xtr, score it on Xte
```

The pipeline is fitted **inside** each fold, so fill values, limits and encodings never see
the test rows. `pipeline_fitted_on_all_rows` is for transforming new data after you are done.
It is **not** for scoring, because it has already seen every row. (The `DEMO_GUIDE.md` runs
this for real.)

## Why it is built this way

| Design choice | Why |
|---|---|
| **Everything is a Decision** with evidence, confidence and a safe default. | You can see, question and change every choice. The same data gives the same plan. |
| **Plan before apply** (`plan()` never writes). | The Plan PDF can show the full plan before anything happens (a dry run). |
| **Read every cell as text first.** | pandas' guesses lose information: `02138` → 2138, `NA` → blank. Typing happens later, with evidence. |
| **Abstain instead of guessing** when two types are close or a trap is found. | A wrong type that nobody notices is the worst outcome. Flagging costs a question. Guessing wrong silently costs a model. |
| **Questions are batched and never block.** | One screen of questions, not interruptions. A server with no terminal gets `strict` (an error listing every question), never a hang. |
| **Fingerprints** on Decisions, answers stored in `answers.json`. | Run again on the same data and you are not asked again. Change the data so the evidence changes, and you are. |
| **Learn-nothing steps on the table, learning steps in an unfitted pipeline.** | Otherwise your cross-validation scores are too optimistic (the fills and encodings have seen the test rows). |
| **Cross-fitted target encoding.** | A row's own answer must not leak into its own encoded value. |
| **Plain language in PDFs, checked by a jargon list.** | The person who signs off may not be a data scientist. |
| **Outputs in `./preflight_out/`, never next to the source.** | The source folder may be read-only or shared. |
| **Files saved before the Report PDF is drawn.** | A PDF bug must never cost you your results. |
| **v1's cleaning and engineering rules were ported, not rewritten.** | They were already tested. v2 only changed *how* choices are made and recorded. |

## What changed from v1 (v1.0.1)

| | v1 | v2 |
|---|---|---|
| Input | a pandas DataFrame you had already loaded | a file path. PreFlight does the reading |
| Reading | your own `pd.read_csv`. Comma, UTF-8 and header row assumed | detected, with confidence. Asks when unsure |
| Types | v1 "Profiler" kinds | 8 types scored with evidence, traps, abstaining. v1 kinds are derived from them (`signals.kind_from_schema`) |
| Choices | made silently, then reported | Decisions: planned, shown, answerable |
| Questions | none | 4 channels, 3 modes, stored answers |
| Pipeline | fitted on all rows | **unfitted** (fit per fold), plus an all-rows copy for deployment |
| Reports | one matplotlib PDF | Plan PDF before + Report PDF after, plain language |

**Measured on the 254-file corpus** (`docs/v1_baseline_report.md` vs `docs/v2_hardening_report.md`;
v2 held-out files were seen during review, so they are not independent):

| | v1.0.1 | v2 (Phase 7c, all 254 files) |
|---|---|---|
| Files that crash | 10/254 (3.94%) | 0/254 |
| Column type, strict | 2302/2850 (80.77%) | 2600/2850 (91.23%) |
| Wrong type with no flag (of all columns) | 268/2850 (9.40%) | 13/2850 (0.46%) |
| Dialect (separator + encoding + header all right) | 233/254 (91.73%) | 233/254 (91.73%) |

Dialect accuracy is the **same count** as v1's. v2 flags uncertain files instead of assuming,
but it still misses 21 files, mostly Latin-1 vs UTF-8 labels.

## Where the project stands (end of Phase 7c)

`docs/phase8_readiness.md`: 9 of 10 release gates pass. **Gate 8 fails**: dialect (91.73%) and
strict column-type accuracy (91.23%) are below the 95% goal. **Verdict: NO-GO** until the owner
either accepts the lower goals or more work raises the accuracy. Known gaps are listed in each
file's "Known limits" below, and in `docs/v2_hardening_report.md`.

## Bugs found while writing this guide (not fixed: Stage D allows no source changes)

1. `plan.schema.dialect.delimiter` / `effective_delimiter()` and `schema.header.header_row_index`
   can disagree with the reading Decisions in force (seen on a headerless semicolon file: the
   schema said `,` and row 0, the Decisions said `separate_by_;` and `no_header`, and the data was
   read correctly). Trust the Decisions.
2. The question preview (`interact/preview.py:converter_for`) does not strip units, so for a unit
   column "Convert to whole numbers" previews as `(left blank)`, while the real run gives the numbers.

See `docs/DEMO_GUIDE.md` steps 3 and 6.

---

# Section 2 — File by file

## 2.1 `src/preflight/` — the top-level files

These are the files directly inside `src/preflight/`. They hold the public entry points
(`plan`, `dry_run`, `apply`, `prepare`, `resolve`) and the shared building blocks
(Decision, schema, signals, the pipeline builder).

---

### `src/preflight/__init__.py` (21 lines)

**Purpose.** This is what you get when you write `import preflight`. It decides which names are public.

**Walk-through.**
- Lines 3–7: import the public pieces from the files that define them:
  `Decision` (decisions.py), `resolve` (interact/resolve.py), `Plan`, `apply`, `dry_run`, `plan`
  (plan.py), `prepare` (prepare.py), `TableSchema` (schema.py).
- Line 9: `__version__ = "2.0.0.dev0"`. It must match `version` in `pyproject.toml`.
- Lines 11–21: `__all__`, the list of names that count as "the public API".

**Connections.** Everything a user touches starts here. Anything not in `__all__` is internal
and may change without notice.

---

### `src/preflight/decisions.py` (188 lines)

**Purpose.** Defines the **Decision**: the single record type used for every choice PreFlight
makes ("this file is UTF-8", "column age is a number", "fill blanks in age with the median").
Every stage produces Decisions; the PDFs, the questions and the pipeline all read them.
This is the most important idea in the project.

**Why it exists (design choice).** In v1 the library just *did* things and wrote a report
afterwards. In v2 every action is first written down as a Decision, with its evidence and how
sure the library is. That makes three things possible: you can see the whole plan before
anything happens (dry run), you can change any choice (questions), and the same file gives
the same plan every time (fingerprints).

**Walk-through.**
- Lines 18–63 `compute_decision_fingerprint(stage, kind, columns, action, default, evidence) -> str`.
  Builds a short ID (16 hex characters) that is the same every time the same file gives the
  same choice.
  - Lines 34–52: tidy the evidence so it prints the same way every run. Floats are rounded to 4
    places (so 0.123456789 and 0.12345679 match), lists and dicts are sorted or converted to
    text, and anything else becomes a string.
  - Lines 54–61: the payload is stage, kind, the *sorted* column names, action, default and the
    tidied evidence. No timestamps and no random values go in, which is why it is stable.
  - Lines 62–63: SHA-256 of the JSON text, first 16 characters kept.
- Lines 66–102 `class Decision` (a dataclass). The fields:
  - `id`: human-readable ID, built automatically if empty (line 118), e.g. `dec_infer_age_1a2b3c4d`.
  - `stage`: which part of the program made it: `reading`, `infer`, `clean`, `engineer`.
  - `kind`: what sort of choice it is, e.g. `encoding`, `type_inference`, `fill_median`.
  - `columns`: which columns it is about (empty list = the whole file).
  - `action`: what will happen, e.g. `cast_to_int`, `use_encoding_utf-8`, `drop_column`.
  - `evidence`: a dict of the numbers behind the choice (scores, counts, examples).
  - `confidence`: 0.0 to 1.0, how sure the library is.
  - `default`: the safe fallback action, used when nobody answers a question.
  - `needs_attention`: True means "a person should look at this". These become questions.
  - `plain_language`: a sentence a non-programmer can read; this is what the PDFs show.
  - `status`: `automatic` (the library chose), `confirmed` (a person said yes), `changed`
    (a person picked something else).
  - `timestamp`, `fingerprint`, `history` (a log of status changes).
- Lines 104–118 `__post_init__`: fills in the timestamp (current UTC time), the fingerprint and
  the ID if they were not given.
- Line 120 `VALID_STATUSES`: the three allowed statuses.
- Lines 122–131 `_log`: appends one entry to `history` (status, time, action, who changed it,
  the answer text).
- Lines 133–143 `confirm(source, answer)`: status becomes `confirmed`, logged. The
  fingerprint is **not** changed: it keeps naming the original proposal.
- Lines 145–162 `change(new_action, source, answer)`: replaces `action`, status becomes
  `changed`, logged.
- Lines 164–179 `mark_automatic(action, source)`: records that the library settled the
  question itself (used in `defaults` mode), optionally switching to the safe default action.
- Lines 181–188 `to_dict` / `from_dict`: turn a Decision into a plain dict for JSON and back.

**Connections.** Created by every stage (`plan.py`, `infer/`, `clean/`, `engineer/`). Read by
`interact/` (questions), `report/` (PDFs) and `assemble.py` (pipeline). Saved into
`plan.json` and `decisions.json`.

**Known limits.** `from_dict` passes the dict straight to the constructor, so a JSON file
with an unknown key raises `TypeError`.

---

### `src/preflight/schema.py` (182 lines)

**Purpose.** Defines **TableSchema**: the description of the *file itself*. It records how the
file is stored (text format, separator, header row), its columns, and each column's
detected type. Think of it as the file's ID card.

**Walk-through.**
- Line 20 `SCHEMA_VERSION = "2.0.0"`: saved into every schema so old files can be recognised later.
- Lines 23–35 `EncodingSchema`: the text format. `detected` (e.g. `utf-8`), `confidence`,
  `has_bom`, `low_confidence`, `raw_detected` (what the detector literally said) and `override`
  (a person's choice). `effective_encoding()` returns the override if there is one, otherwise
  the detected value. Every part of the schema follows this "override wins" pattern.
- Lines 38–54 `DialectSchema`: the separator (`delimiter`), quote character, escape character,
  confidence, `flagged`, `margin` (how far ahead the winner was), and the overrides.
  `effective_delimiter()` follows the same pattern.
- Lines 57–73 `HeaderSchema`: which row holds the column names (`header_row_index`, `None` =
  no names row), plus lists of row numbers that are titles, footers, blanks, totals or ragged
  (wrong number of cells). `effective_header_row()`.
- Lines 76–102 `ColumnRawMetadata`: one column. `name`, `position`, the overrides, then the
  typing results added in Phase 3: `inferred_type`, `confidence`, `missing_markers`,
  `runner_up` (the second-best type), `abstained` (the library refused to guess),
  `anomalies`, `failed_samples` (example values that did not fit) and `evidence`.
  `effective_name()` and `effective_type()` apply the overrides.
- Line 106: `ColumnSchema` is just another name for `ColumnRawMetadata`.
- Lines 109–132 `TableSchema`: puts it all together. It also holds `decisions` and `user_overrides`.
- Lines 134–181: `to_dict`, `to_json`, `from_dict`, `from_json`. The round trip is
  lossless: save and load gives back an equal object (tested in
  `tests/property/test_schema_roundtrip.py`).

**Connections.** Built by the reading stage in `plan.py`, filled in by `infer/`, read by
`signals.py` (to pick each column's kind) and saved as `schema.json` by
`report/report_pdf.py`.

---

### `src/preflight/types.py` (89 lines)

**Purpose.** Old (v1) type names, kept so the v1 regression tests and the v1-style report
code still work.

**Walk-through.**
- Lines 18–28 `SemanticType`: v1's nine column kinds, `NUMERIC_FEATURE`, `NUMERIC_ID`,
  `CATEGORICAL_LOW`, `CATEGORICAL_HIGH`, `DATETIME_NATIVE`, `DATETIME_STRING`, `BOOLEAN`,
  `CONSTANT` and `TEXT`. ("LOW"/"HIGH" means fewer or more than 20 different values.)
- Lines 32–47 `ColumnProfile`: v1's per-column summary.
- Lines 51–67 `ReportEntry`: one line of the v1 report. `__post_init__` (lines 61–67)
  rejects unknown stage or severity names.
- Lines 71–75 `PrepResult`: v1's return value (table, pipeline, report).
- Line 78 `_UNSET`: a "no value given" marker object.

**Connections.** Used by `report/report.py`, `validation.py` and the regression tests
(`tests/regression/`). New v2 code uses the string kinds in `signals.py` instead.

---

### `src/preflight/validation.py` (196 lines)

**Purpose.** Input checks from v1 for DataFrame inputs: empty table, missing target, a
classification task with a number target, and so on.

**Walk-through.**
- Lines 20 and 24: two named thresholds. A target counts as "a continuous number" when it has
  more than 20 different values **and** more than 5% of the rows are different values.
- Lines 27–103 `_validate_task_target_match(target_series, task) -> str | None`:
  classification with a continuous number target raises `ValueError`. Regression with a text
  target, or with 20 or fewer different values, returns a warning string (rating scales like 1–5
  are legitimate, so this only warns).
- Lines 106–196 `_validate_inputs(df, target, task, model_hint, column_types)`: eight checks in
  order: a DataFrame was given, it is not empty, the target exists, there is at least one other
  column, no duplicate names, the task is valid, `model_hint` is `tree`/`linear`, then the
  task/target check and the `column_types` overrides.

**Connections.** In v2 it is only used by the test adapter `tests/regression/adapter.py`,
because v2's public functions take a file path, not a DataFrame. Line 9 still has
`from __future__ import annotations`; it is harmless on Python 3.10+.

**Known limits.** Not called by `plan()` or `prepare()`. A wrong `model_hint` passed to
`prepare()` is not rejected by this file.

---

### `src/preflight/signals.py` (317 lines)

**Purpose.** Computes the **signals**: the per-column numbers the cleaning and engineering
steps need (missing rate, number of different values, rare values, correlation with the
target and so on). v2's schema records only structure and type. These numbers came from v1's
"Profiler", so they were ported here unchanged (see `docs/port_contract.md`, option A).

**Walk-through.**
- Lines 28–41: the kind names (same strings as v1's `SemanticType`), two groups
  (`CATEGORICAL_KINDS`, `DATETIME_KINDS`) and `DEFAULT_CARDINALITY_THRESHOLD = 20`: a
  category column with fewer than 20 different values is "low variety" (numbered for tree
  models, one yes/no column per value for linear models); 20 or more is "high variety"
  (replaced by the typical outcome per value, i.e. target-encoded).
- Lines 44–63 `ColumnSignals`: the numbers for one column, plus `v2_type`, `confidence`,
  `abstained` and `anomalies` copied from the schema.
- Lines 70–112: small v1 functions copied unchanged:
  - `compute_missing_rate`: share of blanks.
  - `compute_outlier_rate` (76–90): share of values outside the IQR fences
    (Q1 − 1.5·IQR, Q3 + 1.5·IQR). It returns `None` for text, yes/no columns, or when the
    middle half of the values has zero spread.
  - `compute_cardinality`: number of different values.
  - `identify_rare_categories` (97–103): values that make up less than 1% of the non-blank values.
  - `compute_class_imbalance_ratio`: largest class count ÷ smallest.
- Lines 115–121 `_text_like`: v1's free-text test on up to 100 sampled values (seed 42):
  average length over 50, or over 30 with 5+ words.
- Lines 124–172 `v1_infer_kind`: v1's whole kind-guessing rule. It is only used when a column
  has no schema entry.
- Lines 179–193 `_v1_numeric_id`: v1's ID rule. At least 95% of values are different **and**
  (the name contains id/_id/pk/index **or** the values only go up, or only go down, and are whole numbers).
- Lines 196–203 `_plain_numbers`: are all values plain numbers with no leading zero? (Leading
  zeros mean a code such as a ZIP code, which must stay text.)
- Lines 206–235 `kind_from_schema(series, v2_type, cardinality_threshold) -> str`: maps v2's
  type to a v1 kind. This is the bridge between v2 typing and the v1 cleaning/engineering code.
  - constant → CONSTANT, boolean → BOOLEAN, datetime → DATETIME_*.
  - int/float → NUMERIC_FEATURE, unless v1's ID rule fires (→ NUMERIC_ID).
  - ID → NUMERIC_ID if the values are numbers, otherwise CATEGORICAL_HIGH.
  - text → TEXT only if it really looks like prose (`_text_like`); short text is treated as
    a category.
  - everything else → CATEGORICAL_LOW or _HIGH by the 20-value threshold.
- Lines 238–317 `compute_signals(df, schema, target, task, cardinality_threshold, rare_threshold)
  -> dict[str, ColumnSignals]`: loops over every column except the target. It picks the kind,
  computes rare values (categories only), the correlation with the target (numbers only),
  mutual information (not for constant or ID columns), text averages (TEXT only) and VIF
  (lines 312–315, for number columns, all at once). It does not change `df`.

**Connections.** Called by `clean/_common.py` (`need_signals`), `assemble.py`
(`plan_time_column_kinds`, `EngineerTransformer._fold_signals`) and `engineer/`.

**Known limits.** Correlation (and so the leakage flag) is computed only for number
columns. A yes/no or text copy of the target is not caught (see the titanic `alive` example in
`docs/v2_vs_v1_scores.md`).

---

### `src/preflight/plan.py` (1198 lines) — the heart of the program

**Purpose.** Runs everything in order. It defines the **stages**, the `Plan` object, and the
public functions `plan()`, `dry_run()` and `apply()`. When you want to know "what happens when
I call X", start here.

**The big idea.** Every stage has two methods:
- `plan(context) -> list[Decision]`: *look* at the data and propose Decisions. Never changes
  anything.
- `apply(context, decisions)`: *do* what the Decisions say.

`plan()` runs the first method of every stage, `apply()` runs the second. That split is
what makes the dry run possible: the plan is complete before anything is done.

**Walk-through.**
- Lines 1–47: description and imports. Note line 25: `plan.py` imports the pipeline builder
  from `assemble.py`, and lines 26–36 import every cleaning and engineering function.
- Lines 53–66 `PlanContext`: the bag of state passed between stages while planning: file path,
  target, task, the raw table, the schema, the Decisions so far, `model_hint`, configs, and
  `prep_decisions` (a cache, see `_ensure_prep_plan`).
- Lines 69–83 `ApplyContext`: the same for applying. It adds `typed_data` (table after typing),
  `anomalies` (values that failed to convert) and `prepared_data` (table after the steps that
  learn nothing).
- Lines 86–104 `ApplyResult`: what `apply()` returns.
  - `data`: the **typed table** (numbers are numbers, dates are dates).
  - `decisions`, `anomalies`.
  - `prepared`: the table after the "learn nothing" steps (fix markers, read numbers, remove
    repeated rows, drop columns, tidy text).
  - `pipeline`: an **unfitted** scikit-learn Pipeline with every step that *does* learn from
    data. You fit it yourself, on training rows only.
  - `specs`: kept for old callers, always empty now.
- Lines 107–171 `Plan`: what `plan()` returns, a list of Decisions plus the schema and the
  settings. `to_dict`/`to_json`/`save_json`/`from_dict`/`from_json` let it be saved as
  `plan.json` and loaded back.
- Lines 178–189 `Stage`: the base class with the two methods above, plus `name` and
  `section` (the heading it appears under in the Plan PDF).
- Lines 192–359 `ReadingStage`, stage 1: reading the file.
  - `plan` (200–324): detects the encoding, the dialect (separator) and the header row,
    using `ingest/`. It makes one Decision for each.
    - Encoding (213–241): needs attention if the detector says low confidence or confidence < 0.80.
    - Dialect (243–267): needs attention if flagged or confidence < 0.80. Default `separate_by_,`.
    - Header (269–304): needs attention if flagged, ragged, or confidence < 0.80. Default
      `use_header_row_0`.
    - The 0.80 cut-off is "below this, ask a person".
    - Then (306–322) it loads the raw table (all cells as text) into memory and creates the
      `TableSchema` with one entry per column.
  - `apply` (326–359): reads the file again using the actions *in force* on the three reading
    Decisions. So if a person changed the separator to `;`, the file is read with `;`. If the
    header choice differs from the detected one, `_header_for_action` builds the right header
    info. A wrong text format gives a clear `ValueError` instead of a raw `UnicodeDecodeError`.
- Lines 362–364 `_action_value("separate_by_;", "separate_by_") -> ";"`: reads the value out of
  an action string.
- Lines 367–393 `_header_for_action`: builds header info for a header row chosen by a person.
  `no_header` names the columns `column_1`, `column_2`, ... A chosen row number uses that row's
  cells as names, and the rows above it become title rows.
- Lines 396–676 `TypingStage`, stage 2: column types.
  - `plan` (405–516): calls `infer_table_types` (infer/) to type every column and keeps the
    `infer` Decisions. If a target was given, it also decides the **task**:
    - If you passed `task=`, that is used (confidence 1.0, no question).
    - Otherwise (457–488), it is **ambiguous** when the target is a whole-number column with 10
      or fewer different values, or has exactly 2 values but is not yes/no. Then it makes an
      `ambiguous_task` Decision (confidence 0.50, needs attention, default
      `classification`), because something like a 1–5 rating could be either.
    - Otherwise (489–514), number targets mean regression and anything else means
      classification (confidence 0.95).
  - `apply` (518–676): converts each column from text to its type, one value at a time.
    - Blanks and the column's missing-value markers become missing (`pd.NA`).
    - For unit columns, the unit is stripped first (`strip_unit`).
    - `cast_to_int`: removes thousands commas, then `int()`.
    - `cast_to_float`: handles decimal commas (`1.234,56` → `1234.56`).
    - `cast_to_boolean`: true/yes/y/t/1 and false/no/n/f/0.
    - `cast_to_datetime`: `dateutil` parser.
    - **Any value that fails is never silently lost**: it is written to `context.anomalies`
      with its row, column, value and message, and the cell becomes missing.
    - Lines 661–673 choose the pandas dtype: `Int64`, `Float64`, `boolean` (these allow missing
      values), datetime, `category`, or `object` (text).
- Lines 683–701 `_working_frame`: turns the typed table into the plain dtypes the v1-ported
  cleaning code expects. Nullable ints become int (or float if there are blanks), categories
  become text, and so on.
- Lines 704–711 `_resolved_task`: the task from the user, otherwise from the task Decision,
  otherwise `classification`.
- Lines 714–725 `_ACTION_TYPE`: maps a typing action to a type name
  (`cast_to_int` → `int`, `keep_as_id` → `ID`, ...).
- Lines 728–737 `_schema_with_actions`: a copy of the schema whose column types follow the
  actions *now in force*, so a type a person changed reaches cleaning and engineering.
- Lines 740–752 `_unfitted_pipeline`: builds the pipeline returned in `ApplyResult.pipeline`.
  It fixes each column's kind from the whole table (`plan_time_column_kinds`), then calls
  `build_unfitted_pipeline` (assemble.py).
- Lines 755–792 `_ensure_prep_plan`: plans all cleaning and engineering **once**. It builds
  the typed table in memory, runs the cleaning planner (`plan_clean_full`) and the engineering
  planner (`plan_engineer`), and caches the result in `context.prep_decisions`. The later
  stages each take their own kinds from this list. Nothing is written to disk.
- Lines 795–823 `PrepStage`: base class for the cleaning/engineering stages.
  - `kinds`: which Decision kinds belong to this stage.
  - `in_pipeline` (line 804): True for steps that **learn from data**. Their `apply` does
    nothing to the full table; the step only runs inside the returned pipeline. This is the
    fit-hygiene rule (see `docs/fit_hygiene.md`).
  - `apply` (814–823): starts `prepared_data` from the typed table on first use, then runs
    `runner` unless `in_pipeline`.
- Lines 826–911: the stages, in run order:
  - `FixingStage`: missing-value markers → blank, then read numbers out of text (`apply_coerce`).
  - `RemovingStage`: repeated rows, then drop unusable columns.
  - `TidyingStage`: lower-case and trim category text. Rare-value grouping is listed here but
    learns from data, so it runs in the pipeline (line 855 comment).
  - `FillingGapsStage` (in_pipeline): blank-value flags and fills.
  - `OutlierStage` (in_pipeline): clip extreme numbers.
  - `LeakageStage` (in_pipeline): columns that may give away the answer, and overlapping columns.
  - `CreatingStage` (in_pipeline): encodings, scaling and generated columns.
- Lines 914–941 `StageRegistry` and `build_default_registry`: the list of stages in run
  order. A new stage is added by registering it here.
- Lines 948–1051 `Planner`:
  - `plan` (954–1009): checks that the file exists, runs `stage.plan` for every stage
    (each stage sees the Decisions of the earlier ones through `context.decisions`), and
    returns a `Plan`.
  - `apply` (1011–1051): runs `stage.apply` for every stage, then returns `ApplyResult` with
    the unfitted pipeline.
- Lines 1054–1071 `replan_prep`: after a person changes a reading or typing answer, the
  cleaning and engineering Decisions may be out of date. This rebuilds them from the
  new answers. It is called by `interact/resolve.py`.
- Lines 1078–1087 `plan(path, target, task, model_hint, feature_config) -> Plan`: public.
  Pure: reads the file, writes nothing.
- Lines 1090–1096 `DEFAULT_OUT_DIR_NAME = "preflight_out"` and `default_out_dir()`: outputs go
  to `./preflight_out/` in the folder you run Python from, **never** next to the source file
  (it may be read-only or shared). This was changed in Phase 7b.
- Lines 1099–1120 `write_plan_artifacts`: writes `plan.json` **first**, then the Plan PDF.
  If the PDF fails, the JSON is already saved and the error says where.
- Lines 1123–1145 `apply(plan, decisions, mode, channel, out_dir) -> ApplyResult`: public.
  With a `mode`, it first answers every open question through `resolve` (interact/), then
  runs the stages.
- Lines 1148–1198 `dry_run(path, target, task, out_dir, mode="auto", channel, model_hint,
  feature_config) -> (Plan, pdf_path)`: public. Plans, resolves the questions (unless
  `mode=None`), and writes `plan.json` + `plan.pdf`. It never writes cleaned data.

**What changed in v2 / Phase 7b.**
- v1 had no plan: it profiled, cleaned and engineered in one go.
- Phase 7b split the steps into "learn nothing" (run on the full table) and "learn from data"
  (`in_pipeline = True`, run inside the unfitted pipeline). It also moved outputs to
  `preflight_out/` and made `auto` the default mode.

**Known limits.**
- `TypingStage.apply` converts values one at a time in Python, so it is slow on very large
  files (still well within the corpus timeouts).
- `_resolved_task` falls back to `classification` when nothing says otherwise.

---

### `src/preflight/prepare.py` (122 lines)

**Purpose.** The one-call "do everything for real" function: `prepare()`. It plans, asks the
questions, applies, and **always** writes the Report PDF at the end.

**Walk-through.**
- Lines 1–19: description. Two key promises:
  1. There is no switch to turn the Report PDF off.
  2. Everything else is saved *before* the PDF is drawn, so a PDF failure loses nothing.
- Lines 34–44 `_fit_on_all_rows(pipeline, result, target)`: a copy (`clone`) of the
  unfitted pipeline fitted on **every** row. It is for transforming new data once you
  have finished choosing a model. **Never** use it to score a model: it has seen the test
  rows (that is "leakage", see the glossary).
- Lines 47–119 `prepare(path, target, task=None, mode="auto", channel=None, out_dir=None,
  model_hint="tree", feature_config=None) -> dict`:
  - Lines 94–96: resolves the output folder (default `./preflight_out`) and creates it.
  - Line 98: `plan(...)`.
  - Line 103: `apply(..., mode=mode)`, which answers the questions and runs the stages.
  - Lines 105–106: the unfitted pipeline and the all-rows fit.
  - Line 108: `write_report_artifacts` saves `cleaned_data.csv`, `schema.json`,
    `decisions.json`, `pipeline.joblib`, `pipeline_fitted_on_all_rows.joblib`, then renders
    `report.pdf`.
  - Line 112: prints `PreFlight: output written to <folder>`.
  - Returns a dict with `apply_result`, `pipeline` (unfitted), `pipeline_fitted_on_all_rows`,
    `plan` and `paths`.

**Connections.** It is the only caller of `report/report_pdf.write_report_artifacts`.

**Known limits.** It takes a file path only; DataFrame input is not part of 2.0.

---

### `src/preflight/assemble.py` (595 lines)

**Purpose.** Builds the **scikit-learn Pipeline**: the object that carries the "learn from
data" steps so you can fit it on training rows and use it on new rows. Ported from v1's
`assembler.py`, then rebuilt for fit hygiene in Phase 7b.

**Two families live here.**
1. **New (Phase 7b):** `FoldCleaner` + `build_unfitted_pipeline`. This is what `apply()` and
   `prepare()` return.
2. **Ported from v1:** `CleanerTransformer`, `build_pipeline`, `build_pipeline_two_phase`,
   `run_assembler`. These are still used by tests and by the v1-style regression adapter.

**Walk-through.**
- Lines 63–77 `overlay_actions(fresh, in_force)`: when a step is re-planned on training rows,
  copy the person's chosen action (and status) onto the new Decision with the same kind and
  columns. The new evidence (numbers from the training rows) is kept. This is how
  "the person said *don't* drop this column" survives refitting.
- Lines 90–97: the two lists that implement fit hygiene.
  - `STATELESS_CLEAN_KINDS`: learn nothing. These are markers, number reading, repeated rows,
    column drops and text tidying.
  - `PIPELINE_CLEAN_KINDS`: learn from data. These are rare values, blank flags and fills, and
    clipping.
- Lines 100–106 `apply_stateless(df, decisions)`: runs the "learn nothing" steps in plan order.
- Lines 109–129 `plan_time_column_kinds(typed, decisions, schema, target, task)`: each
  column's kind (number, low/high-variety category, ...) worked out once on the whole table.
  The kind picks the encoder, and it is fixed for every fold so all folds produce the same columns.
- Lines 132–135 `_with_evidence`: a copy of a Decision with extra evidence values.
- Lines 138–224 `FoldCleaner`, the cleaning steps that learn from data.
  - `fit` (153–191): from **the rows it is given only**, it:
    1. finds rare values (<1%) for each `group_rare_categories` column and groups them;
    2. works out the median (`fill_median`) or most common value (`fill_mode`) and fills;
    3. works out the clipping limits (`clip_outliers`).
    The results are stored in `decisions_`.
  - `_limits` (193–208): IQR method gives Q1 − 1.5·IQR and Q3 + 1.5·IQR. The z-score method
    gives mean ± 3 standard deviations. Zero spread means no limits. Any other method raises
    an error.
  - `transform` / `_apply` (210–219): apply the learned values in the same order
    (normalize → impute → clip).
  - `get_feature_names_out`: the output column names, so `set_output(transform="pandas")` works.
- Lines 227–251 `build_unfitted_pipeline(target, decisions, column_kinds, ...) -> Pipeline`:
  `Pipeline([("cleaner", FoldCleaner), ("engineer", EngineerTransformer)])`, output as pandas.
  **Unfitted**: you call `fit` (or `clone` then `fit` inside each CV fold).
- Lines 254–293 `CleanerTransformer` (v1 style): `fit` re-plans *all* cleaning on the training
  table, and `transform` applies it, except removing repeated rows (removing rows from new data
  would silently lose rows the caller expects back).
- Lines 296–500 `EngineerTransformer`, the engineering steps.
  - `_fold_signals` (323–334): recomputes the signals (e.g. correlation) from these rows, but
    keeps each column's plan kind (`column_kinds`).
  - `fit` (336–376): re-plans engineering on the training rows (`plan_engineer`), overlays
    the person's choices, keeps any column the person chose to leave out as "leaky"
    (352–359), checks the target is present if target encoding is needed, runs
    `apply_engineer` to learn the encodings/scaling (`specs_`), and records the output columns.
  - `fit_transform` (378–396): **target encoding is cross-fitted** here. Each training row
    gets a value computed *without its own target*, so the model cannot cheat by reading its
    own answer. New rows passed to `transform` use the mapping learned in `fit`.
  - `transform` (398–432): replays each learned step from `specs_`:
    - `expand_datetime`: split dates into parts.
    - `ordinal_encode`: map each category to a number, unknown → −1.
    - `one_hot_encode`: one yes/no column per category.
    - `target_encode_cross_fit`: map each category to the mean target, unknown → overall mean.
    - `linear_numeric`: optional log1p, then standardise.
    - `drop_leaky`: remove the column.
    Then it adds the generated columns (`_generated`), and finally makes sure the output has
    exactly the columns seen at fit time (missing one-hot columns are filled with False).
  - `_generated` (434–495): interactions, cyclical date features, cluster label/distance,
    and text features (length, word count, has-text, TF-IDF), all from the fitted state.
- Lines 503–522 `build_pipeline`: the unfitted v1-style pipeline.
- Lines 525–556 `build_pipeline_two_phase(X, y, target, ...)`: the v1-style **fitted**
  pipeline. The cleaner is fitted first, then the engineer on the cleaned columns (hence "two
  phase": the engineer needs to know which columns survived cleaning).
- Lines 559–572 `transform_new_data(pipeline, new_df, target)`: transform new rows and carry
  the target column through untouched.
- Lines 575–595 `run_assembler(df, target, ...)`: v1-style "fit on everything and return the
  table + pipeline".

**What changed in Phase 7b and why.** Before 7b, `apply()` fitted the pipeline on all rows. If you
then evaluated a model with cross-validation, the fill values, limits and encodings had already
seen the test rows, so the scores came out too good. Now `apply()` returns the pipeline
**unfitted** and you fit it inside each fold. `tests/unit/test_fit_hygiene.py` proves it.

**Known limits.**
- `transform` fills missing expected columns with `False` even when the column was numeric.
- An ID-typed text column can still be target-encoded instead of dropped (seen on
  `carData_TitanicSurvival.csv`, see `docs/v2_vs_v1_scores.md`).

---

### `src/preflight/cli.py` (1 line)

A placeholder docstring (`"""CLI module."""`). There is no command-line tool in 2.0; it is
planned for Phase 8. `typer` is listed as a dependency for it but is not imported anywhere yet.

---

## 2.2 `src/preflight/ingest/` — reading the raw file

**What this folder does.** It answers three questions about a file before any data is
interpreted:
1. **Encoding:** how are the bytes turned into letters? (UTF-8, Latin-1, ...)
2. **Dialect:** what separates the cells? (comma, semicolon, tab, pipe) and how are quotes written?
3. **Header:** which row holds the column names, if any? Are there title lines above or
   footer/total lines below?

Then it loads the table with **every cell kept as text**, so nothing is guessed yet (no
"NA" strings turned into blanks, no numbers parsed). Typing happens later in `infer/`.

**Why (design choice).** v1 called `pandas.read_csv(path)` and hoped. That breaks on files with
no header row (`adult.data`, `iris.data`: the first data row becomes the column names) and on
semicolon files (`winequality-red.csv`: everything lands in one column). v2 detects each
property, records it as a Decision with a confidence, and asks when unsure.

---

### `ingest/__init__.py` (185 lines)

**Purpose.** Makes the four detectors importable as `preflight.ingest.detect_encoding` etc.,
and offers `ingest()`, a one-call helper.

**Walk-through.**
- Lines 26–41: re-export `detect_encoding`, `detect_dialect`, `detect_header`, `load_raw` and
  their result types.
- Lines 44–185 `ingest(source, chunksize=None, user_overrides=None) -> (data, TableSchema)`:
  1. Encoding (72–83): use `user_overrides["encoding"]` if given, otherwise detect.
  2. Dialect (85–100): use `user_overrides["delimiter"]` (and quote settings) if given.
  3. Header (102–126): detect, and replace the row number if `header_row_index` is overridden.
  4. Load (128–135), with optional `chunksize` for streaming big files.
  5. Build the `TableSchema` (137–183), keeping both the detected values and the overrides.

**Connections.** `plan.py`'s `ReadingStage` calls the four detectors directly, not `ingest()`.
`ingest()` is used by tests (`tests/unit/test_ingest*.py`, `tests/property/test_ingest_roundtrip.py`)
and the ingest corpus suite.

---

### `ingest/encoding.py` (199 lines)

**Purpose.** Decide the file's text format (encoding).

**Thresholds and why.**
- Line 16 `MAX_ENCODING_ALTERNATIVES = 4`: when unsure, show the person at most 4 other options.
- Line 22 `ENCODING_RIVAL_CHAOS_MARGIN = 0.10`: two candidate encodings are "rivals" when
  charset-normalizer's mess score ("chaos") differs by 0.10 or less **and** they decode the
  sample to *different* text. Old single-byte code pages (cp1252 vs cp775, for example) often tie
  like this. A rival makes the result low-confidence instead of a silent guess (added in Phase 7b).
- Line 72 `confidence_threshold=0.70`: below this, flag as low confidence.
- Line 71 `sample_size=131072`: only the first 128 KB of the file is examined.

**Walk-through.**
- Lines 25–31 `BOM_SIGNATURES`: byte patterns at the very start of a file that *announce*
  the encoding (a "byte order mark"). UTF-32 is checked before UTF-16 because UTF-32-LE starts
  with the same two bytes as UTF-16-LE.
- Lines 34–43 `EncodingResult`: `encoding`, `confidence`, `has_bom`, `low_confidence`,
  `raw_detected`, `alternatives`.
- Lines 46–66 `normalize_encoding_name`: one spelling per encoding (`latin1` → `iso-8859-1`,
  `windows-1252` → `cp1252`, plain ASCII → `utf-8` because ASCII is a subset of UTF-8).
- Lines 69–199 `detect_encoding(source, sample_size, confidence_threshold) -> EncodingResult`.
  The order matters:
  1. Read the sample (91–104). Accepts a path, bytes or an open binary file.
  2. Empty input (106–114) → UTF-8, confidence 0.5, low confidence.
  3. BOM present (116–125) → that encoding, confidence 1.0.
  4. All bytes below 128 (127–136) → plain ASCII → UTF-8, confidence 1.0.
  5. Decodes cleanly as UTF-8 (138–150) → UTF-8, confidence 1.0. Random bytes almost never
     happen to be valid UTF-8, so this is a very reliable test. (The `except` here is one of the
     10 justified "swallowed" exceptions.)
  6. Otherwise ask `charset_normalizer` (152–190). Confidence = 1 − chaos/2. Low confidence when
     confidence < 0.70, when the language check ("coherence") is 0, or when a rival exists. The
     best guess is **kept** (never swapped), and the alternatives are listed for the person.
  7. Nothing matched (192–199) → Latin-1 (it can decode any byte), confidence 0.3, low confidence.

**Known limits.** Only the first 128 KB is examined. Most of the corpus dialect misses are
here: Latin-1 files labelled as UTF-8 or the reverse (`docs/review_notes/phase7b/error_analysis.md`).

---

### `ingest/dialect.py` (267 lines)

**Purpose.** Decide the separator and the quoting style.

**The idea in one sentence.** Try each candidate separator. The right one splits every row into
the **same number of cells**, more than one.

**Walk-through.**
- Line 16 `CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]`: the only separators tried.
- Lines 19–30 `DialectResult`: separator, quote settings, `confidence`, `flagged`, `margin`,
  `field_count_mode` (the most common number of cells per row) and `row_regularity`.
- Lines 33–61 `_read_sample_lines`: up to 1000 lines, decoded with the chosen encoding
  (unreadable bytes are replaced rather than crashing).
- Lines 64–125 `_score_delimiter(lines, delimiter, ...) -> (score, mode, regularity)`:
  - Parse the lines with Python's `csv` module using this separator (85–95). If it cannot
    parse, the score is 0.
  - `mode` = most common cells-per-row. If it is 1, the separator did not split anything →
    score 0 (104–106).
  - `regularity` = share of rows with exactly `mode` cells.
  - A penalty for the spread of cell counts: `stdev / mode`, capped at 1 (117–119).
  - A small bonus for more columns: `0.7 + 0.05 × mode`, capped at 1 (122). This breaks ties
    in favour of the separator that gives a real table.
- Lines 128–161 `_detect_quoting_style`: `\"` in the text means backslash escaping; `""` means
  doubled quotes (the CSV standard). If there are no double quotes but there are single quotes,
  single quotes are tried as the quote character (152–159).
- Lines 164–267 `detect_dialect(source, encoding, sample_size=100, verify_size=1000,
  margin_threshold=0.15)`:
  1. Score all candidates on the first 100 lines (205–217).
  2. **Verify** the winner and runner-up on up to 1000 lines and average the two scores
    (219–231). This catches files whose first lines look different from the rest.
  3. All scores 0 → a single-column file. Return comma, flagged (233–245).
  4. Flag when the winner's lead (`margin`) is under 0.15, or the winning score is under 0.60
    (247–250). Below those values the choice is not clear enough to make silently.

**Known limits.** Only four separators. A file whose separator also appears inside unquoted
cells can still fool it. 4 of the 13 tuning-set dialect misses are separator misses (all
semicolon files read as comma).

---

### `ingest/header.py` (317 lines)

**Purpose.** Find the column-names row, and the junk rows around the table.

**The idea.** Column names are words, while the data under them is often numbers or dates. A
row that contains decimals or negative numbers is data, not names.

**Walk-through.**
- Lines 16–28: helper patterns. `YEAR_PATTERN` (1800–2099; years can appear in headers, e.g.
  "2019"), date shapes, yes/no words, `TOTAL_MARKERS` (total, average, ...) and
  `COMMENT_PREFIXES` (`#`, `//`, ...).
- Lines 31–43 `HeaderResult`: `header_row_index` (None = no names row), `column_names`,
  `confidence`, and lists of title/footer/blank/total/ragged rows.
- Lines 46–67 `_is_numeric`, `_is_float_or_negative`: number tests that accept `%`, `$`, `€`
  and `£`.
- Lines 70–83 `_classify_token_type`: labels a cell as empty, boolean, year, date, numeric or string.
- Lines 86–317 `detect_header(...)`:
  1. Read up to 150 lines and parse them with the detected dialect (114–154).
  2. Note blank rows (156–172) and the most common cells-per-row (174–178).
  3. **Title rows** (180–206): from the top, skip lines that start with a comment prefix, or
     that hold a single cell while the table has several columns (a banner like "Sales report 2020").
  4. The next line is the **candidate** header (208–211); the 50 lines after it are sample data.
  5. **Footer and total rows** (213–227): among the last 3 lines, a first cell such as "Total"
     or "average:" marks a total row. "source:", "note:", "disclaimer:", or a single-cell line
     marks a footer.
  6. **Is the candidate a header?** (229–271)
     - Check 1: a cell with a decimal point or a minus sign → it is data (confidence 0.98).
       This is how `iris.data` (`5.1,3.5,...`) is recognised as having no header.
     - Check 2: any plain number that is not a year → data (0.95). This catches `adult.data`
       (first cell `39`).
     - Check 3: count the columns where the candidate is a word but the data below is mostly
       numbers or dates. Each one raises confidence: 0.8 + 0.05 per column, capped at 1.0.
  7. **Ragged rows** (273–282): rows whose cell count differs from the header's.
  8. **Names** (284–302): empty names become `unnamed_<i>`, and duplicates get `_<n>` added.
     With no header, the names are `col_0`, `col_1`, ...
  9. `flagged` = ragged, or confidence < 0.80.

**Known limits.** Only the top and the last 3 lines are checked for junk rows. A header made
entirely of numbers (e.g. years only) can be read as data. 2 of 13 tuning dialect misses are
header misses.

---

### `ingest/loader.py` (208 lines)

**Purpose.** Load the table with every cell as plain text.

**Walk-through.**
- Lines 20–28 `RawLoadResult`: `data`, `row_count`, `column_count`, `column_names`,
  `is_chunked` and `ragged_rows_found`.
- Lines 31–41 `_clean_chunk`: set the column names, turn real blanks into `""`, and make every cell `str`.
- Lines 44–208 `load_raw(source, encoding=None, dialect=None, header_result=None, chunksize=None)`:
  - Anything not given is detected (71–113).
  - The `pd.read_csv` settings (119–132) are the key lines:
    - `dtype=str`: every cell is text.
    - `keep_default_na=False, na_values=[]`: the text "NA" stays the text "NA". Deciding what
      means "missing" is `infer/missing.py`'s job, with evidence.
    - `skiprows=title_rows`: skip the banner rows.
    - `on_bad_lines="warn"`: a bad line does not stop the load.
  - Header handling (134–140): with a header row, pandas reads it (it is row 0 after the
    skipped title rows). Without one, the generated names are passed in.
  - Lines 142–149: files that use only `\r` as the line ending (old Mac style) crash pandas'
    fast C parser, so the slower Python parser is used for them.
  - Lines 151–157 `_safe_read_csv`: if the C parser fails, retry once with the Python parser.
  - Lines 159–184: streaming mode (`chunksize`) returns a generator. On a parse error it
    **closes the first reader before re-reading** (a Phase 7b fix: the file handle used to leak).
  - Lines 186–199: drop the detected footer and total rows. Their positions are shifted to
    account for the skipped title and header rows.

**Connections.** Called by `ReadingStage.plan` and `.apply` in `plan.py`.

---

## 2.3 `src/preflight/infer/` — deciding each column's type

**What this folder does.** It takes the all-text table from `ingest/` and decides, for every
column:
1. which values secretly mean "missing" (`NA`, `?`, `-999`, ...), and
2. what the column *is*: one of **int, float, boolean, datetime, categorical, text, ID,
   constant**.

It produces one Decision per column with the evidence (scores for every candidate type) and a
confidence. When two types are too close, or a trap is found (dates that could be day/month or
month/day, numbers with units), it **abstains**: it keeps the column as text and asks.

**How the corpus scores types.** The labels use 7 "canonical" types. `int` and `float` both
count as `numeric` (see `to_canonical_type`).

---

### `infer/__init__.py` (32 lines)

Re-exports `detect_missing_markers`, `MissingMarker`, `infer_column_type`, `infer_table_types`,
`TypeInferenceResult`, `to_canonical_type` and `Decision`, so you can write
`from preflight.infer import infer_column_type`.

---

### `infer/config.py` (37 lines)

**Purpose.** Every named setting for typing in one place. The comment on line 2 says it: change
them here, not in the code that uses them.

| Name | Value | Meaning and why |
|---|---|---|
| `TYPE_SAMPLE_SIZE` | 10,000 | Columns with more values are *scored* on a random sample of 10,000. The winner is then checked against **every** value. This made planning 4–6× faster on large files in Phase 7b without changing any corpus result (2841/2841 column types were the same). |
| `TYPE_SAMPLE_SEED` | 20,241,005 | Fixed seed, so the same file always gives the same sample and the same answer. |
| `UNIT_MAX_LETTERS` | 5 | A unit is up to 5 letters after a number (`12kg`, `3.5 mph`). No unit names are listed anywhere, so detection works for any unit. |
| `UNIT_MIN_REPEATS` | 2 | A unit must appear at least twice... |
| `UNIT_FLAG_SHARE` | 0.15 | ...and on at least 15% of values, to be reported. |
| `UNIT_COLUMN_SHARE` | 0.80 | If one unit is on 80%+ of values and every value is "number + that unit" or a bare number, the column is read as numbers, the unit is stored in the Decision, and the column is flagged (the conversion loses the unit text). |
| `PLUS_CODE_MIN_DIGITS` | 7 | `+` followed by 7+ digits (an international phone number) is a code, never a number, because the `+` would be lost. |
| `TEXT_NAME_HINT_MIN_WORDS` | 5 | A column whose **name** suggests prose (description, comment, title, name, ...) is free text if its values average 5+ words. Changed from 3 in Phase 7c: at 3, person names and place names were wrongly called free text (`docs/tuning_log.md`). |

---

### `infer/missing.py` (220 lines)

**Purpose.** Find "disguised missing values": cells that are not blank but mean "no value".
It only **detects**: nothing is changed here. The cleaning step converts them later, so the
Plan PDF can show the evidence first.

**Walk-through.**
- Lines 24–44 `MissingMarker`: `marker` (the exact text), `confidence`, `reason`, `count` and
  `frequency` (share of rows).
- Lines 48–70 `STANDARD_SENTINELS`: well-known "missing" words with their confidence, e.g.
  `""`, `na`, `n/a`, `null`, `?` (1.0); `none`, `nan`, `-` (0.95); `missing`, `.` (0.90);
  `*`, `unspecified` (0.85). Matched ignoring case and surrounding spaces.
- Line 73 `NUMERIC_SENTINEL_CANDIDATES`: numbers people use for "missing": 999, -999, 9999,
  -99, 99, -1.
- Lines 76–83 `_safe_float`: number parse that ignores `,`, `$`, `€` and `£`.
- Lines 86–220 `detect_missing_markers(values, column_name) -> list[MissingMarker]`. Three passes:
  1. **Standard words** (115–140). `unknown` also counts (confidence 0.85).
  2. **Suspicious numbers in context** (142–195), only if at least 3 other values are numbers.
     A sentinel number is only called "missing" when it does not fit the rest:
     - `-1` when every other value is 0 or more (e.g. an age column).
     - `999`/`9999` when every other value is 150 or less.
     - `-99`/`-999` when every other value is -10 or more.
     So `999` in a price column with values up to 5000 is left alone.
  3. **Odd word in a number column** (197–213): if 80%+ of at least 10 values are numbers, a
     short (≤10 characters) non-number like `n.a.` or `unk` is flagged (0.85).
  - Lines 216–220: sorted by confidence, then count.

**Known limits.** `99` is in the candidate list but no rule ever flags it. The number rules
need at least 3 other numeric values.

---

### `infer/types.py` (791 lines) — the type detector

**Purpose.** Score every candidate type for a column, handle the traps, pick a winner or
abstain, and write the Decision.

**The core idea: "parse success rate".** For each candidate type, try to read every value as
that type. The score is the share that succeed. A column where 100% of values read as whole
numbers scores int = 1.0. Then special rules adjust the scores for the known traps.

**Walk-through.**
- Lines 42–70 `TypeInferenceResult`: what one column's inference returns. It holds the type,
  confidence, `runner_up` (the second-best type, filled when the lead is under 0.30), `abstained`,
  `needs_attention`, `anomalies` (trap tags), `missing_markers`, `failed_samples`, `evidence`
  and the `decision`.
- Lines 73–87 `to_canonical_type`: int/float/numeric → `numeric`, id → `ID`, everything else
  unchanged. Used by the accuracy scripts to compare with the labels.
- Lines 94–131: the patterns.
  - `BOOL_LITERAL_PAIRS`: true/false, yes/no, y/n, t/f, 1/0.
  - `ID_NAME_PATTERNS` (line 103): id, _id, id_, pk, key, index, code, fips, zip, postal,
    rownames. A column name containing one of these hints at an ID.
  - `_RE_UNIT_VALUE` (107–110): "optional currency sign, a number, then optional letters (up to
    5) or %".
  - `DATE_PATTERNS` (112–121): ISO dates (`2020-01-31`), with times, `d/m/yyyy`, `d-m-yyyy`,
    `yyyy/m/d`, `Jan 5, 2020`, `5 Jan 2020` and `2020-01`.
  - Number patterns (124–130): plain int, int with thousands commas, float, float with
    thousands commas, decimal comma (`12,5`, `1.234,56`), and the leading-zero shape.
- Lines 138–278: one small parser per type.
  - `_parse_bool` (138–139).
  - `_is_plus_code` (142–144).
  - `_parse_int` (147–174) returns (parses, has_leading_zero). **Leading-zero trap:** `02138`
    is not an int, because reading it as 2138 would destroy a ZIP code.
  - `_unit_of` (177–182) and `strip_unit` (185–196): find a value's unit, and get the number part.
  - `_parse_float` (199–241) returns (parses, is_decimal_comma, unit, has_leading_zero). A
    value with a unit is **not** a float here. The unit is reported separately, so units are
    never stripped silently.
  - `_check_date_ambiguity` (245–271): **date trap.** For `a/b/yyyy` dates, if any `a` > 12 the
    format must be day-first, and if any `b` > 12 it must be month-first. If **every** date has
    both parts ≤ 12, the order cannot be known, so it is ambiguous.
  - `_parse_datetime` (274–278): matches one of the date patterns.
- Lines 285–333: speed helpers.
  - `_value_masks` (299–311) parses each **distinct** value once (`pd.factorize`) and spreads the
    result back to every row. This is much faster when values repeat.
  - `_sample_positions` (314–319): the seeded 10,000-row sample, or None for small columns.
  - `_first_failures` (322–324): the first 5 values that fail.
  - `_unit_counts` (327–333).
- Lines 340–737 `infer_column_type(values, column_name, margin_threshold=0.15) ->
  TypeInferenceResult`:
  - **Step 1** (around 361–407): find missing markers and remove them. A column that is entirely
    missing → `constant`, confidence 1.0.
  - **Step 2, scores** (409–556):
    - A. Constant (425–426): 1.0 if there is only one distinct value.
    - B. Boolean (428–431): only if all distinct values fit **one** pair (e.g. only yes/no).
    - C. Int (433–443), with the leading-zero and plus-code anomalies (at least 2 values, and at
      least 2% of values).
    - D. Float (445–475), with decimal comma (10%+ of values) and units. A **unit column** is
      detected when one unit covers 80%+ of values and every value is "number + that unit" or a
      bare number.
    - E. Datetime (477–484): share of values matching a date pattern. If it is 0.70+, check the
      day/month ambiguity.
    - F. ID (485–508): leading zeros or plus codes → 0.98 (if 80%+ of values are different) or
      0.85. An ID-like name with 85%+ different values (or a counting-up run) → 0.98. A
      counting-up run of whole numbers with every value different → 0.95. Otherwise 0.
    - G. Text vs categorical (510–531): look at the share of values that are words (not
      numbers, dates or yes/no), the average length and the average word count. It is **text**
      if the average length is over 40, or over 25 with 5+ words, or the name suggests prose
      and there are `TEXT_NAME_HINT_MIN_WORDS` (5) or more words. Otherwise it is
      **categorical**. The winner gets 0.95 × textual share and the loser 0.20 ×.
    - Disambiguation (533–544): a 1/0 column is boolean (int is lowered to 0.70). Leading zeros
      or plus codes set int and float to 0 and ID to at least 0.98.
  - **Step 3, rank** (558–590): sort the scores. The **runner-up** is the best type from a
    *different family* (int and float are one family, text and categorical another). `margin` =
    winner − runner-up.
  - **Step 3b, verify** (592–600): with a sample, count how many values in the **whole** column
    fail the winning type, and record that as an anomaly. A rare bad value outside the sample is
    never lost.
  - **Step 4, decide** (602–674):
    - Ambiguous dates → **abstain**: keep as text, needs attention.
    - Unit column → read as numbers, keep the unit in the evidence, needs attention, confidence 0.90.
    - Mixed units → **abstain** (keep as text), needs attention.
    - Winner score under 0.60, or margin under 0.15 when the winner is text or categorical →
      **abstain** (keep as text), needs attention. 0.60 means "most values do not even fit the
      best type". 0.15 means "too close to call".
    - Otherwise a **clear winner** (658–674). `needs_attention` is still set if there is any
      anomaly, confidence < 0.90, or margin < 0.25. The action is `cast_to_int/float/boolean/datetime`,
      `keep_as_id`, `keep_as_constant` or `keep_as_categorical/text`. The safe `default` is always
      `keep_as_string`.
  - Lines 676–707: failed samples and the **evidence** dict: every candidate score, top score,
    runner-up score, margin, counts, and trap flags.
  - Lines 709–737: build the Decision (`kind` is `type_abstain` or `type_inference`) and the result.
- Lines 740–791 `infer_table_types(df, schema) -> TableSchema`: runs `infer_column_type` on every
  column and copies the results into the schema's column entries (matched by name, or by position
  for headerless files). It appends all Decisions to `schema.decisions`.

**What changed in Phase 7b / 7c and why.**
- 7b: the sample-then-verify speed-up; generic unit detection (no unit names in the code); `+`
  phone codes treated as IDs.
- 7c: the name hint for free text needs 5 words instead of 3.

**Known limits.**
- The biggest miss: ID columns (mostly row numbers) are often typed as numbers. On the tuning set,
  112 of 149 misses are of this kind, and 111 of them are flagged for review.
- The narrow-margin abstain rule names `"numeric"`, which is never a candidate name (int and
  float are), so close int/float-vs-other calls are not abstained by that rule. They can still be
  flagged by the margin < 0.25 rule.
- A CSV cannot record that a column was text, so 10-digit phone numbers without `+` look like
  integers (the strict xfail test `test_phone_number_like_column_not_coerced`).

---

## 2.4 `src/preflight/clean/` — cleaning

**What this folder does.** It fixes the table so a model can use it: it turns "missing" words
into real blanks, reads numbers out of text, removes repeated rows and useless columns, tidies
category text, fills blanks and limits extreme numbers.

**The pattern every file follows.** Each step has two functions:
- `plan_<step>(df, ...) -> list[Decision]`: looks at the table and proposes Decisions. It never
  changes `df`.
- `apply_<step>(df, decisions) -> df`: does exactly what the Decisions (with their *current*
  actions) say. If a person changed an action to "keep", apply does nothing for it.

Learned values (the median used, the clipping limits, the rare-value list) are stored **in the
Decision's evidence**, so the plan shows them and the apply step just reads them back.

**Where it came from.** The rules are v1's `cleaner.py`, ported in Phase 6 (see
`docs/port_log.md`, `docs/port_regressions.md`). The thresholds are v1's numbers. New in v2:
every step is a Decision, and three "impact" thresholds ask a person when a step would remove
or change a lot.

**Run order** (in `cleaner.py:plan_clean_full`, the same as v1):
markers → read numbers → signals → repeated rows → drop columns → tidy text and rare values →
fill blanks → clip extremes.

---

### `clean/__init__.py` (1 line)
Just a docstring. Import the step modules directly.

---

### `clean/config.py` (40 lines)

**Purpose.** `CleanerConfig`, every cleaning threshold in one dataclass. `DEFAULT_CLEANER_CONFIG`
(line 40) is the instance used when none is passed.

| Setting | Default | Meaning |
|---|---|---|
| `drop_missing_threshold` | 0.60 | Drop a column if more than 60% of it is blank (v1). |
| `impute_strategy_threshold` | 0.30 | Under 30% blank: fill with the median or the most common value. At 30% or more: numbers are only flagged, categories get the label `__missing__` (v1). |
| `winsorize_missing_limit` | 0.30 | Do not clip a column that is more than 30% blank (v1). |
| `rare_category_threshold` | 0.01 | A category value under 1% of rows is "rare" (v1). |
| `outlier_method` | `"iqr"` | `iqr` or `zscore`. |
| `cardinality_threshold` | 20 | 20+ different values = high-variety category (v1). |
| `dedupe_impact_threshold` | 0.02 | Removing more than 2% of rows as repeats → ask a person. New in v2. |
| `column_drop_impact_threshold` | 0.25 | Dropping more than a quarter of the columns → ask. New in v2. |
| `outlier_impact_threshold` | 0.05 | Clipping more than 5% of a column's values → ask. New in v2. |

**Why the impact thresholds exist.** Dropping one ID column is routine. Dropping 40% of the
rows or half the columns is usually a sign that something was misread, so a person should
confirm. When an impact threshold fires, the Decision's **default flips to the safe option**
(`keep_rows`, `keep_column`, `keep_outliers`).

---

### `clean/_common.py` (31 lines)

- Lines 11–18 `py(value)`: turns numpy/pandas numbers into plain Python numbers, so evidence can
  be saved as JSON.
- Lines 21–31 `need_signals(df, signals, schema, target, cardinality_threshold)`: use the signals
  if given, otherwise compute them (`signals.compute_signals`). Every planner calls this, so the
  planners work on their own (in tests) and also share one computation (in `plan_clean_full`).

---

### `clean/markers.py` (90 lines)

**Purpose.** Turn the disguised missing values found by `infer/missing.py` into real blanks.

- Lines 18–23 `marker_strings`: the list of marker texts from the schema (dicts or strings),
  trimmed, without empty strings.
- Lines 26–42 `apply_missing_markers_to_series`: replaces each marker with `NaN`.
- Lines 45–80 `plan_markers(df, schema)`: one `convert_missing_markers` Decision per column
  whose markers **actually occur** in the data. The evidence holds the markers and the cell count,
  and the confidence is the column's typing confidence. It never needs attention, because the
  evidence was already shown at typing time.
- Lines 83–90 `apply_markers`.

---

### `clean/cleaner.py` (267 lines)

**Purpose.** Runs the cleaning steps in order, and reads numbers out of text.

**Walk-through.**
- Lines 34–48 `coerce_string_dates_to_datetime`: v1 helper. A value that cannot be parsed
  becomes "no date" (`NaT`) instead of raising.
- Lines 51–119 `coerce_numeric_string_column(series) -> (numbers or None, unit or None)`:
  - Only for text columns.
  - Ignores common missing words (line 70).
  - Pattern (line 75): a number, then any text after it (the unit).
  - Succeeds only if **95%+** of values start with a number (line 90) **and** the text after the
    numbers is the same unit 90%+ of the time, with at most 5 different units (91–95). This
    stops it from turning a genuine text column into numbers.
  - Returns the numbers and the most common unit.
- Lines 122–166 `plan_coerce(df, schema, target)`: for columns the schema types as numbers but
  that are still text (e.g. "12 kg"), propose `coerce_numeric`. The plain-language text names
  the unit being removed.
- Lines 169–178 `apply_coerce`.
- Lines 181–186 `CleanPlan`: `decisions`, `cleaned` (the simulated table) and `signals`. The
  engineer planner needs the cleaned table and the signals.
- Lines 189–198 `plan_clean`: returns only the Decisions.
- Lines 201–251 `plan_clean_full(df, schema, target, config, target_column, task) -> CleanPlan`:
  **simulates** every step on a private copy, so each step sees the table as the previous step
  left it. This matters: rare values are found after text tidying, fills are computed after drops,
  and clipping limits after fills. The signals (line 229) are computed once, after markers and
  number reading.
- Lines 254–267 `apply_clean`: runs all the apply functions in the same order.

**Connections.** `plan_clean_full` is called by `plan.py:_ensure_prep_plan` (planning) and
`assemble.CleanerTransformer.fit`. `apply_coerce` is called by `assemble.apply_stateless` and
`plan.FixingStage`.

---

### `clean/dedupe.py` (88 lines)

- Lines 12–20 `remove_duplicate_rows`: pandas `drop_duplicates` (all columns must match).
- Lines 23–69 `plan_dedupe(df, config)`: if there are repeats, one `remove_duplicates`
  Decision. If more than 2% of rows are repeats, it needs attention, the text asks
  "...? This is a large share, so please confirm." and the default becomes `keep_rows`.
- Lines 72–88 `apply_dedupe`: removes them only if the action in force is `remove_duplicates`.

**Why repeated rows are not removed from new data.** In the pipeline (`CleanerTransformer.transform`)
this step is skipped. If you pass 100 rows to predict on, you expect 100 predictions back.

---

### `clean/drops.py` (167 lines)

- Lines 17–22 `should_drop_column`: missing rate > threshold.
- Lines 25–65: small helpers that drop given columns (v1 API).
- Lines 68–148 `plan_drops(df, schema, target, config, target_column, signals)`. In v1 order, one
  reason per column:
  1. more than 60% blank → `drop_high_missingness`;
  2. kind is NUMERIC_ID → `drop_id` ("a unique label for each row, not useful for learning");
  3. kind is CONSTANT → `drop_constant`.
  The **target is never dropped** (line 91). If the dropped columns are more than 25% of all
  columns, **every** drop needs attention and defaults to `keep_column` (106–124). The action is
  always `drop_column`, and the kind tells you why.
- Lines 151–167 `apply_drops`: drop the columns whose action in force is `drop_column`.

**Known limit.** Text-valued ID columns become CATEGORICAL_HIGH (see `signals.kind_from_schema`)
and are **not** dropped. They are encoded instead.

---

### `clean/normalize.py` (167 lines)

- Lines 18–28 `normalize_category_values(series)`: lower-case and trim each text value. Blanks stay blank.
- Lines 31–41 `identify_rare_categories`: values under 1% of non-blank rows.
- Lines 44–53 `group_rare_categories`: replace rare values with `__rare__`.
- Lines 56–142 `plan_normalization(df, schema, target, cardinality_threshold, config, signals)`:
  - For every category column, count how many entries tidying would change. If any would, propose
    `normalize_categories`.
  - **Phase 7c fix (lines 84–90):** the comparison is done on plain values
    (`astype(object)`). The typed table stores low-variety text as pandas `category`. Lower-casing
    `['Male','Female']` makes a *different* category set, and comparing two Categoricals with
    different sets raises `TypeError: Categoricals can only be compared if 'categories' are the
    same`. That crashed the pipeline on most real files (tips, diamonds, titanic, adult).
    Regression test: `tests/unit/test_normalize_categorical.py`.
  - High-variety columns with rare values also get `group_rare_categories` (confidence 0.95).
    The rare list comes from the signals, i.e. the values *before* tidying, as in v1.
- Lines 145–167 `apply_normalization`.

**Fit hygiene.** Tidying learns nothing (runs on the full table). Rare-value grouping learns a
list from the data, so in the returned pipeline `FoldCleaner.fit` recomputes the list from the
training rows only.

---

### `clean/impute.py` (214 lines)

- Lines 18–25 `add_missing_indicator`: a new yes/no column `<col>_missing`, True where the value was blank.
- Lines 28–40 `median_impute`, 43–55 `mode_impute`, 58–62 `constant_fill` (label `__missing__`).
- Lines 65–184 `plan_impute(...)`. v1 rules by kind:
  - **Number** with blanks: add a `<col>_missing` flag. Under 30% blank → also fill with the
    median. 30%+ → flag only (the blanks stay).
  - **Low-variety category**: flag. Under 30% → fill with the most common value. 30%+ →
    `__missing__`.
  - **High-variety category**: `__missing__` fill, no flag.
  - yes/no, text, dates: left alone (v1 did the same).
  The fill value is stored in the evidence and shown in the plan ("Fill blanks in 'age' with the
  middle value (28)").
- Lines 187–214 `apply_impute`.

**Why a "was blank" flag.** Whether a value was missing can itself be useful to the model
(for example, people who skip an income question). The flag keeps that information after the
blank is filled.

**Known limits.** A number column that is 30–60% blank keeps its blanks. Models that cannot
take blanks need an imputer after PreFlight. Line 12 re-exports `apply_missing_markers_to_series`
for old imports (one of the 3 remaining `noqa` comments).

---

### `clean/outliers.py` (170 lines)

- Lines 16–44 `winsorize_outliers(series, method)`: v1's clipper (IQR: Q1 − 1.5·IQR to
  Q3 + 1.5·IQR; z-score: mean ± 3 standard deviations).
- Lines 47–146 `plan_outliers(...)`: for each number column (not the target, not yes/no,
  at most 30% blank, spread not zero), compute the limits and count values outside them.
  - A Decision is made **even when nothing is outside the limits now**, because the limits also
    apply to new data.
  - More than 5% clipped → needs attention, default `keep_outliers`.
  - The limits are stored in the evidence.
- Lines 149–170 `apply_outliers`: clips to the stored limits. Nullable integer columns are first
  widened to `Float64`, because the limits can be fractions (lines 163–167).

**Why IQR by default.** It is not pulled around by the extreme values themselves (a mean and
standard deviation are), so it works on skewed data such as prices.

---

## 2.5 `src/preflight/engineer/` — turning columns into model inputs

**What this folder does.** Models need numbers. This folder turns cleaned columns into numbers
the model can use:
- Categories become numbers (**encoding**).
- Dates are split into parts.
- For linear models, numbers are rescaled, and a long tail is compressed with a logarithm.
- Optionally (`FeatureConfig`), extra columns are generated: combinations, cluster groups,
  date cycles, text statistics.

It also flags numeric columns that may **leak** the answer, and notes overlapping columns.

**The same plan/apply pattern as cleaning.** `plan_engineer` simulates every step on a copy and
records Decisions. `apply_engineer` replays them. Both use one shared function (`_apply_one`), so
planning and applying cannot drift apart.

**The model hint.** `model_hint="tree"` or `"linear"` changes the choices:
- Tree models (random forests, gradient boosting) cope with categories numbered 0, 1, 2... and
  do not care about scale.
- Linear models need one yes/no column per category (one-hot) and rescaled numbers.

---

### `engineer/__init__.py` (1 line)
Docstring only.

---

### `engineer/config.py` (83 lines)

- Lines 10–39 `FeatureConfig`: the **opt-in** extras. All are off by default.
  - `interactions` (+ `interaction_top_k=5`, `interaction_types=["ratio","product"]`).
  - `datetime_cyclical`, `datetime_deltas`, `datetime_reference_col`.
  - `clustering` (+ `cluster_k="auto"`, `cluster_features="numeric_only"`).
  - `text_features` (+ `text_tfidf`, `text_tfidf_top_k=20`).
  - `__post_init__` (27–39) rejects bad values with plain messages.
- Lines 43–57 `EngineerConfig`, the thresholds:
  - `leakage_threshold = 0.95`: a numeric column whose correlation with the target is above 0.95
    in absolute value is a suspected leak (v1's number).
  - `vif_note_threshold = 10.0`: a column with an overlap score above 10 gets a note (new in v2,
    never tuned). 10 is the common rule of thumb for "this column is mostly a combination of others".
  - `skew_threshold = 1.0`: a skewness above 1 gets a log transform for linear models (v1).
- Line 60 `DEFAULT_ENGINEER_CONFIG`.
- Lines 63–83 `PRESETS`: v1's `fast` and `thorough` settings, kept for the v1 regression tests
  (re-exported by `types.py`).

---

### `engineer/numeric.py` (46 lines)

- Lines 11–23 `needs_log_transform(series, skew_threshold)`: True when |skewness| > 1 and every
  value is above −1 (log1p needs x > −1).
- Lines 26–30 `log1p_transform`: `log(1 + x)`. It shrinks big values a lot and small values a
  little, so one huge salary does not dominate a linear model.
- Lines 33–46 `standard_scale`: (x − mean) / standard deviation. It returns the mean and std so
  new data is scaled the same way. A zero spread returns the column unchanged with std = 1
  (avoids dividing by zero).

---

### `engineer/encoding.py` (90 lines)

- Lines 11–20 `ordinal_encode`: sorts the categories and numbers them 0, 1, 2... The mapping is
  stored, and an unseen category later becomes −1.
- Lines 23–30 `one_hot_encode`: one True/False column per category, named `<col>_<value>`,
  sorted so the order is always the same.
- Lines 33–90 `target_encode_cross_fit(series, target, n_folds=5, smoothing=10.0)`: replaces
  each category with **the average target for that category**. This is useful for columns with
  many categories (one-hot would create hundreds of columns).
  - The danger: if a row's own target is included in its category's average, the model is
    partly reading the answer. So the rows are split into 5 parts, and each part is encoded
    using averages from the **other 4** parts only (lines 71–82). This is called **cross-fitting**.
  - **Smoothing** (lines 77–79): rare categories are pulled toward the overall average:
    `(count × category mean + 10 × overall mean) / (count + 10)`. A category seen twice
    cannot get an extreme value from two rows.
  - Text targets are first turned into numbers by sorted label (lines 50–53).
  - Fewer than 5 rows → one smoothed mapping for all (lines 63–69).
  - It returns the cross-fitted values for the training rows **and** a full mapping for new data.

---

### `engineer/datetime.py` (102 lines)

- Lines 16–42 `expand_datetime(series) -> DataFrame`: `<col>_year`, `_month`, `_day`,
  `_dayofweek`, `_hour`, `_is_weekend`. Text dates are parsed first. A missing date gives
  missing parts.
- Lines 45–102 `generate_datetime_cyclical_features(df, dt_columns, config)`: optional extras.
  - Cyclical (64–76): sine and cosine of month/12 and weekday/7. **Why:** December (12) and
    January (1) are neighbours, but as plain numbers they are far apart. On a circle they are close.
  - Deltas (78–87): days between every pair of date columns.
  - Reference column (89–100): days from a chosen date column to each other date column.

---

### `engineer/vif.py` (153 lines)

- Lines 17–54 `compute_vif_scores(df, numeric_columns)`: the **variance inflation factor**
  (overlap score) for each number column: how much of it can be predicted from the other number
  columns. It reads the diagonal of the inverse correlation matrix. Only the 50 columns with the
  largest variance are used (for speed). A matrix that cannot be inverted leaves all scores as None
  (a justified swallowed exception).
- Lines 57–69 `compute_correlation_with_target`: Pearson correlation, **numbers only**.
- Lines 72–81 `flag_leakage_suspect`: |correlation| > 0.95.
- Lines 84–153 `plan_vif_and_leakage(df, signals, target, config)`:
  - **Suspected leakage** (103–129): action `keep_feature` (v1 kept such columns too), always
    needs attention, and the question offers `drop_leaky_feature`. Text: "Column 'x' tracks the
    value to be predicted almost exactly (match 0.97). It may give away the answer. Keep it, or
    leave it out?"
  - **Feature overlap** (131–153): a note only, never asks, keeps the column.

**Known limit.** Text or yes/no copies of the target are not checked (correlation needs numbers).
Example: `alive` vs `survived` in `titanic.csv`.

---

### `engineer/features.py` (259 lines)

- Lines 20–51 `compute_mutual_info_with_target`: how much knowing this column tells you about the
  target (works for non-linear links too). Text columns are ordinal-coded first. Any failure →
  None (justified swallowed exception: it is an optional signal).
- Lines 54–74 `rank_interaction_columns`: the top-k number columns by
  max(|correlation|, mutual information).
- Lines 77–113 `generate_interaction_features`: for each pair of top columns, `a_div_b`
  (division by 0 gives a blank), `a_times_b` and `a_minus_b`. It never overwrites an existing column.
- Lines 116–178 `generate_cluster_features`: k-means clustering on the number columns (blanks
  → 0). With `cluster_k="auto"`, it tries k = 2..10 and keeps the best **silhouette score** (how
  well separated the groups are). It adds `cluster_label` and `cluster_dist_to_centroid`, and
  returns the fitted model for new data.
- Lines 181–259 `generate_text_features`: `<col>_char_length`, `<col>_word_count`, `<col>_has_text`,
  and optionally the TF-IDF scores of the top-k words (`<col>_tfidf_<word>`). Name collisions are
  skipped and reported.

---

### `engineer/engineer.py` (352 lines)

- Lines 41–49 `_target_numeric`: the target as numbers (text targets → sorted label codes, as in v1).
- Lines 52–63 `_cfg_from_evidence`: rebuilds a `FeatureConfig` from a Decision's evidence, so
  the Decision alone is enough to replay the step.
- Lines 66–171 `_apply_one(work, d, specs, target, target_numeric, df_in)`: applies **one**
  Decision and records what was learned in `specs` (mappings, scale parameters, fitted models).
  `keep_feature` does nothing. The target is never transformed.
- Lines 174–188 `_mk`: shorthand to build an engineer Decision (never needs attention).
- Lines 191–269 `plan_engineer(df, schema, target, model_hint, cardinality_threshold,
  feature_config, task, signals, config) -> list[Decision]`:
  - Rejects any `model_hint` other than tree/linear (208–209).
  - Leakage and overlap Decisions first (218).
  - Then for each column, by kind (229–264):
    - real date column → `expand_datetime`;
    - low-variety category → `ordinal_encode` (tree) or `one_hot_encode` (linear);
    - high-variety category → `target_encode` (cross-fit, 5 folds). It is **skipped when there
      is no target**;
    - number, linear only → `log_transform` if skewed, then `scale_numeric`.
  - Then the opt-in extras (266–267).
- Lines 272–332 `_plan_generated`: in v1 order: interactions (needs at least 2 top columns),
  date extras, clusters, text features.
- Lines 335–352 `apply_engineer(df, decisions, target) -> (table, specs)`.

**Known limits.**
- Columns of kind TEXT get no Decision unless `text_features` is on, so long free text stays
  as raw text in the output.
- Date columns kept as text (`DATETIME_STRING`) are not expanded.
- yes/no columns pass through unchanged (True/False).
- High-variety category columns stay as text when there is no target.

---

## 2.6 `src/preflight/interact/` — asking a person

**What this folder does.** Some Decisions have `needs_attention=True` (PreFlight was not sure,
or the step would remove a lot). This folder turns those Decisions into **questions**, gets the
answers (from a terminal, a notebook form, a JSON file, or your own function), applies them,
and remembers them so you are not asked again.

**The five ideas.**
1. **One batch.** All questions for a run are collected into one `ReviewSession` and asked
   together, not one at a time during processing.
2. **Real previews.** Each column question shows 3 real values from your file and what each
   option would turn them into.
3. **Modes.** `ask` (ask a person), `defaults` (take every safe default without asking),
   `strict` (stop with an error listing every open question), `auto` (= ask if a person is
   there, otherwise strict). Nothing can ever hang waiting for input on a server.
4. **Never guess an answer.** A missing or invalid answer raises an error naming the question.
   A default is only used in `defaults` mode, and is logged as such.
5. **Remembered by fingerprint.** Answers are saved in `answers.json` under the Decision's
   fingerprint (a short hash of what was found). Same data → same fingerprint → the stored
   answer is reused. If the data changes the evidence, the fingerprint changes and you are asked again.

**New in v2.** v1 never asked anything. This folder was built in Phase 5
(`docs/v2_interact_report.md`).

---

### `interact/__init__.py` (1 line)
Docstring only. The public entry point is `preflight.resolve` (re-exported by the package).

---

### `interact/triggers.py` (306 lines)

**Purpose.** Gives every question a **reason label** (the "trigger category"). The label only
says *why* a question is asked. *Whether* it is asked is decided upstream by `needs_attention`.

- Line 33 `ANOMALY_RATE_THRESHOLD = 0.10`: if more than 10% of the values fail the proposed
  conversion, the label is "high anomaly rate".
- Lines 36–57 `TriggerCategory`, the 8 labels: `low_confidence_structure`, `ambiguous_type`,
  `destructive_action`, `suspected_leakage`, `disguised_missing`, `high_anomaly_rate`,
  `lossy_coercion`, `ambiguous_structure_or_task`.
- Lines 60–83 `TriggerContext` (the column's schema entry and raw values) and `TriggerRule`
  (category + description + test function).
- Lines 90–120 `column_missing_markers`, and `anomaly_rate` (the share of non-missing raw values the
  proposed conversion fails on).
- Lines 123–187: the test functions.
  - `_match_ragged_or_task`: task questions, or uneven rows.
  - `_match_low_confidence_structure`: any reading Decision.
  - `_match_destructive`: a cleaning drop/remove/clip that needs attention.
  - `_match_leakage`.
  - `_match_lossy`: type trap, or leading zeros with a number cast.
  - `_match_disguised_missing`: contextual sentinel markers.
  - `_match_high_anomaly`.
  - `_match_ambiguous_type`: the catch-all for type questions.
- Lines 190–273 `TriggerRegistry`: the rules in order, **first match wins**. The order is
  structure/task, low-confidence reading, destructive, leakage, lossy, disguised missing, high
  anomaly, ambiguous type (the catch-all last). `register` lets later code add rules.
- Lines 289–306 `classify_decision_trigger`: raises if no rule matches. A new kind of Decision
  cannot be silently filed under the wrong reason.

---

### `interact/preview.py` (174 lines)

**Purpose.** Build the before/after preview for a column question.

- Lines 21–22: `BLANK = "(left blank)"` (shown when a value cannot be converted) and
  `PREVIEW_SAMPLES = 3`.
- Lines 27–84 `converter_for(action, evidence)`: a small function that converts one text value
  the same way the typing step would (int, float with decimal comma, yes/no, date). Returns None
  for actions that do not convert.
- Lines 87–93 `after_value`: one value → its converted text, or `(left blank)`.
- Lines 96–125 `pick_samples`: up to 3 different real values, skipping blanks and missing markers.
  It looks at no more than 200 distinct values. **If any option would fail on some value, one
  failing value is included**, so the preview shows the problem, not only the values that work.
- Lines 128–174 `build_column_preview(...) -> {"before": [...], "after": {option: [...]}, "note"?}`.
  If the raw column is not available, it uses sample values already stored in the evidence (and
  says so). With no values at all, the note says so. **Nothing is invented.**

---

### `interact/question.py` (390 lines)

**Purpose.** The `Question` object, the options for each kind of Decision, and the batch.

- Lines 36–37: every question has **2 to 4 options**.
- Lines 40–71: errors. `QuestionError` (a malformed question), `InvalidAnswerError` (the answer
  is not an option; the message lists the allowed answers), `MissingAnswerError`.
- Lines 74–134 `Question`: `question_id`, `decision_id`, `fingerprint`, `category`, `column`,
  `text`, `explanation`, `evidence_summary`, `options`, `option_actions` (option label →
  Decision action), `recommended_default`, `preview`. `__post_init__` (99–115) **refuses** a
  broken question (wrong option count, duplicate options, default not an option, a column
  question without a preview) instead of quietly fixing it. `check_answer` (121–127) accepts
  only an exact option.
- Lines 141–228: **option builders**, one per Decision kind. Each returns candidate actions.
  - Type questions: the candidate types ranked by score.
  - Encoding: the rivals found, then Latin-1 and UTF-8.
  - Separator: `; \t | ,`.
  - Header: "no header".
  - Task: classification / regression / none.
  - Drops: "keep the column"; repeated rows: "keep the rows"; clipping: "keep the outliers".
  - Leakage: keep / leave out.
  - `canonical_action` (160–162) treats `keep_as_string` and `keep_as_text` as the same thing.
- Lines 231–233 `register_question_builder`: a plug-in point for new kinds.
- Lines 236–270 `_assemble_options`: the order is **safe default first, then the proposed action,
  then the candidates**. No two options may mean the same thing or have the same label, and
  there are at most 4. The default and the proposal are never cut. Type questions with too few
  options get generic ones. Otherwise fewer than 2 options raises.
- Lines 290–335 `build_question_from_decision`: puts it all together. The id is
  `q<NNN>_<first 6 characters of the fingerprint>`, e.g. `q001_3fa2b1`. The wording comes from
  `report/templates.py`.
- Lines 342–350 `ReviewSession`: the source path plus the list of questions.
- Lines 353–376 `validate_answers`: **every** question must have a valid answer. An answer for a
  question that does not exist is also an error. A default is never substituted.
- Lines 379–390 `create_review_session`: one question per `needs_attention` Decision.

---

### `interact/modes.py` (138 lines)

- Line 25 `MODES = ("auto", "ask", "defaults", "strict")`.
- Lines 28–43 `StrictResolutionError`: lists every open item (id, column, question) and explains
  the three ways forward: `mode='defaults'`, a review file, or an interactive run.
- Lines 46–56 `in_notebook_kernel`: True in Jupyter. It checks without importing IPython.
- Lines 59–66 `is_interactive_context`: notebook, or both stdin and stdout are a terminal.
- Lines 69–76 `resolve_mode`: an explicit mode wins. `auto`/None → `ask` if interactive,
  otherwise `strict`. **Why:** a script on a server must never sit waiting for typing that
  never comes.
- Lines 83–106 `_finalize` and `apply_answer`: the same action as proposed → status
  `confirmed`; a different action → `changed`; a reused stored answer → `confirmed`. The sentence
  is re-rendered to describe the final action.
- Lines 109–115 `apply_defaults_mode`: every open Decision gets its safe default, and the status
  stays `automatic` (with the source "defaults mode: safe default").
- Lines 118–121 `apply_strict_mode`: raise if anything is open.
- Lines 124–138 `apply_ask_mode`.

---

### `interact/store.py` (108 lines)

- Lines 16–17: `STORE_VERSION = 1`, `STORE_FILENAME = "answers.json"`.
- Lines 20–31 `StoredAnswer`: fingerprint, question, column, answer, action, status, timestamp.
- Lines 34–108 `AnswerStore(out_dir)`:
  - `_load` (43–58): an unreadable or wrong-version file is **ignored with a warning** (the
    questions are asked again). When new answers are saved, it is kept as `answers.json.bak`.
  - `lookup` (63–70): returns the stored answer **only if** the fingerprint matches exactly
    **and** the stored option still exists and still means the same action.
  - `record` (72–83).
  - `save` (85–98): writes a temporary file, then renames it over the old one, so a crash never
    leaves a half-written file.

**Where.** Always in the output folder (`preflight_out/` by default), never next to your data file.

---

### `interact/resolve.py` (232 lines)

**Purpose.** `resolve()`, the function that answers a Plan's questions.

- Lines 51–58 `_load_raw`: re-reads the raw file for previews. If that fails, previews say
  values are unavailable (a justified swallowed exception).
- Lines 63–86 `_pick_channel`: a channel object; None → notebook form if in Jupyter with
  ipywidgets, otherwise terminal; a function → `CallbackChannel`; or the names `"terminal"`,
  `"notebook"`, `"review_file"`.
- Lines 89–229 `resolve(source_or_plan, mode=None, channel=None, out_dir=None, target=None,
  task=None) -> (Plan, plan_pdf_path)`:
  - Input (116–135): a Plan, a data file (it plans it first), or a **filled-in `review.json`**.
    With a review file, the source, target, task and output folder are read from the file, and
    the mode becomes ask through that file.
  - Safety checks (137–148): the output folder must not be the data file's own folder, and the
    Decision ids must be unique.
  - `_pass(subset)` (157–188): build the session, apply stored answers, then follow the mode.
    In ask mode, new answers are recorded and saved.
  - **Two passes** (220–226). Pass 1 handles reading and typing questions. If any answer changed
    an action, cleaning and engineering are **planned again** (`replan_prep`), because they
    depend on the types. Pass 2 handles the cleaning and engineering questions. Without this, you
    could be asked whether to clip a column that, after your answer, is no longer numeric.
  - Strict mode (190–218) does the same two passes using **only** stored answers, then raises if
    anything is open.
  - Finally `write_plan_artifacts` writes `plan.json` and **then** renders the Plan PDF.

---

### `interact/channels/` — the ways to answer

**`base.py` (35 lines).**
- `ReviewPending` (14–22): raised when a channel has handed out the questions but cannot wait
  (review file written; notebook form shown). You answer, then call `resolve()` again.
- `BaseChannel` (25–32): one method, `review(session) -> {question_id: option}`.

**`callback.py` (34 lines).** `CallbackChannel(handler)`: your function receives the list of
`Question`s and returns `{question_id: option label}`. The result is validated. Good for tests and
for building your own interface.

**`review_file.py` (128 lines).**
- `write_review_file` (30–64): writes `review.json` with every question, its options, the
  recommended default, the preview and `"answer": null`. The write is atomic.
- `load_review_file` (67–84): checks the format name and version.
- `ReviewFileChannel.review` (94–105): if the file does not exist → write it and raise
  `ReviewPending`. If it exists → read and validate the answers.
- `answers_from_review` (108–128): **refuses a stale file**. Its question ids and fingerprints
  must be exactly the current ones. Otherwise: "Delete it and run again."
- Note: the file contains real sample values from your data. Treat it like the data.

**`terminal.py` (185 lines).**
- `NonInteractiveError` (24): there is no terminal to ask in.
- `format_screen` (52–73) plain text, or `_print_rich` (76–109) when the optional `rich`
  package is installed: one screen listing every question with "Why", "What we found", a
  preview table and numbered options. The recommended one is marked `<- recommended`.
- `_interpret` (112–125): accepts an option number, the exact option text, or the word `default`.
- `TerminalChannel.review` (160–185): asks "Item i of n (1-k):" once per question. End of input
  (Ctrl-D) → `MissingAnswerError`.

**`notebook.py` (108 lines).** A Jupyter form (needs `ipywidgets`): radio buttons with **nothing
preselected** and a Submit button. A notebook cannot pause and wait for a click, so the first
call shows the form and raises `ReviewPending`. After you press Submit, running the cell again
collects the answers (kept in `_SUBMITTED`, keyed by the question set).

**`__init__.py` (17 lines).** Re-exports the channel classes.

**Known limits.**
- The notebook channel is tested with a fake widget library, not a real browser.
- The terminal needs a real TTY unless you pass `stdin`.

---

## 2.7 `src/preflight/report/` — words and PDFs

**What this folder does.** It turns Decisions into plain sentences, and produces the two PDFs:
- **Plan PDF** (`plan.pdf`): written **before** anything is changed. It shows what PreFlight
  *will* do and what it needs you to decide.
- **Report PDF** (`report.pdf`): written **after** a real run. It shows what was done, who
  decided each thing, and what looked unusual.

**The plain-language rule.** Words such as "imputation", "winsorize", "schema", "delimiter",
"utf-8", "outlier", "median" and "leakage" are banned from user-facing text. The full list is in
`templates.py`. The Report PDF **checks every body paragraph** against the list and fails if one
slips through. Only the "Technical details" appendix may use the technical terms. **Why:** the
person reading the PDF may not be a data scientist, and the plan must be something they can sign off.

---

### `report/__init__.py` (13 lines)
Re-exports `render_plan_pdf`, `render_decision_text`, `render_report_pdf`,
`write_report_artifacts` and `ReportGenerationError`.

---

### `report/templates.py` (840 lines) — all the sentences

- Lines 24–75 `BANNED_JARGON`: the banned word list.
- Lines 78–94 `find_jargon_violations(text)`: a whole-word, case-insensitive search. Returns the
  banned words found.
- Lines 97–102 `assert_plain_language(text, context)`: raises `ValueError` naming the word.
- Lines 106–112 `RenderedDecisionText`: `summary`, `question`, `options`,
  `recommended_default`, `explanation`.
- Lines 121–156 `_TEXT_FORMAT_FAMILIES`, `friendly_text_format`: encodings become words, e.g.
  `cp1252` → "Western European text format", `utf-8` → "standard text format". An unknown encoding
  becomes "regional text format (name)", so it is never mistaken for the standard one.
- Lines 159–173 `friendly_separator_name`: `,` → "comma", `;` → "semicolon", and so on.
- Lines 176–197 `friendly_type_name`: `int` → "whole numbers", `float` → "decimal numbers",
  `id` → "identifiers", and so on.
- Lines 200–265 `_PREP_ACTION_NAMES`, `friendly_action_name`: every action → a short label.
  For example, `clip_outliers` → "Pull extreme values back into range", `fill_median` → "Fill
  blanks with the middle value", `target_encode_cross_fit` → "Use the typical outcome per
  category". **These labels are the options a person sees.**
- Lines 283–524: one template per reading/typing kind: encoding, separator, header, type,
  missing marker, abstention, anomaly, ambiguous task.
- Lines 531–558 `_SECTION_BY_KIND`, `section_for_decision`: which Plan PDF heading a Decision goes
  under. The headings are reading, fixing, filling gaps, removing, creating.
- Lines 561–609 `_PREP_QUESTIONS`, `template_prep_step`: question titles and reasons for the
  cleaning steps that can ask (e.g. "Remove the repeated rows?" / "Many rows appear more than once...").
- Lines 616–732 `render_decision_text(decision)`: **the dispatcher**. It picks the right template
  by stage and kind.
- Lines 739–799: `_pct`, `describe_type_scores`, `template_type_question`. A type question
  explains the trap in plain words (units, ambiguous dates, leading zeros).
- Lines 802–840 `render_question_text(decision)`: question wording for questions (used by
  `interact/question.py`).

---

### `report/plan_pdf.py` (518 lines)

- Lines 44–90 `NumberedCanvas`: draws "Page X of Y" and the footer on every page. It needs a
  second pass, because Y is only known at the end.
- Lines 93–229 `_build_styles`: fonts and colours.
- Lines 232–248 `_resolution_label`: "You changed this" / "You confirmed this", from the Decision's history.
- Lines 251–518 `render_plan_pdf(plan, output_path)`. Sections in order:
  1. Title (around line 271).
  2. Summary box (281–343): data file, rows/columns, target, goal, number of review items,
     "Source File Protected".
  3. "Your answers" (345–354), only after questions were answered.
  4. **"Needs your decision"** (355–407): one item per open question, with its options and the
     recommended safe default highlighted.
  5. **"What we will do"** (409–466): five groups, Reading / Fixing / Filling gaps / Removing /
     Creating. An empty group says "Nothing to do at this stage." instead of disappearing, so you
     can see that it was checked.
  6. "What we won't touch" (468–485): the promise that the source file is never changed.
  7. Sign-off checklist (486 onward).

**Connections.** Called by `plan.write_plan_artifacts` (after `plan.json` is written) from
`dry_run`, `resolve` and `prepare`.

---

### `report/report_pdf.py` (599 lines)

- Lines 74–76: display caps. `ANOMALY_DISPLAY_CAP = 25`, `LEAKAGE_DISPLAY_CAP = 15`,
  `DECISION_LOG_DISPLAY_CAP = 60`. Longer lists end with "+N more" and are **never cut silently**.
- Lines 78–82 `_STATUS_LABELS`: automatic → "Decided automatically", confirmed → "You confirmed
  this", changed → "You changed this".
- Lines 85–92 `ReportGenerationError`.
- Lines 204–207 `_safe_para`: a paragraph that **raises if it contains banned jargon**.
- Lines 214–259: two charts drawn with reportlab only (no matplotlib): a pie of who decided what,
  and a bar chart of unusual values per column.
- Lines 266–504 `render_report_pdf(plan, apply_result, output_path, ...)`. Sections:
  1. Title (around 296).
  2. "The plan that was applied" (306–349), with the at-a-glance pie.
  3. "What was planned, by stage" (350–373).
  4. "What actually happened vs. what was planned" (374–397), including the steps you changed.
  5. "Who decided what" (398–430), with statuses and timestamps.
  6. "Unusual entries found while reading your data" (431–457), capped.
  7. "Columns that may give away the answer" (458–476), the leakage warnings.
  8. "Technical details" appendix (477 onward), the only place for raw evidence.
- Lines 511–590 `write_report_artifacts(plan, apply_result, pipeline, out_dir, renderer,
  fitted_pipeline) -> dict[str, Path]`. **Order matters:**
  1. `cleaned_data.csv`, 2. `schema.json`, 3. `decisions.json`, 4. `pipeline.joblib` (unfitted)
  and `pipeline_fitted_on_all_rows.joblib`.
  5. **Only then** render `report.pdf`. If rendering fails, `ReportGenerationError` lists every
     file already saved. **Why:** a font or layout bug must never cost you your cleaned data.

**There is no way to switch the Report PDF off.** Every real run (`prepare`) ends with one.
This was a requirement of the v2 plan.

---

### `report/report.py` (489 lines) — the v1 report (legacy)

The v1 `Report` class (lines 27–350: filter, `show`, `to_dict`, `to_dataframe`, `plot`,
`save_pdf` with matplotlib) and four plotting helpers (352–489). In v2 it is **only** used by the
v1 regression tests (`tests/regression/adapter.py`). It uses matplotlib, which is a core dependency in
`pyproject.toml`.

Phase 7c change: `from __future__ import annotations` was removed, together with the move to
Python 3.10 as the minimum version and the ruff `py310` target (stage A3 in `.review/STATE_7c.md`).

---

### `src/preflight/py.typed`

An empty marker file. It tells type checkers (mypy, pyright) that the package ships type hints.

---

## 2.8 `scripts/` — measuring tools

**What this folder is.** These are command-line tools that **measure** PreFlight. They are not
part of the library (nothing in `src/` imports them). They follow one house rule: **scripts
compute, files hold the output, and the screen shows counts only.** The results go to
`docs/` or a JSON file, so a number in a report can always be traced back to a file.

**The tuning / held-out rule.** The corpus is split in two (`corpus/labels/split.json`):
- **Tuning** files may be looked at while improving PreFlight.
- **Held-out** files are only scored once at the end, so they show how PreFlight does on files
  nobody tuned it for.

Several scripts enforce this by never loading held-out files.

| Script | Lines | What it does | Output |
|---|---|---|---|
| `acquire_corpus.py` | 677 | Phase 1: downloads the 200+ public CSVs and writes the first labels. `build_dataset_manifest` (52–351) lists the sources, `analyze_column` (387–493) proposes a label per column, `process_dataset` (500–611) and `main` (618–673). | `corpus/raw/`, `corpus/labels/` |
| `measure_v1_baseline.py` | 373 | Runs v1.0.1's profiler on the corpus to get the "before" numbers (`evaluate_dataset` 63–219, `main` 222–369). | `docs/v1_baseline_report.md` |
| `corpus_accuracy.py` | 199 | **The main measurement.** `measure_file` (65–170) reads one file through ingest, infer and `plan()` and compares it with the labels: dialect right?, each column's type right?, wrong-and-not-flagged?, evidence present?, how many questions? Helpers: `norm_enc` (44–48), `load_labels` (51–58), `load_split` (61–62). `main` (173–180) runs every file in one process. | per-file records |
| `corpus_summary.py` | 281 | Reads one corpus test run's saved records (`.pytest_cache/corpus_results/run_<time>/`) and builds the targets table. Held-out and combined columns appear only with `--final`. | targets table (markdown) |
| `type_accuracy_buckets.py` | 20 | Column-type accuracy per group, strict and "correct or flagged". | counts printed |
| `tuning_eval.py` | 56 | Phase 7c stage B: scores the **tuning** set only, in parallel, and writes the misses. | JSON |
| `error_analysis.py` | 58 | Groups the tuning-set misses by (label → detected type) and dialect part (encoding, separator, header). | markdown |
| `compare_v1_v2.py` | 191 | v1 vs v2 model scores on 10 files. Each version's pipeline is fitted **inside each CV fold**, with the same models, folds and seeds. Leftover text columns are one-hot encoded per fold for both sides alike. | `docs/review_notes/phase7c/compare_v1_v2.json` |
| `compare_table.py` | 49 | Turns that JSON into a table. **Tie rule:** the difference in means ≤ v1 spread + v2 spread. | `compare_table.md` |
| `licence_audit.py` | 93 | Reads every dependency's licence from the installed package metadata (not from memory). | `docs/licence_audit.md` |
| `exception_audit.py` | 57 | Lists every `except` in `src/` that ignores an error, and whether it carries a `# Swallowed:` comment saying why that is safe. | markdown |
| `static_triage.py` | 193 | Phase 7 grep/AST checks (TODOs, bare excepts, prints and so on). | `docs/review_notes/phase7/static_findings.md` |
| `time_plan.py` | 39 | Times `plan()` per file in a fresh process. It also records a fingerprint of the Decisions, to prove a speed change did not change the output. | JSON |
| `timeouts_solo.py` | 37 | Re-runs slow files one at a time with a hard time limit. | `docs/review_notes/phase7/timeouts_solo.md` |
| `infer_fingerprints.py` | 41 | Saves each column's typing result, to check that the 10,000-row sampling changed nothing on small columns. | JSON |

`scripts/_corpus_accuracy_raw.json` and `scripts/_metrics_summary.json` are saved outputs from
earlier measurement runs (data, not code).

**How to run the full measurement.**
```bash
pytest -m corpus -n 4 --dist loadgroup          # one record per file per suite
python scripts/corpus_summary.py --final         # targets table (held-out included)
python scripts/type_accuracy_buckets.py          # strict and correct-or-flagged
```

---

## 2.9 `tests/` — how the project proves it works

**The four folders.**

| Folder | What it checks | Runs by default? |
|---|---|---|
| `tests/unit/` | One part at a time, on small hand-made files. | yes |
| `tests/property/` | Random inputs generated by the `hypothesis` library: rules that must hold for *any* input. | yes |
| `tests/regression/` | v1.0.1's 41-test acceptance suite, run against v2 through an adapter. | yes |
| `tests/corpus/` | Every one of the 254 real files in `corpus/raw/`. Slow. | **no**: `pytest -m corpus` |

`pyproject.toml` sets `addopts = "-m 'not corpus'"`, so a plain `pytest` runs the fast suite
(Phase 7c: 346 passed, 2 skipped, 1 xfailed).

**Rule for this project:** a test assertion is never edited to make a test pass. A known gap is
marked `xfail(strict=True)` instead. The test then *must* keep failing, and if the gap is ever
fixed, the run tells you to remove the marker.

---

### `tests/conftest.py` (88 lines)
Wiring for the corpus suites. Each corpus session gets a fresh folder
`.pytest_cache/corpus_results/run_<time>/`. A test that hits its time limit is saved as a
TIMEOUT record. At the end, `scripts/corpus_summary.py` runs on the records, and a corpus-wide
threshold that fails also fails the session. It lives at the top level because the parallel
runner (pytest-xdist) only loads it from there.

### `tests/unit/` (one file per subject)

| File | Checks |
|---|---|
| `conftest.py` | Shared fixtures: a messy CSV, a plan for it, a review session, and helpers to pick options. |
| `test_ingest.py` | Encoding, separator and header detection on hand-built files: BOM, Latin-1, semicolons, tabs, no header, title rows, footers. |
| `test_ingest_edges.py` | Edge paths: bytes and open files as input, overrides, empty input, fallbacks. |
| `test_encoding_evidence.py` | An unsure encoding is **flagged with alternatives**, never silently swapped. |
| `test_infer.py` | Type traps: leading zeros, ambiguous dates, decimal commas, booleans, IDs, abstaining. |
| `test_infer_sampling.py` | Large columns are scored on a sample and verified on all rows. Small columns give **exactly** the pre-sampling output, compared with the golden file `tests/unit/data/infer_golden_pre_sampling.json`. |
| `test_units_generic.py` | Units are found without any unit name in the code (`12kg`, `3.5 mph`, `$`, `%`). |
| `test_plan.py` | `plan()` builds Decisions in registry order, with ids and fingerprints. |
| `test_dry_run.py` | `dry_run()` writes `plan.json` and the Plan PDF and changes nothing else. |
| `test_plan_apply_equivalence.py` | What the plan says = what `apply` does. |
| `test_jargon_lint.py` | Renders Plan PDFs and checks that **no banned word** appears in them. |
| `test_question.py` | Questions: 2–4 options, default among them, previews, trigger labels. |
| `test_modes.py` | ask / defaults / strict, and auto selection. |
| `test_channels.py` | The terminal, review-file, callback and notebook channels. The notebook channel uses a **fake** widget library. |
| `test_store.py` | `answers.json`: reuse on the same fingerprint, re-ask on a changed one, an unreadable file kept as `.bak`. |
| `test_nonblocking.py` | With stdin closed or an idle pipe, nothing ever waits for input (run in subprocesses). |
| `test_no_extras.py` | Works with neither `rich` nor `ipywidgets` installed. |
| `test_output_location.py` | Output goes to `./preflight_out/`, never next to the source file. A non-interactive run with open questions stops with a helpful error. |
| `test_clean.py`, `test_prep_steps.py` | Cleaning and engineering steps: the plan never changes the input, apply is correct, the impact thresholds flip defaults. |
| `test_engineer.py` | Encodings, scaling, leakage flag, opt-in extras. |
| `test_prep_integration.py` | The cleaning/engineering sections appear in the plan, and the dry run matches the real run across all stages. |
| `test_assemble.py` | The pipeline reproduces `apply`, learns only from training rows, and survives a save and reload. |
| `test_fit_hygiene.py` | Steps that learn live in the **unfitted** pipeline. Fitting inside CV folds gives no leak (a leaky toy dataset is used to show it). |
| `test_report_pdf.py` | `prepare()` writes every output, and a PDF failure still leaves the saved files (`ReportGenerationError`). |
| `test_normalize_categorical.py` | Phase 7c regression: tidying a pandas `category` column must not crash. |

### `tests/property/`

| File | Rule |
|---|---|
| `test_ingest_roundtrip.py` | Loading keeps every cell **exactly** as written. |
| `test_schema_roundtrip.py` | A random `TableSchema` → JSON → back gives the same object. |
| `test_plan_determinism_and_purity.py` | The same file gives the same Decisions twice, and `plan()` never changes the source file. |
| `test_performance_smoke.py` | A ~1-million-row CSV loads in chunks within a memory limit. |

### `tests/regression/`
- `adapter.py` (468 lines): **test-only**. v1 tests pass a DataFrame. v2 only reads files, so
  the adapter writes the DataFrame to a temporary CSV, runs v2 with `mode="defaults"`, and wraps
  the result in v1's shapes (`PrepResult`, `Report`, `ColumnProfile`). Nothing in it is
  exported from `preflight`.
- `test_mvp_acceptance.py` (640 lines, 41 tests): v1's acceptance suite. Skips and divergences are
  listed in `docs/port_regressions.md`. One test, `test_phone_number_like_column_not_coerced`,
  is a strict xfail (Phase 7c): a CSV cannot tell a 10-digit phone number from an integer.

### `tests/corpus/` (run with `pytest -m corpus -n 4 --dist loadgroup`)

| File | Role |
|---|---|
| `corpus_support.py` | Helpers: list the files, load labels, save one JSON record per file, cache ingest/infer results, run a worker in a subprocess. |
| `test_label_integrity.py` | The labels themselves: no duplicate ids, every raw file has a label and every label a raw file, each label has the required fields, column counts match the raw file, and every source has a licence in `sources.csv`. |
| `test_ingest_corpus.py` | Ingest on every file: no crash, dialect checks. |
| `test_infer_corpus.py` | Type inference on every file: evidence and Decisions present. |
| `test_plan_corpus.py` | `dry_run` on every file: the Plan PDF renders, with no jargon. |
| `test_interact_corpus.py` + `interact_worker.py` | plan → resolve(strict) and plan → apply(defaults) on every file, in a subprocess with a hard timeout. |
| `test_pipeline_corpus.py` + `prep_worker.py` | The whole pipeline, including fitting the returned pipeline. |
| `test_accuracy_corpus.py` | Calls `scripts/corpus_accuracy.py:measure_file` per file. These records feed the accuracy numbers. |
| `test_corruption_handling.py` + `corruption.py` | Deliberately damaged copies of real files (wrong encoding, mixed separators, extra title rows, sentinels, uneven rows, duplicate columns, embedded nulls...). PreFlight must flag or cope, never crash silently. |
| `test_degenerate_inputs.py` | Empty file, header only, one column, one row, binary garbage, very wide, one huge cell, a column that is entirely missing. |

---

## 2.10 `corpus/` — the test data

```
corpus/
├── raw/        254 real CSV/TSV/data files (public sources)
└── labels/     257 files:
    ├── <file_id>.json   254 labels, one per raw file
    ├── split.json       which files are tuning (153) and held-out (101)
    ├── sources.csv      where each file came from, with its licence
    └── GUIDE.md         how the labels were made
```

**A label file** (`corpus/labels/<file_id>.json`) holds `file_id`, `filename`, the true
`dialect` (`delimiter`, `encoding`, `header_row_index`) and one entry per column
(`column_name`, `column_index`, `semantic_type`, `anomalies`). Types use the 7 canonical names:
numeric, categorical, boolean, datetime, text, ID, constant.

**Name prefixes** show the source: `fl_` frictionless messy-CSV tests, `fte_` FiveThirtyEight,
`pd_` pandas fixtures, `r_` R datasets (AER, carData, COUNT, DAAG), `sb_` seaborn, `tt_`
TidyTuesday, `uci_` UCI repository.

**The split** (`split.json`): SHA-256 of the file id, mod 100. Under 60 → tuning, otherwise
held-out. It is fixed by a formula, so nobody can choose which files are "easy". The held-out
numbers in Phase 7c are labelled **"seen during review, not independent"**, because the held-out
files were examined in earlier review phases.

---

## 2.11 `docs/` — index

| File | What it is |
|---|---|
| `README.md` | Index of the docs, and which ones are current. |
| `PROJECT_GUIDE.md` | This guide. |
| `DEMO_GUIDE.md` | Step-by-step walkthrough with real output. |
| `PLAN.md` | The v2 plan: phases, goals and gates. |
| `fit_hygiene.md` | Which steps learn from data, and how to use the unfitted pipeline in CV. |
| `v2_hardening_report.md` | **Current numbers** (Phase 7b, updated in 7c): accuracy, gates, what was not verified. |
| `phase8_readiness.md` | The 10 release gates and the GO/NO-GO verdict. |
| `v2_vs_v1_scores.md` | Model scores, v1 vs v2, on 10 files. |
| `tuning_log.md` | Every Phase 7c tuning change and its measured effect. |
| `licence_audit.md` | The licence of every dependency. |
| `hardening_findings.md`, `review_phase7.md`, `review_phase7b.md`, `PHASE_7B_REPORT.md` | Review findings from Phases 7 and 7b. |
| `v1_baseline_report.md` | v1.0.1's scores on the corpus (the "before"). |
| `v2_ingest_report.md`, `v2_infer_report.md`, `v2_plan_report.md`, `v2_interact_report.md`, `v2_port_report.md`, `v2_accuracy_report.md` | Per-phase reports. The ones marked **SUPERSEDED** have old numbers. Use `v2_hardening_report.md` instead. |
| `PHASE6.md`, `port_contract.md`, `port_log.md`, `port_regressions.md`, `v2_port_pytest_output.txt` | The Phase 6 port of v1's cleaning/engineering. |
| `plan_pdf_comprehension_check.md` | A protocol for testing whether people understand the Plan PDF. |
| `review_notes/phase7/`, `phase7b/`, `phase7c/` | Working notes and raw outputs (static findings, error analysis, comparison JSON, final summaries). |

---

## 2.12 Project files at the root

### `pyproject.toml`
- `[build-system]`: hatchling. `[tool.hatch.build.targets.wheel]` packages `src/preflight`.
- `[project]`: the name on PyPI is **`pypreflight`** (you import `preflight`), version
  `2.0.0.dev0`, Python **>= 3.10** (raised from 3.9 in Phase 7c, so the declared minimum
  matches the CI test matrix 3.10/3.11/3.12; that matrix has not run yet).
- Core dependencies: scikit-learn, pandas, numpy, scipy, python-dateutil, matplotlib, typer,
  joblib, charset-normalizer, reportlab>=4. (`typer` is currently not imported by any module;
  `matplotlib` is only used by the legacy v1 `report/report.py`.)
- Extras: `interactive` = rich (prettier terminal questions), `notebook` = ipywidgets (the
  notebook form), `dev` = pytest, pypdf, hypothesis, pytest-cov, ruff, mypy, nox, pre-commit,
  pytest-xdist, pytest-timeout. Each has a licence comment.
- `[tool.ruff]`: line length 100, target py310. Rules E, F, W, I, UP, B. Ignored: E501, and
  UP007/UP045/UP035/B905 (with the reasons in comments).
- `[tool.mypy]`: Python 3.10, `strict = true` for `preflight.*`, `mypy_path = "typings"`.
- `[tool.pytest.ini_options]`: the `corpus` marker, deselected by default.

### `typings/`
Small hand-written type stubs (`.pyi`) for the parts of scikit-learn (`sklearn/base.pyi`) and
reportlab (`reportlab/pdfgen/canvas.pyi`) that PreFlight subclasses. This lets `mypy --strict`
check those classes. See `typings/README.md`.

### `.github/workflows/ci.yml`
On every push to `main`/`v2-dev`, and on every pull request, three jobs run:
- `lint`: ruff on Python 3.12;
- `type-check`: `mypy --strict src/preflight` on 3.12;
- `tests`: `pytest -q` on **3.10, 3.11 and 3.12** (the fast suite; corpus tests are excluded).

### `noxfile.py`
The same three checks as local sessions (`tests`, `lint`, `type_check`) for Python 3.10–3.12.
Only 3.12 is installed on the development machine, so 3.10 and 3.11 are verified in CI only.

### Other root files
`README.md` (user introduction), `CHANGELOG.md`, `LICENSE` (MIT), `CLAUDE.md`, `dist/`
(built packages), `.review/` (Phase 7c working state, including `STATE_7c.md`).

---

# Section 3 — Glossary

**Abstain** — the type detector refuses to guess (scores too low or too close, or a trap). The column stays text and a question is asked.
**Action** — what a Decision will do, e.g. `cast_to_int`, `drop_column`, `separate_by_;`.
**Ambiguous date** — `03/04/2020`: every date could be read day-first or month-first.
**Ambiguous task** — a target such as 1–5 ratings that could be classification or regression.
**Anomaly** — a value that failed the chosen conversion. It is logged, and the cell becomes blank.
**Answer store** — `answers.json`: stored answers keyed by fingerprint.
**Apply** — carry out the Decisions (`preflight.apply`).
**BOM (byte order mark)** — bytes at the start of a file that announce its encoding.
**Canonical type** — the 7 label types: numeric, categorical, boolean, datetime, text, ID, constant (int + float = numeric).
**Cardinality / variety** — the number of different values in a column. 20+ = high variety.
**Channel** — how questions reach you: terminal, notebook, review file or callback.
**Chaos / coherence** — charset-normalizer's mess score and its language check for a decoding.
**Classification / regression** — predicting a category / predicting a number.
**Clip (winsorize)** — pull extreme values back to a limit.
**Clone** — sklearn's `clone()`: a fresh, unfitted copy of a pipeline.
**Confidence** — 0–1, how sure a Decision is.
**Corpus** — the 254 real test files in `corpus/raw/`, with labels in `corpus/labels/`.
**Correct or flagged** — a column type that is right, or wrong but flagged for review.
**Cross-fit** — compute a row's encoding only from *other* rows (5 parts), so its own answer cannot leak in.
**CV fold** — in cross-validation, one train/test split of the rows.
**Decision** — the record of one choice: stage, kind, columns, action, evidence, confidence, default, needs_attention, plain_language, status, fingerprint, history.
**Dedupe** — remove repeated rows.
**Default (safe default)** — the action used when nobody answers. Usually the least destructive one.
**Dialect** — how a CSV is written: the separator (delimiter), quote and escape characters.
**Disguised missing value / marker / sentinel** — a non-blank value that means "missing": `NA`, `?`, `-999`.
**Dry run** — plan + questions + Plan PDF, with no data output (`preflight.dry_run`).
**Encoding (text)** — how bytes become letters: UTF-8, Latin-1, cp1252...
**Encoding (categories)** — turning categories into numbers: ordinal, one-hot, target.
**Evidence** — the numbers behind a Decision (scores, counts, examples, learned values).
**Fingerprint** — the first 16 hex characters of a SHA-256 of a Decision's stage, kind, columns, action, default and rounded evidence. The same data gives the same fingerprint.
**Fit / transform** — learn values from rows / apply learned values to rows.
**Fit hygiene** — learning steps are fitted only on training rows (inside each fold). Learn-nothing steps may run on the full table.
**Gate** — one of the 10 release checks in `docs/phase8_readiness.md`. GO = all pass.
**Header row** — the row holding the column names. Title rows are above it, footer/total rows at the end.
**Held-out set** — 101 corpus files scored only at the end. In 7c they are "seen during review, not independent".
**Impact threshold** — removing or changing more than a set share (2% rows, 25% columns, 5% values) makes the step ask, with a safe default.
**Imputation (fill)** — replacing blanks with a value (median, most common, `__missing__`).
**IQR** — the spread of the middle half of the values (Q3 − Q1). The limits are Q1 − 1.5·IQR and Q3 + 1.5·IQR.
**Jargon lint** — the check that the PDFs contain no banned technical words.
**Kind** — (1) a Decision's category, e.g. `fill_median`. (2) v1's column kind, e.g. `NUMERIC_FEATURE`, `CATEGORICAL_LOW`.
**Leading zero** — `02138`. Reading it as a number would destroy a code.
**Leakage** — information about the answer reaching the model unfairly. Either a column that copies the target, or fitted values that have seen the test rows.
**log1p** — log(1 + x). It compresses a long tail.
**Margin** — the winner's score minus the runner-up's.
**Missing indicator** — an extra `<col>_missing` True/False column.
**Mode** — auto, ask, defaults, strict: how open questions are settled.
**model_hint** — `tree` or `linear`. It changes the encoding and scaling.
**Mutual information** — how much a column tells you about the target (also catches non-linear links).
**needs_attention** — True means the Decision becomes a question.
**One-hot** — one True/False column per category.
**Ordinal encoding** — categories numbered 0, 1, 2... (sorted). An unknown category becomes −1.
**Pipeline (unfitted)** — sklearn `Pipeline(FoldCleaner, EngineerTransformer)` with nothing learned yet.
**pipeline_fitted_on_all_rows** — the same pipeline fitted on every row. For new data only, never for scoring.
**Plan** — the ordered list of Decisions plus the schema (`plan.json`).
**Plan PDF / Report PDF** — before-run plan / after-run record, both in plain language.
**Plus code** — `+44...`: a phone code, kept as an ID.
**Prepared table** — the typed table after the learn-nothing steps (`ApplyResult.prepared`).
**Preview** — 3 real values, before and after each option.
**Property test** — a test run on many random inputs (hypothesis).
**Ragged rows** — rows with a different number of cells from the header.
**Rare category** — a value seen in under 1% of rows. It is grouped as `__rare__`.
**Raw table** — every cell as text, straight from the file.
**Review file** — `review.json`: the questions written to disk, to be filled in and passed back to `resolve()`.
**ReviewPending** — "questions handed out, answer them and call again" (it never waits).
**Runner-up** — the best type from a different family than the winner.
**Schema (TableSchema)** — the file's description: encoding, dialect, header, columns and their types.
**Signals** — per-column numbers (missing rate, variety, correlation, VIF...) computed in `signals.py`.
**Skewness** — how lopsided a distribution is. Above 1 gets log1p (linear models).
**Smoothing** — pulling small categories' averages toward the overall average.
**Stage** — a pipeline step class with `plan` and `apply` (Reading, Typing, Fixing...).
**Standard scaling** — (x − mean) / std.
**Stateless / stateful** — learns nothing from data / learns values from data.
**Status** — `automatic`, `confirmed` or `changed`.
**Strict accuracy** — the column type exactly matches the label.
**StrictResolutionError** — strict mode's error listing every open question.
**Swallowed exception** — an error deliberately ignored. Each one must carry a `# Swallowed:` reason.
**Target** — the column to predict.
**Target encoding** — replacing a category with the average target for it.
**TF-IDF** — word scores: frequent in this row, rare overall.
**Trap** — a type pitfall: leading zero, plus code, decimal comma (`12,5`), unit (`12kg`), ambiguous date.
**Trigger category** — the label for *why* a question is asked (8 of them).
**Tuning set** — 153 corpus files that may be examined while improving PreFlight.
**Typed table** — values converted to real types (`ApplyResult.data`).
**Unit column** — numbers with one repeated suffix (`12 kg`). It is read as numbers, and the unit is kept in the evidence.
**VIF (overlap score)** — how much a number column is predictable from the others. Above 10 gets a note.
**Wrong-no-flag** — a wrong type that was *not* flagged: the silent failure rate.
**xfail (strict)** — a test expected to fail. If it starts passing, the run fails, so the marker gets removed.
**z-score** — (x − mean) / std. The z-score limits are ±3.
