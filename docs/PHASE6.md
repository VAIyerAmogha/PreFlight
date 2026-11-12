# PHASE 6 (M5): Port v1.0.1 cleaning, engineering and assembly into v2

Status: Subphases 6.0-6.7 all DONE. Nothing is committed. Open items for the user: 4 regression divergences in docs/port_regressions.md, and the unidentified "13 bug repros" (see section 4).
Branch: `v2-dev`. v1.0.1 backup (read-only reference): `/home/amg/Desktop/PreFlight v1.0.1`
(v1 sources in `src/preflight/{profiler,cleaner,engineer,assembler}.py`; tests in `tests/mvp_test.py`, `tests/test_full.py`, `tests/1.0.0/`.)

## 0. HANDOVER INSTRUCTIONS (read first if you are a new session)

1. Read this file top to bottom, then `docs/v2_interact_report.md`, `CLAUDE.md`, `docs/PLAN.md`.
2. Find the first subphase in section 3 whose status is not `DONE`. Work only on that one.
3. Append to the ACTION LOG (section 5) after every meaningful action (file read/written, test run, decision, stop). Format: `YYYY-MM-DD | subphase | action | result`.
4. Update the subphase status line when finishing. Never mark `DONE` without the exit check passing.
5. If a STOP condition triggers, write it in section 4 (Open questions), stop, and report to the user. Do not guess.
6. Never run `git add`, `git commit`, `git push`. Leave all changes uncommitted.

## 1. Purpose

Port v1.0.1 cleaning, engineering and assembly logic into v2 so it reads the v2 schema and Decisions instead of the v1 Profiler output. Then integrate into the plan registry / dry_run / Plan PDF, port v1 regression tests, and report real numbers.

## 2. Hard rules (verbatim intent from the user task)

- PORT, not rewrite. Copy v1 logic into the v2 layout (`clean/`, `engineer/`, `assemble.py`); change only what is needed to read schema/Decisions and emit plan Decisions. No redesign, no renamed behaviors, no "improvements".
- Every change to v1 logic is logged in `docs/port_log.md` (file, function, what changed, why).
- v1 Profiler is NOT ported; the v2 schema replaces it.
- No learned models. No CLI commands. No Report PDF. No threshold tuning on the corpus.
- Each step: `plan(context) -> list[Decision]` and `apply(context, decisions)`. `plan()` never mutates data. Decisions carry evidence, confidence, safe default, needs_attention, plain-language sentence template.
- Destructive actions (column drops, dedupe, outlier clipping) emit a `needs_attention` Decision when impact exceeds a configurable threshold (named config value, documented default). Do not tune against the corpus.
- Missing-value markers recorded in the schema (Phase 3) are applied in the cleaner (the first place they are converted).
- Typed DataFrame input comes from the Phase 4 apply (typing stage), not pandas guessing.
- VIF uses numpy (correlation-matrix inverse). No statsmodels.
- Never edit a regression assertion to force a pass. Divergences go to `docs/port_regressions.md`; the user decides each.
- If anything is ambiguous or blocked, stop and report.

## 3. Subphases

Each has: scope, deliverables, exit check. Budget is a rough share of total effort.

### 6.0 Recon and contract (~8%) — STATUS: DONE (Q1 -> audit draft; Q2 -> Option A)
Scope: read v1 profiler/cleaner/engineer; read v2 `schema.py`, `infer/`, Decision model, plan registry, Phase 5 trigger plug-in points.
Deliverables: `docs/port_contract.md` with (a) Profiler fields consumed by Cleaner/Engineer, (b) mapping table v1 type (numeric, categorical, datetime, text, boolean, constant, ID) to v2 `inferred_type`/confidence/missing_markers/anomalies/target/task type, (c) list of v1 fields with NO v2 equivalent.
Exit check: **STOP and ask the user if (c) is non-empty** (compute in port vs drop dependent logic). Do not invent substitutes. Also stop if the v1 backup does not import.

### 6.1 Cleaner port (~18%) — STATUS: DONE (unit tests for new behaviour still to be written in 6.6; `run_cleaner` shim must be rebuilt inside tests/regression/adapter.py in 6.5)
Scope: `clean/`: dedupe, column drops, normalization, missing-value fill, winsorization, rare-category handling; schema missing markers applied here.
Deliverables: one module per step with plan/apply; named config for the destructive-impact threshold with documented default; log entries in `docs/port_log.md`.
Exit check: plan purity (data hash unchanged), apply produces the v1-equivalent output on a hand-built frame.

### 6.2 Engineer port (~22%) — STATUS: DONE except trigger wiring (moved to 6.4: triggers.py rules + question option builders + templates)
Scope: `engineer/`: tree-vs-linear transform choice, cross-fit target encoding, datetime expansion, interactions, clustering features, text features, numpy VIF, leakage signals. Wire leakage into the Phase 5 "suspected leakage" and "destructive action" triggers and remove them from the "empty plug-in points" list. Fix the v1 Pandas4Warning (old `engineer.py:698` equivalent) and log it.
Exit check: same as 6.1 plus triggers fire on a perfectly leaky feature; no pandas warnings.

### 6.3 Assembler port (~10%) — STATUS: DONE (unit tests in 6.6)
Scope: `assemble.py`: two-phase assembler building the fitted sklearn Pipeline from applied Decisions. Keep fit-on-train-only and leakage safeguards exactly as v1.
Exit check: pipeline output equals apply output; serializes and reloads.

### 6.4 Plan integration (~8%) — STATUS: NOT STARTED (a first attempt was REVERTED at user request to stop after 6.3; design notes below)
Scope: register cleaner/engineer planners in the plan registry (dry_run produces fixing / filling gaps / removing / creating Decisions); add sentence templates (extend, not replace); extend banned-jargon test (imputation, winsorize, cardinality, target encoding, interaction, VIF need plain equivalents).
Exit check: Phase 4 equivalence test across ALL stages (dry_run Decisions == real-run Decisions except status/timestamps); dry-run purity (source/output hashes unchanged) still holds.

### 6.5 Regression port (~20%) — STATUS: NOT STARTED (can start in parallel with 6.4 once 6.3 is done)
Scope: port the 47 `test_mvp_acceptance` tests and the 13 bug-repro tests (identify by content) into `tests/regression/`. Add an internal, non-exported adapter: DataFrame -> CSV in tmp dir -> v2 CSV path. Investigate the 15 unexplained v1 skips: fix cause, remove with written reason, or document why they must stay skipped.
Deliverables: `docs/port_regressions.md` (test, v1 behavior, v2 behavior, assessment: v1 wrong / v2 regressed / equivalent).
Exit check: zero unexplained skips; no assertion edited. **STOP** if a dataset is unavailable offline; tell the user which to supply.

### 6.6 New tests and full run (~10%) — STATUS: NOT STARTED
Scope: `tests/unit/` per ported step (plan purity, apply correctness, threshold behavior, edge cases: all-null, constant, single-row, high-cardinality, perfectly leaky); `tests/unit/test_assemble.py`; `tests/corpus/test_pipeline_corpus.py` (dry_run then run under defaults mode on every corpus file with a usable target; record completions, unhandled exceptions, plan/run disagreements (must be 0), needs_attention median/max/files above 10).
Deliverables: raw pytest output of ALL suites including earlier phases; `docs/v2_port_report.md` with actual numbers only (regression pass/fail/skip out of 47+13, remaining unexplained skips, corpus results). No claims of improvement over v1.

### 6.7 Verify (~4%) — STATUS: DONE
Scope: `pip install -e .` works; `import preflight` works with no extras; paste extracted text of one corpus Plan PDF showing populated fixing/filling/removing/creating sections; list which of the 8 Phase 5 trigger categories are implemented (all 8 expected, else say which and why).

Results:
- `pip install -e .` succeeds (reinstalled pypreflight 2.0.0.dev0 editable; all deps already satisfied).
- `python -I -c "import preflight"` succeeds with no extras installed (`interactive`, `notebook`, `dev` are the only optional-dependency groups in `pyproject.toml`; none imported).
- Plan PDF sample: `corpus/raw/titanic.csv`, target `survived`, `dry_run(mode="defaults")`. Extracted text shows all four sections populated: Fixing (tidy text, pclass bounds, age/sibsp/parch/fare winsorize), Filling gaps (age/embarked/embark_town missing-indicator + impute), Removing (107 duplicate rows flagged needs_attention, `deck` column dropped at 77.2% missing), Creating (5 categorical->numeric encodings). Full text in ACTION LOG below.
- Trigger categories: `TriggerRegistry.implemented_categories()` returns all 8 (`ambiguous_structure_or_task`, `low_confidence_structure`, `destructive_action`, `suspected_leakage`, `lossy_coercion`, `disguised_missing`, `high_anomaly_rate`, `ambiguous_type`); `empty_plugin_categories()` is empty. Matches `interact/triggers.py` module docstring claim.
- Pandas4Warning / FutureWarning check: `tests/corpus/prep_worker.py` (used by the 6.6 corpus suite) calls `warnings.filterwarnings("ignore")` in its subprocess, so the 6.6 "790 passed" run did not actually check for pandas warnings. Ran a separate targeted check: `plan`/`dry_run`/`apply` under `warnings.simplefilter("error", FutureWarning)` and `warnings.simplefilter("error", Pandas4Warning)` (pandas 3.0.6, which already defines `Pandas4Warning`) on the 4 string-date files plus the 4 slowest/most feature-heavy corpus files (`big_epa_cars`, `rainfall`, `ipf_lifts`, `COUNT_badhealth`, covering text/datetime/cluster/interaction code paths). 0 warnings raised, 0 errors. Confirmed `select_dtypes(include="object")` (the v1 offending pattern, port_log.md 6.2) does not appear anywhere in `src/`.

### 6.4 design notes for the next session
- Registry order must be v1 execution order, with `section` only a PDF heading: fixing(markers, coerce) -> removing(dedupe, drops) -> fixing(tidy, rare) -> filling gaps -> fixing(outliers) -> removing(leakage) -> creating. A `PrepStage` base class with `kinds` + `runner` and one shared `_ensure_prep_plan(context)` (typed frame built in memory from typing Decisions via `TypingStage().apply`, then `plan_clean_full` + `plan_engineer(signals=cp.signals)`) worked on paper.
- Typed frame has Int64/Float64/boolean/category; convert to numpy dtypes before cleaning (Int64 cannot take a fractional median).
- `ApplyResult.data` must stay the typed table (test_apply_typing_execution); add `prepared`/`specs` fields instead.
- `PlanContext` needs `model_hint`, `feature_config`, `cleaner_config` (use `field(default_factory=...)`; the attempt failed on a mutable dataclass default). `Planner.plan` should set `context.decisions` after each stage.
- Triggers: implement destructive_action (clean stage, kinds remove_duplicates/drop_*/clip_outliers, needs_attention) and suspected_leakage (engineer, kind suspected_leakage); question builders for both; `section_for_decision` + templates + extended banned-jargon list; PDF sections use it.
- tests/unit/test_engineer.py and test_assemble.py (draft-era) call removed signatures and must be rewritten in 6.6.

## 4. Open questions / stops (fill in as they occur)

- **Q1 (2026-10-22, 6.0): repo already contains an earlier, partial Phase 6 attempt, committed in HEAD `0b51162` ("interaction layer implemented").** The task assumes clean/, engineer/, assemble.py, tests/regression/ do not exist yet. Found:
  - `src/preflight/clean/` (cleaner, config, dedupe, drops, impute, normalize, outliers; ~1.2k lines; already imports `Decision` and `TableSchema`, has `plan_*`/`apply_*` functions).
  - `src/preflight/engineer/` (engineer, config, datetime, encoding, features, numeric, vif; ~1.3k lines).
  - `src/preflight/cleaner.py`, `engineer.py`, `assembler.py`, `report.py` are v1-compat shims; `src/preflight/assemble.py` is a 1-line stub.
  - `plan.py`: Fixing (stage 3) and Creating (stage 6) are still stubs (lines ~639, ~675); cleaner/engineer planners are NOT registered.
  - `tests/regression/test_mvp_acceptance.py` (41 test fns, 1 explicit CLI skip) and `tests/regression/adapter.py` (733 lines; a test-only v1 API shim that re-implements the v1 profiler inside tests). Both are in git.
  - `tests/unit/test_clean.py` (7), `test_engineer.py` (7), `test_assemble.py` (3), `tests/corpus/test_pipeline_corpus.py` exist.
  - The docs they cite (`docs/port_contract.md`, `port_log.md`, `port_regressions.md`) do NOT exist. `docs/v2_interact_report.md` finding 4 says these test files import modules that do not exist (`CleanerTransformer`, `build_pipeline_two_phase`).
  - Not yet verified: whether the existing code is faithful to v1, or even imports/passes.
  - RESOLVED 2026-10-22: user chose (a) audit/complete the draft.
  - **Q2 (6.0): v1 EDA signals with no v2 equivalent**, listed in `docs/port_contract.md` section 3. Needs user choice: Option A (compute in a pure `signals.py`, recommended) or Option B (drop dependent logic). 6.1 blocked until answered.
- (old text) Decision needed from user: (a) treat existing code as a draft: audit it against v1 and fix/complete it; (b) discard it and port from scratch per the task; (c) something else. Also confirm whether the 41 vs 47 acceptance-test count and the adapter's in-test profiler are acceptable.

## 5. ACTION LOG

| Date | Subphase | Action | Result |
|------|----------|--------|--------|
| 2026-10-22 | planning | Received Phase 6 task; user supplied v1 path `/home/amg/Desktop/PreFlight v1.0.1` | v1 dir exists |
| 2026-10-22 | planning | Measured v1 source sizes (cleaner 464, engineer 732, assembler 403, profiler 480 lines; tests ~1.4k lines in mvp_test.py + test_full.py) and v2 src (~9.7k lines) | Used for effort estimate |
| 2026-10-22 | planning | Gave user estimate: ~2.5-4M tokens processed, ~15-25% of 15M budget, ~3-5h | Delivered in chat |
| 2026-10-22 | planning | Created this PHASE6.md (subphase split + handover + log) | No code touched |
| 2026-10-22 | 6.0 | Read v1 `profiler.py` (480 lines) and `types.py` in full | Profiler fields noted: SemanticType (9 values incl. NUMERIC_ID/FEATURE, CATEGORICAL_LOW/HIGH, DATETIME_NATIVE/STRING, BOOLEAN, CONSTANT, TEXT), missing_rate, outlier_rate (IQR), cardinality, rare_categories(<1%), vif_score, correlation_with_target, mutual_info_with_target, is_leakage_suspect(|r|>0.95), text_avg_length/word_count; plus class imbalance (>10:1) and mixed-type flags as ReportEntries |
| 2026-10-22 | 6.0 | Listed v2 src/tests/docs, git status, git log for clean/ engineer/ | Found pre-existing partial Phase 6 code committed in 0b51162 (see Q1) |
| 2026-10-22 | 6.0 | STOPPED before writing `docs/port_contract.md` | Contract depends on whether existing code is kept; awaiting user decision on Q1 |
| 2026-10-22 | 6.0 | User chose option (a): audit and complete existing draft | proceed |
| 2026-10-22 | 6.0 | Baseline: `pytest tests/unit --ignore test_assemble.py` = 176 passed; `test_assemble.py` and `tests/regression` fail at import (missing assemble symbols) | recorded |
| 2026-10-22 | 6.0 | Read v1 cleaner/engineer/assembler signatures + v2 clean/, engineer/, plan.py stubs, schema, infer types | feeds contract |
| 2026-10-22 | 6.0 | Wrote `docs/port_contract.md` (consumption table, mapping, no-equivalent list, preliminary audit) | STOPPED on Q2 |
| 2026-10-22 | 6.0 | User chose Option A (compute signals in port) | proceed |
| 2026-10-22 | 6.1 | Created `src/preflight/signals.py` (v1 profiler numbers + kind_from_schema) | imports OK |
| 2026-10-22 | 6.1 | Rewrote clean/{drops,normalize,impute,outliers,dedupe,cleaner}.py, added clean/{markers,_common}.py, config cardinality_threshold | `tests/unit/test_clean.py` 7 passed after 3 draft-era tests corrected (they asserted non-v1 behaviour) |
| 2026-10-22 | 6.1 | Differential test v1 run_profiler+run_cleaner vs v2 plan/apply on 31 corpus files | 28 identical, 3 differ only by intentionally unported string-date / profiler-coercion |
| 2026-10-22 | 6.1 | Wrote `docs/port_log.md` (6.1 section) | done |
| 2026-10-22 | 6.2 | Rewrote engineer/engineer.py (plan/apply via shared `_apply_one`), vif.py (leakage keep-by-default Decision + VIF note), features.py (rank_interaction_columns, seeded MI, no-overwrite), config.py (EngineerConfig); added `plan_clean_full` returning cleaned df + signals | imports OK |
| 2026-10-22 | 6.2 | Differential v1 vs v2 engineer: 68/76 identical (tree+linear), 55/76 with full FeatureConfig; differences explained in port_log.md | logged |
| 2026-10-22 | 6.3 | Wrote assemble.py (CleanerTransformer, EngineerTransformer, build_pipeline[_two_phase], transform_new_data, run_assembler) | 78/78 corpus runs: pipeline.transform(train) == apply output (target-encoded columns excluded by v1 design); pickle round trip OK |
| 2026-10-22 | 6.2/6.3 | Updated docs/port_log.md | done |

| 2026-10-22 | 6.4 | Started plan/trigger/template integration; plan.py import failed (mutable dataclass default) | USER asked to stop after 6.3 |
| 2026-10-22 | 6.4 | Reverted plan.py, report/templates.py, report/plan_pdf.py, interact/triggers.py, interact/question.py with `git checkout` (they had no earlier uncommitted changes) | repo imports again; tests/unit minus test_assemble: 174 passed, 2 failed (draft-era test_engineer tests using removed signatures) |
| 2026-10-22 | - | STOPPED after 6.3 per user | report delivered |

### 6.4 DONE (plan integration) — see docs/port_log.md "6.4"
185 unit tests pass (excluding test_assemble.py, rewritten in 6.6). Next: 6.5 regression port (adapter.py must be rebuilt; run_cleaner/run_engineer/add_features removed from src).

### 6.5 DONE (regression port): 47 collected = 41 pass, 4 fail (listed in docs/port_regressions.md for the user), 2 CLI skips. "13 bug repros" NOT identified: awaiting user. 15 v1 skips explained (one parametrized zero-feature case). Env note: running the v1 suite re-installed v1 editable into .venv; restored with `pip install -e . --no-deps`.

### 6.6 DONE: test_assemble.py rewritten, test_prep_steps.py added, corpus all-files test added (tests/corpus/prep_worker.py), full run 790 passed / 4 failed / 2 skipped (docs/v2_port_pytest_output.txt), docs/v2_port_report.md written. Contract-doc leakage error fixed. Next: 6.7 verify. Open: the 13 bug repros.

| 2026-10-23 | 6.7 | `pip install -e .` | succeeded, pypreflight 2.0.0.dev0 editable reinstalled, no errors |
| 2026-10-23 | 6.7 | `python -I -c "import preflight"` (no extras) | succeeded; confirmed `pyproject.toml` optional-dependencies are only `interactive`, `notebook`, `dev`, none required for import |
| 2026-10-23 | 6.7 | Read `interact/triggers.py` in full; ran `TriggerRegistry().implemented_categories()` / `.empty_plugin_categories()` | all 8 categories implemented, 0 empty plug-ins: `ambiguous_structure_or_task`, `low_confidence_structure`, `destructive_action`, `suspected_leakage`, `lossy_coercion`, `disguised_missing`, `high_anomaly_rate`, `ambiguous_type` |
| 2026-10-23 | 6.7 | Generated Plan PDF via `dry_run(mode="defaults")` on `corpus/raw/exercise.csv` | PDF built, but Filling gaps / Removing sections were empty ("Nothing to do at this stage") — not a representative sample |
| 2026-10-23 | 6.7 | Generated Plan PDF on `corpus/raw/titanic.csv` (target `survived`, classification), extracted text | all 4 sections (Fixing, Filling gaps, Removing, Creating) populated; full text below |
| 2026-10-23 | 6.7 | Noticed `tests/corpus/prep_worker.py:97` calls `warnings.filterwarnings("ignore")` in its subprocess | the 6.6 "790 passed" corpus run did not actually check for pandas warnings; ran a separate check |
| 2026-10-23 | 6.7 | Ran `plan`/`dry_run`/`apply` under `warnings.simplefilter("error", FutureWarning)` + `warnings.simplefilter("error", Pandas4Warning)` (pandas 3.0.6) on `exercise.csv`, `bachelorette.csv`, `majors-list.csv`, `trump-lawsuits.csv` (string-date files from 6.2) plus `big_epa_cars.csv`, `rainfall.csv`, `ipf_lifts.csv`, `COUNT_badhealth.csv` (slowest/most feature-heavy, exercising text/datetime/cluster/interaction code) | 0 warnings raised, 0 errors, all 8 files completed plan+apply cleanly |
| 2026-10-23 | 6.7 | `grep -rn 'select_dtypes(include="object")' src/` | not found, confirms port_log.md 6.2 claim that the v1 Pandas4Warning-causing pattern was removed, not patched |
| 2026-10-23 | 6.7 | Marked subphase DONE | **Phase 6 (M5) subphases 6.0–6.7 all DONE.** Open items carried forward: the 4 regression divergences (port_regressions.md) await user decision; "13 bug repros" unidentified, awaiting user confirmation that the 19 bug-labelled tests already in the 47 are what was meant. Nothing committed (per rule 6: no git add/commit/push). |

#### 6.7 Plan PDF sample (`corpus/raw/titanic.csv`, target=`survived`, task=classification, mode=defaults)

```
PreFlight Plan — Specification for Data Verification and Pipeline Operations
Data File titanic.csv | Rows/Columns 891 rows / 15 columns
Target Column survived | Objective classification
Review Needed 4 question(s) resolved | Safety Mode Source File Protected

Your answers
- Column 'survived': Read column 'survived' as plain text. (safe default applied)
- Column 'deck': Read column 'deck' as plain text. (safe default applied)
- File: Remove 107 repeated rows (12.0% of the table)? This is a large share, so please confirm. (safe default applied)
- Column 'fare': Pull 102 extreme entries in 'fare' (13.0% of entries) back to the range [-31.04, 73.2]? This is a large share, so please confirm. (safe default applied)

Needs your decision
All automated checks completed with high certainty. No manual decisions are required.

Reading
- Read the file using standard text format. Separate columns using comma separators. Use row 1 as column names.
- Read column 'survived' as plain text; 'pclass' as whole numbers; 'sex' as categories; 'age' as decimal numbers;
  'sibsp'/'parch' as whole numbers; 'fare' as decimal numbers; 'embarked'/'class'/'who' as categories;
  'adult_male' as yes/no values; 'deck' as plain text; 'embark_town' as categories; 'alive'/'alone' as yes/no values.
- Predict categories for target column 'survived'.

Fixing
- Tidy the text in column 'embarked': trim extra spaces and use lowercase (782 entries change).
- Tidy the text in column 'class': trim extra spaces and use lowercase (784 entries change).
- Tidy the text in column 'embark_town': trim extra spaces and use lowercase (782 entries change).
- Set limits [-2, 6] for 'pclass' (no current entry is outside them; they apply to new data).
- Pull 39 extreme entries in 'age' back to the range [1, 57].
- Pull 39 extreme entries in 'sibsp' back to the range [-1.5, 2.5].
- Pull 15 extreme entries in 'parch' back to the range [-1.5, 2.5].
- Pull 102 extreme entries in 'fare' (13.0% of entries) back to the range [-31.04, 73.2]? This is a large share, so please confirm.

Filling gaps
- Add a yes/no column 'age_missing' marking where 'age' was blank (13.5% blank).
- Fill blanks in 'age' with the middle value (28.25).
- Add a yes/no column 'embarked_missing' marking where 'embarked' was blank (0.3% blank).
- Fill blanks in 'embarked' with its most common value ('s').
- Add a yes/no column 'embark_town_missing' marking where 'embark_town' was blank (0.3% blank).
- Fill blanks in 'embark_town' with its most common value ('southampton').

Removing
- Remove 107 repeated rows (12.0% of the table)? This is a large share, so please confirm.
- Remove column 'deck' because 77.2% of its entries are blank.

Creating
- Turn the categories in 'sex' into whole numbers (one number per category).
- Turn the categories in 'embarked' into whole numbers (one number per category).
- Turn the categories in 'class' into whole numbers (one number per category).
- Turn the categories in 'who' into whole numbers (one number per category).
- Turn the categories in 'embark_town' into whole numbers (one number per category).

What we won't touch
Data Integrity Guarantee: The original source file will never be changed, modified, or overwritten.
Columns designated as identifiers, constant values, or preserved plain text will not have their values modified.
```
