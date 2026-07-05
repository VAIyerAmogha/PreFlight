# CLAUDE.md

## What this is
A pip-installable Python library that takes a raw pandas DataFrame and returns a cleaned DataFrame, a reusable sklearn Pipeline, and a structured Report — automatically.

## Read first
See PLAN.md for full technical spec, architecture, and implementation order.
Start every session by reading PLAN.md, then this file.

## Navigation

    src/preflight/
      __init__.py     — public API: prepare(), profile(), clean(), engineer(), compare()
      types.py        — SemanticType enum, ColumnProfile, ReportEntry dataclasses
      validation.py   — _validate_inputs(), _validate_task_target_match() API boundary helpers
      profiler.py     — semantic type inference, all EDA signal extraction
      cleaner.py      — per-column remediation strategies
      engineer.py     — encoding, scaling, datetime expansion
      assembler.py    — sklearn Pipeline construction, PrepResult assembly
      report.py       — Report object, .show(), .plot(), .to_html(), .to_dict()
      cli.py          — typer CLI, wraps prepare() for terminal use

    tests/
      test_profiler.py
      test_cleaner.py
      test_engineer.py
      test_assembler.py
      test_report.py
      test_cli.py

## How to run

    # install for development
    pip install -e ".[dev]"

    # run tests
    pytest tests/

    # run CLI
    preflight prepare train.csv --target price --task regression

## Env vars
    None required. No API keys. No external services.

## Code conventions
- Type hints on every function signature, no exceptions
- Every automated decision must emit a ReportEntry — nothing silent
- Profiler output (ColumnProfile[]) is the single source of truth for all downstream decisions
- Cleaner and Engineer never re-infer column types — they consume ColumnProfile only
- No business logic in __init__.py — it imports and exposes only
- All sklearn transformers must use set_output(transform="pandas") for column name preservation
- Rare category grouping happens before cardinality is finalized in Profiler
- VIF computation capped at top 50 numeric features by variance — log a warning if cap kicks in

## What NOT to do
- Do not re-infer SemanticType after Profiler has run — pass ColumnProfile through
- Do not apply outlier handling to columns with missingness > 30%
- Do not apply target encoding without cross-fit leakage prevention
- Do not drop columns based on mutual information scores — surface in Report only
- Do not use WidthType.PERCENTAGE anywhere in sklearn ColumnTransformer widths
- Do not train or select models — PreFlight stops before model training

## Current focus
Last updated: 2026-07-05
Active work: Phase 6 (graphic PDF report export).

Recent completions:
- v0.2.0 Phase 5: CLI exposes FeatureConfig via flags on `prepare`; feature_config stays None unless at least one feature flag is explicitly set, preserving v0.1.0 CLI output for existing users.
- v0.2.0 Phase 4: Improved Report readability. Added shared color/sizing constants to charts, grouped .show() by stage, added severity symbols and verbose toggle.
- v0.2.0 Phase 3: Implemented add_features() public API in __init__.py and engineer.py to apply FeatureConfig to an existing PrepResult post-hoc without rerunning Profiler/Cleaner.
- v0.2.0 Phase 2: Added FeatureConfig and three opt-in feature-engineering steps (interactions, datetime cyclical/deltas/reference, clustering) to the Engineer stage.
- v0.2.0 Phase 1: task/target mismatch validation added (validation.py). classification+continuous raises before any stage; regression+low-cardinality warns via ReportEntry. All four public functions now call _validate_inputs(). tests/test_validation.py added (28 tests).
- Phase 12 (packaging) is FULLY complete (hygiene, pyproject.toml metadata, and >80% test coverage).
- Added tests/edge_cases/test_degenerate_shapes.py — Phase 11 (all 3 edge-case sub-steps) is COMPLETE
- Added tests/edge_cases/test_cardinality_extremes.py — Edge case testing in progress, 1 of 3 sub-steps remaining (degenerate DataFrame shapes)
- Added tests/edge_cases/test_degenerate_columns.py — Edge case testing in progress, 2 of 3 sub-steps remaining (cardinality/ID extremes, degenerate DataFrame shapes)
- Added tests/integration/test_adult_income.py — Phase 10 integration testing on all 3 datasets is COMPLETE
- Added tests/integration/test_house_prices.py to validate regression, target-encoding, and log1p branches — Complete
- Added tests/integration/test_titanic.py as the first real-dataset integration test — Complete
- cli.py FULLY complete (Sub-step 3 of 3)
- Implemented cli.py output file writing logic (Sub-step 2 of 3) — Complete
- Implemented cli.py typer app skeleton and argument validation (Sub-step 1 of 3) — Complete
- Implemented compare() to diff PrepResults in __init__.py (Sub-step 3 of 3) — Complete
- __init__.py fully complete
- Implemented partial-stage public functions (profile, clean, engineer) in __init__.py (Sub-step 2 of 3) — Complete
- Implemented prepare() entry point with input validation in __init__.py (Sub-step 1 of 3) — Complete
- Scaffold the PreFlight-ML repository structure — Complete
- Implemented src/preflight/types.py — Complete
- Added runtime validation for ReportEntry stage and severity fields
- Added comprehensive unit tests for SemanticType, ColumnProfile, ReportEntry, and PrepResult
- Implemented SemanticType inference logic in profiler.py (Sub-step 1 of 4) — Complete
- Implemented target-independent structural signal functions in profiler.py (Sub-step 2 of 4) — Complete
- Implemented target-dependent signal functions in profiler.py (Sub-step 3 of 4) — Complete
- Implemented run_profiler orchestration in profiler.py (Sub-step 4 of 4) — Complete
- Implemented cleaner.py base imputation functions (Sub-step 1 of 4) — Complete
- Implemented cleaner.py column/row structural decisions (Sub-step 2 of 4) — Complete
- Implemented cleaner.py value-level remediation functions (Sub-step 3 of 4) — Complete
- Implemented run_cleaner orchestration in cleaner.py (Sub-step 4 of 4) — Complete
- Implemented engineer.py encoding strategies (ordinal, one-hot, cross-fit target encoding) (Sub-step 1 of 4) — Complete
- Implemented engineer.py scaling and skew transform functions (Sub-step 2 of 4) — Complete
- Implemented engineer.py datetime expansion (Sub-step 3 of 4) — Complete
- Implemented run_engineer orchestration block in engineer.py (Sub-step 4 of 4) — Complete
- Implemented Report class core in report.py (Sub-step 1 of 3) — Complete
- Implemented Report.show() terminal output in report.py (Sub-step 2 of 3) — Complete
- Implemented Report.to_dict() and Report.to_dataframe() export methods (Sub-step 3 of 3) — Complete
- Implemented CleanerTransformer wrapper in assembler.py (Sub-step 1 of 4) — Complete
- Implemented EngineerTransformer wrapper in assembler.py (Sub-step 2 of 4) — Complete
- Implemented two-phase Pipeline construction in assembler.py (Sub-step 3 of 4) — Complete
- Implemented run_assembler and transform_new_data orchestration (Sub-step 4 of 4) — Complete
- Implemented report.py standalone charting functions (Sub-step 1 of 3) — Complete
- Implemented Report constructor extension and unified .plot() method in report.py (Sub-step 2 of 3) — Complete
- Implemented Report.to_html() and Report.save_html() fully offline generators (Sub-step 3 of 3) — Complete

Open questions / blockers:
- Should ColumnProfile be frozen/immutable? (Currently it's a standard dataclass)

## Decision log addendum
17. task/target mismatch validated at the API boundary before any stage runs; classification+continuous raises, regression+low-cardinality warns (2026-07-04)
18. FeatureConfig defaults to all-off; v0.1.0 behavior is the guaranteed fallback (2026-07-04)
19. add_features() lets FeatureConfig be applied post-hoc to an existing PrepResult without rerunning Profiler/Cleaner; returns a new PrepResult, never mutates the input; requires a full prepare() result with profiles/target available (2026-07-04)
20. Report chart functions share one color palette/sizing logic and return Figure objects; .show() groups by stage, uses severity symbols, and truncates info-level entries by default with a verbose=True override (2026-07-04)
21. CLI exposes FeatureConfig via flags on `prepare`; feature_config stays None unless at least one feature flag is explicitly set, preserving v0.1.0 CLI output for existing users (2026-07-04)
22. TEXT SemanticType detection + stats: Added TEXT to SemanticType enum and basic stats to ColumnProfile. TEXT columns currently pass through Cleaner and Engineer untouched; feature generation is deferred to Phase 2 (2026-07-05)
23. FeatureConfig.text_features / text_tfidf — opt-in text feature generation: generates basic stats (char_length, word_count, has_text) and optionally capped TF-IDF features for TEXT columns (2026-07-05)
24. column_types manual SemanticType override, applied post-inference, validated at API boundary (2026-07-05)
25. PRESETS dict + preset param on prepare(), explicit kwargs always override preset values (2026-07-05)
26. dry_run param on prepare() — full decision logic runs, no transform/no pipeline fit, returns original df untouched (2026-07-05)