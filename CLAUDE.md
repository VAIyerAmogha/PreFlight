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
Last updated: 2026-07-03
Active work: engineer.py Implementation (Phase 4)

Recent completions:
- Scaffold the PreFlight-ML repository structure — Complete
- Implement src/preflight/types.py — Complete
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

Open questions / blockers:
- Should ColumnProfile be frozen/immutable? (Currently it's a standard dataclass)