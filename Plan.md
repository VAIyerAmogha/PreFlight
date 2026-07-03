# PLAN.md

## What we are building
PreFlight-ML is a pip-installable Python library for ML engineers and Kaggle-style researchers. It accepts a raw pandas DataFrame and a target column name, and returns a cleaned and feature-engineered DataFrame, a reusable serializable scikit-learn Pipeline, and a structured Report that logs every automated decision with its rationale. The goal is to eliminate mechanical data preparation work without creating a black box — every transform is explainable and every output is reproducible on unseen data.

## Stack
- Language: Python 3.9+
- Framework: scikit-learn (Pipeline, ColumnTransformer, transformers)
- Data layer: pandas + numpy
- Statistics: scipy (skewness, IQR, VIF)
- Date parsing: python-dateutil
- Visualization: matplotlib
- CLI: typer
- Packaging: pyproject.toml + build + twine
- Test runner: pytest
- Serialization: joblib

## Architecture

    Profiler     — semantic type inference and EDA signal extraction only. Owns ColumnProfile[]. No data mutation.
    Cleaner      — consumes ColumnProfile[], applies per-column remediation. Returns cleaned DataFrame + transformer specs.
    Engineer     — consumes ColumnProfile[] and cleaned DataFrame, applies encoding/scaling/feature creation. Returns transform specs.
    Assembler    — consumes all transformer specs, builds sklearn Pipeline, assembles PrepResult and Report.
    Report       — owns all display and export logic. No data processing. Reads ReportEntry[] only.
    CLI          — thin wrapper around prepare(). No business logic. Handles file I/O only.

## Component responsibilities

    types.py        — SemanticType enum, ColumnProfile dataclass, ReportEntry dataclass, PrepResult dataclass
    profiler.py     — infers SemanticType for every column, extracts missingness rate, outlier prevalence, correlation, mutual information, VIF, class imbalance, rare categories, leakage flags
    cleaner.py      — median/mode/constant imputation, missing indicators, column drops, outlier winsorization, duplicate removal, ID column drop, category normalization, rare grouping, dtype coercion
    engineer.py     — ordinal encoding, one-hot encoding, target encoding (cross-fit), StandardScaler, log1p transform, datetime expansion, model_hint branching
    assembler.py    — constructs sklearn ColumnTransformer and Pipeline, calls fit_transform on training data, assembles PrepResult
    report.py       — ReportEntry log, .show() terminal output, .plot() matplotlib charts, .to_html() embedded HTML, .to_dict(), .to_dataframe()
    __init__.py     — exposes prepare(), profile(), clean(), engineer(), compare() as public API
    cli.py          — typer app, preflight prepare command, writes .csv / .joblib / .json outputs

## Data models

    SemanticType    — enum: NUMERIC_FEATURE, NUMERIC_ID, CATEGORICAL_LOW, CATEGORICAL_HIGH,
                      DATETIME_NATIVE, DATETIME_STRING, BOOLEAN, CONSTANT

    ColumnProfile   — name, semantic_type, missing_rate, outlier_rate, cardinality,
                      rare_categories[], vif_score, correlation_with_target,
                      mutual_info_with_target, is_leakage_suspect, dtype

    ReportEntry     — stage (profiler|cleaner|engineer), column, action, rationale,
                      severity (info|warning|critical), before_stats{}, after_stats{}

    PrepResult      — df (DataFrame), pipeline (sklearn Pipeline), report (Report)

## API surface

    pf.prepare(df, target, task, model_hint, drop_threshold, outlier_method, cardinality_threshold)
                            — full pipeline, returns PrepResult

    pf.profile(df, target)  — Profiler only, returns PrepResult with df=original, pipeline=None
    pf.clean(df, target)    — Profiler + Cleaner, returns PrepResult with pipeline=None
    pf.engineer(df, target) — Profiler + Cleaner + Engineer, returns PrepResult with pipeline=None
    pf.compare(a, b)        — diffs two PrepResults, prints decision and feature shape differences

    CLI: preflight prepare <file> --target --task --model-hint --drop-threshold
         outputs: <name>_prepared.csv, <name>_pipeline.joblib, <name>_report.json

## Implementation order

    1.  Scaffold         — src/preflight/ layout, pyproject.toml, tests/, .gitignore, README stub (Complete)
    2.  types.py         — SemanticType, ColumnProfile, ReportEntry, PrepResult dataclasses (Complete)
    3.  profiler.py      — semantic type inference for all 8 types, EDA signal extraction (Complete)
    4.  test_profiler    — unit tests across numeric, categorical, datetime, ID, constant columns (Complete)
    5.  cleaner.py       — all remediation strategies consuming ColumnProfile (Complete)
    6.  test_cleaner     — unit tests per strategy, rare grouping cascade, VIF cap behavior (Complete)
    7.  engineer.py      — all transforms, model_hint branching, cross-fit target encoding
    8.  test_engineer    — both model_hint modes, datetime expansion, skew handling
    9.  report.py        — ReportEntry log, .show(), .to_dict(), .to_dataframe()
    10. assembler.py     — Pipeline construction, column name preservation, PrepResult assembly
    11. test_assembler   — full prepare() round-trip, pipeline.transform() on held-out data
    12. report visuals   — .plot() four chart types, .to_html() embedded export
    13. __init__.py      — wire prepare(), profile(), clean(), engineer(), compare()
    14. cli.py           — typer CLI, file I/O, output naming
    15. test_cli         — CLI integration tests
    16. integration      — real datasets: Titanic, House Prices, Adult Income
    17. edge cases       — all-null columns, single-category, 100% cardinality, zero-variance
    18. packaging        — pyproject.toml metadata, classifiers, README.md, test coverage to 80%
    19. testpypi         — build + upload to TestPyPI, verify install on clean venv
    20. publish          — twine upload to PyPI, tag v0.1.0

## Environment variables
    None. PreFlight-ML requires no environment variables, API keys, or external services.

## Architecture decisions log
    2026-07-02: Scaffolded PreFlight-ML repository structure exactly as defined in CLAUDE.md's "Navigation" section and PLAN.md's "Package structure" section.
    2025-07-02: Single entry point prepare() — reduces API surface, easier to document and test
    2025-07-02: model_hint="tree"|"linear" — encodes the single most impactful structural decision about downstream model family without requiring full model specification
    2025-07-02: Report is a first-class output, not optional — auditability is the core differentiator; a preprocessing library without explainability cannot be trusted
    2025-07-02: Rare grouping runs before cardinality is finalized — grouping can push CATEGORICAL_HIGH to CATEGORICAL_LOW which changes encoding strategy
    2025-07-02: VIF capped at top 50 features by variance — O(n²) cost on wide datasets is prohibitive; cap with logged warning is the right tradeoff
    2025-07-02: Target encoding uses 5-fold cross-fit — naive target encoding leaks target into features during training; cross-fit prevents this
    2025-07-02: set_output(transform="pandas") on all transformers — column name preservation is non-negotiable for Report readability and user trust; requires sklearn >= 1.2
    2025-07-02: No automated feature selection — MI scores surfaced in Report only; dropping features silently is too destructive and requires domain judgment
    2025-07-02: Clean from scratch, nothing from old web app — the FastAPI + Next.js architecture shares no useful primitives with a pip library
    2026-07-03: Implemented SemanticType enum with eight canonical semantic categories to establish a fixed contract between Profiler, Cleaner, and Engineer.
2026-07-03: Implemented ColumnProfile, ReportEntry, and PrepResult as dataclasses to provide strongly typed, lightweight data transfer objects across pipeline stages.
2026-07-03: Added runtime validation for ReportEntry.stage and ReportEntry.severity via __post_init__ since Literal annotations are not enforced at runtime by Python dataclasses.
2026-07-03: Separated semantic type inference and mixed-type detection in Profiler to ensure inference is robust and mixed-type signals are extracted independently.
2026-07-03: Implemented target-independent structural signal functions (missing_rate, outlier_rate via IQR, cardinality, rare_categories) as pure functions without remediation logic, strictly adhering to the architectural boundary between Profiler and Cleaner.
2026-07-03: Implemented target-dependent signal functions (correlation, mutual info, leakage detection, class imbalance, VIF), handling categorical encoding internally for MI computation without mutating global pipeline state, and capping VIF at 50 to prevent O(n²) scaling issues.
2026-07-03: Finalized Profiler orchestration block. `run_profiler` now accurately coordinates type inference, internal metric computation, applies global VIF correctly back to mapped columns, emits non-silent `ReportEntry` warning/critical signals for detected dataset issues (e.g., high missingness, leakage suspects), and strictly respects the one-time inference rule for `SemanticType`.
2026-07-03: Implemented base imputation functions in `cleaner.py` (Sub-step 1 of 4). These functions do not contain any threshold logic (e.g., low-missingness), but merely perform unconditional mechanical data remediation, deferring orchestration logic to a higher layer.
2026-07-03: Implemented column- and row-level structural decision functions in `cleaner.py` (Sub-step 2 of 4). Included `drop_high_missingness_columns`, `drop_numeric_id_columns`, and `remove_duplicate_rows` as independently callable, immutable functions.
2026-07-03: Implemented value-level remediation functions in `cleaner.py` (Sub-step 3 of 4). These functions (`winsorize_outliers`, `normalize_category_values`, `coerce_string_dates_to_datetime`, `group_rare_categories`) execute robust, stateless transformations while delegating conditional checks (e.g., missingness thresholds for winsorization) to the orchestration layer.
2026-07-03: Finalized cleaner orchestration block (`run_cleaner`). Safely aggregates transformation functions dynamically driven by `SemanticType` and exact state logic, explicitly outputting exhaustive transformer specifications (`specs`) and transparent `ReportEntry` arrays while ensuring the DataFrame remains functionally immutable in memory.