# PLAN.md

## Status & Current Focus
- **v1.0.0 Stress-Test / Fix / Verification Cycle**: 100% Complete. All blocker bugs and silent data degradations are resolved, verified on original datasets, and proven generalized on a fresh HR Employee Attrition dataset. All test suites pass.

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
    7.  engineer.py      — all transforms, model_hint branching, cross-fit target encoding (Complete)
    8.  test_engineer    — both model_hint modes, datetime expansion, skew handling (Complete)
    9.  report.py        — ReportEntry log, .show(), .to_dict(), .to_dataframe() (Complete)
    10. assembler.py     — Pipeline construction, column name preservation, PrepResult assembly (Complete)
    11. test_assembler   — full prepare() round-trip, pipeline.transform() on held-out data (Complete)
    12. report visuals   — .plot() four chart types, .to_html() embedded export (Complete)
    13. __init__.py      — wire prepare(), profile(), clean(), engineer(), compare() (Complete)
    14. cli.py           — typer CLI, file I/O, output naming (Complete)
    15. test_cli         — CLI integration tests (Complete)
    16. [x] integration      — real datasets: Titanic, House Prices, Adult Income
    17. [x] edge cases       — degenerate cols (Complete), cardinality/ID extremes (Complete), degenerate DataFrame shapes (Complete)
    18. [x] packaging        — [x] hygiene pass, [x] pyproject.toml metadata, [x] test coverage to 80%
    19. testpypi         — build + upload to TestPyPI, verify install on clean venv
    20. publish          — twine upload to PyPI, tag v0.1.0

## v0.2.0 roadmap
    Phase 1  [x] task/target mismatch validation — validation.py, _validate_inputs(), _validate_task_target_match(); classification+continuous raises at API boundary; regression+low-cardinality warns via ReportEntry (COMPLETE)
    Phase 2  [x] opt-in feature engineering — types.py FeatureConfig, engineer.py generate_interaction_features/datetime_cyclical/cluster_features, run_engineer() modified but defaults to all-off; v0.1.0 behavior is the guaranteed fallback (COMPLETE)
    Phase 3  [x] add_features post-hoc engineering — engineer.py add_features() and FeatureAugmenterTransformer, __init__.py export (COMPLETE)
    Phase 4  [x] Report readability improvements — report.py charts share color/sizing constants, .show() groups by stage with severity symbols and truncates info-level by default (COMPLETE)
    Phase 5  [x] CLI exposes FeatureConfig — cli.py added flags to `prepare`; feature_config stays None unless explicitly set, preserving v0.1.0 output (COMPLETE)

## v1.0.0 roadmap
    Phase 1  [x] TEXT SemanticType detection + stats (COMPLETE)
    Phase 2  [x] Text feature generation (COMPLETE)
    Phase 3  [x] Manual SemanticType overrides (COMPLETE)
    Phase 4  [x] Config presets (COMPLETE)
    Phase 5  [x] dry_run / preview mode (COMPLETE)
    Phase 6  [x] graphic PDF report export (COMPLETE)
    Phase 7  [x] PDF comparison report for compare() (COMPLETE)
    Phase 8  [x] CLI updates for all new v1.0.0 flags (COMPLETE)
    Phase 8.1 [x] Library-wide error message simplification (COMPLETE)

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
2026-07-03: Implemented encoding strategies in `engineer.py` (Sub-step 1 of 4). Included `ordinal_encode`, `one_hot_encode`, and `target_encode_cross_fit`. Explicitly used 5-fold cross-fit for target encoding to prevent target leakage into features during training, as naive target encoding introduces severe bias.
2026-07-03: Implemented scaling and skew transform functions in `engineer.py` (Sub-step 2 of 4). Added explicit checks to prevent log1p transformation on columns with values <= -1 and zero-variance standard scaling protection, preserving mathematical safety without polluting business logic in transformation primitives.
2026-07-03: Implemented datetime expansion in `engineer.py` (Sub-step 3 of 4). Leveraged pandas `.dt` accessors for native NaT propagation, ensuring rows with missing dates maintain their integrity rather than causing downstream pipeline crashes.
2026-07-03: Finalized engineer orchestration block (`run_engineer`). Explicitly enforces the `model_hint` lock on transformation branching (tree vs linear) and unconditionally applies 5-fold cross-fit target encoding for high-cardinality categorical data, fully satisfying the leakage-prevention architecture constraints.
2026-07-03: Implemented `Report` core class in `report.py` (Sub-step 1 of 3). Enforced immutability defensively by copying entry lists, and built filter queries without utilizing Pandas DataFrames to uphold the boundary rule that Report logic performs no data processing.
2026-07-03: Implemented `Report.show()` output routine in `report.py` (Sub-step 2 of 3). Enforced ordered layout grouped strictly by processing stage (`profiler`, `cleaner`, `engineer`) with inner worst-first severity sorting, guaranteeing that critical data issues are immediately visible at the bottom of the user's terminal.
2026-07-03: Finalized Phase 5 report serialization methods `to_dict` and `to_dataframe`. Engineered a custom recursive type caster within `to_dict` specifically to sanitize nested numpy scalar types emitted from sklearn statistics down to pure native Python structs, thereby pre-empting standard library `json.dumps()` failures in the Phase 8 CLI release.
2026-07-03: Implemented `CleanerTransformer` in `assembler.py` (Sub-step 1 of 4). Designed as a stateless wrapper over `cleaner.py` that computes all robust data statistics exclusively inside `fit()`. Explicitly decoupled bounds-computation from `transform()` and skipped duplicate row removal during transform, satisfying the architecture requirement that production pipelines never silently drop user-provided streaming inference records.
2026-07-03: Implemented `EngineerTransformer` in `assembler.py` (Sub-step 2 of 4). Ensures strict output column matching regardless of training-vs-test category drift by statically recording and automatically padding missing dummy variables with `False` values at inference. Guarantees safety during deployment against novel or missing target encoding classes by falling back onto frozen global means.
2026-07-03: Implemented two-phase Pipeline construction logic (`build_pipeline` and `build_pipeline_two_phase`) in `assembler.py` (Sub-step 3 of 4). Solved the severe architectural coupling problem (where `EngineerTransformer` absolutely requires `CleanerTransformer`'s dynamic column-drop outputs to initialize) without breaking core Scikit-learn Pipeline mechanics. Implemented a strict two-phase fit routine that manually steps through the cleaner, extracts the post-cleaning profiles, constructs the engineer stage dynamically, and then packages them together into a pre-fitted Pipeline strictly enforcing pandas DataFrame output formatting.
2026-07-03: Finalized Phase 6 orchestration `run_assembler` and `transform_new_data` ensuring that target columns remain rigorously decoupled from all preprocessing transformations, preserving raw labels for supervised model training and downstream alignment.
2026-07-03: Designed Phase 7 `report.py` visual generators as completely standalone pure functions that each explicitly instantiate and close over their own independent `Figure` and `Axes` objects. This was a deliberate architectural choice to prevent catastrophic state bleed between consecutive Matplotlib charts when invoked inside interactive global contexts (like Jupyter notebooks).
2026-07-03: Extended the Report constructor to optionally accept df, profiles, and target. This maintains the Phase 5 boundary where core report features only read ReportEntry[], while providing the necessary state to the visual layer (.plot()) without silently breaking constructor contracts.
2026-07-04: Implemented prepare() in __init__.py as a thin wrapper that strictly performs input validation before delegating to the assembler, keeping business logic strictly out of the __init__.py namespace.
2026-07-04: Implemented profile(), clean(), and engineer() exploratory functions in __init__.py using the pure stateless orchestration functions rather than sklearn Transformers. This adheres to the API requirement that intermediate functions do not emit fitted Pipelines and purely facilitate interactive data inspection.
2026-07-04: Implemented initial CLI skeleton in cli.py using Typer. Early validation errors (file not found, invalid CSV, ValueError from PreFlight) are caught and handled with clean stderr echos and Exit(1) instead of raw stack traces, enforcing good CLI hygiene.
2026-07-04: Implemented write_outputs() in cli.py handling conditional joblib and json serialization. Opted to fail fast on OSError during file writing and map those to CLI error echoes to ensure users are immediately alerted to permission or disk issues.
2026-07-04: Hardened cli.py with strict CLI-layer defensiveness (e.g. range-checking for thresholds, intercepting empty/null payloads, parsing exceptions) to provide a polished terminal UX without polluting core library primitives.
2026-07-04: Added tests/integration/test_titanic.py for the full prepare() round-trip on a real-world dataset (Titanic). Fixed CleanerTransformer get_feature_names_out to properly emit generated indicator column names, ensuring pandas set_output behaves deterministically.
2026-07-04: Added tests/integration/test_house_prices.py to validate the pipeline on regression tasks with high cardinality categoricals and skewed features. Rigorously tested cross-fit target encoding, log1p transformation gates, and verified that target columns remain rigorously untouched.
2026-07-04: Added tests/integration/test_adult_income.py as the final dataset integration test, verifying class imbalance detection, consistent encoding on medium-cardinality boundaries, and explicitly asserting the absence of `.predict()` methods to enforce architectural scope.
    2026-07-04: Added tests/edge_cases/test_degenerate_columns.py to validate the pipeline gracefully handles degenerate columns (all-null, single-category, constant numeric) without crashing and properly flags or drops them according to Profiler and Cleaner rules.
    2026-07-04: Added tests/edge_cases/test_cardinality_extremes.py to validate the pipeline gracefully handles cardinality and variance extremes, confirming that high-cardinality strings are properly smoothed by cross-fit target encoding and near-zero variance numerics are safely scaled.
    2026-07-04: Added tests/edge_cases/test_degenerate_shapes.py to validate the pipeline robustly fast-fails on single-column, empty, and duplicate-column DataFrames, and handles single-row DataFrames gracefully without raw exceptions.
    2026-07-04: Performed repository hygiene pass, removing manual test artifacts and explicitly updating .gitignore to exclude them. Added pytest testpaths and markers to pyproject.toml to ensure clean test runs.
    17. 2026-07-04: task/target mismatch validated at the API boundary before any stage runs; classification+continuous raises, regression+low-cardinality warns. Extracted all repeated validation blocks from prepare/profile/clean/engineer into _validate_inputs() in validation.py. _validate_task_target_match() uses a dual-threshold heuristic (>20 unique values AND >5% of row count) to avoid false positives on small integer-class datasets.
    18. 2026-07-04: FeatureConfig defaults to all-off; v0.1.0 behavior is the guaranteed fallback. Opt-in feature engineering (interactions, datetime, clustering) is safely scoped to run_engineer without modifying the default pipeline logic.
    19. 2026-07-04: add_features() lets FeatureConfig be applied post-hoc to an existing PrepResult without rerunning Profiler/Cleaner; returns a new PrepResult, never mutates the input; requires a full prepare() result with profiles/target available.
    20. 2026-07-04: Report chart functions share one color palette/sizing logic and return Figure objects; .show() groups by stage, uses severity symbols, and truncates info-level entries by default with a verbose=True override.
    21. 2026-07-04: CLI exposes FeatureConfig via flags on `prepare`; feature_config stays None unless at least one feature flag is explicitly set, preserving v0.1.0 CLI output for existing users.
    22. 2026-07-05: TEXT SemanticType detection + stats: Added TEXT to SemanticType enum and basic stats to ColumnProfile. TEXT columns currently pass through Cleaner and Engineer untouched; feature generation is deferred to Phase 2.
    23. 2026-07-05: FeatureConfig.text_features / text_tfidf — opt-in text feature generation: added generator for character length, word count, has_text boolean, and bounded TF-IDF term vectors. Collisions gracefully skipped.
    24. 2026-07-05: column_types manual SemanticType override, applied post-inference, validated at API boundary.
    25. 2026-07-05: PRESETS dict + preset param on prepare(), explicit kwargs always override preset values.
    26. 2026-07-05: dry_run param on prepare() — full decision logic runs, no transform/no pipeline fit, returns original df untouched.
    27. 2026-07-05: Report.save_pdf() — graphic-first PDF export reusing existing chart functions and severity palette, appendix table last.
    28. 2026-07-05: save_compare_pdf() — visual before/after PDF diff report built on compare(), reuses Phase 6 chart/PDF infrastructure.
    29. 2026-07-05: CLI consolidation pass — all v1.0.0 flags verified complete and consistent with the Python API.
    30. 2026-07-05: Library-wide error message simplification pass — plain language, no internal jargon, no raw tracebacks reach the user.
    31. 2026-07-05: Internal Target Label Encoding — Coerces string-labeled classification targets to numeric integers (based on lexicographically sorted order) internally within target_encode_cross_fit and run_engineer/fit to support target encoding of high-cardinality features, while retaining original string labels in all user-facing reports and returned dataframes.
    32. 2026-07-05: Outlier Winsorization Collapse Protection — Prevented outlier winsorization from collapsing low-variance or zero-inflated columns by checking for zero IQR or standard deviation before applying bounds clipping. Added explicit, non-silent `ReportEntry` logs indicating when winsorization is skipped to protect column variance.
    33. 2026-07-05: Numeric Coercion of String Columns — Integrated a robust numeric string-coercion mechanism in both `run_profiler` and `run_cleaner`/`CleanerTransformer`. It automatically parses string columns containing whitespace and common unit suffixes (like CC, kmpl, bhp, Nm) into float columns if at least 95% of non-missing values are parseable.
    34. 2026-07-05: Robust Subprocess Testing and Environment Validation — Resolved shell path dependency issues by using `sys.executable` to run pytest in subprocesses and dynamically checking local virtualenv binary folders for `preflight`. Prevented infinite recursion in `test_coverage_gate.py` by explicitly ignoring itself in recursive pytest calls.
    35. 2026-07-05: Completed the formal v1.0.0 verification and generalization pass on a fresh HR Employee Attrition dataset, ensuring all blockers and silent data issues are resolved.