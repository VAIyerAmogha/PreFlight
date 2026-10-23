# Phase 6 Port Log

Every change to v1.0.1 logic. Format: file, function, what changed, why.
Differential check (6.1): v1 `run_profiler`+`run_cleaner` vs v2 `plan_clean`+`apply_clean` (no schema) on 31 random corpus files: 28 identical frames; the 3 differences (`taxis.csv`, `raw-responses.csv`, `pulitzer-circulation-data.csv`) are the two deliberate removals below (string date parsing, profiler-side numeric-string coercion).

## 6.1 Cleaner

| File | Function | Change | Why |
|---|---|---|---|
| `signals.py` (new) | all | v1 profiler functions `compute_missing_rate`, `compute_outlier_rate`, `compute_cardinality`, `identify_rare_categories`, `compute_class_imbalance_ratio`, `infer_semantic_type` (as `v1_infer_kind`, fallback only) copied unchanged; correlation, MI, VIF, leakage flag reused from the existing `engineer/vif.py`, `engineer/features.py` | Option A (contract section 3). Profiler class itself not ported |
| `signals.py` | `kind_from_schema` | NEW. Kind comes from v2 `inferred_type`: ID+numeric dtype -> NUMERIC_ID, ID+non-numeric -> CATEGORICAL_HIGH, boolean -> BOOLEAN (v1: any 2-unique column), text -> TEXT only if v1 length rule holds else categorical | contract item 11; v2 inference replaces v1 ID/boolean rules |
| `clean/drops.py` | `plan_drops` | ID rule now `kind == NUMERIC_ID` (draft used name-part match without cardinality test); constant = kind CONSTANT; missing/ID/constant precedence as v1 | restore v1 order and rules |
| `clean/drops.py` | `drop_*_columns` | Helpers take column lists instead of v1 `ColumnProfile` list | Profiler not ported |
| `clean/*` | destructive thresholds | `needs_attention` when impact `>` threshold (draft used `>=`); when needs_attention the safe default is the keep action (`keep_column`, `keep_rows`, `keep_outliers`), otherwise default equals the action | task: "exceeds a threshold"; safe default |
| `clean/config.py` | `CleanerConfig` | added `cardinality_threshold` (v1 20). Existing thresholds `dedupe_impact_threshold` 0.02, `column_drop_impact_threshold` 0.05, `outlier_impact_threshold` 0.05 are NEW (no v1 equivalent), not tuned | task requirement |
| `clean/markers.py` (new) | `plan_markers`/`apply_markers` | Schema missing markers converted to blanks as the first cleaning step | task: markers applied in cleaner; v1 had a fixed list inside `coerce_numeric_string_column` only |
| `clean/cleaner.py` | `plan_clean`/`apply_clean` | Steps simulated in order on a private copy so bounds/medians are computed after prior steps (as v1's sequential run). Apply order: markers, coerce, dedupe, drops, normalize, impute, outliers (draft ran outliers before impute) | faithful to v1 order; plan == apply |
| `clean/cleaner.py` | `plan_coerce` | v1 cleaner's numeric-string coercion kept, only for columns the schema types int/float/numeric but still hold text | v1 Cleaner branch NUMERIC_FEATURE |
| `clean/cleaner.py` | (removed) DATETIME_STRING coercion | NOT ported: typing stage (Phase 4) converts dates | task: typed input from Phase 4 apply |
| `clean/cleaner.py` | (removed) profiler-side `coerce_numeric_string_column` pass | NOT ported for columns with no schema entry | Profiler not ported |
| `clean/cleaner.py` | `run_cleaner` | removed from `src/`; v1-style call lives in the test adapter | shim does not belong in the package |
| `clean/impute.py` | `plan_impute` | v1 per-kind rules restored: numeric: flag + median if rate < 0.3 (flag only otherwise); low-variety categorical: flag + mode if < 0.3 else constant; high-variety categorical: constant, NO flag; boolean/text/datetime untouched. Draft flagged every column and used a different categorical split. Fill values stored exact (draft rounded to 4 decimals, and stringified modes) | v1 fidelity |
| `clean/impute.py` | `apply_impute` | no longer applies schema markers (moved to `markers.py`), `schema` param removed | separation |
| `clean/normalize.py` | `plan_normalization` | applies to CATEGORICAL_LOW/HIGH kinds; rare values come from signals computed BEFORE tidying (v1 behavior: rare list from profiler on raw values, applied after lowercasing); full rare list stored (draft truncated to 10 and recomputed after tidying); Decision only when tidying changes something (v1 always ran it, same result) | v1 fidelity |
| `clean/normalize.py` | `normalize_category_values` | also accepts pandas Categorical dtype | typed input is `category` after Phase 4 |
| `clean/outliers.py` | `plan_outliers` | NUMERIC_FEATURE only; gate uses the original missing rate (as v1); bounds computed after gap filling (as v1); Decision emitted even with zero outliers because bounds also apply to new data (v1 stored `outlier_method` in specs whenever winsorize ran); bounds exact (draft rounded) | v1 fidelity |
| `clean/outliers.py` | `apply_outliers` | only nullable-integer dtypes are widened to Float64; numpy ints use pandas clip as in v1 | v1 fidelity |
| `clean/outliers.py` | `winsorize_outliers` | error message wording only | plain language |

## 6.2 Engineer

Differential check (v1 `run_profiler`+`run_cleaner`+`run_engineer` vs v2 `plan_clean_full`+`plan_engineer`+`apply_engineer`, no schema, tree and linear, 38 corpus files): 68 of 76 identical; all 8 differences are the unported string-date branch (`exercise`, `bachelorette`, `majors-list`, `trump-lawsuits`). With the full FeatureConfig (interactions, dates, clusters, text): 55 of 76 identical; remaining differences are (a) the same string-date cases, (b) interaction column choice on files where v1's mutual-information ranking is non-deterministic (v1 `mutual_info_*` called without a seed; v2 seeds it), (c) one file (`COUNT_badhealth`) where cluster distances differ on 12% of rows although features, k and cleaned input match; not explained (KMeans ties on integer data suspected), see port_regressions.md if it recurs.

| File | Function | Change | Why |
|---|---|---|---|
| `engineer/engineer.py` | `plan_engineer`/`apply_engineer` | Rewritten around one shared `_apply_one`, kinds from column signals (not Profiler). Steps simulated on a private copy so scale parameters/limits are computed in v1 order; apply replays. Decisions only for columns that change (passthrough kinds emit nothing). DATETIME_STRING gets no expansion (v1 expanded DATETIME_NATIVE only). | Decision-based port |
| `engineer/engineer.py` | (draft) categorical split | Replaced `nunique >= threshold` inline test with signal kind CATEGORICAL_LOW/HIGH (same threshold, from `signals.py`) | v1 fidelity |
| `engineer/engineer.py` | feature_config | Draft ignored `feature_config` in `plan_engineer`. Interactions, datetime parts, clusters and text features are now Decisions (`create_interactions`, `create_datetime_features`, `create_cluster_features`, `create_text_features`) | needed for "creating" section |
| `engineer/engineer.py` | `run_engineer`, `add_features`, `FeatureAugmenterTransformer` | removed from `src/` (v1 PrepResult-era public API; augmenter logic moved into `EngineerTransformer`) | shim does not belong in the package |
| `engineer/engineer.py` | target encoding | Skipped (no Decision) when there is no target column | v1 raised KeyError |
| `engineer/features.py` | `generate_interaction_features` | takes ranked column list; ranking in `rank_interaction_columns` from signals (v1 read profile correlation/MI); generated names never overwrite an existing column (v1 `add_features` path skipped them, `run_engineer` overwrote) | v1 fidelity + deterministic Decisions |
| `engineer/features.py` | mutual information | `random_state=0` added to `mutual_info_regression/classif` | v1 was non-deterministic, which breaks plan == run |
| `engineer/features.py` | `generate_cluster_features` | returns early for fewer than 2 rows | v1 crashed |
| `engineer/vif.py` | `plan_vif_and_leakage` | now reads signals. Suspected leakage (|r| > 0.95, v1 flag): v1 only reported it and kept the column; v2 proposes `keep_feature` (default) with `needs_attention=True`, and `drop_leaky_feature` is the alternative. Draft proposed dropping by default | v1 never dropped; safe default |
| `engineer/vif.py` | VIF note | note-only Decision `feature_overlap` when VIF > 10 (`EngineerConfig.vif_note_threshold`). NEW: v1 only stored the score. Draft used 20 | task: VIF signal in plan |
| `engineer/config.py` | `EngineerConfig` | new: `leakage_threshold` 0.95 (v1), `vif_note_threshold` 10 (new), `skew_threshold` 1.0 (v1) | named config values |
| `engineer/numeric.py` | `needs_log_transform` | empty-column guard; `standard_scale` NaN-safe mean/std | draft change, kept (no effect on valid input) |
| `engineer/encoding.py` | `target_encode_cross_fit` | fewer rows than folds: falls back to the full-data smoothed mapping | v1 crashed (KFold) |
| `engineer/datetime.py` | `expand_datetime` | string input is parsed instead of raising | draft change, kept; typed input never hits it |
| not ported | class imbalance, mixed-type, high-missing report flags | report-only in v1 (Report PDF is out of scope) | scope |
| not ported | Pandas4Warning fix at v1 `engineer.py:698` | the code path (`add_features` text-collision restore using `select_dtypes(include="object")`) is replaced by `generate_text_features` collision skipping + `EngineerTransformer`; `select_dtypes(include="object")` no longer appears in `src/` | removed the offending code instead of patching it (verify with `-W error`, see 6.6) |
| `signals.py` | `v1_infer_kind` | unchanged v1 | fallback |

## 6.3 Assembler

| File | Function | Change | Why |
|---|---|---|---|
| `assemble.py` | `CleanerTransformer` | built from Decisions: `fit` re-plans on training data and carries over the action in force from supplied Decisions (matched by kind and column); `transform` = `apply_clean` without row dedupe (as v1) | fit on train only; pipeline matches plan |
| `assemble.py` | `EngineerTransformer` | `fit` re-plans and applies on cleaned training data, keeps fitted state (`specs_`); `transform` ported from v1 incl. missing-column fill with `False` and reindex; target-encoded columns use the full-data mapping + global mean for unseen (v1 behavior, so transformed training rows differ from cross-fit rows by design) | v1 fidelity |
| `assemble.py` | generated features in `transform` | ported from v1 `FeatureAugmenterTransformer`; datetime features are computed from the original date columns before they are split into parts (v1 augmenter looked for them after the split and silently lost them for new data) | v1 bug fixed, logged |
| `assemble.py` | `build_pipeline_two_phase` | signature now (X, y, target, model_hint, decisions, ...); `profiles` argument gone (Profiler not ported) | no Profiler |
| `assemble.py` | `run_assembler` | returns (table, pipeline); v1 returned PrepResult with Report (Report PDF out of scope) | scope |

## 6.4 Plan integration (no v1 logic changed; wiring only)

- Cleaning and engineering steps are registered as `PrepStage`s in v1 execution order (fixing, removing, tidying, filling gaps, outliers, leakage, creating). `section` only picks the Plan PDF heading. All steps read one shared in-memory simulation (`_ensure_prep_plan`), so plan values equal what apply produces.
- Typing answers feed cleaning: `_schema_with_actions` applies the typing actions in force to a schema copy. `resolve` therefore resolves reading/typing first, re-plans cleaning/engineering (`replan_prep`), then asks about those. Strict mode skips re-planning and lists everything open.
- `Plan` now stores `model_hint` and `feature_config` (round-trips through JSON). `ApplyResult` gains `prepared` and `specs`; `data` stays the typed table.
- Triggers: `destructive_action` (clean kinds remove_duplicates, drop_*, clip_outliers with needs_attention) and `suspected_leakage` (engineer kind) are implemented; no empty plug-in points remain. Question builders offer the keep action; safe default keeps the data.
- THRESHOLD CHANGE to flag: `column_drop_impact_threshold` 0.05 -> 0.25. With 0.05 any single column drop on a table of 20 or fewer columns asked a question, which made every identifier/constant column an interruption. Not tuned on the corpus; user may reverse.
- `signals.kind_from_schema`: numeric columns v2 did not call IDs now still get v1's numeric-ID rule; ID-typed text of plain numbers maps to NUMERIC_ID.
- Banned-jargon list extended (interaction, target encoding, cross-fit, VIF, one-hot, ordinal, tf-idf, k-means, centroid, outlier, median, log1p, z-score, leakage).
- Tests: new `tests/unit/test_prep_integration.py` (9); updated test_question, test_store, test_engineer for the new behaviour (no v1 assertions touched).
- Result: `tests/unit` minus `test_assemble.py` = 185 passed. Smoke over 30 corpus files: dry_run Decisions == actual-run Decisions, 30/30. Known: hotels.csv planning takes ~195 s, almost all in Phase 3 type inference (pre-existing).

## 6.5 Regression port (no src logic changed)

- `tests/regression/adapter.py` rewritten: DataFrame -> temp CSV -> `Planner.plan` -> `resolve(mode="defaults")` -> `apply`; no v1 profiler is re-implemented in tests any more (kinds and numbers come from `preflight.signals`). Pipeline `transform` of new rows also goes through a temp CSV and the plan's typing Decisions.
- `tests/regression/test_mvp_acceptance.py` replaced by a byte-faithful copy of v1 `mvp_test.py` (imports + two CLI skip markers only). The draft's reworded docstrings/comments were removed.
- Results and divergences: `docs/port_regressions.md`.
