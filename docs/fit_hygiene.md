# Fit hygiene: which steps learn from data

Rule: steps that learn nothing from data run on the full table in the applied path.
Steps that compute a statistic from data run inside the returned sklearn Pipeline. That
pipeline is unfitted and works with `sklearn.base.clone`, so callers fit it inside their
CV folds.

Two rulings from the project owner (Phase 7b, Stage 2):

- **Q1, plan-time column choices.** Some choices pick columns from whole-table statistics
  but learn no value that is applied to rows: dropping a column for a high missing rate,
  for being constant or for being a numeric ID; giving a column a blank-flag column; and
  the column kind (numeric, low-variety or high-variety categorical) that picks the
  encoder. These are reviewed plan Decisions. They are fixed for every fold, so the column
  set is the same in each one.
- **Q2, order.** A stateless step that today runs after a stateful step stays in that
  order inside the pipeline, so the numbers do not change. Examples are log1p after gap
  filling and clipping, and date and text-length features after the encoders.

So the applied path holds every step before the first one that learns from data. Every
step from rare-value grouping onwards runs in the pipeline, in the same order as before.

| # | Step (Decision kind -> action) | Class | Runs in | Justification |
|---|---|---|---|---|
| 1 | convert_missing_markers | STATELESS | applied path | Replaces the marker strings named by the typing Decisions with blanks, one cell at a time. |
| 2 | coerce_numeric | STATELESS | applied path | Parses each value on its own (removes a unit or stray spaces). Typing picks the columns. |
| 3 | remove_duplicates | STATELESS | applied path | Compares rows within the table given. Nothing carries over to new rows. |
| 4 | drop_high_missingness / drop_constant / drop_id -> drop_column | STATELESS (plan-time choice, Q1) | applied path | A reviewed column choice. No value is learned or applied to rows. |
| 5 | normalize_categories | STATELESS | applied path | Trims and lower-cases each value. |
| 6 | group_rare_categories | STATEFUL | pipeline (`FoldCleaner`) | The rare-value list is a frequency cutoff, learned from the training rows. |
| 7 | add_missing_indicator | STATELESS (plan-time choice, Q1; order, Q2) | pipeline | The flag is worked out row by row. It runs after step 6, as before. |
| 8 | fill_median / fill_mode | STATEFUL | pipeline (`FoldCleaner`) | The fill value is the median or the most common value of the training rows. |
| 9 | constant_fill | STATELESS (order, Q2) | pipeline | Fills with the fixed value `__missing__`. It runs after step 6, so a filled blank is never grouped as a rare value. |
| 10 | clip_outliers | STATEFUL | pipeline (`FoldCleaner`) | The IQR or z-score limits come from the training rows. |
| 11 | suspected_leakage -> drop_leaky_feature | STATEFUL | pipeline (`EngineerTransformer`) | Correlation with the target, recomputed in each fold. If the person chose to leave a column out, it stays out in every fold. |
| 12 | feature_overlap (VIF) | STATEFUL | pipeline (`EngineerTransformer`) | VIF comes from the correlation matrix of the training rows. |
| 13 | expand_datetime | STATELESS (order, Q2) | pipeline | Splits each date into year, month and day parts. |
| 14 | ordinal_encode | STATEFUL | pipeline | The category-to-code mapping is learned from the training rows. |
| 15 | one_hot_encode | STATEFUL | pipeline | The output columns are the categories in the training rows. |
| 16 | target_encode -> target_encode_cross_fit | STATEFUL | pipeline | The mean target per category comes from the training rows. In `fit_transform` each training row gets an out-of-fold value. In `transform`, new rows get the mapping learned in `fit`. |
| 17 | log_transform -> log1p_transform | STATELESS (order, Q2) | pipeline | log1p of each value, after steps 8 and 10 as before. |
| 18 | scale_numeric -> standard_scale | STATEFUL | pipeline | Mean and standard deviation of the training rows. |
| 19 | create_interactions | STATEFUL | pipeline | Columns are chosen by mutual information with the target on the training rows. |
| 20 | create_datetime_features | STATELESS (order, Q2) | pipeline | Cyclical date parts and date differences within a row. |
| 21 | create_cluster_features | STATEFUL | pipeline | k-means centroids from the training rows. |
| 22 | create_text_features | MIXED (order, Q2) | pipeline | Length and word counts are per value. The TF-IDF vocabulary is learned from the training rows. |

The column kinds that pick each encoder (Q1) come from `assemble.plan_time_column_kinds`.
It runs on the full typed table, the same way the plan does, and passes the result to
`EngineerTransformer(column_kinds=...)`. Signals that involve the target, such as
correlation and mutual information, are recomputed on the training rows inside `fit`.

## What the API returns

- `apply(plan)` returns `ApplyResult.prepared`, the full table after steps 1 to 5 only, and
  `ApplyResult.pipeline`, the unfitted pipeline that holds steps 6 to 22. The pipeline's
  input is `prepared` without the target column.
- `prepare()` returns `pipeline` (unfitted, the same object as above) and
  `pipeline_fitted_on_all_rows`, a clone of it fitted on every row. The second one is a
  convenience for transforming new data after model choice is finished. It has seen every
  row, so **it must not be used to evaluate a model**. Both are saved as
  `pipeline.joblib` and `pipeline_fitted_on_all_rows.joblib`, before the Report PDF is
  rendered. `cleaned_data.csv` is `prepared`.
- The plan's Decisions are unchanged. Their evidence values, such as fill values and
  limits, are the whole-table preview shown in the Plan PDF. `pipeline.fit` recomputes
  them on the rows it is given, and the fitted values are in
  `pipeline.named_steps["cleaner"].decisions_` and `named_steps["engineer"].specs_`.
- `build_pipeline_two_phase` and `run_assembler` (the older fit-on-what-you-pass helpers)
  are unchanged. The v1 regression adapter still uses them.

## Tests

- `tests/unit/test_fit_hygiene.py` covers the following:
  - On a leaky toy table with a noise target, the in-fold pipeline scores at chance
    (AUC below 0.60). A full-data fit scores more than 0.10 higher.
  - `clone` round-trips, and the returned pipeline is unfitted.
  - Fill values, clipping limits and the target-encoding mapping fitted inside a fold
    match the training rows only.
  - The rare-value list fitted inside a fold differs from the whole-table list.
  - Target encoding is cross-fit in `fit_transform`.
  - The full table is never filled or encoded.
  - `prepare()` returns both pipelines.
- One test was changed: `tests/unit/test_prep_integration.py::test_leakage_decision_asks_and_answer_is_applied`.
  The person's "leave the column out" answer is now checked on the fitted pipeline's
  output, because leaving a leaky column out is a stateful step. The test still requires
  the column to be kept under defaults and removed when chosen.
- `tests/regression/adapter.py` (test plumbing, not an assertion) rebuilds v1's
  `prepare(df)` table, which v1 fitted on all rows, with
  `clone(result.pipeline).fit_transform(...)`.
