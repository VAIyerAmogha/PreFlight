# v1.0.1 regression port: divergences and skips (Phase 6 / 6.5)

Source: `/home/amg/Desktop/PreFlight v1.0.1/tests/mvp_test.py` (the "test_mvp_acceptance" suite). v1 baseline run: 47 collected, 47 passed.
Port: `tests/regression/test_mvp_acceptance.py` (bodies and assertions unchanged; only the import line and two skip markers differ, check with `diff`), run through `tests/regression/adapter.py` (DataFrame -> CSV in a temp dir -> v2 `Planner.plan` -> `resolve(mode="defaults")` -> `apply`).

Result of the last run (raw output reproduced in the 6.5 report): **47 collected: 41 passed, 4 failed, 2 skipped.** Repeated 4 times: same result (the tests use unseeded random data; the outcome did not vary).

No assertion was edited. The 4 failures are listed for you to decide.

## Failures (you decide each)

| # | Test | v1 behaviour | v2 behaviour | Assessment |
|---|------|--------------|--------------|------------|
| 1 | `TestRegression_StringAndMultiClassTargets::test_multiclass_string_target_does_not_crash` (asserts `result.df.shape[0] == 150`) | `prepare` returned 150 rows. v1's `clean()` removed duplicate rows, but `prepare()` built its output through the pipeline, whose transform skips row de-duplication. | `apply` removes the one exact duplicate row of iris (the removal is below the 2% impact threshold, so it is applied without asking): 149 rows. | **Equivalent in intent / v1 inconsistent.** v1's own `clean()` would also give 149; only `prepare()` skipped it. v2's pipeline output still keeps all rows (de-dup is skipped on new data, as in v1). Decide: keep v2, or change the dedupe default. |
| 2 | `TestRegression_IdLikeColumnsNotCoercedToNumeric::test_phone_number_like_column_not_coerced` | Test passes a DataFrame of strings; v1 kept them from becoming a numeric feature. | A CSV cannot carry "this was text". v2 reads the 10-digit values as whole numbers (type_inference `int`, confidence 1.0, no question) so the column becomes NUMERIC_FEATURE. | **Equivalent for CSV input; the test input is not representable through a CSV.** If you want phone-like digit runs kept as identifiers, that is a Phase 3 inference rule (e.g. digit-run length), not a port change. |
| 3 | `TestRegression_IdLikeColumnsNotCoercedToNumeric::test_genuinely_messy_numeric_string_still_coerces_correctly` | v1 coerced the messy numeric string column (blanks, "N/A", padded values) to NUMERIC_FEATURE automatically. | v2 proposes `cast_to_float` but marks it `needs_attention` (lossy: 5 entries become blank) with the safe default "Keep as plain text". Under defaults mode the column stays text. Under `ask` the recommended option is also "Keep as plain text"; "Convert to decimal numbers" is offered. | **Deliberate Phase 4/5 behaviour, not a regression of the port.** v1 never asked. Decide whether the safe default for a column that is 99% numbers should be the conversion. |
| 4 | `TestRegression_UnitSuffixNumericParsing::test_simple_unit_suffix_column_parses_as_numeric` ("1248 CC") | v1 coerced common unit suffixes (CC, kmpl, Nm, bhp) to numeric. | v2's unit list (`infer/types.py` `UNIT_REGEXES`: $, %, kg/g/mg, km/m/cm/mm, kmpl/kpl/mpg/mph/kph, years) has no `CC`, so the column is inferred categorical (confidence 0.95), no question is raised. Units that are on the list (e.g. `kmpl`) do raise a question. | **v2 regressed** relative to v1 for suffixes outside its list. Fix would be in Phase 3 inference (extend the list or ask), which is out of scope for this port. |

## Skips in the port (2)

| Test | Reason |
|------|--------|
| `TestCliRoundTrip::test_cli_full_flag_combination_on_string_target_dataset` | v2 has no CLI in this phase (task: no CLI commands). Body is the v1 body, unchanged, with a skip marker. Must stay skipped until the CLI exists. |
| `TestCliRoundTrip::test_cli_errors_are_clean_not_tracebacks` | Same. |

## Tests that pass but assert little (for your awareness; not changed)

- `test_iqr_zero_skip_is_logged` ends with `assert "skip" in ... or True`: always true.
- `test_complex_nested_unit_expression_documented_limitation` accepts either of two outcomes.
- `test_save_compare_pdf_validates_prepresult_types_before_comparing`, `test_compare_and_pdf_comparison_on_rich_dataset`, `test_report_has_all_export_methods`, `test_all_expected_top_level_functions_exist`, `test_presets_include_fast_and_thorough`, `test_feature_config_has_all_v1_fields`, `test_semantic_type_includes_text` exercise the v1 result containers (`Report`, `PrepResult`, `compare`, `save_compare_pdf`), which live in the compatibility modules (`src/preflight/types.py`, `report/report.py`, `validation.py`) and in the adapter. They pass without touching the v2 planner, so they say little about the port.

## The 15 v1 skips

All 15 come from one place: `tests/test_full.py:770`, in `TestFuzzSweepRandomizedColumnCounts::test_randomized_shape_and_missingness_combinations`. The test is parametrized over `n_rows` (5 values) x `n_numeric_cols` (0, 1, 5) x `null_rate` (3 values); the `n_numeric_cols == 0` cases (5 x 3 = 15) call `pytest.skip("no feature columns is covered explicitly in edge case section")`.

- Cause: not an environment problem. The skip is hard-coded for the zero-feature case.
- Is the reason true? Yes: `tests/test_full.py::TestEdgeCasesDegenerateShapes::test_single_column_dataframe_plus_target` asserts the zero-feature case raises `ValueError`, and it passes in the v1 run (402 passed).
- Resolution: **document, leave as is.** These 15 are one duplicate case counted 15 times; `test_full.py` is not part of the 47 + 13 port, so nothing is ported or removed. Nothing is skipped in the v2 port for this reason.
- Also seen in the v1 run: `tests/test_coverage_gate.py` fails (it shells out to `pytest --cov`, and `pytest-cov` is not installed in this venv). It is unrelated to the skips.

## "13 bug repros": NOT ported, needs your decision

I could not identify a separate set of 13 bug-repro tests in v1. The only tests that are labelled as bug regressions are in `mvp_test.py` itself ("SECTION 1 - Regression tests for every bug found in the stress-test cycle", bugs 1.1, 1.2, 2.1-2.5, 3.1-3.3): 19 tests in 6 classes, and they are already inside the 47. A search of every other v1 test file for bug/regression/repro wording found nothing that is a bug repro. I did not guess. Please tell me which 13 tests you mean (file and names), or confirm the 19 above are what you meant.

## Environment note

While investigating the skips I ran the v1 suite with this repo's venv. v1's `test_scaffold.py` runs `pip install -e .` inside the v1 folder, which replaced the editable install of this repo with v1.0.1 in `.venv`. I reinstalled this repo (`pip install -e . --no-deps`); `import preflight` resolves to `src/preflight` (2.0.0.dev0) again. No project files were changed by this.
