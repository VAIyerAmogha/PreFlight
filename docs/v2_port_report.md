> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# v2 port report (Phase 6 / M5): actual numbers only

All numbers are from one full run: `pytest tests -v` (raw output: `docs/v2_port_pytest_output.txt`, 1825.7 s).
**Total: 790 passed, 4 failed, 2 skipped** (796 collected).

| Suite | Passed | Failed | Skipped |
|-------|-------:|-------:|--------:|
| tests/unit | 209 | 0 | 0 |
| tests/property | 0 collected | - | - |
| tests/regression | 41 | 4 | 2 |
| tests/corpus | 540 | 0 | 0 |

## Regression port (v1 `mvp_test.py` = "test_mvp_acceptance")
- 47 collected: 41 pass, 4 fail, 2 skip (the two CLI tests; v2 has no CLI).
- The 4 failures are v2/v1 divergences, no assertion edited; each is described in `docs/port_regressions.md` for your decision.
- **Bug repros: 0 of 13 ported.** I could not identify a separate set of 13; the 19 bug-labelled tests in `mvp_test.py` are already inside the 47. Waiting for your answer.
- Unexplained skips remaining: **0**. The 15 v1 skips are one parametrized zero-feature case (`tests/test_full.py:770`), explained in `docs/port_regressions.md`; they are not part of the port.

## Corpus: full pipeline on every file with a usable target
The corpus labels carry no target, so the rule is mine (`tests/corpus/prep_worker.py`): target = last column; usable if the table has at least 20 rows, the column has at least 2 distinct values, and (if text) at most 50 distinct values. Each file runs `dry_run(mode="defaults")`, then `apply(plan, mode="defaults")`, in its own subprocess.

| Measure | Value |
|---------|------:|
| Files in corpus | 254 |
| Files with a usable target | 142 (not usable: 93 text target with more than 50 values, 17 under 20 rows, 1 single column, 1 constant target) |
| Completed | 142 |
| Unhandled exceptions / hangs | 0 / 0 |
| Files where plan and run disagree | 0 |
| needs_attention per file (median / max) | 3 / 109 |
| Files above 10 needs_attention | 15 |
| Files with cleaning or engineering Decisions | 142 |
| Slowest files | big_epa_cars.csv 204.5 s, rainfall.csv 117.7 s, ipf_lifts.csv 57.8 s |

The 10 hand-picked datasets in the same file also pass (dry_run Plan PDF sections populated, apply under defaults).
Note: needs_attention counts include the reading and typing questions, not only cleaning and engineering.

## New unit tests this step
- `tests/unit/test_assemble.py` (8 tests, rewritten): pipeline from Decisions reproduces apply for tree and linear; held-out rows use training-only values; row removal skipped on new data; unseen values; pickle and joblib reload; action overlay.
- `tests/unit/test_prep_steps.py` (16 tests): plan purity, apply correctness, tree vs linear, thresholds for row removal, column removal and outlier pull-back (below / above), and edge cases (all-blank column, constant column, single row, high-variety column, perfectly leaky feature, target untouched).
- Earlier in 6.4: `tests/unit/test_prep_integration.py` (9 tests).

## 6.7 Verify (done)
- `pip install -e .`: succeeds.
- `import preflight` with no extras installed: succeeds (`interactive`, `notebook`, `dev` are the only optional-dependency groups; none are needed to import).
- Plan PDF text sample (`corpus/raw/titanic.csv`, target `survived`): all four sections populated (Fixing, Filling gaps, Removing, Creating). Full text in `PHASE6.md` 6.7 action log.
- Phase 5 trigger categories: all 8 implemented, 0 empty plug-ins (`TriggerRegistry.empty_plugin_categories()` returns `[]`).
- Pandas4Warning / `-W error`: `tests/corpus/prep_worker.py` suppresses all warnings in its subprocess, so the corpus run above did not check this. A separate targeted run (`plan`/`dry_run`/`apply` under `warnings.simplefilter("error", ...)` for `FutureWarning` and `Pandas4Warning`, pandas 3.0.6) on 8 corpus files covering the string-date and text/datetime/cluster/interaction code paths: 0 warnings, 0 errors. `select_dtypes(include="object")` (the v1 offending pattern) is absent from `src/`.

## Known caveats
- `column_drop_impact_threshold` was changed 0.05 to 0.25 (logged in `docs/port_log.md`).
- v1 `Report`/`types`/`validation` compat modules remain in `src/preflight` (committed in the earlier draft); the regression adapter uses them.
- Nothing was committed.
