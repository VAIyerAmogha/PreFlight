# Phase 8 readiness (Phase 7c)

The held-out set was **seen during review, not independent** (see docs/v2_hardening_report.md).
Corpus run: `.pytest_cache/corpus_results/run_20261030_162635`.

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | No unhandled exceptions or timeouts | PASS | `pytest -m corpus -n 4 --timeout=300`: 2393 passed, 0 failed in 431 s. Accuracy suite 254/254 files with no unhandled exception, 0 timeouts in every suite (ingest, infer, plan, interact, pipeline, accuracy). |
| 2 | Wrong type with no flag ≤1% (held-out) | PASS | Held-out 10/1085 = 0.92% (Phase 7b: 1.38%). Tuning 0.17%, combined 0.46%. The held-out set was seen during review, so this is not an independent estimate. |
| 3 | No silent failure paths | PASS | `scripts/exception_audit.py`: 10 handlers swallow an exception, 10 justified, 0 unjustified. Listed below. |
| 4 | Fit hygiene: test passes, pipeline unfitted and cloneable | PASS | `tests/unit/test_fit_hygiene.py` 7/7 passed, including `test_pipeline_is_unfitted_and_clone_round_trips` and `test_fitting_inside_a_fold_uses_only_the_training_rows`. |
| 5 | Dry-run purity, plan/apply equivalence, never blocks non-interactively | PASS | `tests/unit/test_dry_run.py`, `test_plan_apply_equivalence.py`, `test_nonblocking.py`: 17/17 passed. Corpus interact suite: 254/254, 0 hung files. |
| 6 | ruff 0, mypy 0, zero unexplained skips | PASS | `ruff check src tests`: 0. `mypy --strict src/preflight`: 0 issues in 55 files. Fast suite: 346 passed, 2 skipped, 1 xfailed (2 warnings: pytest deprecation notices about class-scoped fixtures). The 2 skips are the CLI tests (`test_mvp_acceptance.py:517,540`: "v2 has no CLI in this phase"). The 1 xfail is `test_phone_number_like_column_not_coerced` (strict xfail: a CSV cannot record that a column was text, and 10-digit strings look the same as integers; real phone formats are covered in `test_units_generic.py`). |
| 7 | Report PDF cannot be disabled; results saved before render | PASS | `prepare()` has no argument to turn the Report PDF off (`src/preflight/prepare.py`). `write_report_artifacts` writes cleaned data, schema, decisions and pipelines before rendering. `tests/unit/test_report_pdf.py::test_pdf_render_failure_preserves_already_saved_artifacts` passes. |
| 8 | Dialect and column-type accuracy vs 95% | **FAIL** | Dialect: tuning 91.50%, held-out 92.08%, combined 91.73%. Column type, strict: tuning 92.01%, held-out 89.95%, combined 91.23%. Column type, correct or flagged: tuning 99.83%, held-out 99.08%, combined 99.54%. Neither 95% goal is met on strict accuracy. |
| 9 | Open decisions, with what the code does today | PASS (documented) | See "Open decisions" below. |
| 10 | Not-verified list | PASS (documented) | See "Not verified" below; the same list is in docs/v2_hardening_report.md. |

**Verdict: NO-GO.** Gate 8 fails: dialect and strict column-type accuracy are about 92% and
91%, not 95%. All other gates pass. To get to GO, the owner can either accept the 95% goals as
unmet and publish the limitation wording in docs/v2_hardening_report.md, or fund another
accuracy round. That round needs a new, unseen test set, because the current held-out set
has been seen.

## Gate 3: remaining swallowed exceptions (all justified)

| Location | Exception | Why it is safe |
|---|---|---|
| src/preflight/engineer/features.py:50 | Exception | Mutual information is an optional signal; None means "not available". |
| src/preflight/engineer/vif.py:51 | LinAlgError | A singular correlation matrix leaves every VIF score as None ("not computed"). |
| src/preflight/infer/types.py:496 | ValueError | A value that is not a plain integer means the column is not a sequence. |
| src/preflight/ingest/dialect.py:94 | Exception | A candidate dialect that cannot parse the sample scores zero. |
| src/preflight/ingest/encoding.py:149 | UnicodeDecodeError | Not valid UTF-8, so the next detector decides. |
| src/preflight/interact/modes.py:54 | Exception | If IPython cannot report its shell, it is not a usable notebook kernel. |
| src/preflight/interact/resolve.py:57 | Exception | Previews are optional; with None, every preview says the values are unavailable. |
| src/preflight/interact/resolve.py:72 | ImportError | No widgets installed: fall back to the terminal prompt. |
| src/preflight/report/report.py:289 | Exception | Reported as a warning; the PDF still renders without that plot. |
| src/preflight/signals.py:167 | ValueError, TypeError, OverflowError, ParserError | Counting parse failures is the measurement. |

Source: docs/review_notes/phase7c/exceptions.md. One quiet gap that is not an exception: the leakage check
(`engineer/vif.py`, `compute_correlation_with_target`) only measures numeric columns, so a text or yes/no
copy of the target is not flagged. On titanic.csv, `alive` (yes/no) copies `survived` and stays in the
features (docs/v2_vs_v1_scores.md). v1 has the same gap.

## Open decisions (what the code does today)

| Decision | Today |
|---|---|
| DataFrame input | Not in 2.0. `plan()`, `dry_run()` and `prepare()` take a file path only. The regression suite reaches v2 from a DataFrame through a test-only adapter (`tests/regression/adapter.py`) that writes a CSV first. |
| CLI | Phase 8. `src/preflight/cli.py` is an empty placeholder and no console entry point is declared. `typer` is still a runtime dependency although nothing imports it. |
| Notebook widgets | Tested only with mocked widgets, plus one headless build with real ipywidgets objects (`tests/unit/test_channels.py`). Never run in a real Jupyter front end. |
| Python floor | `requires-python >=3.10`; CI matrix 3.10, 3.11, 3.12 (not yet run). |
| Dependency pins | Not pinned. Licences in docs/licence_audit.md (1 flag: hypothesis, MPL-2.0, dev only). |

## Not verified

- Python 3.10 and 3.11: only 3.12 is installed locally; the CI matrix has not run.
- Independent accuracy: no unseen test set exists; the held-out set was seen during review.
- Files from other sources (Excel exports, database dumps, logs, very wide files). The corpus is 254
  public files; the largest has about 179,000 lines.
- Notebook widgets in a real notebook; the terminal channel with a real person (scripted input only).
- A built wheel installed in a clean environment (tests used the editable install).
- Leakage detection for non-numeric copies of the target (known gap, above).
- Model-quality gains in general: the 10-file v1-vs-v2 comparison uses assumed targets and is too small
  to support a general claim.
