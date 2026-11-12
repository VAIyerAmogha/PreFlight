# v2 hardening report (Phase 7b, updated in Phase 7c)

**About the held-out numbers:** the held-out set (101 files) was **seen during review, not
independent**. Phase 7b scored it once and reported it, and Phase 7c scored it again after
one tuning change. Phase 7c tuning looked only at tuning-set results
(`docs/tuning_log.md`), but the held-out set no longer gives an unbiased estimate. Treat the
held-out column as one more measured split, not as a blind test.

## Phase 7c final measurement

One run: `.pytest_cache/corpus_results/run_20261030_162635`
(`pytest -m corpus -n 4 --timeout=300`: 2393 passed in 431 s, 0 failed, 0 timeouts).
Scored with `scripts/corpus_summary.py --final` (full output in
`docs/review_notes/phase7c/final_summary.md`) and `scripts/type_accuracy_buckets.py`.

Code changes since Phase 7b that affect these numbers: one tuning change (the name-word rule
for free text needs 5 words per value instead of 3; `docs/tuning_log.md`), and the
Categorical comparison fix in `clean/normalize.py` (affects cleaning, not typing).

| Target | Goal | Tuning (153 files) | Held-out (101, seen during review, not independent) | Combined (254) |
|---|---|---|---|---|
| Files without unhandled exceptions | 100% | PASS 153/153 | PASS 101/101 | PASS 254/254 |
| Dialect accuracy | ≥95% | FAIL 140/153 = 91.50% | FAIL 93/101 = 92.08% | FAIL 233/254 = 91.73% |
| Column type accuracy, strict | ≥95% | FAIL 1624/1765 = 92.01% | FAIL 976/1085 = 89.95% | FAIL 2600/2850 = 91.23% |
| Column type, correct or flagged for review | (no goal set) | 1762/1765 = 99.83% | 1075/1085 = 99.08% | 2837/2850 = 99.54% |
| Wrong type, no flag | ≤1% | PASS 3/1765 = 0.17% | PASS 10/1085 = 0.92% | PASS 13/2850 = 0.46% |
| Decisions with evidence+confidence | 100% | PASS 4736/4736 | PASS 3050/3050 | PASS 7786/7786 |
| needs_attention median | ≤10 | PASS 3 (max 109) | PASS 2 (max 39) | PASS 3 (max 109) |
| Timeouts | 0 | PASS 0 | PASS 0 | PASS 0 |

Change from Phase 7b: tuning strict type accuracy 91.56% -> 92.01%, held-out 89.68% -> 89.95%;
wrong type with no flag on held-out 1.38% -> 0.92%, tuning 0.45% -> 0.17%. Dialect accuracy is
unchanged (no dialect change was made).

The legacy corpus-wide test thresholds all pass: ingest delimiter 99.61%, encoding 98.43%,
header 100%, dialect 98.43%; infer accuracy 91.23%, wrong-unflagged 0.39% (that suite's own
column matching); plan 254/254; interact 254/254 with no hangs; pipeline 142/142 files with a
usable target.

**The 95% goals for dialect and strict column-type accuracy are not met.** Most column-type
misses are ID-like columns (mostly row-number columns) read as numbers, and nearly all of
them are flagged for review. That is why "correct or flagged" sits above 99% while strict
accuracy is about 91%.

## Suggested public limitation wording

> PreFlight reads the file format (encoding, separator, header row) correctly for about 92% of
> the 254 test files, and gives the exact expected column type for about 91% of columns. Most of
> the remaining columns are flagged for your review instead of being changed silently: fewer
> than 1 in 100 columns get a wrong type with no flag. Check the flagged items in the Plan PDF
> before you rely on the output, especially row-number and ID columns.

## Phase 7b record (kept for history)

Phase 7b final run `run_20261030_154227` (no tuning): tuning dialect 91.50%, type 91.56%,
wrong-no-flag 0.45%; held-out dialect 92.08%, type 89.68%, wrong-no-flag 1.38%. Phase 7b error
analysis (tuning only): `docs/review_notes/phase7b/error_analysis.md`.

## Not verified

These have not been checked, and nothing in this report covers them:

- **Python 3.10 and 3.11:** only 3.12 is installed locally. CI is set up for 3.10/3.11/3.12
  but has not been run.
- **Independent accuracy:** no unseen test set exists. The held-out set was seen during review.
- **Files outside the corpus:** 254 public files (Rdatasets, TidyTuesday, FiveThirtyEight,
  UCI, seaborn, pandas and Frictionless parser fixtures; `scripts/acquire_corpus.py`). Behaviour on other sources (exports from Excel, databases or logging
  systems; very wide or very large files) is not measured. The largest corpus file has about 179,000 lines (rainfall.csv).
- **Notebook widgets in a real notebook:** tested headless with mocked widgets, plus one
  headless build with real ipywidgets objects. No real Jupyter front end was used.
- **Terminal channel with a real person:** tested with scripted input only.
- **CLI:** `src/preflight/cli.py` is an empty placeholder; there is no command-line entry
  point (Phase 8).
- **DataFrame input:** `prepare()`/`plan()` take a file path only.
- **Pinned dependency versions:** not pinned. Tested only against the versions in
  `docs/licence_audit.md` (pandas 3.0.6, scikit-learn 1.9.1, numpy 2.5.3, ...).
- **Model quality gains:** the 10-file v1-vs-v2 comparison (`docs/v2_vs_v1_scores.md`) uses
  assumed targets and is too small to claim a general gain.
- **Installed wheel:** the tests ran against the editable install; a built wheel was not
  installed into a clean environment and tested in this phase.
- **Leakage from non-numeric columns (known gap, found in Phase 7c):** the leakage check only
  measures numeric columns (`engineer/vif.py`, `compute_correlation_with_target`). A text or
  yes/no copy of the target is not flagged. Example: `alive` in titanic.csv
  (docs/v2_vs_v1_scores.md). v1 has the same gap.
