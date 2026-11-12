> **Updated in Phase 7c (2026-10-30):** the score table was re-run after the Categorical fix; the Phase 7 table is replaced. Held-out numbers in this document come from a set **seen during review, not independent**.

# v2 vs v1.0.1: feature-building comparison on 10 corpus files

## IMPORTANT DISCLOSURE — targets are my assumption, not corpus ground truth

The corpus label schema (`corpus/labels/*.json`, `corpus/labels/split.json`) has **no
target/task field anywhere** — only `file_id`, `filename`, `dialect`, and per-column
`column_name`/`column_index`/`semantic_type`/`anomalies`. There is no ground-truth target
for any corpus file.

Every target column and task (classification/regression) used below was **assigned by me**,
using the rule stated in the next section, specifically to make this comparison runnable.
These are not corpus-labeled targets. Where the file is a recognizable, well-known public
dataset I used that dataset's standard, commonly-documented outcome variable (disclosed
per-file below as "recognized standard"); I did not need to fall back to a generic rule
(e.g. "last column") for any of the 10 files chosen, because all 10 are standard named
public datasets with well-known targets — this is disclosed explicitly per file in the table
below, not left implicit.

## Selection rule

Candidates were restricted to files that are (a) a recognizable named public dataset so a
target assignment is defensible, and (b) inspectable by eye for messy/clean structure
without consulting the label JSON. I inspected the raw bytes of each candidate directly
(`head`, line counts, delimiter check) before picking. "Messy" = at least one of: no header
row, non-comma delimiter, missing-value markers requiring interpretation, or heavy missing
data / mixed-type columns observed in the raw file. "Clean" = proper header, consistent
comma delimiter, directly parseable with plain `pandas.read_csv`, no misleading markers.

I picked 5 messy + 5 clean (satisfies the ≥4 messy / ≥3 clean requirement) specifically
because the messy set splits into two sub-cases that matter for different reasons:
(1) files where naive `pandas.read_csv` structurally misparses the file (no header, wrong
delimiter) — a direct test of v2's ingest/dialect-detection value, and (2) a file that
parses fine but has real missing-value/mixed-type content — a test of actual cleaning
logic, not just ingest.

## The 10 files, messy/clean judgment, and target assignment

| # | File | Messy/Clean | Why (raw-file inspection) | Target | Task | Target rule |
|---|---|---|---|---|---|---|
| 1 | `adult.data` | Messy | No header row at all; raw first line is already a data row (`39, State-gov, 77516, ...`) | `income` (last column) | classification | **Recognized standard** — UCI "Adult" dataset; `income` (`<=50K`/`>50K`) is the dataset's documented target |
| 2 | `iris.data` | Messy | No header row; first line is a data row (`5.1,3.5,1.4,0.2,Iris-setosa`) | `species` (5th/last column) | classification | **Recognized standard** — UCI Iris; species is the canonical target |
| 3 | `winequality-red.csv` | Messy | Semicolon-delimited with quoted headers (`"fixed acidity";"volatile acidity";...`), not comma | `quality` (last column) | regression | **Recognized standard** — UCI Wine Quality; `quality` is the documented target |
| 4 | `winequality-white.csv` | Messy | Same semicolon-delimiter issue as #3 | `quality` (last column) | regression | **Recognized standard** — same dataset family as #3 |
| 5 | `titanic.csv` | Messy | Proper header/comma delimiter, but real missing values (`deck` mostly blank, `age` has gaps) and mixed representations (`class` string duplicates `pclass` numeric, `alive` string duplicates `survived`) | `survived` | classification | **Recognized standard** — seaborn "titanic" dataset; `survived` is the standard outcome |
| 6 | `tips.csv` | Clean | Quoted, comma-delimited, no missing values, consistent header | `tip` | regression | **Recognized standard** — seaborn "tips"; `tip` is the standard regression target |
| 7 | `AER_Affairs.csv` | Clean | Comma-delimited, consistent header, no missing markers | `affairs` | regression | **Recognized standard** — AER (Fair, 1978) "Affairs" dataset; `affairs` (count of extramarital encounters) is the documented target. This is the exact example given in the task's own resolution text. |
| 8 | `diamonds.csv` | Clean | Comma-delimited, quoted header, consistent, no missing markers (large: 53,940 rows) | `price` | regression | **Recognized standard** — ggplot2 "diamonds" dataset; `price` is the standard regression target |
| 9 | `carData_TitanicSurvival.csv` | Clean | Comma-delimited, consistent header, `rownames` is a quoted passenger-name string (clearly an ID, not messy content) | `survived` | classification | **Recognized standard** — car package "TitanicSurvival"; `survived` is the documented target |
| 10 | `COUNT_titanic.csv` | Clean | Comma-delimited, consistent header, fully categorical/coded columns | `survived` | classification | **Recognized standard** — COUNT package "titanic" (Hosmer & Lemeshow); `survived` is the documented target |

No fallback rule ("last column" / "only clear numeric non-ID column") was needed for any of
the 10 — all are recognized named public datasets with a standard outcome variable. This is
stated explicitly rather than left to be inferred.

## Method (Phase 7c re-run, `scripts/compare_v1_v2.py`)

This is the same method as the Phase 7 comparison: fitting inside each fold, fixed seeds,
repeated CV, and ties within one standard deviation. It is now a script, so it can be re-run.

- **v1.0.1 path:** raw file -> plain `pandas.read_csv(path)` (no delimiter or header fix-ups,
  on purpose) -> in each fold, v1's `profiler.run_profiler` and
  `assembler.build_pipeline_two_phase` on the training rows only -> `transform` on the test rows.
  Code from `/home/amg/Desktop/PreFlight v1.0.1/src/preflight/`.
- **v2 path:** `preflight.plan(path, target=, task=, model_hint=)` ->
  `preflight.apply(plan, mode="defaults")` -> in each fold, `sklearn.base.clone(result.pipeline)`
  fitted on the training rows of `result.prepared` only -> `transform` on the test rows. This is the
  Phase 7b fit-hygiene pattern (docs/fit_hygiene.md). Column choices made at plan time (types,
  dropped columns, encoder choice) are fixed for all folds.
- **Models:** linear (`LogisticRegression` / `LinearRegression` after standardizing) and tree
  (`RandomForestClassifier` / `RandomForestRegressor`, 100 trees), `random_state=42`. A median
  imputer runs first on both paths, in case either leaves a blank.
- **Leftover text columns:** if a version's output still has a non-numeric column, it is
  one-hot encoded with the training-fold categories. This applies to both versions alike.
  In practice it applies to v1, which leaves yes/no-style text columns (`sex` = man/woman)
  unencoded. Without this step those columns became blanks, and v1 scored at the
  majority-class baseline (0.6824 on COUNT_titanic.csv). That would be a harness artefact, not
  a v1 result.
- **CV:** `RepeatedStratifiedKFold` / `RepeatedKFold`, 3 splits x 3 repeats, `random_state=42`:
  9 scored folds per file, model and version.
- **Subsampling:** `adult.data`, `diamonds.csv` and `winequality-white.csv` have more than 4,000
  data rows. They are cut to 4,000 rows (seed 42) at the file level, so both versions read the
  same rows.
- **Tie rule:** `|v1_mean - v2_mean| <= v1_std + v2_std` -> TIE. Equal means with zero spread
  count as a tie.

Raw fold scores: `docs/review_notes/phase7c/compare_v1_v2.json`; table built by
`scripts/compare_table.py`.

## Results (Phase 7c, replaces the Phase 7 table)

| File | Model | v1 mean ± std | v2 mean ± std | Verdict | Note |
|---|---|---|---|---|---|
| adult.data | linear (acc) | failed | 0.8466 ± 0.0074 | v1 failed | No header row: plain `read_csv` uses the first data row as names, so there is no `income` column. Subsampled to 4,000 rows. |
| adult.data | tree (acc) | failed | 0.8502 ± 0.0056 | v1 failed | Same. |
| iris.data | linear (acc) | failed | 0.9546 ± 0.0245 | v1 failed | No header row; no `species` column. |
| iris.data | tree (acc) | failed | 0.9569 ± 0.0215 | v1 failed | Same. |
| winequality-red.csv | linear (R²) | failed | 0.3597 ± 0.0382 | v1 failed | Semicolon separator: plain `read_csv` gives one column. |
| winequality-red.csv | tree (R²) | failed | 0.4557 ± 0.0271 | v1 failed | Same. |
| winequality-white.csv | linear (R²) | failed | 0.2794 ± 0.0242 | v1 failed | Same. Subsampled to 4,000 rows. |
| winequality-white.csv | tree (R²) | failed | 0.4702 ± 0.0194 | v1 failed | Same. |
| titanic.csv | linear (acc) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | TIE (leakage, not a real score) | `alive` (yes/no) is a copy of `survived`; neither version removes it. See finding 3. |
| titanic.csv | tree (acc) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | TIE (leakage, not a real score) | Same. |
| tips.csv | linear (R²) | 0.3825 ± 0.0689 | 0.3964 ± 0.0631 | TIE | |
| tips.csv | tree (R²) | 0.1956 ± 0.1719 | 0.1968 ± 0.1187 | TIE | |
| AER_Affairs.csv | linear (R²) | 0.0869 ± 0.0460 | 0.0932 ± 0.0422 | TIE | |
| AER_Affairs.csv | tree (R²) | 0.0459 ± 0.0635 | 0.0660 ± 0.0651 | TIE | |
| diamonds.csv | linear (R²) | 0.9202 ± 0.0053 | 0.9183 ± 0.0097 | TIE | Subsampled to 4,000 rows. |
| diamonds.csv | tree (R²) | 0.9616 ± 0.0036 | 0.9635 ± 0.0062 | TIE | Subsampled to 4,000 rows. |
| carData_TitanicSurvival.csv | linear (acc) | 0.6175 ± 0.0022 | 0.7841 ± 0.0093 | v2 better | v1 is at the majority-class rate: it reads `passengerClass` (1st/2nd/3rd) as a date, which gives one constant value. v2 ordinal-encodes `passengerClass`. Both versions target-encode the unique passenger-name column `rownames`. v2 types it as an ID but still encodes it, using cross-fitting; this is noted, not fixed. |
| carData_TitanicSurvival.csv | tree (acc) | 0.6180 ± 0.0010 | 0.7545 ± 0.0207 | v2 better | Same. |
| COUNT_titanic.csv | linear (acc) | 0.7773 ± 0.0119 | 0.7773 ± 0.0119 | TIE | |
| COUNT_titanic.csv | tree (acc) | 0.7923 ± 0.0116 | 0.7844 ± 0.0132 | TIE | |

Counts over 20 file/model pairs: v1 failed 8, TIE 10 (2 of them are the leaky titanic.csv
pairs), v2 better 2, v1 better 0.

## Findings

### 1. v1 cannot read 4 of the 10 files with a plain read

Plain `pandas.read_csv(path)` misreads `adult.data` and `iris.data` (no header row) and both
wine files (semicolon separator) before any v1 code runs. v2's reading stage (dialect, header
and encoding detection) reads all 4 correctly, and all 8 v2 file/model pairs produce scores.

### 2. The v2 Categorical defect from Phase 7 is fixed

Phase 7 found that `clean/normalize.py` raised `TypeError: Categoricals can only be compared if
'categories' are the same` on any Categorical column with upper-case letters or extra spaces.
That error blocked 6 of the 10 files. It was fixed in Phase 7c, which now compares plain values,
and has a regression test (`tests/unit/test_normalize_categorical.py`). All 10 files now run
on v2.

### 3. Target leakage on titanic.csv (both versions)

`alive` (yes/no) is a direct re-encoding of the target `survived`, and both versions keep it. v2
reads it as a yes/no column, and its leakage check only measures numeric columns
(`engineer/vif.py`, `compute_correlation_with_target`), so it is not flagged. Both versions
therefore score 1.000. This is a known gap and is listed in docs/v2_hardening_report.md.

## Summary

- On the 6 files both versions can read, 10 of 12 pairs are TIEs. 2 of those ties are
  meaningless because of the leakage in titanic.csv. The remaining 2 pairs (carData_TitanicSurvival)
  are v2 better, because v1 reads a class column as a date. No pair is v1 better.
- On the 4 messy files, v1 fails to read the file; v2 reads and scores all of them.
- 10 files and assumed targets are too few to claim a general accuracy gain. v2's clear
  advantage here is that it reads messy files and gets column types right; on files both
  versions read, the feature building gives about the same scores.

---

## Corpus scores: v1.0.1 vs v2 after Phase 7c (254 files, 2,850 labelled columns)

| Metric | v1.0.1 (docs/v1_baseline_report.md) | v2 Phase 7c (combined) |
|---|---|---|
| Files with an unhandled exception | 10/254 (3.94%) | 0/254 |
| Dialect accuracy | 91.73% | 91.73% (233/254) |
| Column type accuracy, strict | 80.77% | 91.23% (2600/2850) |
| Wrong type with no flag (all columns) | 9.40% | 0.46% (13/2850) |

v2 numbers come from the single Phase 7c run (docs/v2_hardening_report.md). They combine the
tuning set and the held-out set, and the held-out set was **seen during review, not
independent**. Dialect accuracy is level with v1: v2 gets different files right and wrong, and
the totals happen to match.
