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

## Method

- **v1.0.1 path**: raw file → plain `pandas.read_csv(path)` (no manual delimiter/header
  fix-ups — this is deliberately naive, to test what actually happens when v1 is pointed at
  these files with no special handling) → v1.0.1's own `profiler.run_profiler` →
  `assembler.build_pipeline_two_phase` (which internally calls v1's `cleaner.run_cleaner`
  and `engineer.run_engineer`), from `/home/amg/Desktop/PreFlight v1.0.1/src/preflight/`.
- **v2 path**: raw file → `preflight.plan(path, target=, task=, model_hint=)` →
  `preflight.apply(plan_obj, mode="defaults")` → `result.data` (the typed table) →
  `preflight.assemble.build_pipeline_two_phase(X, y, target, model_hint=, decisions=,
  schema=, task=)`, i.e. the same public ingest/plan/apply/assemble flow used elsewhere in
  this repo (`prepare()`).
- **Models**: one linear model (`LogisticRegression`/`LinearRegression`, `random_state=42`),
  one tree model (`RandomForestClassifier`/`RandomForestRegressor`, 100 trees,
  `random_state=42`), fit on standardized (linear) or raw (tree) features.
- **CV**: `RepeatedStratifiedKFold`/`RepeatedKFold`, `n_splits=3`, `n_repeats=3`,
  `random_state=42` (9 scored folds per file/model/path).
- **No leakage**: for every fold, v1's `profiler.run_profiler` + `build_pipeline_two_phase`
  and v2's `build_pipeline_two_phase` are **re-fit on the training fold only**
  (`cleaner.fit(X_train, y_train)`, `engineer.fit(X_cleaned_train, y_train)` for both
  versions); medians, scalers, encodings, winsorize bounds, interaction rankings are all
  computed from train rows only and then applied to the held-out test fold via each
  pipeline's own `transform()`. The one thing computed once outside the fold is column
  *semantic typing* (which columns are numeric/categorical/ID etc.) — standard schema
  inference, not a statistic of the target, so not treated as leakage here.
- `diamonds.csv` (53,940 rows) and `adult.data` (32,561 rows) were subsampled to 4,000 rows
  (fixed seed) purely for CV runtime; this is noted, not hidden.

## Results table

Only files where **both** v1 and v2 produced usable, scoreable features are included here.
Files where one side failed outright are in the Robustness section instead (not folded into
means). Tie rule: if v1 vs v2 means differ by less than one combined standard deviation
(`|v1_mean - v2_mean| < v1_std + v2_std`), it's reported TIE.

| File | Model | v1 mean±std | v2 mean±std | Verdict | Note |
|---|---|---|---|---|---|
| AER_Affairs.csv | linear (R²) | 0.0869 ± 0.0434 | 0.0869 ± 0.0434 | TIE | Identical — no categorical normalization changes anything here (gender/children values are already lowercase, no v2 defect triggered, see Robustness) |
| AER_Affairs.csv | tree (R²) | 0.0456 ± 0.0650 | 0.0479 ± 0.0635 | TIE | Difference (0.002) is far under 1 combined std (0.13) |
| carData_TitanicSurvival.csv | linear (acc) | 0.6259 ± 0.0066 | 0.6264 ± 0.0059 | TIE | Negligible difference |
| carData_TitanicSurvival.csv | tree (acc) | 0.6180 ± 0.0009 | 0.6180 ± 0.0009 | TIE | Identical |
| COUNT_titanic.csv | linear (acc) | 0.7773 ± 0.0112 | 0.7773 ± 0.0112 | TIE | Identical |
| COUNT_titanic.csv | tree (acc) | 0.7923 ± 0.0110 | 0.7923 ± 0.0110 | TIE | Identical |

**On these 3 "clean" files, v2 shows no measurable gain over v1 — stated plainly, as
instructed.** The feature pipelines produce essentially the same numbers because both
v1 and v2 implement the same cleaning/engineering rules by design (v2 was ported from v1,
per `docs/port_log.md`), and these 3 files contain nothing (no messy delimiters, no
case/whitespace-inconsistent categorical text) that would make the two diverge.

No file in this set showed v1-better or v2-better by more than 1 combined std — every
scoreable file/model pair is a TIE.

A separate, unexplained observation on `titanic.csv` (v1 side only, since v2 could not run
on it — see Robustness): v1's linear and tree models both scored 1.000 ± 0.000 accuracy for
`survived`. This is very likely because the raw file contains an `alive` column
(`"yes"`/`"no"`) that is a direct re-encoding of the `survived` target, and `class`/`who`
columns that are near-duplicates of `pclass`/`sex`/`age`; v1's NUMERIC_ID/CONSTANT drop rules
do not catch string-valued leakage columns like `alive`. This is a property of the raw
dataset plus v1's drop rules, not a v2 comparison point (v2 never got the chance to run on
this file, due to the defect below), and is reported here as an observation rather than a
score claim.

## Robustness findings (not folded into the score table)

### 1. v1 fails outright on every file with a naive read (4 files)

Plain `pandas.read_csv(path)` with no manual fix-ups mis-parses 4 of the 10 files before any
v1 profiler/cleaner/engineer code runs at all:

| File | What naive `pandas.read_csv` produces | Why it breaks v1 |
|---|---|---|
| `adult.data` | First data row (`39, State-gov, 77516, Bachelors, 13, ...`) is consumed as the header; resulting columns are string literals like `'39'`, `' State-gov'` | No column is or can be named/indexed `income`; `run_profiler(df, target, task)` cannot locate the target |
| `iris.data` | Same problem: first row (`5.1,3.5,1.4,0.2,Iris-setosa`) becomes the header | No `species` column exists |
| `winequality-red.csv` | Entire semicolon-delimited header+rows load into a single column (comma is the wrong delimiter) | No `quality` column exists — the whole file is one text column |
| `winequality-white.csv` | Same single-column misparse as red wine | Same |

v2's `preflight.plan()`/`apply()` (via `detect_dialect`/`detect_header`/`detect_encoding` in
`preflight.ingest`) correctly detects the no-header case for `adult.data`/`iris.data` and the
semicolon delimiter for both wine-quality files, and produced a typed table with the target
column present and usable in all 4 cases (`v2_status: "ok"` in the raw run output). This is
a genuine, reproducible robustness advantage for v2's ingest stage on these 4 files — but
see finding 2, which prevented these 4 files from reaching a scoreable comparison anyway.

### 2. v2 has a reproducible defect that fails on virtually every categorical column with mixed case or whitespace (6 of 10 files, including all 4 files from finding 1)

`src/preflight/clean/normalize.py`, `normalize_category_values()` (line ~22) lowercases and
strips a categorical-dtype `pandas.Series` with `.apply(...)`. When the column actually
needs changing (any value isn't already fully lowercase/trimmed), pandas returns a new
`Categorical` whose **category set differs** from the original column's category set (e.g.
`['Female','Male']` → `['female','male']`). `plan_normalization()` in
`src/preflight/clean/cleaner.py` (via `clean/normalize.py:82`) then does
`normalized != series`, which pandas raises on for two `Categorical`s with different
category sets:

```
TypeError: Categoricals can only be compared if 'categories' are the same.
```

Minimal reproduction:
```python
import pandas as pd
s = pd.Series(['Female', 'Male', 'Female'], dtype='category')
normalized = s.apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
(normalized != s)   # raises TypeError
```

This crashes `plan_clean_full` → `CleanerTransformer.fit` → `build_pipeline_two_phase`
deterministically, for **every** fold, on any file that has a categorical column with
uppercase letters or leading/trailing whitespace anywhere in its values. That covers 6 of
the 10 files selected here:

| File | Example categorical value that triggers it |
|---|---|
| `adult.data` | `' State-gov'` (leading space + mixed case, workclass) |
| `iris.data` | `'Iris-setosa'` (mixed case, species) |
| `titanic.csv` | `'S'`/`'C'`/`'Q'` (embarked, uppercase) |
| `tips.csv` | `'Female'`/`'Male'`, `'No'`/`'Yes'`, `'Sun'`, `'Dinner'` |
| `diamonds.csv` | `'Ideal'`, `'E'`, `'SI2'` (cut/color/clarity) |

(`winequality-red/white.csv` have no categorical columns at all, so this specific defect
does not apply to them — but they are already excluded by finding 1.)

**Net effect**: of the 10 files, v2 could only be scored on 3 (`AER_Affairs.csv`,
`carData_TitanicSurvival.csv`, `COUNT_titanic.csv`) — the only 3 whose categorical columns
happened to already be lowercase/whitespace-clean in the raw data (e.g. `'female'`/`'male'`,
`'yes'`/`'no'`, `'1st class'`). This is a real, src-level defect in v2's `clean/normalize.py`
discovered during this comparison, not a limitation of the comparison methodology. I did not
patch it — the user's resolution authorized target inference, not scope to fix unrelated
pipeline bugs, and fixing it would change what's being measured.

## Summary

- 3 of 10 files produced a genuine head-to-head score comparison: **all 6 model/file pairs
  are TIEs** (v2 shows no regression and no gain vs. v1 on these).
- 4 of 10 files show v2's ingest succeeding where v1's naive `pandas.read_csv` fails outright
  (robustness finding in v2's favor, but these never reached scoring because of finding 2).
- 6 of 10 files (a superset including those 4) hit a reproducible v2 defect in
  `clean/normalize.py` that crashes cleaning whenever a categorical column has mixed-case or
  whitespace content — these never reached scoring either.
- No file in this set shows v1-better or v2-better by score; the only real differentiators
  found were robustness ones (in opposite directions, on different files).
