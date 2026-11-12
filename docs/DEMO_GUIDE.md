# PreFlight — Demo Guide

Every step below was run for real on 2026-10-30, in a fresh virtual environment, on Linux
with Python 3.12. The output shown is the actual output, shortened only where marked `...`.
Two steps revealed bugs, marked **⚠ BUG**. The source was not changed.

## 1. Install in a clean environment

```bash
cd PreFlight                       # the repository root
python3 -m venv .venv-demo
.venv-demo/bin/pip install .                          # core
.venv-demo/bin/pip install ".[interactive,notebook]"  # extras: rich (terminal), ipywidgets (notebook)
.venv-demo/bin/python -c "import preflight; print(preflight.__version__)"
```
Output:
```
2.0.0.dev0
```
Without the extras everything still works: the terminal uses plain text, and the notebook form
is unavailable. `tests/unit/test_no_extras.py` checks this.

## 2. The three sample files (`examples/`)

| File | What is special |
|---|---|
| `examples/clean.csv` | 60 rows: `age,income,city,bought`. Comma-separated, has a header. |
| `examples/messy_noheader_semicolon.csv` | 60 rows: **no header row**, **semicolons**, e.g. `5.5;2.9;2.2;versicolor`. |
| `examples/disguised_missing_units.csv` | `patient_id,weight,temp,smoker,outcome`. `weight` has a unit (`69 kg`) and `N/A`, `temp` has `-999`, `smoker` has `?`. |

The values are random, so the models below cannot learn much. These files are for showing
behaviour, not accuracy. Run every command from inside `examples/`.

## 3. Dry run and the Plan PDF

```python
import preflight
plan, pdf = preflight.dry_run("messy_noheader_semicolon.csv", mode="defaults")
print([c.name for c in plan.schema.columns])
print([c.inferred_type for c in plan.schema.columns])
for d in plan.decisions[:3]:
    print(d.kind, "->", d.action, "| conf", round(d.confidence, 2), "| ask:", d.needs_attention)
```
Output:
```
PreFlight dry run: output written to /home/amg/Desktop/PreFlight/examples/preflight_out
['col_0', 'col_1', 'col_2', 'col_3']
['float', 'float', 'float', 'categorical']
encoding -> use_encoding_utf-8 | conf 1.0 | ask: False
dialect -> separate_by_; | conf 0.9 | ask: False
header -> no_header | conf 0.98 | ask: False
```
Files: `preflight_out/plan.json`, `preflight_out/plan.pdf`. The semicolon and the missing header
were both detected. Plain `pandas.read_csv` would have produced one column, with the first data
row used as names.

**⚠ BUG:** `plan.schema.dialect.delimiter` and `plan.schema.dialect.effective_delimiter()`
return `','` for this file, and `plan.schema.header.header_row_index` returns `0`. The reading
Decisions (`separate_by_;`, `no_header`) are what is actually used, and the data is read
correctly. **Read the Decisions, not those schema fields.**

**Reading the Plan PDF** (`plan.pdf`), top to bottom:
1. A summary box: file, rows/columns, target, goal, and how many items need review.
2. **Needs your decision**: each open question, its options, and the recommended safe choice.
3. **What we will do**: Reading / Fixing / Filling gaps / Removing / Creating. An empty group
   says "Nothing to do at this stage."
4. **What we won't touch**: your source file is never changed.
5. A sign-off checklist.

## 4. `prepare()` in auto mode (a file with no open questions)

```python
out = preflight.prepare("clean.csv", target="bought")
print(sorted(out))
for k, p in out["paths"].items(): print(k, p.name)
print("prepared shape:", out["apply_result"].prepared.shape)
```
Output:
```
PreFlight: output written to /home/amg/Desktop/PreFlight/examples/preflight_out
['apply_result', 'paths', 'pipeline', 'pipeline_fitted_on_all_rows', 'plan']
cleaned_data cleaned_data.csv
schema schema.json
decisions decisions.json
pipeline pipeline.joblib
pipeline_fitted_on_all_rows pipeline_fitted_on_all_rows.joblib
report_pdf report.pdf
plan_json plan.json
plan_pdf plan.pdf
prepared shape: (60, 4)
```
`mode="auto"` (the default) means: ask if you are at a terminal or in a notebook, otherwise
`strict`. This file had no open questions, so it simply ran.

## 5. Open questions with no terminal: strict stops and tells you what to do

```python
preflight.prepare("disguised_missing_units.csv", target="outcome", out_dir="out_disguised")
```
Run as a script (no terminal attached), the output is:
```
preflight.interact.modes.StrictResolutionError: Strict mode: 5 unresolved item(s), and this session cannot ask (no terminal or notebook). To proceed: rerun with mode='defaults' to accept the safe default for each item; or write a review file with mode='ask', channel='review_file', fill in its answers, and run again (stored answers are reused); or run interactively with mode='ask'.
  - id: dec_infer_patient_id_1e3f2400 | column: patient_id | How should column 'patient_id' be read?
  - id: dec_infer_weight_b4b5b4ac | column: weight | How should column 'weight' be read?
  - id: dec_infer_temp_07b0442b | column: temp | How should column 'temp' be read?
  - id: dec_infer_smoker_5ba7cb44 | column: smoker | How should column 'smoker' be read?
  - id: dec_infer_outcome_7068fdc6 | column: outcome | What type of prediction should be made for target 'outcome'?
```
It never hangs. Choose one of the three ways forward (steps 6–8).

## 6. Answering in the terminal

At a real terminal, `prepare(..., mode="ask")` shows one screen and then asks once per item.
Here the input was scripted, so it could be recorded:
```python
import io
from preflight.interact.channels.terminal import TerminalChannel
ch = TerminalChannel(stdin=io.StringIO("default\n" * 5), use_rich=False)
preflight.resolve("disguised_missing_units.csv", target="outcome", mode="ask",
                  channel=ch, out_dir="out_terminal")
```
Output (first two of the five items):
```
========================================================================
PreFlight needs your input on 5 item(s)
File: /home/amg/Desktop/PreFlight/examples/disguised_missing_units.csv
========================================================================

1. How should column 'patient_id' be read?  [column: patient_id]
   Why: The values fit more than one reading closely enough that a wrong guess could change them.
   What we found: Best matches: identifiers fit 98% of the values; categories fit 95% of the values; plain text fit 20% of the values.
   Preview, real values from your file:
     now:  P0000 | P0001 | P0002
     Keep as plain text: P0000 | P0001 | P0002
     Keep identifier text: P0000 | P0001 | P0002
     Convert to categories: P0000 | P0001 | P0002
   1) Keep as plain text   <- recommended
   2) Keep identifier text
   3) Convert to categories

2. How should column 'weight' be read?  [column: weight]
   Why: Values include units (kg); turning them into numbers would lose the units.
   What we found: Best matches: categories fit 95% of the values; plain text fit 20% of the values.
   Preview, real values from your file:
     now:  69 kg | 88 kg | 61 kg
     Convert to whole numbers: (left blank) | (left blank) | (left blank)
     Convert to categories: 69 kg | 88 kg | 61 kg
     Keep as plain text: 69 kg | 88 kg | 61 kg
     ('(left blank)' means that value cannot be converted and is recorded as a problem value.)
   1) Convert to whole numbers   <- recommended
   2) Convert to categories
   3) Keep as plain text
...
```
You can answer with an option number, the exact option text, or `default`. Written:
`out_terminal/answers.json`, `out_terminal/plan.json`, `out_terminal/plan.pdf`.

**⚠ BUG (preview):** for `weight`, the preview says "Convert to whole numbers" leaves every value
blank. The real run strips the unit and gives `69.0`, `88.0`, ... (see step 7). The preview's
converter does not strip units the way the typing step does, so it can scare you away from the
correct option.

## 7. The review-file flow with `resolve()`

Useful on a server, or when someone else answers.
```python
try:
    preflight.resolve("disguised_missing_units.csv", target="outcome",
                      mode="ask", channel="review_file", out_dir="out_disguised")
except preflight.interact.channels.base.ReviewPending as e:
    print("ReviewPending:", e)
```
Output:
```
ReviewPending: Wrote 5 question(s) to /home/amg/Desktop/PreFlight/examples/out_disguised/review.json. Fill in the "answer" fields, then call resolve('/home/amg/Desktop/PreFlight/examples/out_disguised/review.json').
```
Open `review.json` and set every `"answer"` to one of that question's `"options"` strings.
For the demo, each was set to its `recommended_default`:
```
patient_id -> Keep as plain text | options: ['Keep as plain text', 'Keep identifier text', 'Convert to categories']
weight -> Convert to whole numbers | options: ['Convert to whole numbers', 'Convert to categories', 'Keep as plain text']
temp -> Keep as plain text | options: ['Keep as plain text', 'Convert to decimal numbers', 'Convert to categories']
smoker -> Keep as plain text | options: ['Keep as plain text', 'Convert to yes/no values', 'Convert to categories']
outcome -> Predict categories (classification) | options: ['Predict categories (classification)', 'Predict numerical values (regression)', 'No prediction goal (data inspection only)']
```
Then:
```python
plan, pdf = preflight.resolve("out_disguised/review.json")
print(pdf.name, sorted({d.status for d in plan.decisions}))
```
```
plan.pdf ['automatic', 'changed', 'confirmed']
```
The answers are now in `out_disguised/answers.json`. A later run with the same `out_dir` reuses
them, because the fingerprints match, so it does not stop:
```python
out = preflight.prepare("disguised_missing_units.csv", target="outcome", out_dir="out_disguised")
print(out["apply_result"].prepared.head(3).to_string())
```
```
PreFlight: output written to /home/amg/Desktop/PreFlight/examples/out_disguised
  patient_id  weight  temp smoker  outcome
0      p0000    69.0  39.0      y     sick
1      p0001    88.0  <NA>   None     sick
2      p0002    61.0  36.9   None  healthy
```
`69 kg` became `69.0`, `-999` became blank (`<NA>`), `?` became blank (`None`), and the text
was tidied to lower case. A blank answer, or one that is not an option, raises an error
naming the question. Defaults are never filled in for you.

## 8. Fit hygiene: clone the unfitted pipeline inside each CV fold

```python
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

out = preflight.prepare("clean.csv", target="bought", mode="defaults", out_dir="out_cv")
df = out["apply_result"].prepared
X, y = df.drop(columns=["bought"]), df["bought"].astype(str)   # pipeline input: no target column
scores = []
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    pipe = clone(out["pipeline"])                              # fresh and unfitted every fold
    Xtr = pipe.fit_transform(X.iloc[tr], y.iloc[tr])
    Xte = pipe.transform(X.iloc[te])
    m = RandomForestClassifier(random_state=0).fit(Xtr, y.iloc[tr])
    scores.append(accuracy_score(y.iloc[te], m.predict(Xte)))
print("columns out:", list(Xtr.columns))
print("fold accuracies:", [round(s, 2) for s in scores])
```
Output:
```
PreFlight: output written to /home/amg/Desktop/PreFlight/examples/out_cv
columns out: ['age', 'income', 'city']
fold accuracies: [0.33, 0.25, 0.42, 0.42, 0.42]
```
The scores are low because the demo data is random, and that is the honest result.
**Do not** score with `out["pipeline_fitted_on_all_rows"]`: it was fitted on every row, test
rows included, so any score from it is too optimistic. Use it only to transform *new* data
after you have chosen your model.

## 9. Reading the Report PDF (`report.pdf`)

1. **The plan that was applied**, with a pie chart of who decided what.
2. **What was planned, by stage.**
3. **What actually happened vs. what was planned**, including the steps you changed.
4. **Who decided what**: automatic / you confirmed / you changed, with times.
5. **Unusual entries found while reading your data** (at most 25 shown, with "+N more").
6. **Columns that may give away the answer** (leakage warnings).
7. **Technical details**: the only part that uses technical terms and raw evidence.

## 10. What is in the output folder

| File | Written by | What it is |
|---|---|---|
| `plan.json` | dry_run / resolve / prepare | every Decision, with status |
| `plan.pdf` | same | the plan, before running |
| `answers.json` | resolve (ask mode) | your answers, keyed by fingerprint |
| `review.json` | review-file channel | the questions to fill in (contains sample data values) |
| `cleaned_data.csv` | prepare | the prepared table |
| `schema.json` | prepare | the file's description and column types |
| `decisions.json` | prepare | the final Decisions |
| `pipeline.joblib` | prepare | the **unfitted** pipeline |
| `pipeline_fitted_on_all_rows.joblib` | prepare | for new data only, never for scoring |
| `report.pdf` | prepare | the after-run record (always written, and written last) |

The default folder is `./preflight_out/` (where you run Python). Passing the source file's own
folder as `out_dir` is refused.

## 11. strict vs defaults vs ask

| Mode | Open questions are... | Use when |
|---|---|---|
| `ask` | asked through a channel (terminal, notebook, review file, callback) | a person is available |
| `defaults` | settled with each safe default, logged as `automatic` | you accept the safe choices |
| `strict` | an error listing all of them | pipelines and CI: never guess |
| `auto` (default) | `ask` if a terminal or notebook is present, otherwise `strict` | normal use |

## 12. When something fails

| Symptom | What to do |
|---|---|
| `StrictResolutionError` | Use `mode="defaults"`, or the review file (step 7), or run in a terminal with `mode="ask"`. |
| `ReviewPending` | Not a failure: fill in `review.json` (or press Submit in the notebook), then call `resolve()` again. |
| `ReviewFileError: ... does not match the current questions` | The data or plan changed. Delete `review.json` and run again. |
| `InvalidAnswerError` / `MissingAnswerError` | An answer is blank or not one of the listed options. The message names the question. |
| `NonInteractiveError` | The terminal channel has no terminal. Use the review file or a callback. |
| `ReportGenerationError` | The PDF failed, but every other file was already saved. The message lists their paths. |
| Wrong separator / encoding / header | Answer the reading question, or check the reading Decisions in `plan.json`. |
| Warning "Ignoring stored answers" | `answers.json` was unreadable. You will be asked again, and the old file is kept as `.bak`. |
