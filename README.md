<div align="center">

# PreFlight

**Turn a messy CSV into model-ready data, with every choice written down.**

[![PyPI](https://img.shields.io/pypi/v/pypreflight?include_prereleases&label=PyPI)](https://pypi.org/project/pypreflight/)
[![Python](https://img.shields.io/pypi/pyversions/pypreflight)](https://pypi.org/project/pypreflight/)
[![CI](https://github.com/VAIyerAmogha/PreFlight/actions/workflows/ci.yml/badge.svg)](https://github.com/VAIyerAmogha/PreFlight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/VAIyerAmogha/PreFlight/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational)](https://github.com/VAIyerAmogha/PreFlight/tree/main/typings)

[Installation](#installation) ·
[Quickstart](#quickstart) ·
[How it works](#how-it-works) ·
[API](#api-reference) ·
[Accuracy](#accuracy) ·
[Limitations](#known-limitations) ·
[Contributing](#contributing)

</div>

---

PreFlight is a Python library that prepares tabular data for machine learning. You give it the
path to a CSV file and the name of your target column. It works out how the file is encoded,
what separates the cells and where the header is. It then infers a type for every column,
plans the cleaning and feature engineering, and gives you back:

- a typed, cleaned **pandas DataFrame**;
- an **unfitted scikit-learn `Pipeline`** holding every step that learns from data, ready to be
  cloned and fitted inside each cross-validation fold;
- a **Plan PDF** before anything runs and a **Report PDF** after, both readable by someone who
  has never opened the code;
- **JSON records** of the schema, every decision and every answer a person gave.

Each choice PreFlight makes is a `Decision` carrying its evidence, a confidence score, a safe
default and a one-sentence explanation. When the evidence is too close to call, PreFlight
leaves the data as text and asks a person. It does not guess, and it never modifies the source file.

> **Status:** PreFlight 2.0 is in **alpha** (`2.0.0.dev0`). The public API listed below is stable in shape,
> but details may change before 2.0.0. Please [open an issue](https://github.com/VAIyerAmogha/PreFlight/issues)
> if something surprises you.

> [!WARNING]
> **PreFlight can make mistakes.** It infers types, file formats and cleaning steps from the data,
> and inference can be wrong. Review the Plan PDF and `decisions.json` before you rely on the output,
> especially for high-stakes work. See [Known limitations](#known-limitations).

## Why PreFlight

`pandas.read_csv(path)` assumes UTF-8, commas and a header on the first line. Real files break
those assumptions all the time: European exports separated by semicolons, Latin-1 text,
headerless dumps, title rows above the table, `-999` or `?` standing in for "missing",
`69 kg` in a numeric column. Most preprocessing tools either fail on these files or, worse,
handle them silently in a way nobody checks.

PreFlight is built on five rules:

| Rule | What it means in practice |
|---|---|
| **Everything is a Decision** | Encoding, separator, header row, every column type, every fill value and encoder is a record with evidence and confidence. |
| **Plan before apply** | `plan()` only reads and proposes. Nothing changes until `apply()`, so a dry run and a Plan PDF come for free. |
| **Abstain, don't guess** | If two readings of a column score too closely, or a conversion would lose information, the column stays as text and becomes a question. |
| **Ask once, never hang** | All questions are batched into one review. With no person available, the run stops with a clear list instead of waiting. Answers are stored by fingerprint and reused. |
| **Fit hygiene** | Steps that learn nothing run on the table. Steps that learn from data (medians, encoders, clipping limits) go into an unfitted pipeline you fit per fold, so CV scores stay honest. |

## Installation

```bash
pip install --pre pypreflight
```

The distribution is named `pypreflight`. The import name is `preflight`.

Optional extras:

```bash
pip install --pre "pypreflight[interactive]"   # rich-formatted terminal questions
pip install --pre "pypreflight[notebook]"      # ipywidgets questions inside Jupyter
```

PreFlight supports Python **3.10, 3.11 and 3.12**. It builds on pandas, NumPy, SciPy,
scikit-learn, charset-normalizer and ReportLab.

## Quickstart

### One call

```python
import preflight

out = preflight.prepare("patients.csv", target="outcome")
```

In a terminal or notebook, PreFlight shows any open questions on one screen and waits for your
answers. In a script, CI job or other non-interactive session, it stops and lists the questions
instead of hanging (see [Resolution modes](#resolution-modes)).

`out` is a dictionary:

| Key | Contents |
|---|---|
| `apply_result` | `ApplyResult` with `.data` (typed table), `.prepared` (typed and cleaned), `.pipeline`, `.decisions`, `.anomalies` |
| `pipeline` | **Unfitted** scikit-learn `Pipeline`. Fit it on training rows only. |
| `pipeline_fitted_on_all_rows` | The same pipeline fitted on every row, for transforming new data once model selection is done. Not for evaluation. |
| `plan` | The final `Plan` |
| `paths` | Paths of every file written |

All files are written to `./preflight_out/` unless you pass `out_dir`:

```
preflight_out/
├── plan.json                           # the full plan, machine-readable
├── plan.pdf                            # what PreFlight intends to do, and what it asked
├── cleaned_data.csv                    # the prepared table
├── schema.json                         # encoding, dialect, header, per-column types and anomalies
├── decisions.json                      # every final Decision with evidence and who decided it
├── pipeline.joblib                     # unfitted pipeline
├── pipeline_fitted_on_all_rows.joblib
└── report.pdf                          # what was done, by whom, and what looked unusual
```

### Look before you leap

`plan()` is pure. It reads the file, proposes, and writes nothing.

```python
import preflight

p = preflight.plan("examples/disguised_missing_units.csv", target="outcome")

for d in p.decisions:
    flag = "ASK" if d.needs_attention else ""
    print(f"{d.stage:<9}{d.kind:<22}{str(d.columns):<15}{d.action:<26}{d.confidence:.2f} {flag}")
```

```
reading  encoding              []             use_encoding_utf-8        1.00
reading  dialect               []             separate_by_,             0.95
reading  header                []             use_header_row_0          0.85
infer    type_inference        ['patient_id'] keep_as_id                0.98 ASK
infer    type_inference        ['weight']     cast_to_int               0.90 ASK
infer    type_inference        ['temp']       cast_to_float             1.00 ASK
infer    type_inference        ['smoker']     cast_to_boolean           1.00 ASK
infer    type_inference        ['outcome']    keep_as_categorical       0.95
infer    ambiguous_task        ['outcome']    classification            0.50 ASK
clean    normalize_categories  ['patient_id'] normalize_categories      1.00
clean    add_missing_indicator ['weight']     add_missing_indicator     1.00
clean    fill_median           ['weight']     fill_median               0.95
clean    add_missing_indicator ['temp']       add_missing_indicator     1.00
clean    fill_median           ['temp']       fill_median               0.95
clean    clip_outliers         ['weight']     clip_outliers             0.95
clean    clip_outliers         ['temp']       clip_outliers             0.95
engineer target_encode         ['patient_id'] target_encode_cross_fit   0.95
```

This file contains four traps, and each one is caught and explained:

- `weight` holds values like `69 kg` and some `N/A`. The Decision reads: *"Column 'weight' holds
  numbers with the unit 'kg'. Read as numbers; the unit 'kg' is kept in this record."* It asks
  first, because stripping a unit loses information.
- `temp` uses `-999` for "no reading". PreFlight flags it because every other value is at least
  36.1, not because `-999` is on a list.
- `smoker` is a `Y`/`N` column where `?` means unknown.
- `patient_id` scores almost equally as an identifier and as a category, so it is marked for review.

`preflight.dry_run()` does the same and also resolves the questions and writes `plan.json`
and `plan.pdf`, without producing any cleaned data.

### Files `pandas.read_csv` gets wrong

```python
p = preflight.plan("examples/messy_noheader_semicolon.csv")   # 5.5;2.9;2.2;versicolor ...

for d in p.decisions[:3]:
    print(d.action, d.confidence, "|", d.plain_language)
```

```
use_encoding_utf-8 1.0 | Read the file using standard text format.
separate_by_; 0.9 | Separate columns using semicolon separators.
no_header 0.98 | No column names row found; assign automatic column names.
```

`pandas.read_csv` returns a single column named `5.5;2.9;2.2;versicolor` for this file. PreFlight
splits it into `col_0` … `col_3` and types them as float, float, float and categorical.

### Cross-validation without leakage

The returned pipeline is unfitted and works with `sklearn.base.clone`, so every statistic
(fill values, clipping limits, target encodings) is learned from the training fold alone:

```python
from sklearn.base import clone
from sklearn.model_selection import KFold

df = out["apply_result"].prepared
X, y = df.drop(columns=["outcome"]), df["outcome"]

for train, test in KFold(5, shuffle=True, random_state=0).split(X):
    pipe = clone(out["pipeline"])
    X_train = pipe.fit_transform(X.iloc[train], y.iloc[train])
    X_test = pipe.transform(X.iloc[test])
    # fit and score your model here
```

Which steps run on the table and which go into the pipeline is documented step by step in
[`docs/fit_hygiene.md`](https://github.com/VAIyerAmogha/PreFlight/blob/main/docs/fit_hygiene.md).

## How it works

```
 your_file.csv  (never modified)
      │
      ▼
 1. READ     encoding → separator → header row            → raw table, all text
      ▼
 2. TYPE     per column: find disguised missing values,    → TableSchema
             score candidate types, pick a clear winner
             or abstain (keep as text and ask)
      ▼
 3. PLAN     simulated on a private copy:                  → Plan (ordered Decisions)
             markers → units → duplicates → column drops      plan.json · plan.pdf
             → text tidying → fills → clipping → leakage
             → encoding
      ▼
 4. RESOLVE  open questions batched into one review;
             stored answers reused by fingerprint
      ▼
 5. APPLY    learn-nothing steps        → prepared DataFrame
             learn-from-data steps      → unfitted sklearn Pipeline
      ▼
 report.pdf · decisions.json · schema.json · pipeline.joblib
```

### Decisions

Every step above produces `Decision` objects. The fields you will use most:

| Field | Meaning |
|---|---|
| `stage`, `kind` | Where it came from (`reading`, `infer`, `clean`, `engineer`) and what it is |
| `columns` | Columns it applies to |
| `action` | What will be done, e.g. `cast_to_float`, `fill_median` |
| `evidence` | The measurements behind it: scores, counts, learned values |
| `confidence` | 0 to 1 |
| `default` | The safe choice used when nobody answers |
| `needs_attention` | `True` if a person should confirm it |
| `plain_language` | One sentence a non-specialist can read |
| `status`, `history` | Who decided it (automatic, default or user) and how it changed |

### Resolution modes

Pass `mode=` to `prepare()`, `dry_run()`, `resolve()` or `apply()`:

| Mode | Behaviour |
|---|---|
| `"auto"` *(default)* | `"ask"` when the session is interactive (terminal or notebook), `"strict"` otherwise |
| `"ask"` | Show all open questions through a channel and use the answers |
| `"defaults"` | Accept the safe default for every question. Uncertain types stay as text. |
| `"strict"` | Raise `StrictResolutionError` listing every open question. Never waits. |

A strict run on the example file stops like this:

```
Strict mode: 5 unresolved item(s), and this session cannot ask (no terminal or notebook).
To proceed: rerun with mode='defaults' to accept the safe default for each item; or write a
review file with mode='ask', channel='review_file', fill in its answers, and run again (stored
answers are reused); or run interactively with mode='ask'.
  - id: dec_infer_patient_id_1e3f2400 | column: patient_id | How should column 'patient_id' be read?
  - id: dec_infer_weight_b4b5b4ac | column: weight | How should column 'weight' be read?
  ...
```

> **Tip:** `"defaults"` is deliberately cautious: the safe default for an uncertain column type is to keep
> the original text, because keeping text never destroys information. For the best result on a
> new dataset, answer the questions once. The answers are stored and reused on later runs.

### Channels

With `mode="ask"`, the `channel=` argument decides how questions reach a person:

| Channel | Use it for |
|---|---|
| `"terminal"` | Interactive prompt. Formatted with `rich` if the `interactive` extra is installed. |
| `"notebook"` | ipywidgets form inside Jupyter (`notebook` extra) |
| `"review_file"` | Asynchronous review. Writes `review.json`, raises `ReviewPending`, and continues when you call `preflight.resolve("…/review.json")` after filling in the answers. |
| a callable | Your own UI or automation. Receives the list of `Question` objects and returns `{question_id: option}`. |

Each question carries its text, an explanation, a summary of the evidence, the options, the
recommended default and a preview of real values before and after each option. A callable
channel looks like this:

```python
def answer(questions):
    choices = {}
    for q in questions:
        print(q.column, q.text, q.options)
        choices[q.question_id] = q.recommended_default
    return choices

out = preflight.prepare("patients.csv", target="outcome", mode="ask", channel=answer)
```

## API reference

Everything below is importable from the top-level `preflight` package.

| Function | Purpose | Writes files |
|---|---|---|
| `plan(path, target=None, task=None, model_hint="tree", feature_config=None)` | Read, infer and propose. Returns a `Plan`. | No |
| `dry_run(path, target=None, task=None, out_dir=None, mode="auto", channel=None, ...)` | `plan` plus question resolution. Returns `(Plan, plan_pdf_path)`. | `plan.json`, `plan.pdf`, `answers.json` |
| `resolve(plan_or_path, mode=None, channel=None, out_dir=None, target=None, task=None)` | Answer every open question. Accepts a `Plan`, a data file, or a filled-in `review.json`. | `plan.json`, `plan.pdf`, `answers.json`, maybe `review.json` |
| `apply(plan, decisions=None, mode=None, channel=None, out_dir=None)` | Run the plan. Returns an `ApplyResult`. | Only when `mode` is given (resolved plan) |
| `prepare(path, target, task=None, mode="auto", channel=None, out_dir=None, model_hint="tree", feature_config=None)` | Everything: plan → resolve → apply → save → Report PDF | All outputs |

Classes: `Plan` (`to_json`, `save_json`, `from_json`, `from_dict`), `Decision`, `TableSchema`.

Common arguments:

- **`task`**: `"classification"`, `"regression"` or `None` to infer it from the target.
- **`model_hint`**: `"tree"` or `"linear"`. Shapes the engineering step: `"tree"` ordinal-encodes
  low-variety categories, `"linear"` one-hot encodes them and standard-scales numeric features.
- **`out_dir`**: defaults to `./preflight_out/`. PreFlight never writes next to the source file,
  which may be read-only or shared.

## Accuracy

PreFlight is measured against a corpus of **254 real public CSV files** (Rdatasets, TidyTuesday,
FiveThirtyEight, UCI, seaborn, pandas test fixtures and the Frictionless messy-CSV tests) with
**2,850 hand-labelled columns**. The labels and their sources live in
[`corpus/labels/`](https://github.com/VAIyerAmogha/PreFlight/tree/main/corpus/labels).

| Metric | v1.0.1 | v2 |
|---|---:|---:|
| Files that crash | 10 / 254 | **0 / 254** |
| Column type exactly right | 80.77% | **91.23%** |
| Column type right *or* flagged for review | n/a | **99.54%** |
| Wrong type with no flag (silent failures) | 9.40% | **0.46%** |
| File format right (encoding + separator + header) | 91.73% | 91.73% |

In plain terms, fewer than 1 column in 100 gets a wrong type without a warning.

These numbers come with caveats. The held-out part of the corpus was examined during
development, so it is not a fully independent test. Most remaining type misses are row-number
columns read as integers, and nearly all of those are flagged for review.

## Known limitations

**PreFlight can make mistakes.** Every type, format and cleaning choice is inferred from the data
by heuristics, not known for certain, and some of them will be wrong for your file:

- On the benchmark corpus, about **1 column in 11** gets the wrong type (most are flagged for review,
  but **0.46% are wrong with no flag**), and the file format is misread for about **1 file in 12**.
- Your data may look nothing like the corpus, so your error rate may be higher.
- A confident decision is not a guaranteed one. Confidence scores measure how clearly the evidence
  points one way, not whether the answer is right.
- Recommended answers to questions are a safe default, not domain knowledge. Only you know what a
  column means.

Treat the output as a well-documented first draft. Read the Plan PDF before running `prepare()`,
check the Report PDF and the cleaned data afterwards, and override any decision you disagree with.
PreFlight is provided as is, without warranty (see the [license](https://github.com/VAIyerAmogha/PreFlight/blob/main/LICENSE)).

Specific known gaps:

- **CSV-like files only.** No Excel, JSON or Parquet, and no DataFrame input in 2.0.
- **No command-line interface yet.** Use the Python API.
- **The leakage check is numeric-only.** A text or yes/no copy of the target is not detected.
- **Text identifier columns** with many unique values are target-encoded rather than dropped.
- **Dialect detection** tries four separators and looks at the first 128 KB of the file.
- **Type conversion** runs per value in Python, so very large files are slower than they need to be.
- **Numeric columns that are 30–60% blank** keep their blanks. Add an imputer if your model cannot
  handle missing values.

Bug reports and pull requests for any of these are welcome.

## Contributing

```bash
git clone https://github.com/VAIyerAmogha/PreFlight.git
cd PreFlight
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,interactive,notebook]"
```

Run the checks CI runs:

```bash
pytest -q                     # fast suite: unit, property-based and regression tests
ruff check src/ tests/        # lint
mypy --strict src/preflight   # type check
nox                           # all of the above on Python 3.10, 3.11 and 3.12
```

The corpus suites run every test against all 254 files. They are slow and deselected by
default. The raw files are not stored in the repository, so download them first:

```bash
python scripts/acquire_corpus.py
pytest -m corpus -n 4 --dist loadgroup
```

Repository layout:

```
src/preflight/
├── ingest/      encoding, dialect and header detection
├── infer/       per-column type inference and disguised-missing detection
├── clean/       markers, duplicates, drops, normalisation, imputation, outliers
├── engineer/    encoding, scaling, datetime and numeric features, VIF
├── interact/    questions, modes, channels and the answer store
├── report/      Plan PDF and Report PDF
├── plan.py      the orchestrator: plan(), apply(), dry_run()
├── prepare.py   the one-call entry point
└── assemble.py  builds the scikit-learn pipeline
tests/           unit/ · property/ · regression/ · corpus/
corpus/labels/   ground-truth labels for the 254-file corpus
examples/        small CSVs used in this README
```

Before opening a pull request, please make sure the fast suite, ruff and mypy all pass, and add
a test for any behaviour you change. Changes to type inference or ingestion should also be checked
against the corpus suite.

## License

PreFlight is released under the [MIT License](https://github.com/VAIyerAmogha/PreFlight/blob/main/LICENSE).
All runtime dependencies are under permissive licences.
