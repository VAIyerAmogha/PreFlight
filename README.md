# PreFlight

**Automated, auditable DataFrame preprocessing for ML workflows.**

PreFlight takes a raw pandas DataFrame and a target column, and returns a cleaned dataset, a reusable `sklearn` Pipeline, and a structured report explaining every decision it made along the way. Nothing happens silently.

```bash
pip install pypreflight
```

> **Note:** the PyPI distribution is named `pypreflight`, but you still `import preflight` and use the `preflight` CLI command.

- **GitHub**: [github.com/VAIyerAmogha/PreFlight](https://github.com/VAIyerAmogha/PreFlight)
- **PyPI**: [pypi.org/project/pypreflight](https://pypi.org/project/pypreflight/)
- **License**: MIT
- **Python**: 3.9+

---

## Table of Contents

- [Why PreFlight](#why-preflight)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Core API](#core-api)
  - [`prepare()`](#prepare)
  - [`profile()` / `clean()` / `engineer()`](#profile--clean--engineer)
  - [`compare()`](#compare)
  - [`add_features()`](#add_features)
- [Feature engineering with `FeatureConfig`](#feature-engineering-with-featureconfig)
- [The Report](#the-report)
- [Command line interface](#command-line-interface)
- [Semantic types](#semantic-types)
- [`model_hint`: tree vs. linear](#model_hint-tree-vs-linear)
- [Scope](#scope)
- [FAQ](#faq)

---

## Why PreFlight

Most preprocessing code is either a pile of one-off `pandas` snippets that don't survive contact with new data, or a black-box pipeline that makes decisions you can't see or explain. PreFlight aims to sit in between:

- **One function call** — `pf.prepare(df, target="price")` handles missing values, outliers, encoding, scaling, and datetime features.
- **A reusable Pipeline** — every transformation is captured in a fitted `sklearn.Pipeline`, so you can call `.transform()` on new data (a test set, production data) and get identical results.
- **A full audit trail** — every automated decision (a column dropped, a value imputed, an encoding chosen) is logged in a structured `Report`, with a plain-English reason.
- **No silent behavior** — if something happens to your data, you can always find out why.

PreFlight never trains or selects a model. It stops right before that step, handing you a clean `DataFrame` and pipeline ready for whatever model you choose.

---

## Quickstart

```python
import pandas as pd
import preflight as pf

df = pd.read_csv("train.csv")

result = pf.prepare(df, target="price", task="regression", model_hint="linear")

result.df          # cleaned, feature-engineered DataFrame
result.pipeline     # fitted sklearn Pipeline — reusable on new data
result.report.show()  # human-readable audit log of every decision made
```

Apply the same pipeline to new data later:

```python
new_data = pd.read_csv("test.csv")
transformed = result.pipeline.transform(new_data)
```

---

## How it works

PreFlight runs your data through four internal stages:

```
Profiler  →  Cleaner  →  Engineer  →  Assembler
```

| Stage | What it does |
|---|---|
| **Profiler** | Infers each column's semantic type (numeric, categorical, datetime, etc.) and computes diagnostic signals — missingness, outlier rate, cardinality, correlation/mutual information with the target, VIF, leakage suspicion. Never modifies data. |
| **Cleaner** | Imputes missing values, handles outliers, drops unusable columns (constants, IDs, duplicates), groups rare categories. |
| **Engineer** | Encodes categoricals, scales/transforms numerics, expands datetime columns, and (optionally) generates new engineered features — see [`FeatureConfig`](#feature-engineering-with-featureconfig). |
| **Assembler** | Packages everything into a fitted `sklearn.Pipeline` and a `PrepResult` (`df`, `pipeline`, `report`). |

Every stage writes to the same `Report`, so the full history of what happened to your data is available in one place.

---

## Core API

### `prepare()`

The main entry point. Runs all four stages and returns a fitted pipeline.

```python
pf.prepare(
    df,
    target,                      # str: name of the target column
    task="classification",       # "classification" | "regression"
    model_hint="tree",            # "tree" | "linear"
    drop_threshold=0.5,           # missingness fraction above which a column is dropped
    outlier_method="winsorize",   # outlier handling strategy
    cardinality_threshold=20,     # categorical cardinality cutoff (low vs high)
    feature_config=None,          # optional FeatureConfig — see below
) -> PrepResult
```

Returns a `PrepResult` with:
- `result.df` — cleaned, engineered DataFrame
- `result.pipeline` — fitted, reusable `sklearn.Pipeline`
- `result.report` — structured audit log (a `Report` object)

### `profile()` / `clean()` / `engineer()`

Stateless, single-stage versions for inspecting intermediate output. Each returns a `PrepResult` with `pipeline=None` — useful for debugging or understanding what a single stage would do without committing to the full pipeline.

```python
pf.profile(df, target, task="classification", cardinality_threshold=20)
pf.clean(df, target, task="classification", drop_threshold=0.5, outlier_method="winsorize")
pf.engineer(df, target, task="classification", model_hint="tree", feature_config=None)
```

### `compare()`

Diffs two `PrepResult` objects — useful for comparing different settings (e.g. `model_hint="tree"` vs `"linear"`, or two `FeatureConfig`s).

```python
result_a = pf.prepare(df, target="price", model_hint="tree")
result_b = pf.prepare(df, target="price", model_hint="linear")

diff = pf.compare(result_a, result_b)
# prints a summary: shape differences, column differences, report entry counts
```

### `add_features()`

Apply new `FeatureConfig` options to a result you've **already prepared**, without rerunning Profiler/Cleaner from scratch.

```python
from preflight.types import FeatureConfig

result = pf.prepare(df, target="price")

# later, without starting over:
config = FeatureConfig(clustering=True, cluster_k=4)
result2 = pf.add_features(result, config)
```

`result2` is a brand-new `PrepResult` — the original `result` is left untouched. `result2.pipeline` includes the new step, so `result2.pipeline.transform(new_data)` reproduces the same engineered features on new data.

> `add_features()` requires a `PrepResult` from `prepare()` (i.e. one with a fitted pipeline) — it will raise a clear error if given output from `profile()`, `clean()`, or `engineer()`.

---

## Feature engineering with `FeatureConfig`

By default, PreFlight's Engineer stage only encodes, scales, and expands datetime columns — nothing more. `FeatureConfig` lets you opt in to additional, automatically-generated features. **Everything is off by default** — passing no `FeatureConfig` produces identical output to earlier versions.

```python
from preflight.types import FeatureConfig

config = FeatureConfig(
    interactions=True,
    interaction_top_k=5,
    interaction_types=["ratio", "product"],

    datetime_cyclical=True,
    datetime_deltas=True,
    datetime_reference_col=None,

    clustering=True,
    cluster_k="auto",
    cluster_features="numeric_only",
)

result = pf.prepare(df, target="price", model_hint="linear", feature_config=config)
```

| Option | Type | Default | What it does |
|---|---|---|---|
| `interactions` | `bool` | `False` | Generate ratio/product/difference features between your most target-relevant numeric columns. |
| `interaction_top_k` | `int` | `5` | How many top numeric columns (by correlation/mutual information with target) are considered as candidates. |
| `interaction_types` | `list[str]` | `["ratio", "product"]` | Which combinations to generate: `"ratio"`, `"product"`, `"difference"`. |
| `datetime_cyclical` | `bool` | `False` | Adds `sin`/`cos` encodings of month and day-of-week, plus `is_weekend`. |
| `datetime_deltas` | `bool` | `False` | Adds day-difference features between pairs of datetime columns. |
| `datetime_reference_col` | `str \| None` | `None` | If set, adds "days since this column" for every other datetime column. |
| `clustering` | `bool` | `False` | Adds a `cluster_label` and `cluster_dist_to_centroid` column via KMeans. |
| `cluster_k` | `int \| "auto"` | `"auto"` | Number of clusters, or `"auto"` to pick automatically via silhouette score. |
| `cluster_features` | `"numeric_only" \| list[str]` | `"numeric_only"` | Which columns to cluster on. |

Every generated column is logged in the `Report` with a plain-English reason — call `result.report.show()` to see exactly what was created and why.

---

## The Report

`result.report` is a first-class part of every `PrepResult` — it's how PreFlight stays auditable instead of a black box.

```python
result.report.show()             # grouped, readable terminal summary
result.report.show(verbose=True) # show every entry, no truncation
result.report.to_dict()          # structured dict export
result.report.to_dataframe()     # entries as a DataFrame (stage, column, action, rationale, severity)
result.report.plot()             # correlation heatmap, missingness heatmap, MI bar chart, class distribution
result.report.to_html("report.html")  # self-contained HTML export, no external references
```

`.show()` groups entries by stage and marks severity with symbols — `⚠` for warnings, `✕` for critical issues, `·` for informational entries. By default, long lists of informational entries are summarized; pass `verbose=True` to see everything.

---

## Command line interface

```bash
preflight prepare train.csv --target price --task regression --model-hint linear
```

This writes three files next to your input: `train_prepared.csv`, `train_pipeline.joblib`, and `train_report.json`.

**Common options:**

| Flag | Description |
|---|---|
| `--target` | Name of the target column (required) |
| `--task` | `classification` or `regression` |
| `--model-hint` | `tree` or `linear` |
| `--drop-threshold` | Missingness fraction above which a column is dropped |
| `--outlier-method` | Outlier handling strategy |
| `--cardinality-threshold` | Cutoff between low- and high-cardinality categoricals |
| `--verbose` | Print the full report to the console |

**Feature engineering flags (v0.2.0+):**

| Flag | Description |
|---|---|
| `--interactions` / `--no-interactions` | Enable numeric interaction features |
| `--interaction-top-k` | Number of top columns considered for interactions |
| `--interaction-types` | Comma-separated: `ratio,product,difference` |
| `--datetime-cyclical` / `--no-datetime-cyclical` | Enable cyclical datetime features |
| `--datetime-deltas` / `--no-datetime-deltas` | Enable cross-column date deltas |
| `--datetime-reference-col` | Column to compute "days since" against |
| `--clustering` / `--no-clustering` | Enable cluster-based features |
| `--cluster-k` | `auto` or an integer |
| `--cluster-features` | Comma-separated column list, or `numeric_only` |

Example with feature engineering enabled:

```bash
preflight prepare train.csv --target price --task regression \
  --interactions --interaction-top-k 5 \
  --clustering --cluster-k auto
```

---

## Semantic types

The Profiler classifies every column into one of eight semantic types, which determines how Cleaner and Engineer treat it:

| Type | Meaning |
|---|---|
| `NUMERIC_FEATURE` | A regular numeric column |
| `NUMERIC_ID` | A numeric column that looks like an identifier (dropped by Cleaner) |
| `CATEGORICAL_LOW` | A categorical column below the cardinality threshold |
| `CATEGORICAL_HIGH` | A categorical column above the cardinality threshold (always cross-fit target encoded) |
| `DATETIME_NATIVE` | Already a proper datetime dtype |
| `DATETIME_STRING` | A string column that parses as a date |
| `BOOLEAN` | A true/false column |
| `CONSTANT` | A column with only one unique value (dropped by Cleaner) |

---

## `model_hint`: tree vs. linear

`model_hint` tells Engineer which family of downstream model you plan to use, since tree-based and linear models want different preprocessing:

| | `model_hint="tree"` | `model_hint="linear"` |
|---|---|---|
| Low-cardinality categorical | Ordinal encoding | One-hot encoding |
| Numeric scaling | None | `StandardScaler` |
| Skewed numeric columns | Untouched | `log1p` transform |
| High-cardinality categorical | Cross-fit (5-fold) target encoding | Cross-fit (5-fold) target encoding |

High-cardinality categoricals are **always** cross-fit target encoded regardless of `model_hint`, since naive target encoding would leak the target into the features.

---

## Scope

**In scope (v0.1–v0.2):** tabular data, supervised learning, CSV/DataFrame input, regression and classification, numeric/categorical/datetime features, optional interaction/datetime/cluster feature engineering.

**Out of scope (for now):** text columns, time series, multi-label targets, image data, automated feature *selection* (mutual information scores are surfaced in the Report, never used to drop columns), and resampling techniques like SMOTE.

---

## FAQ

**Does PreFlight train models?**
No. `result.pipeline` has `.fit()` and `.transform()`, but never `.predict()`. Model training and selection is intentionally left to you.

**What happens if I mix up `task` and my data?**
PreFlight checks this before doing any work. If you pass `task="classification"` on data that looks continuous, you'll get a clear error suggesting `task="regression"` instead of a confusing crash. The reverse case (regression on a low-cardinality target) produces a warning in the Report rather than an error, since that's sometimes legitimate.

**Will using `FeatureConfig` change my existing results?**
No. Every `FeatureConfig` option defaults to off. If you don't pass `feature_config`, behavior is identical to earlier versions.

**Can I see exactly why a column was dropped or transformed?**
Yes — call `result.report.show()`. Every automated decision has a logged rationale.

---

## License

MIT — see [LICENSE](https://github.com/VAIyerAmogha/PreFlight/blob/main/LICENSE).