# PreFlight

**Automated, auditable DataFrame preprocessing for ML workflows.**

PreFlight takes a raw pandas DataFrame and a target column, and returns three things:

- a **cleaned, feature-engineered DataFrame**
- a **reusable, fitted scikit-learn `Pipeline`**
- a **structured audit `Report`** of every automated decision — nothing happens silently

```python
import preflight as pf

result = pf.prepare(df, target="price", task="regression")

result.df           # cleaned + engineered DataFrame
result.pipeline      # fitted sklearn Pipeline — reusable on new data
result.report.show() # human-readable audit of every decision made
```

> Distribution name on PyPI is `pypreflight`. Import name and CLI command are both `preflight`.

---

## Install

```bash
pip install pypreflight
```

```python
import preflight as pf
```

Requires Python 3.9+.

---

## Why PreFlight

Most preprocessing code is either:
- hand-rolled per-project boilerplate that silently makes assumptions, or
- a black-box AutoML pipeline that decides things for you and doesn't explain why.

PreFlight sits in between: it automates the tedious parts (missing values, encoding, scaling, outliers, feature generation) while logging **every decision it makes and why**, so you can trust — and inspect — what happened to your data before it reaches a model.

It deliberately **never selects features for you and never trains a model**. It stops right before that line, on purpose.

---

## Quickstart

```python
import pandas as pd
import preflight as pf

df = pd.read_csv("train.csv")

result = pf.prepare(df, target="price", task="regression")

print(result.df.head())
result.report.show()                   # readable summary in the console
result.report.save_html("report.html")
result.report.save_pdf("report.pdf")   # graphic-first PDF report
```

To use the fitted pipeline on new data later:

```python
new_data = pd.read_csv("test.csv")
transformed = result.pipeline.transform(new_data)
```

---

## The 4-stage pipeline

```
Profiler → Cleaner → Engineer → Assembler
```

| Stage | What it does |
|---|---|
| **Profiler** | Infers each column's semantic type (numeric, categorical, datetime, text, boolean, constant, ID) and computes EDA signals — no data is changed here. |
| **Cleaner** | Handles missing values, outliers, and rare categories based on what the Profiler found. |
| **Engineer** | Encodes, scales, expands datetimes, and (optionally) generates new features. |
| **Assembler** | Builds the final `sklearn.Pipeline` and packages the `Report`. |

You can call each stage individually for inspection:

```python
pf.profile(df, target="price")   # inspect inferred types only
pf.clean(df, target="price")     # see cleaning decisions
pf.engineer(df, target="price")  # see encoding/scaling decisions
```

These return the same `PrepResult` shape as `prepare()`, but with `pipeline=None` — they're for inspection, not production use.

---

## Core API

### `pf.prepare(...)`

```python
pf.prepare(
    df,
    target: str,
    task: str = "classification",            # or "regression"
    model_hint: str = "tree",                # or "linear"
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
    feature_config: FeatureConfig | None = None,
    column_types: dict[str, SemanticType] | None = None,
    preset: str | None = None,               # "fast" or "thorough"
    dry_run: bool = False,
) -> PrepResult
```

**`model_hint`** controls encoding/scaling strategy:

| | `model_hint="tree"` | `model_hint="linear"` |
|---|---|---|
| Low-cardinality categorical | ordinal encoding | one-hot encoding |
| Scaling | none | `StandardScaler` |
| Skewed numerics | untouched | `log1p` transform |
| High-cardinality categorical | cross-fit (5-fold) target encoding in both cases | same |

### `pf.profile()` / `pf.clean()` / `pf.engineer()`

Same signature family as `prepare()` minus the arguments that don't apply to that stage. All return `pipeline=None`.

### `pf.compare(result_a, result_b)`

Diffs two `PrepResult` objects — shape, columns, report entry counts, and decision differences. Prints a summary and returns a dict.

### `pf.add_features(result, feature_config)`

Applies a `FeatureConfig` to an **already-prepared** `PrepResult`, without rerunning the Profiler or Cleaner:

```python
base = pf.prepare(df, target="price", task="regression")
enriched = pf.add_features(base, pf.FeatureConfig(interactions=True))
```

- Requires `result.pipeline` to not be `None` — i.e. must come from `prepare()`, not `profile()`/`clean()`/`engineer()`.
- Never mutates the input `result`.
- Can be called multiple times in sequence to stack different configs.
- Column name collisions are skipped with a warning, never silently overwritten.

---

## Feature engineering — `FeatureConfig`

All feature generation is **off by default**. Nothing changes in your output unless you explicitly turn it on.

```python
from preflight import FeatureConfig

config = FeatureConfig(
    interactions=True,
    interaction_top_k=5,
    interaction_types=["ratio", "product"],

    datetime_cyclical=True,
    datetime_deltas=True,
    datetime_reference_col="signup_date",

    clustering=True,
    cluster_k="auto",              # or an explicit int
    cluster_features="numeric_only",

    text_features=True,            # basic text stats: length, word count, has_text
    text_tfidf=True,               # adds a small TF-IDF-lite feature set
    text_tfidf_top_k=20,
)

result = pf.prepare(df, target="price", task="regression", feature_config=config)
```

| Category | What it generates |
|---|---|
| **Interactions** | Ratio/product columns between the top-K target-correlated numeric columns, zero-guarded. |
| **Datetime** | Cyclical sin/cos month & day-of-week, weekend flags, cross-column date deltas. |
| **Clustering** | KMeans cluster label + distance-to-centroid, with automatic or manual K. |
| **Text** | Character length, word count, a `has_text` flag, and (optionally) a capped TF-IDF-lite term set. This is intentionally basic — full NLP (embeddings, sentiment, language models) is out of scope. |

Every generated column gets a logged reason in the `Report` — nothing is added silently.

---

## Manual column type overrides

Auto-inference is good, but not perfect for every dataset. Force a column's type directly:

```python
from preflight import SemanticType

result = pf.prepare(
    df,
    target="price",
    column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
)
```

Overrides are applied **after** auto-inference and logged in the Report, stating both the original inferred type and the forced type. Overriding a nonexistent column or the target column itself raises a clear error before anything runs.

---

## Presets — less to configure

```python
pf.prepare(df, target="price", preset="fast")       # speed-favoring defaults
pf.prepare(df, target="price", preset="thorough")    # completeness-favoring defaults
```

Any parameter you also pass explicitly always overrides the preset's value for that parameter. Using a preset is logged in the Report so it's never a hidden configuration.

---

## Preview mode — `dry_run`

See what PreFlight *would* do, without touching your data or fitting a pipeline:

```python
preview = pf.prepare(df, target="price", dry_run=True)

preview.df        # your original DataFrame, completely untouched
preview.pipeline  # None
preview.report.show()  # the full, real decision log — same as a real run
```

Useful for trusting the library on a new dataset before committing to a real transform.

---

## The Report

```python
result.report.show()                       # console summary, grouped by stage
result.report.show(severity_filter="warning")
result.report.plot(kind="all")             # list of matplotlib Figures
result.report.to_html()                    # HTML string
result.report.save_html("report.html")
result.report.save_pdf("report.pdf")       # graphic-first PDF export
result.report.summary_counts()             # {'info': 132, 'warning': 14, 'critical': 0}
result.report.to_dataframe()
result.report.to_dict()
```

`.save_pdf()` produces a visual, chart-led PDF: a summary cover page, per-stage chart pages, and a compact appendix table last — built for skimming, not just archiving.

To compare two results visually:

```python
pf.save_compare_pdf(result_a, result_b, "comparison.pdf")
```

---

## CLI

```bash
preflight prepare train.csv --target price --task regression
```

Common flags:

```bash
preflight prepare train.csv --target price --task regression \
  --model-hint linear \
  --preset thorough \
  --column-type "zip_code:CATEGORICAL_LOW" \
  --text-features --text-tfidf --text-tfidf-top-k 15 \
  --interactions --interaction-top-k 5 \
  --clustering --cluster-k auto \
  --dry-run \
  --save-pdf report.pdf
```

Run `preflight prepare --help` for the full flag list.

---

## Semantic types

| Type | Meaning |
|---|---|
| `NUMERIC_FEATURE` | Ordinary numeric column |
| `NUMERIC_ID` | Numeric but identifier-like (excluded from most transforms) |
| `CATEGORICAL_LOW` | Low-cardinality categorical |
| `CATEGORICAL_HIGH` | High-cardinality categorical (cross-fit target encoded) |
| `DATETIME_NATIVE` | Already a datetime dtype |
| `DATETIME_STRING` | String that parses as a date |
| `BOOLEAN` | Two-valued column |
| `CONSTANT` | Single unique value (flagged, not useful) |
| `TEXT` | Free text — long, high-uniqueness strings |

---

## Scope

**In scope:** tabular data, supervised learning, CSV/DataFrame input, regression + classification, numeric/categorical/datetime/text features, optional interaction/datetime/cluster/text feature engineering.

**Explicitly out of scope:** full NLP (embeddings, transformers, sentiment analysis), time series, multi-label targets, image data, automated feature *selection*, SMOTE/resampling, model training or selection of any kind.

If your DataFrame has feature signals PreFlight surfaces (mutual information, correlation) — it will tell you about them in the Report, but it will never act on them by dropping or selecting features for you.

---

## Known limitations

These are documented, intentional boundaries of the current release — not bugs waiting to be fixed, but things worth knowing before you rely on them:

- **Complex unit-suffix parsing.** Simple numeric-with-unit strings (e.g. `"1248 CC"`, `"74 bhp"`) are automatically parsed into numeric columns. More complex, nested expressions (e.g. `"190Nm@ 2000rpm"`) are not — they'll fall back to being treated as a high-cardinality categorical. If you have columns like this, clean them with a quick regex before passing them in, or use `column_types` to tell PreFlight how to treat them.

- **Saved pipelines require `pypreflight` installed wherever they're loaded.** `result.pipeline` uses PreFlight's own custom transformers internally. If you save it with `joblib` and load it later in a different environment (e.g. a production server), that environment needs `pypreflight` installed too — it's not a fully standalone, dependency-free sklearn pipeline.

- **Columns mixing short labels and long free text can go either way.** If a string column contains a blend of short category-like values and longer descriptive text, it may get classified as `TEXT` and go through basic text feature generation instead of being treated as a category — which can blur category structure that would otherwise be useful. If you know a column should be treated as categorical regardless of string length, use `column_types` to force it.

---

## FAQ

**Does this train a model?**
No. The `Pipeline` returned has `.fit()` and `.transform()`, never `.predict()`. Model training is intentionally out of scope.

**Will it silently drop my columns?**
Only above `drop_threshold` missingness, and every drop is logged in the Report with a reason.

**Can I use the pipeline on new data later?**
Yes — `result.pipeline.transform(new_data)` reproduces the exact same transformation, including any fitted feature generators (frozen KMeans centroids, learned TF-IDF vocabulary, etc.).

**What if I disagree with an auto-inferred column type?**
Use `column_types` to override it directly — see above.

**Is text handling here "AI-powered"?**
No. Text support is intentionally basic (length, word count, TF-IDF-lite) — a lightweight signal boost, not an NLP pipeline.

---

## License

MIT