# PreFlight-ML

A pip-installable Python library that takes a raw pandas DataFrame and returns a cleaned DataFrame, a reusable sklearn Pipeline, and a structured Report — automatically.

![PyPI version placeholder] ![License placeholder]

## Installation

```bash
# Coming soon to PyPI!
# pip install preflight-ml

# For now, install from source:
git clone https://github.com/VAIyerAmogha/PreFlight.git
cd PreFlight
pip install .
```

## Quickstart

```python
import preflight as pf
import pandas as pd

# Load your raw, messy dataset
df = pd.read_csv("data.csv")

# Run the full preparation pipeline
result = pf.prepare(
    df=df,
    target="price",
    task="regression",
    model_hint="tree"
)

# 1. Inspect the fully cleaned and engineered dataset
print(result.df.head())

# 2. Review the automated decisions report
result.report.show()

# 3. Use the scikit-learn Pipeline on new data
pipeline = result.pipeline
new_data = pd.read_csv("new_data.csv")
# predictions = my_model.predict(pipeline.transform(new_data))
```

## Features

PreFlight-ML eliminates mechanical data preparation work without creating a black box. Every transform is explainable and reproducible.

### Profiler
- **Semantic Type Inference**: Automatically infers 8 semantic types (e.g. `NUMERIC_FEATURE`, `CATEGORICAL_HIGH`, `DATETIME_NATIVE`).
- **Signal Extraction**: Calculates missingness rates, outlier prevalence, cardinality, correlation, mutual information, class imbalance, and leakage flags.
- **VIF & Multicollinearity**: Detects collinear features to prevent mathematical instability.

### Cleaner
- **Imputation**: Intelligent median, mode, and constant imputation with automatic missing indicators.
- **Structural Remediation**: Drops high-missingness and numeric ID columns, removes duplicate rows, and coerces string dates.
- **Value-Level Fixing**: Winsorizes outliers, normalizes category strings, and groups rare categories.

### Engineer
- **Encoding Strategies**: Applies ordinal encoding, one-hot encoding, and 5-fold cross-fit target encoding (to prevent target leakage).
- **Scaling & Transformations**: Applies `StandardScaler` and `log1p` transforms where mathematically safe.
- **Datetime Expansion**: Automatically extracts features from date columns (year, month, day, day of week).

### Report
- **Transparent Logging**: Every automated decision is logged with its rationale and severity.
- **Visualizations**: Generates EDA charts using `result.report.plot()`.
- **Export Options**: Export the report to terminal, DataFrame, JSON, or embedded HTML.

## CLI Usage

PreFlight-ML can be used directly from the command line:

```bash
preflight prepare data.csv --target price --task regression --model-hint tree
```
This will generate:
- `data_prepared.csv`
- `data_pipeline.joblib`
- `data_report.json`

## Scope and Boundaries

What is in scope (v0.1.0):
- Fully automated data cleaning and feature engineering.
- Explainable preprocessing logs.
- Generation of an exportable scikit-learn `Pipeline`.

Explicitly **OUT OF SCOPE**:
- Model training, hyperparameter tuning, or AutoML model selection.
- Destructive feature selection (we do not drop columns silently based on mutual information).
- Target variable transformation.

For full architectural details, see the [Architecture Docs](PLAN.md) (Note: currently an internal development document, but will be expanded in future releases).
