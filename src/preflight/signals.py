"""
Column signals for the cleaning and engineering stages.

The v2 schema records structure and type only. The cleaner and engineer also
need numbers v1's Profiler computed (missing rate, cardinality, rare values,
target relationships...). They are ported here unchanged from v1
`profiler.py`; nothing else from the Profiler is ported. The column kind comes
from the schema's inferred type, not from v1's heuristics (see
docs/port_contract.md section 2); v1's own inference is only used as a
fallback when no schema entry exists for a column.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dateutil.parser import ParserError, parse

from preflight.engineer.features import compute_mutual_info_with_target
from preflight.engineer.vif import (
    compute_correlation_with_target,
    compute_vif_scores,
    flag_leakage_suspect,
)
from preflight.schema import TableSchema

# Kind names are v1's SemanticType values.
NUMERIC_FEATURE = "NUMERIC_FEATURE"
NUMERIC_ID = "NUMERIC_ID"
CATEGORICAL_LOW = "CATEGORICAL_LOW"
CATEGORICAL_HIGH = "CATEGORICAL_HIGH"
DATETIME_NATIVE = "DATETIME_NATIVE"
DATETIME_STRING = "DATETIME_STRING"
BOOLEAN = "BOOLEAN"
CONSTANT = "CONSTANT"
TEXT = "TEXT"

CATEGORICAL_KINDS = (CATEGORICAL_LOW, CATEGORICAL_HIGH)
DATETIME_KINDS = (DATETIME_NATIVE, DATETIME_STRING)

DEFAULT_CARDINALITY_THRESHOLD = 20


@dataclass
class ColumnSignals:
    """Per-column numbers the cleaner and engineer read (v1 ColumnProfile fields)."""
    name: str
    kind: str
    missing_rate: float
    dtype: str = ""
    outlier_rate: Optional[float] = None
    cardinality: int = 0
    rare_categories: Optional[List[str]] = None
    vif_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    mutual_info_with_target: Optional[float] = None
    is_leakage_suspect: bool = False
    text_avg_length: Optional[float] = None
    text_avg_word_count: Optional[float] = None
    v2_type: Optional[str] = None
    confidence: float = 1.0
    abstained: bool = False
    anomalies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ported verbatim from v1 profiler.py
# ---------------------------------------------------------------------------

def compute_missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def compute_outlier_rate(series: pd.Series) -> Optional[float]:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
    return float(len(outliers) / len(non_null))


def compute_cardinality(series: pd.Series) -> int:
    return int(series.nunique())


def identify_rare_categories(series: pd.Series, threshold: float = 0.01) -> List[str]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    freqs = non_null.value_counts(normalize=True)
    rare = freqs[freqs < threshold]
    return [str(x) for x in rare.index]


def compute_class_imbalance_ratio(target: pd.Series) -> Optional[float]:
    if pd.api.types.is_float_dtype(target):
        return None
    counts = target.dropna().value_counts()
    if len(counts) < 2:
        return None
    return float(counts.max() / counts.min())


def _text_like(non_null: pd.Series) -> bool:
    sample_size = min(100, len(non_null))
    sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
    sample_str = sample.astype(str)
    avg_len = sample_str.str.len().mean()
    avg_words = sample_str.str.split().str.len().mean()
    return bool(avg_len > 50 or (avg_len > 30 and avg_words >= 5))


def v1_infer_kind(series: pd.Series, cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD) -> str:
    """v1 `infer_semantic_type`, unchanged. Fallback for columns with no schema entry."""
    non_null = series.dropna()
    n_unique = non_null.nunique()
    total_len = len(series)

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        if len(non_null) > 0 and _text_like(non_null):
            return TEXT

    if n_unique <= 1:
        return CONSTANT
    if pd.api.types.is_bool_dtype(series) or n_unique == 2:
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME_NATIVE

    if pd.api.types.is_numeric_dtype(series):
        name_lower = str(series.name).lower() if series.name else ""
        if any(pat in name_lower for pat in ["zip", "postal", "postcode"]):
            return CATEGORICAL_LOW if n_unique < cardinality_threshold else CATEGORICAL_HIGH
        is_high_cardinality = (n_unique / total_len >= 0.95) if total_len > 0 else False
        matches_id_pattern = any(pat in name_lower for pat in ["id", "_id", "pk", "index"])
        is_integer = pd.api.types.is_integer_dtype(series)
        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
        if is_high_cardinality and (matches_id_pattern or (is_monotonic and is_integer)):
            return NUMERIC_ID
        return NUMERIC_FEATURE

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        sample_size = min(100, len(non_null))
        sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
        if len(sample) > 0:
            success_count = 0
            for val in sample:
                try:
                    if isinstance(val, str):
                        parse(val, fuzzy=False)
                        success_count += 1
                except (ValueError, TypeError, OverflowError, ParserError):
                    pass
            if success_count / len(sample) > 0.9:
                return DATETIME_STRING

    return CATEGORICAL_LOW if n_unique < cardinality_threshold else CATEGORICAL_HIGH


# ---------------------------------------------------------------------------
# v2: kind from the schema's inferred type
# ---------------------------------------------------------------------------

def _v1_numeric_id(series: pd.Series, n_unique: int) -> bool:
    """v1 rule: nearly all values unique AND (ID-like name OR an ascending/descending integer run)."""
    total_len = len(series)
    if total_len == 0 or n_unique / total_len < 0.95:
        return False
    name_lower = str(series.name).lower() if series.name else ""
    if any(pat in name_lower for pat in ["zip", "postal", "postcode"]):
        return False
    matches_id_pattern = any(pat in name_lower for pat in ["id", "_id", "pk", "index"])
    is_integer = pd.api.types.is_integer_dtype(series)
    if not is_integer and pd.api.types.is_float_dtype(series):
        nn = series.dropna()
        is_integer = bool(len(nn) > 0 and (nn % 1 == 0).all())
    is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
    return bool(matches_id_pattern or (is_monotonic and is_integer))


def _plain_numbers(non_null: pd.Series) -> bool:
    """Text values that are all plain numbers without leading zeros (so safe to treat as numeric IDs)."""
    if len(non_null) == 0:
        return False
    strs = non_null.astype(str).str.strip()
    if strs.str.match(r"^0\d").any():
        return False
    return bool(pd.to_numeric(strs, errors="coerce").notna().all())


def kind_from_schema(
    series: pd.Series,
    v2_type: str,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
) -> str:
    """Map a v2 inferred type to a v1 kind (docs/port_contract.md section 2)."""
    t = (v2_type or "").lower()
    non_null = series.dropna()
    n_unique = non_null.nunique()

    if t == "constant":
        return CONSTANT
    if t == "boolean":
        return BOOLEAN
    if t == "datetime":
        return DATETIME_NATIVE if pd.api.types.is_datetime64_any_dtype(series) else DATETIME_STRING
    if t in ("int", "float", "numeric"):
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # v1's numeric-ID rule is kept for numeric columns v2 did not call IDs
            if _v1_numeric_id(series, n_unique):
                return NUMERIC_ID
        return NUMERIC_FEATURE  # string-with-units columns are also numeric; the cleaner reads them
    if t == "id":
        if pd.api.types.is_numeric_dtype(series) or _plain_numbers(non_null):
            return NUMERIC_ID
        return CATEGORICAL_HIGH
    if t == "text" and len(non_null) > 0 and _text_like(non_null):
        return TEXT
    # "categorical", short "text", and anything else
    return CATEGORICAL_LOW if n_unique < cardinality_threshold else CATEGORICAL_HIGH


def compute_signals(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
    rare_threshold: float = 0.01,
) -> Dict[str, ColumnSignals]:
    """
    Compute ColumnSignals for every non-target column of `df`.

    Pure: `df` is not modified. `task` is "classification" or "regression";
    when None it is skipped for mutual information.
    """
    meta = {}
    if schema and schema.columns:
        for c in schema.columns:
            meta[c.name] = c

    target_series = df[target] if target and target in df.columns else None
    out: Dict[str, ColumnSignals] = {}
    numeric_feature_cols: List[str] = []

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        m = meta.get(col)
        v2_type = m.effective_type() if m is not None else None
        if v2_type:
            kind = kind_from_schema(series, v2_type, cardinality_threshold)
        else:
            kind = v1_infer_kind(series, cardinality_threshold)

        rare = None
        if kind in CATEGORICAL_KINDS:
            rare = identify_rare_categories(series, threshold=rare_threshold)

        corr = None
        if kind in (NUMERIC_FEATURE, NUMERIC_ID) and target_series is not None:
            corr = compute_correlation_with_target(series, target_series)

        mi = None
        if target_series is not None and task and kind not in (CONSTANT, NUMERIC_ID):
            mi = compute_mutual_info_with_target(series, target_series, task=task)

        sig = ColumnSignals(
            name=col,
            kind=kind,
            missing_rate=compute_missing_rate(series),
            dtype=str(series.dtype),
            outlier_rate=compute_outlier_rate(series),
            cardinality=compute_cardinality(series),
            rare_categories=rare,
            correlation_with_target=corr,
            mutual_info_with_target=mi,
            is_leakage_suspect=flag_leakage_suspect(corr),
            v2_type=v2_type,
            confidence=getattr(m, "confidence", 1.0) if m is not None else 1.0,
            abstained=bool(getattr(m, "abstained", False)) if m is not None else False,
            anomalies=list(getattr(m, "anomalies", []) or []) if m is not None else [],
        )
        if kind == TEXT:
            non_null_text = series.dropna().astype(str)
            if len(non_null_text) > 0:
                sig.text_avg_length = float(non_null_text.str.len().mean())
                sig.text_avg_word_count = float(non_null_text.str.split().str.len().mean())
            else:
                sig.text_avg_length = 0.0
                sig.text_avg_word_count = 0.0
        if kind == NUMERIC_FEATURE:
            numeric_feature_cols.append(col)
        out[col] = sig

    if numeric_feature_cols:
        scores, _ = compute_vif_scores(df, numeric_feature_cols)
        for col in numeric_feature_cols:
            out[col].vif_score = scores.get(col)

    return out
