"""
Category normalization and rare-category grouping for PreFlight v2.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG


def normalize_category_values(series: pd.Series) -> pd.Series:
    """
    Lowercases and strips whitespace from string values.
    Leaves nulls as null. Non-string values passed through unchanged.
    """
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        normalized = series.apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
        return normalized.replace({np.nan: None})
    return series.copy()


def identify_rare_categories(series: pd.Series, threshold: float = 0.01) -> List[str]:
    """
    Identify categories occurring below the threshold frequency of non-null values.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return []

    freqs = non_null.value_counts(normalize=True)
    rare = freqs[freqs < threshold]
    return [str(x) for x in rare.index]


def group_rare_categories(
    series: pd.Series,
    rare_categories: Sequence[str],
    label: str = "__rare__",
) -> pd.Series:
    """
    Replaces any value in rare_categories with label.
    """
    rare_set = set(rare_categories)
    return series.apply(lambda x: label if x in rare_set else x)


def plan_normalization(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    cardinality_threshold: int = 20,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> List[Decision]:
    """
    Plan category normalization and rare category grouping for categorical columns.
    """
    type_map = {}
    if schema and schema.columns:
        for c in schema.columns:
            eff_type = c.effective_type() or "text"
            type_map[c.name] = eff_type.lower()

    decisions: List[Decision] = []

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        col_type = type_map.get(col, "")
        n_unique = int(series.dropna().nunique())

        is_categorical = col_type == "categorical" or isinstance(series.dtype, pd.CategoricalDtype)
        if not is_categorical:
            # Check if text column has low/high cardinality categorical behavior
            if col_type == "text" and n_unique > 0 and n_unique < len(series) * 0.5:
                is_categorical = True

        if not is_categorical:
            continue

        # 1. Normalization Decision (strip whitespace and lowercase)
        norm_dec = Decision(
            id="",
            stage="clean",
            kind="normalize_categories",
            columns=[col],
            action="normalize_categories",
            evidence={
                "unique_count": n_unique,
            },
            confidence=1.0,
            default="normalize_categories",
            needs_attention=False,
            plain_language=f"Standardize text in column '{col}' by stripping extra spaces and converting to lowercase.",
            status="automatic",
        )
        decisions.append(norm_dec)

        # 2. Rare Category Grouping (for high-cardinality categoricals)
        if n_unique >= cardinality_threshold:
            normalized_series = normalize_category_values(series)
            rare_cats = identify_rare_categories(normalized_series, threshold=config.rare_category_threshold)
            if rare_cats:
                rare_dec = Decision(
                    id="",
                    stage="clean",
                    kind="group_rare_categories",
                    columns=[col],
                    action="group_rare_categories",
                    evidence={
                        "rare_categories": rare_cats[:10],
                        "rare_count": len(rare_cats),
                        "threshold": config.rare_category_threshold,
                    },
                    confidence=0.95,
                    default="group_rare_categories",
                    needs_attention=False,
                    plain_language=f"Group {len(rare_cats)} infrequent category values in column '{col}' into a single group.",
                    status="automatic",
                )
                decisions.append(rare_dec)

    return decisions


def apply_normalization(
    df: pd.DataFrame,
    decisions: List[Decision],
) -> pd.DataFrame:
    """
    Apply category normalization and rare category grouping from decisions.
    """
    df_out = df.copy()

    for d in decisions:
        if not d.columns:
            continue
        col = d.columns[0]
        if col not in df_out.columns:
            continue

        if d.kind == "normalize_categories" and d.action == "normalize_categories":
            df_out[col] = normalize_category_values(df_out[col])

        elif d.kind == "group_rare_categories" and d.action == "group_rare_categories":
            rare_cats = d.evidence.get("rare_categories", [])
            if rare_cats:
                df_out[col] = group_rare_categories(df_out[col], rare_cats)

    return df_out
