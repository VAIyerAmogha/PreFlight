"""
Category normalization and rare-category grouping for PreFlight v2.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.clean._common import need_signals
from preflight.signals import CATEGORICAL_HIGH, CATEGORICAL_KINDS, ColumnSignals
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
    cardinality_threshold: Optional[int] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    signals: Optional[Dict[str, ColumnSignals]] = None,
) -> List[Decision]:
    """
    Plan text tidying for categorical columns (v1: CATEGORICAL_LOW and HIGH) and
    rare-value grouping for high-variety columns (CATEGORICAL_HIGH only).

    Rare values come from the column signals, i.e. they are found on the
    values as they were before tidying (as in v1).
    """
    card = cardinality_threshold if cardinality_threshold is not None else config.cardinality_threshold
    sigs = need_signals(df, signals, schema, target, card)
    decisions: List[Decision] = []

    for col in df.columns:
        if col == target or col not in sigs:
            continue
        sig = sigs[col]
        if sig.kind not in CATEGORICAL_KINDS:
            continue

        series = df[col]
        normalized = normalize_category_values(series)
        changed = int(((normalized != series) & series.notna()).sum()) if len(series) else 0
        if changed > 0:
            decisions.append(
                Decision(
                    id="",
                    stage="clean",
                    kind="normalize_categories",
                    columns=[col],
                    action="normalize_categories",
                    evidence={"unique_count": sig.cardinality, "entries_changed": changed},
                    confidence=1.0,
                    default="normalize_categories",
                    needs_attention=False,
                    plain_language=(
                        f"Tidy the text in column '{col}': trim extra spaces and use lowercase "
                        f"({changed} {'entry' if changed == 1 else 'entries'} change)."
                    ),
                    status="automatic",
                )
            )

        if sig.kind == CATEGORICAL_HIGH and sig.rare_categories:
            rare = list(sig.rare_categories)
            decisions.append(
                Decision(
                    id="",
                    stage="clean",
                    kind="group_rare_categories",
                    columns=[col],
                    action="group_rare_categories",
                    evidence={
                        "rare_categories": rare,
                        "rare_count": len(rare),
                        "threshold": config.rare_category_threshold,
                    },
                    confidence=0.95,
                    default="group_rare_categories",
                    needs_attention=False,
                    plain_language=(
                        f"Combine {len(rare)} very uncommon values in column '{col}' "
                        f"(each under {config.rare_category_threshold:.0%} of entries) into one group called '__rare__'."
                    ),
                    status="automatic",
                )
            )

    return decisions


def apply_normalization(
    df: pd.DataFrame,
    decisions: List[Decision],
) -> pd.DataFrame:
    """Apply text tidying and rare-value grouping from decisions."""
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
