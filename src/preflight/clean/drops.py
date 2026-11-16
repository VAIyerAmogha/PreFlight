"""
Column drop planning and execution for PreFlight v2.0 (high missingness, ID, constant).
"""

import re
from collections.abc import Sequence
from typing import Any, Optional

import pandas as pd

from preflight.clean._common import need_signals
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.signals import CONSTANT, NUMERIC_ID, ColumnSignals, _text_like

# "id" as its own word ("ID", "user_id", "Sample ID") or a camel-case suffix
# ("customerID", "PassengerId"); never inside a word ("valid", "width", "Paid").
_ID_NAME = re.compile(r"(?:^|[^A-Za-z])(?:id|ID|Id)(?:[^A-Za-z]|$)|[a-z](?:Id|ID)$")
# A row label is (nearly) unique per row; v1's numeric-ID rule uses the same bar.
ROW_LABEL_UNIQUE_RATIO = 0.95
ROW_LABEL_MIN_ROWS = 20


def looks_like_id_name(name: Any) -> bool:
    """True when a column name reads as an identifier ("customerID", "user_id", "ID")."""
    return bool(_ID_NAME.search(str(name)))


def unique_ratio(series: pd.Series) -> float:
    """Distinct non-blank values / non-blank values (0.0 for an empty column)."""
    nn = series.dropna()
    return float(nn.nunique() / len(nn)) if len(nn) else 0.0


def is_row_label(series: pd.Series, sig: ColumnSignals) -> bool:
    """
    A column that labels rows instead of describing them, whatever its stored type:
    nearly every value is unique AND the column was read as an identifier or is named
    like one. Covers text IDs ("7590-VHVEG") that the numeric-ID rule cannot see, and
    numeric IDs that were kept as text. Free text (names, comments) is never a label,
    and an "identifier" with few distinct values (codes with leading zeros) is a feature.
    """
    nn = series.dropna()
    if len(nn) < ROW_LABEL_MIN_ROWS or unique_ratio(series) < ROW_LABEL_UNIQUE_RATIO:
        return False
    if _text_like(nn):
        return False
    return (sig.v2_type or "").lower() == "id" or looks_like_id_name(series.name)


def should_drop_column(missing_rate: float, drop_threshold: float = 0.6) -> bool:
    """
    Pure decision function, no side effects.
    True if missing_rate exceeds drop_threshold.
    """
    return missing_rate > drop_threshold


def drop_high_missingness_columns(
    df: pd.DataFrame,
    columns_with_missing_rate: Sequence[tuple[str, float]],
    drop_threshold: float = 0.6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns exceeding drop_threshold missingness.
    """
    cols_to_drop = [
        col for col, rate in columns_with_missing_rate
        if should_drop_column(rate, drop_threshold) and col in df.columns
    ]
    if not cols_to_drop:
        return df.copy(), []
    return df.drop(columns=cols_to_drop), cols_to_drop


def drop_numeric_id_columns(
    df: pd.DataFrame,
    id_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop identified ID columns.
    """
    cols_to_drop = [c for c in id_columns if c in df.columns]
    if not cols_to_drop:
        return df.copy(), []
    return df.drop(columns=cols_to_drop), cols_to_drop


def drop_constant_columns(
    df: pd.DataFrame,
    constant_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop identified constant columns.
    """
    cols_to_drop = [c for c in constant_columns if c in df.columns]
    if not cols_to_drop:
        return df.copy(), []
    return df.drop(columns=cols_to_drop), cols_to_drop


def plan_drops(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    target_column: Optional[str] = None,
    signals: Optional[dict[str, ColumnSignals]] = None,
) -> list[Decision]:
    """
    Plan column drops (v1 order: high missingness, then ID, then constant).
    Never drops the target. If the fraction of columns dropped exceeds
    config.column_drop_impact_threshold, every drop is needs_attention and its
    safe default is to keep the column.
    """
    target = target_column or target
    total_cols = len(df.columns)
    if total_cols == 0:
        return []

    sigs = need_signals(df, signals, schema, target, config.cardinality_threshold)

    cols_to_drop: list[tuple[str, str, dict[str, Any]]] = []
    for col in df.columns:
        if col == target or col not in sigs:
            continue
        sig = sigs[col]
        if should_drop_column(sig.missing_rate, config.drop_missing_threshold):
            cols_to_drop.append(
                (col, "high_missingness", {"missing_rate": sig.missing_rate, "threshold": config.drop_missing_threshold})
            )
        elif sig.kind == NUMERIC_ID:
            cols_to_drop.append((col, "numeric_id", {"cardinality": sig.cardinality}))
        elif is_row_label(df[col], sig):
            cols_to_drop.append(
                (col, "row_label", {"cardinality": sig.cardinality, "unique_ratio": unique_ratio(df[col])})
            )
        elif sig.kind == CONSTANT:
            cols_to_drop.append((col, "constant", {"unique_count": sig.cardinality}))

    if not cols_to_drop:
        return []

    col_impact = len(cols_to_drop) / total_cols
    needs_att = col_impact > config.column_drop_impact_threshold

    decisions: list[Decision] = []
    for col, reason, ev_detail in cols_to_drop:
        if reason == "high_missingness":
            summary = f"Remove column '{col}' because {ev_detail['missing_rate']:.1%} of its entries are blank."
            kind = "drop_high_missingness"
        elif reason in ("numeric_id", "row_label"):
            summary = f"Remove identifier column '{col}' (a unique label for each row, not useful for learning)."
            kind = "drop_id"
        else:
            summary = f"Remove column '{col}' because every row has the same value."
            kind = "drop_constant"
        if needs_att:
            summary = (
                f"{summary} This is one of {len(cols_to_drop)} columns "
                f"({col_impact:.0%} of the table) planned for removal; please confirm."
            )

        decisions.append(
            Decision(
                id="",
                stage="clean",
                kind=kind,
                columns=[col],
                action="drop_column",
                evidence={
                    "reason": reason,
                    "total_columns": total_cols,
                    "columns_dropped_count": len(cols_to_drop),
                    "column_impact_fraction": col_impact,
                    "impact_threshold": config.column_drop_impact_threshold,
                    **ev_detail,
                },
                confidence=0.95,
                default="keep_column" if needs_att else "drop_column",
                needs_attention=needs_att,
                plain_language=summary,
                status="automatic",
            )
        )
    return decisions


def apply_drops(
    df: pd.DataFrame,
    decisions: list[Decision],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply column drops whose action in force is drop_column."""
    dropped_cols = []
    for d in decisions:
        if d.kind in ("drop_column", "drop_high_missingness", "drop_id", "drop_constant"):
            if d.action == "drop_column" and d.columns:
                col = d.columns[0]
                if col in df.columns:
                    dropped_cols.append(col)

    if not dropped_cols:
        return df.copy(), []

    return df.drop(columns=dropped_cols), dropped_cols
