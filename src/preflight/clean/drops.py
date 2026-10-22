"""
Column drop planning and execution for PreFlight v2.0 (high missingness, ID, constant).
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG


def should_drop_column(missing_rate: float, drop_threshold: float = 0.6) -> bool:
    """
    Pure decision function, no side effects.
    True if missing_rate exceeds drop_threshold.
    """
    return missing_rate > drop_threshold


def drop_high_missingness_columns(
    df: pd.DataFrame,
    columns_with_missing_rate: Sequence[Tuple[str, float]],
    drop_threshold: float = 0.6,
) -> Tuple[pd.DataFrame, List[str]]:
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
) -> Tuple[pd.DataFrame, List[str]]:
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
) -> Tuple[pd.DataFrame, List[str]]:
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
) -> List[Decision]:
    """
    Plan column drops for high-missingness, ID, and constant columns.
    Never drops the designated target column.
    Destructive threshold check: if fraction of columns dropped exceeds
    config.column_drop_impact_threshold, needs_attention is set to True.
    """
    target = target_column or target
    total_cols = len(df.columns)
    if total_cols == 0:
        return []

    # Map column types from schema if available
    type_map = {}
    if schema and schema.columns:
        for c in schema.columns:
            eff_type = c.effective_type() or "text"
            type_map[c.name] = eff_type.lower()

    cols_to_drop: List[Tuple[str, str, Dict[str, Any]]] = []

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        missing_rate = float(series.isna().mean())
        col_type = type_map.get(col, "")
        n_unique = int(series.nunique(dropna=False))
        n_unique_non_null = int(series.dropna().nunique())

        # 1. High Missingness
        if should_drop_column(missing_rate, config.drop_missing_threshold):
            cols_to_drop.append(
                (
                    col,
                    "high_missingness",
                    {
                        "missing_rate": round(missing_rate, 4),
                        "threshold": config.drop_missing_threshold,
                    },
                )
            )
            continue

        # 2. ID Column
        is_id = col_type == "id" or (
            (col_type in ("int", "numeric") or pd.api.types.is_numeric_dtype(series))
            and len(series) > 0
            and any(
                part in ("id", "pk", "index")
                for part in col.lower().replace("-", "_").split("_")
            )
        )
        if is_id:
            cols_to_drop.append(
                (
                    col,
                    "numeric_id",
                    {
                        "cardinality": n_unique_non_null,
                        "uniqueness_ratio": round(n_unique_non_null / max(1, len(series)), 4),
                    },
                )
            )
            continue

        # 3. Constant Column
        is_constant = col_type == "constant" or n_unique_non_null <= 1
        if is_constant:
            cols_to_drop.append(
                (
                    col,
                    "constant",
                    {
                        "unique_count": n_unique_non_null,
                    },
                )
            )

    if not cols_to_drop:
        return []

    col_impact = len(cols_to_drop) / total_cols
    needs_att = col_impact >= config.column_drop_impact_threshold

    decisions: List[Decision] = []
    for col, reason, ev_detail in cols_to_drop:
        if reason == "high_missingness":
            m_rate = ev_detail["missing_rate"]
            summary = f"Remove column '{col}' due to high missing data rate ({m_rate:.1%})."
            kind = "drop_high_missingness"
        elif reason == "numeric_id":
            summary = f"Remove identifier column '{col}' to prevent spurious correlations."
            kind = "drop_id"
        else:
            summary = f"Remove constant column '{col}' containing single repeated value."
            kind = "drop_constant"

        ev = {
            "reason": reason,
            "total_columns": total_cols,
            "columns_dropped_count": len(cols_to_drop),
            "column_impact_fraction": round(col_impact, 4),
            "impact_threshold": config.column_drop_impact_threshold,
            **ev_detail,
        }

        dec = Decision(
            id="",
            stage="clean",
            kind=kind,
            columns=[col],
            action="drop_column",
            evidence=ev,
            confidence=0.95,
            default="keep_column",
            needs_attention=needs_att,
            plain_language=summary,
            status="automatic",
        )
        decisions.append(dec)

    return decisions


def apply_drops(
    df: pd.DataFrame,
    decisions: List[Decision],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply column drops specified by confirmed/automatic Decisions.
    """
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
