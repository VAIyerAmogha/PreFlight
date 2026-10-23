"""Shared helpers for cleaning planners."""

from typing import Any, Dict, Optional

import pandas as pd

from preflight.schema import TableSchema
from preflight.signals import ColumnSignals, compute_signals


def py(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python values (JSON-safe, exact)."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def need_signals(
    df: pd.DataFrame,
    signals: Optional[Dict[str, ColumnSignals]],
    schema: Optional[TableSchema],
    target: Optional[str],
    cardinality_threshold: int,
) -> Dict[str, ColumnSignals]:
    """Return `signals`, computing them from `df` when not supplied."""
    if signals is not None:
        return signals
    return compute_signals(df, schema=schema, target=target, cardinality_threshold=cardinality_threshold)
