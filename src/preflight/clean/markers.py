"""
Missing-value marker conversion for PreFlight v2.0.

Phase 3 records disguised missing values ("N/A", "?", "unknown"...) in the
schema. This is the stage that converts them into real blanks.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema


def marker_strings(missing_markers: Sequence[Union[str, Dict[str, Any]]]) -> List[str]:
    out = set()
    for m in missing_markers:
        out.add(str(m.get("marker", "")).strip() if isinstance(m, dict) else str(m).strip())
    out.discard("")
    return sorted(out)


def apply_missing_markers_to_series(
    series: pd.Series,
    missing_markers: Sequence[Union[str, Dict[str, Any]]],
) -> pd.Series:
    """Replace detected missing-value markers with np.nan."""
    markers = set(marker_strings(missing_markers))
    if not markers:
        return series

    def _clean(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        if str(val).strip() in markers:
            return np.nan
        return val

    return series.apply(_clean)


def plan_markers(df: pd.DataFrame, schema: Optional[TableSchema] = None) -> List[Decision]:
    """One Decision per column whose schema markers actually occur in the data."""
    decisions: List[Decision] = []
    if not schema or not schema.columns:
        return decisions

    for c in schema.columns:
        if c.name not in df.columns or not c.missing_markers:
            continue
        markers = marker_strings(c.missing_markers)
        if not markers:
            continue
        series = df[c.name]
        hits = int(series.map(lambda v: (not pd.isna(v)) and str(v).strip() in set(markers)).sum())
        if hits == 0:
            continue
        decisions.append(
            Decision(
                id="",
                stage="clean",
                kind="convert_missing_markers",
                columns=[c.name],
                action="convert_missing_markers",
                evidence={"markers": markers, "cells": hits, "total_rows": len(series)},
                confidence=float(min(1.0, max(0.0, c.confidence))),
                default="convert_missing_markers",
                needs_attention=False,
                plain_language=(
                    f"Treat {', '.join(repr(m) for m in markers[:3])} in column '{c.name}' as blank "
                    f"({hits} {'entry' if hits == 1 else 'entries'})."
                ),
                status="automatic",
            )
        )
    return decisions


def apply_markers(df: pd.DataFrame, decisions: List[Decision]) -> pd.DataFrame:
    df_out = df.copy()
    for d in decisions:
        if d.kind == "convert_missing_markers" and d.action == "convert_missing_markers" and d.columns:
            col = d.columns[0]
            if col in df_out.columns:
                df_out[col] = apply_missing_markers_to_series(df_out[col], d.evidence.get("markers", []))
    return df_out
