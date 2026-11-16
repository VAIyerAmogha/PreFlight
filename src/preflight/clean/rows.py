"""
Row-level checks around the target for PreFlight v2.0.

- missing_target_rows: rows whose target is blank cannot be learned from. They are
  set aside before any step learns from the table (stateless; a no-op on new data,
  which has no target column).
- class_imbalance: a note when the rarest answer is a small share of the rows, so a
  plain random split or plain accuracy would mislead.
- grouped_rows: a note when an identifier-like column repeats, so rows belong to
  groups (the same patient, customer, animal) and a random split puts one group on
  both sides.

The two notes change nothing; they make the risk visible in the plan and report.
"""

from typing import Optional

import pandas as pd

from preflight.clean.drops import looks_like_id_name, unique_ratio
from preflight.decisions import Decision
from preflight.signals import ColumnSignals

# Rarest answer below this share of rows -> class_imbalance note.
IMBALANCE_MIN_SHARE = 0.20
# Identifier-named column: repeats count as groups when distinct values are at least
# this share of rows (fewer distinct values reads as a code, not an identity).
GROUP_MIN_UNIQUE_RATIO = 0.05
# Column read as an identifier but not named like one (e.g. codes with leading zeros):
# a stricter bar, so short code lists are not mistaken for groups.
GROUP_MIN_UNIQUE_RATIO_UNNAMED = 0.5
GROUP_MAX_UNIQUE_RATIO = 0.95  # at or above this it is a row label (dropped instead)


def plan_missing_target(df: pd.DataFrame, target: Optional[str]) -> list[Decision]:
    """One Decision setting aside the rows whose target is blank, if there are any."""
    if not target or target not in df.columns or len(df) == 0:
        return []
    n_missing = int(df[target].isna().sum())
    # Nothing to set aside; and an entirely blank target is not a row problem (setting
    # every row aside would silently empty the table): planning stops on it earlier.
    if n_missing == 0 or n_missing == len(df):
        return []
    share = n_missing / len(df)
    row_word = "row" if n_missing == 1 else "rows"
    return [
        Decision(
            id="",
            stage="clean",
            kind="missing_target_rows",
            columns=[target],
            action="drop_rows",
            evidence={"missing_count": n_missing, "total_rows": len(df), "missing_fraction": round(share, 4)},
            confidence=1.0,
            default="drop_rows",
            needs_attention=False,
            plain_language=(
                f"Set aside {n_missing} {row_word} ({share:.1%} of the table) where '{target}', "
                "the value to predict, is blank: a model cannot learn from rows without an answer."
            ),
            status="automatic",
        )
    ]


def apply_missing_target(df: pd.DataFrame, decisions: list[Decision]) -> tuple[pd.DataFrame, int]:
    """Drop rows with a blank target. No-op when the target column is absent (new data)."""
    for d in decisions:
        if d.kind == "missing_target_rows" and d.action == "drop_rows" and d.columns:
            target = d.columns[0]
            if target in df.columns:
                keep = df[target].notna()
                n = int((~keep).sum())
                if n:
                    return df[keep].reset_index(drop=True), n
    return df, 0


def plan_class_imbalance(df: pd.DataFrame, target: Optional[str], task: Optional[str]) -> list[Decision]:
    """A note when the rarest answer of a classification target is under IMBALANCE_MIN_SHARE."""
    if task != "classification" or not target or target not in df.columns:
        return []
    counts = df[target].dropna().astype(str).value_counts()
    if len(counts) < 2:
        return []
    total = int(counts.sum())
    rare_label, rare_count = str(counts.index[-1]), int(counts.iloc[-1])
    share = rare_count / total
    if share >= IMBALANCE_MIN_SHARE:
        return []
    return [
        Decision(
            id="",
            stage="clean",
            kind="class_imbalance",
            columns=[target],
            action="note_class_imbalance",
            evidence={
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "rarest_class": rare_label,
                "rarest_share": round(share, 4),
                "threshold": IMBALANCE_MIN_SHARE,
            },
            confidence=1.0,
            default="note_class_imbalance",
            needs_attention=False,
            plain_language=(
                f"Note: the answers in '{target}' are uneven. The rarest one ('{rare_label}') is only "
                f"{share:.1%} of rows ({rare_count} of {total}). Split the data so each part keeps "
                "this mix, and judge results with a score that counts the rare answer fairly, "
                "not plain accuracy."
            ),
            status="automatic",
        )
    ]


def plan_grouped_rows(
    df: pd.DataFrame,
    target: Optional[str],
    signals: Optional[dict[str, ColumnSignals]] = None,
) -> list[Decision]:
    """A note per identifier-like column whose values repeat (rows come in groups)."""
    decisions: list[Decision] = []
    for col in df.columns:
        if col == target:
            continue
        sig = (signals or {}).get(col)
        named = looks_like_id_name(col)
        typed_id = sig is not None and (sig.v2_type or "").lower() == "id"
        if not (named or typed_id):
            continue
        series = df[col]
        nn = series.dropna()
        if len(nn) < 2:
            continue
        ratio = unique_ratio(series)
        low = GROUP_MIN_UNIQUE_RATIO if named else GROUP_MIN_UNIQUE_RATIO_UNNAMED
        if not (low <= ratio < GROUP_MAX_UNIQUE_RATIO):
            continue
        counts = nn.astype(str).value_counts()
        repeated = counts[counts > 1]
        if repeated.empty:
            continue
        n_groups, n_rows = int(len(repeated)), int(repeated.sum())
        decisions.append(
            Decision(
                id="",
                stage="clean",
                kind="grouped_rows",
                columns=[col],
                action="note_grouped_rows",
                evidence={
                    "distinct_values": int(nn.nunique()),
                    "repeated_values": n_groups,
                    "rows_in_repeats": n_rows,
                    "unique_ratio": round(ratio, 4),
                    "examples": [str(x) for x in repeated.index[:5]],
                },
                confidence=0.8,
                default="note_grouped_rows",
                needs_attention=False,
                plain_language=(
                    f"Note: '{col}' looks like an identifier, but {n_groups} of its values repeat "
                    f"across {n_rows} rows, so rows come in groups (for example the same person or "
                    "item measured more than once). When checking a model, keep each group "
                    "entirely in one part of the split, or results will look better than they "
                    f"are. Consider leaving '{col}' out as an input."
                ),
                status="automatic",
            )
        )
    return decisions
