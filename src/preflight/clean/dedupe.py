"""
Row deduplication planning and execution for PreFlight v2.0.
"""

from typing import List, Tuple
import pandas as pd

from preflight.decisions import Decision
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG


def remove_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Returns (deduplicated df, count of rows removed).
    Uses standard full-row duplicate definition without mutating input.
    """
    initial_len = len(df)
    new_df = df.drop_duplicates()
    removed_count = initial_len - len(new_df)
    return new_df, removed_count


def plan_dedupe(
    df: pd.DataFrame,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> List[Decision]:
    """
    Plan row deduplication. Emits a Decision if duplicate rows are present.
    If the fraction of rows removed exceeds config.dedupe_impact_threshold,
    needs_attention is set to True.
    """
    total_rows = len(df)
    if total_rows == 0:
        return []

    duplicate_mask = df.duplicated()
    dup_count = int(duplicate_mask.sum())
    if dup_count == 0:
        return []

    impact = dup_count / total_rows
    needs_att = impact >= config.dedupe_impact_threshold

    row_word = "row" if dup_count == 1 else "rows"
    plain_lang = (
        f"Remove {dup_count} duplicate {row_word} ({impact:.1%} of table)."
        if not needs_att
        else f"Review removal of {dup_count} duplicate {row_word} ({impact:.1%} of table)."
    )

    decision = Decision(
        id="",
        stage="clean",
        kind="remove_duplicates",
        columns=[],
        action="remove_duplicates",
        evidence={
            "duplicate_count": dup_count,
            "total_rows": total_rows,
            "impact_fraction": round(impact, 4),
            "threshold": config.dedupe_impact_threshold,
        },
        confidence=1.0,
        default="remove_duplicates",
        needs_attention=needs_att,
        plain_language=plain_lang,
        status="automatic",
    )
    return [decision]


def apply_dedupe(
    df: pd.DataFrame,
    decisions: List[Decision],
) -> Tuple[pd.DataFrame, int]:
    """
    Apply deduplication decision if confirmed or automatic with action 'remove_duplicates'.
    """
    should_dedupe = False
    for d in decisions:
        if d.kind == "remove_duplicates" and d.action == "remove_duplicates":
            should_dedupe = True
            break

    if not should_dedupe:
        return df.copy(), 0

    return remove_duplicate_rows(df)
