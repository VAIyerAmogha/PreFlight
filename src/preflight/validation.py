"""
Centralized input validation helpers for the PreFlight public API.

All four public functions (prepare, profile, clean, engineer) delegate their
input checks here so that the same rules fire consistently and in the same order,
and so that task/target mismatch is detected *before* any pipeline stage runs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Heuristic thresholds — documented so they are easy to tune.
# ---------------------------------------------------------------------------
#: Maximum number of unique values a target may have before we consider it
#: "continuous" for the purposes of task mismatch detection.
_UNIQUE_COUNT_HARD_THRESHOLD: int = 20

#: Maximum fraction of rows that may be unique values before the target is
#: considered continuous.  E.g. 0.05 means "more than 5 % of rows are unique."
_UNIQUE_FRACTION_THRESHOLD: float = 0.05


def _validate_task_target_match(
    target_series: pd.Series,
    task: str,
) -> Optional[str]:
    """Detect a probable task/target mismatch and either raise or warn.

    Heuristic rules
    ---------------
    *classification + continuous target* → **raise** ``ValueError``.
        A target is considered "continuous" when its dtype is numeric AND either:
        - it has more than ``_UNIQUE_COUNT_HARD_THRESHOLD`` (default 20) unique
          values, **or**
        - its unique-value count exceeds 5 % of the total row count.

        Both thresholds must be exceeded for the error to fire, so a small
        dataset with 25 integer class labels (e.g. 0–24) does not raise.

    *regression + low-cardinality / non-numeric target* → **return warning string**.
        Low-cardinality regression targets (e.g. star ratings 1–5, or an integer
        score column) are legitimate, so we never raise.  The caller is expected
        to attach the returned string as a ``ReportEntry`` warning so it surfaces
        in ``Report.show()``.

    Parameters
    ----------
    target_series:
        The raw target column extracted from the user's DataFrame.
    task:
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    Optional[str]
        ``None`` if everything looks fine, or a warning message string for the
        regression + low-cardinality case.

    Raises
    ------
    ValueError
        When ``task="classification"`` but the target looks continuous.
    """
    name = target_series.name
    n_rows = len(target_series)
    n_unique = target_series.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    if task == "classification":
        # A numeric column with many unique values is almost certainly continuous.
        exceeds_hard_threshold = n_unique > _UNIQUE_COUNT_HARD_THRESHOLD
        exceeds_fraction = (n_rows > 0) and (n_unique / n_rows > _UNIQUE_FRACTION_THRESHOLD)

        if is_numeric and exceeds_hard_threshold and exceeds_fraction:
            raise ValueError(
                f"target '{name}' looks continuous ({n_unique} unique values) but "
                f"task='classification' was passed. "
                f"Did you mean task='regression'?"
            )

    elif task == "regression":
        # Non-numeric targets are always suspicious for regression.
        # Numeric targets with very few unique values are also worth flagging.
        is_non_numeric = not is_numeric
        is_low_cardinality_int = is_numeric and n_unique <= _UNIQUE_COUNT_HARD_THRESHOLD

        if is_non_numeric or is_low_cardinality_int:
            cardinality_info = (
                f"non-numeric dtype ({target_series.dtype})"
                if is_non_numeric
                else f"only {n_unique} unique value(s)"
            )
            return (
                f"target '{name}' has {cardinality_info} but task='regression' was passed. "
                f"If this is intentional (e.g. an ordinal score or integer rating) "
                f"you can ignore this warning."
            )

    return None


def _validate_inputs(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_hint: Optional[str] = None,
) -> Optional[str]:
    """Run all standard API boundary checks for the four public functions.

    Checks performed (in order)
    ----------------------------
    1. DataFrame is non-empty.
    2. ``target`` column exists.
    3. At least one feature column besides the target.
    4. No duplicate column names.
    5. ``task`` is ``"classification"`` or ``"regression"``.
    6. ``model_hint`` (when provided) is ``"tree"`` or ``"linear"``.
    7. Task / target mismatch via :func:`_validate_task_target_match`.

    Parameters
    ----------
    df:
        The raw user DataFrame.
    target:
        Name of the target column.
    task:
        ``"classification"`` or ``"regression"``.
    model_hint:
        Optional model family hint.  Pass ``None`` to skip check #6.

    Returns
    -------
    Optional[str]
        A warning message string if a regression + low-cardinality target is
        detected, otherwise ``None``.  The caller should convert this into a
        ``ReportEntry`` with ``stage="profiler", severity="warning"``.

    Raises
    ------
    ValueError
        On any structural problem or a classification + continuous-target mismatch.
    """
    # 1. Non-empty
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")

    # 2. Target exists
    if target not in df.columns:
        raise ValueError(
            f"target '{target}' not found in DataFrame columns: {list(df.columns)}"
        )

    # 3. At least one feature besides target
    if len(df.columns) < 2:
        raise ValueError(
            "DataFrame must contain at least one feature column besides the target"
        )

    # 4. No duplicate column names
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")

    # 5. Valid task
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got '{task}'"
        )

    # 6. Valid model_hint (only when the caller supplies one)
    if model_hint is not None and model_hint not in ("tree", "linear"):
        raise ValueError(
            f"model_hint must be 'tree' or 'linear', got '{model_hint}'"
        )

    # 7. Task / target mismatch — may raise (classification+continuous) or warn
    warning = _validate_task_target_match(df[target], task)
    return warning
