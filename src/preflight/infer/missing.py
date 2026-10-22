"""
Missing-value marker detection module for PreFlight v2.0 (Phase 3 / M2).

Detects disguised missing values per column before type scoring:
- Standard sentinels (empty string, NA, N/A, null, -, ?, etc.)
- Contextually suspicious numeric sentinels (999, -999, 9999, -99, -1 when other
  values are strictly non-negative or bounded)
- Column-specific frequency outliers (implausibly high frequency relative to
  the rest of the column distribution in mostly numeric or high-entropy columns).

Flags detected markers with confidence scores without auto-applying or mutating
the underlying raw data (mutation belongs to the Clean phase).
"""

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import pandas as pd


@dataclass(frozen=True)
class MissingMarker:
    """
    Metadata for a detected missing value marker in a column.

    Attributes:
        marker: The exact string representation of the marker.
        confidence: Confidence score between 0.0 and 1.0.
        reason: Explanation of why this token was classified as a missing marker.
        count: Number of occurrences in the evaluated column.
        frequency: Proportion of column rows matching this marker (0.0 to 1.0).
    """
    marker: str
    confidence: float
    reason: str
    count: int
    frequency: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert marker to dictionary."""
        return asdict(self)


# Standard literal string sentinels (case-insensitive)
STANDARD_SENTINELS: Dict[str, float] = {
    "": 1.0,
    "na": 1.0,
    "n/a": 1.0,
    "#n/a": 1.0,
    "null": 1.0,
    "none": 0.95,
    "nil": 0.95,
    "nan": 0.95,
    "-nan": 0.95,
    "nan.": 0.95,
    "missing": 0.90,
    "-": 0.95,
    "--": 0.95,
    "---": 0.95,
    "?": 1.0,
    "??": 0.95,
    ".": 0.90,
    "*": 0.85,
    "not available": 0.95,
    "not applicable": 0.95,
    "unspecified": 0.85,
}

# Contextually suspicious numeric sentinel candidates
NUMERIC_SENTINEL_CANDIDATES = {"999", "-999", "9999", "-99", "99", "-1"}


def _safe_float(val: str) -> Optional[float]:
    """Attempt parsing string as float, returning None on failure."""
    try:
        # Strip currency symbols and whitespace if needed
        cleaned = re.sub(r"[,$€£]", "", val.strip())
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def detect_missing_markers(
    values: Union[pd.Series, Sequence[Any]],
    column_name: str = "",
) -> List[MissingMarker]:
    """
    Detect missing value markers for a single column.

    Examines raw values to identify both standard sentinels and column-specific
    anomalous markers (e.g. context-dependent "-1" or "999").

    Args:
        values: Sequence or pandas Series of raw cell values.
        column_name: Optional column name for contextual inspection.

    Returns:
        List of MissingMarker objects detected in this column.
    """
    if isinstance(values, pd.Series):
        raw_list = [str(x) if x is not None else "" for x in values.tolist()]
    else:
        raw_list = [str(x) if x is not None else "" for x in values]

    total_rows = len(raw_list)
    if total_rows == 0:
        return []

    val_counts = Counter(raw_list)
    detected: Dict[str, MissingMarker] = {}

    # 1. Standard literal string sentinels
    for raw_val, count in val_counts.items():
        v_stripped = raw_val.strip()
        v_lower = v_stripped.lower()

        if v_lower in STANDARD_SENTINELS:
            conf = STANDARD_SENTINELS[v_lower]
            freq = count / total_rows
            reason = "empty_string" if v_stripped == "" else f"standard_sentinel: {v_stripped}"
            detected[raw_val] = MissingMarker(
                marker=raw_val,
                confidence=conf,
                reason=reason,
                count=count,
                frequency=freq,
            )
        elif v_lower == "unknown" and count > 0:
            # "unknown" in categorical columns is frequently a missing marker
            freq = count / total_rows
            detected[raw_val] = MissingMarker(
                marker=raw_val,
                confidence=0.85,
                reason="standard_sentinel: unknown",
                count=count,
                frequency=freq,
            )

    # 2. Contextually suspicious numeric sentinels
    # Parse non-sentinel values to observe the valid distribution
    remaining_vals = [v for v in raw_list if v not in detected]
    parsed_nums: List[float] = []
    num_strings: List[str] = []

    for v in remaining_vals:
        f = _safe_float(v)
        if f is not None:
            parsed_nums.append(f)
            num_strings.append(v.strip())

    if len(parsed_nums) >= 3:
        # Check for candidates like "-1", "999", "-999", "9999", "-99"
        for cand in NUMERIC_SENTINEL_CANDIDATES:
            if cand in val_counts and cand not in detected:
                cand_float = float(cand)
                cand_count = val_counts[cand]
                other_nums = [x for x in parsed_nums if x != cand_float]

                if len(other_nums) >= 3:
                    min_other = min(other_nums)
                    max_other = max(other_nums)

                    # Context 1: "-1" when all other values are strictly non-negative
                    if cand == "-1" and min_other >= 0:
                        detected[cand] = MissingMarker(
                            marker=cand,
                            confidence=0.85,
                            reason="contextually_suspicious: -1 in non-negative distribution",
                            count=cand_count,
                            frequency=cand_count / total_rows,
                        )

                    # Context 2: "999" or "9999" when other values are bounded far below (e.g. max <= 150)
                    elif cand in ("999", "9999") and max_other <= 150:
                        detected[cand] = MissingMarker(
                            marker=cand,
                            confidence=0.90,
                            reason=f"contextually_suspicious: {cand} far above distribution max {max_other}",
                            count=cand_count,
                            frequency=cand_count / total_rows,
                        )

                    # Context 3: "-99" or "-999" when other values are bounded well above (e.g. min >= -10)
                    elif cand in ("-99", "-999") and min_other >= -10:
                        detected[cand] = MissingMarker(
                            marker=cand,
                            confidence=0.90,
                            reason=f"contextually_suspicious: {cand} far below distribution min {min_other}",
                            count=cand_count,
                            frequency=cand_count / total_rows,
                        )

    # 3. Column-specific outlier detection (implausibly high frequency or isolated non-numeric in numeric column)
    # If a column is predominantly numeric (>80% numeric), any isolated non-numeric string is suspicious
    numeric_ratio = len(parsed_nums) / len(remaining_vals) if remaining_vals else 0.0
    if numeric_ratio >= 0.80 and len(remaining_vals) >= 10:
        for raw_val, count in val_counts.items():
            if raw_val not in detected:
                v_str = raw_val.strip()
                if _safe_float(v_str) is None and len(v_str) <= 10:
                    # Short non-numeric token in an otherwise numeric column
                    freq = count / total_rows
                    detected[raw_val] = MissingMarker(
                        marker=raw_val,
                        confidence=0.85,
                        reason=f"column_specific_outlier: non-numeric token '{raw_val}' in {numeric_ratio:.0%} numeric column",
                        count=count,
                        frequency=freq,
                    )

    # Return sorted by confidence descending, count descending
    return sorted(
        detected.values(),
        key=lambda m: (m.confidence, m.count),
        reverse=True,
    )
