"""
Header and table structure detection module for PreFlight v2.0 ingest pipeline.

Distinguishes header rows from data rows using type-consistency across candidate
header positions. Detects and excludes title/preamble lines, footer notes, blank rows,
and total/summary rows. Flags ragged rows with inconsistent field counts.
Returns a typed HeaderResult.
"""

import csv
from dataclasses import dataclass, field
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

YEAR_PATTERN = re.compile(r"^(?:1[89]|20)\d{2}$")
DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{4}"),
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}"),
]
BOOLEAN_TOKENS = {"true", "false", "t", "f", "yes", "no", "y", "n"}
TOTAL_MARKERS = {
    "total", "grand total", "summary", "average", "avg", "subtotal",
    "all", "overall", "mean", "median"
}
COMMENT_PREFIXES = ("#", "//", "/*", "--", "%", "rem ")


@dataclass(frozen=True)
class HeaderResult:
    """Typed result object for header and table boundary detection."""
    header_row_index: Optional[int]
    column_names: List[str]
    confidence: float
    title_rows: List[int] = field(default_factory=list)
    footer_rows: List[int] = field(default_factory=list)
    blank_rows: List[int] = field(default_factory=list)
    total_rows: List[int] = field(default_factory=list)
    ragged_rows: List[int] = field(default_factory=list)
    is_ragged: bool = False
    flagged: bool = False


def _is_numeric(val: str) -> bool:
    """Check if token can be parsed as float or integer, allowing units/currencies."""
    if not val:
        return False
    v = val.strip().rstrip("%").lstrip("$").lstrip("€").lstrip("£")
    try:
        float(v)
        return True
    except ValueError:
        return False


def _is_float_or_negative(val: str) -> bool:
    """Check if token is explicitly a float (decimal point) or negative number."""
    if not val:
        return False
    v = val.strip().rstrip("%").lstrip("$").lstrip("€").lstrip("£")
    try:
        f = float(v)
        return ("." in v) or (f < 0)
    except ValueError:
        return False


def _classify_token_type(token: str) -> str:
    """Classify token into coarse semantic group for type-consistency checking."""
    t = token.strip()
    if not t:
        return "empty"
    if t.lower() in BOOLEAN_TOKENS:
        return "boolean"
    if YEAR_PATTERN.match(t):
        return "year"
    if any(p.match(t) for p in DATE_PATTERNS):
        return "date"
    if _is_numeric(t):
        return "numeric"
    return "string"


def detect_header(
    source: Union[str, Path, bytes, List[str]],
    delimiter: str = ",",
    quotechar: str = '"',
    escapechar: Optional[str] = None,
    doublequote: bool = True,
    encoding: str = "utf-8",
    sample_rows_count: int = 100,
) -> HeaderResult:
    """
    Detect header row and table structural boundaries.

    Distinguishes header from data rows by type consistency across positions.
    Identifies and excludes title lines, footers, blank rows, and total lines.
    Flags ragged rows.

    Args:
        source: File path, raw bytes, or list of strings.
        delimiter: Delimiter character.
        quotechar: Quote character.
        escapechar: Escape character.
        doublequote: Whether quotes are doubled.
        encoding: Text encoding.
        sample_rows_count: Number of rows to inspect for table structure.

    Returns:
        HeaderResult object.
    """
    # 1. Read and parse lines
    lines: List[str]
    if isinstance(source, list):
        lines = source
    elif isinstance(source, bytes):
        lines = source.decode(encoding, errors="replace").splitlines()
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        with open(p, "r", encoding=encoding, errors="replace") as f:
            lines = [f.readline().rstrip("\r\n") for _ in range(sample_rows_count + 50)]
            lines = [l for l in lines if l]
    else:
        raise TypeError(f"Unsupported source type for header detection: {type(source)}")

    if not lines:
        return HeaderResult(
            header_row_index=None,
            column_names=[],
            confidence=0.5,
            flagged=True,
        )

    # Parse with CSV reader
    reader = csv.reader(
        lines,
        delimiter=delimiter,
        quotechar=quotechar,
        doublequote=doublequote,
        escapechar=escapechar,
    )
    raw_rows = list(reader)

    if not raw_rows:
        return HeaderResult(
            header_row_index=None,
            column_names=[],
            confidence=0.5,
            flagged=True,
        )

    # 2. Track blank rows
    blank_rows = []
    non_blank_rows: List[Tuple[int, List[str]]] = []
    for idx, row in enumerate(raw_rows):
        if not row or not any(cell.strip() for cell in row):
            blank_rows.append(idx)
        else:
            non_blank_rows.append((idx, row))

    if not non_blank_rows:
        return HeaderResult(
            header_row_index=None,
            column_names=[],
            confidence=0.5,
            blank_rows=blank_rows,
            flagged=True,
        )

    # Determine mode field count from non-blank rows
    field_counts = [len(r) for _, r in non_blank_rows]
    mode_field_count = (
        max(set(field_counts), key=field_counts.count) if field_counts else 1
    )

    # 3. Detect title/preamble rows above header
    title_rows = []
    start_pointer = 0
    while start_pointer < len(non_blank_rows):
        orig_idx, row = non_blank_rows[start_pointer]
        first_cell = row[0].strip() if row else ""

        # Comment syntax check
        is_comment = any(first_cell.startswith(p) for p in COMMENT_PREFIXES)
        # Single cell banner check when table is multi-column
        is_single_cell_banner = (len(row) == 1 and mode_field_count > 1)

        if is_comment or is_single_cell_banner:
            title_rows.append(orig_idx)
            start_pointer += 1
        else:
            break

    if start_pointer >= len(non_blank_rows):
        return HeaderResult(
            header_row_index=None,
            column_names=[f"col_{i}" for i in range(mode_field_count)],
            confidence=0.5,
            title_rows=title_rows,
            blank_rows=blank_rows,
            flagged=True,
        )

    # Candidate header row
    candidate_orig_idx, candidate_row = non_blank_rows[start_pointer]
    data_rows_entries = non_blank_rows[start_pointer + 1:start_pointer + 51]
    data_rows = [r for _, r in data_rows_entries]

    # 4. Detect footer rows and total/summary rows from the bottom of sample
    footer_rows = []
    total_rows = []
    if len(non_blank_rows) > 5:
        # Check last 3 rows
        for orig_idx, row in non_blank_rows[-3:]:
            if orig_idx <= candidate_orig_idx:
                continue
            first_cell_lower = row[0].strip().lower() if row else ""
            if any(first_cell_lower == m or first_cell_lower.startswith(f"{m}:") for m in TOTAL_MARKERS):
                total_rows.append(orig_idx)
            elif any(first_cell_lower.startswith(prefix) for prefix in ("source:", "note:", "disclaimer:")) or (
                len(row) == 1 and mode_field_count > 1
            ):
                footer_rows.append(orig_idx)

    # 5. Distinguish header row vs data row by type-consistency
    is_header: bool = True
    confidence: float = 0.95

    # Check 1: If candidate header contains floats or negative numbers, it is data (e.g. iris.data)
    if any(_is_float_or_negative(cell) for cell in candidate_row):
        is_header = False
        confidence = 0.98

    # Check 2: Count numeric tokens that are NOT 4-digit years (e.g. adult.data with age 39)
    if is_header:
        pure_numeric_count = 0
        for cell in candidate_row:
            c_strip = cell.strip()
            if _is_numeric(c_strip) and not YEAR_PATTERN.match(c_strip):
                pure_numeric_count += 1
        if pure_numeric_count > 0:
            is_header = False
            confidence = 0.95

    # Check 3: Type disparity against data rows
    if is_header and data_rows:
        num_cols = len(candidate_row)
        disparities = 0
        comparable_cols = 0

        for col_i in range(num_cols):
            h_type = _classify_token_type(candidate_row[col_i])
            col_vals = [r[col_i] for r in data_rows if col_i < len(r) and r[col_i].strip()]
            if not col_vals:
                continue
            comparable_cols += 1
            data_types = [_classify_token_type(v) for v in col_vals]
            most_common_type = max(set(data_types), key=data_types.count)

            # If header is string/year and data is numeric/date
            if h_type in ("string", "year") and most_common_type in ("numeric", "date"):
                disparities += 1

        if comparable_cols > 0 and disparities > 0:
            # Strong type disparity confirms header
            is_header = True
            confidence = min(1.0, 0.8 + 0.05 * disparities)

    # 6. Detect ragged rows
    ragged_rows = []
    expected_col_count = len(candidate_row) if is_header else mode_field_count
    for orig_idx, row in non_blank_rows:
        if orig_idx in title_rows or orig_idx in footer_rows:
            continue
        if len(row) != expected_col_count:
            ragged_rows.append(orig_idx)

    is_ragged = len(ragged_rows) > 0

    # 7. Construct column names
    if is_header:
        detected_header_idx = candidate_orig_idx
        # Deduplicate and sanitize empty column names
        raw_names = candidate_row
        seen_names: Set[str] = set()
        clean_names = []
        for i, name in enumerate(raw_names):
            n_clean = name.strip() if name.strip() else f"unnamed_{i}"
            if n_clean in seen_names:
                # Disambiguate duplicate header names
                count = sum(1 for existing in clean_names if existing.startswith(n_clean))
                n_clean = f"{n_clean}_{count}"
            seen_names.add(n_clean)
            clean_names.append(n_clean)
        column_names = clean_names
    else:
        detected_header_idx = None
        column_names = [f"col_{i}" for i in range(expected_col_count)]

    flagged = is_ragged or (confidence < 0.80)

    return HeaderResult(
        header_row_index=detected_header_idx,
        column_names=column_names,
        confidence=round(confidence, 4),
        title_rows=sorted(title_rows),
        footer_rows=sorted(footer_rows),
        blank_rows=sorted(blank_rows),
        total_rows=sorted(total_rows),
        ragged_rows=sorted(ragged_rows),
        is_ragged=is_ragged,
        flagged=flagged,
    )
