"""
Dialect detection module for PreFlight v2.0 ingest pipeline.

Tests candidate delimiters (comma, semicolon, tab, pipe) and quoting styles.
Scores delimiters by row-length regularity across a sample, verifies against a
larger sample or full file, and flags low-margin winners. Returns a typed DialectResult.
"""

from collections import Counter
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]


@dataclass(frozen=True)
class DialectResult:
    """Typed result object for CSV dialect detection."""
    delimiter: str
    quotechar: str
    escapechar: Optional[str]
    doublequote: bool
    confidence: float
    flagged: bool
    margin: float
    field_count_mode: int
    row_regularity: float


def _read_sample_lines(
    source: Union[str, Path, bytes, List[str]],
    encoding: str = "utf-8",
    max_lines: int = 1000,
) -> List[str]:
    """Read and decode lines from source up to max_lines."""
    if isinstance(source, list):
        return [line.rstrip("\r\n") for line in source[:max_lines]]

    raw_text: str
    if isinstance(source, bytes):
        raw_text = source.decode(encoding, errors="replace")
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        # Read up to ~1MB or full file to get lines
        with open(p, "r", encoding=encoding, errors="replace") as f:
            lines = []
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\r\n"))
            return lines
    else:
        raise TypeError(f"Unsupported source type for dialect detection: {type(source)}")

    return [line.rstrip("\r\n") for line in raw_text.splitlines()[:max_lines]]


def _score_delimiter(
    lines: List[str],
    delimiter: str,
    quotechar: str = '"',
    doublequote: bool = True,
    escapechar: Optional[str] = None,
) -> Tuple[float, int, float]:
    """
    Score a candidate delimiter based on row-length regularity.

    Returns:
        (score, mode_field_count, regularity_ratio)
    """
    if not lines:
        return 0.0, 0, 0.0

    # Filter out empty or whitespace-only lines
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0, 0, 0.0

    try:
        reader = csv.reader(
            non_empty,
            delimiter=delimiter,
            quotechar=quotechar,
            doublequote=doublequote,
            escapechar=escapechar,
        )
        counts = [len(row) for row in reader]
    except Exception:
        return 0.0, 0, 0.0

    if not counts:
        return 0.0, 0, 0.0

    # Mode field count
    counter = Counter(counts)
    mode_count, mode_freq = counter.most_common(1)[0]

    # If mode is 1, the delimiter did not split rows into tabular fields
    if mode_count <= 1:
        return 0.0, mode_count, 0.0

    total_rows = len(counts)
    regularity = mode_freq / total_rows

    # Measure variance across row field lengths
    if total_rows > 1:
        stdev = statistics.stdev(counts)
    else:
        stdev = 0.0

    # Penalty for irregular row lengths
    variance_penalty = min(1.0, stdev / mode_count)
    regularity_score = regularity * max(0.0, 1.0 - variance_penalty)

    # Slight bonus for structural multi-column clarity (diminishing returns)
    col_bonus = min(1.0, 0.7 + 0.05 * mode_count)
    final_score = regularity_score * col_bonus

    return round(final_score, 4), mode_count, round(regularity, 4)


def _detect_quoting_style(
    lines: List[str],
    delimiter: str,
) -> Tuple[str, bool, Optional[str]]:
    """Detect quote character, doublequote, and escape character."""
    sample_text = "\n".join(lines[:100])

    # Check for escaped quotes \"
    has_backslash_escaped_quote = r'\"' in sample_text
    # Check for doubled quotes ""
    has_doubled_quote = r'""' in sample_text

    # Default to double quote
    quotechar = '"'
    doublequote = True
    escapechar = None

    if has_backslash_escaped_quote:
        escapechar = "\\"
        doublequote = False
    elif has_doubled_quote:
        doublequote = True
        escapechar = None

    # Check if single quote is used as quotechar instead of double quote
    # If double quotes are completely absent, but single quotes wrap fields containing delimiter
    if '"' not in sample_text and "'" in sample_text:
        # Check if single quote parses cleanly
        score_sq, mode_sq, _ = _score_delimiter(lines[:50], delimiter, quotechar="'")
        score_dq, mode_dq, _ = _score_delimiter(lines[:50], delimiter, quotechar='"')
        if score_sq > score_dq:
            quotechar = "'"

    return quotechar, doublequote, escapechar


def detect_dialect(
    source: Union[str, Path, bytes, List[str]],
    encoding: str = "utf-8",
    sample_size: int = 100,
    verify_size: int = 1000,
    margin_threshold: float = 0.15,
) -> DialectResult:
    """
    Detect the CSV dialect (delimiter and quoting style) of a dataset.

    Evaluates candidates across an initial sample, verifies against a larger
    window, and flags ambiguous or low-margin decisions.

    Args:
        source: File path, raw bytes, or list of text lines.
        encoding: Character encoding to use when decoding bytes.
        sample_size: Number of lines for initial candidate scoring.
        verify_size: Number of lines for verification pass.
        margin_threshold: Score difference below which detection is flagged.

    Returns:
        DialectResult object.
    """
    # 1. Read initial sample and verification lines
    all_lines = _read_sample_lines(source, encoding=encoding, max_lines=verify_size)
    initial_sample = all_lines[:sample_size]

    if not initial_sample or not any(l.strip() for l in initial_sample):
        # Empty or whitespace only
        return DialectResult(
            delimiter=",",
            quotechar='"',
            escapechar=None,
            doublequote=True,
            confidence=0.5,
            flagged=True,
            margin=0.0,
            field_count_mode=1,
            row_regularity=0.0,
        )

    # 2. Score candidate delimiters on initial sample
    candidate_scores: Dict[str, Tuple[float, int, float]] = {}
    for delim in CANDIDATE_DELIMITERS:
        score, mode_cnt, reg = _score_delimiter(initial_sample, delim)
        candidate_scores[delim] = (score, mode_cnt, reg)

    # Sort delimiters by score descending
    ranked_initial = sorted(
        candidate_scores.items(), key=lambda item: item[1][0], reverse=True
    )

    top_delim, (top_score, top_mode, top_reg) = ranked_initial[0]
    runner_delim, (runner_score, _, _) = ranked_initial[1]

    # 3. Verification pass against larger verification sample
    if len(all_lines) > len(initial_sample) and top_score > 0.0:
        verified_score, verified_mode, verified_reg = _score_delimiter(all_lines, top_delim)
        # Runner-up verification
        verified_runner_score, _, _ = _score_delimiter(all_lines, runner_delim)

        final_top_score = (top_score + verified_score) / 2.0
        final_runner_score = (runner_score + verified_runner_score) / 2.0
        top_mode = verified_mode
        top_reg = verified_reg
    else:
        final_top_score = top_score
        final_runner_score = runner_score

    # 4. Handle edge case: single-column dataset (all delimiters score 0)
    if final_top_score == 0.0:
        return DialectResult(
            delimiter=",",
            quotechar='"',
            escapechar=None,
            doublequote=True,
            confidence=0.5,
            flagged=True,
            margin=0.0,
            field_count_mode=1,
            row_regularity=0.0,
        )

    # 5. Margin and Confidence Computation
    margin = max(0.0, final_top_score - final_runner_score)
    # Low margin winner or weak overall regularity triggers flagged = True
    is_low_margin = (margin < margin_threshold) or (final_top_score < 0.60)

    # Detect quoting parameters
    quotechar, doublequote, escapechar = _detect_quoting_style(all_lines, top_delim)

    confidence = round(min(1.0, max(0.0, final_top_score)), 4)

    return DialectResult(
        delimiter=top_delim,
        quotechar=quotechar,
        escapechar=escapechar,
        doublequote=doublequote,
        confidence=confidence,
        flagged=is_low_margin,
        margin=round(margin, 4),
        field_count_mode=top_mode,
        row_regularity=round(top_reg, 4),
    )
