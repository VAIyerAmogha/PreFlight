"""
Type inference, confidence scoring, trap detection, and abstain logic for PreFlight v2.0.

Evaluates raw non-missing column values against 8 candidate types:
- int, float, boolean, datetime, categorical, text, ID, constant.

Explicitly handles known traps:
- Leading-zero IDs / postal codes (rejects int coercion, preserving leading zeroes)
- European decimal comma vs. decimal point and thousands separators
- Ambiguous date formats (dd/mm vs mm/dd ambiguity flagged and abstained)
- Embedded units (a number with a short letter suffix, a currency symbol or a percent sign),
  found generically and flagged as a lossy conversion with the unit kept in the Decision

Emits a Decision record per column with confidence scoring, safe defaults,
and abstain behavior when candidate score margins are low or traps are unresolvable.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import ColumnRawMetadata, TableSchema

from .config import (
    PLUS_CODE_MIN_DIGITS,
    TEXT_NAME_HINT_MIN_WORDS,
    TYPE_SAMPLE_SEED,
    TYPE_SAMPLE_SIZE,
    UNIT_COLUMN_SHARE,
    UNIT_FLAG_SHARE,
    UNIT_MAX_LETTERS,
    UNIT_MIN_REPEATS,
)
from .missing import MissingMarker, detect_missing_markers


@dataclass(frozen=True)
class TypeInferenceResult:
    """
    Per-column result of type inference and anomaly detection.

    Attributes:
        column_name: Name of evaluated column.
        inferred_type: Winner candidate type or safe fallback if abstained.
        confidence: Numerical score (0.0 to 1.0) of winning type.
        runner_up: Close second-place candidate type if margin was tight.
        abstained: True if inference abstained rather than guessing.
        needs_attention: True if review is recommended (low confidence, trap, or abstain).
        anomalies: Detected anomaly tags (e.g. 'leading_zero', 'ambiguous_date_format').
        missing_markers: Detected missing value markers.
        failed_samples: Sample of values that failed to parse under inferred type.
        evidence: Diagnostic candidate scores, stats, and detection flags.
        decision: Decision dataclass instance representing this outcome.
    """
    column_name: str
    inferred_type: str
    confidence: float
    runner_up: Optional[str]
    abstained: bool
    needs_attention: bool
    anomalies: list[str]
    missing_markers: list[MissingMarker]
    failed_samples: list[str]
    evidence: dict[str, Any]
    decision: Decision


def to_canonical_type(inferred_type: str) -> str:
    """
    Map candidate inference type to canonical 7 ground-truth semantic types.

    Ground-truth semantic types:
    - numeric, categorical, datetime, text, boolean, constant, ID.
    """
    t = inferred_type.lower()
    if t in ("int", "float", "numeric"):
        return "numeric"
    if t == "id":
        return "ID"
    if t in ("boolean", "categorical", "datetime", "text", "constant"):
        return t
    return inferred_type


# ---------------------------------------------------------------------------
# Regex patterns and constants
# ---------------------------------------------------------------------------

BOOL_LITERAL_PAIRS = [
    ({"true", "false"}),
    ({"yes", "no"}),
    ({"y", "n"}),
    ({"t", "f"}),
    ({"1", "0"}),
]
BOOL_ALL_LITERALS = {v for pair in BOOL_LITERAL_PAIRS for v in pair}

ID_NAME_PATTERNS = ["id", "_id", "id_", "pk", "key", "index", "code", "fips", "zip", "postal", "rownames"]

# A number with a unit: currency symbol before it, or a short run of letters / a percent
# sign after it. The unit itself is whatever the data says (nothing is listed here).
_RE_UNIT_VALUE = re.compile(
    r"^(?P<pre>[$€£¥₹])?\s*(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)\s*"
    rf"(?P<suf>[A-Za-z]{{1,{UNIT_MAX_LETTERS}}}|%)?$"
)

DATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?",
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^\d{1,2}-\d{1,2}-\d{4}$",
    r"^\d{4}/\d{1,2}/\d{1,2}$",
    r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$",
    r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
    r"^\d{4}-\d{2}$",
]


_RE_INT = re.compile(r"^[-+]?\d+$")
_RE_INT_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(,\d{3})+$")
_RE_FLOAT_LZ = re.compile(r"^[-+]?0\d+(\.\d*)?$")
_RE_FLOAT = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
_RE_FLOAT_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(,\d{3})*(\.\d+)?$")
_RE_DEC_COMMA = re.compile(r"^[-+]?\d+(,\d+)$")
_RE_DEC_COMMA_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(\.\d{3})*(,\d+)$")
_DATE_COMPILED = [re.compile(p) for p in DATE_PATTERNS]


# ---------------------------------------------------------------------------
# Individual Candidate Type Parsers
# ---------------------------------------------------------------------------

def _parse_bool(v: str) -> bool:
    return v.strip().lower() in BOOL_ALL_LITERALS


def _is_plus_code(s: str) -> bool:
    """'+' and a long run of digits (e.g. +447911123456): a code whose '+' a number would drop."""
    return s.startswith("+") and s[1:].isdigit() and len(s) - 1 >= PLUS_CODE_MIN_DIGITS


def _parse_int(v: str) -> tuple[bool, bool]:
    """
    Parse integer without leading zero corruption.

    Returns:
        Tuple of (parses_cleanly, has_leading_zero).
    """
    s = v.strip()
    if not s:
        return False, False

    # Check for leading zero trap (e.g. "02138", "0012")
    if len(s) > 1 and s.isdigit() and s.startswith("0"):
        return False, True

    # A '+' code is not an integer (the sign is part of the code)
    if _is_plus_code(s):
        return False, False

    # Standard integer with optional sign
    if _RE_INT.match(s):
        return True, False

    # Integer with thousands comma separators (e.g. "1,000,000")
    if _RE_INT_THOUSANDS.match(s):
        return True, False

    return False, False


def _unit_of(s: str) -> Optional[str]:
    """The unit of a number-with-unit value ('12 abc' -> 'abc', '$12.50' -> '$'), else None."""
    m = _RE_UNIT_VALUE.match(s)
    if m is None:
        return None
    return m.group("pre") or m.group("suf")


def strip_unit(value: str, unit: str) -> Optional[str]:
    """
    The number part of `value` when it carries `unit` (or no unit at all), else None.
    Used to read a unit column as numbers; the unit itself is kept in the Decision.
    """
    m = _RE_UNIT_VALUE.match(value.strip())
    if m is None:
        return None
    found = m.group("pre") or m.group("suf")
    if found is not None and found != unit:
        return None
    return m.group("num")


def _parse_float(v: str) -> tuple[bool, bool, Optional[str], bool]:
    """
    Parse float handling decimal comma, embedded unit, and leading-zero traps.

    Returns:
        Tuple of (parses_cleanly, is_decimal_comma, embedded_unit_symbol, has_leading_zero).
    """
    s = v.strip()
    if not s:
        return False, False, None, False

    # Check for leading zero digit string (e.g. "02138", "0012")
    if len(s) > 1 and s.isdigit() and s.startswith("0"):
        return False, False, None, True

    # A '+' code is not a number either
    if _is_plus_code(s):
        return False, False, None, False

    # Check for embedded unit trap
    unit_sym = _unit_of(s)
    if unit_sym is not None:
        return False, False, unit_sym, False

    # Standard float (e.g. 12.5, 0.123, -4.5, 1e-5)
    # Exclude digit strings starting with 0 that aren't 0.xxx or 0
    if _RE_FLOAT_LZ.match(s):
        return False, False, None, True

    if _RE_FLOAT.match(s):
        return True, False, None, False

    # Float with thousands comma separators: 1,000.50
    if _RE_FLOAT_THOUSANDS.match(s):
        return True, False, None, False

    # Decimal comma (European format): 12,5 or 1.234,56
    if _RE_DEC_COMMA.match(s):
        return True, True, None, False
    if _RE_DEC_COMMA_THOUSANDS.match(s):
        return True, True, None, False

    return False, False, None, False



def _check_date_ambiguity(values: Sequence[str]) -> bool:
    """
    Check if date strings formatted as A/B/YYYY or A-B-YYYY have dd/mm ambiguity.

    If across all slash/dash dates in the column both parts are <= 12,
    it is impossible to determine day vs. month order without locale.
    """
    slash_dates = []
    for v in values:
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", v.strip())
        if m:
            p1, p2 = int(m.group(1)), int(m.group(2))
            slash_dates.append((p1, p2))

    if not slash_dates:
        return False

    # If any date has p1 > 12, p1 must be day (dd/mm format)
    # If any date has p2 > 12, p2 must be day (mm/dd format)
    has_p1_gt_12 = any(p1 > 12 for p1, p2 in slash_dates)
    has_p2_gt_12 = any(p2 > 12 for p1, p2 in slash_dates)

    if has_p1_gt_12 or has_p2_gt_12:
        return False  # Disambiguated by data!

    # All dates have both parts <= 12 -> ambiguous!
    return True


def _parse_datetime(v: str) -> bool:
    s = v.strip()
    if not s:
        return False
    return any(p.match(s) for p in _DATE_COMPILED)


# ---------------------------------------------------------------------------
# Parse results for a whole column (each distinct value parsed once)
# ---------------------------------------------------------------------------

# Which mask proves a value parses as the winning type (used to verify the winner).
_VERIFY_MASK = {"int": "int", "float": "float", "datetime": "datetime", "boolean": "bool"}

_MASK_KEYS = ("int", "int_lz", "float", "float_lz", "dec_comma", "datetime", "bool", "plus_code")


def _classify(v: str) -> tuple[tuple[bool, ...], Optional[str]]:
    p_int, int_lz = _parse_int(v)
    p_flt, dec_comma, unit, flt_lz = _parse_float(v)
    flags = (p_int, int_lz, p_flt, flt_lz, p_flt and dec_comma, _parse_datetime(v), _parse_bool(v),
             _is_plus_code(v.strip()))
    return flags, unit


def _value_masks(values: Sequence[str]) -> dict[str, np.ndarray]:
    """
    For every value, whether it parses as int, float, datetime or boolean, plus the trap
    flags (leading zero, decimal comma, embedded unit). Each distinct value is parsed
    once with the _parse_* functions and the result is spread back to every row.
    """
    codes, uniques = pd.factorize(pd.Series(list(values), dtype=object), sort=False)
    parsed = [_classify(u) for u in uniques]
    flags = np.array([f for f, _ in parsed], dtype=bool).reshape(len(parsed), len(_MASK_KEYS))
    units = np.array([u for _, u in parsed] + [None], dtype=object)[:-1]
    masks: dict[str, np.ndarray] = {k: flags[codes, i] for i, k in enumerate(_MASK_KEYS)}
    masks["unit"] = units[codes]
    return masks


def _sample_positions(n: int) -> Optional[np.ndarray]:
    """Sorted, seeded sample of row positions, or None when the column is small enough to score whole."""
    if n <= TYPE_SAMPLE_SIZE:
        return None
    rng = np.random.default_rng(TYPE_SAMPLE_SEED)
    return np.sort(rng.choice(n, size=TYPE_SAMPLE_SIZE, replace=False))


def _first_failures(values: Sequence[str], ok: np.ndarray, limit: int = 5) -> list[str]:
    """The first `limit` values (in column order) that do not parse."""
    return [values[i] for i in np.flatnonzero(~ok)[:limit]]


def _unit_counts(unit: np.ndarray) -> dict[str, int]:
    """Count of each embedded unit, keyed in order of first appearance in the column."""
    counts: dict[str, int] = {}
    for u in unit:
        if u is not None:
            counts[u] = counts.get(u, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main Type Inference Logic
# ---------------------------------------------------------------------------

def infer_column_type(
    values: Union[pd.Series, Sequence[Any]],
    column_name: str = "",
    margin_threshold: float = 0.15,
) -> TypeInferenceResult:
    """
    Infer the semantic type of a column with confidence, anomaly capture, and abstain logic.

    1. Detects disguised missing values and filters them out.
    2. Scores candidate types on remaining non-missing values.
    3. Handles known traps: leading zeros, decimal commas, ambiguous dates, embedded units.
    4. Evaluates winning score margin; abstains if margin is insufficient or trap is unresolved.
    5. Emits a versioned Decision record.
    """
    if isinstance(values, pd.Series):
        raw_list = [str(x) if x is not None else "" for x in values.tolist()]
    else:
        raw_list = [str(x) if x is not None else "" for x in values]

    col_name_lower = column_name.strip().lower()

    # Step 1: Detect missing-value markers
    missing_markers = detect_missing_markers(raw_list, column_name=column_name)
    missing_tokens: set[str] = {m.marker for m in missing_markers}

    # Extract non-missing values
    non_missing = [v for v in raw_list if v not in missing_tokens and v.strip() != ""]
    total_non_missing = len(non_missing)

    anomalies: list[str] = []
    for m in missing_markers:
        if m.reason != "empty_string":
            anomalies.append(f"disguised_missing: {m.marker}")

    # Edge Case: Column is entirely empty or missing
    if total_non_missing == 0:
        evidence = {
            "candidate_scores": {"constant": 1.0},
            "non_missing_count": 0,
            "total_rows": len(raw_list),
            "margin": 1.0,
        }
        dec = Decision(
            id="",
            stage="infer",
            kind="type_inference",
            columns=[column_name],
            action="keep_as_constant",
            evidence=evidence,
            confidence=1.0,
            default="keep_as_string",
            needs_attention=False,
            plain_language=f"Column '{column_name}' contains only empty/missing values. Classified as constant.",
            status="automatic",
        )
        return TypeInferenceResult(
            column_name=column_name,
            inferred_type="constant",
            confidence=1.0,
            runner_up=None,
            abstained=False,
            needs_attention=False,
            anomalies=anomalies,
            missing_markers=missing_markers,
            failed_samples=[],
            evidence=evidence,
            decision=dec,
        )

    # Step 2: Score candidate types
    unique_vals = set(non_missing)
    uniqueness_ratio = len(unique_vals) / total_non_missing

    # Per-value parse results for the WHOLE column, computed with vectorized string
    # operations (same rules as the _parse_* functions above). Candidate scores come
    # from a seeded sample of these rows; trap counts, anomalies and failed samples
    # always come from the whole column, so nothing is found only in the sample.
    masks = _value_masks(non_missing)
    sample_idx = _sample_positions(total_non_missing)
    sampled = sample_idx is not None
    n_scored = len(sample_idx) if sample_idx is not None else total_non_missing

    def _rate(mask: Any) -> int:
        return int(mask[sample_idx].sum()) if sample_idx is not None else int(mask.sum())

    # A. Constant Score
    const_score = 1.0 if len(unique_vals) <= 1 else 0.0

    # B. Boolean Score
    unique_lower = {v.strip().lower() for v in unique_vals}
    is_binary_pair = any(unique_lower.issubset(pair) and len(unique_lower) > 0 for pair in BOOL_LITERAL_PAIRS)
    bool_score = _rate(masks["bool"]) / n_scored if is_binary_pair else 0.0

    # C. Int Score & Traps
    int_score = _rate(masks["int"]) / n_scored
    failed_int_samples = _first_failures(non_missing, masks["int"])

    # The leading-zero threshold is checked on the int parser's count alone
    leading_zero_count = int(masks["int_lz"].sum())
    if leading_zero_count >= max(2, int(0.02 * total_non_missing)):
        anomalies.append("leading_zero")
    plus_code_count = int(masks["plus_code"].sum())
    if plus_code_count >= max(2, int(0.02 * total_non_missing)):
        anomalies.append("plus_sign_code")

    # D. Float Score & Traps
    float_score = _rate(masks["float"]) / n_scored
    failed_float_samples = _first_failures(non_missing, masks["float"])
    decimal_comma_count = int(masks["dec_comma"].sum())
    decimal_comma_detected = decimal_comma_count >= max(2, int(0.1 * total_non_missing))

    # Check for embedded units trap (units in order of first appearance in the column)
    has_embedded_unit = False
    detected_unit = None
    unit_counts = _unit_counts(masks["unit"])
    for u, count in unit_counts.items():
        if count >= max(UNIT_MIN_REPEATS, int(UNIT_FLAG_SHARE * total_non_missing)):
            anomalies.append(f"embedded_unit: {u}")
            has_embedded_unit = True
            detected_unit = u
            break

    # A unit column: one unit on most values and every value a number with that unit
    # (or a bare number). It can be read as numbers with the unit kept in the Decision.
    unit_column_type: Optional[str] = None
    if has_embedded_unit:
        dominant, dom_count = max(unit_counts.items(), key=lambda kv: kv[1])
        with_dominant = masks["unit"] == dominant
        bare_number = masks["int"] | masks["float"]
        if dom_count >= UNIT_COLUMN_SHARE * total_non_missing and bool((with_dominant | bare_number).all()):
            detected_unit = dominant
            if f"embedded_unit: {dominant}" not in anomalies:
                anomalies = [a for a in anomalies if not a.startswith("embedded_unit:")]
                anomalies.append(f"embedded_unit: {dominant}")
            numbers = [strip_unit(v, dominant) or "" for v in non_missing]
            unit_column_type = "float" if any("." in x for x in numbers) else "int"

    # E. Datetime Score & Traps
    dt_score = _rate(masks["datetime"]) / n_scored
    failed_dt_samples = _first_failures(non_missing, masks["datetime"])
    has_ambiguous_date = False
    if dt_score >= 0.70:
        if _check_date_ambiguity(non_missing):
            anomalies.append("ambiguous_date_format")
            has_ambiguous_date = True
    # F. ID Score
    is_id_name = any(
        re.search(rf"\b{p}\b", col_name_lower) or col_name_lower == p or col_name_lower.endswith(p)
        for p in ID_NAME_PATTERNS
    )
    is_sequential_int = False
    if int_score >= 0.95 and uniqueness_ratio >= 0.95 and total_non_missing >= 5:
        try:
            ints = [int(v.replace(",", "")) for v in non_missing[:100]]
            if sorted(ints) == ints and len(set(ints)) == len(ints):
                is_sequential_int = True
        except ValueError:  # Swallowed: a value that is not a plain integer means the column is not a sequence
            pass

    has_leading_zeros = "leading_zero" in anomalies
    has_plus_codes = "plus_sign_code" in anomalies
    if has_leading_zeros or has_plus_codes:
        id_score = 0.98 if uniqueness_ratio >= 0.8 else 0.85
    elif is_id_name and (uniqueness_ratio >= 0.85 or is_sequential_int):
        id_score = 0.98
    elif is_sequential_int and uniqueness_ratio == 1.0:
        id_score = 0.95
    else:
        id_score = 0.0

    # G. Text vs Categorical Score
    # Count textual (non-numeric, non-date, non-bool) tokens
    textual = ~(masks["int"] | masks["float"] | masks["bool"] | masks["datetime"])
    textual_rate = _rate(textual) / n_scored

    avg_len = sum(len(v) for v in non_missing) / total_non_missing
    avg_words = sum(len(v.split()) for v in non_missing) / total_non_missing
    is_text_domain = any(
        term in col_name_lower
        for term in ["description", "comment", "review", "summary", "text", "abstract", "question", "title", "name"]
    )

    if textual_rate > 0.0:
        if avg_len > 40 or (avg_len > 25 and avg_words >= 5) or (is_text_domain and avg_words >= TEXT_NAME_HINT_MIN_WORDS):
            text_score = 0.95 * textual_rate
            cat_score = 0.20 * textual_rate
        else:
            cat_score = 0.95 * textual_rate
            text_score = 0.20 * textual_rate
    else:
        cat_score = 0.05
        text_score = 0.05

    # Disambiguation: binary 1/0 is boolean, not integer
    if is_binary_pair and unique_lower in [{"1", "0"}, {"0", "1"}]:
        bool_score = 1.0
        int_score = 0.70  # Prioritize boolean for 2-state 0/1 indicator flags

    # Disambiguation: leading zeros and '+' codes reject integer and float coercion
    if has_leading_zeros or has_plus_codes:
        int_score = 0.0
        float_score = 0.0
        id_score = max(id_score, 0.98)



    # Assemble candidate scores
    scores: dict[str, float] = {
        "constant": const_score,
        "boolean": bool_score,
        "ID": id_score,
        "int": int_score,
        "float": float_score,
        "datetime": dt_score,
        "categorical": cat_score,
        "text": text_score,
    }

    # Step 3: Trap Handling & Abstain Triggers
    unresolved_trap: Optional[str] = None
    if has_ambiguous_date:
        unresolved_trap = "ambiguous_date_format"
    elif has_embedded_unit:
        unresolved_trap = f"embedded_unit: {detected_unit}"

    # Rank candidate types by score descending
    # Special hierarchy: if int is 1.0 and float is 1.0, int is the winner
    sorted_candidates = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cand, top_score = sorted_candidates[0]

    # Find runner up from a different semantic family
    def _is_same_family(t1: str, t2: str) -> bool:
        if t1 in ("int", "float") and t2 in ("int", "float"):
            return True
        if t1 in ("text", "categorical") and t2 in ("text", "categorical"):
            return True
        return False

    runner_cand = None
    runner_score = 0.0
    for cand, score in sorted_candidates[1:]:
        if not _is_same_family(top_cand, cand):
            runner_cand = cand
            runner_score = score
            break

    if runner_cand is None and len(sorted_candidates) > 1:
        runner_cand = sorted_candidates[1][0]
        runner_score = sorted_candidates[1][1]

    margin = top_score - runner_score

    # Step 3b: Verify the winning type against the WHOLE column. With a sample, any value
    # that does not parse as the winner is recorded as an anomaly, so a rare bad value
    # outside the sample is never lost. Without a sample the scores already cover
    # every value (failed values show in the score and in failed_samples).
    verified_failures: Optional[int] = None
    if top_cand in _VERIFY_MASK:
        verified_failures = int((~masks[_VERIFY_MASK[top_cand]]).sum())
        if sampled and verified_failures > 0:
            anomalies.append(f"verification_failed: {verified_failures} values are not {top_cand}")

    # Step 4: Decision & Abstain Evaluation
    abstained = False
    needs_attention = False
    inferred_type = top_cand

    if unresolved_trap == "ambiguous_date_format":
        # DD/MM vs MM/DD ambiguity cannot be resolved without locale
        abstained = True
        needs_attention = True
        inferred_type = "text"
        action = "keep_as_string"
        default_act = "keep_as_string"
        confidence = round(top_score, 4)
        plain_lang = (
            f"Abstained from inferring datetime for column '{column_name}' due to "
            f"unresolved DD/MM vs MM/DD ambiguity across all date values. Defaulted to string."
        )
    elif unresolved_trap and unresolved_trap.startswith("embedded_unit") and unit_column_type is not None:
        # Every value is a number with the same unit: read as numbers, keep the unit in
        # this Decision, and flag it (the text form is lost) so a person can check.
        abstained = False
        needs_attention = True
        inferred_type = unit_column_type
        action = f"cast_to_{unit_column_type}"
        default_act = action
        confidence = 0.90
        plain_lang = (
            f"Column '{column_name}' holds numbers with the unit '{detected_unit}'. Read as numbers; "
            f"the unit '{detected_unit}' is kept in this record."
        )
    elif unresolved_trap and unresolved_trap.startswith("embedded_unit"):
        # Units that are not the same on every value would be lost if coerced
        abstained = True
        needs_attention = True
        inferred_type = "text"
        action = "keep_as_string"
        default_act = "keep_as_string"
        confidence = 0.90
        plain_lang = (
            f"Detected embedded units ({detected_unit}) in column '{column_name}'. Coercion to "
            f"numeric would be lossy without explicit user confirmation. Abstained and kept as string."
        )
    elif top_score < 0.60 or (margin < margin_threshold and top_cand in ("text", "categorical", "numeric")):
        # Low confidence or narrow margin between distinct type families
        abstained = True
        needs_attention = True
        inferred_type = "text"
        action = "keep_as_string"
        default_act = "keep_as_string"
        confidence = round(top_score, 4)
        plain_lang = (
            f"Abstained from type coercion for column '{column_name}' due to close margin "
            f"({top_cand}: {top_score:.2f} vs {runner_cand}: {runner_score:.2f}, margin: {margin:.2f}). "
            f"Kept as safe string default."
        )
    else:
        # Clear winner
        inferred_type = top_cand
        confidence = round(top_score, 4)
        needs_attention = bool(anomalies) or (confidence < 0.90) or (margin < 0.25)
        if inferred_type in ("int", "float", "boolean", "datetime"):
            action = f"cast_to_{inferred_type}"
        elif inferred_type == "ID":
            action = "keep_as_id"
        elif inferred_type == "constant":
            action = "keep_as_constant"
        else:
            action = f"keep_as_{inferred_type}"
        # The safe default is plain text whenever the TYPE itself is in doubt. A column
        # flagged only because some values are missing-value placeholders (like '?' or
        # -999) has a clear type, so its default (and the recommended answer) is that
        # type. Unflagged decisions keep the text default; it is never used for them.
        type_in_doubt = (
            confidence < 0.90
            or margin < 0.25
            or any(not a.startswith("disguised_missing:") for a in anomalies)
        )
        default_act = action if (needs_attention and not type_in_doubt) else "keep_as_string"
        plain_lang = (
            f"Inferred column '{column_name}' as {inferred_type} with confidence {confidence:.2f} "
            f"(margin: {margin:.2f} over {runner_cand})."
        )

    # Failed samples for winner
    if inferred_type == "int":
        failed_samples = failed_int_samples
    elif inferred_type == "float":
        failed_samples = failed_float_samples
    elif inferred_type == "datetime":
        failed_samples = failed_dt_samples
    else:
        failed_samples = []

    evidence = {
        "candidate_scores": {k: round(v, 4) for k, v in scores.items()},
        "top_score": round(top_score, 4),
        "runner_score": round(runner_score, 4),
        "margin": round(margin, 4),
        "non_missing_count": total_non_missing,
        "total_rows": len(raw_list),
        "decimal_comma_detected": decimal_comma_detected,
        "leading_zero_detected": has_leading_zeros,
        "ambiguous_date_detected": has_ambiguous_date,
        "embedded_unit_detected": detected_unit,
        "unresolved_trap": unresolved_trap,
    }
    if has_plus_codes:
        evidence["plus_sign_code_detected"] = True
    if unit_column_type is not None:
        evidence["unit_column"] = True
        evidence["lossy_coercion"] = True
        failed_samples = []
    if sampled:
        evidence["scored_on_sample"] = n_scored
        evidence["verified_failures"] = verified_failures

    # Construct Decision record
    kind = "type_abstain" if abstained else "type_inference"
    decision = Decision(
        id="",
        stage="infer",
        kind=kind,
        columns=[column_name],
        action=action,
        evidence=evidence,
        confidence=confidence,
        default=default_act,
        needs_attention=needs_attention,
        plain_language=plain_lang,
        status="automatic",
    )

    return TypeInferenceResult(
        column_name=column_name,
        inferred_type=inferred_type,
        confidence=confidence,
        runner_up=runner_cand if margin < 0.30 else None,
        abstained=abstained,
        needs_attention=needs_attention,
        anomalies=sorted(list(set(anomalies))),
        missing_markers=missing_markers,
        failed_samples=failed_samples,
        evidence=evidence,
        decision=decision,
    )


def infer_table_types(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
) -> TableSchema:
    """
    Run type inference across all columns of a DataFrame and populate TableSchema.

    Args:
        df: Raw DataFrame containing cell strings.
        schema: Optional existing TableSchema from ingest. If omitted, builds fresh schema.

    Returns:
        Updated TableSchema with inferred column types, missing markers, and emitted decisions.
    """
    if schema is None:
        cols_meta = [
            ColumnRawMetadata(name=str(col), position=idx)
            for idx, col in enumerate(df.columns)
        ]
        res_schema = TableSchema(
            row_count=len(df),
            column_count=len(df.columns),
            columns=cols_meta,
        )
    else:
        res_schema = schema

    col_meta_by_name = {c.name: c for c in res_schema.columns}
    col_meta_by_idx = {c.position: c for c in res_schema.columns}

    new_decisions: list[Decision] = []

    for idx, col_name in enumerate(df.columns):
        col_series = df[col_name]
        res = infer_column_type(col_series, column_name=str(col_name))

        # Retrieve target column schema
        c_meta = col_meta_by_name.get(str(col_name)) or col_meta_by_idx.get(idx)
        if c_meta is not None:
            c_meta.inferred_type = res.inferred_type
            c_meta.confidence = res.confidence
            c_meta.missing_markers = [m.to_dict() for m in res.missing_markers]
            c_meta.runner_up = res.runner_up
            c_meta.abstained = res.abstained
            c_meta.anomalies = res.anomalies
            c_meta.failed_samples = res.failed_samples
            c_meta.evidence = res.evidence

        new_decisions.append(res.decision)

    res_schema.decisions.extend(new_decisions)
    return res_schema
