"""
Type inference, confidence scoring, trap detection, and abstain logic for PreFlight v2.0.

Evaluates raw non-missing column values against 8 candidate types:
- int, float, boolean, datetime, categorical, text, ID, constant.

Explicitly handles known traps:
- Leading-zero IDs / postal codes (rejects int coercion, preserving leading zeroes)
- European decimal comma vs. decimal point and thousands separators
- Ambiguous date formats (dd/mm vs mm/dd ambiguity flagged and abstained)
- Embedded units (e.g. '15 kmpl', '$12.50' detected as text/numeric mix, flagged lossy-if-coerced, abstained)

Emits a Decision record per column with confidence scoring, safe defaults,
and abstain behavior when candidate score margins are low or traps are unresolvable.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import datetime as dt
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from preflight.decisions import Decision, compute_decision_fingerprint
from preflight.schema import ColumnRawMetadata, TableSchema

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
    anomalies: List[str]
    missing_markers: List[MissingMarker]
    failed_samples: List[str]
    evidence: Dict[str, Any]
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

UNIT_REGEXES = [
    (r"^[$€£]\s*[\d,]+(\.\d+)?$", "$"),
    (r"^[\d,]+(\.\d+)?\s*%$", "%"),
    (r"^[\d,]+(\.\d+)?\s*(?:kg|g|mg)$", "kg"),
    (r"^[\d,]+(\.\d+)?\s*(?:km|m|cm|mm)$", "km"),
    (r"^[\d,]+(\.\d+)?\s*(?:kmpl|kpl|mpg|mph|kph)$", "kmpl"),
    (r"^[\d,]+(\.\d+)?\s*(?:years|yrs|yr)$", "years"),
]

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


# ---------------------------------------------------------------------------
# Individual Candidate Type Parsers
# ---------------------------------------------------------------------------

def _parse_bool(v: str) -> bool:
    return v.strip().lower() in BOOL_ALL_LITERALS


def _parse_int(v: str) -> Tuple[bool, bool]:
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

    # Standard integer with optional sign
    if re.match(r"^[-+]?\d+$", s):
        return True, False

    # Integer with thousands comma separators (e.g. "1,000,000")
    if re.match(r"^[-+]?\d{1,3}(,\d{3})+$", s):
        return True, False

    return False, False


def _parse_float(v: str) -> Tuple[bool, bool, Optional[str], bool]:
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

    # Check for embedded unit trap
    for pat, unit_sym in UNIT_REGEXES:
        if re.match(pat, s, re.IGNORECASE):
            return False, False, unit_sym, False

    # Standard float (e.g. 12.5, 0.123, -4.5, 1e-5)
    # Exclude digit strings starting with 0 that aren't 0.xxx or 0
    if re.match(r"^[-+]?0\d+(\.\d*)?$", s):
        return False, False, None, True

    if re.match(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$", s):
        return True, False, None, False

    # Float with thousands comma separators: 1,000.50
    if re.match(r"^[-+]?\d{1,3}(,\d{3})*(\.\d+)?$", s):
        return True, False, None, False

    # Decimal comma (European format): 12,5 or 1.234,56
    if re.match(r"^[-+]?\d+(,\d+)$", s):
        return True, True, None, False
    if re.match(r"^[-+]?\d{1,3}(\.\d{3})*(,\d+)$", s):
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
    return any(re.match(p, s) for p in DATE_PATTERNS)


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
    missing_tokens: Set[str] = {m.marker for m in missing_markers}

    # Extract non-missing values
    non_missing = [v for v in raw_list if v not in missing_tokens and v.strip() != ""]
    total_non_missing = len(non_missing)

    anomalies: List[str] = []
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

    # Trap detection flags
    leading_zero_count = 0
    decimal_comma_count = 0
    embedded_unit_counts: Dict[str, int] = {}
    failed_int_samples: List[str] = []
    failed_float_samples: List[str] = []
    failed_dt_samples: List[str] = []

    # A. Constant Score
    const_score = 1.0 if len(unique_vals) <= 1 else 0.0

    # B. Boolean Score
    bool_matches = 0
    unique_lower = {v.strip().lower() for v in unique_vals}
    is_binary_pair = any(unique_lower.issubset(pair) and len(unique_lower) > 0 for pair in BOOL_LITERAL_PAIRS)

    for v in non_missing:
        if _parse_bool(v):
            bool_matches += 1
    bool_score = bool_matches / total_non_missing if is_binary_pair else 0.0

    # C. Int Score & Traps
    int_matches = 0
    for v in non_missing:
        p_int, has_lz = _parse_int(v)
        if p_int:
            int_matches += 1
        else:
            if len(failed_int_samples) < 5:
                failed_int_samples.append(v)
        if has_lz:
            leading_zero_count += 1

    int_score = int_matches / total_non_missing

    # If leading zeros detected in digit strings, record anomaly
    if leading_zero_count >= max(2, int(0.02 * total_non_missing)):
        anomalies.append("leading_zero")

    # D. Float Score & Traps
    float_matches = 0
    for v in non_missing:
        p_flt, is_dec_comma, unit_sym, flt_lz = _parse_float(v)
        if p_flt:
            float_matches += 1
            if is_dec_comma:
                decimal_comma_count += 1
        else:
            if len(failed_float_samples) < 5:
                failed_float_samples.append(v)
        if unit_sym:
            embedded_unit_counts[unit_sym] = embedded_unit_counts.get(unit_sym, 0) + 1
        if flt_lz:
            leading_zero_count += 1

    float_score = float_matches / total_non_missing
    decimal_comma_detected = decimal_comma_count >= max(2, int(0.1 * total_non_missing))

    # Check for embedded units trap
    has_embedded_unit = False
    detected_unit = None
    for u, count in embedded_unit_counts.items():
        if count >= max(2, int(0.15 * total_non_missing)):
            anomalies.append(f"embedded_unit: {u}")
            has_embedded_unit = True
            detected_unit = u
            break

    # E. Datetime Score & Traps
    dt_matches = 0
    for v in non_missing:
        if _parse_datetime(v):
            dt_matches += 1
        else:
            if len(failed_dt_samples) < 5:
                failed_dt_samples.append(v)

    dt_score = dt_matches / total_non_missing
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
        except ValueError:
            pass

    has_leading_zeros = "leading_zero" in anomalies
    if has_leading_zeros:
        id_score = 0.98 if uniqueness_ratio >= 0.8 else 0.85
    elif is_id_name and (uniqueness_ratio >= 0.85 or is_sequential_int):
        id_score = 0.98
    elif is_sequential_int and uniqueness_ratio == 1.0:
        id_score = 0.95
    else:
        id_score = 0.0

    # G. Text vs Categorical Score
    # Count textual (non-numeric, non-date, non-bool) tokens
    textual_count = 0
    for v in non_missing:
        s_clean = v.strip()
        p_i, _ = _parse_int(s_clean)
        p_f, _, _, _ = _parse_float(s_clean)
        p_b = _parse_bool(s_clean)
        p_d = _parse_datetime(s_clean)
        if not (p_i or p_f or p_b or p_d):
            textual_count += 1

    textual_rate = textual_count / total_non_missing

    avg_len = sum(len(v) for v in non_missing) / total_non_missing
    avg_words = sum(len(v.split()) for v in non_missing) / total_non_missing
    is_text_domain = any(
        term in col_name_lower
        for term in ["description", "comment", "review", "summary", "text", "abstract", "question", "title", "name"]
    )

    if textual_rate > 0.0:
        if avg_len > 40 or (avg_len > 25 and avg_words >= 5) or (is_text_domain and avg_words >= 3):
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

    # Disambiguation: leading zeros reject integer and float coercion
    if has_leading_zeros:
        int_score = 0.0
        float_score = 0.0
        id_score = max(id_score, 0.98)



    # Assemble candidate scores
    scores: Dict[str, float] = {
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
    elif unresolved_trap and unresolved_trap.startswith("embedded_unit"):
        # Embedded units (e.g. 15 kmpl) would lose units if coerced
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
        default_act = "keep_as_string"
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

    new_decisions: List[Decision] = []

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
