"""
Plain-language sentence templates for PreFlight v2.0 Decision records.

Generates human-readable descriptions, questions, selectable options, and
safe defaults for every pipeline Decision kind.

Design Rules:
- Plain language only: no technical jargon (no "schema", "delimiter", "dtype",
  "coercion", "parse", "imputation", "winsorize", "cardinality", "heuristic",
  and no raw encoding names like "utf-8").
- All templates accept only structured fields (column names, counts, example values),
  never raw unvetted free text from internal diagnostic dumps.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence

from preflight.decisions import Decision


# Banned jargon terms that must never appear in user-facing sentences or PDFs
BANNED_JARGON = [
    "imputation",
    "impute",
    "imputing",
    "winsorize",
    "winsorizing",
    "winsorization",
    "cardinality",
    "dtype",
    "dtypes",
    "coercion",
    "coerce",
    "coerced",
    "coercing",
    "heuristic",
    "heuristics",
    "parse",
    "parsing",
    "parser",
    "parsed",
    "schema",
    "schemas",
    "delimiter",
    "delimiters",
    "utf-8",
    "utf8",
    "latin-1",
    "windows-1252",
    "iso-8859-1",
]


@dataclass(frozen=True)
class RenderedDecisionText:
    """Structured plain-language rendered text for a pipeline Decision."""
    summary: str
    question: Optional[str] = None
    options: List[str] = field(default_factory=list)
    recommended_default: Optional[str] = None
    explanation: Optional[str] = None


# ---------------------------------------------------------------------------
# Friendly Term Mappings (Plain Equivalents)
# ---------------------------------------------------------------------------

def friendly_text_format(enc_name: Optional[str]) -> str:
    """Map character encoding identifiers to user-friendly plain descriptions."""
    if not enc_name:
        return "standard text format"
    norm = enc_name.lower().replace("_", "-").strip()
    if norm in ("utf-8", "utf8", "utf-8-sig"):
        return "standard text format"
    if norm in ("windows-1252", "cp1252", "iso-8859-1", "latin-1", "iso8859-1"):
        return "Western European text format"
    if norm == "ascii":
        return "basic text format"
    if norm in ("utf-16", "utf-16-le", "utf-16-be"):
        return "wide character text format"
    return "standard text format"


def friendly_separator_name(sep_char: Optional[str]) -> str:
    """Map delimiter characters to plain-language names."""
    if not sep_char:
        return "comma"
    if sep_char == ",":
        return "comma"
    if sep_char == ";":
        return "semicolon"
    if sep_char == "\t":
        return "tab"
    if sep_char == "|":
        return "pipe"
    if sep_char == " ":
        return "space"
    return "symbol"


def friendly_type_name(inferred_type: Optional[str]) -> str:
    """Map type keywords to non-jargon plain descriptions."""
    if not inferred_type:
        return "plain text"
    t = inferred_type.lower()
    if t in ("int", "integer"):
        return "whole numbers"
    if t in ("float", "numeric", "number"):
        return "decimal numbers"
    if t in ("bool", "boolean"):
        return "yes/no values"
    if t in ("datetime", "date", "timestamp"):
        return "dates and times"
    if t in ("categorical", "category"):
        return "categories"
    if t in ("text", "string"):
        return "plain text"
    if t == "id":
        return "identifiers"
    if t == "constant":
        return "constant values"
    return "plain text"


def friendly_action_name(action: str) -> str:
    """Map raw code actions to human-friendly descriptions."""
    if not action:
        return "Preserve plain text"
    act = action.lower()
    if act == "keep_as_string":
        return "Keep original plain text"
    if act == "keep_as_constant":
        return "Keep constant value"
    if act == "keep_as_id":
        return "Keep identifier text"
    if act in ("keep_as_categorical", "cast_to_categorical"):
        return "Convert to categories"
    if act == "cast_to_int":
        return "Convert to whole numbers"
    if act == "cast_to_float":
        return "Convert to decimal numbers"
    if act == "cast_to_boolean":
        return "Convert to yes/no values"
    if act == "cast_to_datetime":
        return "Convert to dates and times"
    if act.startswith("use_encoding_"):
        return f"Use {friendly_text_format(act.replace('use_encoding_', ''))}"
    if act.startswith("separate_by_"):
        return f"Separate by {friendly_separator_name(act.replace('separate_by_', ''))}"
    if act.startswith("use_header_row_"):
        return f"Use row {int(act.replace('use_header_row_', '')) + 1} as column names"
    if act == "no_header":
        return "No title row in file"
    if act == "classification":
        return "Predict categories (classification)"
    if act == "regression":
        return "Predict numerical values (regression)"
    return act


def _format_examples(examples: Optional[Sequence[Any]], max_items: int = 3) -> str:
    """Format sample values safely without technical syntax."""
    if not examples:
        return ""
    clean = [str(x).strip() for x in examples if str(x).strip()]
    if not clean:
        return ""
    sample = clean[:max_items]
    return f" (examples: {', '.join(sample)})"


# ---------------------------------------------------------------------------
# Per-Kind Plain-Language Templates
# ---------------------------------------------------------------------------

def template_encoding(
    text_format_name: str,
    confidence: float = 1.0,
    has_bom: bool = False,
    needs_attention: bool = False,
) -> RenderedDecisionText:
    """Plain-language template for text format (encoding) decisions."""
    friendly = friendly_text_format(text_format_name)
    summary = f"Read the file using {friendly}."

    if needs_attention or confidence < 0.80:
        question = "Which text format should be used to read this file?"
        options = [
            f"Use {friendly} (recommended)",
            "Use Western European text format",
            "Use standard text format",
        ]
        recommended = f"Use {friendly} (recommended)"
        explanation = (
            f"The file text format was identified as {friendly} with moderate confidence. "
            "Confirm if special characters appear correctly."
        )
        return RenderedDecisionText(
            summary=summary,
            question=question,
            options=options,
            recommended_default=recommended,
            explanation=explanation,
        )

    return RenderedDecisionText(summary=summary)


def template_dialect(
    separator_char: str,
    confidence: float = 1.0,
    needs_attention: bool = False,
) -> RenderedDecisionText:
    """Plain-language template for column separator (dialect) decisions."""
    friendly = friendly_separator_name(separator_char)
    summary = f"Separate columns using {friendly} separators."

    if needs_attention or confidence < 0.80:
        question = "Which separator splits columns into fields?"
        options = [
            f"Separate columns by {friendly} (recommended)",
            "Separate columns by semicolon",
            "Separate columns by tab",
            "Separate columns by comma",
        ]
        recommended = f"Separate columns by {friendly} (recommended)"
        explanation = (
            f"Multiple candidate column separators were detected. "
            f"Using {friendly} provided the most consistent column structure."
        )
        return RenderedDecisionText(
            summary=summary,
            question=question,
            options=options,
            recommended_default=recommended,
            explanation=explanation,
        )

    return RenderedDecisionText(summary=summary)


def template_header(
    header_row_index: Optional[int],
    title_rows_count: int = 0,
    is_ragged: bool = False,
    needs_attention: bool = False,
) -> RenderedDecisionText:
    """Plain-language template for table header and boundary decisions."""
    if header_row_index is not None:
        row_num = header_row_index + 1
        if title_rows_count > 0:
            summary = (
                f"Skip {title_rows_count} title row(s) and use row {row_num} for column names."
            )
        else:
            summary = f"Use row {row_num} as column names."
    else:
        summary = "No column names row found; assign automatic column names."

    if needs_attention or is_ragged:
        question = "Which row contains the column names?"
        options = [
            "Use row 1 as column names",
            "No column names row in file",
            "Skip leading rows before reading column names",
        ]
        recommended = options[0] if header_row_index is not None else options[1]
        explanation = (
            "Table row lengths vary across the file or the header row was ambiguous."
        )
        return RenderedDecisionText(
            summary=summary,
            question=question,
            options=options,
            recommended_default=recommended,
            explanation=explanation,
        )

    return RenderedDecisionText(summary=summary)


def template_type(
    column_name: str,
    inferred_type: str,
    example_values: Optional[Sequence[Any]] = None,
) -> RenderedDecisionText:
    """Plain-language template for standard column type decisions."""
    friendly = friendly_type_name(inferred_type)
    eg_str = _format_examples(example_values)
    summary = f"Read column '{column_name}' as {friendly}{eg_str}."
    return RenderedDecisionText(summary=summary)


def template_missing_marker(
    column_name: str,
    marker_value: str,
    count: int = 1,
    needs_attention: bool = False,
) -> RenderedDecisionText:
    """Plain-language template for detected missing-value markers."""
    occ_str = f"{count} occurrence" if count == 1 else f"{count} occurrences"
    summary = f"Treat '{marker_value}' as missing data in column '{column_name}' ({occ_str})."

    if needs_attention:
        question = f"Should '{marker_value}' in column '{column_name}' be treated as missing data?"
        options = [
            f"Treat '{marker_value}' as missing data (recommended)",
            f"Keep '{marker_value}' as a valid value",
        ]
        recommended = options[0]
        explanation = (
            f"The value '{marker_value}' looks like a placeholder for missing information "
            f"rather than a regular measurement."
        )
        return RenderedDecisionText(
            summary=summary,
            question=question,
            options=options,
            recommended_default=recommended,
            explanation=explanation,
        )

    return RenderedDecisionText(summary=summary)


def template_abstention(
    column_name: str,
    reason_description: str = "",
    example_values: Optional[Sequence[Any]] = None,
) -> RenderedDecisionText:
    """Plain-language template for column type abstentions."""
    eg_str = _format_examples(example_values)
    summary = f"Keep column '{column_name}' as plain text to preserve original content{eg_str}."
    question = f"How should column '{column_name}' be interpreted?"
    options = [
        "Keep as plain text (safest)",
        "Convert to whole numbers",
        "Convert to decimal numbers",
        "Convert to categories",
    ]
    recommended = "Keep as plain text (safest)"
    explanation = (
        f"Column '{column_name}' has mixed formats or ambiguous values. "
        "Keeping plain text prevents accidental corruption or loss of detail."
    )
    return RenderedDecisionText(
        summary=summary,
        question=question,
        options=options,
        recommended_default=recommended,
        explanation=explanation,
    )


def template_anomaly(
    column_name: str,
    anomaly_description: str,
    example_values: Optional[Sequence[Any]] = None,
    needs_attention: bool = False,
) -> RenderedDecisionText:
    """Plain-language template for detected data anomalies."""
    eg_str = _format_examples(example_values)
    summary = f"Preserve values in column '{column_name}': {anomaly_description}{eg_str}."

    if needs_attention:
        question = f"How should unusual entries in column '{column_name}' be handled?"
        options = [
            "Preserve original values without modification (recommended)",
            "Standardize values into common format",
        ]
        recommended = options[0]
        explanation = f"Values in column '{column_name}' contain {anomaly_description}."
        return RenderedDecisionText(
            summary=summary,
            question=question,
            options=options,
            recommended_default=recommended,
            explanation=explanation,
        )

    return RenderedDecisionText(summary=summary)


def template_ambiguous_task(
    column_name: str,
    unique_count: int = 0,
    example_values: Optional[Sequence[Any]] = None,
) -> RenderedDecisionText:
    """Plain-language template for ambiguous modeling task inference."""
    eg_str = _format_examples(example_values)
    summary = (
        f"Select prediction objective for target column '{column_name}' "
        f"({unique_count} distinct values){eg_str}."
    )
    question = f"What type of prediction should be made for target '{column_name}'?"
    options = [
        "Predict categories (classification)",
        "Predict numerical values (regression)",
        "No prediction goal (data inspection only)",
    ]
    # If 2 to 10 unique values, default to classification; otherwise regression
    recommended = (
        "Predict categories (classification)"
        if 2 <= unique_count <= 10
        else "Predict numerical values (regression)"
    )
    explanation = (
        f"Target column '{column_name}' has {unique_count} distinct values. "
        "These could represent discrete groups or a numerical scale."
    )
    return RenderedDecisionText(
        summary=summary,
        question=question,
        options=options,
        recommended_default=recommended,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Dispatcher for Arbitrary Decision Records
# ---------------------------------------------------------------------------

def render_decision_text(decision: Decision) -> RenderedDecisionText:
    """
    Render a Decision dataclass into verified plain-language text using structured fields.

    Extracts strictly structured fields from decision attributes and evidence,
    dispatching to the corresponding per-kind template.
    """
    kind = decision.kind.lower().replace("-", "_")
    col_name = decision.columns[0] if decision.columns else "table"
    ev = decision.evidence or {}

    if kind in ("encoding", "text_format"):
        enc_name = (
            decision.action[len("use_encoding_"):]
            if decision.action.startswith("use_encoding_")
            else str(ev.get("encoding", decision.action))
        )
        return template_encoding(
            text_format_name=enc_name,
            confidence=decision.confidence,
            has_bom=bool(ev.get("has_bom", False)),
            needs_attention=decision.needs_attention,
        )

    if kind in ("dialect", "separator"):
        sep = (
            decision.action[len("separate_by_"):]
            if decision.action.startswith("separate_by_")
            else str(ev.get("delimiter", ","))
        )
        return template_dialect(
            separator_char=sep,
            confidence=decision.confidence,
            needs_attention=decision.needs_attention,
        )

    if kind in ("header", "table_boundary"):
        h_idx = ev.get("header_row_index")
        if decision.action.startswith("use_header_row_"):
            h_idx = int(decision.action[len("use_header_row_"):])
        t_count = len(ev.get("title_rows", [])) if decision.action.startswith("use_header_row_") else 0
        ragged = bool(ev.get("is_ragged", False))
        if decision.action == "no_header":
            return RenderedDecisionText(summary=template_header(None).summary)
        return template_header(
            header_row_index=h_idx if isinstance(h_idx, int) else 0,
            title_rows_count=t_count,
            is_ragged=ragged,
            needs_attention=decision.needs_attention,
        )

    if kind in ("type_abstain", "abstention", "abstain"):
        if decision.action != "keep_as_string" and not decision.needs_attention:
            return template_type(
                column_name=col_name,
                inferred_type=decision.action.replace("cast_to_", "").replace("keep_as_", ""),
                example_values=ev.get("sample_values") or [],
            )
        samples = ev.get("failed_samples") or ev.get("sample_values") or []
        reason = str(ev.get("unresolved_trap") or "uncertain format")
        return template_abstention(
            column_name=col_name,
            reason_description=reason,
            example_values=samples,
        )

    if kind in ("missing_marker", "missing_value"):
        marker = str(ev.get("marker", "?"))
        count = int(ev.get("count", 1))
        return template_missing_marker(
            column_name=col_name,
            marker_value=marker,
            count=count,
            needs_attention=decision.needs_attention,
        )

    if kind in ("anomaly", "data_anomaly"):
        desc = str(ev.get("anomaly_type") or "unusual pattern")
        samples = ev.get("sample_values", [])
        return template_anomaly(
            column_name=col_name,
            anomaly_description=desc,
            example_values=samples,
            needs_attention=decision.needs_attention,
        )

    if kind in ("ambiguous_task", "task_inference", "task"):
        u_count = int(ev.get("num_unique", 0))
        samples = ev.get("unique_samples", [])
        if decision.needs_attention:
            return template_ambiguous_task(
                column_name=col_name,
                unique_count=u_count,
                example_values=samples,
            )
        else:
            task_name = decision.action
            if task_name == "classification":
                summary = f"Predict categories for target column '{col_name}'."
            elif task_name == "regression":
                summary = f"Predict numerical values for target column '{col_name}'."
            else:
                summary = f"Target column '{col_name}' designated for inspection."
            return RenderedDecisionText(summary=summary)

    # Standard column type inference (kind == "type_inference" or "type")
    # Action typically is 'cast_to_int', 'keep_as_string', etc.
    t_name = decision.action.replace("cast_to_", "").replace("keep_as_", "")
    samples = ev.get("sample_values") or []
    return template_type(
        column_name=col_name,
        inferred_type=t_name,
        example_values=samples,
    )


# ---------------------------------------------------------------------------
# Question Phrasing (interaction layer)
# ---------------------------------------------------------------------------

def _pct(value: Any) -> str:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "an unknown share"


def describe_type_scores(evidence: Dict[str, Any]) -> str:
    """Plain-language summary of how well each reading fits a column's values."""
    scores = evidence.get("candidate_scores") or {}
    ranked = sorted(
        ((str(k).lower(), v) for k, v in scores.items() if isinstance(v, (int, float)) and v > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        return "No clear pattern was found in the values."
    parts = [f"{friendly_type_name(k)} fit {_pct(v)} of the values" for k, v in ranked[:3]]
    return "Best matches: " + "; ".join(parts) + "."


def template_type_question(
    column_name: str,
    action: str,
    evidence: Dict[str, Any],
) -> RenderedDecisionText:
    """Plain-language question for an uncertain column type decision."""
    ev = evidence or {}
    trap = str(ev.get("unresolved_trap") or "")
    reasons: List[str] = []
    if trap == "ambiguous_date_format":
        reasons.append(
            "Some dates could be read day-first or month-first, so they cannot be "
            "interpreted without guessing."
        )
    elif trap.startswith("embedded_unit"):
        unit = ev.get("embedded_unit_detected")
        unit_txt = f" ({unit})" if unit else ""
        reasons.append(
            f"Values include units{unit_txt}; turning them into numbers would lose the units."
        )
    if ev.get("leading_zero_detected") and action in ("cast_to_int", "cast_to_float"):
        reasons.append(
            "Some values start with zero (like 02138); turning them into numbers would drop the zeros."
        )
    if ev.get("decimal_comma_detected"):
        reasons.append("Numbers appear to use a comma as the decimal mark.")
    if not reasons:
        reasons.append(
            "The values fit more than one reading closely enough that a wrong guess could change them."
        )
    return RenderedDecisionText(
        summary=describe_type_scores(ev),
        question=f"How should column '{column_name}' be read?",
        explanation=" ".join(reasons),
    )


def render_question_text(decision: Decision) -> RenderedDecisionText:
    """
    Question phrasing for a decision that needs attention: `summary` holds the
    evidence in plain language, `question` the title, `explanation` the reason.
    """
    kind = decision.kind.lower().replace("-", "_")
    col_name = decision.columns[0] if decision.columns else "table"
    ev = decision.evidence or {}

    if kind in ("type_inference", "type_abstain", "type"):
        return template_type_question(col_name, decision.action, ev)

    base = render_decision_text(decision)
    if kind in ("encoding", "text_format"):
        enc = decision.action[len("use_encoding_"):] if decision.action.startswith("use_encoding_") else None
        summary = f"The file looks like {friendly_text_format(enc)}."
        if ev.get("low_confidence") or (ev.get("confidence") or 1.0) < 0.80:
            summary += " The check was not certain about this."
    elif kind in ("dialect", "separator"):
        sep = decision.action[len("separate_by_"):] if decision.action.startswith("separate_by_") else ","
        summary = f"Columns appear to be split by {friendly_separator_name(sep)}."
        if ev.get("flagged") or (ev.get("confidence") or 1.0) < 0.80:
            summary += " Another separator fit almost as well."
    elif kind in ("header", "table_boundary"):
        summary = (
            "Row lengths vary across the file."
            if ev.get("is_ragged")
            else f"The column-names row was found with {_pct(ev.get('confidence'))} certainty."
        )
    elif kind in ("ambiguous_task", "task_inference", "task"):
        n = ev.get("num_unique")
        summary = f"The target column has {n} distinct values." if n is not None else base.summary
    else:
        summary = base.summary
    return RenderedDecisionText(
        summary=summary,
        question=base.question,
        explanation=base.explanation,
    )
