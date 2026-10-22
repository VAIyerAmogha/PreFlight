"""
Effect previews for column questions in the PreFlight v2.0 interaction layer.

A preview shows a few real values from the file and what each answer option would
turn them into. Conversion rules mirror TypingStage.apply in preflight.plan so the
preview and the real run agree.

Preview shape:
    {"before": ["01", "02", "03"],
     "after": {"<option label>": ["1", "2", "3"], ...},
     "note": "..."}          # only when something needs explaining
"""

from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import pandas as pd
from dateutil import parser as date_parser

BLANK = "(left blank)"
PREVIEW_SAMPLES = 3

Converter = Callable[[str], Optional[str]]


def converter_for(action: str, evidence: Dict[str, Any]) -> Optional[Converter]:
    """
    Return a function str -> converted text (or None if the value cannot be
    converted) for a conversion action, or None if the action is not a conversion.
    """
    dec_comma = bool((evidence or {}).get("decimal_comma_detected", False))

    if action == "cast_to_int":

        def to_int(s: str) -> Optional[str]:
            try:
                return str(int(s.replace(",", "")))
            except (ValueError, TypeError):
                return None

        return to_int

    if action == "cast_to_float":

        def to_float(s: str) -> Optional[str]:
            try:
                t = s
                if dec_comma:
                    t = t.replace(".", "").replace(",", ".") if ("." in t and "," in t) else t.replace(",", ".")
                else:
                    t = t.replace(",", "")
                return str(float(t))
            except (ValueError, TypeError):
                return None

        return to_float

    if action == "cast_to_boolean":

        def to_bool(s: str) -> Optional[str]:
            low = s.lower()
            if low in ("true", "yes", "y", "t", "1"):
                return "yes"
            if low in ("false", "no", "n", "f", "0"):
                return "no"
            return None

        return to_bool

    if action == "cast_to_datetime":

        def to_dt(s: str) -> Optional[str]:
            try:
                dt = date_parser.parse(s)
            except Exception:
                return None
            if isinstance(dt, datetime):
                return dt.strftime("%Y-%m-%d") if (dt.hour, dt.minute, dt.second) == (0, 0, 0) else dt.isoformat(" ")
            return str(dt)

        return to_dt

    return None


def after_value(value: str, action: str, evidence: Dict[str, Any]) -> str:
    """What a single raw value becomes under an action."""
    conv = converter_for(action, evidence)
    if conv is None:
        return value
    out = conv(value)
    return BLANK if out is None else out


def pick_samples(
    series: Optional[pd.Series],
    missing: Iterable[str] = (),
    prefer_failing: Optional[Callable[[str], bool]] = None,
    n: int = PREVIEW_SAMPLES,
) -> List[str]:
    """
    Pick up to n distinct real values from a raw column, skipping empty cells and
    known missing-value markers. If `prefer_failing` is given, one value for which it
    returns True is included (so the preview shows what goes wrong, not just what works).
    """
    if series is None:
        return []
    skip: Set[str] = {m for m in missing} | {""}
    seen: List[str] = []
    for v in series.astype(str).tolist():
        v = v.strip() if isinstance(v, str) else v
        if v in skip or v in seen:
            continue
        seen.append(v)
        if len(seen) >= 200:  # enough to find a failing example without scanning huge columns
            break
    if not seen:
        return []
    picked = seen[:n]
    if prefer_failing is not None and not any(prefer_failing(v) for v in picked):
        failing = next((v for v in seen[n:] if prefer_failing(v)), None)
        if failing is not None:
            picked = picked[: n - 1] + [failing]
    return picked


def build_column_preview(
    series: Optional[pd.Series],
    option_actions: Dict[str, str],
    evidence: Dict[str, Any],
    missing: Iterable[str] = (),
    fallback_values: Optional[List[Any]] = None,
    action_effects: Optional[Dict[str, Callable[[str], str]]] = None,
) -> Dict[str, Any]:
    """
    Build a before/after preview for a column question.

    option_actions maps each option label to the action it would apply. Values come
    from the column itself; if the column is unavailable, from evidence samples the
    pipeline already recorded. Nothing is invented: with no values at all, the
    preview says so in `note`.
    """
    effects = action_effects or {}

    def changes(v: str) -> bool:
        return any(
            (c := converter_for(a, evidence)) is not None and c(v) is None
            for a in option_actions.values()
        )

    before = pick_samples(series, missing, prefer_failing=changes)
    note: Optional[str] = None
    if not before and fallback_values:
        before = [str(x) for x in fallback_values if str(x).strip()][:PREVIEW_SAMPLES]
        note = "Sample values come from the earlier scan."
    if not before:
        return {
            "before": [],
            "after": {label: [] for label in option_actions},
            "note": "No non-empty values were available to show.",
        }

    after: Dict[str, List[str]] = {}
    for label, action in option_actions.items():
        fx = effects.get(action)
        after[label] = [fx(v) if fx is not None else after_value(v, action, evidence) for v in before]
    preview: Dict[str, Any] = {"before": before, "after": after}
    if any(BLANK in vals for vals in after.values()):
        extra = f"'{BLANK}' means that value cannot be converted and is recorded as a problem value."
        note = f"{note} {extra}" if note else extra
    if note:
        preview["note"] = note
    return preview
