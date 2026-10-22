"""
Trigger registry for the PreFlight v2.0 interaction layer.

A trigger names the reason a Decision needs a human. The eight categories are:

1. low_confidence_structure  - text format, separator, or column-names row detected with low confidence
2. ambiguous_type            - close call between column types, or the type check abstained
3. destructive_action        - EMPTY PLUG-IN POINT (no Decision of this kind exists yet)
4. suspected_leakage         - EMPTY PLUG-IN POINT (no Decision of this kind exists yet)
5. disguised_missing         - values that may be placeholders for missing data
6. high_anomaly_rate         - the proposed conversion fails for a large share of values
7. lossy_coercion            - conversion would lose information (units, ambiguous dates, leading zeros)
8. ambiguous_structure_or_task - ambiguous prediction goal or ragged table structure

The category only labels *why* a question is asked. It never decides *whether* one is
asked: that is Decision.needs_attention, set upstream.

Rules are evaluated in registration order; the first match wins. Later phases fill the
empty plug-in points with `TriggerRegistry.register(...)` (see `register_trigger`).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from preflight.decisions import Decision
from preflight.schema import ColumnRawMetadata

# Share of non-missing values failing the proposed conversion above which the
# question is labelled "high anomaly rate". Labelling only: does not gate asking.
ANOMALY_RATE_THRESHOLD = 0.10


class TriggerCategory:
    """Canonical names for the 8 interaction trigger categories."""

    LOW_CONFIDENCE_STRUCTURE = "low_confidence_structure"
    AMBIGUOUS_TYPE = "ambiguous_type"
    DESTRUCTIVE_ACTION = "destructive_action"
    SUSPECTED_LEAKAGE = "suspected_leakage"
    DISGUISED_MISSING = "disguised_missing"
    HIGH_ANOMALY_RATE = "high_anomaly_rate"
    LOSSY_COERCION = "lossy_coercion"
    AMBIGUOUS_STRUCTURE_OR_TASK = "ambiguous_structure_or_task"

    ALL_CATEGORIES = [
        LOW_CONFIDENCE_STRUCTURE,
        AMBIGUOUS_TYPE,
        DESTRUCTIVE_ACTION,
        SUSPECTED_LEAKAGE,
        DISGUISED_MISSING,
        HIGH_ANOMALY_RATE,
        LOSSY_COERCION,
        AMBIGUOUS_STRUCTURE_OR_TASK,
    ]


@dataclass
class TriggerContext:
    """
    Extra evidence available when classifying a Decision.

    column: schema metadata for the Decision's column (missing-value markers etc.).
    raw_series: the column's raw string values from the file, if available.
    """

    column: Optional[ColumnRawMetadata] = None
    raw_series: Optional[pd.Series] = None


Matcher = Callable[[Decision, TriggerContext], bool]


@dataclass(frozen=True)
class TriggerRule:
    """A registered trigger: a category plus the test that recognises it."""

    category: str
    description: str
    matcher: Matcher
    implemented: bool = True


def _never(_d: Decision, _c: TriggerContext) -> bool:
    return False


def column_missing_markers(column: Optional[ColumnRawMetadata]) -> List[str]:
    """Missing-value marker strings recorded for a column."""
    if column is None:
        return []
    return [str(m.get("marker", "")) for m in column.missing_markers if isinstance(m, dict)]


def anomaly_rate(
    decision: Decision,
    series: Optional[pd.Series],
    missing: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Share of non-missing raw values that fail the Decision's proposed conversion.

    Returns None when the rate cannot be measured (no raw values, or the proposed
    action is not a conversion).
    """
    if series is None:
        return None
    from preflight.interact.preview import converter_for  # local: avoids import cycle

    conv = converter_for(decision.action, decision.evidence or {})
    if conv is None:
        return None
    skip = set(missing or []) | {""}
    values = [v for v in series.astype(str).str.strip().tolist() if v not in skip]
    if not values:
        return None
    failed = sum(1 for v in values if conv(v) is None)
    return failed / len(values)


def _is_reading(d: Decision) -> bool:
    return d.stage in ("reading", "ingest")


def _is_type(d: Decision) -> bool:
    return d.stage == "infer" and d.kind in ("type_inference", "type_abstain")


def _match_ragged_or_task(d: Decision, _c: TriggerContext) -> bool:
    if d.kind in ("ambiguous_task", "task_inference", "task"):
        return True
    return _is_reading(d) and bool((d.evidence or {}).get("is_ragged"))


def _match_low_confidence_structure(d: Decision, _c: TriggerContext) -> bool:
    return _is_reading(d)


def _match_lossy(d: Decision, _c: TriggerContext) -> bool:
    if not _is_type(d):
        return False
    ev = d.evidence or {}
    if ev.get("unresolved_trap"):
        return True
    return bool(ev.get("leading_zero_detected")) and d.action in ("cast_to_int", "cast_to_float")


def _match_disguised_missing(d: Decision, c: TriggerContext) -> bool:
    if not _is_type(d) or c.column is None:
        return False
    for m in c.column.missing_markers:
        reason = str(m.get("reason", "")) if isinstance(m, dict) else ""
        if reason.startswith(("contextually_suspicious", "column_specific_outlier")):
            return True
    return False


def _match_high_anomaly(d: Decision, c: TriggerContext) -> bool:
    if not _is_type(d):
        return False
    rate = anomaly_rate(d, c.raw_series, column_missing_markers(c.column))
    return rate is not None and rate > ANOMALY_RATE_THRESHOLD


def _match_ambiguous_type(d: Decision, _c: TriggerContext) -> bool:
    return _is_type(d)


class TriggerRegistry:
    """Ordered registry of trigger rules."""

    def __init__(self) -> None:
        self._rules: List[TriggerRule] = []
        self._register_defaults()

    @property
    def rules(self) -> List[TriggerRule]:
        return list(self._rules)

    def register(self, rule: TriggerRule) -> None:
        """
        Register a rule. A rule for a category whose plug-in point is still empty
        replaces the empty placeholder; anything else is appended (and so is
        evaluated after the built-in rules for the same category).
        """
        for i, existing in enumerate(self._rules):
            if existing.category == rule.category and not existing.implemented:
                self._rules[i] = rule
                return
        self._rules.append(rule)

    def classify(self, decision: Decision, context: Optional[TriggerContext] = None) -> Optional[str]:
        """Return the first matching category, or None if no rule matches."""
        ctx = context or TriggerContext()
        for rule in self._rules:
            if rule.implemented and rule.matcher(decision, ctx):
                return rule.category
        return None

    def implemented_categories(self) -> List[str]:
        return [r.category for r in self._rules if r.implemented]

    def empty_plugin_categories(self) -> List[str]:
        return [r.category for r in self._rules if not r.implemented]

    def _register_defaults(self) -> None:
        C = TriggerCategory
        # Order matters: first match wins. Structure and task first (they use
        # different stages), then the most specific type reasons, then the catch-all.
        self._rules = [
            TriggerRule(
                C.AMBIGUOUS_STRUCTURE_OR_TASK,
                "Ambiguous prediction goal, or table rows of uneven length",
                _match_ragged_or_task,
            ),
            TriggerRule(
                C.LOW_CONFIDENCE_STRUCTURE,
                "Text format, column separator, or column-names row detected with low confidence",
                _match_low_confidence_structure,
            ),
            TriggerRule(
                C.DESTRUCTIVE_ACTION,
                "Dropping columns or rows above an impact threshold (empty plug-in point: "
                "no such Decision exists until the cleaning port)",
                _never,
                implemented=False,
            ),
            TriggerRule(
                C.SUSPECTED_LEAKAGE,
                "Feature that may give away the target (empty plug-in point: "
                "no such Decision exists until the engineering port)",
                _never,
                implemented=False,
            ),
            TriggerRule(
                C.LOSSY_COERCION,
                "Conversion would lose information (embedded units, ambiguous dates, leading zeros)",
                _match_lossy,
            ),
            TriggerRule(
                C.DISGUISED_MISSING,
                "Values that may be placeholders for missing data",
                _match_disguised_missing,
            ),
            TriggerRule(
                C.HIGH_ANOMALY_RATE,
                "Proposed conversion fails for more than "
                f"{int(ANOMALY_RATE_THRESHOLD * 100)}% of values",
                _match_high_anomaly,
            ),
            TriggerRule(
                C.AMBIGUOUS_TYPE,
                "Close call between column types, or the type check abstained",
                _match_ambiguous_type,
            ),
        ]


_DEFAULT_REGISTRY = TriggerRegistry()


def default_registry() -> TriggerRegistry:
    """Return the process-wide trigger registry."""
    return _DEFAULT_REGISTRY


def register_trigger(rule: TriggerRule) -> None:
    """Plug a rule into the process-wide registry (e.g. destructive action, leakage)."""
    _DEFAULT_REGISTRY.register(rule)


def classify_decision_trigger(
    decision: Decision,
    context: Optional[TriggerContext] = None,
) -> str:
    """
    Classify a Decision into one of the trigger categories.

    Raises ValueError if no registered rule recognises the Decision, so a new kind
    of Decision cannot silently be filed under the wrong reason.
    """
    cat = _DEFAULT_REGISTRY.classify(decision, context)
    if cat is None:
        raise ValueError(
            f"No trigger rule recognises decision '{decision.id}' "
            f"(stage='{decision.stage}', kind='{decision.kind}'). "
            "Register one with register_trigger()."
        )
    return cat
