"""
Question model and batched review session for the PreFlight v2.0 interaction layer.

A Question is built from a Decision with needs_attention=True:
  - plain-language text, evidence summary, and reason,
  - 2 to 4 options, each tied to the Decision action it would apply,
  - one recommended safe default (the Decision's own default),
  - for column questions, a preview of real values before and after each option.

All questions for a run are collected into ONE ReviewSession.

Nothing here repairs bad input silently: a Question that breaks its own rules
(option count, default not among options) raises instead of being quietly fixed.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from preflight.decisions import Decision
from preflight.interact.preview import build_column_preview
from preflight.interact.triggers import (
    TriggerContext,
    classify_decision_trigger,
    column_missing_markers,
)
from preflight.report.templates import (
    friendly_action_name,
    render_question_text,
)
from preflight.schema import ColumnRawMetadata, TableSchema

MIN_OPTIONS = 2
MAX_OPTIONS = 4


class QuestionError(ValueError):
    """A Question could not be built or is malformed."""


class InvalidAnswerError(ValueError):
    """An answer is not one of the question's listed options."""

    def __init__(self, question: "Question", answer: Any, detail: str = "") -> None:
        self.question_id = question.question_id
        self.column = question.column
        self.answer = answer
        col = f" (column '{question.column}')" if question.column else ""
        msg = (
            f"Invalid answer {answer!r} for question {question.question_id}{col}: "
            f"\"{question.text}\" Allowed answers: {question.options}."
        )
        super().__init__(f"{msg} {detail}".strip())


class MissingAnswerError(InvalidAnswerError):
    """A question has no answer."""

    def __init__(self, question: "Question") -> None:
        ValueError.__init__(
            self,
            f"No answer for question {question.question_id}"
            + (f" (column '{question.column}')" if question.column else "")
            + f": \"{question.text}\" Allowed answers: {question.options}.",
        )
        self.question_id = question.question_id
        self.column = question.column
        self.answer = None


@dataclass
class Question:
    """
    One user-facing question.

    options: 2 to 4 distinct plain-language choices.
    option_actions: the Decision action each option applies (same keys as options).
    recommended_default: the safe default; always one of options.
    preview: for column questions, {"before": [...], "after": {option: [...]}, "note"?: str}.
    """

    question_id: str
    decision_id: str
    fingerprint: str
    category: str
    column: Optional[str]
    text: str
    explanation: str
    evidence_summary: str
    options: List[str]
    option_actions: Dict[str, str]
    recommended_default: str
    preview: Optional[Dict[str, Any]] = None
    selected_option: Optional[str] = None

    def __post_init__(self) -> None:
        if not (MIN_OPTIONS <= len(self.options) <= MAX_OPTIONS):
            raise QuestionError(
                f"Question {self.question_id} must have {MIN_OPTIONS} to {MAX_OPTIONS} options, "
                f"got {len(self.options)}: {self.options}"
            )
        if len(set(self.options)) != len(self.options):
            raise QuestionError(f"Question {self.question_id} has duplicate options: {self.options}")
        if self.recommended_default not in self.options:
            raise QuestionError(
                f"Question {self.question_id}: recommended default "
                f"{self.recommended_default!r} is not among options {self.options}"
            )
        if set(self.option_actions) != set(self.options):
            raise QuestionError(f"Question {self.question_id}: option_actions do not match options")
        if self.column is not None and self.preview is None:
            raise QuestionError(f"Question {self.question_id} concerns a column but has no preview")

    def action_for(self, answer: str) -> str:
        """Decision action applied by an answer."""
        return self.option_actions[answer]

    def check_answer(self, answer: Any) -> str:
        """Return the answer if it is one of the options; raise otherwise."""
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            raise MissingAnswerError(self)
        if not isinstance(answer, str) or answer not in self.options:
            raise InvalidAnswerError(self, answer)
        return answer

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        return cls(**data)


# ---------------------------------------------------------------------------
# Option builders (one per Decision kind; later phases register their own)
# ---------------------------------------------------------------------------

# A builder returns the candidate ACTIONS in priority order. The Decision's
# default and proposed action are always put first by `_assemble_options`.
OptionBuilder = Callable[[Decision], List[str]]

_TEXT_ACTIONS = ("keep_as_string", "keep_as_text")

_TYPE_TO_ACTION = {
    "constant": "keep_as_constant",
    "boolean": "cast_to_boolean",
    "id": "keep_as_id",
    "int": "cast_to_int",
    "float": "cast_to_float",
    "datetime": "cast_to_datetime",
    "categorical": "keep_as_categorical",
    "text": "keep_as_string",
}
_GENERIC_TYPE_CHOICES = ["keep_as_categorical", "cast_to_float", "cast_to_int"]


def canonical_action(action: str) -> str:
    """Actions that mean the same thing compare equal (plain text has two spellings)."""
    return "keep_as_string" if action in _TEXT_ACTIONS else action


def option_label(action: str) -> str:
    """Plain-language label for an action."""
    if action == "none":
        return "No prediction goal (data inspection only)"
    if canonical_action(action) == "keep_as_string":
        return "Keep as plain text"
    return friendly_action_name(action)


def _type_actions(d: Decision) -> List[str]:
    scores = (d.evidence or {}).get("candidate_scores") or {}
    ranked = sorted(
        ((str(k).lower(), v) for k, v in scores.items() if isinstance(v, (int, float)) and v > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [_TYPE_TO_ACTION[k] for k, _ in ranked if k in _TYPE_TO_ACTION]


def _encoding_actions(d: Decision) -> List[str]:
    return ["use_encoding_iso-8859-1"]


def _dialect_actions(d: Decision) -> List[str]:
    return [f"separate_by_{c}" for c in (";", "\t", "|", ",")]


def _header_actions(d: Decision) -> List[str]:
    return ["no_header"]


def _task_actions(d: Decision) -> List[str]:
    return ["classification", "regression", "none"]


_BUILDERS: Dict[str, OptionBuilder] = {
    "type_inference": _type_actions,
    "type_abstain": _type_actions,
    "encoding": _encoding_actions,
    "dialect": _dialect_actions,
    "header": _header_actions,
    "ambiguous_task": _task_actions,
    "task_inference": _task_actions,
}


def register_question_builder(kind: str, builder: OptionBuilder) -> None:
    """Plug in extra option candidates for a Decision kind (used by later phases)."""
    _BUILDERS[kind] = builder


def _assemble_options(d: Decision, candidates: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Order: safe default, proposed action, then candidates. Distinct by meaning and by
    label, capped at MAX_OPTIONS without ever dropping the default or the proposal.
    """
    ordered: List[str] = []
    seen_canon = set()
    seen_labels = set()
    for action in [d.default, d.action, *candidates]:
        canon = canonical_action(action)
        label = option_label(action)
        if canon in seen_canon or label in seen_labels:
            continue
        seen_canon.add(canon)
        seen_labels.add(label)
        ordered.append(action)
        if len(ordered) == MAX_OPTIONS:
            break

    if len(ordered) < MIN_OPTIONS and d.kind in ("type_inference", "type_abstain"):
        for action in _GENERIC_TYPE_CHOICES:
            if canonical_action(action) not in seen_canon:
                ordered.append(action)
                seen_canon.add(canonical_action(action))
            if len(ordered) >= MIN_OPTIONS:
                break

    if len(ordered) < MIN_OPTIONS:
        raise QuestionError(
            f"Decision '{d.id}' (kind '{d.kind}') offers fewer than {MIN_OPTIONS} distinct "
            f"choices (action '{d.action}', default '{d.default}'). "
            "Register a question builder for this kind with register_question_builder()."
        )
    labels = [option_label(a) for a in ordered]
    return labels, dict(zip(labels, ordered))


# ---------------------------------------------------------------------------
# Question construction
# ---------------------------------------------------------------------------

def _column_meta(schema: Optional[TableSchema], name: str) -> Optional[ColumnRawMetadata]:
    if schema is None:
        return None
    return next((c for c in schema.columns if c.name == name), None)


_TASK_EFFECTS = {
    "classification": lambda v: f"{v} (a category)",
    "regression": lambda v: f"{v} (a number)",
    "none": lambda v: f"{v} (not used)",
}


def build_question_from_decision(
    decision: Decision,
    index: int = 1,
    raw_df: Optional[pd.DataFrame] = None,
    schema: Optional[TableSchema] = None,
) -> Question:
    """Construct a Question from a needs_attention Decision."""
    col = decision.columns[0] if decision.columns else None
    ev = decision.evidence or {}
    meta = _column_meta(schema, col) if col else None
    series = raw_df[col] if (raw_df is not None and col is not None and col in raw_df.columns) else None

    category = classify_decision_trigger(decision, TriggerContext(column=meta, raw_series=series))
    text = render_question_text(decision)

    builder = _BUILDERS.get(decision.kind)
    candidates = builder(decision) if builder is not None else []
    options, option_actions = _assemble_options(decision, candidates)

    preview: Optional[Dict[str, Any]] = None
    if col is not None:
        is_task = decision.kind in ("ambiguous_task", "task_inference", "task")
        preview = build_column_preview(
            series,
            option_actions,
            ev,
            missing=column_missing_markers(meta),
            fallback_values=ev.get("unique_samples") or ev.get("sample_values") or [],
            action_effects=_TASK_EFFECTS if is_task else None,
        )

    fp = decision.fingerprint or ""
    return Question(
        question_id=f"q{index:03d}_{fp[:6]}",
        decision_id=decision.id,
        fingerprint=fp,
        category=category,
        column=col,
        text=text.question or (f"How should column '{col}' be handled?" if col else "How should this be handled?"),
        explanation=text.explanation or "Automatic checks were not certain about this.",
        evidence_summary=text.summary,
        options=options,
        option_actions=option_actions,
        recommended_default=option_label(decision.default),
        preview=preview,
    )


# ---------------------------------------------------------------------------
# Batched session and answer validation
# ---------------------------------------------------------------------------

@dataclass
class ReviewSession:
    """All questions for one run, asked together."""

    source_path: str
    questions: List[Question] = field(default_factory=list)

    def by_id(self) -> Dict[str, Question]:
        return {q.question_id: q for q in self.questions}


def validate_answers(
    session_or_questions: Union[ReviewSession, Sequence[Question]],
    answers: Mapping[str, Any],
) -> Dict[str, str]:
    """
    Check an answer for EVERY question, keyed by question_id.

    Raises MissingAnswerError / InvalidAnswerError naming the first offending question,
    and ValueError for answers addressed to questions that do not exist. Never
    substitutes a default.
    """
    questions = (
        session_or_questions.questions
        if isinstance(session_or_questions, ReviewSession)
        else list(session_or_questions)
    )
    known = {q.question_id for q in questions}
    unknown = sorted(set(answers) - known)
    if unknown:
        raise ValueError(f"Answers given for unknown question id(s): {unknown}. Known: {sorted(known)}")
    clean: Dict[str, str] = {}
    for q in questions:
        clean[q.question_id] = q.check_answer(answers.get(q.question_id))
    return clean


def create_review_session(
    decisions: Sequence[Decision],
    source_path: Union[str, Path],
    raw_df: Optional[pd.DataFrame] = None,
    schema: Optional[TableSchema] = None,
) -> ReviewSession:
    """Collect every needs_attention Decision into ONE batched session."""
    questions = [
        build_question_from_decision(d, index=i, raw_df=raw_df, schema=schema)
        for i, d in enumerate((d for d in decisions if d.needs_attention), start=1)
    ]
    return ReviewSession(source_path=str(Path(source_path).resolve()), questions=questions)
