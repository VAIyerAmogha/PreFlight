"""
Resolution modes for the PreFlight v2.0 interaction layer.

  ask       resolve questions through a channel
  defaults  apply each Decision's safe default, status stays 'automatic', logged as such
  strict    if any question is unresolved, raise an error listing every one

Mode selection: an explicit argument always wins. With no argument, a non-interactive
context (no TTY and no notebook kernel) selects strict, so nothing can ever block
waiting for input.
"""

import sys
from typing import Dict, List, Optional, Sequence

from preflight.decisions import Decision
from preflight.interact.question import (
    Question,
    ReviewSession,
    canonical_action,
)
from preflight.report.templates import render_decision_text

MODES = ("ask", "defaults", "strict")


class StrictResolutionError(RuntimeError):
    """Raised in strict mode when any question is unresolved. Lists every one."""

    def __init__(self, unresolved: Sequence[Question]) -> None:
        self.unresolved_questions = list(unresolved)
        lines = [
            f"  - id: {q.decision_id} | column: {q.column if q.column else '(whole file)'} | {q.text}"
            for q in self.unresolved_questions
        ]
        super().__init__(
            f"Strict mode: {len(lines)} unresolved item(s). Answer them (mode='ask'), "
            "accept safe defaults (mode='defaults'), or reuse stored answers.\n" + "\n".join(lines)
        )


def in_notebook_kernel() -> bool:
    """True inside a Jupyter/IPython kernel (checked without importing IPython)."""
    ipython = sys.modules.get("IPython")
    get_ip = getattr(ipython, "get_ipython", None)
    if not callable(get_ip):
        return False
    try:
        shell = get_ip()
    except Exception:
        return False
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def is_interactive_context() -> bool:
    """True inside a notebook kernel, or when both stdin and stdout are a terminal."""
    if in_notebook_kernel():
        return True
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except (ValueError, OSError):  # closed stream
        return False


def resolve_mode(explicit: Optional[str] = None) -> str:
    """Explicit mode wins; otherwise 'ask' if interactive, else 'strict'."""
    if explicit is not None:
        mode = str(explicit).lower().strip()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {explicit!r}; choose one of {list(MODES)}.")
        return mode
    return "ask" if is_interactive_context() else "strict"


def _decision_index(decisions: Sequence[Decision]) -> Dict[str, Decision]:
    return {d.id: d for d in decisions}


def _finalize(decision: Decision) -> None:
    """A resolved Decision no longer needs attention; its sentence reflects the final action."""
    decision.needs_attention = False
    decision.plain_language = render_decision_text(decision).summary


def apply_answer(decision: Decision, question: Question, answer: str, source: str, reuse: bool = False) -> str:
    """
    Apply a chosen option to its Decision. Returns the new status.

    Same action as proposed -> 'confirmed'. A different action -> 'changed'.
    reuse=True (a stored answer for the same fingerprint) always ends 'confirmed': the
    stored action is applied and the person is not asked again.
    """
    action = question.action_for(answer)
    if reuse:
        decision.action = action
        decision.confirm(source=source, answer=answer)
    elif canonical_action(action) == canonical_action(decision.action):
        decision.confirm(source=source, answer=answer)
    else:
        decision.change(action, source=source, answer=answer)
    _finalize(decision)
    return decision.status


def apply_defaults_mode(session: ReviewSession, decisions: Sequence[Decision]) -> None:
    """Apply every unresolved Decision's safe default; status 'automatic', logged as such."""
    index = _decision_index(decisions)
    for q in session.questions:
        dec = index[q.decision_id]
        dec.mark_automatic(action=dec.default, source="defaults mode: safe default")
        _finalize(dec)


def apply_strict_mode(session: ReviewSession) -> None:
    """Raise if anything is unresolved."""
    if session.questions:
        raise StrictResolutionError(session.questions)


def apply_ask_mode(
    session: ReviewSession,
    decisions: Sequence[Decision],
    answers: Dict[str, str],
    source: str,
) -> Dict[str, str]:
    """
    Apply validated channel answers {question_id: option}. Returns {question_id: status}.
    The answers must already have passed validation (every channel does this).
    """
    index = _decision_index(decisions)
    statuses: Dict[str, str] = {}
    for q in session.questions:
        statuses[q.question_id] = apply_answer(index[q.decision_id], q, q.check_answer(answers.get(q.question_id)), source)
    return statuses
