"""
Terminal channel: one batched review screen, then one prompt per question.

Baseline is the standard library only (print and input()). If `rich` is installed it
is used for the screen's formatting; input still goes through input().

An answer is a number from the list, the exact option text, or the word "default"
(typed on purpose, to take the recommended choice). An empty line is NOT an answer.
"""

import sys
from typing import Any, Callable, Dict, List, Optional, TextIO

from preflight.interact.channels.base import BaseChannel
from preflight.interact.question import (
    InvalidAnswerError,
    MissingAnswerError,
    Question,
    ReviewSession,
    validate_answers,
)


class NonInteractiveError(RuntimeError):
    """Asking was requested where no person can answer (no TTY, no notebook kernel)."""


def rich_available() -> bool:
    try:
        import rich  # noqa: F401
    except Exception:
        return False
    return True


def _preview_lines(q: Question) -> List[str]:
    if not q.preview:
        return []
    pv = q.preview
    lines: List[str] = []
    if not pv.get("before"):
        return [f"   Preview: {pv.get('note', 'no values to show')}"]
    lines.append("   Preview, real values from your file:")
    lines.append(f"     now:  {' | '.join(pv['before'])}")
    for label, vals in pv.get("after", {}).items():
        lines.append(f"     {label}: {' | '.join(vals)}")
    if pv.get("note"):
        lines.append(f"     ({pv['note']})")
    return lines


def format_screen(session: ReviewSession) -> str:
    """The batched review screen as plain text."""
    n = len(session.questions)
    out = [
        "=" * 72,
        f"PreFlight needs your input on {n} item(s)",
        f"File: {session.source_path}",
        "=" * 72,
    ]
    for i, q in enumerate(session.questions, start=1):
        where = f"  [column: {q.column}]" if q.column else ""
        out.append("")
        out.append(f"{i}. {q.text}{where}")
        out.append(f"   Why: {q.explanation}")
        out.append(f"   What we found: {q.evidence_summary}")
        out.extend(_preview_lines(q))
        for j, opt in enumerate(q.options, start=1):
            mark = "   <- recommended" if opt == q.recommended_default else ""
            out.append(f"   {j}) {opt}{mark}")
    out.append("")
    out.append("Answer each item with its option number (or 'default' for the recommended choice).")
    return "\n".join(out)


def _print_rich(session: ReviewSession, stream: TextIO) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    console = Console(file=stream, highlight=False)
    console.print(
        Panel.fit(
            f"{len(session.questions)} item(s) need your input\n[dim]File:[/dim] {escape(session.source_path)}",
            title="PreFlight review",
        )
    )
    for i, q in enumerate(session.questions, start=1):
        where = f"  [dim]column:[/dim] [cyan]{escape(q.column)}[/cyan]" if q.column else ""
        console.print(f"\n[bold]{i}. {escape(q.text)}[/bold]{where}")
        console.print(f"[dim]Why:[/dim] {escape(q.explanation)}")
        console.print(f"[dim]What we found:[/dim] {escape(q.evidence_summary)}")
        if q.preview:
            pv = q.preview
            if pv.get("before"):
                t = Table(title="Preview, real values from your file", show_lines=False)
                t.add_column("Option")
                t.add_column("Values")
                t.add_row("now", " | ".join(escape(v) for v in pv["before"]))
                for label, vals in pv.get("after", {}).items():
                    t.add_row(escape(label), " | ".join(escape(v) for v in vals))
                console.print(t)
            if pv.get("note"):
                console.print(f"[dim]{escape(pv['note'])}[/dim]")
        for j, opt in enumerate(q.options, start=1):
            mark = "  [green]<- recommended[/green]" if opt == q.recommended_default else ""
            console.print(f"  {j}) {escape(opt)}{mark}")
    console.print("\nAnswer each item with its option number (or 'default' for the recommended choice).")


def _interpret(q: Question, raw: str) -> str:
    text = raw.strip()
    if not text:
        raise MissingAnswerError(q)
    if text.lower() == "default":
        return q.recommended_default
    if text.isdigit():
        n = int(text)
        if 1 <= n <= len(q.options):
            return q.options[n - 1]
        raise InvalidAnswerError(q, text, f"Choose a number from 1 to {len(q.options)}.")
    if text in q.options:
        return text
    raise InvalidAnswerError(q, text)


class TerminalChannel(BaseChannel):
    """
    stdin / stdout: optional streams (for scripted input and tests). When stdin is not
    given, the builtin input() is used and a real terminal is required.
    use_rich: None = use rich if installed; False = plain text only.
    require_tty: only relevant when stdin is not given.
    """

    name = "terminal"

    def __init__(
        self,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        use_rich: Optional[bool] = None,
        require_tty: bool = True,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.use_rich = use_rich
        self.require_tty = require_tty

    def _read(self, prompt: str, out: TextIO) -> str:
        if self.stdin is None:
            return input(prompt)
        out.write(prompt)
        out.flush()
        line = self.stdin.readline()
        if line == "":
            raise EOFError
        return line

    def review(self, session: ReviewSession) -> Dict[str, str]:
        if not session.questions:
            return {}
        if self.stdin is None and self.require_tty:
            from preflight.interact.modes import is_interactive_context

            if not is_interactive_context():
                raise NonInteractiveError(
                    "The terminal channel needs a real terminal to ask "
                    f"{len(session.questions)} question(s), and none is attached. "
                    "Use channel='review_file' or a callback, or mode='defaults' / mode='strict'."
                )
        out = self.stdout or sys.stdout
        want_rich = rich_available() if self.use_rich is None else self.use_rich
        if want_rich:
            _print_rich(session, out)
        else:
            out.write(format_screen(session) + "\n")
        answers: Dict[str, str] = {}
        for i, q in enumerate(session.questions, start=1):
            try:
                raw = self._read(f"Item {i} of {len(session.questions)} (1-{len(q.options)}): ", out)
            except EOFError:
                raise MissingAnswerError(q) from None
            answers[q.question_id] = _interpret(q, raw)
        return validate_answers(session, answers)
