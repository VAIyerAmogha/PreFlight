"""
Notebook channel (ipywidgets, optional).

A notebook cell cannot block on a widget click without freezing the kernel that
delivers the click, so this channel never waits. First call: it shows a form and
raises ReviewPending. After the user picks answers and presses Submit, calling
resolve() again with the same channel (or channel="notebook") reads them.

Nothing is preselected: a question left unanswered raises, it is never defaulted.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from preflight.interact.channels.base import BaseChannel, ReviewPending
from preflight.interact.question import Question, ReviewSession, validate_answers

# Submitted-but-not-yet-collected answers, keyed by the question set they belong to.
_SUBMITTED: Dict[Tuple[str, ...], Dict[str, Any]] = {}


def _session_key(session: ReviewSession) -> Tuple[str, ...]:
    return (session.source_path,) + tuple(sorted(q.fingerprint for q in session.questions))


def _import_widgets() -> Any:
    try:
        import ipywidgets
    except ImportError as exc:
        raise ImportError(
            "The notebook channel needs ipywidgets: pip install 'pypreflight[notebook]'"
        ) from exc
    return ipywidgets


def _default_display(obj: Any) -> None:
    from IPython.display import display

    display(obj)


class NotebookChannel(BaseChannel):
    name = "notebook"

    def __init__(self, widgets: Any = None, display_fn: Optional[Callable[[Any], None]] = None) -> None:
        self._widgets = widgets
        self._display = display_fn
        self.controls: Dict[str, Any] = {}

    def build(self, session: ReviewSession) -> Any:
        """Build the form. Returns the container widget."""
        w = self._widgets or _import_widgets()
        key = _session_key(session)
        self.controls = {}
        boxes = [
            w.HTML(
                f"<b>PreFlight needs {len(session.questions)} answer(s)</b> "
                "(nothing is preselected; the recommended choice is marked in each question)."
            )
        ]
        for i, q in enumerate(session.questions, start=1):
            col = f" &mdash; column <code>{_esc(q.column)}</code>" if q.column else ""
            boxes.append(
                w.HTML(
                    f"<p><b>{i}. {_esc(q.text)}</b>{col}<br>{_esc(q.explanation)}<br>"
                    f"<i>{_esc(q.evidence_summary)}</i><br>"
                    f"Recommended: {_esc(q.recommended_default)}</p>"
                )
            )
            radio = w.RadioButtons(options=list(q.options), value=None, description="")
            self.controls[q.question_id] = radio
            boxes.append(radio)
        status = w.HTML("")
        submit = w.Button(description="Submit answers")

        def on_submit(_btn: Any) -> None:
            raw = {qid: c.value for qid, c in self.controls.items()}
            try:
                validate_answers(session, raw)
            except ValueError as exc:
                status.value = f"<span style='color:#b91c1c'>{_esc(str(exc))}</span>"
                _SUBMITTED.pop(key, None)
                return
            _SUBMITTED[key] = raw
            status.value = "Answers saved. Run the resolve() cell again to continue."

        submit.on_click(on_submit)
        self.on_submit = on_submit
        boxes.extend([submit, status])
        return w.VBox(boxes)

    def review(self, session: ReviewSession) -> Dict[str, str]:
        if not session.questions:
            return {}
        key = _session_key(session)
        if key in _SUBMITTED:
            return validate_answers(session, _SUBMITTED.pop(key))
        form = self.build(session)
        (self._display or _default_display)(form)
        raise ReviewPending(
            "Answer the questions in the form above, press Submit, then run this cell again."
        )


def _esc(text: Any) -> str:
    import html

    return html.escape(str(text))
