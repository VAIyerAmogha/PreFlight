"""Callback channel: a user-supplied function answers the whole batch."""

from collections.abc import Mapping
from typing import Callable, Dict, List

from preflight.interact.channels.base import BaseChannel
from preflight.interact.question import Question, ReviewSession, validate_answers


class CallbackChannel(BaseChannel):
    """
    handler(questions: List[Question]) -> {question_id: option label}

    The handler is called once with every question. Exceptions it raises propagate
    unchanged.
    """

    name = "callback"

    def __init__(self, handler: Callable[[List[Question]], Dict[str, str]]) -> None:
        if not callable(handler):
            raise TypeError("callback channel needs a callable that takes a list of Question")
        self.handler = handler

    def review(self, session: ReviewSession) -> Dict[str, str]:
        if not session.questions:
            return {}
        result = self.handler(list(session.questions))
        if not isinstance(result, Mapping):
            raise TypeError(
                "callback must return a mapping of question_id -> option label, "
                f"got {type(result).__name__}"
            )
        return validate_answers(session, result)
