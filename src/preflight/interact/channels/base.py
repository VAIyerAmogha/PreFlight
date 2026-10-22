"""
Base channel interface for the PreFlight v2.0 interaction layer.

A channel gets the whole batched ReviewSession and returns one answer per question,
keyed by question_id. Every channel validates through `validate_answers`: an answer
must be exactly one of the question's listed options, and a missing or invalid answer
raises an error naming the question. A channel never substitutes a default.
"""

from typing import Dict

from preflight.interact.question import ReviewSession, validate_answers


class ReviewPending(RuntimeError):
    """
    The channel cannot collect answers right now and has not blocked waiting for them
    (review file written; notebook form shown). Answer, then call resolve() again.
    """

    def __init__(self, message: str, path: str = "") -> None:
        super().__init__(message)
        self.path = path


class BaseChannel:
    """Interface implemented by every channel."""

    name: str = "base"

    def review(self, session: ReviewSession) -> Dict[str, str]:
        """Return validated answers {question_id: option label} for every question."""
        raise NotImplementedError


__all__ = ["BaseChannel", "ReviewPending", "validate_answers"]
