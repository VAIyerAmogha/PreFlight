"""
Review-file channel: write review.json, the user edits it, resolve() reads and validates.

The file holds real sample values from the user's data (previews), so treat it like
the data itself.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from preflight.interact.channels.base import BaseChannel, ReviewPending
from preflight.interact.question import ReviewSession, validate_answers

REVIEW_FORMAT = "preflight-review"
REVIEW_VERSION = 1

_INSTRUCTIONS = (
    "For each question, set \"answer\" to exactly one of the strings in \"options\" "
    "(\"recommended_default\" is the safe choice). Leave nothing blank, then run "
    "preflight.resolve() on this file."
)


class ReviewFileError(ValueError):
    """review.json is unreadable or does not match the current questions."""


def write_review_file(
    session: ReviewSession,
    path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write review.json with empty answer fields."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": REVIEW_FORMAT,
        "version": REVIEW_VERSION,
        "source_path": session.source_path,
        **(meta or {}),
        "instructions": _INSTRUCTIONS,
        "questions": [
            {
                "question_id": q.question_id,
                "fingerprint": q.fingerprint,
                "column": q.column,
                "category": q.category,
                "text": q.text,
                "explanation": q.explanation,
                "evidence_summary": q.evidence_summary,
                "options": q.options,
                "recommended_default": q.recommended_default,
                "preview": q.preview,
                "answer": None,
            }
            for q in session.questions
        ],
    }
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_review_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and shape-check review.json."""
    p = Path(path).resolve()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ReviewFileError(f"Could not read review file {p}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != REVIEW_FORMAT:
        raise ReviewFileError(f"{p} is not a PreFlight review file")
    if data.get("version") != REVIEW_VERSION:
        raise ReviewFileError(
            f"{p} has review format version {data.get('version')!r}; this release reads version {REVIEW_VERSION}"
        )
    if not isinstance(data.get("questions"), list):
        raise ReviewFileError(f"{p} has no question list")
    return data


class ReviewFileChannel(BaseChannel):
    name = "review_file"

    def __init__(self, path: Union[str, Path], meta: Optional[Dict[str, Any]] = None) -> None:
        self.path = Path(path).resolve()
        self.meta = meta

    def review(self, session: ReviewSession) -> Dict[str, str]:
        if not session.questions:
            return {}
        if not self.path.exists():
            write_review_file(session, self.path, self.meta)
            raise ReviewPending(
                f"Wrote {len(session.questions)} question(s) to {self.path}. Fill in the "
                f"\"answer\" fields, then call resolve({str(self.path)!r}).",
                path=str(self.path),
            )
        data = load_review_file(self.path)
        return validate_answers(session, answers_from_review(session, data, self.path))


def answers_from_review(session: ReviewSession, data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """
    Extract answers from a loaded review file, refusing a file whose questions are not
    exactly the session's (stale file for different data or a changed plan).
    """
    file_fps = {}
    for item in data["questions"]:
        if not isinstance(item, dict) or "question_id" not in item:
            raise ReviewFileError(f"{path}: malformed question entry: {item!r}")
        file_fps[item["question_id"]] = item.get("fingerprint")
    expected = {q.question_id: q.fingerprint for q in session.questions}
    if file_fps != expected:
        missing = sorted(set(expected) - set(file_fps))
        extra = sorted(set(file_fps) - set(expected))
        changed = sorted(k for k in set(expected) & set(file_fps) if expected[k] != file_fps[k])
        raise ReviewFileError(
            f"{path} does not match the current questions "
            f"(missing: {missing}, unexpected: {extra}, changed: {changed}). "
            "Delete it and run again to write a fresh review file."
        )
    return {item["question_id"]: item.get("answer") for item in data["questions"]}
