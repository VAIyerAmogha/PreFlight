"""
Answer persistence for the PreFlight v2.0 interaction layer.

Answers are kept in a versioned JSON file (answers.json) inside the run's out_dir,
keyed by Decision.fingerprint. Never written next to the source file.

Re-run rules:
  - A question whose fingerprint matches a stored answer is not asked again; its
    Decision is marked 'confirmed' and the stored action is applied.
  - A different fingerprint means different evidence: the question is asked again.
    A stored answer is never applied to a non-matching fingerprint.
  - A stored answer is also ignored (and the question asked again) if its option is no
    longer one of the question's options or no longer maps to the same action.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import warnings

from preflight.interact.question import Question

STORE_VERSION = 1
STORE_FILENAME = "answers.json"


@dataclass
class StoredAnswer:
    fingerprint: str
    question: str
    column: Optional[str]
    answer: str
    action: str
    status: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnswerStore:
    """Fingerprint-keyed answers in <out_dir>/answers.json."""

    def __init__(self, out_dir: Union[str, Path]) -> None:
        self.path = Path(out_dir).resolve() / STORE_FILENAME
        self._answers: Dict[str, StoredAnswer] = {}
        self._unreadable = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("store_version") != STORE_VERSION:
                raise ValueError(f"unsupported store version {data.get('store_version')!r}" if isinstance(data, dict) else "not an object")
            self._answers = {fp: StoredAnswer(**rec) for fp, rec in data["answers"].items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._answers = {}
            self._unreadable = True
            warnings.warn(
                f"Ignoring stored answers in {self.path} ({exc}); questions will be asked again. "
                "The unreadable file is kept as answers.json.bak when new answers are saved.",
                stacklevel=3,
            )

    def __len__(self) -> int:
        return len(self._answers)

    def lookup(self, question: Question) -> Optional[StoredAnswer]:
        """The stored answer for this question, only if it still applies exactly."""
        rec = self._answers.get(question.fingerprint)
        if rec is None or rec.fingerprint != question.fingerprint:
            return None
        if question.option_actions.get(rec.answer) != rec.action:
            return None
        return rec

    def record(self, question: Question, answer: str, status: str, when: Optional[str] = None) -> StoredAnswer:
        rec = StoredAnswer(
            fingerprint=question.fingerprint,
            question=question.text,
            column=question.column,
            answer=answer,
            action=question.action_for(answer),
            status=status,
            timestamp=when or datetime.now(timezone.utc).isoformat(),
        )
        self._answers[question.fingerprint] = rec
        return rec

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._unreadable and self.path.exists():
            os.replace(self.path, self.path.with_name(STORE_FILENAME + ".bak"))
            self._unreadable = False
        payload = {
            "store_version": STORE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "answers": {fp: rec.to_dict() for fp, rec in self._answers.items()},
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path
