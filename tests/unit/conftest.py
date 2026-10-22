"""Shared fixtures for interaction-layer unit tests."""

from pathlib import Path
from typing import Callable, Dict, List

import pytest

from preflight.interact.question import Question, ReviewSession, create_review_session
from preflight.interact.resolve import _load_raw
from preflight.plan import Plan, plan


def make_messy_csv(path: Path) -> Path:
    """A small file that raises several kinds of questions (units, dates, ids, numbers)."""
    rows = ["id,zip,mileage,when,score,target"]
    for i in range(12):
        rows.append(f"{i + 1},0{2100 + i},{10 + i} kmpl,0{(i % 9) + 1}/0{(i % 8) + 1}/2020,{i % 3},{i % 2}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return make_messy_csv(d / "messy.csv")


@pytest.fixture
def task_csv(tmp_path: Path) -> Path:
    """Target with three integer values: classification vs regression is ambiguous."""
    d = tmp_path / "data"
    d.mkdir()
    p = d / "task.csv"
    p.write_text("feature,target\n1.0,0\n2.0,1\n3.0,2\n4.0,1\n5.0,0\n6.0,2\n", encoding="utf-8")
    return p


@pytest.fixture
def messy_plan(messy_csv: Path) -> Plan:
    return plan(messy_csv, target="target")


def session_of(p: Plan) -> ReviewSession:
    return create_review_session(p.decisions, p.source_path, raw_df=_load_raw(p), schema=p.schema)


@pytest.fixture
def messy_session(messy_plan: Plan) -> ReviewSession:
    s = session_of(messy_plan)
    assert len(s.questions) >= 4, "fixture should raise several questions"
    return s


def pick(index: int) -> Callable[[List[Question]], Dict[str, str]]:
    """Callback that answers every question with its index-th option (clamped)."""
    return lambda qs: {q.question_id: q.options[min(index, len(q.options) - 1)] for q in qs}


def recommended(qs: List[Question]) -> Dict[str, str]:
    return {q.question_id: q.recommended_default for q in qs}
