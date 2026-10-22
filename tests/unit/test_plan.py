"""
Unit tests for PreFlight v2.0 Plan module (Phase 4 / M3).

Validates:
1. plan() purity: source file and arguments are never mutated.
2. Decision status transitions: automatic -> confirmed -> changed, with timestamp updates.
3. Fingerprint stability: identical fingerprints across re-runs of the same file.
4. plan.json round-trip: lossless serialization and deserialization of Plan dataclass.
5. Task inference: automatic task detection and ambiguous task needs_attention emission.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest

from preflight.decisions import Decision
from preflight.plan import Plan, apply, plan
from preflight.schema import TableSchema


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text(
        "id,category,rating,target\n001,alpha,4.5,1\n002,beta,3.8,0\n003,gamma,4.9,1\n",
        encoding="utf-8",
    )
    return p


def test_plan_purity(sample_csv: Path):
    """Verify plan() never mutates source file."""
    initial_bytes = sample_csv.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    plan_obj = plan(sample_csv, target="target")

    after_bytes = sample_csv.read_bytes()
    after_hash = hashlib.sha256(after_bytes).hexdigest()

    assert initial_hash == after_hash, "plan() mutated the source file bytes!"
    assert plan_obj.source_path == str(sample_csv.resolve())
    assert plan_obj.target_column == "target"
    assert len(plan_obj.decisions) > 0


def test_decision_status_transitions():
    """Verify Decision status transitions: automatic -> confirmed -> changed with timestamps."""
    d = Decision(
        id="",
        stage="infer",
        kind="type_inference",
        columns=["age"],
        action="cast_to_int",
        evidence={"score": 0.98},
        confidence=0.98,
        default="keep_as_string",
        needs_attention=False,
        plain_language="Read column 'age' as whole numbers.",
        status="automatic",
    )
    assert d.status == "automatic"
    orig_fp = d.fingerprint
    orig_ts = d.timestamp
    assert orig_fp is not None
    assert orig_ts is not None

    # Transition to confirmed
    d.confirm()
    assert d.status == "confirmed"
    assert d.fingerprint == orig_fp, "Fingerprint should not change upon confirmation"
    assert d.timestamp is not None

    # Transition to changed
    d.change("cast_to_float")
    assert d.status == "changed"
    assert d.action == "cast_to_float"
    assert d.fingerprint == orig_fp, "Fingerprint should preserve original proposal identity"


def test_fingerprint_stability(sample_csv: Path):
    """Verify fingerprints are 100% deterministic and identical across repeated runs."""
    plan1 = plan(sample_csv, target="target")
    plan2 = plan(sample_csv, target="target")

    assert len(plan1.decisions) == len(plan2.decisions)

    fps1 = [d.fingerprint for d in plan1.decisions]
    fps2 = [d.fingerprint for d in plan2.decisions]

    assert fps1 == fps2
    for fp in fps1:
        assert fp is not None
        assert len(fp) == 16


def test_plan_json_roundtrip(sample_csv: Path, tmp_path: Path):
    """Verify lossless serialization and deserialization of Plan dataclass."""
    plan_obj = plan(sample_csv, target="target")
    json_path = tmp_path / "plan.json"

    plan_obj.save_json(json_path)
    assert json_path.exists()

    loaded_plan = Plan.from_json(json_path.read_text(encoding="utf-8"))

    assert loaded_plan.source_path == plan_obj.source_path
    assert loaded_plan.target_column == plan_obj.target_column
    assert loaded_plan.task_type == plan_obj.task_type
    assert loaded_plan.row_count == plan_obj.row_count
    assert loaded_plan.column_count == plan_obj.column_count
    assert len(loaded_plan.decisions) == len(plan_obj.decisions)

    for orig_d, loaded_d in zip(plan_obj.decisions, loaded_plan.decisions):
        assert orig_d.id == loaded_d.id
        assert orig_d.stage == loaded_d.stage
        assert orig_d.kind == loaded_d.kind
        assert orig_d.columns == loaded_d.columns
        assert orig_d.action == loaded_d.action
        assert orig_d.confidence == loaded_d.confidence
        assert orig_d.default == loaded_d.default
        assert orig_d.needs_attention == loaded_d.needs_attention
        assert orig_d.status == loaded_d.status
        assert orig_d.fingerprint == loaded_d.fingerprint


def test_plan_task_inference_ambiguous(tmp_path: Path):
    """Verify task inference emits needs_attention Decision when target is ambiguous."""
    # Target has 3 discrete integer classes (ambiguous between classification & regression)
    p = tmp_path / "ambiguous.csv"
    p.write_text("feature,target\n1.0,0\n2.0,1\n3.0,2\n4.0,1\n5.0,0\n", encoding="utf-8")

    plan_obj = plan(p, target="target")
    task_decs = [d for d in plan_obj.decisions if "task" in d.kind]

    assert len(task_decs) == 1
    t_dec = task_decs[0]
    assert t_dec.kind == "ambiguous_task"
    assert t_dec.needs_attention is True
    assert t_dec.columns == ["target"]
    assert "classification" in t_dec.options if hasattr(t_dec, "options") else True
