"""Answer persistence keyed by Decision.fingerprint (Phase 5 / M4)."""

import hashlib
import json
from pathlib import Path

import pytest

from conftest import pick, recommended, session_of
from preflight import resolve
from preflight.interact.modes import StrictResolutionError
from preflight.interact.store import STORE_VERSION, AnswerStore
from preflight.plan import plan


def tree_hash(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def fail_if_called(qs):
    raise AssertionError(f"asked again: {[q.text for q in qs]}")


def test_rerun_does_not_re_ask_and_marks_confirmed(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    second, _ = resolve(messy_csv, mode="ask", channel=fail_if_called, out_dir=out, target="target")
    reused = [d for d in second.decisions if d.history]
    assert reused
    for d in reused:
        assert d.status == "confirmed"
        assert d.history[-1]["source"].startswith("stored answer")
        assert d.history[-1]["timestamp"]
        assert d.needs_attention is False


def test_stored_answers_satisfy_strict_mode(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    resolve(messy_csv, mode="strict", out_dir=out, target="target")  # must not raise


def test_stored_changed_answer_is_reapplied(messy_csv, tmp_path):
    out = tmp_path / "out"
    first, _ = resolve(messy_csv, mode="ask", channel=pick(2), out_dir=out, target="target")
    chosen = {d.id: d.action for d in first.decisions if d.history}
    second, _ = resolve(messy_csv, mode="ask", channel=fail_if_called, out_dir=out, target="target")
    for d in second.decisions:
        if d.history:
            assert d.action == chosen[d.id]
            assert d.status == "confirmed"


def test_changed_fingerprint_is_asked_again(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    old = json.loads((out / "answers.json").read_text())["answers"]

    # same column names, different values -> different evidence -> different fingerprints
    rows = ["id,zip,mileage,when,score,target"]
    for i in range(20):
        rows.append(f"{i * 7 + 1},0{5000 + i * 3},{30 + i} mpg,1{i % 3}/1{i % 2}/2021,{i % 5},{i % 2}")
    messy_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    seen = []

    def spy(qs):
        seen.extend(qs)
        return recommended(qs)

    resolve(messy_csv, mode="ask", channel=spy, out_dir=out, target="target")
    assert seen, "changed data must be asked again"
    assert not ({q.fingerprint for q in seen} & set(old)), "no old fingerprint was reused"
    now = json.loads((out / "answers.json").read_text())["answers"]
    assert set(old) <= set(now), "old answers are kept, not overwritten"


def test_partial_match_asks_only_what_changed(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    txt = messy_csv.read_text()
    messy_csv.write_text(txt.replace("02100", "ZZ100", 1), encoding="utf-8")  # only the zip column changes
    seen = []
    resolve(messy_csv, mode="ask", channel=lambda qs: (seen.extend(qs), recommended(qs))[1], out_dir=out, target="target")
    assert seen and {q.column for q in seen} <= {"zip"}


def test_stale_answers_are_never_misapplied(messy_plan, tmp_path):
    session = session_of(messy_plan)
    q = session.questions[0]
    store = AnswerStore(tmp_path / "out")

    # answer filed under a different fingerprint
    store.record(q, q.options[0], "confirmed")
    store._answers["ffffffffffffffff"] = store._answers.pop(q.fingerprint)
    assert store.lookup(q) is None

    # right fingerprint, but the option no longer exists / maps elsewhere
    store2 = AnswerStore(tmp_path / "out2")
    rec = store2.record(q, q.options[0], "confirmed")
    rec.answer = "An option that is no longer offered"
    assert store2.lookup(q) is None
    rec.answer = q.options[0]
    rec.action = "some_other_action"
    assert store2.lookup(q) is None

    # key and record disagree
    store3 = AnswerStore(tmp_path / "out3")
    store3.record(q, q.options[0], "confirmed")
    store3._answers[q.fingerprint].fingerprint = "0000000000000000"
    assert store3.lookup(q) is None


def test_stale_answer_leads_to_a_question_not_a_silent_apply(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    path = out / "answers.json"
    data = json.loads(path.read_text())
    for rec in data["answers"].values():
        rec["answer"] = "A choice from an older release"
    path.write_text(json.dumps(data))
    with pytest.raises(StrictResolutionError):
        resolve(messy_csv, mode="strict", out_dir=out, target="target")


def test_store_is_versioned_json_inside_out_dir_only(messy_csv, tmp_path):
    out = tmp_path / "out"
    before = tree_hash(messy_csv.parent)
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    assert tree_hash(messy_csv.parent) == before, "nothing written next to the source file"
    data = json.loads((out / "answers.json").read_text())
    assert data["store_version"] == STORE_VERSION
    for fp, rec in data["answers"].items():
        assert rec["fingerprint"] == fp and rec["timestamp"] and rec["status"] in ("confirmed", "changed")


def test_out_dir_beside_the_source_file_is_refused(messy_plan):
    with pytest.raises(ValueError, match="next to the source"):
        resolve(messy_plan, mode="defaults", out_dir=Path(messy_plan.source_path).parent)


def test_unreadable_store_warns_asks_again_and_keeps_a_backup(messy_csv, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "answers.json").write_text("{not json")
    seen = []
    with pytest.warns(UserWarning, match="Ignoring stored answers"):
        resolve(messy_csv, mode="ask", channel=lambda qs: (seen.extend(qs), recommended(qs))[1], out_dir=out, target="target")
    assert seen
    assert (out / "answers.json.bak").read_text() == "{not json"
    assert json.loads((out / "answers.json").read_text())["store_version"] == STORE_VERSION


def test_unknown_store_version_is_not_trusted(messy_csv, tmp_path):
    out = tmp_path / "out"
    resolve(messy_csv, mode="ask", channel=recommended, out_dir=out, target="target")
    p = out / "answers.json"
    data = json.loads(p.read_text())
    data["store_version"] = 999
    p.write_text(json.dumps(data))
    with pytest.warns(UserWarning), pytest.raises(StrictResolutionError):
        resolve(messy_csv, mode="strict", out_dir=out, target="target")


def test_status_changes_are_recorded_with_timestamps_on_the_decision(messy_plan, tmp_path):
    resolved, _ = resolve(messy_plan, mode="ask", channel=pick(1), out_dir=tmp_path / "out")
    for d in resolved.decisions:
        for h in d.history:
            assert h["status"] in ("automatic", "confirmed", "changed")
            assert h["timestamp"] and h["source"]
    assert any(d.history for d in resolved.decisions)


def test_fingerprint_survives_resolution(messy_csv, tmp_path):
    p = plan(messy_csv, target="target")
    before = {d.id: d.fingerprint for d in p.decisions}
    resolve(p, mode="ask", channel=pick(2), out_dir=tmp_path / "out")
    assert before == {d.id: d.fingerprint for d in p.decisions}
