"""ask / defaults / strict behaviour and mode auto-selection (Phase 5 / M4)."""

from pathlib import Path

import pandas as pd
import pytest

from conftest import pick, recommended
from preflight import apply, resolve
from preflight.interact import modes
from preflight.interact.channels import NonInteractiveError, ReviewPending
from preflight.interact.modes import StrictResolutionError, resolve_mode
from preflight.plan import plan


# --- defaults ---------------------------------------------------------------

def test_defaults_applies_safe_default_marks_automatic_and_logs_it(messy_plan, tmp_path):
    open_before = {d.id: d for d in messy_plan.decisions if d.needs_attention}
    assert open_before
    resolved, _ = resolve(messy_plan, mode="defaults", out_dir=tmp_path / "out")
    for d in resolved.decisions:
        if d.id in open_before:
            assert d.action == d.default
            assert d.status == "automatic"
            assert d.needs_attention is False
            assert d.history and d.history[-1]["status"] == "automatic"
            assert "defaults" in d.history[-1]["source"]
            assert d.history[-1]["timestamp"]


def test_defaults_does_not_write_answers(messy_plan, tmp_path):
    out = tmp_path / "out"
    resolve(messy_plan, mode="defaults", out_dir=out)
    assert not (out / "answers.json").exists()
    assert (out / "plan.json").exists() and (out / "plan.pdf").exists()


# --- strict -----------------------------------------------------------------

def test_strict_lists_every_unresolved_item(messy_plan, tmp_path):
    expected = [d for d in messy_plan.decisions if d.needs_attention]
    with pytest.raises(StrictResolutionError) as exc:
        resolve(messy_plan, mode="strict", out_dir=tmp_path / "out")
    msg = str(exc.value)
    assert len(exc.value.unresolved_questions) == len(expected) > 1
    for d in expected:
        assert d.id in msg
        assert (d.columns[0] if d.columns else "(whole file)") in msg
    for q in exc.value.unresolved_questions:
        assert q.text in msg


def test_strict_leaves_decisions_untouched_and_writes_no_plan(messy_plan, tmp_path):
    before = [(d.action, d.status, d.needs_attention, list(d.history)) for d in messy_plan.decisions]
    out = tmp_path / "out"
    with pytest.raises(StrictResolutionError):
        resolve(messy_plan, mode="strict", out_dir=out)
    assert before == [(d.action, d.status, d.needs_attention, list(d.history)) for d in messy_plan.decisions]
    assert not (out / "plan.pdf").exists()


def test_strict_passes_when_nothing_is_open(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    f = src / "clean.csv"
    f.write_text("name,city\nann,paris\nbob,rome\ncat,oslo\ndan,lima\n", encoding="utf-8")
    p = plan(f)
    assert not [d for d in p.decisions if d.needs_attention]
    resolve(p, mode="strict", out_dir=tmp_path / "out")


# --- ask --------------------------------------------------------------------

def test_ask_confirm_vs_change_statuses(messy_plan, tmp_path):
    open_ids = {d.id: (d.action, d.default) for d in messy_plan.decisions if d.needs_attention}
    resolved, _ = resolve(messy_plan, mode="ask", channel=recommended, out_dir=tmp_path / "out")
    for d in resolved.decisions:
        if d.id in open_ids:
            proposed, default = open_ids[d.id]
            same = proposed == default or {proposed, default} <= {"keep_as_string", "keep_as_text"}
            assert d.status == ("confirmed" if same else "changed")
            assert d.action == default or d.status == "confirmed"
            assert d.history[-1]["source"] == "ask: callback channel"
            assert d.history[-1]["answer"]


def test_ask_applying_the_proposed_option_confirms(messy_plan, tmp_path):
    proposed = {d.id: d.action for d in messy_plan.decisions if d.needs_attention}

    def choose_proposed(qs):
        return {q.question_id: next(o for o, a in q.option_actions.items() if a == proposed[q.decision_id]) for q in qs}

    resolved, _ = resolve(messy_plan, mode="ask", channel=choose_proposed, out_dir=tmp_path / "out")
    for d in resolved.decisions:
        if d.id in proposed:
            assert d.status == "confirmed" and d.action == proposed[d.id]


def test_answers_flow_into_the_final_plan_files(messy_plan, tmp_path):
    out = tmp_path / "out"
    resolve(messy_plan, mode="ask", channel=pick(1), out_dir=out)
    import json

    data = json.loads((out / "plan.json").read_text())
    assert not [d for d in data["decisions"] if d["needs_attention"]]
    assert any(d["status"] in ("confirmed", "changed") and d["history"] for d in data["decisions"])
    assert (out / "plan.pdf").stat().st_size > 0


def test_ask_never_substitutes_default_for_bad_answer(messy_plan, tmp_path):
    from preflight.interact.question import InvalidAnswerError

    with pytest.raises(InvalidAnswerError):
        resolve(messy_plan, mode="ask", channel=lambda qs: {q.question_id: "nonsense" for q in qs}, out_dir=tmp_path / "out")
    assert all(d.status == "automatic" for d in messy_plan.decisions)
    assert [d for d in messy_plan.decisions if d.needs_attention], "nothing was resolved"


def test_ask_in_terminal_without_tty_errors_instead_of_blocking(messy_plan, tmp_path):
    # pytest has no TTY; the default terminal channel must refuse, not wait.
    with pytest.raises(NonInteractiveError):
        resolve(messy_plan, mode="ask", out_dir=tmp_path / "out")


# --- mode selection ---------------------------------------------------------

def test_explicit_mode_beats_auto_selection(monkeypatch):
    monkeypatch.setattr(modes, "is_interactive_context", lambda: True)
    assert resolve_mode("strict") == "strict"
    assert resolve_mode("defaults") == "defaults"
    assert resolve_mode(None) == "ask"
    monkeypatch.setattr(modes, "is_interactive_context", lambda: False)
    assert resolve_mode("ask") == "ask"
    assert resolve_mode("defaults") == "defaults"
    assert resolve_mode(None) == "strict"


def test_non_interactive_default_is_strict_end_to_end(messy_plan, tmp_path):
    # Under pytest there is no TTY, so no mode means strict.
    with pytest.raises(StrictResolutionError):
        resolve(messy_plan, out_dir=tmp_path / "out")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        resolve_mode("yolo")


def test_no_tty_under_test_runner():
    assert modes.is_interactive_context() is False


def test_notebook_kernel_counts_as_interactive(monkeypatch):
    import sys
    import types

    class ZMQInteractiveShell:
        pass

    fake = types.SimpleNamespace(get_ipython=lambda: ZMQInteractiveShell())
    monkeypatch.setitem(sys.modules, "IPython", fake)
    assert modes.in_notebook_kernel() is True
    assert modes.is_interactive_context() is True


# --- answers change what the run does --------------------------------------

def test_defaults_mode_keeps_ambiguous_columns_as_text(messy_csv, tmp_path):
    res = apply(plan(messy_csv, target="target"), mode="defaults", out_dir=tmp_path / "out")
    assert not pd.api.types.is_integer_dtype(res.data["id"])
    assert res.data["zip"].tolist()[0] == "02100", "leading zeros survive the safe default"


def test_ask_answer_changes_how_the_data_is_read(messy_csv, tmp_path):
    def whole_numbers(qs):
        return {
            q.question_id: ("Convert to whole numbers" if "Convert to whole numbers" in q.options else q.recommended_default)
            for q in qs
        }

    res = apply(plan(messy_csv, target="target"), mode="ask", channel=whole_numbers, out_dir=tmp_path / "out")
    assert pd.api.types.is_integer_dtype(res.data["id"])


def test_apply_without_mode_is_unchanged(messy_csv):
    res = apply(plan(messy_csv, target="target"))
    assert pd.api.types.is_integer_dtype(res.data["id"])


def test_apply_mode_requires_a_plan(messy_csv):
    from preflight.plan import ApplyContext

    with pytest.raises(ValueError):
        apply(ApplyContext(source_path=messy_csv), mode="defaults")


def test_reading_actions_in_force_are_used(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    f = src / "latin.csv"
    f.write_bytes("name,city\nzo\xeb,m\xfcnchen\nren\xe9,k\xf6ln\n".encode("latin-1"))
    p = plan(f)
    enc = next(d for d in p.decisions if d.kind == "encoding")

    enc.action = "use_encoding_utf-8"
    with pytest.raises(ValueError, match="text format"):
        apply(p)

    enc.action = "use_encoding_iso-8859-1"
    assert apply(p).data["name"].tolist() == ["zoë", "rené"]


def test_header_and_separator_actions_in_force_are_used(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    f = src / "t.csv"
    f.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    p = plan(f)
    next(d for d in p.decisions if d.kind == "header").action = "no_header"
    df = apply(p).data
    assert list(df.columns) == ["column_1", "column_2"] and len(df) == 3

    p = plan(f)
    next(d for d in p.decisions if d.kind == "dialect").action = "separate_by_,"
    assert list(apply(p).data.columns) == ["a;b"]
