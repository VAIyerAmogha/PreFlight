"""Question model, option bounds, previews, and trigger registry (Phase 5 / M4)."""

import pytest

from conftest import session_of
from preflight.decisions import Decision
from preflight.interact.question import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    Question,
    QuestionError,
    build_question_from_decision,
    register_question_builder,
)
from preflight.interact.triggers import (
    TriggerCategory,
    TriggerRegistry,
    TriggerRule,
    classify_decision_trigger,
    default_registry,
)
from preflight.plan import plan


def test_one_batched_session_covers_every_open_decision(messy_plan, messy_session):
    open_ids = [d.id for d in messy_plan.decisions if d.needs_attention]
    assert [q.decision_id for q in messy_session.questions] == open_ids
    assert len(open_ids) > 1


def test_option_bounds_and_default_membership(messy_session):
    for q in messy_session.questions:
        assert MIN_OPTIONS <= len(q.options) <= MAX_OPTIONS
        assert len(set(q.options)) == len(q.options)
        assert q.recommended_default in q.options
        assert set(q.option_actions) == set(q.options)


def test_recommended_default_is_the_decisions_own_default(messy_plan, messy_session):
    by_id = {d.id: d for d in messy_plan.decisions}
    for q in messy_session.questions:
        dec = by_id[q.decision_id]
        assert q.option_actions[q.recommended_default] == dec.default or (
            q.option_actions[q.recommended_default] in ("keep_as_string", "keep_as_text")
            and dec.default in ("keep_as_string", "keep_as_text")
        )
        # the proposed action is offered too, so a person can confirm it
        assert any(a == dec.action for a in q.option_actions.values()) or dec.action in (
            "keep_as_text",
            "keep_as_string",
        )


def test_column_questions_carry_a_real_preview(messy_session):
    column_questions = [q for q in messy_session.questions if q.column]
    assert column_questions
    zip_q = next(q for q in column_questions if q.column == "zip")
    assert zip_q.preview is not None
    assert zip_q.preview["before"][:2] == ["02100", "02101"], "preview must show the file's real values"
    assert set(zip_q.preview["after"]) == set(zip_q.options)
    for q in column_questions:
        assert q.preview is not None and q.preview["before"], q.column
        assert all(len(v) == len(q.preview["before"]) for v in q.preview["after"].values())


def test_preview_shows_what_would_go_wrong(messy_session):
    when_q = next(q for q in messy_session.questions if q.column == "when")
    after = when_q.preview["after"]
    assert "Convert to dates and times" in after


def test_preview_blank_marker_is_explained():
    from preflight.interact.preview import BLANK, build_column_preview
    import pandas as pd

    pv = build_column_preview(
        pd.Series(["1", "2", "oops"]),
        {"Convert to whole numbers": "cast_to_int", "Keep as plain text": "keep_as_string"},
        {},
    )
    assert "oops" in pv["before"], "a value that fails conversion is included on purpose"
    assert pv["after"]["Convert to whole numbers"][pv["before"].index("oops")] == BLANK
    assert "note" in pv


def test_preview_with_no_values_says_so_instead_of_inventing(tmp_path):
    from preflight.interact.preview import build_column_preview

    pv = build_column_preview(None, {"A": "keep_as_string", "B": "cast_to_int"}, {})
    assert pv["before"] == [] and "note" in pv


def test_task_question_has_preview_and_three_goals(task_csv):
    p = plan(task_csv, target="target")
    q = next(q for q in session_of(p).questions if q.decision_id.startswith("dec_infer_target") and "classification" in q.option_actions.values())
    assert q.category == TriggerCategory.AMBIGUOUS_STRUCTURE_OR_TASK
    assert set(q.option_actions.values()) == {"classification", "regression", "none"}
    assert q.preview and q.preview["before"]


def test_table_level_question_has_no_preview_and_plain_options():
    dec = Decision(
        id="", stage="reading", kind="encoding", columns=[], action="use_encoding_iso-8859-1",
        evidence={"encoding": "iso-8859-1", "confidence": 0.6}, confidence=0.6,
        default="use_encoding_utf-8", needs_attention=True, plain_language="x",
    )
    q = build_question_from_decision(dec)
    assert q.column is None and q.preview is None
    assert q.category == TriggerCategory.LOW_CONFIDENCE_STRUCTURE
    assert q.option_actions[q.recommended_default] == "use_encoding_utf-8"
    assert "use_encoding_iso-8859-1" in q.option_actions.values()


def test_dialect_question_offers_alternatives_without_losing_default_or_proposal():
    dec = Decision(
        id="", stage="reading", kind="dialect", columns=[], action="separate_by_;",
        evidence={"delimiter": ";", "confidence": 0.5}, confidence=0.5,
        default="separate_by_,", needs_attention=True, plain_language="x",
    )
    q = build_question_from_decision(dec)
    acts = list(q.option_actions.values())
    assert acts[0] == "separate_by_," and "separate_by_;" in acts and len(acts) == MAX_OPTIONS


def test_question_never_silently_repairs_itself():
    base = dict(
        question_id="q", decision_id="d", fingerprint="f", category="ambiguous_type", column=None,
        text="t", explanation="e", evidence_summary="s",
    )
    with pytest.raises(QuestionError):
        Question(**base, options=["only"], option_actions={"only": "a"}, recommended_default="only")
    five = [f"o{i}" for i in range(5)]
    with pytest.raises(QuestionError):
        Question(**base, options=five, option_actions={o: o for o in five}, recommended_default="o0")
    with pytest.raises(QuestionError):
        Question(**base, options=["a", "b"], option_actions={"a": "a", "b": "b"}, recommended_default="c")
    with pytest.raises(QuestionError):  # column question without a preview
        Question(**{**base, "column": "x"}, options=["a", "b"], option_actions={"a": "a", "b": "b"}, recommended_default="a")


def test_decision_with_single_choice_and_no_builder_raises_naming_the_kind():
    dec = Decision(
        id="", stage="clean", kind="mystery_kind", columns=[], action="same", evidence={},
        confidence=0.1, default="same", needs_attention=True, plain_language="x",
    )
    with pytest.raises((QuestionError, ValueError)) as exc:
        build_question_from_decision(dec)
    assert "mystery_kind" in str(exc.value)


def test_plugin_builder_lets_a_new_kind_ask(monkeypatch):
    import preflight.interact.question as qmod

    monkeypatch.setitem(qmod._BUILDERS, "drop_column_demo", lambda d: ["drop_column", "keep_column"])
    reg = TriggerRegistry()
    reg.register(TriggerRule(TriggerCategory.DESTRUCTIVE_ACTION, "demo", lambda d, c: d.kind == "drop_column_demo"))
    monkeypatch.setattr("preflight.interact.triggers._DEFAULT_REGISTRY", reg)
    dec = Decision(
        id="", stage="clean", kind="drop_column_demo", columns=["x"], action="drop_column", evidence={},
        confidence=0.5, default="keep_column", needs_attention=True, plain_language="x",
    )
    q = build_question_from_decision(dec)
    assert q.category == TriggerCategory.DESTRUCTIVE_ACTION and len(q.options) == 2


# --- trigger registry -------------------------------------------------------

def test_eight_categories_registered_two_are_empty_plugin_points():
    reg = TriggerRegistry()
    assert sorted(TriggerCategory.ALL_CATEGORIES) == sorted(
        reg.implemented_categories() + reg.empty_plugin_categories()
    )
    assert len(TriggerCategory.ALL_CATEGORIES) == 8
    assert sorted(reg.empty_plugin_categories()) == [
        TriggerCategory.DESTRUCTIVE_ACTION,
        TriggerCategory.SUSPECTED_LEAKAGE,
    ]


def test_empty_plugin_points_never_match_and_can_be_filled():
    reg = TriggerRegistry()
    dec = Decision(
        id="", stage="clean", kind="drop_column", columns=["x"], action="drop", evidence={},
        confidence=0.5, default="keep", needs_attention=True, plain_language="x",
    )
    assert reg.classify(dec) is None
    reg.register(TriggerRule(TriggerCategory.DESTRUCTIVE_ACTION, "drops", lambda d, c: d.kind == "drop_column"))
    assert reg.classify(dec) == TriggerCategory.DESTRUCTIVE_ACTION
    assert TriggerCategory.DESTRUCTIVE_ACTION not in reg.empty_plugin_categories()


def test_unrecognised_decision_raises_rather_than_misfiling():
    dec = Decision(
        id="", stage="clean", kind="drop_column", columns=["x"], action="drop", evidence={},
        confidence=0.5, default="keep", needs_attention=True, plain_language="x",
    )
    with pytest.raises(ValueError, match="drop_column"):
        classify_decision_trigger(dec)


def test_implemented_triggers_classify_real_decisions(messy_session):
    cats = {q.column: q.category for q in messy_session.questions if q.column}
    assert cats["mileage"] == TriggerCategory.LOSSY_COERCION
    assert cats["when"] == TriggerCategory.LOSSY_COERCION
    assert cats["id"] == TriggerCategory.AMBIGUOUS_TYPE


def test_disguised_missing_and_high_anomaly_use_column_evidence():
    import pandas as pd
    from preflight.interact.triggers import TriggerContext, anomaly_rate
    from preflight.schema import ColumnRawMetadata

    dec = Decision(
        id="", stage="infer", kind="type_inference", columns=["v"], action="cast_to_int",
        evidence={}, confidence=0.6, default="keep_as_string", needs_attention=True, plain_language="x",
    )
    meta = ColumnRawMetadata(name="v", position=0, missing_markers=[
        {"marker": "999", "reason": "contextually_suspicious: 999 far above distribution max 40", "count": 2}
    ])
    assert default_registry().classify(dec, TriggerContext(column=meta)) == TriggerCategory.DISGUISED_MISSING

    series = pd.Series(["1", "2", "x", "y", "5", "6", "7", "8", "9", "10"])
    assert anomaly_rate(dec, series) == pytest.approx(0.2)
    assert default_registry().classify(dec, TriggerContext(raw_series=series)) == TriggerCategory.HIGH_ANOMALY_RATE
    ok = pd.Series([str(i) for i in range(30)])
    assert default_registry().classify(dec, TriggerContext(raw_series=ok)) == TriggerCategory.AMBIGUOUS_TYPE
