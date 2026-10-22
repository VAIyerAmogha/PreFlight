"""
The four channels: terminal, review_file, callback, notebook (Phase 5 / M4).

NOTE ON THE NOTEBOOK CHANNEL: it is tested headless with a mocked ipywidgets module
only. The widget UI has NOT been exercised in a real Jupyter notebook.
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import pick, recommended
from preflight import resolve
from preflight.interact.channels import (
    CallbackChannel,
    NonInteractiveError,
    NotebookChannel,
    ReviewFileChannel,
    ReviewPending,
    TerminalChannel,
)
from preflight.interact.channels import notebook as nb_mod
from preflight.interact.channels.review_file import ReviewFileError, load_review_file
from preflight.interact.channels.terminal import format_screen
from preflight.interact.question import InvalidAnswerError, MissingAnswerError


# --- terminal ----------------------------------------------------------------

def run_terminal(session, text, **kw):
    out = io.StringIO()
    ch = TerminalChannel(stdin=io.StringIO(text), stdout=out, use_rich=False, **kw)
    return ch.review(session), out.getvalue()


def test_terminal_batched_screen_then_answers(messy_session):
    n = len(messy_session.questions)
    lines = "\n".join(["2"] * n) + "\n"
    answers, screen = run_terminal(messy_session, lines)
    assert screen.count("<- recommended") == n
    for q in messy_session.questions:
        assert q.text in screen and q.options[0] in screen
        assert answers[q.question_id] == q.options[1]
    # ONE screen: every question is printed before the first prompt
    assert screen.index(messy_session.questions[-1].text) < screen.index("Item 1 of")


def test_terminal_accepts_number_label_and_explicit_default(messy_session):
    qs = messy_session.questions
    text = "1\n" + qs[1].options[1] + "\n" + "default\n" * (len(qs) - 2)
    answers, _ = run_terminal(messy_session, text)
    assert answers[qs[0].question_id] == qs[0].options[0]
    assert answers[qs[1].question_id] == qs[1].options[1]
    assert answers[qs[2].question_id] == qs[2].recommended_default


@pytest.mark.parametrize("bad", ["9", "0", "banana", "-1"])
def test_terminal_rejects_invalid_answer_naming_the_question(messy_session, bad):
    first = messy_session.questions[0]
    with pytest.raises(InvalidAnswerError) as exc:
        run_terminal(messy_session, f"{bad}\n")
    assert first.question_id in str(exc.value)


def test_terminal_empty_line_is_not_an_answer(messy_session):
    first = messy_session.questions[0]
    with pytest.raises(MissingAnswerError) as exc:
        run_terminal(messy_session, "\n")
    assert first.question_id in str(exc.value)


def test_terminal_eof_is_a_missing_answer_not_a_default(messy_session):
    with pytest.raises(MissingAnswerError):
        run_terminal(messy_session, "")


def test_terminal_uses_builtin_input_as_baseline(messy_session, monkeypatch):
    replies = iter(["1"] * len(messy_session.questions))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    ch = TerminalChannel(stdout=io.StringIO(), use_rich=False, require_tty=False)
    answers = ch.review(messy_session)
    assert answers[messy_session.questions[0].question_id] == messy_session.questions[0].options[0]


def test_terminal_without_tty_refuses_to_prompt(messy_session):
    with pytest.raises(NonInteractiveError):
        TerminalChannel(stdout=io.StringIO()).review(messy_session)


def test_terminal_rich_screen_when_installed(messy_session):
    pytest.importorskip("rich")
    out = io.StringIO()
    ch = TerminalChannel(stdin=io.StringIO("1\n" * len(messy_session.questions)), stdout=out)
    ch.review(messy_session)
    text = out.getvalue()
    assert messy_session.questions[0].text in text and "recommended" in text


def test_rich_screen_does_not_choke_on_markup_in_data():
    pytest.importorskip("rich")
    from preflight.interact.question import Question, ReviewSession

    q = Question(
        question_id="q1", decision_id="d", fingerprint="f", category="ambiguous_type", column="[bold]x[/]",
        text="How should column '[bold]x[/]' be read?", explanation="e", evidence_summary="s",
        options=["A [red]", "B"], option_actions={"A [red]": "a", "B": "b"}, recommended_default="B",
        preview={"before": ["[/oops]"], "after": {"A [red]": ["[/oops]"], "B": ["b"]}},
    )
    out = io.StringIO()
    TerminalChannel(stdin=io.StringIO("2\n"), stdout=out).review(ReviewSession("/x.csv", [q]))
    assert "[/oops]" in out.getvalue()


# --- review_file ---------------------------------------------------------------

def test_review_file_round_trip_with_edited_answers(messy_csv, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ReviewPending) as pending:
        resolve(messy_csv, mode="ask", channel="review_file", out_dir=out, target="target")
    rf = Path(pending.value.path)
    assert rf == out / "review.json"

    data = json.loads(rf.read_text())
    assert data["questions"], "questions written"
    for item in data["questions"]:
        assert item["answer"] is None
        assert 2 <= len(item["options"]) <= 4
        assert item["recommended_default"] in item["options"]
        assert "preview" in item
    assert not (out / "plan.pdf").exists(), "nothing is resolved before answers exist"

    for i, item in enumerate(data["questions"]):
        item["answer"] = item["options"][i % len(item["options"])]
    rf.write_text(json.dumps(data))

    plan_obj, pdf = resolve(str(rf))  # resolve(path) reads and validates
    assert pdf.exists()
    assert not [d for d in plan_obj.decisions if d.needs_attention]
    by_id = {d.id: d for d in plan_obj.decisions}
    for item in data["questions"]:
        assert any(h["answer"] == item["answer"] for d in plan_obj.decisions for h in d.history)
    assert json.loads((out / "answers.json").read_text())["answers"]


def test_review_file_second_call_with_same_plan_object_also_works(messy_plan, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ReviewPending):
        resolve(messy_plan, mode="ask", channel="review_file", out_dir=out)
    data = json.loads((out / "review.json").read_text())
    for item in data["questions"]:
        item["answer"] = item["recommended_default"]
    (out / "review.json").write_text(json.dumps(data))
    resolve(messy_plan, mode="ask", channel="review_file", out_dir=out)
    assert not [d for d in messy_plan.decisions if d.needs_attention]


def _write_review(messy_csv, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ReviewPending):
        resolve(messy_csv, mode="ask", channel="review_file", out_dir=out, target="target")
    return out / "review.json", json.loads((out / "review.json").read_text())


def test_review_file_blank_answer_raises_naming_question(messy_csv, tmp_path):
    rf, data = _write_review(messy_csv, tmp_path)
    for item in data["questions"][1:]:
        item["answer"] = item["recommended_default"]
    rf.write_text(json.dumps(data))
    with pytest.raises(MissingAnswerError) as exc:
        resolve(str(rf))
    assert data["questions"][0]["question_id"] in str(exc.value)


def test_review_file_invalid_answer_raises_naming_question(messy_csv, tmp_path):
    rf, data = _write_review(messy_csv, tmp_path)
    for item in data["questions"]:
        item["answer"] = item["recommended_default"]
    data["questions"][2]["answer"] = "something else"
    rf.write_text(json.dumps(data))
    with pytest.raises(InvalidAnswerError) as exc:
        resolve(str(rf))
    assert data["questions"][2]["question_id"] in str(exc.value)


def test_review_file_for_changed_questions_is_rejected(messy_csv, tmp_path):
    rf, data = _write_review(messy_csv, tmp_path)
    for item in data["questions"]:
        item["answer"] = item["recommended_default"]
    data["questions"][0]["fingerprint"] = "0" * 16
    rf.write_text(json.dumps(data))
    with pytest.raises(ReviewFileError, match="does not match"):
        resolve(str(rf))


def test_non_review_json_is_not_mistaken_for_one(tmp_path):
    p = tmp_path / "other.json"
    p.write_text("{}")
    with pytest.raises(ReviewFileError):
        load_review_file(p)


# --- callback ------------------------------------------------------------------

def test_callback_receives_all_questions_in_one_call(messy_session):
    calls = []

    def handler(qs):
        calls.append(list(qs))
        return recommended(qs)

    answers = CallbackChannel(handler).review(messy_session)
    assert len(calls) == 1 and calls[0] == messy_session.questions
    assert set(answers) == {q.question_id for q in messy_session.questions}


def test_callback_missing_answer_names_question(messy_session):
    def handler(qs):
        a = recommended(qs)
        del a[qs[1].question_id]
        return a

    with pytest.raises(MissingAnswerError) as exc:
        CallbackChannel(handler).review(messy_session)
    assert messy_session.questions[1].question_id in str(exc.value)


def test_callback_unknown_question_id_and_wrong_return_type(messy_session):
    with pytest.raises(ValueError, match="unknown question"):
        CallbackChannel(lambda qs: {**recommended(qs), "q_nope": "x"}).review(messy_session)
    with pytest.raises(TypeError):
        CallbackChannel(lambda qs: ["a", "b"]).review(messy_session)
    with pytest.raises(TypeError):
        CallbackChannel("not callable")


# --- notebook (mocked widgets only) ---------------------------------------------

class FakeWidget:
    def __init__(self, *a, **kw):
        self.args, self.kw = a, kw
        self.value = kw.get("value")
        self._cb = None

    def on_click(self, fn):
        self._cb = fn

    def click(self):
        self._cb(self)


def fake_widgets():
    ns = SimpleNamespace(
        HTML=lambda v="": FakeWidget(value=v),
        RadioButtons=lambda **kw: FakeWidget(**kw),
        Button=lambda **kw: FakeWidget(**kw),
        VBox=lambda children: SimpleNamespace(children=children),
    )
    return ns


def make_notebook():
    shown = []
    return NotebookChannel(widgets=fake_widgets(), display_fn=shown.append), shown


@pytest.fixture(autouse=True)
def _clear_notebook_state():
    nb_mod._SUBMITTED.clear()
    yield
    nb_mod._SUBMITTED.clear()


def test_notebook_shows_form_and_does_not_block(messy_session):
    ch, shown = make_notebook()
    with pytest.raises(ReviewPending):
        ch.review(messy_session)
    assert len(shown) == 1
    radios = list(ch.controls.values())
    assert len(radios) == len(messy_session.questions)
    assert all(r.kw["value"] is None for r in radios), "nothing preselected"


def test_notebook_submit_then_resolve_again(messy_session):
    ch, _ = make_notebook()
    with pytest.raises(ReviewPending):
        ch.review(messy_session)
    for q in messy_session.questions:
        ch.controls[q.question_id].value = q.options[1]
    ch.on_submit(None)
    answers = ch.review(messy_session)
    assert answers == {q.question_id: q.options[1] for q in messy_session.questions}


def test_notebook_unanswered_question_is_not_defaulted(messy_session):
    ch, _ = make_notebook()
    with pytest.raises(ReviewPending):
        ch.review(messy_session)
    for q in messy_session.questions[1:]:
        ch.controls[q.question_id].value = q.options[0]
    ch.on_submit(None)  # first question left unanswered
    with pytest.raises(ReviewPending):  # not accepted: shows the form again
        ch.review(messy_session)


def test_notebook_invalid_value_is_rejected(messy_session):
    ch, _ = make_notebook()
    with pytest.raises(ReviewPending):
        ch.review(messy_session)
    for q in messy_session.questions:
        ch.controls[q.question_id].value = q.options[0]
    ch.controls[messy_session.questions[0].question_id].value = "not an option"
    ch.on_submit(None)
    assert not nb_mod._SUBMITTED


def test_notebook_invalid_answer_that_reaches_validation_names_question(messy_session):
    nb_mod._SUBMITTED[nb_mod._session_key(messy_session)] = {
        q.question_id: q.options[0] for q in messy_session.questions
    } | {messy_session.questions[0].question_id: "bad"}
    ch, _ = make_notebook()
    with pytest.raises(InvalidAnswerError) as exc:
        ch.review(messy_session)
    assert messy_session.questions[0].question_id in str(exc.value)


def test_notebook_without_ipywidgets_raises_clear_error(messy_session, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    with pytest.raises(ImportError, match=r"pypreflight\[notebook\]"):
        NotebookChannel().review(messy_session)


def test_notebook_form_builds_with_real_ipywidgets_objects_headless(messy_session):
    """Constructs real widget objects (no display, no kernel). Not a UI test."""
    pytest.importorskip("ipywidgets")
    ch = NotebookChannel(display_fn=lambda w: None)
    form = ch.build(messy_session)
    assert form.children
    assert all(r.value is None for r in ch.controls.values())


# --- invalid answers are rejected on EVERY channel ---------------------------------

def _bad(qs):
    out = recommended(qs)
    out[qs[0].question_id] = "not one of the options"
    return out


def test_every_channel_rejects_invalid_answers(messy_csv, messy_session, tmp_path):
    first = messy_session.questions[0].question_id
    n = len(messy_session.questions)

    with pytest.raises(InvalidAnswerError, match=first):
        CallbackChannel(_bad).review(messy_session)

    with pytest.raises(InvalidAnswerError, match=first):
        run_terminal(messy_session, "x\n" + "1\n" * (n - 1))

    rf, data = _write_review(messy_csv, tmp_path)
    for item in data["questions"]:
        item["answer"] = item["options"][0]
    data["questions"][0]["answer"] = "nope"
    rf.write_text(json.dumps(data))
    with pytest.raises(InvalidAnswerError, match=data["questions"][0]["question_id"]):
        resolve(str(rf))

    ch, _ = make_notebook()
    with pytest.raises(ReviewPending):
        ch.review(messy_session)
    for q in messy_session.questions:
        ch.controls[q.question_id].value = q.options[0]
    ch.controls[first].value = "nope"
    ch.on_submit(None)
    assert not nb_mod._SUBMITTED  # rejected; never reaches resolution


def test_format_screen_is_plain_text_without_rich(messy_session):
    screen = format_screen(messy_session)
    assert messy_session.source_path in screen
    for q in messy_session.questions:
        assert q.text in screen
