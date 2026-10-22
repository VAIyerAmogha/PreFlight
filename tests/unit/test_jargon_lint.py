"""
Jargon lint test suite for PreFlight v2.0 Plan PDF output (Phase 4 / M3).

Renders Plan PDFs across various hand-built messy files covering edge cases
and traps, extracts all text using pypdf, and asserts that NO banned technical
jargon terms appear anywhere in the document.
"""

from pathlib import Path
import re
import pytest
import pypdf

from preflight.plan import dry_run
from preflight.report.templates import BANNED_JARGON


# Extended list of banned technical jargon and raw encoding tokens
STRICT_BANNED_PATTERNS = [
    r"\bimputation\b",
    r"\bimpute\b",
    r"\bwinsoriz\w*",
    r"\bcardinality\b",
    r"\bdtypes?\b",
    r"\bcoerc\w*",
    r"\bheuristics?\b",
    r"\bpars\w*",
    r"\bschemas?\b",
    r"\bdelimit\w*",
    r"\butf-?8\b",
    r"\bwindows-1252\b",
    r"\biso-8859-1\b",
    r"\blatin-?1\b",
]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text across all pages of a PDF using pypdf."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def check_for_banned_jargon(text: str) -> None:
    """Assert that none of the strict banned patterns appear in text."""
    lower_text = text.lower()
    for pattern in STRICT_BANNED_PATTERNS:
        match = re.search(pattern, lower_text)
        assert match is None, (
            f"Found banned jargon '{match.group(0)}' (pattern: '{pattern}') in PDF text:\n{text}"
        )


def test_jargon_lint_standard_numeric(tmp_path: Path):
    """Verify standard tabular data produces jargon-free Plan PDF."""
    p = tmp_path / "numbers.csv"
    p.write_text("id,val1,val2\n1,10.5,100\n2,20.5,200\n3,30.5,300\n", encoding="utf-8")

    out_dir = tmp_path / "out1"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_leading_zero_ids(tmp_path: Path):
    """Verify ID columns with leading zeros produce jargon-free Plan PDF."""
    p = tmp_path / "zips.csv"
    p.write_text("zip,state\n02138,MA\n01001,MA\n07030,NJ\n", encoding="utf-8")

    out_dir = tmp_path / "out2"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_european_decimal_comma(tmp_path: Path):
    """Verify decimal comma floats produce jargon-free Plan PDF."""
    p = tmp_path / "euro.csv"
    p.write_text("item,price\ncoffee,3,50\ntea,2,75\ncake,4,20\n", encoding="utf-8")

    out_dir = tmp_path / "out3"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_ambiguous_dates_abstention(tmp_path: Path):
    """Verify ambiguous date abstention produces jargon-free Plan PDF."""
    p = tmp_path / "dates.csv"
    p.write_text("event,date\nmeeting,01/02/2020\nlunch,03/04/2020\ncall,05/06/2020\n", encoding="utf-8")

    out_dir = tmp_path / "out4"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_embedded_units_abstention(tmp_path: Path):
    """Verify embedded currency and physical units produce jargon-free Plan PDF."""
    p = tmp_path / "units.csv"
    p.write_text("car,mileage,cost\nA,15 kmpl,$12000\nB,18 kmpl,$15500\n", encoding="utf-8")

    out_dir = tmp_path / "out5"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_semicolon_separator(tmp_path: Path):
    """Verify semicolon separator produces jargon-free Plan PDF."""
    p = tmp_path / "semi.csv"
    p.write_text("code;label;count\n10;first;100\n20;second;200\n", encoding="utf-8")

    out_dir = tmp_path / "out6"
    _, pdf_path = dry_run(p, out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


def test_jargon_lint_ambiguous_task(tmp_path: Path):
    """Verify ambiguous task question produces jargon-free Plan PDF."""
    p = tmp_path / "task.csv"
    p.write_text("f1,f2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")

    out_dir = tmp_path / "out7"
    _, pdf_path = dry_run(p, target="target", out_dir=out_dir)

    pdf_text = extract_pdf_text(pdf_path)
    check_for_banned_jargon(pdf_text)


# ---------------------------------------------------------------------------
# Phase 5 (M4): questions, options, previews, terminal screen, and answered PDFs
# ---------------------------------------------------------------------------

from preflight.decisions import Decision
from preflight.interact.channels.terminal import format_screen
from preflight.interact.question import ReviewSession, build_question_from_decision, create_review_session
from preflight.interact.resolve import _load_raw
from preflight.plan import plan
from preflight import resolve

HAND_BUILT_FILES = {
    "numbers": ("id,val1,val2\n1,10.5,100\n2,20.5,200\n3,30.5,300\n", None),
    "zips": ("zip,state\n02138,MA\n01001,MA\n07030,NJ\n", None),
    "euro": ("item,price\ncoffee,3,50\ntea,2,75\ncake,4,20\n", None),
    "dates": ("event,date\nmeeting,01/02/2020\nlunch,03/04/2020\ncall,05/06/2020\n", None),
    "units": ("car,mileage,cost\nA,15 kmpl,$12000\nB,18 kmpl,$15500\n", None),
    "semi": ("code;label;count\n10;first;100\n20;second;200\n", None),
    "task": ("f1,f2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", "target"),
    "mixed": ("id,zip,mileage,when,score,target\n" + "\n".join(
        f"{i + 1},0{2100 + i},{10 + i} kmpl,0{(i % 9) + 1}/0{(i % 8) + 1}/2020,{i % 3},{i % 2}" for i in range(12)
    ) + "\n", "target"),
}


def question_strings(session: ReviewSession):
    """Every user-facing string of every question except the user's own data values."""
    for q in session.questions:
        yield q.text
        yield q.explanation
        yield q.evidence_summary
        yield q.recommended_default
        yield from q.options
        if q.preview:
            yield q.preview.get("note", "")
            yield from q.preview["after"].keys()


@pytest.mark.parametrize("name", sorted(HAND_BUILT_FILES))
def test_jargon_lint_question_text_options_previews_and_screen(name, tmp_path: Path):
    content, target = HAND_BUILT_FILES[name]
    src = tmp_path / "data"
    src.mkdir()
    p = src / f"{name}.csv"
    p.write_text(content, encoding="utf-8")
    plan_obj = plan(p, target=target)
    session = create_review_session(plan_obj.decisions, p, raw_df=_load_raw(plan_obj), schema=plan_obj.schema)
    for s in question_strings(session):
        check_for_banned_jargon(s)
    check_for_banned_jargon(format_screen(session))


def _reading_decision(kind, action, default, evidence):
    return Decision(
        id="", stage="reading", kind=kind, columns=[], action=action, evidence=evidence,
        confidence=0.5, default=default, needs_attention=True, plain_language="x",
    )


def test_jargon_lint_reading_questions():
    decisions = [
        _reading_decision("encoding", "use_encoding_iso-8859-1", "use_encoding_utf-8", {"encoding": "iso-8859-1", "confidence": 0.5}),
        _reading_decision("encoding", "use_encoding_utf-16", "use_encoding_utf-8", {"encoding": "utf-16", "confidence": 0.5}),
        _reading_decision("dialect", "separate_by_;", "separate_by_,", {"delimiter": ";", "confidence": 0.5}),
        _reading_decision("dialect", "separate_by_\t", "separate_by_,", {"delimiter": "\t", "confidence": 0.5}),
        _reading_decision("header", "use_header_row_2", "use_header_row_0", {"header_row_index": 2, "title_rows": [0, 1], "is_ragged": True, "confidence": 0.5}),
        _reading_decision("header", "no_header", "use_header_row_0", {"header_row_index": None, "confidence": 0.5}),
    ]
    session = create_review_session(decisions, "/x/data.csv")
    assert len(session.questions) == len(decisions)
    for s in question_strings(session):
        check_for_banned_jargon(s)
    check_for_banned_jargon(format_screen(session))


def test_jargon_lint_plan_pdf_after_answers(tmp_path: Path):
    content, target = HAND_BUILT_FILES["mixed"]
    src = tmp_path / "data"
    src.mkdir()
    p = src / "mixed.csv"
    p.write_text(content, encoding="utf-8")
    for mode, channel in (("defaults", None), ("ask", lambda qs: {q.question_id: q.options[-1] for q in qs})):
        _, pdf_path = resolve(p, mode=mode, channel=channel, out_dir=tmp_path / f"out_{mode}", target=target)
        text = extract_pdf_text(pdf_path)
        assert "Your answers" in text
        check_for_banned_jargon(text)
