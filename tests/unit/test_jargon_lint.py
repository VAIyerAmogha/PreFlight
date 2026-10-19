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
