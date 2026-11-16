"""
Encoding detection evidence: a low-confidence or ambiguous result is flagged (with the
other plausible encodings listed), never silently swapped for another encoding.
"""
from pathlib import Path

import pytest

from preflight.ingest.encoding import detect_encoding, normalize_encoding_name
from preflight.plan import apply, plan

WESTERN = "name,city,price\n" + "".join(f"Café {i},Zürich,€{i}.50 – naïve “quoted”\n" for i in range(40))
CYRILLIC = "имя,город,цена\n" + "".join(f"Иван {i},Москва,{i} рублей за штуку\n" for i in range(40))
JAPANESE = "名前,都市,価格\n" + "".join(f"山田 {i},東京都,{i}円です\n" for i in range(40))
GREEK = "όνομα,πόλη\n" + "".join(f"Γιάννης {i},Αθήνα\n" for i in range(40))


@pytest.mark.parametrize("text,real", [
    (WESTERN, "cp1252"), (CYRILLIC, "cp1251"), (JAPANESE, "cp932"), (GREEK, "cp1253"),
])
def test_wrong_decode_is_never_silent(text, real):
    raw = text.encode(real)
    res = detect_encoding(raw)
    decodes_right = raw.decode(res.encoding) == text
    # Either the detected encoding gives back the original text, or the result is
    # flagged and the right encoding is offered as an alternative.
    assert decodes_right or (res.low_confidence and real in res.alternatives), res


def test_multi_byte_text_is_detected_exactly_and_not_flagged():
    res = detect_encoding(JAPANESE.encode("cp932"))
    assert res.encoding == "cp932" and not res.low_confidence


def test_tied_single_byte_candidates_make_the_result_low_confidence():
    # cp1252 Western text ties with other Western code pages that decode it differently
    res = detect_encoding(WESTERN.encode("cp1252"))
    assert res.low_confidence
    assert "cp1252" in (res.encoding,) + res.alternatives


def test_low_confidence_result_keeps_the_detected_encoding():
    # Bytes that only decode as some legacy code page: the old code swapped this for
    # ISO-8859-1 on a check that can never fail. Now the detected name is kept.
    raw = b"a,b\n" + bytes([0x82, 0xA0, 0x2C, 0x31, 0x0A]) * 3
    res = detect_encoding(raw)
    assert res.low_confidence
    assert res.raw_detected is not None
    assert res.encoding == normalize_encoding_name(res.raw_detected)


def test_plan_flags_an_ambiguous_encoding_and_defaults_still_read_the_file(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "western.csv"
    f.write_bytes(WESTERN.encode("cp1252"))
    p = plan(f)
    enc = next(x for x in p.decisions if x.kind == "encoding")
    assert enc.needs_attention
    assert enc.action == f"use_encoding_{enc.evidence['encoding']}"
    assert enc.default == enc.action  # the default is an encoding that decoded the sample
    assert "cp1252" in [enc.evidence["encoding"]] + enc.evidence["alternatives"]
    res = apply(p, mode="defaults", out_dir=tmp_path / "out")
    assert len(res.data) == 40


def test_utf8_file_is_not_flagged(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "u.csv"
    f.write_text(WESTERN, encoding="utf-8")
    enc = next(x for x in plan(f).decisions if x.kind == "encoding")
    assert enc.evidence["encoding"] == "utf-8" and not enc.needs_attention


def test_ambiguous_encoding_question_offers_the_rival_encodings(tmp_path: Path):
    # Corpus regression (avengers.csv): the question offered only one distinct choice and
    # could not be built, because every rival encoding was labelled "standard text format".
    from preflight.interact.question import build_question_from_decision

    f = tmp_path / "western.csv"
    f.write_bytes(WESTERN.encode("cp1252"))
    enc = next(x for x in plan(f).decisions if x.kind == "encoding")
    assert enc.needs_attention and enc.evidence["alternatives"]
    q = build_question_from_decision(enc)
    assert len(q.options) >= 2 and len(set(q.options)) == len(q.options)
    assert sum("standard text format" in o for o in q.options) <= 1


@pytest.mark.parametrize("enc", ["cp932", "cp1257", "cp1250", "mac-greek", "cp1251", "big5"])
def test_regional_encodings_are_never_called_the_standard_format(enc):
    from preflight.report.templates import friendly_text_format

    name = friendly_text_format(enc)
    assert name not in ("standard text format", "Western European text format")


def test_encoding_family_names_are_distinct_and_aliases_agree():
    from preflight.report.templates import friendly_text_format

    assert friendly_text_format("latin-1") == friendly_text_format("cp1252") == "Western European text format"
    assert friendly_text_format("UTF8") == "standard text format"
    assert len({friendly_text_format(e) for e in ["cp1257", "cp1250", "cp932", "cp1253", "cp1251"]}) == 5
    assert "made-up-enc" in friendly_text_format("made-up-enc")
