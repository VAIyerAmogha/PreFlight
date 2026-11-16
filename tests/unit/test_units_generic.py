"""
Generic embedded-unit detection: a number with a short repeated letter suffix (or a
currency / percent symbol) is found without any unit name being listed in the code.
"""
import random
from pathlib import Path

import pytest

from preflight.infer.types import infer_column_type, strip_unit
from preflight.plan import apply, plan

SRC = Path(__file__).resolve().parents[2] / "src" / "preflight"

r = random.Random(11)
CASES = {
    "CC": ([f"{r.randint(800, 2500)} CC" for _ in range(120)], "int"),
    "kmpl": ([f"{r.randint(10, 28)} kmpl" for _ in range(120)], "int"),
    "kg": ([f"{r.randint(1, 95)} kg" for _ in range(120)], "int"),
    "L": ([f"{r.choice(['1.2', '3.5', '2.0', '4.4'])} L" for _ in range(120)], "float"),
}


@pytest.mark.parametrize("unit", list(CASES))
def test_unit_column_is_read_as_numbers_with_the_unit_kept_and_flagged(unit):
    values, typ = CASES[unit]
    res = infer_column_type(values, column_name="measure")
    assert res.inferred_type == typ
    assert res.decision.action == f"cast_to_{typ}"
    assert res.needs_attention  # lossy conversion: always flagged
    assert res.evidence["embedded_unit_detected"] == unit  # unit preserved in the Decision
    assert res.evidence["lossy_coercion"] is True
    assert f"embedded_unit: {unit}" in res.anomalies


@pytest.mark.parametrize("unit", ["CC", "kmpl", "kg"])
def test_no_unit_name_is_listed_in_the_source(unit):
    # 'L' is too short to search for meaningfully; the other three must not appear as
    # quoted literals anywhere in the package.
    for f in SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert f'"{unit}"' not in text and f"'{unit}'" not in text, f
        assert f"{unit}|" not in text and f"|{unit})" not in text, f


@pytest.mark.parametrize("values", [
    [r.choice(["the cat sat on the mat", "a dog ran fast", "birds fly south"]) for _ in range(80)],
    [f"A{r.randint(1, 99)}" for _ in range(80)],  # IDs with a letter prefix
    [f"{r.randint(1, 99)}" for _ in range(80)],  # plain integers
    [f"{r.uniform(0, 9):.2f}" for _ in range(80)],  # plain floats
])
def test_negative_cases_are_not_units(values):
    res = infer_column_type(values, column_name="col")
    assert res.evidence["embedded_unit_detected"] is None
    assert not any(a.startswith("embedded_unit") for a in res.anomalies)
    assert "lossy_coercion" not in res.evidence


def test_partial_unit_column_is_kept_as_text_and_flagged():
    values = [f"{r.randint(1, 95)} kg" for _ in range(90)] + ["unknown weight"] * 10
    res = infer_column_type(values, column_name="weight")
    assert res.inferred_type == "text" and res.abstained and res.needs_attention
    assert res.decision.default == "keep_as_string"


def test_strip_unit_only_strips_the_named_unit():
    assert strip_unit("1248 CC", "CC") == "1248"
    assert strip_unit(" 3.5 L ", "L") == "3.5"
    assert strip_unit("42", "kg") == "42"
    assert strip_unit("12 lb", "kg") is None
    assert strip_unit("A12", "A") is None


def test_unit_column_end_to_end_values_and_decision(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "cars.csv"
    vals = [f"{800 + 7 * i} CC" for i in range(60)]
    f.write_text("engine,target\n" + "".join(f"{v},{i % 2}\n" for i, v in enumerate(vals)), encoding="utf-8")
    p = plan(f, target="target")
    dec = next(x for x in p.decisions if x.kind == "type_inference" and x.columns == ["engine"])
    assert dec.action == "cast_to_int" and dec.needs_attention
    assert dec.evidence["embedded_unit_detected"] == "CC"
    res = apply(p, mode="defaults", out_dir=tmp_path / "out")
    assert list(res.data["engine"].astype(int)[:3]) == [800, 807, 814]
    assert not [a for a in res.anomalies if a["column"] == "engine"]


# --------------------------------------------------------------------------- phone-like codes

@pytest.mark.parametrize("make", [
    lambda i: f"+44{7000000000 + 7919 * i}",  # international, leading plus
    lambda i: f"0{100000000 + 7919 * i}",  # leading zero
    lambda i: f"{200 + i % 700}-{100 + i % 800}-{1000 + i}",  # dashes
    lambda i: f"+1-{200 + i % 700}-{100 + i % 800}-{1000 + i}",  # plus and dashes
])
def test_phone_like_values_are_never_cast_to_numbers(make):
    res = infer_column_type([make(i) for i in range(200)], column_name="contact")
    assert res.inferred_type not in ("int", "float")
    assert res.decision.action not in ("cast_to_int", "cast_to_float")


def test_plus_codes_are_flagged_as_identifiers():
    res = infer_column_type([f"+44{7000000000 + 7919 * i}" for i in range(200)], column_name="contact")
    assert res.inferred_type == "ID" and res.needs_attention
    assert "plus_sign_code" in res.anomalies and res.evidence["plus_sign_code_detected"] is True


def test_signed_measurements_are_still_numbers():
    res = infer_column_type([f"+{i % 9}.5" for i in range(60)] + ["-2.0", "+3"], column_name="delta")
    assert res.inferred_type == "float"
    assert "plus_sign_code" not in res.anomalies
