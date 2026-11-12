"""
Type inference on large columns: scores come from a seeded sample, the winner is then
checked against every value. Small columns must give exactly the pre-sampling output
(golden file captured from the implementation before sampling was added).
"""
import json
import random
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preflight.infer import config as infer_config
from preflight.infer.types import (
    _parse_bool,
    _parse_datetime,
    _parse_float,
    _parse_int,
    _sample_positions,
    _value_masks,
    infer_column_type,
)

GOLDEN = Path(__file__).parent / "data" / "infer_golden_pre_sampling.json"


def _golden_cases():
    # Must stay byte-for-byte identical to the generator used to write GOLDEN.
    r = random.Random(7)
    cases = {
        "ints": [str(r.randint(-500, 5000)) for _ in range(300)],
        "ints_thousands": [f"{r.randint(1000, 999999):,}" for _ in range(200)],
        "floats": [f"{r.uniform(-10, 10):.3f}" for _ in range(400)] + ["NA", "", "1e-5"],
        "dec_comma": [f"{r.randint(0, 99)},{r.randint(0, 99)}" for _ in range(150)],
        "leading_zero": [f"{r.randint(0, 99999):05d}" for _ in range(200)],
        "units_kg": [f"{r.randint(1, 90)} kg" for _ in range(120)],
        "money": [f"${r.randint(1, 900)}.{r.randint(10, 99)}" for _ in range(120)],
        "pct": [f"{r.randint(1, 99)}%" for _ in range(80)],
        "dates_iso": [f"20{r.randint(10, 24)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}" for _ in range(300)],
        "dates_ambig": [f"{r.randint(1, 12)}/{r.randint(1, 12)}/2020" for _ in range(100)],
        "dates_dmy": [f"{r.randint(13, 28)}/{r.randint(1, 12)}/2020" for _ in range(100)],
        "bool_yn": [r.choice(["yes", "no", "Yes"]) for _ in range(90)],
        "bool_01": [r.choice(["0", "1"]) for _ in range(90)],
        "cats": [r.choice(["red", "green", "blue", "cyan"]) for _ in range(500)],
        "text": [" ".join(r.choice(["alpha", "beta", "gamma", "delta", "x"]) for _ in range(9)) for _ in range(60)],
        "mixed": [r.choice(["12", "3.5", "abc", "2020-01-01", "yes", "007", "5 km"]) for _ in range(400)],
        "seq_id": [str(i) for i in range(1, 400)],
        "constant": ["x"] * 50,
        "empty": ["", "NA"] * 20,
        "sentinel": [str(r.randint(1, 80)) for _ in range(200)] + ["-999"] * 15,
        "spaces": [f"  {r.randint(1, 50)} " for _ in range(100)],
        "near_10k": [str(r.randint(0, 10**6)) for _ in range(10_000)],
    }
    names = {"seq_id": "row_id", "text": "review_text"}
    return cases, names


def _summary(res):
    return json.loads(json.dumps(
        [res.inferred_type, res.confidence, res.needs_attention, res.anomalies,
         res.failed_samples, res.decision.action, res.evidence],
        sort_keys=True, default=str,
    ))


# Phase 7b Stage 3 deliberately changed these (a column whose every value is a number with
# one unit is read as numbers, unit kept in the Decision, still flagged); everything else
# must still match the pre-sampling output exactly.
UNIT_POLICY_CHANGED = {
    "units_kg": ("int", "cast_to_int", "kg"),
    "money": ("float", "cast_to_float", "$"),
    "pct": ("int", "cast_to_int", "%"),
}


def test_small_columns_match_pre_sampling_output():
    golden = json.loads(GOLDEN.read_text())
    cases, names = _golden_cases()
    assert set(golden) == set(cases)
    for key, values in cases.items():
        assert len(values) <= infer_config.TYPE_SAMPLE_SIZE
        res = infer_column_type(values, column_name=names.get(key, key))
        if key in UNIT_POLICY_CHANGED:
            typ, action, unit = UNIT_POLICY_CHANGED[key]
            assert (res.inferred_type, res.decision.action, res.decision.default) == (typ, action, action), key
            assert res.needs_attention and res.evidence["embedded_unit_detected"] == unit
            assert golden[key][0] == "text" and golden[key][2] is True  # was: abstained and flagged
            continue
        assert _summary(res) == golden[key], key


_VALUE_CHARS = "0123456789,.-+eE $%kgmLyrs/:T "


@settings(max_examples=400, deadline=None)
@given(st.lists(st.text(alphabet=_VALUE_CHARS, min_size=1, max_size=12), min_size=1, max_size=40))
def test_vectorized_masks_agree_with_per_value_parsers(values):
    m = _value_masks(values)
    for i, v in enumerate(values):
        p_int, int_lz = _parse_int(v)
        p_flt, dec_comma, unit, flt_lz = _parse_float(v)
        assert bool(m["int"][i]) == p_int, v
        assert bool(m["int_lz"][i]) == int_lz, v
        assert bool(m["float"][i]) == p_flt, v
        assert bool(m["dec_comma"][i]) == (p_flt and dec_comma), v
        assert m["unit"][i] == unit, v
        assert bool(m["float_lz"][i]) == flt_lz, v
        assert bool(m["datetime"][i]) == _parse_datetime(v), v
        assert bool(m["bool"][i]) == _parse_bool(v), v


def _big_int_column(n=50_000, seed=3):
    r = random.Random(seed)
    return [str(r.randint(1, 10**6)) for _ in range(n)]


def test_large_column_is_scored_on_a_sample_and_is_deterministic():
    col = _big_int_column()
    a = infer_column_type(col, column_name="amount")
    b = infer_column_type(list(col), column_name="amount")
    assert a.evidence["scored_on_sample"] == infer_config.TYPE_SAMPLE_SIZE
    assert _summary(a) == _summary(b)
    assert a.decision.fingerprint is not None
    assert a.decision.fingerprint == b.decision.fingerprint
    assert a.inferred_type == "int"
    assert a.evidence["verified_failures"] == 0


def test_rare_bad_value_outside_the_sample_is_an_anomaly():
    col = _big_int_column()
    picked = set(_sample_positions(len(col)).tolist())
    outside = next(i for i in range(len(col) - 1, -1, -1) if i not in picked)
    col[outside] = "12.5"  # a float among ints: not a missing marker, not an int
    res = infer_column_type(col, column_name="amount")
    assert res.evidence["candidate_scores"]["int"] == 1.0  # the sample never saw it
    assert res.inferred_type == "int"
    assert res.evidence["verified_failures"] == 1
    assert "verification_failed: 1 values are not int" in res.anomalies
    assert res.needs_attention
    assert res.failed_samples == ["12.5"]


def test_trap_anomalies_use_the_whole_column_not_the_sample():
    # 300 leading-zero codes, all placed outside the sample: still detected
    col = _big_int_column()
    picked = set(_sample_positions(len(col)).tolist())
    outside = [i for i in range(len(col)) if i not in picked][:2000]
    for i in outside:
        col[i] = f"0{i:05d}"
    res = infer_column_type(col, column_name="amount")
    assert "leading_zero" in res.anomalies


@pytest.mark.parametrize("n", [infer_config.TYPE_SAMPLE_SIZE, infer_config.TYPE_SAMPLE_SIZE + 1])
def test_sampling_starts_just_above_the_named_size(n):
    res = infer_column_type(_big_int_column(n), column_name="amount")
    assert ("scored_on_sample" in res.evidence) == (n > infer_config.TYPE_SAMPLE_SIZE)


def test_single_leading_zero_value_is_not_a_leading_zero_anomaly():
    # Pre-sampling behaviour: the threshold (at least 2 values) counts the int parser only.
    res = infer_column_type(["0123"], column_name="fieldNameA")
    assert res.anomalies == []
    assert res.evidence["leading_zero_detected"] is False
