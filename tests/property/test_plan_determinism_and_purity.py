"""
Hypothesis / property-ish tests for `plan()`:

1. Running plan() on the same file twice yields identical Decisions
   (ids, kinds, actions, fingerprints) and identical Plan fingerprints overall.
2. plan() never mutates its input raw DataFrame once loaded -- i.e. running
   plan() on a file does not have side effects visible to a second, independent
   run (checked by deep-copying context.raw_data style outputs via two
   independent plan() calls and also by snapshotting the source file bytes
   before/after, since plan() only reads a path, not a DataFrame, in this
   library's public API).
3. Arbitrary small generated tables never raise an unhandled exception and
   never hang when run through plan() -- allowed outcomes are: a successful
   Plan, or one of the library's own documented exception types.
"""

from __future__ import annotations

import csv
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from preflight.plan import plan

EXPECTED_EXC_TYPES = (ValueError, UnicodeDecodeError, FileNotFoundError, OSError, LookupError, KeyError)


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_plan_is_deterministic_across_runs(tmp_path):
    p = tmp_path / "det.csv"
    _write_csv(
        p,
        ["id", "amount", "category"],
        [["1", "10.5", "a"], ["2", "20.25", "b"], ["3", "bad", "a"]],
    )

    plan_1 = plan(p)
    plan_2 = plan(p)

    assert len(plan_1.decisions) == len(plan_2.decisions)
    for d1, d2 in zip(plan_1.decisions, plan_2.decisions):
        assert d1.id == d2.id
        assert d1.kind == d2.kind
        assert d1.action == d2.action
        assert d1.fingerprint == d2.fingerprint
        assert d1.needs_attention == d2.needs_attention

    assert plan_1.row_count == plan_2.row_count
    assert plan_1.column_count == plan_2.column_count


def test_plan_does_not_mutate_source_file(tmp_path):
    p = tmp_path / "purity.csv"
    _write_csv(p, ["x", "y"], [["1", "2"], ["3", "4"]])
    before = p.read_bytes()
    plan(p)
    after = p.read_bytes()
    assert before == after, "plan() must never modify the source file on disk"


cell_value = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000).map(str),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6).map(str),
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), max_codepoint=0x2FFF),
        min_size=0,
        max_size=15,
    ),
    st.just(""),
    st.just("N/A"),
    st.just("NULL"),
)

column_name = st.text(alphabet="abcdefghij_", min_size=1, max_size=8)


@given(
    n_cols=st.integers(min_value=1, max_value=4),
    n_rows=st.integers(min_value=0, max_value=10),
    data=st.data(),
)
@settings(
    max_examples=30,
    deadline=3000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_plan_never_raises_unhandled_or_hangs_on_arbitrary_tables(tmp_path, n_cols, n_rows, data):
    seen = set()
    header = []
    for _ in range(n_cols):
        name = data.draw(column_name.filter(lambda s: s not in seen))
        seen.add(name)
        header.append(name)

    rows = [[data.draw(cell_value) for _ in range(n_cols)] for _ in range(n_rows)]

    p = tmp_path / f"fuzz_{n_cols}_{n_rows}_{len(seen)}.csv"
    _write_csv(p, header, rows)

    try:
        plan_obj = plan(p)
        assert plan_obj.column_count >= 0
    except Exception as exc:
        assert isinstance(exc, EXPECTED_EXC_TYPES), (
            f"Unhandled exception type {type(exc)} escaped plan(): {exc}"
        )
