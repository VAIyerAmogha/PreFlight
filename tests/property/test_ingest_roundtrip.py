"""
Hypothesis property test: raw loading preserves every cell exactly.

A generated table of printable strings (avoiding the library's own missing
sentinels, delimiters and quote characters, since a cell that happens to look
like one of those is handled by a *different* contract: missing-value or
dialect detection, not raw-preservation) is written to CSV and loaded through
`preflight.ingest.load_raw`. Every cell must come back identical to what was
written, and column names must round-trip too.
"""

from __future__ import annotations

import csv

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from preflight.ingest import load_raw

SAFE_CHARS = st.characters(
    whitelist_categories=("Ll", "Lu", "Nd"),
    min_codepoint=32,
    max_codepoint=0x2FFF,
)

cell_strategy = st.text(alphabet=SAFE_CHARS, min_size=1, max_size=20).filter(
    lambda s: s.strip() != ""
)

column_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu"), min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=8,
)


@given(
    n_cols=st.integers(min_value=1, max_value=5),
    n_rows=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
@settings(
    max_examples=40,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_raw_load_preserves_every_cell_exactly(tmp_path, n_cols, n_rows, data):
    col_names = []
    seen = set()
    for _ in range(n_cols):
        name = data.draw(column_name_strategy.filter(lambda s: s not in seen))
        seen.add(name)
        col_names.append(name)

    rows = []
    for _ in range(n_rows):
        row = [data.draw(cell_strategy) for _ in range(n_cols)]
        rows.append(row)

    p = tmp_path / f"roundtrip_{n_cols}_{n_rows}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(col_names)
        for row in rows:
            writer.writerow(row)

    load_res = load_raw(p, encoding="utf-8")
    df = load_res.data

    assert list(df.columns) == col_names
    assert len(df) == n_rows
    for r_idx, row in enumerate(rows):
        for c_idx, expected in enumerate(row):
            actual = df.iloc[r_idx, c_idx]
            assert actual == expected, (
                f"cell ({r_idx},{c_idx}) mismatch: wrote {expected!r}, loaded {actual!r}"
            )

