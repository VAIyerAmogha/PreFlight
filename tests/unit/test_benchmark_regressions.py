"""
Regressions from the 10-dataset public benchmark (playground/benchmark/REPORT.md).

F1: whitespace-delimited files (UCI horse-colic, auto-mpg) were read as one text
    column with the first data row as the header, at reported confidence 1.0.
F2: a target whose class label is also a missing sentinel (credit-approval's "+"/"-")
    had a whole class blanked, leaving a constant target and a "perfect" CV score.
"""

from pathlib import Path

import pytest

import preflight
from preflight.infer.missing import detect_missing_markers
from preflight.ingest.dialect import WHITESPACE_DELIMITER, detect_dialect
from preflight.ingest.loader import load_raw

# ---------------------------------------------------------------- F1: whitespace

HORSE_LIKE = "".join(
    f"2 1 5301{i:02d} 38.50 66 28 3 3 ? 2 5 4 4 ? ? ? 3 5 45.00 8.40 ? ? 2 2 11300 00000 00000 {1 + i % 2} \n"
    for i in range(30)
)

AUTO_MPG_LIKE = "".join(
    f'{18 + i % 7}.0   8   307.0      {130 + i}.0      3504.      12.0   70  1\t"chevrolet chevelle {i}"\n'
    for i in range(30)
)


def test_space_delimited_file_is_detected():
    res = detect_dialect(HORSE_LIKE.encode())
    assert res.delimiter == WHITESPACE_DELIMITER
    assert res.field_count_mode == 28
    assert res.flagged is False


def test_space_runs_plus_tab_with_quoted_names(tmp_path: Path):
    res = detect_dialect(AUTO_MPG_LIKE.encode())
    assert res.delimiter == WHITESPACE_DELIMITER
    assert res.field_count_mode == 9

    f = tmp_path / "auto.data"
    f.write_text(AUTO_MPG_LIKE)
    loaded = load_raw(f)
    df = loaded.data
    assert df.shape == (30, 9)  # no data row was promoted to a header
    assert df.iloc[0, 8] == "chevrolet chevelle 0"
    assert df.iloc[0, 0] == "18.0"


def test_whitespace_split_of_unquoted_words_is_flagged():
    # bird_call.csv: values contain spaces, so a whitespace split may misalign columns
    content = "Species Family Count\n" + "".join(
        f"Zonotrichia albicollis{i} Passerellidae {i}\n" for i in range(20)
    )
    res = detect_dialect(content.encode())
    assert res.delimiter == WHITESPACE_DELIMITER
    assert res.flagged is True
    assert res.confidence < 0.8  # the reading stage asks instead of deciding


GERMAN_LIKE = "".join(
    f"A1{i % 4 + 1} {6 + i} A3{i % 5} A4{i % 3} {1000 + 37 * i} A6{i % 5 + 1} {i % 4 + 1} {1 + i % 2}\n"
    for i in range(40)
)


def test_flagged_whitespace_keeps_whitespace_as_safe_default(tmp_path: Path):
    # german-credit: letter codes make the split "ambiguous", but comma cannot split
    # the rows at all, so defaults mode must not fall back to a one-column table.
    f = tmp_path / "german.data"
    f.write_text(GERMAN_LIKE)
    p = preflight.plan(f, target="col_7", task="classification")
    dial = next(d for d in p.decisions if d.kind == "dialect")
    assert dial.needs_attention is True  # still asked about in ask mode
    assert dial.default == f"separate_by_{WHITESPACE_DELIMITER}"
    res = preflight.apply(p, mode="defaults", out_dir=tmp_path / "out")
    assert res.prepared.shape[1] == 8


def test_target_missing_after_reading_stops_the_run(tmp_path: Path):
    f = tmp_path / "t.csv"
    f.write_text("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(10)))
    with pytest.raises(ValueError, match=r"Target column 'y' was not found in t\.csv.*2 column"):
        preflight.plan(f, target="y", task="classification")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # ", " separated (UCI adult): whitespace gives the same field count, comma wins
        ("".join(f"{30 + i}, State-gov, {7000 + i}, Bachelors, <=50K\n" for i in range(20)), ","),
        # free text with spaces inside comma fields (titanic)
        ("id,name,age\n" + "".join(f'{i},"Braund, Mr. Owen {i}",22\n' for i in range(20)), ","),
        # tab file whose cells contain single spaces: whitespace is not 2x the fields
        ("city\tpop\tstate\n" + "".join(f"New York {i}\t{i}\tNY\n" for i in range(20)), "\t"),
        # comma file of identical multi-word sentences: whitespace looks more regular,
        # but prose has no column padding, so comma stands
        ("review,rating\n" + "".join(
            f"This product exceeded my expectations in every way,{i % 5}\n" for i in range(20)
        ), ","),
    ],
)
def test_explicit_delimiters_still_win(content: str, expected: str):
    assert detect_dialect(content.encode()).delimiter == expected


def test_schema_records_detected_reading_results(tmp_path: Path):
    f = tmp_path / "horse.data"
    f.write_text(HORSE_LIKE)
    p = preflight.plan(f, target="col_27", task="classification")
    assert p.schema.dialect.delimiter == WHITESPACE_DELIMITER
    assert p.schema.header.header_row_index is None  # headerless, not the default row 0
    assert p.column_count == 28


# ---------------------------------------------------------------- F2: label vs sentinel


def test_binary_dash_label_is_not_a_missing_marker():
    col = ["+"] * 307 + ["-"] * 383
    assert detect_missing_markers(col, "class") == []


def test_dash_is_still_missing_among_many_values():
    col = ["10", "20", "-", "30", "40", "-"]
    assert "-" in {m.marker for m in detect_missing_markers(col)}


def test_unambiguous_sentinel_stays_missing_in_two_value_column():
    col = ["yes"] * 10 + ["?"] * 5
    assert "?" in {m.marker for m in detect_missing_markers(col)}


def test_plus_minus_target_keeps_both_classes(tmp_path: Path):
    f = tmp_path / "crx.data"
    f.write_text("".join(f"b,{20 + i}.5,u,{'+' if i % 2 else '-'}\n" for i in range(40)))
    p = preflight.plan(f, target="col_3", task="classification")
    col = next(c for c in p.schema.columns if c.name == "col_3")
    assert col.missing_markers == []
    assert col.inferred_type != "constant"


def _write_table(tmp_path: Path, n: int = 60) -> Path:
    """IDs, codes, groups, an uneven target with a few blanks."""
    rows = ["customerID,code,patient_id,x,y"]
    for i in range(n):
        y = "" if i % 10 == 9 else ("b" if i % 10 == 0 else "a")  # 6 blank, 6 'b', 48 'a'
        rows.append(f"{1000 + i}-VHVEG{i},0{i % 5 + 1},P{i // 2:03d},{i * 1.5:.1f},{y}")
    f = tmp_path / "table.csv"
    f.write_text("\n".join(rows) + "\n")
    return f


def _by_kind(p, kind: str) -> dict:
    return {tuple(d.columns): d for d in p.decisions if d.kind == kind}


def test_text_id_column_is_dropped_but_short_codes_are_kept(tmp_path: Path):
    p = preflight.plan(_write_table(tmp_path), target="y", task="classification")
    drops = _by_kind(p, "drop_id")
    assert ("customerID",) in drops and drops[("customerID",)].action == "drop_column"
    assert ("code",) not in drops  # leading-zero codes with 5 values are a feature
    assert ("x",) not in drops


def test_id_name_rule_does_not_match_inside_words():
    from preflight.clean.drops import looks_like_id_name

    for name in ("customerID", "PassengerId", "user_id", "ID", "Sample ID", "id"):
        assert looks_like_id_name(name), name
    for name in ("valid", "width", "Paid", "humidity", "Bid price", "idle_time"):
        assert not looks_like_id_name(name), name


def test_rows_with_blank_target_are_set_aside(tmp_path: Path):
    p = preflight.plan(_write_table(tmp_path), target="y", task="classification")
    d = _by_kind(p, "missing_target_rows")[("y",)]
    assert d.action == "drop_rows" and d.evidence["missing_count"] == 6
    res = preflight.apply(p)
    assert len(res.prepared) == 54
    assert res.prepared["y"].notna().all()


def test_missing_target_step_is_a_no_op_on_new_data():
    import pandas as pd

    from preflight.clean.rows import apply_missing_target, plan_missing_target

    train = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", None, "b", "a"]})
    decs = plan_missing_target(train, "y")
    new = pd.DataFrame({"x": [5, 6]})  # no target column at prediction time
    out, n = apply_missing_target(new, decs)
    assert n == 0 and len(out) == 2


def test_uneven_target_gets_a_note(tmp_path: Path):
    p = preflight.plan(_write_table(tmp_path), target="y", task="classification")
    d = _by_kind(p, "class_imbalance")[("y",)]
    assert d.evidence["rarest_class"] == "b"
    assert d.needs_attention is False
    assert "accuracy" in d.plain_language


def test_repeating_identifier_gets_a_grouping_note(tmp_path: Path):
    p = preflight.plan(_write_table(tmp_path), target="y", task="classification")
    notes = _by_kind(p, "grouped_rows")
    assert ("patient_id",) in notes
    assert ("customerID",) not in notes  # unique: dropped as a row label instead
    assert ("code",) not in notes  # not named like an identifier and few distinct values


# ---------------------------------------------------------------- F3: resolved wording


def _proposal(kind: str, action: str, evidence: dict, text: str):
    from preflight.decisions import Decision

    return Decision(
        id="", stage="clean", kind=kind, columns=["x"] if kind != "remove_duplicates" else [],
        action=action, evidence=evidence, confidence=0.95, default=action,
        needs_attention=False, plain_language=text, status="automatic",
    )


def test_declined_clip_is_described_as_kept():
    from preflight.report.templates import find_jargon_violations, render_decision_text

    d = _proposal(
        "clip_outliers", "keep_outliers", {"outlier_count": 20, "impact_fraction": 0.098},
        "Pull 20 extreme entries in 'x' (9.8% of entries) back to the range [1, 2]? "
        "This is a large share, so please confirm.",
    )
    text = render_decision_text(d).summary
    assert text.startswith("Leave the 20 extreme entries in 'x'")
    assert "?" not in text and "please confirm" not in text
    assert find_jargon_violations(text) == []


def test_declined_dedupe_is_described_as_kept():
    from preflight.report.templates import render_decision_text

    d = _proposal(
        "remove_duplicates", "keep_rows", {"duplicate_count": 240, "impact_fraction": 0.15},
        "Remove 240 repeated rows (15.0% of the table)? This is a large share, so please confirm.",
    )
    assert render_decision_text(d).summary.startswith("Keep all rows, including 240 repeated ones")


def test_confirmed_proposal_loses_its_question():
    from preflight.report.templates import render_decision_text

    d = _proposal(
        "clip_outliers", "clip_outliers", {"outlier_count": 20, "impact_fraction": 0.098},
        "Pull 20 extreme entries in 'x' (9.8% of entries) back to the range [1, 2]? "
        "This is a large share, so please confirm.",
    )
    text = render_decision_text(d).summary
    assert "please confirm" not in text and not text.endswith("?")


def test_constant_target_stops_the_run(tmp_path: Path):
    f = tmp_path / "const.csv"
    f.write_text("x,y\n" + "".join(f"{i},?\n" for i in range(10)) + "".join(f"{i},1\n" for i in range(10)))
    with pytest.raises(ValueError, match="Target column 'y' has 1 distinct value"):
        preflight.plan(f, target="y", task="classification")
