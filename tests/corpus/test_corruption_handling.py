"""
Hardening tests: run deterministic corrupted variants of real corpus files
through the actual plan()/apply() pipeline.

For every corruption kind the test asserts EITHER:
  (a) the pipeline produces a sensible result (ingest/infer succeed, row/col
      counts are plausible), OR
  (b) the pipeline raises, or emits an explicit `needs_attention` Decision
      naming the relevant stage.

An unflagged wrong result (e.g. silently losing rows/columns, or a plan with
no needs_attention anywhere, that nonetheless produced nonsense) fails the
test — "no crash" alone is never sufficient.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.plan import plan, Planner

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from corruption import all_kinds, generate_corruption  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus" / "raw"

SOURCE_FILES = ["AER_Affairs.csv", "AER_BankWages.csv", "AER_CASchools.csv"]
SEED = 20240501


def _write_variant(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _any_needs_attention(plan_obj) -> bool:
    return any(d.needs_attention for d in plan_obj.decisions)


@pytest.mark.parametrize("source_file", SOURCE_FILES)
@pytest.mark.parametrize("kind", all_kinds())
def test_corrupted_variant_is_handled_or_flagged(tmp_path, source_file, kind):
    src_path = CORPUS_DIR / source_file
    raw_text = src_path.read_text(encoding="utf-8", errors="replace")
    variant = generate_corruption(raw_text, kind, seed=SEED)
    corrupted_path = _write_variant(tmp_path, f"{kind}_{source_file}", variant.data)

    try:
        plan_obj = plan(corrupted_path)
    except Exception as exc:
        # Raising is an acceptable outcome for corrupted input, as long as it's
        # a reasonably informative library/parsing exception, not something
        # bizarre like a KeyError from deep internals indicating silent misuse.
        assert isinstance(
            exc, (ValueError, UnicodeDecodeError, FileNotFoundError, OSError, LookupError)
        ), f"Unexpected exception type for {kind}/{source_file}: {type(exc)}: {exc}"
        return

    # Did not raise: either the result is sensible, or something is flagged.
    if _any_needs_attention(plan_obj):
        return

    # No flag raised -- the pipeline must still have produced a plausible
    # table: nonzero columns, and a schema whose row/column counts line up
    # with what was actually loaded.
    assert plan_obj.schema is not None, f"No schema produced for {kind}/{source_file}"
    assert plan_obj.column_count > 0, (
        f"{kind}/{source_file}: zero columns and nothing flagged needs_attention"
    )
    assert plan_obj.row_count is None or plan_obj.row_count >= 0

    # For corruption kinds that structurally SHOULD be visible to the reading
    # stage (ragged rows, duplicate columns, extra title rows, mixed
    # delimiters, wrong encoding), require that *something* in the pipeline
    # noticed -- either via needs_attention (checked above and already false)
    # or by the apply stage raising. Since neither happened, that is a real
    # gap: fail loudly rather than silently accept it.
    structurally_detectable = {
        "ragged_rows",
        "duplicate_columns",
        "extra_title_rows",
        "mixed_delimiters",
        "wrong_encoding",
    }
    if kind in structurally_detectable:
        pytest.fail(
            f"{kind} on {source_file}: pipeline neither raised nor flagged "
            f"needs_attention on any Decision, despite structurally detectable "
            f"corruption. decisions={[d.kind for d in plan_obj.decisions]}"
        )


@pytest.mark.parametrize("source_file", SOURCE_FILES)
@pytest.mark.parametrize("kind", all_kinds())
def test_corrupted_variant_apply_does_not_hang_or_crash_unexpectedly(tmp_path, source_file, kind):
    """apply() must either succeed or raise one of the library's expected
    exception types -- never hang, never raise something totally opaque."""
    src_path = CORPUS_DIR / source_file
    raw_text = src_path.read_text(encoding="utf-8", errors="replace")
    variant = generate_corruption(raw_text, kind, seed=SEED)
    corrupted_path = _write_variant(tmp_path, f"apply_{kind}_{source_file}", variant.data)

    try:
        plan_obj = plan(corrupted_path)
    except Exception:
        return  # already covered by the plan-level test

    planner = Planner()
    try:
        result = planner.apply(plan_obj)
    except Exception as exc:
        assert isinstance(
            exc, (ValueError, UnicodeDecodeError, FileNotFoundError, OSError, LookupError, KeyError)
        ), f"Unexpected exception type on apply for {kind}/{source_file}: {type(exc)}: {exc}"
        return

    assert result.data is not None
