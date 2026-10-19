"""
Unit tests for PreFlight v2.0 dry_run function (Phase 4 / M3).

Validates:
1. Purity: source file hash is identical before and after dry_run.
2. No cleaned output produced: dry_run writes only plan.json and plan.pdf.
3. Failure isolation: when PDF rendering raises an error, plan.json still exists
   on disk and dry_run raises a clear error naming the saved file.
"""

import hashlib
from pathlib import Path
import pytest

from preflight.plan import dry_run


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    p = tmp_path / "messy.csv"
    p.write_text(
        "id,value,comment\n01,12.5,normal\n02,NA,missing\n03,34.0,valid\n",
        encoding="utf-8",
    )
    return p


def test_dry_run_source_file_hash_unchanged(messy_csv: Path, tmp_path: Path):
    """Verify source file hash is 100% unchanged before and after dry_run."""
    initial_hash = hashlib.sha256(messy_csv.read_bytes()).hexdigest()
    out_dir = tmp_path / "run_out"

    plan_obj, pdf_path = dry_run(messy_csv, out_dir=out_dir)

    after_hash = hashlib.sha256(messy_csv.read_bytes()).hexdigest()
    assert initial_hash == after_hash, "dry_run altered the source file!"
    assert pdf_path.exists()
    assert (out_dir / "plan.json").exists()


def test_dry_run_no_cleaned_data_written(messy_csv: Path, tmp_path: Path):
    """Verify dry_run never writes cleaned data files or modifies working directory."""
    out_dir = tmp_path / "clean_check"
    dry_run(messy_csv, out_dir=out_dir)

    # Output directory must only contain plan.json and plan.pdf
    generated_files = sorted([f.name for f in out_dir.iterdir()])
    assert generated_files == ["plan.json", "plan.pdf"], (
        f"Unexpected files produced by dry_run: {generated_files}"
    )


def test_dry_run_pdf_failure_leaves_plan_json(messy_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify failure handling: if PDF rendering fails, plan.json still exists on disk
    and dry_run raises a clear error naming the saved plan.json file.
    """
    out_dir = tmp_path / "fail_check"
    plan_json_path = out_dir / "plan.json"

    import sys
    # Monkeypatch render_plan_pdf to raise an exception simulating rendering failure
    def mock_render_fail(plan, output_path):
        raise RuntimeError("Simulated PDF engine crash")

    plan_mod = sys.modules["preflight.plan"]
    monkeypatch.setattr(plan_mod, "render_plan_pdf", mock_render_fail)

    with pytest.raises(RuntimeError) as exc_info:
        dry_run(messy_csv, out_dir=out_dir)

    # 1. Error message must name the saved plan.json file
    err_msg = str(exc_info.value)
    assert str(plan_json_path) in err_msg or "plan.json" in err_msg, (
        f"Error message does not name saved plan.json file: {err_msg}"
    )

    # 2. plan.json must still exist on disk
    assert plan_json_path.exists(), "plan.json was not saved to disk before PDF failure!"
    assert plan_json_path.stat().st_size > 0, "plan.json on disk is empty!"
