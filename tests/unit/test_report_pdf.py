"""
Tests for the compulsory Report PDF (preflight.prepare()).

1. End-to-end: prepare() on a small real corpus file writes every real-run
   artifact (cleaned data, schema, decisions, pipeline, report.pdf) to out_dir.
2. Failure isolation: if PDF rendering fails, the cleaned data/schema/decisions/
   pipeline files must still exist on disk at the paths named in the raised
   ReportGenerationError's message.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

import preflight as pf
from preflight.report.report_pdf import ReportGenerationError

CORPUS_FILE = Path(__file__).resolve().parents[2] / "corpus" / "raw" / "AER_Affairs.csv"


def test_prepare_end_to_end_writes_all_artifacts(tmp_path):
    assert CORPUS_FILE.exists(), f"corpus fixture missing: {CORPUS_FILE}"
    out_dir = tmp_path / "out"

    result = pf.prepare(
        CORPUS_FILE,
        target="rating",
        task="regression",
        mode="defaults",
        out_dir=out_dir,
    )

    paths = result["paths"]
    for key in ("cleaned_data", "schema", "decisions", "pipeline", "report_pdf", "plan_json", "plan_pdf"):
        assert key in paths, f"missing path key: {key}"
        assert paths[key].exists(), f"missing artifact file: {paths[key]}"

    # Cleaned data is readable and non-trivial
    cleaned = pd.read_csv(paths["cleaned_data"])
    assert len(cleaned) > 0

    # The fitted pipeline round-trips through joblib and is also the object returned
    loaded_pipe = joblib.load(paths["pipeline"])
    assert loaded_pipe is not None
    assert result["pipeline"] is not None

    # Report PDF is a real, non-empty PDF
    pdf_bytes = paths["report_pdf"].read_bytes()
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 1000

    assert result["apply_result"].decisions, "decisions should not be empty"


def test_pdf_render_failure_preserves_already_saved_artifacts(tmp_path, monkeypatch):
    out_dir = tmp_path / "out_fail"

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic rendering failure")

    monkeypatch.setattr("preflight.report.report_pdf.render_report_pdf", _boom)

    with pytest.raises(ReportGenerationError) as excinfo:
        pf.prepare(
            CORPUS_FILE,
            target="rating",
            task="regression",
            mode="defaults",
            out_dir=out_dir,
        )

    message = str(excinfo.value)
    for fname in ("cleaned_data.csv", "schema.json", "decisions.json", "pipeline.joblib"):
        full_path = out_dir / fname
        assert str(full_path) in message, f"{full_path} not named in error message"
        assert full_path.exists(), f"{full_path} should still exist on disk after PDF failure"

    # And the report.pdf itself must NOT have been produced
    assert not (out_dir / "report.pdf").exists()
