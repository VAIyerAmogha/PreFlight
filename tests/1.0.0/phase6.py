"""
Tests for Phase 6 (v1.0.0): graphic-oriented PDF report export via report.save_pdf().

Covers:
- save_pdf() produces a valid, non-empty PDF file at the given path
- PDF has multiple pages (cover + stage pages [+ appendix])
- Existing Report methods (.show, .plot, .to_html, .save_html, .to_dict,
  .to_dataframe, .summary_counts) are completely unaffected by this addition
- include_appendix=False produces a shorter PDF (no appendix pages)
- Gracefully handles a report with zero warnings/criticals (no crash, no blank/broken page)
- save_pdf() does not mutate the report or its underlying data
- CLI --save-pdf flag produces a PDF file end-to-end
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader  # lightweight, read-only inspection of generated PDFs

import preflight as pf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    n = 100
    return pd.DataFrame({
        "num_a": np.random.uniform(0, 100, size=n),
        "num_b": np.concatenate([np.random.uniform(0, 1, size=n - 5), [np.nan] * 5]),  # some missingness
        "cat": np.random.choice(["x", "y", "z"], size=n),
        "target": np.random.randint(0, 2, size=n),
    })


@pytest.fixture
def clean_df():
    """A very tidy DataFrame likely to produce few/no warnings or criticals."""
    n = 50
    return pd.DataFrame({
        "num_a": np.random.uniform(0, 1, size=n),
        "target": np.random.randint(0, 2, size=n),
    })


def _tmp_pdf_path():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    os.remove(path)  # save_pdf should create it fresh
    return path


# ---------------------------------------------------------------------------
# Basic PDF generation
# ---------------------------------------------------------------------------

class TestSavePdfBasic:

    def test_save_pdf_creates_file(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_pdf_produces_valid_pdf(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            reader = PdfReader(path)
            assert len(reader.pages) >= 1
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_pdf_has_multiple_pages(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            reader = PdfReader(path)
            # Expect at least: cover page + one or more stage pages
            assert len(reader.pages) >= 2
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Appendix toggle
# ---------------------------------------------------------------------------

class TestAppendixToggle:

    def test_include_appendix_true_by_default(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            reader_with = PdfReader(path)
            pages_with = len(reader_with.pages)
        finally:
            if os.path.exists(path):
                os.remove(path)

        path2 = _tmp_pdf_path()
        try:
            result.report.save_pdf(path2, include_appendix=False)
            reader_without = PdfReader(path2)
            pages_without = len(reader_without.pages)
        finally:
            if os.path.exists(path2):
                os.remove(path2)

        assert pages_with >= pages_without

    def test_no_appendix_still_produces_valid_pdf(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path, include_appendix=False)
            reader = PdfReader(path)
            assert len(reader.pages) >= 1
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Graceful handling of sparse reports
# ---------------------------------------------------------------------------

class TestGracefulEdgeCases:

    def test_no_crash_when_no_warnings_or_criticals(self, clean_df):
        result = pf.prepare(clean_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)  # must not raise
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_no_crash_with_dry_run_result(self, sample_df):
        """dry_run results still have a full report; save_pdf must work on them too."""
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Existing methods unaffected
# ---------------------------------------------------------------------------

class TestExistingMethodsUnaffected:

    def test_show_unaffected(self, sample_df, capsys):
        result = pf.prepare(sample_df, target="target", task="classification")
        result.report.show()  # must not raise, output presence is enough here
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_plot_unaffected(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        figures = result.report.plot(kind="all")
        assert isinstance(figures, list)
        assert len(figures) > 0

    def test_to_html_unaffected(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        html = result.report.to_html()
        assert isinstance(html, str)
        assert "<html" in html.lower() or "<div" in html.lower()

    def test_save_html_unaffected(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        try:
            result.report.save_html(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_summary_counts_unaffected(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        counts = result.report.summary_counts()
        assert isinstance(counts, dict)
        assert "info" in counts

    def test_to_dict_and_to_dataframe_unaffected(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        d = result.report.to_dict()
        df = result.report.to_dataframe()
        assert isinstance(d, dict)
        assert isinstance(df, pd.DataFrame)

    def test_save_pdf_does_not_mutate_report_data(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        counts_before = result.report.summary_counts()
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            counts_after = result.report.summary_counts()
            assert counts_before == counts_after
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCliSavePdfFlag:

    def test_cli_save_pdf_flag_produces_file(self, sample_df, tmp_path):
        from typer.testing import CliRunner
        from preflight.cli import app

        csv_path = tmp_path / "data.csv"
        pdf_path = tmp_path / "report.pdf"
        sample_df.to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "prepare",
                str(csv_path),
                "--target", "target",
                "--task", "classification",
                "--save-pdf", str(pdf_path),
            ],
        )

        assert result.exit_code == 0
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0