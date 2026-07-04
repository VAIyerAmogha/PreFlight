# tests/test_report.py (part 6 — .to_html())
import pandas as pd
import numpy as np
import os
import tempfile
import pytest
import matplotlib
matplotlib.use("Agg")
from preflight.report import Report
from preflight.types import ReportEntry, ColumnProfile, SemanticType

def make_profile(name, mi=0.3):
    return ColumnProfile(
        name=name, semantic_type=SemanticType.NUMERIC_FEATURE, missing_rate=0.0,
        outlier_rate=None, cardinality=10, rare_categories=[], vif_score=None,
        correlation_with_target=None, mutual_info_with_target=mi,
        is_leakage_suspect=False, dtype="float64",
    )

def test_to_html_returns_string():
    entries = [ReportEntry(stage="cleaner", column="a", action="x", rationale="y",
                            severity="info", before_stats={}, after_stats={})]
    r = Report(entries)
    html = r.to_html(include_plots=False)
    assert isinstance(html, str)
    assert "<html" in html.lower() or "<table" in html.lower()

def test_to_html_escapes_special_characters():
    entries = [ReportEntry(stage="cleaner", column="<script>alert(1)</script>",
                            action="x", rationale="y", severity="info",
                            before_stats={}, after_stats={})]
    r = Report(entries)
    html = r.to_html(include_plots=False)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html

def test_to_html_degrades_gracefully_without_plot_context():
    entries = [ReportEntry(stage="cleaner", column="a", action="x", rationale="y",
                            severity="info", before_stats={}, after_stats={})]
    r = Report(entries)
    html = r.to_html(include_plots=True)  # should not raise despite no df/profiles
    assert isinstance(html, str)

def test_to_html_embeds_base64_images_when_context_present():
    n = 50
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, n),
        "b": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n),
    })
    profiles = [make_profile("a"), make_profile("b")]
    r = Report([], df=df, profiles=profiles, target="target")
    html = r.to_html(include_plots=True)
    assert "data:image/png;base64," in html

def test_to_html_no_external_references():
    entries = [ReportEntry(stage="cleaner", column="a", action="x", rationale="y",
                            severity="info", before_stats={}, after_stats={})]
    r = Report(entries)
    html = r.to_html(include_plots=False)
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn." not in html.lower()

def test_to_html_severity_present_in_output():
    entries = [
        ReportEntry(stage="profiler", column="x", action="flag", rationale="leak",
                    severity="critical", before_stats={}, after_stats={}),
    ]
    r = Report(entries)
    html = r.to_html(include_plots=False)
    assert "critical" in html.lower()

def test_save_html_writes_file():
    entries = [ReportEntry(stage="cleaner", column="a", action="x", rationale="y",
                            severity="info", before_stats={}, after_stats={})]
    r = Report(entries)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        r.save_html(path, include_plots=False)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert len(content) > 0

def test_to_html_empty_report_no_crash():
    r = Report([])
    html = r.to_html(include_plots=False)
    assert isinstance(html, str)