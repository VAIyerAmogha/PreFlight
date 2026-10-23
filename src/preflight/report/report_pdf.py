"""
Report PDF generator for PreFlight v2.0 using ReportLab.

This is the compulsory, real-run deliverable produced by `preflight.prepare()`.
There is no option to disable it: every real run ends with a Report PDF, after
the cleaned data, schema, decisions, and fitted pipeline have already been
written to disk.

Produces a plain-language document covering:
1. The plan that was applied.
2. What actually happened vs. what was planned.
3. Evidence per Decision (plain-language explanation in the body; raw evidence
   values only in the "Technical details" appendix).
4. Who decided what: automatic / confirmed / changed, with timestamps.
5. Anomaly lists, capped with a "+N more" note when truncated.
6. Leakage warnings.
7. Simple charts drawn with reportlab.graphics only (no matplotlib).
8. A "Technical details" appendix: the only section allowed to contain raw
   evidence internals and statistical terms.

Design rules:
- Plain language only outside the appendix: every paragraph is checked against
  the same banned-jargon list the Plan PDF uses (see templates.find_jargon_violations).
- Sample values shown here are restricted to two sources: examples already
  surfaced through `render_decision_text` (the same text the Plan PDF showed),
  and capped anomaly samples recorded during apply(). No other raw data values
  are pulled into this report.
- The footer discloses that the report contains sample values from the user's
  own data, not synthetic examples.

Failure handling: `write_report_artifacts()` writes the cleaned data,
schema.json, decisions.json, and the fitted pipeline (joblib) to out_dir
BEFORE attempting to render the PDF. If rendering fails for any reason, the
already-saved files are not lost: a `ReportGenerationError` is raised naming
every one of their paths.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from xml.sax.saxutils import escape as xml_escape
import json

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from preflight.decisions import Decision
from preflight.report.templates import (
    assert_plain_language,
    find_jargon_violations,
    friendly_type_name,
    render_decision_text,
    section_for_decision,
)

# ---------------------------------------------------------------------------
# Caps (anomaly lists and leakage lists are always truncated, never silently)
# ---------------------------------------------------------------------------

ANOMALY_DISPLAY_CAP = 25
LEAKAGE_DISPLAY_CAP = 15
DECISION_LOG_DISPLAY_CAP = 60

_STATUS_LABELS = {
    "automatic": "Decided automatically",
    "confirmed": "You confirmed this",
    "changed": "You changed this",
}


class ReportGenerationError(Exception):
    """
    Raised when PDF rendering fails AFTER the run's real outputs (cleaned
    data, schema, decisions, and fitted pipeline) were already saved.

    The message names every one of those saved file paths, so a failure in
    the PDF step never costs the caller their results.
    """


# ---------------------------------------------------------------------------
# Styles (reuses the Plan PDF's typography system)
# ---------------------------------------------------------------------------

def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "DocTitle": ParagraphStyle(
            "RDocTitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, textColor=colors.HexColor("#0F172A"), spaceAfter=4,
        ),
        "DocSubtitle": ParagraphStyle(
            "RDocSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12,
        ),
        "SectionHeading": ParagraphStyle(
            "RSectionHeading", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=colors.HexColor("#1E293B"),
            spaceBefore=14, spaceAfter=6, keepWithNext=True,
        ),
        "SubSectionHeading": ParagraphStyle(
            "RSubSectionHeading", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=11, leading=15, textColor=colors.HexColor("#334155"),
            spaceBefore=8, spaceAfter=4, keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "RBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=13, textColor=colors.HexColor("#334155"), spaceAfter=4,
        ),
        "BulletItem": ParagraphStyle(
            "RBulletItem", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=12, textColor=colors.HexColor("#1E293B"),
            leftIndent=10, spaceAfter=2,
        ),
        "EmptyNotice": ParagraphStyle(
            "REmptyNotice", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=13, textColor=colors.HexColor("#64748B"), spaceAfter=6,
        ),
        "WarningText": ParagraphStyle(
            "RWarningText", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9.5, leading=13, textColor=colors.HexColor("#9A3412"), spaceAfter=3,
        ),
        "SummaryLabel": ParagraphStyle(
            "RSummaryLabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"),
        ),
        "SummaryValue": ParagraphStyle(
            "RSummaryValue", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=colors.HexColor("#0F172A"),
        ),
        "AppendixHeading": ParagraphStyle(
            "RAppendixHeading", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, textColor=colors.HexColor("#1E293B"),
            spaceBefore=18, spaceAfter=8,
        ),
        "AppendixBody": ParagraphStyle(
            "RAppendixBody", parent=base["Normal"], fontName="Courier",
            fontSize=7.5, leading=10.5, textColor=colors.HexColor("#334155"), spaceAfter=3,
        ),
    }
    return styles


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas; the footer carries the data-sample disclosure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Any] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))

        if self._pageNumber > 1:
            self.drawString(36, 760, "PreFlight Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 25, page_str)
        self.drawString(
            36, 25,
            "This report contains sample values taken from your own data, not synthetic examples.",
        )
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 35, 576, 35)
        self.restoreState()


# ---------------------------------------------------------------------------
# Plain-language safety helper
# ---------------------------------------------------------------------------

def _safe_para(text: str, style: ParagraphStyle, context: str = "") -> Paragraph:
    """Paragraph for the main body: raises if `text` contains banned jargon."""
    assert_plain_language(text, context=context)
    return Paragraph(text, style)


# ---------------------------------------------------------------------------
# Charts (reportlab.graphics only)
# ---------------------------------------------------------------------------

def _status_pie_chart(decisions: Sequence[Decision]) -> Optional[Drawing]:
    counts = {"automatic": 0, "confirmed": 0, "changed": 0}
    for d in decisions:
        counts[d.status] = counts.get(d.status, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None

    labels = [k for k in ("automatic", "confirmed", "changed") if counts[k] > 0]
    data = [counts[k] for k in labels]

    drawing = Drawing(360, 170)
    pie = Pie()
    pie.x, pie.y = 60, 10
    pie.width, pie.height = 140, 140
    pie.data = data
    pie.labels = [f"{_STATUS_LABELS[k]} ({counts[k]})" for k in labels]
    pie.slices.strokeWidth = 0.5
    palette = [colors.HexColor("#2563EB"), colors.HexColor("#16A34A"), colors.HexColor("#D97706")]
    for i in range(len(data)):
        pie.slices[i].fillColor = palette[i % len(palette)]
    drawing.add(pie)
    return drawing


def _anomaly_bar_chart(anomaly_counts_by_column: Dict[str, int]) -> Optional[Drawing]:
    if not anomaly_counts_by_column:
        return None
    top = sorted(anomaly_counts_by_column.items(), key=lambda kv: kv[1], reverse=True)[:10]
    cats = [c for c, _ in top]
    vals = [n for _, n in top]

    drawing = Drawing(480, 200)
    chart = VerticalBarChart()
    chart.x, chart.y = 50, 40
    chart.width, chart.height = 400, 130
    chart.data = [vals]
    chart.categoryAxis.categoryNames = cats
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.dx = -8
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#DC2626")
    drawing.add(chart)
    drawing.add(String(50, 185, "Unusual entries found per column", fontSize=9, fillColor=colors.HexColor("#1E293B")))
    return drawing


# ---------------------------------------------------------------------------
# Report PDF rendering
# ---------------------------------------------------------------------------

def render_report_pdf(
    plan: Any,
    apply_result: Any,
    output_path: Union[str, Path],
    anomaly_cap: int = ANOMALY_DISPLAY_CAP,
    leakage_cap: int = LEAKAGE_DISPLAY_CAP,
) -> Path:
    """
    Render the Report PDF describing a completed real run.

    Args:
        plan: the Plan that was applied (preflight.plan.Plan).
        apply_result: the ApplyResult returned by apply() for this run.
        output_path: destination path for the PDF.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story: List[Any] = []

    decisions: List[Decision] = list(getattr(apply_result, "decisions", None) or getattr(plan, "decisions", []))
    anomalies: List[Dict[str, Any]] = list(getattr(apply_result, "anomalies", []) or [])

    filename = Path(getattr(plan, "source_path", "data.csv")).name
    row_count = getattr(plan, "row_count", None)
    rows_text = f"{row_count:,}" if row_count is not None else "Unknown"
    target_text = getattr(plan, "target_column", None) or "None specified"
    task_text = getattr(plan, "task_type", None) or "Not specified"

    # ---- 1. Title ----
    story.append(Paragraph("PreFlight Report", styles["DocTitle"]))
    story.append(
        Paragraph(
            "What was planned, what actually happened, and who decided what",
            styles["DocSubtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#16A34A"), spaceAfter=10))

    # ---- 2. Summary: the plan that was applied ----
    story.append(Paragraph("The plan that was applied", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    summary_data = [
        [
            Paragraph("Data File", styles["SummaryLabel"]),
            Paragraph(xml_escape(filename), styles["SummaryValue"]),
            Paragraph("Rows", styles["SummaryLabel"]),
            Paragraph(rows_text, styles["SummaryValue"]),
        ],
        [
            Paragraph("Target Column", styles["SummaryLabel"]),
            Paragraph(xml_escape(str(target_text)), styles["SummaryValue"]),
            Paragraph("Objective", styles["SummaryLabel"]),
            Paragraph(xml_escape(str(task_text)), styles["SummaryValue"]),
        ],
        [
            Paragraph("Total Decisions", styles["SummaryLabel"]),
            Paragraph(str(len(decisions)), styles["SummaryValue"]),
            Paragraph("Unusual Entries Found", styles["SummaryLabel"]),
            Paragraph(str(len(anomalies)), styles["SummaryValue"]),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[100, 170, 100, 170])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BBF7D0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCFCE7")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    pie = _status_pie_chart(decisions)
    if pie is not None:
        story.append(Paragraph("Who decided what, at a glance", styles["SubSectionHeading"]))
        story.append(pie)
        story.append(Spacer(1, 8))

    # ---- 3. What was planned, section by section ----
    story.append(Paragraph("What was planned, by stage", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    sections = ["reading", "fixing", "filling gaps", "removing", "creating"]
    section_titles = {
        "reading": "Reading", "fixing": "Fixing", "filling gaps": "Filling gaps",
        "removing": "Removing", "creating": "Creating",
    }
    for sec in sections:
        story.append(Paragraph(section_titles[sec], styles["SubSectionHeading"]))
        sec_decisions = [
            d for d in decisions
            if (section_for_decision(d) if d.stage != "reading" else "reading") == sec
            and (sec != "reading" or d.stage in ("reading", "ingest", "infer"))
        ]
        if not sec_decisions:
            story.append(Paragraph("Nothing happened at this stage.", styles["EmptyNotice"]))
            continue
        for dec in sec_decisions:
            bundle = render_decision_text(dec)
            story.append(_safe_para(f"• {bundle.summary}", styles["BulletItem"], context=f"decision {dec.id}"))

    story.append(Spacer(1, 8))

    # ---- 4. What actually happened vs. what was planned ----
    story.append(Paragraph("What actually happened vs. what was planned", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    changed = [d for d in decisions if d.status == "changed"]
    confirmed = [d for d in decisions if d.status == "confirmed"]
    automatic = [d for d in decisions if d.status == "automatic"]
    story.append(
        _safe_para(
            f"{len(automatic)} step(s) ran automatically on safe defaults, "
            f"{len(confirmed)} step(s) were confirmed as proposed, and "
            f"{len(changed)} step(s) were changed from the proposed action. "
            f"The data that came out reflects the actions actually in force below, "
            f"not only the original proposal.",
            styles["Body"],
        )
    )
    if changed:
        story.append(Paragraph("Steps you changed from the original proposal:", styles["SubSectionHeading"]))
        for dec in changed:
            where = f"Column '{dec.columns[0]}'" if dec.columns else "File"
            bundle = render_decision_text(dec)
            story.append(_safe_para(f"• {where}: now '{bundle.summary}'", styles["BulletItem"]))
    story.append(Spacer(1, 8))

    # ---- 5. Who decided what (status + timestamps) ----
    story.append(Paragraph("Who decided what", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    log_rows = [[
        Paragraph("<b>Where</b>", styles["SummaryLabel"]),
        Paragraph("<b>Decided</b>", styles["SummaryLabel"]),
        Paragraph("<b>When</b>", styles["SummaryLabel"]),
    ]]
    shown = decisions[:DECISION_LOG_DISPLAY_CAP]
    for dec in shown:
        where = ", ".join(dec.columns) if dec.columns else "File"
        log_rows.append([
            Paragraph(xml_escape(where), styles["SummaryValue"]),
            Paragraph(_STATUS_LABELS.get(dec.status, dec.status), styles["SummaryValue"]),
            Paragraph(xml_escape(str(dec.timestamp)), styles["SummaryValue"]),
        ])
    log_table = Table(log_rows, colWidths=[200, 150, 190])
    log_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(log_table)
    remaining = len(decisions) - len(shown)
    if remaining > 0:
        story.append(Paragraph(f"+ {remaining} more decision(s) not shown above.", styles["EmptyNotice"]))
    story.append(Spacer(1, 8))

    # ---- 6. Anomaly lists (capped) ----
    story.append(Paragraph("Unusual entries found while reading your data", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    anomaly_counts_by_column: Dict[str, int] = {}
    for a in anomalies:
        col = str(a.get("column", "unknown"))
        anomaly_counts_by_column[col] = anomaly_counts_by_column.get(col, 0) + 1

    if not anomalies:
        story.append(Paragraph("No unusual entries were found.", styles["EmptyNotice"]))
    else:
        bar = _anomaly_bar_chart(anomaly_counts_by_column)
        if bar is not None:
            story.append(bar)
            story.append(Spacer(1, 6))
        shown_anomalies = anomalies[:anomaly_cap]
        for a in shown_anomalies:
            col = xml_escape(str(a.get("column", "unknown")))
            val = xml_escape(str(a.get("value", "")))
            friendly_target = friendly_type_name(str(a.get("target_type", "")))
            line = f"• Column '{col}': the entry '{val}' could not be read as {friendly_target}."
            story.append(_safe_para(line, styles["BulletItem"]))
        extra = len(anomalies) - len(shown_anomalies)
        if extra > 0:
            story.append(Paragraph(f"+ {extra} more unusual entr{'y' if extra == 1 else 'ies'} not shown above.", styles["EmptyNotice"]))
    story.append(Spacer(1, 8))

    # ---- 7. Leakage warnings ----
    story.append(Paragraph("Columns that may give away the answer", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    leakage_decisions = [d for d in decisions if d.kind in ("suspected_leakage", "feature_overlap")]
    if not leakage_decisions:
        story.append(Paragraph("No columns were flagged as giving away the answer.", styles["EmptyNotice"]))
    else:
        shown_leak = leakage_decisions[:leakage_cap]
        for dec in shown_leak:
            col = ", ".join(dec.columns) if dec.columns else "a column"
            bundle = render_decision_text(dec)
            story.append(
                _safe_para(f"• Column '{col}': {bundle.summary} ({_STATUS_LABELS.get(dec.status, dec.status)})", styles["WarningText"])
            )
        extra_leak = len(leakage_decisions) - len(shown_leak)
        if extra_leak > 0:
            story.append(Paragraph(f"+ {extra_leak} more flagged column(s) not shown above.", styles["EmptyNotice"]))
    story.append(Spacer(1, 10))

    # ---- 8. Technical details appendix (jargon allowed here only) ----
    appendix_flowables: List[Any] = [
        Paragraph("Technical details", styles["AppendixHeading"]),
        Paragraph(
            "This appendix is for technical readers. It repeats the decisions above with their "
            "raw evidence values, exactly as recorded internally.",
            styles["Body"],
        ),
        Spacer(1, 6),
    ]
    for dec in decisions:
        where = ", ".join(dec.columns) if dec.columns else "table"
        header = f"[{dec.stage}/{dec.kind}] {where} -> action={dec.action!r} status={dec.status} confidence={dec.confidence:.2f}"
        appendix_flowables.append(Paragraph(xml_escape(header), styles["AppendixBody"]))
        ev_str = json.dumps(dec.evidence, default=str, sort_keys=True)
        if len(ev_str) > 600:
            ev_str = ev_str[:600] + " ...(truncated)"
        appendix_flowables.append(Paragraph(xml_escape(ev_str), styles["AppendixBody"]))
        appendix_flowables.append(Spacer(1, 3))
    story.append(KeepTogether(appendix_flowables[:4]))
    story.extend(appendix_flowables[4:])

    doc = SimpleDocTemplate(
        str(out_p), pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=45,
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    return out_p


# ---------------------------------------------------------------------------
# write_report_artifacts(): write real outputs FIRST, render PDF LAST
# ---------------------------------------------------------------------------

def write_report_artifacts(
    plan: Any,
    apply_result: Any,
    pipeline: Any,
    out_dir: Union[str, Path],
    renderer: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Write the real-run outputs (cleaned data, schema.json, decisions.json, and
    the fitted pipeline) to out_dir, THEN render the Report PDF.

    If PDF rendering raises, the exception is caught and re-raised as
    ReportGenerationError, whose message names every one of the already-saved
    file paths below, so the caller never loses results just because the PDF
    step failed.

    Returns a dict with keys: cleaned_data, schema, decisions, pipeline, report_pdf
    (all absolute Paths; report_pdf is only present if rendering succeeded, but
    the exception message on failure names the other four, which are always
    written before rendering is attempted).
    """
    out_d = Path(out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_d / "cleaned_data.csv"
    schema_path = out_d / "schema.json"
    decisions_path = out_d / "decisions.json"
    pipeline_path = out_d / "pipeline.joblib"
    report_pdf_path = out_d / "report.pdf"

    # 1. Cleaned data
    cleaned_df: Optional[pd.DataFrame] = getattr(apply_result, "prepared", None)
    if cleaned_df is None:
        cleaned_df = getattr(apply_result, "data", None)
    if cleaned_df is None:
        cleaned_df = pd.DataFrame()
    cleaned_df.to_csv(cleaned_path, index=False)

    # 2. schema.json
    schema_obj = getattr(plan, "schema", None)
    schema_json = schema_obj.to_json() if schema_obj is not None else "{}"
    schema_path.write_text(schema_json, encoding="utf-8")

    # 3. decisions.json
    decisions: List[Decision] = list(getattr(apply_result, "decisions", None) or getattr(plan, "decisions", []))
    decisions_path.write_text(
        json.dumps([d.to_dict() for d in decisions], indent=2, default=str), encoding="utf-8"
    )

    # 4. Fitted pipeline (joblib)
    import joblib

    joblib.dump(pipeline, pipeline_path)

    saved_paths = {
        "cleaned_data": cleaned_path,
        "schema": schema_path,
        "decisions": decisions_path,
        "pipeline": pipeline_path,
    }

    # Only now attempt PDF rendering.
    render = renderer if renderer is not None else render_report_pdf
    try:
        render(plan, apply_result, report_pdf_path)
    except Exception as exc:
        saved_list = "\n".join(f"  - {name}: {p}" for name, p in saved_paths.items())
        raise ReportGenerationError(
            "Failed to render the Report PDF, but your real outputs were already saved:\n"
            f"{saved_list}\nUnderlying error: {exc}"
        ) from exc

    saved_paths["report_pdf"] = report_pdf_path
    return saved_paths


__all__ = [
    "ReportGenerationError",
    "render_report_pdf",
    "write_report_artifacts",
    "ANOMALY_DISPLAY_CAP",
    "LEAKAGE_DISPLAY_CAP",
]
