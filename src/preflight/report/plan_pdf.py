"""
Plan PDF generator for PreFlight v2.0 using ReportLab.

Produces a clean, human-readable specification document detailing:
1. Summary box (dataset metrics, target, review items count)
2. 'Needs your decision' (flagged decisions as questions with options and safe default)
3. 'What we will do' (grouped by reading, fixing, filling gaps, removing, creating)
4. 'What we won't touch' (guarantee that source file is never modified)
5. Sign-off checklist

Design Rules:
- Plain language only: strictly zero banned technical jargon.
- Every decision text item is generated from plain-language templates.
- Groups without active decisions (e.g. fixing, filling gaps, removing, creating)
  render 'Nothing to do at this stage' rather than being hidden.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
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
    friendly_action_name,
    friendly_separator_name,
    friendly_text_format,
    friendly_type_name,
    render_decision_text,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running header on later pages
        if self._pageNumber > 1:
            self.drawString(
                36,
                760,
                "PreFlight Plan",
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)

        # Running footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 25, page_str)
        self.drawString(
            36,
            25,
            "Confidential - PreFlight Automated Data Plan",
        )
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 35, 576, 35)
        self.restoreState()


def _build_styles() -> Dict[str, ParagraphStyle]:
    """Build typography and element styling without jargon."""
    base = getSampleStyleSheet()

    styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4,
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "SubSectionHeading": ParagraphStyle(
            "SubSectionHeading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        ),
        "BodyBold": ParagraphStyle(
            "BodyBold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=2,
        ),
        "EmptyNotice": ParagraphStyle(
            "EmptyNotice",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=6,
        ),
        "QuestionText": ParagraphStyle(
            "QuestionText",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#9A3412"),
            spaceAfter=3,
        ),
        "OptionItem": ParagraphStyle(
            "OptionItem",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1E293B"),
            leftIndent=12,
            spaceAfter=2,
        ),
        "SafeDefault": ParagraphStyle(
            "SafeDefault",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#15803D"),
            leftIndent=12,
            spaceAfter=4,
        ),
        "SummaryLabel": ParagraphStyle(
            "SummaryLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
        ),
        "SummaryValue": ParagraphStyle(
            "SummaryValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0F172A"),
        ),
        "SummaryValueHighlight": ParagraphStyle(
            "SummaryValueHighlight",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#C2410C"),
        ),
        "BulletItem": ParagraphStyle(
            "BulletItem",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1E293B"),
            leftIndent=10,
            spaceAfter=2,
        ),
    }
    return styles


def render_plan_pdf(
    plan: Any,
    output_path: Union[str, Path],
) -> Path:
    """
    Render a complete, validated Plan PDF document.

    Args:
        plan: Plan dataclass instance from preflight.plan.
        output_path: Destination file path for generated PDF.

    Returns:
        Path to generated PDF file.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story: List[Any] = []

    # 1. Document Title Banner
    story.append(Paragraph("PreFlight Plan", styles["DocTitle"]))
    story.append(
        Paragraph(
            "Specification for Data Verification and Pipeline Operations",
            styles["DocSubtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # 2. Summary Box
    decisions: List[Decision] = getattr(plan, "decisions", [])
    attention_decisions = [d for d in decisions if getattr(d, "needs_attention", False)]

    filename = Path(getattr(plan, "source_path", "data.csv")).name
    row_count = getattr(plan, "row_count", None)
    rows_text = f"{row_count:,}" if row_count is not None else "Unknown"
    cols_text = str(getattr(plan, "column_count", len(getattr(plan, "schema", None).columns if getattr(plan, "schema", None) else [])))
    target_text = getattr(plan, "target_column", None) or "None specified"
    task_text = getattr(plan, "task_type", None) or "Not specified"

    if attention_decisions:
        rev_count_style = styles["SummaryValueHighlight"]
        rev_text = f"{len(attention_decisions)} item(s) require review"
    else:
        rev_count_style = styles["SummaryValue"]
        rev_text = "0 items (all safe defaults ready)"

    summary_data = [
        [
            Paragraph("Data File", styles["SummaryLabel"]),
            Paragraph(filename, styles["SummaryValue"]),
            Paragraph("Rows / Columns", styles["SummaryLabel"]),
            Paragraph(f"{rows_text} rows / {cols_text} columns", styles["SummaryValue"]),
        ],
        [
            Paragraph("Target Column", styles["SummaryLabel"]),
            Paragraph(target_text, styles["SummaryValue"]),
            Paragraph("Objective", styles["SummaryLabel"]),
            Paragraph(task_text, styles["SummaryValue"]),
        ],
        [
            Paragraph("Review Needed", styles["SummaryLabel"]),
            Paragraph(rev_text, rev_count_style),
            Paragraph("Safety Mode", styles["SummaryLabel"]),
            Paragraph("Source File Protected", styles["SummaryValue"]),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[100, 170, 100, 170])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 3. Section: Needs your decision
    story.append(Paragraph("Needs your decision", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))

    if not attention_decisions:
        story.append(
            Paragraph(
                "All automated checks completed with high certainty. No manual decisions are required. "
                "All operations will proceed automatically using safe defaults.",
                styles["Body"],
            )
        )
        story.append(Spacer(1, 8))
    else:
        for idx, dec in enumerate(attention_decisions, start=1):
            text_bundle = render_decision_text(dec)
            q_title = text_bundle.question or f"Review column '{dec.columns[0] if dec.columns else 'data'}'"

            q_flowables: List[Any] = [
                Paragraph(f"Item {idx}: {q_title}", styles["QuestionText"]),
            ]
            if text_bundle.explanation:
                q_flowables.append(Paragraph(text_bundle.explanation, styles["Body"]))

            # Render selectable options
            if text_bundle.options:
                for opt in text_bundle.options:
                    q_flowables.append(Paragraph(f"- [ ] {opt}", styles["OptionItem"]))

            # Highlight recommended safe default
            rec_def = text_bundle.recommended_default or friendly_action_name(dec.default)
            q_flowables.append(
                Paragraph(f"Recommended safe default: {rec_def}", styles["SafeDefault"])
            )
            q_flowables.append(Spacer(1, 4))

            q_box = Table([[q_flowables]], colWidths=[540])
            q_box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FDE68A")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(q_box)
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))

    # 4. Section: What we will do
    story.append(Paragraph("What we will do", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))

    # Group 1: Reading
    story.append(Paragraph("Reading", styles["SubSectionHeading"]))
    reading_decisions = [
        d for d in decisions if getattr(d, "stage", "") in ("reading", "ingest", "infer")
    ]
    if reading_decisions:
        for dec in reading_decisions:
            text_bundle = render_decision_text(dec)
            bullet = f"• {text_bundle.summary}"
            story.append(Paragraph(bullet, styles["BulletItem"]))
    else:
        story.append(Paragraph("Nothing to do at this stage.", styles["EmptyNotice"]))

    # Group 2: Fixing
    story.append(Paragraph("Fixing", styles["SubSectionHeading"]))
    fixing_decisions = [
        d for d in decisions if getattr(d, "stage", "") == "clean" and getattr(d, "kind", "") == "fixing"
    ]
    if fixing_decisions:
        for dec in fixing_decisions:
            text_bundle = render_decision_text(dec)
            story.append(Paragraph(f"• {text_bundle.summary}", styles["BulletItem"]))
    else:
        story.append(Paragraph("Nothing to do at this stage.", styles["EmptyNotice"]))

    # Group 3: Filling gaps
    story.append(Paragraph("Filling gaps", styles["SubSectionHeading"]))
    gaps_decisions = [
        d for d in decisions if getattr(d, "stage", "") == "clean" and "gap" in getattr(d, "kind", "")
    ]
    if gaps_decisions:
        for dec in gaps_decisions:
            text_bundle = render_decision_text(dec)
            story.append(Paragraph(f"• {text_bundle.summary}", styles["BulletItem"]))
    else:
        story.append(Paragraph("Nothing to do at this stage.", styles["EmptyNotice"]))

    # Group 4: Removing
    story.append(Paragraph("Removing", styles["SubSectionHeading"]))
    removing_decisions = [
        d for d in decisions if getattr(d, "stage", "") == "clean" and "remove" in getattr(d, "kind", "")
    ]
    if removing_decisions:
        for dec in removing_decisions:
            text_bundle = render_decision_text(dec)
            story.append(Paragraph(f"• {text_bundle.summary}", styles["BulletItem"]))
    else:
        story.append(Paragraph("Nothing to do at this stage.", styles["EmptyNotice"]))

    # Group 5: Creating
    story.append(Paragraph("Creating", styles["SubSectionHeading"]))
    creating_decisions = [
        d for d in decisions if getattr(d, "stage", "") in ("engineer", "create")
    ]
    if creating_decisions:
        for dec in creating_decisions:
            text_bundle = render_decision_text(dec)
            story.append(Paragraph(f"• {text_bundle.summary}", styles["BulletItem"]))
    else:
        story.append(Paragraph("Nothing to do at this stage.", styles["EmptyNotice"]))

    story.append(Spacer(1, 10))

    # 5. Section: What we won't touch
    story.append(Paragraph("What we won't touch", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    story.append(
        Paragraph(
            "<b>Data Integrity Guarantee:</b> The original source file will never be changed, modified, or overwritten. "
            "All operations create completely new datasets and outputs, preserving your source data in its exact original condition.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Columns designated as identifiers, constant values, or preserved plain text will not have their values modified.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 10))

    # 6. Section: Sign-off checklist
    checklist_flowables: List[Any] = [
        Paragraph("Sign-off checklist", styles["SectionHeading"]),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6),
        Paragraph("- [ ] I have reviewed the file reading settings and column titles.", styles["Body"]),
        Paragraph("- [ ] I have reviewed the proposed column types and interpretations.", styles["Body"]),
        Paragraph("- [ ] I confirm the selections above and approve proceeding with execution.", styles["Body"]),
        Spacer(1, 6),
        Table(
            [
                [
                    Paragraph("<b>Signature:</b> ___________________________", styles["Body"]),
                    Paragraph("<b>Date:</b> ______________", styles["Body"]),
                ]
            ],
            colWidths=[340, 200],
        ),
    ]

    story.append(KeepTogether(checklist_flowables))

    # Build Document
    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=45,
    )
    doc.build(story, canvasmaker=NumberedCanvas)

    return out_p
