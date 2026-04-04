"""
PRISM — Chat Agent Export Tools
Generate downloadable artefacts: cleaned CSV, charts ZIP, and PDF report.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

from langchain_core.tools import tool

EXPORT_DIR = os.getenv("EXPORT_DIR", "/tmp/prism_exports")


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


@tool
def export_clean_csv(clean_csv_path: str, session_id: str) -> dict:
    """
    Copy the cleaned CSV to the export directory and return its download path.
    Returns {"path": str, "filename": str, "type": "csv"}.
    """
    out_dir = os.path.join(EXPORT_DIR, session_id)
    _ensure_dir(out_dir)
    filename = f"prism_clean_{session_id}.csv"
    out_path = os.path.join(out_dir, filename)
    shutil.copy(clean_csv_path, out_path)
    return {"path": out_path, "filename": filename, "type": "csv"}


@tool
def export_charts_zip(chart_specs_json: str, session_id: str) -> dict:
    """
    Render all ChartSpec objects to PNG using Plotly kaleido and zip them.
    chart_specs_json: JSON array of ChartSpec dicts (each must have 'plotly_json' and 'chart_type').
    Returns {"path": str, "filename": str, "chart_count": int, "type": "zip"}.
    Requires kaleido: `uv add kaleido`
    """
    import plotly.io as pio

    specs = json.loads(chart_specs_json)
    out_dir = os.path.join(EXPORT_DIR, session_id, "charts")
    _ensure_dir(out_dir)

    png_paths: list[str] = []
    for i, spec in enumerate(specs):
        try:
            fig = pio.from_json(spec["plotly_json"])
            chart_type = spec.get("chart_type", f"chart_{i}")
            png_path = os.path.join(out_dir, f"{i:02d}_{chart_type}.png")
            fig.write_image(png_path, width=1400, height=800, scale=2)
            png_paths.append(png_path)
        except Exception as exc:
            # Don't fail the whole export for one bad chart
            print(f"[PRISM] chart {i} failed to render: {exc}")

    zip_filename = f"prism_charts_{session_id}.zip"
    zip_path = os.path.join(EXPORT_DIR, session_id, zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in png_paths:
            zf.write(p, os.path.basename(p))

    return {
        "path": zip_path,
        "filename": zip_filename,
        "chart_count": len(png_paths),
        "type": "zip",
    }


@tool
def generate_pdf_report(
    profile_json: str,
    stats_json: str,
    narrative: str,
    session_id: str,
) -> dict:
    """
    Generate a structured PDF analysis report using ReportLab.
    Sections: Dataset Overview | Data Quality | Statistical Findings | Key Insights.
    Returns {"path": str, "filename": str, "type": "pdf"}.
    Requires reportlab: `uv add reportlab`
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    out_dir = os.path.join(EXPORT_DIR, session_id)
    _ensure_dir(out_dir)
    filename = f"prism_report_{session_id}.pdf"
    pdf_path = os.path.join(out_dir, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("PRISM — Agentic Analyst Report", styles["Title"]))
    story.append(Spacer(1, 18))

    # ── Dataset Overview ─────────────────────────────────────────────────────
    profile = json.loads(profile_json)
    story.append(Paragraph("Dataset Overview", styles["Heading1"]))
    rows_val = profile.get("shape", [0, 0])[0]
    cols_val = profile.get("shape", [0, 0])[1]
    story.append(Paragraph(
        f"Rows: <b>{rows_val:,}</b> &nbsp;|&nbsp; Columns: <b>{cols_val}</b>",
        styles["Normal"],
    ))
    story.append(Spacer(1, 10))

    # Data quality summary table
    null_pcts = profile.get("null_pcts", {})
    max_null = max(null_pcts.values(), default=0)
    dupe_pct = profile.get("duplicate_pct", 0)
    quality_data = [
        ["Metric", "Value"],
        ["Max null rate", f"{max_null:.1f}%"],
        ["Duplicate rows", f"{profile.get('duplicate_count', 0)} ({dupe_pct:.1f}%)"],
        ["Numeric columns", str(len(profile.get("numeric_cols", [])))],
        ["Categorical columns", str(len(profile.get("categorical_cols", [])))],
        ["Date columns", str(len(profile.get("date_cols", [])))],
    ]
    tbl = Table(quality_data, colWidths=[8 * cm, 8 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # ── Key Insights (narrative) ─────────────────────────────────────────────
    story.append(Paragraph("Key Insights", styles["Heading1"]))
    for para in narrative.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["Normal"]))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 10))

    # ── Statistical Findings ─────────────────────────────────────────────────
    stats = json.loads(stats_json)
    if stats:
        story.append(Paragraph("Statistical Findings", styles["Heading1"]))
        for s in stats:
            sig_label = "Significant ✓" if s.get("significant") else "Not Significant ✗"
            story.append(Paragraph(
                f"<b>{s.get('test_name', '').upper()}</b> | {s.get('col_a', '')} "
                f"{'& ' + s['col_b'] if s.get('col_b') else ''} | "
                f"p={s.get('p_value', '')} | {sig_label}",
                styles["Normal"],
            ))
            if s.get("interpretation"):
                story.append(Paragraph(
                    f"&nbsp;&nbsp;→ {s['interpretation']}", styles["Normal"]
                ))
            story.append(Spacer(1, 4))

    doc.build(story)
    return {"path": pdf_path, "filename": filename, "type": "pdf"}
