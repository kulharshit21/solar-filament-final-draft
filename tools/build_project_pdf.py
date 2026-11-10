#!/usr/bin/env python3
"""Build the polished FINAL DRAFT project report PDF from README.md."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
OUTPUT = ROOT / "output" / "pdf" / "FINAL_DRAFT_Project_Report.pdf"

GITHUB = "https://github.com/kulharshit21/solar-filament-final-draft"
DRIVE = "https://drive.google.com/open?id=1qikTNZkhUWEC3vB0Yz2_5l5Zwr1AehFN"
KAGGLE = "https://www.kaggle.com/competitions/filament-segmentation-2026"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1976D2")
CYAN = colors.HexColor("#00A6A6")
GOLD = colors.HexColor("#F5A623")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#F0F7FF")
PALE_GREEN = colors.HexColor("#ECFDF5")
LINE = colors.HexColor("#D9E2EC")


def ascii_text(value: str) -> str:
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00d7": "x",
        "\u00b1": "+/-", "\u03b1": "alpha", "\u2265": ">=", "\u2264": "<=", "\u2026": "...",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def inline_markup(text: str) -> str:
    text = ascii_text(text)
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"@@TOKEN{len(tokens) - 1}@@"

    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: stash(f'<link href="{html.escape(m.group(2), quote=True)}" color="#1976D2"><u>{html.escape(m.group(1))}</u></link>'),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: stash(f'<font color="#1976D2"><u>{html.escape(m.group(1))}</u></font>'),
        text,
    )
    text = re.sub(r"`([^`]+)`", lambda m: stash(f'<font name="Courier" color="#0B7285">{html.escape(m.group(1))}</font>'), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: stash(f"<b>{html.escape(m.group(1))}</b>"), text)
    text = html.escape(text)
    for index, token in enumerate(tokens):
        text = text.replace(f"@@TOKEN{index}@@", token)
    return text


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 250

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setFillColor(colors.white)
        canvas.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=1)

        boxes = {
            "input": (10, 190, 78, 40, "2048 x 2048\nH-alpha image", NAVY),
            "prep": (105, 190, 80, 40, "CLAHE\npreprocessing", CYAN),
            "yolo": (210, 195, 104, 46, "YOLO11x-seg\ninstance proposals", BLUE),
            "m2f": (210, 125, 104, 46, "Mask2Former\nSwin-L topology", BLUE),
            "sam": (210, 55, 104, 46, "SAM 2.1 Hiera-L\nboundary refinement", BLUE),
            "avg": (350, 58, 126, 40, "YOLO + SAM\nprobability average", GOLD),
            "fuse": (350, 130, 126, 46, "IoU association\nEqual fusion 1:1", GOLD),
            "gate": (210, 5, 104, 40, "Locked selector\n0.45 / 0.53", CYAN),
            "out": (350, 5, 126, 40, "Area + overlap gates\nFortran RLE", NAVY),
        }

        def arrow(x1, y1, x2, y2):
            canvas.setStrokeColor(MUTED)
            canvas.setFillColor(MUTED)
            canvas.setLineWidth(1.4)
            canvas.line(x1, y1, x2, y2)
            angle = 5
            canvas.line(x2, y2, x2 - angle, y2 + 3)
            canvas.line(x2, y2, x2 - angle, y2 - 3)

        arrow(88, 210, 105, 210)
        arrow(185, 210, 210, 218)
        arrow(185, 210, 210, 148)
        arrow(185, 210, 210, 78)
        arrow(314, 218, 344, 150)
        arrow(314, 148, 350, 153)
        arrow(314, 78, 350, 78)
        arrow(413, 98, 413, 130)
        arrow(350, 145, 314, 45)
        arrow(314, 25, 350, 25)

        for _, (x, y, w, h, label, fill) in boxes.items():
            canvas.setFillColor(fill)
            canvas.setStrokeColor(fill)
            canvas.roundRect(x, y, w, h, 6, fill=1, stroke=1)
            canvas.setFillColor(colors.white)
            canvas.setFont("Helvetica-Bold", 8.2)
            lines = label.split("\n")
            start = y + h / 2 + (len(lines) - 1) * 5
            for idx, line in enumerate(lines):
                canvas.drawCentredString(x + w / 2, start - idx * 10, line)

        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=28, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=14, leading=20, textColor=MUTED, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceBefore=12, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=CYAN, spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.5, textColor=INK, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK, leftIndent=14, firstLineIndent=-7, bulletIndent=4, spaceAfter=3),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.4, leading=10, textColor=colors.HexColor("#102A43"), backColor=colors.HexColor("#F5F7FA"), borderColor=LINE, borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=8),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED),
        "link": ParagraphStyle("Link", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=BLUE, spaceAfter=4),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=NAVY, backColor=PALE_GREEN, borderColor=CYAN, borderWidth=1, borderPadding=10, spaceAfter=10),
    }


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 15 * mm, width, 15 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(18 * mm, height - 9.5 * mm, "FINAL DRAFT - Solar Filament Segmentation")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 10 * mm, "Verified public leaderboard PQ: 0.40 | Code and artifacts linked in this report")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.restoreState()


def split_table_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    for character in line:
        if character == "`":
            in_code = not in_code
            current.append(character)
        elif character == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    return cells


def table_from_lines(lines: list[str], style_map) -> Table:
    rows = []
    for line in lines:
        cells = split_table_row(line)
        rows.append(cells)
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in rows[1]):
        rows.pop(1)
    data = [[Paragraph(inline_markup(cell), style_map["small"]) for cell in row] for row in rows]
    width = 174 * mm
    count = max(len(row) for row in data)
    table = Table(data, colWidths=[width / count] * count, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def markdown_story(text: str, style_map) -> list:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 1."))
    lines = lines[start:]
    story: list = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    code_lang = ""

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(part.strip() for part in paragraph)), style_map["body"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                flush_paragraph()
                in_code = True
                code_lang = stripped[3:].strip()
                code = []
            else:
                in_code = False
                if code_lang == "mermaid":
                    story.append(Spacer(1, 5))
                    story.append(ArchitectureDiagram(174 * mm))
                    story.append(Spacer(1, 8))
                else:
                    story.append(Paragraph(html.escape(ascii_text("\n".join(code))).replace("\n", "<br/>"), style_map["code"]))
                code_lang = ""
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if not stripped or stripped == "---":
            flush_paragraph()
            if stripped == "---":
                story.append(Spacer(1, 5))
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(table_from_lines(table_lines, style_map))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), style_map["h2"]))
        elif stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), style_map["h1"]))
        elif re.match(r"^[-*] ", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), style_map["bullet"], bulletText="-"))
        elif re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            marker, content = stripped.split(". ", 1)
            story.append(Paragraph(inline_markup(content), style_map["bullet"], bulletText=f"{marker}."))
        else:
            paragraph.append(stripped)
        i += 1
    flush_paragraph()
    return story


def build():
    style_map = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=19 * mm,
        title="FINAL DRAFT - Reproducible 0.40-PQ Solar Filament Segmentation",
        author="Harshit Kulkarni",
        subject="Architecture, training, validation, integrity, and reproduction report",
    )

    story: list = [
        Spacer(1, 11 * mm),
        Paragraph("FINAL DRAFT", style_map["title"]),
        Paragraph("Reproducible 0.40-PQ Solar Filament Instance Segmentation", style_map["subtitle"]),
        Spacer(1, 5 * mm),
        Paragraph("KAGGLE SCORE RECONFIRMED: 0.40", style_map["callout"]),
        Paragraph(f'<link href="{GITHUB}" color="#1976D2"><u>Private GitHub code repository</u></link>', style_map["link"]),
        Paragraph(f'<link href="{DRIVE}" color="#1976D2"><u>Google Drive weights and exact submission</u></link>', style_map["link"]),
        Paragraph(f'<link href="{KAGGLE}" color="#1976D2"><u>Kaggle competition</u></link>', style_map["link"]),
        Spacer(1, 8 * mm),
        ArchitectureDiagram(174 * mm),
        Spacer(1, 8 * mm),
        Paragraph("Canonical submission", style_map["h2"]),
        Paragraph("1,261 instances across 180 test images; zero overlaps; exact SHA-256:", style_map["body"]),
        Paragraph('<font name="Courier" size="7.5">4ee79cb588f9415eb2f48ebb1a129382f0d7f5e6410724be8b9f6b213b18e512</font>', style_map["body"]),
        Spacer(1, 5 * mm),
        Paragraph("Prepared 10 November 2026 | Asia/Calcutta", style_map["small"]),
        PageBreak(),
        Paragraph("Executive summary", style_map["h1"]),
        Paragraph(
            "This report documents the exact code, architecture, training stages, calibration logic, integrity boundary, artifact verification, and reproduction process behind the independently reconfirmed 0.40 public leaderboard submission. The decisive improvement over nearby 0.39 outputs was the locked equal-weight fusion and post-processing policy, not a different set of model weights.",
            style_map["body"],
        ),
        Paragraph("Release links", style_map["h2"]),
        Paragraph(f'<link href="{GITHUB}" color="#1976D2"><u>{GITHUB}</u></link>', style_map["body"]),
        Paragraph(f'<link href="{DRIVE}" color="#1976D2"><u>{DRIVE}</u></link>', style_map["body"]),
        Paragraph("Evidence highlights", style_map["h2"]),
        Table([
            [Paragraph("Evidence", style_map["small"]), Paragraph("Verified result", style_map["small"])],
            [Paragraph("Kaggle status", style_map["small"]), Paragraph("COMPLETE - public PQ 0.40", style_map["small"])],
            [Paragraph("Artifact verification", style_map["small"]), Paragraph("100 passes, 600 comparisons, zero mismatches", style_map["small"])],
            [Paragraph("Drive synchronization", style_map["small"]), Paragraph("Zero checksum differences", style_map["small"])],
            [Paragraph("Publication boundary", style_map["small"]), Paragraph("Code in GitHub; weights and CSV in Drive", style_map["small"])],
        ], colWidths=[55 * mm, 115 * mm], style=[
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]),
        PageBreak(),
    ]

    story.extend(markdown_story(SOURCE.read_text(encoding="utf-8"), style_map))
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    build()

