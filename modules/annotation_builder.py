"""
annotation_builder.py
----------------------
Turns the structured DifferenceRecord list into a real, professionally
annotated Target PDF -- Adobe Acrobat-style review comments (color-coded
bounding boxes + sticky notes) stamped directly onto the target file,
rather than a separate report document.

Open the output in Acrobat/Preview/any PDF viewer: every issue shows up as
a colored box on the page, with a comment bubble you can click to read
Issue Type / Expected / Found / Recommendation / Confidence -- exactly
like a human reviewer's markup pass.

This module does NOT detect anything -- it only visualizes the
DifferenceRecord objects that difference_builder.py already produced. That
boundary (detection vs. visualization) is deliberate, same as
report_builder.py.
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

from .difference_builder import DifferenceRecord
from .document_object import BBox
from .utils import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except Exception:  # pragma: no cover
    fitz = None
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed -- annotated PDF export disabled.")


def is_available() -> bool:
    return _FITZ_AVAILABLE


# --------------------------------------------------------------------------
# Fixed color legend (RGB 0..1 for PyMuPDF) -- matches the spec:
#   Red = missing/extra objects | Orange = overflow/clipping
#   Purple = layout/movement/alignment/margins | Blue = typography/color
#   Green = bullets/numbering | Yellow = tables | Cyan = images | Gray = page-level
# --------------------------------------------------------------------------

RED = (0.80, 0.15, 0.15)
ORANGE = (0.90, 0.49, 0.13)
PURPLE = (0.56, 0.27, 0.68)
BLUE = (0.16, 0.50, 0.73)
GREEN = (0.16, 0.60, 0.32)
YELLOW = (0.80, 0.62, 0.07)
CYAN = (0.10, 0.63, 0.66)
GRAY = (0.45, 0.49, 0.49)

_LAYOUT_STATUSES = {
    "MOVED", "PAGE_MOVED", "RESIZED", "ALIGNMENT_CHANGED",
    "MARGIN_CHANGED", "ROTATED", "SPLIT", "MERGED",
}
_OVERFLOW_STATUSES = {"OVERFLOW", "UNDERFLOW"}
_PRESENCE_STATUSES = {"MISSING", "EXTRA"}

_ISSUE_TYPE_LABEL = {
    "MISSING": "Missing Element",
    "EXTRA": "Extra Element",
    "MOVED": "Object Shifted",
    "PAGE_MOVED": "Moved to Different Page",
    "RESIZED": "Object Resized",
    "OVERFLOW": "Text Overflow",
    "UNDERFLOW": "Text Underflow",
    "ALIGNMENT_CHANGED": "Alignment Changed",
    "MARGIN_CHANGED": "Margin / Header-Footer Changed",
    "CROPPED": "Image Cropped",
    "ROTATED": "Rotation Changed",
    "SPLIT": "Table Split",
    "MERGED": "Table Merged",
    "COLOR_CHANGED": "Color Changed",
    "FORMATTING_CHANGED": "Bold/Italic Changed",
    "BULLET_MISMATCH": "Bullet Count Mismatch",
    "STRUCTURE_CHANGED": "Table Rows/Columns Changed",
}


def _category_color(r: DifferenceRecord) -> Tuple[Tuple[float, float, float], str]:
    """(rgb, category_label) per the fixed annotation-color legend.

    Order matters: presence (red) and overflow (orange) are checked before
    the type-based table/image colors, so e.g. a MISSING table is still red,
    not yellow -- "object presence" and "overflow/clipping" outrank the
    generic per-type color, matching the spec's own color list.
    """
    if r.status in _PRESENCE_STATUSES:
        return RED, "Missing/Extra Object"
    if r.status in _OVERFLOW_STATUSES or r.status == "CROPPED":
        return ORANGE, "Overflow / Clipping"
    if r.object_type == "table":
        return YELLOW, "Table"
    if r.object_type == "image":
        return CYAN, "Image"
    if r.object_type == "page":
        return GRAY, "Page-level"
    if r.status == "BULLET_MISMATCH":
        return GREEN, "Bullets / Numbering"
    if r.status in ("COLOR_CHANGED", "FORMATTING_CHANGED"):
        return BLUE, "Typography / Color"
    if r.status in _LAYOUT_STATUSES:
        return PURPLE, "Layout / Position"
    return GRAY, "Other"


def _fmt_bbox(bbox: Optional[BBox]) -> str:
    if not bbox:
        return "n/a"
    return f"X={bbox[0]:.1f}, Y={bbox[1]:.1f}, W={(bbox[2] - bbox[0]):.1f}, H={(bbox[3] - bbox[1]):.1f}"


def _expected_found(r: DifferenceRecord) -> Tuple[str, str]:
    """Best-effort Expected/Found pair, tailored to the issue type."""
    d = r.details or {}
    if r.status == "MISSING":
        return f"{r.object_type.capitalize()} present, as in source", "Not found in target"
    if r.status == "EXTRA":
        return "No corresponding element in source", f"Extra {r.object_type} present in target"
    if r.status in ("MOVED", "PAGE_MOVED"):
        return f"Position {_fmt_bbox(r.old_bbox)}", f"Position {_fmt_bbox(r.new_bbox)} (moved {r.movement_pt:.1f}pt)"
    if r.status == "RESIZED":
        return f"Size {_fmt_bbox(r.old_bbox)}", f"Size {_fmt_bbox(r.new_bbox)}"
    if r.status == "OVERFLOW":
        return "Content fits fully inside its frame", "Content exceeds the frame / is being clipped"
    if r.status == "UNDERFLOW":
        return "Content fills the frame as in source", "Content occupies noticeably less space than source"
    if r.status == "ALIGNMENT_CHANGED":
        delta = d.get("alignment_delta_pt")
        found = f"Left edge offset by {delta:.1f}pt" if delta is not None else "Left edge misaligned"
        return "Left edge aligned with source", found
    if r.status == "MARGIN_CHANGED":
        check = d.get("check", "margin")
        if check == "margin":
            return f"{str(d.get('side', '')).capitalize()} margin matching source", f"Off by {d.get('delta_pt', 0):.1f}pt"
        if check == "header_consistency":
            return "Header present/absent as in source", "Header presence differs from source"
        if check == "footer_consistency":
            return "Footer present/absent as in source", "Footer presence differs from source"
        if check == "page_numbering":
            return f"Page number '{d.get('source')}'", f"Page number '{d.get('target')}'"
        if check == "stray_highlight":
            return f"{d.get('source_count', 0)} highlight marking(s)", f"{d.get('target_count', 0)} highlight marking(s)"
        return "Matches source", "Differs from source"
    if r.status == "CROPPED":
        return "Full image visible, as in source", "Image appears cropped"
    if r.status == "ROTATED":
        return "No rotation (matching source)", "Rotation differs from source"
    if r.status == "SPLIT":
        return "Single table, as in source", "Table appears split across frames"
    if r.status == "MERGED":
        return "Separate tables, as in source", "Tables appear merged into one"
    if r.status == "COLOR_CHANGED":
        return "Text color matching source", "Text color differs from source"
    if r.status == "FORMATTING_CHANGED":
        src_bold, tgt_bold = d.get("source_bold"), d.get("target_bold")
        src_italic, tgt_italic = d.get("source_italic"), d.get("target_italic")

        def _style_label(bold, italic):
            if bold and italic:
                return "bold italic"
            if bold:
                return "bold"
            if italic:
                return "italic"
            return "regular"
        return f"Text styled {_style_label(src_bold, src_italic)}, as in source", \
               f"Text styled {_style_label(tgt_bold, tgt_italic)}"
    if r.status == "BULLET_MISMATCH":
        src_n, tgt_n = d.get("source_bullet_count"), d.get("target_bullet_count")
        return f"{src_n} bullet/list item(s), as in source", f"{tgt_n} bullet/list item(s) found"
    if r.status == "STRUCTURE_CHANGED":
        src_rows, tgt_rows = d.get("source_rows"), d.get("target_rows")
        src_cols, tgt_cols = d.get("source_cols"), d.get("target_cols")
        return f"{src_rows} rows x {src_cols} cols, as in source", f"{tgt_rows} rows x {tgt_cols} cols"
    return "Matches source", "Differs from source"


def _issue_bbox(r: DifferenceRecord) -> Optional[BBox]:
    return r.exact_position or r.new_bbox or r.old_bbox


def _note_text(r: DifferenceRecord, category_label: str) -> str:
    issue_type = _ISSUE_TYPE_LABEL.get(r.status, r.status.replace("_", " ").title())
    expected, found = _expected_found(r)
    lines = [
        f"Issue Type: {issue_type}",
        f"Category: {category_label}",
        f"Object: {r.object_type.capitalize()}",
        f"Page: {r.page}",
        f"Bounding Box: {_fmt_bbox(_issue_bbox(r))}",
        f"Severity: {r.severity.upper()}",
        f"Confidence: {r.confidence:.0%}",
        "",
        f"Expected: {expected}",
        f"Found: {found}",
        "",
        f"Recommendation: {r.recommendation}",
    ]
    return "\n".join(lines)


def build_annotated_pdf(
    target_path: str,
    records: List[DifferenceRecord],
    author: str = "AI DTP QA Agent",
) -> bytes:
    """Opens the target PDF and stamps every DifferenceRecord onto it as a
    color-coded bounding-box (Square annot) + sticky-note (Text annot) pair.
    These are real PDF annotation objects -- they open as native review
    comments in Acrobat, Preview, Chrome, etc., not baked-in pixels.

    Records with no usable bbox (a handful of page-level checks, e.g. a
    page-number mismatch) still get a sticky note pinned near the top of
    the page so the issue isn't lost, just without a bounding box.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is not installed. Run `pip install pymupdf`.")

    doc = fitz.open(target_path)
    try:
        stacked_notes = 0
        for r in records:
            page_index = r.page - 1
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc[page_index]
            color, category_label = _category_color(r)
            bbox = _issue_bbox(r)
            note_text = _note_text(r, category_label)
            note_title = f"{r.severity.upper()} - {_ISSUE_TYPE_LABEL.get(r.status, r.status)}"
            info = {"title": author, "subject": note_title, "content": note_text}

            note_point = None
            if bbox:
                rect = fitz.Rect(*bbox)
                rect.intersect(page.rect)
                if not rect.is_empty and not rect.is_infinite:
                    box_annot = page.add_rect_annot(rect)
                    box_annot.set_colors(stroke=color)
                    box_annot.set_border(width=1.5)
                    box_annot.set_opacity(0.9)
                    box_annot.set_info(info)
                    box_annot.update()
                    note_point = fitz.Point(rect.x0, max(rect.y0 - 14, 2))

            if note_point is None:
                # No usable geometry -- stack the sticky note near the page corner
                # instead of dropping the issue silently.
                note_point = fitz.Point(10, 10 + 16 * (stacked_notes % 30))
                stacked_notes += 1

            text_annot = page.add_text_annot(note_point, note_text, icon="Comment")
            text_annot.set_colors(stroke=color)
            text_annot.set_info(info)
            text_annot.update()

        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True)
        return buf.getvalue()
    finally:
        doc.close()
