"""
difference_builder.py
-----------------------
Consolidates paragraph/image/table match results plus page-level checks
(header/footer consistency, page numbering, margins, white space) into a
single flat list of DifferenceRecord objects -- the canonical structured
output of the detection pipeline.

This is the boundary between "deterministic layout analysis" and
"AI-written report": everything past this module is either serialization
(to JSON for the LLM) or formatting (to TXT/DOCX/PDF). No new issues may be
invented after this point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .document_object import BBox, DocumentObject, ObjectStatus
from .image_matcher import ImageMatch, match_images
from .paragraph_matcher import ParagraphMatch, match_paragraphs
from .table_matcher import TableMatch, match_tables
from .utils import QAConfig, get_logger

logger = get_logger(__name__)

_RECOMMENDATIONS = {
    ObjectStatus.MISSING: "Restore the missing element in the target layout; confirm it was not dropped during import.",
    ObjectStatus.EXTRA: "Verify this element is intentional -- it has no counterpart in the source document.",
    ObjectStatus.MOVED: "Reposition the element to match the source layout's relative position.",
    ObjectStatus.PAGE_MOVED: "Element now falls on a different page than the source -- check page flow/pagination.",
    ObjectStatus.RESIZED: "Resize the element back to the source's proportions.",
    ObjectStatus.OVERFLOW: "Text/content exceeds its container -- expand the frame, reduce leading, or adjust the text box to prevent clipping.",
    ObjectStatus.UNDERFLOW: "Content occupies noticeably less space than source -- check for accidental truncation or an oversized frame.",
    ObjectStatus.ALIGNMENT_CHANGED: "Re-align the element's left edge to match the source layout.",
    ObjectStatus.MARGIN_CHANGED: "Adjust page margins to match the source template.",
    ObjectStatus.CROPPED: "Image appears cropped relative to source -- confirm the full image is visible.",
    ObjectStatus.ROTATED: "Element rotation differs from source -- confirm the rotation is intentional.",
    ObjectStatus.SPLIT: "Table appears split across multiple frames -- confirm this matches intended pagination.",
    ObjectStatus.MERGED: "Multiple source tables appear merged into one -- verify table structure.",
    ObjectStatus.MATCH: "No action needed.",
    ObjectStatus.COLOR_CHANGED: "Text color no longer matches the source -- restore the original font/highlight color unless the change was intentional.",
    ObjectStatus.FORMATTING_CHANGED: "Bold/italic emphasis no longer matches the source -- restore the original character formatting unless the change was intentional.",
    ObjectStatus.BULLET_MISMATCH: "The number of bullet/numbered list items in this block no longer matches the source -- confirm no bullet point was dropped or accidentally added.",
    ObjectStatus.STRUCTURE_CHANGED: "The table's row and/or column count no longer matches the source -- confirm no row or column was dropped, merged, or added.",
}

_SEVERITY = {
    ObjectStatus.MISSING: "high",
    ObjectStatus.EXTRA: "medium",
    ObjectStatus.MOVED: "medium",
    ObjectStatus.PAGE_MOVED: "high",
    ObjectStatus.RESIZED: "medium",
    ObjectStatus.OVERFLOW: "high",
    ObjectStatus.UNDERFLOW: "medium",
    ObjectStatus.ALIGNMENT_CHANGED: "low",
    ObjectStatus.MARGIN_CHANGED: "medium",
    ObjectStatus.CROPPED: "high",
    ObjectStatus.ROTATED: "medium",
    ObjectStatus.SPLIT: "medium",
    ObjectStatus.MERGED: "medium",
    ObjectStatus.MATCH: "none",
    ObjectStatus.COLOR_CHANGED: "medium",
    ObjectStatus.FORMATTING_CHANGED: "medium",
    ObjectStatus.BULLET_MISMATCH: "high",
    ObjectStatus.STRUCTURE_CHANGED: "high",
}


@dataclass
class DifferenceRecord:
    page: int
    object_type: str          # "paragraph" | "image" | "table" | "page"
    status: str
    severity: str
    exact_position: Optional[BBox]
    movement_pt: float
    old_bbox: Optional[BBox]
    new_bbox: Optional[BBox]
    recommendation: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _para_record(m: ParagraphMatch) -> DifferenceRecord:
    page = (m.target.page if m.target else m.source.page)
    recommendation = _RECOMMENDATIONS[m.status]
    if m.status == ObjectStatus.BULLET_MISMATCH and m.source and m.target:
        src_n, tgt_n = m.source.bullet_count, m.target.bullet_count
        direction = "an extra bullet/list item was added" if tgt_n > src_n else "a bullet/list item is missing"
        recommendation = (
            f"Bullet/list item count differs from source ({src_n} in source vs {tgt_n} in target) -- "
            f"{direction}. Confirm this was not lost or introduced during import/translation."
        )
    return DifferenceRecord(
        page=page,
        object_type="paragraph",
        status=m.status.value,
        severity=_SEVERITY[m.status],
        exact_position=(m.target.bbox if m.target else m.source.bbox),
        movement_pt=round(m.movement_pt, 2),
        old_bbox=m.source.bbox if m.source else None,
        new_bbox=m.target.bbox if m.target else None,
        recommendation=recommendation,
        confidence=round(m.score, 3),
        details={
            "source_text_preview": (m.source.text[:80] if m.source else None),
            "target_text_preview": (m.target.text[:80] if m.target else None),
            "sub_scores": m.sub_scores,
            "height_ratio": round(m.height_ratio, 3),
            "alignment_delta_pt": round(m.alignment_delta_pt, 2),
            "source_bold": m.source.is_bold if m.source else None,
            "target_bold": m.target.is_bold if m.target else None,
            "source_italic": m.source.is_italic if m.source else None,
            "target_italic": m.target.is_italic if m.target else None,
            "source_bullet_count": m.source.bullet_count if m.source else None,
            "target_bullet_count": m.target.bullet_count if m.target else None,
        },
    )


def _image_record(m: ImageMatch) -> DifferenceRecord:
    page = (m.target.page if m.target else m.source.page)
    return DifferenceRecord(
        page=page,
        object_type="image",
        status=m.status.value,
        severity=_SEVERITY[m.status],
        exact_position=(m.target.bbox if m.target else m.source.bbox),
        movement_pt=round(m.movement_pt, 2),
        old_bbox=m.source.bbox if m.source else None,
        new_bbox=m.target.bbox if m.target else None,
        recommendation=_RECOMMENDATIONS[m.status],
        confidence=round(m.score, 3),
        details={"phash_distance": m.hash_distance},
    )


def _table_record(m: TableMatch) -> DifferenceRecord:
    page = (m.target.page if m.target else m.source.page)
    recommendation = _RECOMMENDATIONS[m.status]
    if m.status == ObjectStatus.STRUCTURE_CHANGED and m.source and m.target:
        parts = []
        if m.row_delta != 0:
            parts.append(f"{m.source.rows} row(s) in source vs {m.target.rows} in target")
        if m.col_delta != 0:
            parts.append(f"{m.source.cols} column(s) in source vs {m.target.cols} in target")
        recommendation = (
            "Table structure differs from source (" + "; ".join(parts) + "). "
            "Confirm no row or column was dropped, merged, or added during layout."
        )
    return DifferenceRecord(
        page=page,
        object_type="table",
        status=m.status.value,
        severity=_SEVERITY[m.status],
        exact_position=(m.target.bbox if m.target else m.source.bbox),
        movement_pt=0.0,
        old_bbox=m.source.bbox if m.source else None,
        new_bbox=m.target.bbox if m.target else None,
        recommendation=recommendation,
        confidence=round(m.score, 3),
        details={
            "row_delta": m.row_delta,
            "col_delta": m.col_delta,
            "source_rows": m.source.rows if m.source else None,
            "target_rows": m.target.rows if m.target else None,
            "source_cols": m.source.cols if m.source else None,
            "target_cols": m.target.cols if m.target else None,
        },
    )


def _page_level_records(source_doc: DocumentObject, target_doc: DocumentObject, config: QAConfig) -> List[DifferenceRecord]:
    records: List[DifferenceRecord] = []
    n = min(source_doc.page_count, target_doc.page_count)
    for i in range(n):
        sp, tp = source_doc.pages[i], target_doc.pages[i]

        if bool(sp.header_text) != bool(tp.header_text):
            records.append(DifferenceRecord(
                page=tp.page_number, object_type="page", status="MARGIN_CHANGED",
                severity="low", exact_position=tp.header_bbox, movement_pt=0.0,
                old_bbox=sp.header_bbox, new_bbox=tp.header_bbox,
                recommendation="Header presence differs between source and target -- confirm consistency.",
                confidence=0.8, details={"check": "header_consistency"},
            ))
        if bool(sp.footer_text) != bool(tp.footer_text):
            records.append(DifferenceRecord(
                page=tp.page_number, object_type="page", status="MARGIN_CHANGED",
                severity="low", exact_position=tp.footer_bbox, movement_pt=0.0,
                old_bbox=sp.footer_bbox, new_bbox=tp.footer_bbox,
                recommendation="Footer presence differs between source and target -- confirm consistency.",
                confidence=0.8, details={"check": "footer_consistency"},
            ))
        if sp.page_number_text and tp.page_number_text and sp.page_number_text != tp.page_number_text:
            records.append(DifferenceRecord(
                page=tp.page_number, object_type="page", status="MARGIN_CHANGED",
                severity="medium", exact_position=tp.footer_bbox, movement_pt=0.0,
                old_bbox=None, new_bbox=None,
                recommendation="Page number sequence does not align with source -- verify pagination.",
                confidence=0.7,
                details={"check": "page_numbering", "source": sp.page_number_text, "target": tp.page_number_text},
            ))

        # NOTE: top/bottom margins are derived from where the first/last content
        # actually sits on the page (see analyze_page_layout), not from a fixed
        # page-template value. On a translated document, the amount of text is
        # expected to differ from the source, which shifts the top/bottom content
        # edge even when nothing is actually wrong -- so only left/right (the
        # horizontal frame, which text reflow doesn't affect) are checked here.
        for side in ("left", "right"):
            delta = abs(sp.margins.get(side, 0.0) - tp.margins.get(side, 0.0))
            if delta > config.margin_tolerance_pt:
                records.append(DifferenceRecord(
                    page=tp.page_number, object_type="page", status="MARGIN_CHANGED",
                    severity="medium", exact_position=None, movement_pt=round(delta, 2),
                    old_bbox=None, new_bbox=None,
                    recommendation=f"{side.capitalize()} margin differs from source by {delta:.1f}pt -- align to template.",
                    confidence=0.75, details={"check": "margin", "side": side, "delta_pt": round(delta, 2)},
                ))

        extra_highlights = len(tp.highlight_bboxes) - len(sp.highlight_bboxes)
        if extra_highlights > 0:
            records.append(DifferenceRecord(
                page=tp.page_number, object_type="page", status="EXTRA",
                severity="medium",
                exact_position=tp.highlight_bboxes[-1] if tp.highlight_bboxes else None,
                movement_pt=0.0, old_bbox=None, new_bbox=None,
                recommendation="A highlight marking appears in the target that has no counterpart in the "
                                "source -- confirm it's intentional (not a leftover reviewer/translator mark).",
                confidence=0.7,
                details={"check": "stray_highlight", "source_count": len(sp.highlight_bboxes),
                          "target_count": len(tp.highlight_bboxes)},
            ))
    return records


def build_differences(
    source_doc: DocumentObject,
    target_doc: DocumentObject,
    config: QAConfig,
) -> List[DifferenceRecord]:
    """Run all matchers and flatten results into the canonical difference list."""
    logger.info("Matching paragraphs...")
    para_matches = match_paragraphs(source_doc, target_doc, config)
    logger.info("Matching images...")
    image_matches = match_images(source_doc, target_doc, config)
    logger.info("Matching tables...")
    table_matches = match_tables(source_doc, target_doc, config)

    records: List[DifferenceRecord] = []
    records.extend(_para_record(m) for m in para_matches if m.status != ObjectStatus.MATCH)
    records.extend(_image_record(m) for m in image_matches if m.status != ObjectStatus.MATCH)
    records.extend(_table_record(m) for m in table_matches if m.status != ObjectStatus.MATCH)
    records.extend(_page_level_records(source_doc, target_doc, config))

    records.sort(key=lambda r: (r.page, r.object_type, r.status))
    logger.info("Total differences found: %d", len(records))
    return records


def summarize(records: List[DifferenceRecord]) -> Dict[str, Any]:
    """Aggregate stats used by both the report header and the LLM prompt."""
    by_status: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    by_page: Dict[int, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_severity[r.severity] = by_severity.get(r.severity, 0) + 1
        by_page[r.page] = by_page.get(r.page, 0) + 1

    if by_severity.get("high", 0) > 0:
        overall = "FAIL"
    elif by_severity.get("medium", 0) > 0:
        overall = "REVIEW REQUIRED"
    else:
        overall = "PASS"

    return {
        "total_issues": len(records),
        "by_status": by_status,
        "by_severity": by_severity,
        "by_page": by_page,
        "overall_status": overall,
    }
