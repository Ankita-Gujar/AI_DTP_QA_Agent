"""
layout_analyzer.py
-------------------
Two responsibilities:

1. Per-document analysis: derive page margins, header/footer bands and
   text, and page-number strings for each PageObject.
2. Per-pair analysis: given a source object and a candidate target object,
   compute the layout-similarity sub-scores (movement, size delta,
   alignment, margin delta, reading-order delta) that paragraph_matcher /
   image_matcher / table_matcher combine into a final match score.

Keeping this math in one place means "what counts as MOVED" or
"what counts as MARGIN_CHANGED" is defined once and reused everywhere.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .document_object import BaseObject, DocumentObject, PageObject
from .utils import (
    QAConfig,
    bbox_height,
    bbox_width,
    euclidean_distance,
    get_logger,
    normalize_score,
    relative_position,
    safe_ratio,
)

logger = get_logger(__name__)

_PAGE_NUM_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")


def analyze_page_layout(page: PageObject, config: QAConfig) -> None:
    """Populate margins, header/footer bboxes+text, and page number in-place."""
    header_limit = page.height * config.header_zone_ratio
    footer_limit = page.height * (1 - config.footer_zone_ratio)

    header_texts, footer_texts = [], []
    content_bboxes = []

    for para in page.paragraphs:
        y0, y1 = para.bbox[1], para.bbox[3]
        if y1 <= header_limit:
            header_texts.append(para.text)
        elif y0 >= footer_limit:
            footer_texts.append(para.text)
        else:
            content_bboxes.append(para.bbox)

    for obj in list(page.images) + list(page.tables):
        y0, y1 = obj.bbox[1], obj.bbox[3]
        if header_limit < y0 < footer_limit or header_limit < y1 < footer_limit:
            content_bboxes.append(obj.bbox)

    page.header_text = " ".join(t.strip() for t in header_texts if t.strip())
    page.footer_text = " ".join(t.strip() for t in footer_texts if t.strip())

    if header_texts:
        page.header_bbox = (0, 0, page.width, header_limit)
    if footer_texts:
        page.footer_bbox = (0, footer_limit, page.width, page.height)

    match = _PAGE_NUM_RE.search(page.footer_text) or _PAGE_NUM_RE.search(page.header_text)
    page.page_number_text = match.group(1) if match else None

    if content_bboxes:
        left = min(b[0] for b in content_bboxes)
        top = min(b[1] for b in content_bboxes)
        right = max(b[2] for b in content_bboxes)
        bottom = max(b[3] for b in content_bboxes)
        page.margins = {
            "left": left,
            "top": top,
            "right": page.width - right,
            "bottom": page.height - bottom,
        }
    else:
        page.margins = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}


def analyze_document_layout(doc: DocumentObject, config: QAConfig) -> None:
    for page in doc.pages:
        analyze_page_layout(page, config)


# --------------------------------------------------------------------------
# Pairwise layout comparison
# --------------------------------------------------------------------------

def layout_similarity(
    src: BaseObject,
    src_page_size: Tuple[float, float],
    tgt: BaseObject,
    tgt_page_size: Tuple[float, float],
    config: QAConfig,
) -> float:
    """0..1 similarity based on relative on-page position, independent of page dimension changes."""
    src_rel = relative_position(src.bbox, src_page_size)
    tgt_rel = relative_position(tgt.bbox, tgt_page_size)
    src_center = ((src_rel[0] + src_rel[2]) / 2, (src_rel[1] + src_rel[3]) / 2)
    tgt_center = ((tgt_rel[0] + tgt_rel[2]) / 2, (tgt_rel[1] + tgt_rel[3]) / 2)
    dist = euclidean_distance(src_center, tgt_center)  # 0..~1.4 in relative units
    return normalize_score(dist, scale=0.25)


def size_similarity(src: BaseObject, tgt: BaseObject) -> float:
    w_ratio = safe_ratio(tgt.width, src.width)
    h_ratio = safe_ratio(tgt.height, src.height)
    w_dev = abs(1.0 - w_ratio) if w_ratio != float("inf") else 1.0
    h_dev = abs(1.0 - h_ratio) if h_ratio != float("inf") else 1.0
    return max(0.0, 1.0 - (w_dev + h_dev) / 2.0)


def reading_order_similarity(src_order: int, src_total: int, tgt_order: int, tgt_total: int) -> float:
    if src_total <= 1 or tgt_total <= 1:
        return 1.0
    src_frac = src_order / max(1, src_total - 1)
    tgt_frac = tgt_order / max(1, tgt_total - 1)
    return max(0.0, 1.0 - abs(src_frac - tgt_frac))


def movement_pt(src: BaseObject, tgt: BaseObject) -> float:
    return euclidean_distance(src.center, tgt.center)


def horizontal_movement_pt(src: BaseObject, tgt: BaseObject) -> float:
    """Horizontal-only shift between object centers, in points.

    Deliberately ignores vertical movement: when target text is longer or
    shorter than the source (normal for translation), every object below the
    change cascades down/up the page even though nothing is actually wrong.
    Horizontal drift, by contrast, is never an expected side effect of
    translation length changes -- it means something about the layout
    (indentation, column, frame) really shifted -- so it's the right signal
    for a "MOVED" flag on translated documents.
    """
    return abs(src.center[0] - tgt.center[0])


def alignment_delta(src: BaseObject, tgt: BaseObject) -> float:
    """Difference in left-edge (x0) alignment, in points -- a common DTP QA check."""
    return abs(src.bbox[0] - tgt.bbox[0])


def margin_delta(src_page: PageObject, tgt_page: PageObject) -> dict:
    return {
        side: abs(src_page.margins.get(side, 0.0) - tgt_page.margins.get(side, 0.0))
        for side in ("left", "top", "right", "bottom")
    }


def height_ratio(src: BaseObject, tgt: BaseObject) -> float:
    return safe_ratio(tgt.height, src.height)


def rotation_delta(src: BaseObject, tgt: BaseObject) -> float:
    return abs((tgt.rotation - src.rotation + 180) % 360 - 180)
