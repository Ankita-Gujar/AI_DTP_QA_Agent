"""
paragraph_detector.py
----------------------
Turns raw PyMuPDF text blocks/lines into ParagraphObject instances.

Responsibilities:
  * Merge lines that clearly belong to the same paragraph (small vertical
    gap, consistent font) -- PyMuPDF's own "blocks" are usually good enough,
    but we re-validate the gap so a paragraph is not split just because of
    an odd line-break in the source PDF.
  * Split a PyMuPDF block into two paragraphs if it detects a clear gap
    larger than the configured threshold (handles the reverse case).
  * Detect simple multi-column layouts (by clustering x0 of blocks) so
    reading order reflects "down column 1, then down column 2" rather than
    raw top-to-bottom which would interleave columns incorrectly.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .document_object import ParagraphObject
from .pdf_reader import RawPage, RawTextBlock, RawTextLine, span_is_bold, span_is_italic
from .utils import QAConfig, bbox_height, get_logger

logger = get_logger(__name__)

# Leading marker of a bullet/numbered list line: bullet glyphs, dash/asterisk
# bullets, or numbered/lettered/roman-numeral markers ("1.", "1)", "a.", "iv)").
# The marker must be followed by whitespace OR by nothing at all (end of the
# line) -- the latter is deliberate: a bullet/number stub with no text after
# it (e.g. a stray "3." or a lone "\u2756" left behind by a DTP import) is
# itself exactly the kind of defect this check exists to catch, so it must
# still count as a bullet line. Requiring "marker + boundary" (rather than
# marker glued directly to a following character, e.g. "-5") is what keeps
# ordinary text like a negative number or hyphenated word from matching.
_BULLET_MARKER_RE = re.compile(
    r"^\s*("
    r"[\u2022\u2023\u25E6\u25AA\u25B8\u25CF\u25CB\u2756\u2765\u2794\u2192\u2713\u2714]"  # bullet glyphs
    r"|[-*\u2013\u2014]"                                                     # dash / asterisk bullets
    r"|\(?[0-9]{1,3}[.\)]"                                                    # 1.  1)  (1)
    r"|\(?[a-zA-Z][.\)]"                                                      # a.  a)  (a)
    r"|\(?[ivxlcdmIVXLCDM]{1,6}[.\)]"                                         # i.  ii)  IV.
    r")(?:\s|$)"
)


def _block_text(block: RawTextBlock) -> str:
    parts = []
    for line in block.lines:
        line_text = "".join(span.text for span in line.spans)
        parts.append(line_text)
    return "\n".join(p for p in parts if p.strip())


def _dominant_font(block: RawTextBlock) -> tuple[str, float, int]:
    """Return the (font, size, color) of the most common span in the block."""
    counts: dict = {}
    for line in block.lines:
        for span in line.spans:
            key = (span.font, round(span.size, 1), span.color)
            counts[key] = counts.get(key, 0) + len(span.text)
    if not counts:
        return ("", 0.0, 0)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _dominant_style(block: RawTextBlock) -> Tuple[bool, bool, float, float]:
    """Return (is_bold, is_italic, bold_ratio, italic_ratio) for the block.

    is_bold/is_italic are the majority-vote (>=50% of characters) booleans,
    consistent with how _dominant_font picks the block's representative font.
    The ratios are returned too (not just the booleans) so a caller can catch
    emphasis added/removed on only *part* of a block -- e.g. one sentence
    bolded inside an otherwise-plain paragraph -- which would never flip the
    50%-majority boolean but is still a real, visible formatting change.
    """
    bold_chars = 0
    italic_chars = 0
    total_chars = 0
    for line in block.lines:
        for span in line.spans:
            n = max(1, len(span.text))
            total_chars += n
            if span_is_bold(span):
                bold_chars += n
            if span_is_italic(span):
                italic_chars += n
    if total_chars == 0:
        return False, False, 0.0, 0.0
    bold_ratio = bold_chars / total_chars
    italic_ratio = italic_chars / total_chars
    return bold_ratio >= 0.5, italic_ratio >= 0.5, bold_ratio, italic_ratio


def _line_text(line: RawTextLine) -> str:
    return "".join(span.text for span in line.spans)


def _detect_bullets(block: RawTextBlock) -> Tuple[int, List[str]]:
    """Count how many lines in this block look like bullet/numbered list
    items, and return their (marker-stripped) text -- used to catch a bullet
    silently dropped or added within an otherwise-matched text block, which
    a whole-paragraph MISSING/EXTRA check alone would not surface."""
    items: List[str] = []
    for line in block.lines:
        text = _line_text(line)
        if _BULLET_MARKER_RE.match(text):
            items.append(text.strip())
    return len(items), items


def _split_block_by_gap(block: RawTextBlock, config: QAConfig) -> List[RawTextBlock]:
    """Split a PyMuPDF block into multiple sub-blocks if a line gap is abnormally large."""
    if len(block.lines) <= 1:
        return [block]

    groups: List[List] = [[block.lines[0]]]
    for prev, cur in zip(block.lines, block.lines[1:]):
        prev_h = max(1.0, bbox_height(prev.bbox))
        gap = cur.bbox[1] - prev.bbox[3]
        if gap > prev_h * config.paragraph_line_gap_factor:
            groups.append([])
        groups[-1].append(cur)

    if len(groups) == 1:
        return [block]

    sub_blocks = []
    for group in groups:
        xs0 = [l.bbox[0] for l in group]
        ys0 = [l.bbox[1] for l in group]
        xs1 = [l.bbox[2] for l in group]
        ys1 = [l.bbox[3] for l in group]
        bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
        sub_blocks.append(RawTextBlock(lines=group, bbox=bbox))
    return sub_blocks


def _assign_columns(blocks: List[RawTextBlock], page_width: float) -> List[int]:
    """Cheap column clustering: bucket blocks by x0 into up to 3 bands."""
    if not blocks:
        return []
    band_width = page_width / 3.0 if page_width else 1.0
    return [int(min(2, b.bbox[0] // band_width)) if band_width > 0 else 0 for b in blocks]


def _bbox_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _inside_any_table(block_bbox: Tuple[float, float, float, float],
                       table_bboxes: List[Tuple[float, float, float, float]], threshold: float = 0.6) -> bool:
    """True if most of this text block's area falls inside a detected table.

    Tables and ordinary text blocks are extracted independently (PyMuPDF's
    get_text("dict") sees every line on the page, table or not), so without
    this check a table's cell text would ALSO get turned into ordinary
    paragraph blocks -- reported (and mismatched) twice: once correctly as
    a table, once spuriously as a "paragraph" with no sensible counterpart.
    """
    w, h = block_bbox[2] - block_bbox[0], block_bbox[3] - block_bbox[1]
    area = max(0.0, w) * max(0.0, h)
    if area <= 0:
        return False
    return any(_bbox_intersection_area(block_bbox, tb) / area >= threshold for tb in table_bboxes)


def detect_paragraphs(raw_page: RawPage, page_index: int, config: QAConfig) -> List[ParagraphObject]:
    """Convert a RawPage's text blocks into ordered ParagraphObject list."""
    table_bboxes = [t.bbox for t in raw_page.tables]
    expanded_blocks: List[RawTextBlock] = []
    for block in raw_page.text_blocks:
        if table_bboxes and _inside_any_table(block.bbox, table_bboxes):
            continue  # this text belongs to a detected table -- compared there, not as a paragraph
        expanded_blocks.extend(_split_block_by_gap(block, config))

    columns = _assign_columns(expanded_blocks, raw_page.width)

    # Order: column ascending, then top-to-bottom within column.
    indexed = list(zip(expanded_blocks, columns, range(len(expanded_blocks))))
    indexed.sort(key=lambda t: (t[1], t[0].bbox[1], t[0].bbox[0]))

    paragraphs: List[ParagraphObject] = []
    for order, (block, col, _orig_idx) in enumerate(indexed):
        text = _block_text(block)
        if not text.strip():
            continue
        font, size, color = _dominant_font(block)
        is_bold, is_italic, bold_ratio, italic_ratio = _dominant_style(block)
        bullet_count, bullet_items = _detect_bullets(block)
        line_count = len(block.lines)
        paragraphs.append(
            ParagraphObject(
                id=f"p{page_index}_{order}",
                page=page_index,
                bbox=block.bbox,
                text=text,
                font=font,
                font_size=size,
                color=f"#{color:06x}" if color else None,
                line_count=line_count,
                reading_order=order,
                column_index=col,
                is_bold=is_bold,
                is_italic=is_italic,
                bold_ratio=bold_ratio,
                italic_ratio=italic_ratio,
                bullet_count=bullet_count,
                bullet_items=bullet_items,
                confidence=1.0,
                metadata={"raw_line_count": len(block.lines)},
            )
        )
    return paragraphs
