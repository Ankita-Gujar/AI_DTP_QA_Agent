"""
document_object.py
-------------------
The Document Object Model (DOM) for the DTP QA Agent.

Every extracted PDF is normalized into these dataclasses. Nothing downstream
(matchers, difference builder, report) ever touches PyMuPDF objects directly --
they only ever see ParagraphObject / ImageObject / TableObject / PageObject /
DocumentObject. This keeps the extraction layer swappable (e.g. a future
InDesign IDML reader could populate the same DOM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points


class ObjectStatus(str, Enum):
    """Classification applied to every matched (or unmatched) object."""
    MATCH = "MATCH"
    MOVED = "MOVED"
    RESIZED = "RESIZED"
    OVERFLOW = "OVERFLOW"
    UNDERFLOW = "UNDERFLOW"
    PAGE_MOVED = "PAGE_MOVED"
    ALIGNMENT_CHANGED = "ALIGNMENT_CHANGED"
    MARGIN_CHANGED = "MARGIN_CHANGED"
    CROPPED = "CROPPED"
    ROTATED = "ROTATED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    MISSING = "MISSING"
    EXTRA = "EXTRA"
    COLOR_CHANGED = "COLOR_CHANGED"
    FORMATTING_CHANGED = "FORMATTING_CHANGED"   # bold/italic emphasis differs on matched text
    BULLET_MISMATCH = "BULLET_MISMATCH"         # bullet/numbered list item count differs within a matched block
    STRUCTURE_CHANGED = "STRUCTURE_CHANGED"     # table row and/or column count differs from source


def _bbox_props(bbox: BBox) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {
        "width": max(0.0, x1 - x0),
        "height": max(0.0, y1 - y0),
        "center_x": (x0 + x1) / 2.0,
        "center_y": (y0 + y1) / 2.0,
    }


@dataclass
class BaseObject:
    """Common fields shared by every visual object on a page."""
    id: str
    page: int
    bbox: BBox
    rotation: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return _bbox_props(self.bbox)["width"]

    @property
    def height(self) -> float:
        return _bbox_props(self.bbox)["height"]

    @property
    def center(self) -> Tuple[float, float]:
        p = _bbox_props(self.bbox)
        return (p["center_x"], p["center_y"])

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 1e-6 else 0.0


@dataclass
class ParagraphObject(BaseObject):
    text: str = ""
    font: str = ""
    font_size: float = 0.0
    color: Optional[str] = None
    line_count: int = 1
    reading_order: int = 0
    column_index: int = 0
    is_bold: bool = False              # dominant span in the block is bold (>=50% of chars)
    is_italic: bool = False            # dominant span in the block is italic (>=50% of chars)
    bold_ratio: float = 0.0            # fraction (0-1) of this block's characters that are bold
    italic_ratio: float = 0.0          # fraction (0-1) of this block's characters that are italic
    bullet_count: int = 0              # number of bullet/numbered list lines detected inside this block
    bullet_items: List[str] = field(default_factory=list)  # the detected bullet line texts (for reporting)


@dataclass
class ImageObject(BaseObject):
    image_hash: Optional[str] = None      # perceptual hash (phash)
    dpi: Optional[Tuple[float, float]] = None
    is_cropped: bool = False
    clip_embedding: Optional[List[float]] = None


@dataclass
class TableObject(BaseObject):
    rows: int = 0
    cols: int = 0
    cell_texts: List[List[str]] = field(default_factory=list)


@dataclass
class PageObject:
    page_number: int
    width: float
    height: float
    margins: Dict[str, float] = field(default_factory=dict)   # {"top":.., "bottom":.., "left":.., "right":..}
    header_bbox: Optional[BBox] = None
    footer_bbox: Optional[BBox] = None
    header_text: str = ""
    footer_text: str = ""
    page_number_text: Optional[str] = None
    paragraphs: List[ParagraphObject] = field(default_factory=list)
    images: List[ImageObject] = field(default_factory=list)
    tables: List[TableObject] = field(default_factory=list)
    highlight_bboxes: List[BBox] = field(default_factory=list)

    @property
    def size(self) -> Tuple[float, float]:
        return (self.width, self.height)


@dataclass
class DocumentObject:
    path: str
    language: str = ""
    pages: List[PageObject] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def all_paragraphs(self) -> List[ParagraphObject]:
        return [p for pg in self.pages for p in pg.paragraphs]

    def all_images(self) -> List[ImageObject]:
        return [im for pg in self.pages for im in pg.images]

    def all_tables(self) -> List[TableObject]:
        return [t for pg in self.pages for t in pg.tables]
