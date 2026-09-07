"""
pdf_reader.py
-------------
Low-level PDF extraction using PyMuPDF (fitz). Produces raw per-page data
(text blocks/lines/spans, raw image rects, raw table rects) that the
detector modules (paragraph_detector, image_detector, table_detector) turn
into DOM objects.

This module deliberately does NOT build ParagraphObject/ImageObject/etc.
itself -- it only reads. Keeping extraction and interpretation separate
means a bug in "how we decide something is a paragraph" never requires
touching PDF parsing code, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    fitz = None
    _FITZ_AVAILABLE = False
    logger.warning(
        "PyMuPDF (fitz) is not installed. Install with `pip install pymupdf` "
        "to enable real PDF reading. Falling back to a no-op reader."
    )


@dataclass
class RawTextSpan:
    text: str
    bbox: Tuple[float, float, float, float]
    font: str
    size: float
    color: int
    rotation: float = 0.0
    flags: int = 0  # PyMuPDF span flags bitmask (bit 1=italic, bit 4=bold); see _span_is_bold/_span_is_italic


@dataclass
class RawTextLine:
    spans: List[RawTextSpan]
    bbox: Tuple[float, float, float, float]


@dataclass
class RawTextBlock:
    lines: List[RawTextLine]
    bbox: Tuple[float, float, float, float]


@dataclass
class RawImage:
    bbox: Tuple[float, float, float, float]
    rotation: float
    xref: int
    width_px: int
    height_px: int
    dpi: Tuple[float, float]
    raw_bytes: Optional[bytes] = None


@dataclass
class RawTable:
    bbox: Tuple[float, float, float, float]
    rows: int
    cols: int
    cell_texts: List[List[str]]


@dataclass
class RawPage:
    page_number: int
    width: float
    height: float
    text_blocks: List[RawTextBlock] = field(default_factory=list)
    images: List[RawImage] = field(default_factory=list)
    tables: List[RawTable] = field(default_factory=list)
    highlight_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


class PDFReader:
    """Reads a PDF file and yields RawPage objects, one page at a time.

    Streaming page-by-page (rather than loading the whole document into
    memory) is what lets the pipeline scale to 1000+ page documents.
    """

    def __init__(self, path: str):
        self.path = path
        self._doc = None
        if _FITZ_AVAILABLE:
            self._doc = fitz.open(path)

    def __len__(self) -> int:
        if self._doc is None:
            return 0
        return self._doc.page_count

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()

    def __enter__(self) -> "PDFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def read_page(self, index: int) -> RawPage:
        if self._doc is None:
            raise RuntimeError(
                "PyMuPDF is not installed -- cannot read real PDFs. "
                "Run `pip install pymupdf` in your environment."
            )
        page = self._doc[index]
        rect = page.rect
        raw_page = RawPage(page_number=index + 1, width=rect.width, height=rect.height)

        raw_page.text_blocks = self._extract_text_blocks(page)
        raw_page.images = self._extract_images(page)
        raw_page.tables = self._extract_tables(page)
        raw_page.highlight_bboxes = self._extract_highlights(page)
        return raw_page

    def iter_pages(self):
        for i in range(len(self)):
            yield self.read_page(i)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text_blocks(page) -> List[RawTextBlock]:
        blocks: List[RawTextBlock] = []
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:
                continue  # skip image blocks here, handled separately
            lines: List[RawTextLine] = []
            for line in block.get("lines", []):
                spans: List[RawTextSpan] = []
                for span in line.get("spans", []):
                    spans.append(
                        RawTextSpan(
                            text=span.get("text", ""),
                            bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                            font=span.get("font", ""),
                            size=span.get("size", 0.0),
                            color=span.get("color", 0),
                            rotation=0.0,
                            flags=span.get("flags", 0),
                        )
                    )
                if spans:
                    lines.append(RawTextLine(spans=spans, bbox=tuple(line.get("bbox", (0, 0, 0, 0)))))
            if lines:
                blocks.append(RawTextBlock(lines=lines, bbox=tuple(block.get("bbox", (0, 0, 0, 0)))))
        return blocks

    @staticmethod
    def _extract_images(page) -> List[RawImage]:
        images: List[RawImage] = []
        for img in page.get_images(full=True):
            xref = img[0]
            width_px, height_px = img[2], img[3]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects:
                dpi_x = (width_px / rect.width * 72.0) if rect.width else 0.0
                dpi_y = (height_px / rect.height * 72.0) if rect.height else 0.0
                images.append(
                    RawImage(
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        rotation=0.0,
                        xref=xref,
                        width_px=width_px,
                        height_px=height_px,
                        dpi=(dpi_x, dpi_y),
                    )
                )
        return images

    @staticmethod
    def _extract_highlights(page) -> List[Tuple[float, float, float, float]]:
        """Bounding boxes of PDF highlight annotations (the yellow-marker style
        highlight, PDF Subtype /Highlight) on this page -- used to catch stray
        highlight formatting that leaked into a translated file (e.g. a
        reviewer's or translator's highlight left in by mistake)."""
        boxes: List[Tuple[float, float, float, float]] = []
        try:
            annots = page.annots()
        except Exception:
            return boxes
        if not annots:
            return boxes
        for annot in annots:
            try:
                if annot.type[1] == "Highlight":
                    r = annot.rect
                    boxes.append((r.x0, r.y0, r.x1, r.y1))
            except Exception:
                continue
        return boxes

    @staticmethod
    def _extract_tables(page) -> List[RawTable]:
        tables: List[RawTable] = []
        finder = getattr(page, "find_tables", None)
        if finder is None:
            return tables  # older PyMuPDF without table support

        def _run(**kwargs) -> List:
            try:
                result = finder(**kwargs)
            except Exception as exc:  # pragma: no cover
                logger.debug("Table detection failed on page (%s): %s", kwargs, exc)
                return []
            return list(getattr(result, "tables", []))

        # Default strategy relies on visible ruling lines, which real-world
        # print-ready PDFs frequently don't have (borderless/whitespace-
        # aligned tables are extremely common in DTP exports). When it finds
        # nothing, retry with the "text" strategy, which clusters text into
        # a table purely from column/row alignment -- no lines required.
        # This is why tables with no visible grid (e.g. a plain two-column
        # term/definition layout) were previously never detected as tables
        # at all, and so could never be compared as one.
        found = _run()
        if not found:
            found = _run(strategy="text")

        for tbl in found:
            try:
                cell_texts = tbl.extract()
            except Exception:
                cell_texts = []
            bbox = tuple(tbl.bbox) if hasattr(tbl, "bbox") else (0, 0, 0, 0)
            rows = len(cell_texts)
            cols = len(cell_texts[0]) if cell_texts else 0
            if rows < 2 or cols < 2:
                # A 1-row or 1-column "table" from the text-alignment
                # strategy is usually just an ordinary paragraph that
                # happened to align into a single column -- skip it rather
                # than report every plain paragraph as a table.
                continue
            tables.append(RawTable(bbox=bbox, rows=rows, cols=cols, cell_texts=cell_texts))
        return tables


# PyMuPDF span "flags" bitmask constants (see fitz docs: get_text("dict")).
_FLAG_ITALIC = 1 << 1   # 2
_FLAG_BOLD = 1 << 4     # 16


def span_is_bold(span: "RawTextSpan") -> bool:
    """True if a text span is bold, using both the PDF font-descriptor flag
    (reliable when the PDF embeds proper font metadata) and a fallback check
    on the font name (some exporters -- InDesign, Word-to-PDF, etc. -- don't
    set the flag bit correctly but always name the font e.g. 'Arial-Bold')."""
    if span.flags & _FLAG_BOLD:
        return True
    name = (span.font or "").lower()
    return "bold" in name or "black" in name or "heavy" in name


def span_is_italic(span: "RawTextSpan") -> bool:
    """True if a text span is italic/oblique -- see span_is_bold for why both
    the flag bit and the font name are checked."""
    if span.flags & _FLAG_ITALIC:
        return True
    name = (span.font or "").lower()
    return "italic" in name or "oblique" in name


def is_pymupdf_available() -> bool:
    return _FITZ_AVAILABLE
