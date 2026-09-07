"""
page_renderer.py
------------------
Renders PDF pages to raster images (via PyMuPDF) and draws color-coded
overlay rectangles for each difference, for the visual side-by-side
comparison view in the Streamlit UI.

Color legend (fixed, matches the spec):
    Red    = Missing
    Blue   = Moved
    Orange = Resized
    Purple = Overflow
    Green  = Match (rarely drawn -- usually omitted to reduce clutter)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .difference_builder import DifferenceRecord
from .utils import QAConfig, get_logger

logger = get_logger(__name__)

try:
    import fitz
    _FITZ_AVAILABLE = True
except Exception:  # pragma: no cover
    fitz = None
    _FITZ_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    _PIL_AVAILABLE = False


STATUS_COLOR = {
    "MISSING": (192, 57, 43),        # red
    "MOVED": (41, 128, 185),         # blue
    "PAGE_MOVED": (41, 128, 185),    # blue
    "RESIZED": (211, 84, 0),         # orange
    "OVERFLOW": (142, 68, 173),      # purple
    "UNDERFLOW": (142, 68, 173),     # purple
    "MATCH": (39, 174, 96),          # green
    "EXTRA": (192, 57, 43),          # red (treated as a "new, unexpected" flag)
    "ALIGNMENT_CHANGED": (211, 84, 0),
    "MARGIN_CHANGED": (211, 84, 0),
    "CROPPED": (192, 57, 43),
    "ROTATED": (211, 84, 0),
    "SPLIT": (142, 68, 173),
    "MERGED": (142, 68, 173),
    "COLOR_CHANGED": (192, 24, 122),  # magenta/pink
    "FORMATTING_CHANGED": (192, 24, 122),  # magenta/pink -- typography, same family as COLOR_CHANGED
    "BULLET_MISMATCH": (39, 174, 96),      # green -- bullets/numbering
    "STRUCTURE_CHANGED": (211, 84, 0),     # orange -- table row/column count changed
}
DEFAULT_COLOR = (127, 140, 141)  # grey


def render_page_to_image(path: str, page_index: int, dpi: int = 150):
    """Returns a PIL.Image of the given 0-indexed page, or None if rendering is unavailable."""
    if not (_FITZ_AVAILABLE and _PIL_AVAILABLE):
        logger.warning("PyMuPDF/Pillow not available -- cannot render page previews.")
        return None
    doc = fitz.open(path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return img.convert("RGB")
    finally:
        doc.close()


def draw_highlights(image, records: List[DifferenceRecord], page_number: int, dpi: int = 150):
    """Draw color-coded rectangles for every difference on this page onto a copy of `image`."""
    if not _PIL_AVAILABLE or image is None:
        return image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    scale = dpi / 72.0  # PDF points -> pixels at render DPI

    for r in records:
        if r.page != page_number or not r.exact_position:
            continue
        color = STATUS_COLOR.get(r.status, DEFAULT_COLOR)
        x0, y0, x1, y1 = [v * scale for v in r.exact_position]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = r.status
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)

    return overlay


def build_comparison_gallery(
    source_path: str,
    target_path: str,
    records: List[DifferenceRecord],
    config: QAConfig,
    max_pages: Optional[int] = None,
    dpi_override: Optional[int] = None,
) -> List[Dict]:
    """Returns a list of {"page": n, "source_image": PIL.Image, "target_image": PIL.Image}
    for use in the Streamlit visual comparison tab.

    dpi_override lets the caller render at a different resolution than
    config.render_dpi (e.g. a UI "zoom" control) without mutating the shared
    QAConfig -- higher DPI = a larger, more zoomed-in image.
    """
    gallery = []
    if not (_FITZ_AVAILABLE and _PIL_AVAILABLE):
        return gallery

    dpi = dpi_override or config.render_dpi
    src_doc = fitz.open(source_path)
    tgt_doc = fitz.open(target_path)
    try:
        n_pages = tgt_doc.page_count
        if max_pages:
            n_pages = min(n_pages, max_pages)
        for i in range(n_pages):
            src_img = render_page_to_image(source_path, i, dpi) if i < src_doc.page_count else None
            tgt_img = render_page_to_image(target_path, i, dpi)
            tgt_img_highlighted = draw_highlights(tgt_img, records, i + 1, dpi)
            gallery.append({"page": i + 1, "source_image": src_img, "target_image": tgt_img_highlighted})
    finally:
        src_doc.close()
        tgt_doc.close()
    return gallery
