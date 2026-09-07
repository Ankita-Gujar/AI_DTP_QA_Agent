"""
image_detector.py
------------------
Converts RawImage entries into ImageObject instances, computing a
perceptual hash (phash) for content-based matching later on and flagging
likely crops based on aspect-ratio / DPI inconsistency.
"""

from __future__ import annotations

from typing import List, Optional

from .document_object import ImageObject
from .pdf_reader import RawImage, RawPage
from .utils import QAConfig, get_logger

logger = get_logger(__name__)

try:
    import imagehash
    from PIL import Image
    import io
    _HASHING_AVAILABLE = True
except Exception:  # pragma: no cover
    imagehash = None
    Image = None
    _HASHING_AVAILABLE = False
    logger.warning(
        "Pillow/imagehash not fully available -- image content hashing disabled. "
        "Install with `pip install pillow imagehash` for perceptual image matching."
    )


def _compute_phash(raw_image: RawImage, doc) -> Optional[str]:
    """Compute a perceptual hash for the image, if the raw bytes can be pulled from the PDF."""
    if not _HASHING_AVAILABLE or doc is None:
        return None
    try:
        pix_info = doc.extract_image(raw_image.xref)
        image_bytes = pix_info["image"]
        pil_img = Image.open(io.BytesIO(image_bytes))
        return str(imagehash.phash(pil_img))
    except Exception as exc:  # pragma: no cover
        logger.debug("phash computation failed for xref %s: %s", raw_image.xref, exc)
        return None


def _looks_cropped(raw_image: RawImage) -> bool:
    """Heuristic: extreme DPI mismatch between x/y often indicates a non-uniform crop/stretch."""
    dpi_x, dpi_y = raw_image.dpi
    if dpi_x <= 0 or dpi_y <= 0:
        return False
    ratio = max(dpi_x, dpi_y) / min(dpi_x, dpi_y)
    return ratio > 1.35


def detect_images(raw_page: RawPage, page_index: int, config: QAConfig, fitz_doc=None) -> List[ImageObject]:
    images: List[ImageObject] = []
    kept = 0
    for i, raw_image in enumerate(raw_page.images):
        x0, y0, x1, y1 = raw_image.bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < config.min_image_area_pt:
            # Small inline glyphs (warning triangles, checkmarks, tiny bullet
            # icons, etc.) aren't the "images" a DTP QA pass cares about --
            # counting each occurrence as a separate image produces a flood of
            # false MISSING/MOVED findings. Skip them.
            continue
        phash = _compute_phash(raw_image, fitz_doc)
        images.append(
            ImageObject(
                id=f"img{page_index}_{kept}",
                page=page_index,
                bbox=raw_image.bbox,
                rotation=raw_image.rotation,
                confidence=1.0 if phash else 0.7,
                image_hash=phash,
                dpi=raw_image.dpi,
                is_cropped=_looks_cropped(raw_image),
                metadata={"xref": raw_image.xref, "px_size": (raw_image.width_px, raw_image.height_px)},
            )
        )
        kept += 1
    return images
