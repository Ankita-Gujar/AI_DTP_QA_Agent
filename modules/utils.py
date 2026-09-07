"""
utils.py
--------
Shared utilities: logging setup, configuration, geometry helpers, and
small numeric helpers used across the DTP QA Agent.

No module in this project should re-implement geometry math -- everything
lives here so behavior stays consistent across detectors and matchers.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


def _default_level() -> int:
    """Root log level for the app's own loggers, overridable via the
    LOG_LEVEL environment variable (DEBUG/INFO/WARNING/ERROR/CRITICAL) so a
    production deployment can turn verbosity up or down without a code
    change. Falls back to INFO on an unset or unrecognized value."""
    name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, name, None)
    return level if isinstance(level, int) else logging.INFO


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly (no duplicate handlers).

    `level` defaults to the LOG_LEVEL environment variable (see
    `_default_level`) rather than a hardcoded INFO, so it can be tuned per
    deployment.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level if level is not None else _default_level())
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class QAConfig:
    """Central tunable configuration for the whole pipeline.

    Keeping every threshold here (instead of scattered magic numbers)
    means QA behavior can be tuned per-client without touching logic code.
    """

    # --- matching weights (must sum to 1.0) ---
    weight_semantic: float = 0.40
    weight_layout: float = 0.30
    weight_size: float = 0.15
    weight_reading_order: float = 0.15

    # --- thresholds ---
    match_score_threshold: float = 0.55      # below this -> considered MISSING/EXTRA
    move_tolerance_pt: float = 6.0           # HORIZONTAL movement below this is noise, not MOVED
                                              # (only horizontal shift is checked -- vertical shift is
                                              # expected whenever translated text is longer/shorter than
                                              # the source and should not be treated as a layout defect)
    resize_tolerance_ratio: float = 0.06      # relative size change tolerance (6%)
    overflow_height_ratio: float = 1.18       # target height / source height above this -> OVERFLOW
    underflow_height_ratio: float = 0.82      # below this -> UNDERFLOW
    alignment_tolerance_pt: float = 3.0
    margin_tolerance_pt: float = 4.0
    rotation_tolerance_deg: float = 2.0
    image_phash_distance_threshold: int = 10  # hamming distance above this -> not same image
    header_zone_ratio: float = 0.10           # top 10% of page = header zone
    footer_zone_ratio: float = 0.10           # bottom 10% of page = footer zone
    paragraph_line_gap_factor: float = 1.6     # merge lines into a paragraph if gap < factor * line height
    min_image_area_pt: float = 400.0          # skip raster images smaller than this (~20x20pt) --
                                               # filters out inline icons/glyphs (e.g. warning triangles)
                                               # that would otherwise be reported as separate "images"
    flag_paragraph_flow_moves: bool = False   # if False (default), vertical-only paragraph shifts caused
                                               # by translated text being longer/shorter are not flagged
    formatting_ratio_tolerance: float = 0.15  # if a block's bold (or italic) character ratio shifts by more
                                               # than this between source and target -> FORMATTING_CHANGED,
                                               # even if neither side crosses the 50% "dominant style" mark.
                                               # Catches e.g. one sentence bolded inside an otherwise-plain
                                               # paragraph, which a whole-block majority-vote check would miss.

    # --- performance ---
    max_workers: int = 4
    render_dpi: int = 150
    enable_parallel: bool = True

    # --- embeddings ---
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 32


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

Number = float


def bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned bboxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def safe_ratio(a: Number, b: Number) -> float:
    """a / b guarding against zero division. Returns 1.0 if both are ~0."""
    if abs(b) < 1e-6:
        return 1.0 if abs(a) < 1e-6 else float("inf")
    return a / b


def clamp(value: Number, lo: Number, hi: Number) -> Number:
    return max(lo, min(hi, value))


def normalize_score(distance: float, scale: float) -> float:
    """Convert a distance (0..inf) into a similarity score (1..0) with exponential decay."""
    if scale <= 0:
        scale = 1e-6
    return math.exp(-distance / scale)


def relative_position(bbox: Tuple[float, float, float, float],
                       page_size: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """BBox expressed as a fraction of page width/height (0..1). Enables cross-page-size comparison."""
    pw, ph = page_size
    pw = pw or 1.0
    ph = ph or 1.0
    return (bbox[0] / pw, bbox[1] / ph, bbox[2] / pw, bbox[3] / ph)


def movement_points(src_center: Tuple[float, float], tgt_center: Tuple[float, float]) -> float:
    """Absolute on-page movement in points (PDF units)."""
    return euclidean_distance(src_center, tgt_center)


@dataclass
class Thresholded:
    """Small helper to describe a pass/fail with a margin, useful for report explanations."""
    passed: bool
    value: float
    threshold: float

    def describe(self, unit: str = "pt") -> str:
        status = "within tolerance" if self.passed else "out of tolerance"
        return f"{self.value:.2f}{unit} ({status}, threshold {self.threshold:.2f}{unit})"
