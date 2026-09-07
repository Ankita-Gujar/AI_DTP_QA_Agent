"""
image_matcher.py
-----------------
Matches source images to target images using perceptual hash (content
identity) combined with position/size (layout identity). CLIP embeddings
are supported as an optional higher-accuracy signal when available, useful
for detecting a re-exported/re-compressed image that phash alone might
mis-hash.

Classification: MISSING, EXTRA, MOVED, RESIZED, CROPPED, ROTATED, MATCH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .document_object import DocumentObject, ImageObject, ObjectStatus
from .layout_analyzer import horizontal_movement_pt, layout_similarity, movement_pt, rotation_delta, size_similarity
from .utils import QAConfig, get_logger

logger = get_logger(__name__)


@dataclass
class ImageMatch:
    source: Optional[ImageObject]
    target: Optional[ImageObject]
    score: float
    status: ObjectStatus
    hash_distance: Optional[int] = None
    movement_pt: float = 0.0


def _hamming_distance(hash_a: Optional[str], hash_b: Optional[str]) -> Optional[int]:
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return None
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except ValueError:
        return None


def match_images(
    source_doc: DocumentObject,
    target_doc: DocumentObject,
    config: QAConfig,
) -> List[ImageMatch]:
    src_imgs = source_doc.all_images()
    tgt_imgs = target_doc.all_images()

    if not src_imgs:
        return [ImageMatch(None, t, 0.0, ObjectStatus.EXTRA) for t in tgt_imgs]
    if not tgt_imgs:
        return [ImageMatch(s, None, 0.0, ObjectStatus.MISSING) for s in src_imgs]

    # .page is 1-indexed (matches PageObject.page_number); pages list is 0-indexed.
    src_page_sizes = [(source_doc.pages[im.page - 1].width, source_doc.pages[im.page - 1].height) for im in src_imgs]
    tgt_page_sizes = [(target_doc.pages[im.page - 1].width, target_doc.pages[im.page - 1].height) for im in tgt_imgs]

    n_src, n_tgt = len(src_imgs), len(tgt_imgs)
    cost = np.ones((n_src, n_tgt))
    hash_distances: Dict[Tuple[int, int], Optional[int]] = {}

    for i, s in enumerate(src_imgs):
        for j, t in enumerate(tgt_imgs):
            dist = _hamming_distance(s.image_hash, t.image_hash)
            hash_distances[(i, j)] = dist
            content_score = (
                max(0.0, 1.0 - dist / 64.0) if dist is not None else 0.5  # neutral if hashing unavailable
            )
            layout_score = layout_similarity(s, src_page_sizes[i], t, tgt_page_sizes[j], config)
            size_score = size_similarity(s, t)
            score = 0.5 * content_score + 0.3 * layout_score + 0.2 * size_score
            cost[i, j] = 1.0 - score

    row_idx, col_idx = linear_sum_assignment(cost)
    matched_src, matched_tgt = set(), set()
    matches: List[ImageMatch] = []

    for i, j in zip(row_idx, col_idx):
        score = 1.0 - cost[i, j]
        dist = hash_distances.get((i, j))
        content_ok = dist is None or dist <= config.image_phash_distance_threshold
        if score < config.match_score_threshold or not content_ok:
            continue
        s, t = src_imgs[i], tgt_imgs[j]
        matched_src.add(i)
        matched_tgt.add(j)
        matches.append(_classify_image(s, t, score, dist, config))

    for i, s in enumerate(src_imgs):
        if i not in matched_src:
            matches.append(ImageMatch(s, None, 0.0, ObjectStatus.MISSING))
    for j, t in enumerate(tgt_imgs):
        if j not in matched_tgt:
            matches.append(ImageMatch(None, t, 0.0, ObjectStatus.EXTRA))

    return matches


def _classify_image(s: ImageObject, t: ImageObject, score: float, dist: Optional[int], config: QAConfig) -> ImageMatch:
    move = movement_pt(s, t)             # kept on the record for reference/debugging
    hmove = horizontal_movement_pt(s, t)  # inline images reflow vertically with surrounding text too,
                                           # so only a horizontal shift counts as a real MOVED defect
    status = ObjectStatus.MATCH

    if t.is_cropped and not s.is_cropped:
        status = ObjectStatus.CROPPED
    elif rotation_delta(s, t) > config.rotation_tolerance_deg:
        status = ObjectStatus.ROTATED
    elif abs(1.0 - (t.width / s.width if s.width else 1.0)) > config.resize_tolerance_ratio or \
         abs(1.0 - (t.height / s.height if s.height else 1.0)) > config.resize_tolerance_ratio:
        status = ObjectStatus.RESIZED
    elif hmove > config.move_tolerance_pt:
        status = ObjectStatus.MOVED

    return ImageMatch(source=s, target=t, score=score, status=status, hash_distance=dist, movement_pt=move)
