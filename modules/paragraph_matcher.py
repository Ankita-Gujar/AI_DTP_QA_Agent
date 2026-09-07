"""
paragraph_matcher.py
---------------------
Matches source paragraphs to target paragraphs using a weighted blend of:

    40% semantic similarity      (ai_matcher -- cross-lingual embeddings)
    30% layout similarity        (layout_analyzer -- relative on-page position)
    15% object size similarity   (layout_analyzer -- width/height ratio)
    15% reading order similarity (layout_analyzer -- sequence position)

We deliberately do NOT match on raw x/y/width/height/line-count alone --
that produces false positives whenever translation expands/contracts text
(very common, e.g. German vs. Chinese). Semantic similarity anchors the
match on *which paragraph this actually is*, then layout signals classify
*what changed about it*.

Matching itself is solved as an optimal assignment problem (Hungarian
algorithm via scipy) over each page-neighborhood, not greedy nearest-match,
which avoids one strong match "stealing" a paragraph that another needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ai_matcher import SemanticMatcher
from .document_object import DocumentObject, ObjectStatus, ParagraphObject
from .layout_analyzer import (
    alignment_delta,
    height_ratio,
    horizontal_movement_pt,
    layout_similarity,
    margin_delta,
    movement_pt,
    reading_order_similarity,
    size_similarity,
)
from .utils import QAConfig, get_logger

logger = get_logger(__name__)


@dataclass
class ParagraphMatch:
    source: Optional[ParagraphObject]
    target: Optional[ParagraphObject]
    score: float
    status: ObjectStatus
    sub_scores: Dict[str, float]
    movement_pt: float = 0.0
    alignment_delta_pt: float = 0.0
    height_ratio: float = 1.0


def _candidate_window(src_index: int, total: int, window: int = 8) -> Tuple[int, int]:
    """Restrict candidate matches to a neighborhood around the source's position in
    reading order -- keeps the assignment problem tractable on very long documents
    and avoids matching paragraph #3 to paragraph #480 by coincidence."""
    lo = max(0, src_index - window)
    hi = min(total, src_index + window + 1)
    return lo, hi


def match_paragraphs(
    source_doc: DocumentObject,
    target_doc: DocumentObject,
    config: QAConfig,
) -> List[ParagraphMatch]:
    src_paras = source_doc.all_paragraphs()
    tgt_paras = target_doc.all_paragraphs()

    matcher = SemanticMatcher.get(config)
    matches: List[ParagraphMatch] = []

    if not src_paras:
        for t in tgt_paras:
            matches.append(_extra_match(t))
        return matches
    if not tgt_paras:
        for s in src_paras:
            matches.append(_missing_match(s))
        return matches

    n_src, n_tgt = len(src_paras), len(tgt_paras)
    cost_matrix = np.ones((n_src, n_tgt))
    sub_score_cache: Dict[Tuple[int, int], Dict[str, float]] = {}

    src_texts = [p.text for p in src_paras]
    tgt_texts = [p.text for p in tgt_paras]
    full_sim = matcher.similarity_matrix(src_texts, tgt_texts)

    # .page is 1-indexed (matches PageObject.page_number); pages list is 0-indexed.
    src_page_sizes = [(source_doc.pages[p.page - 1].width, source_doc.pages[p.page - 1].height) for p in src_paras]
    tgt_page_sizes = [(target_doc.pages[p.page - 1].width, target_doc.pages[p.page - 1].height) for p in tgt_paras]

    for i, src in enumerate(src_paras):
        lo, hi = _candidate_window(i, n_tgt, window=max(8, abs(n_tgt - n_src) + 8))
        for j in range(lo, hi):
            tgt = tgt_paras[j]
            semantic = float(full_sim[i, j])
            layout = layout_similarity(src, src_page_sizes[i], tgt, tgt_page_sizes[j], config)
            size = size_similarity(src, tgt)
            order = reading_order_similarity(src.reading_order, n_src, tgt.reading_order, n_tgt)
            score = (
                config.weight_semantic * semantic
                + config.weight_layout * layout
                + config.weight_size * size
                + config.weight_reading_order * order
            )
            cost_matrix[i, j] = 1.0 - score
            sub_score_cache[(i, j)] = {
                "semantic": semantic, "layout": layout, "size": size, "reading_order": order,
            }

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matched_src, matched_tgt = set(), set()
    for i, j in zip(row_idx, col_idx):
        score = 1.0 - cost_matrix[i, j]
        if score < config.match_score_threshold or (i, j) not in sub_score_cache:
            continue  # treat as no match; handled as MISSING/EXTRA below
        src, tgt = src_paras[i], tgt_paras[j]
        matched_src.add(i)
        matched_tgt.add(j)
        matches.append(_classify(src, tgt, score, sub_score_cache[(i, j)], config))

    for i, src in enumerate(src_paras):
        if i not in matched_src:
            matches.append(_missing_match(src))
    for j, tgt in enumerate(tgt_paras):
        if j not in matched_tgt:
            matches.append(_extra_match(tgt))

    return matches


def _missing_match(src: ParagraphObject) -> ParagraphMatch:
    return ParagraphMatch(source=src, target=None, score=0.0, status=ObjectStatus.MISSING, sub_scores={})


def _extra_match(tgt: ParagraphObject) -> ParagraphMatch:
    return ParagraphMatch(source=None, target=tgt, score=0.0, status=ObjectStatus.EXTRA, sub_scores={})


def _colors_differ(src: ParagraphObject, tgt: ParagraphObject) -> bool:
    """True if both paragraphs have a known dominant font color and they differ.

    Skipped when either side has no color info, or when the target text is
    trivially short (e.g. a lone bullet/number glyph), since dominant-color
    detection on a couple of characters is unreliable.
    """
    if not src.color or not tgt.color:
        return False
    if len(tgt.text.strip()) < 2 or len(src.text.strip()) < 2:
        return False
    return src.color.lower() != tgt.color.lower()


def _formatting_differs(src: ParagraphObject, tgt: ParagraphObject, config: QAConfig) -> bool:
    """True if this block's bold/italic emphasis differs between source and
    target enough to be a real, visible change -- catches both:
      * a whole block's dominant weight flipping (e.g. a bolded heading that
        came through as regular weight), and
      * emphasis added/removed on only *part* of a block (e.g. one sentence
        bolded inside an otherwise-plain paragraph) that never flips either
        side's 50%-majority is_bold/is_italic flag but is still a visible
        formatting defect a designer would want flagged.
    Falls back to the plain majority-vote flags if ratio data isn't present
    (e.g. objects built by older code paths / tests), so this stays backward
    compatible.
    """
    if src.is_bold != tgt.is_bold or src.is_italic != tgt.is_italic:
        return True
    bold_delta = abs(getattr(src, "bold_ratio", 0.0) - getattr(tgt, "bold_ratio", 0.0))
    italic_delta = abs(getattr(src, "italic_ratio", 0.0) - getattr(tgt, "italic_ratio", 0.0))
    return bold_delta > config.formatting_ratio_tolerance or italic_delta > config.formatting_ratio_tolerance


def _bullet_counts_differ(src: ParagraphObject, tgt: ParagraphObject) -> bool:
    """True if this matched text block contains a different number of
    bullet/numbered list lines in target vs. source -- catches a bullet
    point silently dropped or an extra one added *inside* an otherwise-
    matched block (a whole bullet missing as its own block is instead
    caught by the normal paragraph MISSING/EXTRA check)."""
    return src.bullet_count != tgt.bullet_count


def _classify(
    src: ParagraphObject,
    tgt: ParagraphObject,
    score: float,
    sub_scores: Dict[str, float],
    config: QAConfig,
) -> ParagraphMatch:
    move = movement_pt(src, tgt)                 # kept on the record for reference/debugging
    hmove = horizontal_movement_pt(src, tgt)      # what actually drives the MOVED flag
    align = alignment_delta(src, tgt)
    h_ratio = height_ratio(src, tgt)

    status = ObjectStatus.MATCH

    if src.page != tgt.page:
        status = ObjectStatus.PAGE_MOVED
    elif _bullet_counts_differ(src, tgt):
        # Checked before OVERFLOW/UNDERFLOW/etc: a dropped or added bullet line
        # almost always also changes the block's height, so if height were
        # checked first the more specific, actionable "bullet count changed"
        # finding would be masked by a generic "overflow" one.
        status = ObjectStatus.BULLET_MISMATCH
    elif _formatting_differs(src, tgt, config):
        # Same reasoning as bullets: bold/italic changes are frequently on
        # headings or short blocks whose height also shifts (different font
        # metrics, different translated length) -- checking this before
        # OVERFLOW/UNDERFLOW is what makes it actually surface instead of
        # being silently swallowed by the height check almost every time.
        status = ObjectStatus.FORMATTING_CHANGED
    elif _colors_differ(src, tgt):
        status = ObjectStatus.COLOR_CHANGED
    elif h_ratio > config.overflow_height_ratio:
        status = ObjectStatus.OVERFLOW
    elif h_ratio < config.underflow_height_ratio:
        status = ObjectStatus.UNDERFLOW
    elif config.flag_paragraph_flow_moves and move > config.move_tolerance_pt:
        # Opt-in strict mode: flags any positional drift, including the
        # vertical reflow that's expected on translated documents.
        status = ObjectStatus.MOVED
    elif hmove > config.move_tolerance_pt:
        # Horizontal-only check (default): a paragraph sliding sideways is a
        # real layout defect; sliding down/up because earlier text got
        # longer/shorter is not.
        status = ObjectStatus.MOVED
    elif align > config.alignment_tolerance_pt:
        status = ObjectStatus.ALIGNMENT_CHANGED
    elif abs(1.0 - h_ratio) > config.resize_tolerance_ratio:
        status = ObjectStatus.RESIZED

    return ParagraphMatch(
        source=src,
        target=tgt,
        score=score,
        status=status,
        sub_scores=sub_scores,
        movement_pt=move,
        alignment_delta_pt=align,
        height_ratio=h_ratio,
    )
