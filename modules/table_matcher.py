"""
table_matcher.py
-----------------
Matches source tables to target tables by position/size, then classifies
structural changes (split, merged, overflow, resized, missing/extra).

"Split" and "merged" are inferred by comparing row/column counts: if a
single source table's row range spans two nearby target tables (or vice
versa), we mark them as SPLIT/MERGED rather than one MATCH + one EXTRA/MISSING.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ai_matcher import SemanticMatcher
from .document_object import DocumentObject, ObjectStatus, TableObject
from .layout_analyzer import layout_similarity, size_similarity
from .utils import QAConfig, iou


def _table_text(t: TableObject) -> str:
    """Flatten a table's extracted cell text into one string for semantic
    comparison -- this is what lets table matching survive the table simply
    moving down/up the page (e.g. because a paragraph above it got longer or
    shorter in translation), the same way paragraph matching survives it via
    semantic similarity instead of relying on raw position alone."""
    return " ".join(cell for row in t.cell_texts for cell in row if cell and cell.strip())


@dataclass
class TableMatch:
    source: Optional[TableObject]
    target: Optional[TableObject]
    score: float
    status: ObjectStatus
    row_delta: int = 0
    col_delta: int = 0


def match_tables(source_doc: DocumentObject, target_doc: DocumentObject, config: QAConfig) -> List[TableMatch]:
    src_tables = source_doc.all_tables()
    tgt_tables = target_doc.all_tables()

    if not src_tables:
        return [TableMatch(None, t, 0.0, ObjectStatus.EXTRA) for t in tgt_tables]
    if not tgt_tables:
        return [TableMatch(s, None, 0.0, ObjectStatus.MISSING) for s in src_tables]

    # .page is 1-indexed (matches PageObject.page_number); pages list is 0-indexed.
    src_page_sizes = [(source_doc.pages[t.page - 1].width, source_doc.pages[t.page - 1].height) for t in src_tables]
    tgt_page_sizes = [(target_doc.pages[t.page - 1].width, target_doc.pages[t.page - 1].height) for t in tgt_tables]

    matcher = SemanticMatcher.get(config)
    src_texts = [_table_text(t) for t in src_tables]
    tgt_texts = [_table_text(t) for t in tgt_tables]
    full_sim = matcher.similarity_matrix(src_texts, tgt_texts)

    n_src, n_tgt = len(src_tables), len(tgt_tables)
    cost = np.ones((n_src, n_tgt))
    for i, s in enumerate(src_tables):
        for j, t in enumerate(tgt_tables):
            layout = layout_similarity(s, src_page_sizes[i], t, tgt_page_sizes[j], config)
            size = size_similarity(s, t)
            semantic = float(full_sim[i, j])
            # Content-anchored, same philosophy as paragraph matching: semantic
            # similarity of the extracted cell text identifies *which* table
            # this is even when it has drifted on the page; layout/size then
            # only need to describe *what changed about it*, not find it.
            score = 0.5 * semantic + 0.3 * layout + 0.2 * size
            cost[i, j] = 1.0 - score

    row_idx, col_idx = linear_sum_assignment(cost)
    matched_src, matched_tgt = set(), set()
    matches: List[TableMatch] = []

    for i, j in zip(row_idx, col_idx):
        score = 1.0 - cost[i, j]
        if score < config.match_score_threshold:
            continue
        s, t = src_tables[i], tgt_tables[j]
        matched_src.add(i)
        matched_tgt.add(j)
        matches.append(_classify_table(s, t, score, config))

    unmatched_src = [(i, s) for i, s in enumerate(src_tables) if i not in matched_src]
    unmatched_tgt = [(j, t) for j, t in enumerate(tgt_tables) if j not in matched_tgt]

    # Try to detect split/merge among the leftover, spatially-close tables before
    # giving up and calling them plain MISSING/EXTRA.
    used_extra_idx = set()
    for i, s in unmatched_src:
        nearby = [
            (j, t) for j, t in unmatched_tgt
            if j not in used_extra_idx and t.page == s.page and iou(s.bbox, t.bbox) > 0.05
        ]
        if len(nearby) >= 2:
            for j, t in nearby:
                used_extra_idx.add(j)
                matches.append(TableMatch(s, t, 0.5, ObjectStatus.SPLIT))
        else:
            matches.append(TableMatch(s, None, 0.0, ObjectStatus.MISSING))

    for j, t in unmatched_tgt:
        if j in used_extra_idx:
            continue
        matches.append(TableMatch(None, t, 0.0, ObjectStatus.EXTRA))

    return matches


def _classify_table(s: TableObject, t: TableObject, score: float, config: QAConfig) -> TableMatch:
    row_delta = t.rows - s.rows
    col_delta = t.cols - s.cols
    status = ObjectStatus.MATCH

    if t.height > s.height * config.overflow_height_ratio:
        status = ObjectStatus.OVERFLOW
    elif row_delta != 0 or col_delta != 0:
        # Row/column *count* mismatch is a structural defect (a row or column
        # was dropped/added/merged) -- flagged distinctly from a plain
        # RESIZED (same structure, just bigger/smaller cells), since a
        # missing row/column is a much more actionable, higher-severity issue.
        status = ObjectStatus.STRUCTURE_CHANGED
    elif abs(1.0 - size_similarity(s, t)) > config.resize_tolerance_ratio:
        status = ObjectStatus.RESIZED

    return TableMatch(source=s, target=t, score=score, status=status, row_delta=row_delta, col_delta=col_delta)
