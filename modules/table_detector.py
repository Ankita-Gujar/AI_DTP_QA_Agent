"""
table_detector.py
------------------
Converts RawTable entries (from PyMuPDF's find_tables()) into TableObject
instances. Kept as its own module so a future upgrade (e.g. a dedicated
table-detection model for PDFs without embedded table structure) only
needs to change this file.
"""

from __future__ import annotations

from typing import List

from .document_object import TableObject
from .pdf_reader import RawPage
from .utils import QAConfig


def detect_tables(raw_page: RawPage, page_index: int, config: QAConfig) -> List[TableObject]:
    tables: List[TableObject] = []
    for i, raw_table in enumerate(raw_page.tables):
        tables.append(
            TableObject(
                id=f"tbl{page_index}_{i}",
                page=page_index,
                bbox=raw_table.bbox,
                rows=raw_table.rows,
                cols=raw_table.cols,
                cell_texts=raw_table.cell_texts,
                confidence=1.0,
                metadata={},
            )
        )
    return tables
