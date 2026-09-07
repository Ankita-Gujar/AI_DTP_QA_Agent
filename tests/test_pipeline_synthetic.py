"""
Synthetic end-to-end test that exercises detection->matching->difference
building->report generation WITHOUT needing real PDFs or PyMuPDF, so it can
run in any environment (including one without pymupdf installed). This
validates the core matching/classification/report logic directly against
the DOM layer.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.document_object import (
    DocumentObject, PageObject, ParagraphObject, ImageObject, TableObject,
)
from modules.layout_analyzer import analyze_document_layout
from modules.difference_builder import build_differences, summarize
from modules.prompt_builder import build_prompt
from modules.report_builder import build_fallback_report_text, export_all
from modules.utils import QAConfig


def make_source_doc():
    page = PageObject(page_number=1, width=595, height=842)
    page.paragraphs = [
        ParagraphObject(id="p0", page=0, bbox=(50, 50, 545, 90), text="Welcome to our product guide.",
                         font="Helvetica", font_size=12, line_count=1, reading_order=0),
        ParagraphObject(id="p1", page=0, bbox=(50, 100, 545, 160), text="This section explains installation steps.",
                         font="Helvetica", font_size=11, line_count=2, reading_order=1),
        ParagraphObject(id="p2", page=0, bbox=(50, 170, 545, 210), text="Contact support for further help.",
                         font="Helvetica", font_size=11, line_count=1, reading_order=2),
        # p3 -> bold heading in source; target loses the bold => FORMATTING_CHANGED
        ParagraphObject(id="p3", page=0, bbox=(50, 470, 545, 495), text="Important Safety Notice",
                         font="Helvetica-Bold", font_size=12, line_count=1, reading_order=3, is_bold=True),
        # p4 -> 3-item bullet list in source; target drops one bullet => BULLET_MISMATCH
        ParagraphObject(id="p4", page=0, bbox=(50, 500, 545, 560), reading_order=4,
                         text="\u2022 Item one\n\u2022 Item two\n\u2022 Item three",
                         font="Helvetica", font_size=11, line_count=3,
                         bullet_count=3, bullet_items=["Item one", "Item two", "Item three"]),
    ]
    page.images = [
        ImageObject(id="img0", page=0, bbox=(50, 220, 250, 350), image_hash="abcd1234abcd1234"),
    ]
    page.tables = [
        TableObject(id="tbl0", page=0, bbox=(50, 360, 545, 460), rows=3, cols=2,
                    cell_texts=[["A", "B"], ["1", "2"], ["3", "4"]]),
        # tbl1 -> same size/position in target but a column is dropped => STRUCTURE_CHANGED
        TableObject(id="tbl1", page=0, bbox=(50, 600, 545, 650), rows=2, cols=3,
                    cell_texts=[["A", "B", "C"], ["1", "2", "3"]]),
    ]
    doc = DocumentObject(path="source.pdf", language="English", pages=[page])
    return doc


def make_target_doc():
    page = PageObject(page_number=1, width=595, height=842)
    page.paragraphs = [
        # p0 -> matched, moved slightly (within tolerance) => MATCH
        ParagraphObject(id="t0", page=0, bbox=(50, 50, 545, 92), text="Bienvenue dans notre guide produit.",
                         font="Helvetica", font_size=12, line_count=1, reading_order=0),
        # p1 -> matched but much taller (translation expanded) => OVERFLOW
        ParagraphObject(id="t1", page=0, bbox=(50, 100, 545, 230), text="Cette section explique les etapes d'installation en detail supplementaire.",
                         font="Helvetica", font_size=11, line_count=4, reading_order=1),
        # p2 is MISSING in target (dropped)
        # extra paragraph not in source => EXTRA
        ParagraphObject(id="t2", page=0, bbox=(50, 700, 545, 740), text="Nouvelle mention legale ajoutee.",
                         font="Helvetica", font_size=9, line_count=1, reading_order=2),
        # t3 -> same heading, but bold was lost during DTP => FORMATTING_CHANGED
        ParagraphObject(id="t3", page=0, bbox=(50, 470, 545, 495), text="Avis de securite important",
                         font="Helvetica", font_size=12, line_count=1, reading_order=3, is_bold=False),
        # t4 -> only 2 of the 3 source bullets survived => BULLET_MISMATCH
        ParagraphObject(id="t4", page=0, bbox=(50, 500, 545, 555), reading_order=4,
                         text="\u2022 Premier point\n\u2022 Deuxieme point",
                         font="Helvetica", font_size=11, line_count=2,
                         bullet_count=2, bullet_items=["Premier point", "Deuxieme point"]),
    ]
    page.images = [
        # image moved significantly
        ImageObject(id="imgt0", page=0, bbox=(300, 250, 500, 380), image_hash="abcd1234abcd1230"),
    ]
    page.tables = [
        # table resized (rows differ)
        TableObject(id="tblt0", page=0, bbox=(50, 400, 545, 520), rows=4, cols=2,
                    cell_texts=[["A", "B"], ["1", "2"], ["3", "4"], ["5", "6"]]),
        # tblt1 -> same bbox as tbl1, but a column was dropped => STRUCTURE_CHANGED
        TableObject(id="tblt1", page=0, bbox=(50, 600, 545, 650), rows=2, cols=2,
                    cell_texts=[["A", "B"], ["1", "2"]]),
    ]
    doc = DocumentObject(path="target.pdf", language="French", pages=[page])
    return doc


def main():
    config = QAConfig()
    source_doc = make_source_doc()
    target_doc = make_target_doc()

    analyze_document_layout(source_doc, config)
    analyze_document_layout(target_doc, config)

    differences = build_differences(source_doc, target_doc, config)
    summary = summarize(differences)

    print("=== SUMMARY ===")
    print(summary)
    print()
    print("=== DIFFERENCES ===")
    for r in differences:
        print(f"page={r.page} type={r.object_type} status={r.status} sev={r.severity} conf={r.confidence} rec={r.recommendation}")

    assert summary["total_issues"] > 0, "Expected at least one difference to be detected"
    statuses = {r.status for r in differences}
    assert "MISSING" in statuses, "Expected a MISSING paragraph to be detected"
    assert "EXTRA" in statuses, "Expected an EXTRA paragraph to be detected"
    assert "OVERFLOW" in statuses, "Expected an OVERFLOW paragraph to be detected"
    assert "FORMATTING_CHANGED" in statuses, "Expected a bold/italic change to be detected"
    assert "BULLET_MISMATCH" in statuses, "Expected a bullet-count mismatch to be detected"
    assert "STRUCTURE_CHANGED" in statuses, "Expected a table row/column count change to be detected"

    prompt = build_prompt(differences, "source.pdf", "target.pdf", "English", "French", target_doc.page_count)
    assert "differences" in prompt["user"]

    report_text = build_fallback_report_text(differences, "source.pdf", "target.pdf", "English", "French", target_doc.page_count)
    assert "## Overall Status" in report_text
    assert "## Page-wise Issues" in report_text

    files = export_all(report_text, differences)
    assert "txt" in files
    print()
    print("=== EXPORTED FORMATS ===", list(files.keys()))
    for fmt, data in files.items():
        out_path = os.path.join(os.path.dirname(__file__), f"sample_report.{fmt}")
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Wrote {out_path} ({len(data)} bytes)")

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
