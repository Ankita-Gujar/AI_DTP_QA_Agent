"""
ai_agent.py
------------
Top-level orchestrator. Wires together:

    pdf_reader -> paragraph/image/table detectors -> layout_analyzer
        -> difference_builder -> prompt_builder -> (LLM) -> report_builder

`DTPQAAgent.run()` is the single entry point the Streamlit UI (and any
future CLI/batch runner) calls. It reports progress via an optional
callback so the UI can drive a progress bar.

The LLM is used strictly for the WRITING step (prompt_builder's contract).
If no API key is configured, the agent falls back to the deterministic
template report from report_builder -- the tool remains fully functional
without any LLM at all, since layout detection never depended on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from dotenv import load_dotenv

from .annotation_builder import build_annotated_pdf, is_available as is_annotation_available
from .difference_builder import DifferenceRecord, build_differences, summarize
from .document_object import DocumentObject, PageObject
from .image_detector import detect_images
from .layout_analyzer import analyze_document_layout
from .paragraph_detector import detect_paragraphs
from .pdf_reader import PDFReader, is_pymupdf_available
from .prompt_builder import build_prompt
from .report_builder import build_fallback_report_text, export_all
from .table_detector import detect_tables
from .utils import QAConfig, get_logger

load_dotenv()
logger = get_logger(__name__)

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover - environment dependent
    # Deliberately broad, same reasoning as ai_matcher.py's sentence-transformers
    # import guard: this dependency is optional (see requirements.txt), and its
    # absence should degrade to the deterministic report template, never crash
    # the whole app at import time.
    genai = None
    _GENAI_AVAILABLE = False
    logger.info(
        "google-generativeai is not installed (%s) -- AI-written report prose is "
        "disabled and the deterministic template will be used instead.",
        _import_exc,
    )

ProgressCallback = Optional[Callable[[str, float], None]]


@dataclass
class QARunResult:
    source_doc: DocumentObject
    target_doc: DocumentObject
    differences: List[DifferenceRecord]
    summary: dict
    report_text: str
    report_files: dict  # {"txt": bytes, "docx": bytes, "pdf": bytes}
    llm_used: bool
    annotated_pdf: Optional[bytes] = None  # target PDF with color-coded review annotations stamped on it


def _report_progress(cb: ProgressCallback, message: str, fraction: float) -> None:
    logger.info("[%.0f%%] %s", fraction * 100, message)
    if cb is not None:
        cb(message, fraction)


def _read_document(path: str, language: str, config: QAConfig, progress_cb: ProgressCallback,
                    progress_start: float, progress_end: float) -> DocumentObject:
    if not is_pymupdf_available():
        raise RuntimeError(
            "PyMuPDF is not installed in this environment. Run `pip install pymupdf` "
            "and restart the app to enable real PDF reading."
        )

    doc = DocumentObject(path=path, language=language)
    with PDFReader(path) as reader:
        total_pages = len(reader)
        fitz_doc = reader._doc  # only used to pass into image hashing (extract_image)

        def _process(index: int) -> PageObject:
            raw = reader.read_page(index)
            page = PageObject(page_number=raw.page_number, width=raw.width, height=raw.height)
            # IMPORTANT: raw.page_number is 1-indexed (matches what the UI, the
            # renderer, and DifferenceRecord.page all expect). `index` here is the
            # 0-indexed loop counter -- passing it instead of raw.page_number was
            # the root cause of every paragraph/image/table being tagged one page
            # too early (page 2's content labeled as page 1, page 3's as page 2,
            # etc.), which is why highlights only ever appeared to land on page 1
            # and the last page never got any annotations at all.
            page.paragraphs = detect_paragraphs(raw, raw.page_number, config)
            page.images = detect_images(raw, raw.page_number, config, fitz_doc=fitz_doc)
            page.tables = detect_tables(raw, raw.page_number, config)
            page.highlight_bboxes = list(raw.highlight_bboxes)
            return page

        # NOTE on config.enable_parallel / config.max_workers: reading is
        # sequential here, not multi-threaded -- PyMuPDF page objects aren't
        # safe to share across threads, and this reader streams one page at a
        # time by design (see PDFReader's docstring) so memory stays flat
        # regardless of document length. True parallelism for very large
        # batches means sharding page ranges across separate processes, each
        # with its own PDFReader instance over its own page range; that's a
        # future extension point, not something toggled by this flag today.
        # The flag/max_workers are kept in QAConfig as the stable interface
        # for that extension rather than removed.
        pages: List[Optional[PageObject]] = [None] * total_pages
        progress_every = max(1, total_pages // 20)  # ~20 updates across the whole document,
                                                      # so a 5-page and a 5000-page file both
                                                      # get a responsive, non-spammy progress bar
        for i in range(total_pages):
            pages[i] = _process(i)
            if i % progress_every == 0 or i == total_pages - 1:
                frac = progress_start + (progress_end - progress_start) * ((i + 1) / max(1, total_pages))
                _report_progress(progress_cb, f"Reading page {i + 1}/{total_pages} of {os.path.basename(path)}", frac)

        doc.pages = [p for p in pages if p is not None]

    analyze_document_layout(doc, config)
    return doc


_GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
_GEMINI_TIMEOUT_SECONDS = 60  # bail out to the deterministic template rather than hang the UI


def _call_llm_for_report(prompt: dict) -> Optional[str]:
    """Ask Gemini to write the report prose from the already-final structured
    differences. Returns None (never raises) on any failure -- missing SDK,
    missing/invalid API key, network error, timeout, or an empty response --
    so the caller always has a safe deterministic fallback to use instead."""
    if not _GENAI_AVAILABLE:
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.debug("No GEMINI_API_KEY configured -- using the deterministic report template.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_GEMINI_MODEL_NAME)
        full_prompt = prompt["system"] + "\n\n" + prompt["user"]

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,       # low temperature: this is a report-writing task
                                        # over already-final facts, not a creative one
                max_output_tokens=4096,
            ),
            request_options={"timeout": _GEMINI_TIMEOUT_SECONDS},
        )
        text = getattr(response, "text", None)
        if not text or not text.strip():
            logger.warning("Gemini returned an empty response. Using fallback report.")
            return None
        return text
    except Exception as exc:
        # Broad on purpose: an LLM writing step failing (bad key, rate limit,
        # network blip, safety filter, SDK version mismatch, timeout) must
        # never take down an otherwise-successful QA run -- the deterministic
        # report template covers every case below with the same content.
        logger.warning("Gemini report generation failed (%s). Using fallback report.", exc)
        return None

class DTPQAAgent:

    def __init__(self, config: Optional[QAConfig] = None):

        self.config = config or QAConfig()

    def run(
        self,
        source_path: str,
        target_path: str,
        source_lang: str,
        target_lang: str,
        progress_cb: ProgressCallback = None,
    ) -> QARunResult:
        _report_progress(progress_cb, "Reading source PDF...", 0.02)
        source_doc = _read_document(source_path, source_lang, self.config, progress_cb, 0.02, 0.30)

        _report_progress(progress_cb, "Reading target PDF...", 0.32)
        target_doc = _read_document(target_path, target_lang, self.config, progress_cb, 0.32, 0.60)

        _report_progress(progress_cb, "Matching paragraphs, images and tables...", 0.62)
        differences = build_differences(source_doc, target_doc, self.config)
        summary = summarize(differences)

        _report_progress(progress_cb, "Building AI report prompt...", 0.80)
        prompt = build_prompt(
            differences,
            os.path.basename(source_path),
            os.path.basename(target_path),
            source_lang,
            target_lang,
            target_doc.page_count,
        )

        _report_progress(progress_cb, "Writing QA report...", 0.85)
        llm_text = _call_llm_for_report(prompt)
        llm_used = llm_text is not None
        report_text = llm_text or build_fallback_report_text(
            differences, os.path.basename(source_path), os.path.basename(target_path),
            source_lang, target_lang, target_doc.page_count,
        )

        _report_progress(progress_cb, "Exporting report files...", 0.95)
        report_files = export_all(report_text, differences)

        annotated_pdf: Optional[bytes] = None
        if is_annotation_available():
            try:
                annotated_pdf = build_annotated_pdf(target_path, differences)
            except Exception as exc:
                # Never let the annotated-PDF step take down an otherwise-successful
                # QA run -- the report/differences are still fully usable without it.
                logger.warning("Annotated PDF generation failed (%s); download will be unavailable.", exc)

        _report_progress(progress_cb, "Done.", 1.0)
        return QARunResult(
            source_doc=source_doc,
            target_doc=target_doc,
            differences=differences,
            summary=summary,
            report_text=report_text,
            report_files=report_files,
            llm_used=llm_used,
            annotated_pdf=annotated_pdf,
        )
