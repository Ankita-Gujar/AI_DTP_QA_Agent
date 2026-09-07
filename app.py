from __future__ import annotations

import atexit
import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file (if present) before anything else
# reads os.environ -- must happen before modules.ai_agent is imported, since
# it reads GEMINI_API_KEY at call time, not at import time, but keeping this
# first keeps the "config is loaded before app code runs" contract obvious.
load_dotenv()

import streamlit as st

from modules.ai_agent import DTPQAAgent
from modules.annotation_builder import build_annotated_pdf, is_available as is_annotation_available
from modules.difference_builder import summarize
from modules.pdf_reader import is_pymupdf_available
from modules.report_builder import build_fallback_report_text, export_all
from modules.utils import QAConfig, get_logger

logger = get_logger(__name__)

# Streamlit Cloud / hosted deployments configure secrets via st.secrets
# (secrets.toml), not a .env file or OS environment variable. Mirror any
# GEMINI_API_KEY found there into os.environ so modules/ai_agent.py (which
# only ever reads os.environ, deliberately, to stay UI-framework agnostic)
# picks it up transparently regardless of how the app was deployed.
if not os.environ.get("GEMINI_API_KEY"):
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None
    if secret_key:
        os.environ["GEMINI_API_KEY"] = secret_key

# Maximum pages we'll attempt to fully process in one run. This is a safety
# rail, not a hard technical limit -- PDFReader streams pages one at a time
# so memory isn't the constraint, but a many-thousand-page document run
# through the Streamlit request/response cycle can exceed browser/session
# timeouts. Override with the DTP_QA_MAX_PAGES environment variable.
MAX_PAGES = int(os.environ.get("DTP_QA_MAX_PAGES", "500"))

LANGUAGES = [
    "English", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch",
    "Russian", "Polish", "Turkish", "Arabic", "Hebrew", "Hindi", "Chinese (Simplified)",
    "Chinese (Traditional)", "Japanese", "Korean", "Thai", "Vietnamese", "Indonesian", "Other",
]

STATUS_META = {
    "PASS": {"badge": "\u2713", "label": "Pass", "cls": "pass"},
    "REVIEW REQUIRED": {"badge": "!", "label": "Review required", "cls": "warn"},
    "FAIL": {"badge": "\u2715", "label": "Fail", "cls": "fail"},
}


# Exact status -> color mapping used by modules/page_renderer.py for the
# annotated PDF overlay -- mirrored here so the in-app legend always matches
# what a reviewer actually sees on the highlighted pages.
LEGEND = [
    ("#C0392B", "Missing / extra"),
    ("#2980B9", "Moved"),
    ("#D35400", "Resized / table rows-cols"),
    ("#8E44AD", "Overflow / underflow"),
    ("#C0187A", "Color / bold-italic changed"),
    ("#27AE60", "Bullets / numbering"),
]

# Maps each legend / checkbox label to the underlying DifferenceRecord.status
# values it covers, so the "criteria to check" checkboxes can filter both the
# page-by-page overlay and the annotated PDF -- mirrors the exact color
# grouping in modules/page_renderer.py's STATUS_COLOR table.
CATEGORY_STATUS_MAP = {
    "Missing / extra": {"MISSING", "EXTRA", "CROPPED"},
    "Moved": {"MOVED", "PAGE_MOVED"},
    "Resized / table rows-cols": {"RESIZED", "ALIGNMENT_CHANGED", "MARGIN_CHANGED", "ROTATED", "STRUCTURE_CHANGED"},
    "Overflow / underflow": {"OVERFLOW", "UNDERFLOW", "SPLIT", "MERGED"},
    "Color / bold-italic changed": {"COLOR_CHANGED", "FORMATTING_CHANGED"},
    "Bullets / numbering": {"BULLET_MISMATCH"},
}

# --------------------------------------------------------------------------
# Registration mark -- the app's signature motif, borrowed straight from
# the print shop: the crosshair-in-circle a printer uses to check that
# color plates are aligned. Used as the logomark, as corner "crop marks"
# bracketing panels, and (spinning) as the busy/processing indicator.
# --------------------------------------------------------------------------
def _regmark_svg(extra_class: str = "") -> str:
    return (
        f'<svg class="regmark {extra_class}" width="30" height="30" viewBox="0 0 34 34" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<circle cx="17" cy="17" r="12" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        '<circle cx="17" cy="17" r="3.2" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        '<line x1="17" y1="0" x2="17" y2="8.3" stroke="currentColor" stroke-width="1.5"/>'
        '<line x1="17" y1="25.7" x2="17" y2="34" stroke="currentColor" stroke-width="1.5"/>'
        '<line x1="0" y1="17" x2="8.3" y2="17" stroke="currentColor" stroke-width="1.5"/>'
        '<line x1="25.7" y1="17" x2="34" y2="17" stroke="currentColor" stroke-width="1.5"/>'
        "</svg>"
    )


# --------------------------------------------------------------------------
# Design system -- one CSS block, everything else in the app only ever
# assigns these classes/tokens. Keeps the look consistent instead of one-off
# inline styles scattered through the render functions.
# --------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #101828;
    --ink-soft: #45526B;
    --ink-faint: #8592AC;
    --paper: #F5F8FE;
    --paper-alt: #E8F0FC;
    --line: #D6E2F5;
    --accent: #2451B7;
    --accent-dark: #163B8F;
    --accent-light: #4C7BE0;
    --register: #1D4FA6;
    --spark: #F0A63C;
    --danger: #C31F35;
    --pass: #1B7A4B;
    --pass-bg: #E9F6EF;
    --pass-line: #BCE3CC;
    --warn: #B4650A;
    --warn-bg: #FBF1E3;
    --warn-line: #F0D6A8;
    --fail-bg: #FBEAEA;
    --fail-line: #F0BFC2;
    --font-display: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
    --font-body: 'IBM Plex Sans', -apple-system, sans-serif;
    --font-mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
}

html, body, [class*="css"] { font-family: var(--font-body); color: var(--ink); }
.stApp {
    background:
        radial-gradient(1100px 460px at 100% -8%, #D7E6FC 0%, transparent 60%),
        radial-gradient(900px 420px at -8% 18%, #E3EDFC 0%, transparent 55%),
        radial-gradient(800px 500px at 50% 115%, #E7F0FD 0%, transparent 60%),
        var(--paper);
}

/* Drop Streamlit's default chrome: menu / footer badge / decoration bar.
   toolbarMode="minimal" in config.toml already removes the Deploy button
   and hamburger menu -- this just cleans up what's left. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
div.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: var(--font-display); color: var(--ink); }
a { color: var(--register); }
::selection { background: #CFE0FB; color: var(--ink); }

@media (prefers-reduced-motion: no-preference) {
    .regmark.spin-idle { animation: regspin 22s linear infinite; }
    .regmark.spin-active { animation: regspin 1.1s linear infinite; }
}
@keyframes regspin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

/* ---------------------------------------------------------------- Hero */
.app-hero {
    position: relative;
    background:
        radial-gradient(120% 160% at 100% 0%, #3E74D6 0%, transparent 55%),
        linear-gradient(135deg, #0C2E6E 0%, #1A4CAE 60%, #2E68D6 100%);
    border-radius: 4px;
    padding: 30px 34px 22px 34px;
    margin-bottom: 30px;
    box-shadow: 0 12px 28px rgba(20, 60, 130, 0.22);
    overflow: hidden;
}
.app-hero .crop { position: absolute; width: 15px; height: 15px; opacity: 0.55; }
.app-hero .crop.tl { top: 10px; left: 10px; border-top: 1.5px solid #EAF1FF; border-left: 1.5px solid #EAF1FF; }
.app-hero .crop.tr { top: 10px; right: 10px; border-top: 1.5px solid #EAF1FF; border-right: 1.5px solid #EAF1FF; }
.app-hero .crop.bl { bottom: 10px; left: 10px; border-bottom: 1.5px solid #EAF1FF; border-left: 1.5px solid #EAF1FF; }
.app-hero .crop.br { bottom: 10px; right: 10px; border-bottom: 1.5px solid #EAF1FF; border-right: 1.5px solid #EAF1FF; }
.app-hero .hero-row { display: flex; align-items: center; gap: 14px; }
.app-hero .regmark { color: #F0B25F; flex-shrink: 0; }
.app-hero .eyebrow {
    color: #BBD2F7; font-family: var(--font-mono); font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.16em; text-transform: uppercase; margin: 0 0 5px 0;
}
.app-hero .title {
    color: #FAFCFF; font-family: var(--font-display); font-size: 2.05rem; font-weight: 700;
    margin: 0; line-height: 1.15; letter-spacing: -0.01em;
}
.app-hero .subtitle { color: #CBDBF7; font-size: 0.95rem; margin: 10px 0 0 0; max-width: 700px; line-height: 1.55; }
.app-hero .ruler {
    margin-top: 22px; height: 8px; opacity: 0.45;
    background-image: repeating-linear-gradient(90deg, #DCE9FC 0, #DCE9FC 1px, transparent 1px, transparent 10px);
    background-position: bottom; background-size: 100% 8px; background-repeat: repeat-x;
}

/* --------------------------------------------------------- Section labels */
.section-label {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-mono); font-size: 0.76rem; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-soft); margin: 6px 0 14px 0;
}
.section-label .idx {
    display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
    border-radius: 3px; background: var(--accent); color: #fff; font-size: 0.68rem; font-weight: 600;
}
.section-label .rule { flex: 1; height: 1px; background: var(--line); }

/* ------------------------------------------------------------ Process strip */
.process-strip { display: flex; align-items: stretch; gap: 0; margin: 6px 0 26px 0; }
.process-node {
    flex: 1; background: #FFFFFF; border: 1px solid var(--line); border-radius: 4px;
    padding: 16px 18px; position: relative;
}
.process-node .pn-idx { font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent); font-weight: 600; letter-spacing: 0.05em; }
.process-node .pn-title { font-family: var(--font-display); font-size: 1.02rem; font-weight: 700; color: var(--ink); margin: 4px 0 4px 0; }
.process-node .pn-desc { font-size: 0.83rem; color: var(--ink-soft); line-height: 1.45; }
.process-link {
    width: 28px; flex-shrink: 0; align-self: center; height: 1px;
    background-image: repeating-linear-gradient(90deg, var(--ink-faint) 0, var(--ink-faint) 4px, transparent 4px, transparent 8px);
}

/* ------------------------------------------------------------ Ticket labels */
.ticket-label {
    font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--ink); margin: 0 0 6px 2px; display: flex; align-items: center; gap: 7px;
}
.ticket-label .tk-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: inline-block; }
.ticket-label span.tk-sub { color: var(--ink-faint); font-weight: 400; text-transform: none; letter-spacing: 0; }

[data-testid="stFileUploaderDropzone"] {
    border-radius: 4px; border: 1.5px dashed #C7C3B6 !important; background: #FFFFFF !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; background: #F2F6FE !important; }

/* -------------------------------------------------------------- Status pill */
.status-row { display: flex; align-items: center; gap: 16px; margin: 4px 0 24px 0; }
.status-title { font-family: var(--font-display); font-size: 1.35rem; font-weight: 700; color: var(--ink); margin: 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 9px; padding: 7px 18px 7px 12px;
    border-radius: 999px; font-weight: 600; font-size: 0.95rem; border: 1px solid; font-family: var(--font-mono);
    letter-spacing: 0.02em;
}
.status-pill.pass { background: var(--pass-bg); color: var(--pass); border-color: var(--pass-line); }
.status-pill.warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-line); }
.status-pill.fail { background: var(--fail-bg); color: var(--danger); border-color: var(--fail-line); }
.status-pill .dot {
    width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.72rem; font-weight: 700; color: #fff; background: currentColor;
}
.status-pill .dot span { color: #fff; }
@media (prefers-reduced-motion: no-preference) {
    .status-pill.fail { animation: pillring 2.4s ease-in-out infinite; }
}
@keyframes pillring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(195, 31, 53, 0.0); }
    50% { box-shadow: 0 0 0 5px rgba(195, 31, 53, 0.10); }
}

/* ------------------------------------------------------------------ KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 26px; }
.kpi-card {
    background: #FFFFFF; border: 1px solid var(--line); border-top: 3px solid var(--accent, var(--register));
    border-radius: 4px; padding: 16px 18px; transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(20, 23, 28, 0.08); }
.kpi-label {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-soft); font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.kpi-value { font-family: var(--font-mono); font-size: 1.95rem; font-weight: 600; color: var(--ink); line-height: 1.35; font-variant-numeric: tabular-nums; }

/* ---------------------------------------------------------------- Legend chips */
.legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 18px 0; }
.legend-chip {
    display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 999px;
    background: #FFFFFF; border: 1px solid var(--line); font-size: 0.78rem; color: var(--ink-soft);
    font-family: var(--font-mono);
}
.legend-chip .lc-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

/* -------------------------------------------------------------------- Sidebar */
section[data-testid="stSidebar"] { background: var(--paper-alt); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
section[data-testid="stSidebar"] .section-label .rule { background: #C7D8F2; }

/* ----------------------------------------------------------------- Controls */
.stButton > button, .stDownloadButton > button {
    border-radius: 3px; font-weight: 600; padding: 0.55rem 1.2rem; font-family: var(--font-body);
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease, transform 0.1s ease;
    border: 1.5px solid var(--ink);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }
button[kind="primary"] {
    background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important;
}
button[kind="primary"]:hover { background: var(--accent-dark) !important; border-color: var(--accent-dark) !important; }
button[kind="secondary"]:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

input[type="range"], input[type="checkbox"] { accent-color: var(--accent); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono); font-size: 0.82rem; color: var(--ink-soft); padding: 10px 16px;
    letter-spacing: 0.01em; transition: color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 4px !important; background: #FFFFFF; }

[data-testid="stProgress"] div[role="progressbar"] > div { background-color: var(--accent) !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }

hr { border-color: var(--line); }
</style>
"""


def _init_state():
    if "result" not in st.session_state:
        st.session_state.result = None
    if "source_tmp" not in st.session_state:
        st.session_state.source_tmp = None
    if "target_tmp" not in st.session_state:
        st.session_state.target_tmp = None


def _cleanup_tmp_path(path: Optional[str]) -> None:
    """Best-effort removal of a temp file. Never raises -- a leftover temp
    file is a disk-space nuisance, not a reason to crash the app."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup_session_tmp_files() -> None:
    """Registered with atexit as a last-resort sweep for the current
    session's temp uploads if the process is torn down mid-run (e.g. the
    server restarts) without a normal rerun ever reaching the cleanup below."""
    _cleanup_tmp_path(st.session_state.get("source_tmp"))
    _cleanup_tmp_path(st.session_state.get("target_tmp"))


atexit.register(_cleanup_session_tmp_files)


def _save_upload(uploaded_file) -> str:
    """Persist an uploaded file to a temp path on disk so PyMuPDF can open it
    by path. Every temp file created this way is removed either when the next
    QA run starts (see the click handler below) or at process exit -- without
    that, a long-running server accumulates one orphaned file per upload and
    slowly fills its disk, which is exactly the kind of failure that only
    shows up after real production use, not in a quick local test."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="dtp_qa_")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _pdf_page_count(path: str) -> Optional[int]:
    """Best-effort page count for a validation pre-check, independent of the
    main PDFReader pipeline. Returns None if the file can't be opened as a
    PDF at all (corrupted, password-protected, or not actually a PDF)."""
    if not is_pymupdf_available():
        return None
    import fitz  # local import: only needed for this one check

    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception as exc:
        logger.warning("Failed to open '%s' as a PDF: %s", path, exc)
        return None


def _kpi_card(label: str, value, accent: str) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    )


def _section_label(idx: str, text: str) -> str:
    return (
        f'<p class="section-label"><span class="idx">{idx}</span>{text}<span class="rule"></span></p>'
    )


def _section_label_plain(text: str) -> str:
    return f'<p class="section-label">{text}<span class="rule"></span></p>'


def _ticket_label(title: str, sub: str) -> str:
    return f'<p class="ticket-label"><span class="tk-dot"></span>{title}<span class="tk-sub">&nbsp;&middot; {sub}</span></p>'


def _legend_html() -> str:
    chips = "".join(
        f'<span class="legend-chip"><span class="lc-dot" style="background:{color};"></span>{label}</span>'
        for color, label in LEGEND
    )
    return f'<div class="legend-row">{chips}</div>'


def _process_strip_html() -> str:
    steps = [
        ("01", "Upload", "Drop in the approved source PDF and the translated target PDF."),
        ("02", "Analyze", "The agent reads, matches and compares every paragraph, image and table."),
        ("03", "Review", "Get an annotated PDF, a page-wise issue browser and a downloadable report."),
    ]
    nodes = []
    for i, (idx, title, desc) in enumerate(steps):
        nodes.append(
            f'<div class="process-node"><div class="pn-idx">{idx}</div>'
            f'<div class="pn-title">{title}</div><div class="pn-desc">{desc}</div></div>'
        )
        if i < len(steps) - 1:
            nodes.append('<div class="process-link"></div>')
    return f'<div class="process-strip">{"".join(nodes)}</div>'


def main():
    st.set_page_config(page_title="AI DTP QA Agent", page_icon="\U0001F3AF", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="app-hero">'
        '<span class="crop tl"></span><span class="crop tr"></span>'
        '<span class="crop bl"></span><span class="crop br"></span>'
        '<div class="hero-row">'
        + _regmark_svg("spin-idle")
        + '<div><p class="eyebrow">Localization &amp; Publishing QA</p>'
        '<p class="title">AI DTP QA Agent</p></div>'
        "</div>"
        '<p class="subtitle">Compares a source and translated PDF for layout, formatting and '
        "positioning issues only -- not a translation or grammar checker -- and returns an "
        "annotated PDF a designer can act on directly.</p>"
        '<div class="ruler"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    _init_state()

    if not is_pymupdf_available():
        st.warning(
            "PyMuPDF (`pymupdf`) is not installed in this environment, so real PDF reading is "
            "disabled. Install dependencies with `pip install -r requirements.txt` and restart."
        )

    # ----------------------------------------------------------------
    # Sidebar -- kept intentionally. Source/target language and the
    # matching thresholds below aren't cosmetic settings: they're passed
    # straight into QAConfig / DTPQAAgent.run() and change what the
    # pipeline actually detects (e.g. how much movement counts as
    # "moved" vs. noise). Removing this would remove the ability to tune
    # or even run a correctly-labeled QA pass.
    # ----------------------------------------------------------------
    with st.sidebar:
        st.markdown(_section_label_plain("Run configuration"), unsafe_allow_html=True)
        if os.environ.get("GEMINI_API_KEY"):
            st.success("Gemini API key detected -- report prose will be AI-written.", icon="\u2705")
        else:
            st.warning(
                "No `GEMINI_API_KEY` found -- reports use the built-in template instead of "
                "AI-written prose. Add it as a Streamlit secret or environment variable, then restart.",
                icon="\u26A0\uFE0F",
            )

        source_lang = st.selectbox("Source language", LANGUAGES, index=0)
        target_lang = st.selectbox("Target language", LANGUAGES, index=1)

        with st.expander("Advanced matching thresholds", expanded=False):
            config = QAConfig()
            config.move_tolerance_pt = st.slider("Movement tolerance (pt)", 0.0, 20.0, config.move_tolerance_pt)
            config.resize_tolerance_ratio = st.slider("Resize tolerance (ratio)", 0.0, 0.3, config.resize_tolerance_ratio)
            config.overflow_height_ratio = st.slider("Overflow height ratio", 1.0, 2.0, config.overflow_height_ratio)
            config.match_score_threshold = st.slider("Match score threshold", 0.1, 0.9, config.match_score_threshold)
            config.min_image_area_pt = st.slider(
                "Ignore images smaller than (pt\u00b2)", 0.0, 2000.0, config.min_image_area_pt,
                help="Filters out tiny inline icons/glyphs (e.g. warning-triangle symbols) so they "
                     "aren't reported as separate missing/moved images.",
            )
            config.flag_paragraph_flow_moves = st.checkbox(
                "Flag vertical text reflow as 'Moved' (not recommended for translations)",
                value=config.flag_paragraph_flow_moves,
                help="Off by default: when target text is longer/shorter than the source, everything "
                     "below it shifts up/down on the page -- that's expected, not a layout defect. "
                     "Turn this on only if source and target are meant to have identical line breaks.",
            )

    st.markdown(_section_label("1", "Upload files"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_ticket_label("Source", "approved master"), unsafe_allow_html=True)
        source_file = st.file_uploader("Source PDF", type=["pdf"], key="source_upload", label_visibility="collapsed")
    with col2:
        st.markdown(_ticket_label("Target", "translated proof"), unsafe_allow_html=True)
        target_file = st.file_uploader("Target PDF", type=["pdf"], key="target_upload", label_visibility="collapsed")

    start = st.button("Start AI QA", type="primary", disabled=not (source_file and target_file))

    if start and source_file and target_file:
        # A fresh run replaces any files left over from a previous run in this
        # session -- clean those up now rather than leaking one temp file per
        # click for the lifetime of the server process.
        _cleanup_tmp_path(st.session_state.source_tmp)
        _cleanup_tmp_path(st.session_state.target_tmp)
        st.session_state.source_tmp = _save_upload(source_file)
        st.session_state.target_tmp = _save_upload(target_file)

        # Pre-validate before running the full pipeline, so a bad upload fails
        # fast with a clear message instead of a stack trace mid-run. Skipped
        # entirely if PyMuPDF isn't installed -- the warning banner at the top
        # of the page already covers that case, and the pipeline itself will
        # raise a clear RuntimeError caught below.
        if is_pymupdf_available():
            source_pages = _pdf_page_count(st.session_state.source_tmp)
            target_pages = _pdf_page_count(st.session_state.target_tmp)
            if source_pages is None or target_pages is None:
                st.error(
                    "Could not open one of the uploaded files as a PDF. It may be "
                    "corrupted, password-protected, or not a real PDF despite the "
                    "file extension. Please re-export it and try again."
                )
                st.session_state.result = None
                return
            if source_pages == 0 or target_pages == 0:
                st.error("One of the uploaded PDFs has no pages. Please upload a non-empty PDF.")
                st.session_state.result = None
                return
            if max(source_pages, target_pages) > MAX_PAGES:
                st.error(
                    f"This document has {max(source_pages, target_pages)} pages, which exceeds "
                    f"the configured limit of {MAX_PAGES} pages for a single run "
                    f"(set the `DTP_QA_MAX_PAGES` environment variable to raise it). Very long "
                    "documents are still supported -- split the PDF into smaller sections first."
                )
                st.session_state.result = None
                return

        st.markdown(
            '<div class="hero-row" style="margin: 4px 0 10px 0;">'
            + _regmark_svg("spin-active")
            + f'<span style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--ink-soft);">'
            f"Aligning plates &mdash; comparing source and target&hellip;</span></div>",
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0.0, text="Starting...")

        def on_progress(message: str, fraction: float):
            progress_bar.progress(min(1.0, fraction), text=message)

        agent = DTPQAAgent(config=config)
        try:
            result = agent.run(
                st.session_state.source_tmp,
                st.session_state.target_tmp,
                source_lang,
                target_lang,
                progress_cb=on_progress,
            )
            st.session_state.result = result
            progress_bar.progress(1.0, text="Complete.")
        except Exception as exc:
            logger.exception("QA run failed for source=%s target=%s", source_file.name, target_file.name)
            st.error(
                f"QA run failed: {exc}\n\n"
                "If this keeps happening, check the app logs for the full traceback, "
                "and confirm both files are valid, unencrypted PDFs."
            )
            st.session_state.result = None

    result = st.session_state.result
    if result is None:
        st.markdown(_process_strip_html(), unsafe_allow_html=True)
        st.info("Upload both PDFs and press **Start AI QA** to generate a report.")
        return

    summary_unfiltered = result.summary

    st.markdown(_section_label("2", "Results"), unsafe_allow_html=True)

    st.markdown(_section_label_plain("Criteria to check"), unsafe_allow_html=True)
    st.caption(
        "Choose which issue types this QA pass should count. Unchecked criteria are "
        "excluded from the status, summary, AI report, issue browser, and visual "
        "comparison below -- handy for expected translation effects (e.g. text "
        "reflow/overflow or minor margin drift from a longer or shorter language) "
        "that aren't real layout problems."
    )
    crit_cols = st.columns(3)
    selected_categories = []
    for i, (color, label) in enumerate(LEGEND):
        with crit_cols[i % 3]:
            if st.checkbox(label, value=True, key=f"crit_{label}"):
                selected_categories.append(label)

    all_statuses = {s for m in CATEGORY_STATUS_MAP.values() for s in m}
    selected_statuses = set()
    for label in selected_categories:
        selected_statuses |= CATEGORY_STATUS_MAP.get(label, set())
    filtering_active = selected_statuses != all_statuses

    if not selected_categories:
        st.warning("No criteria selected -- check at least one box above to evaluate this document.")

    filtered_diffs = [r for r in result.differences if r.status in selected_statuses]
    summary = summarize(filtered_diffs)
    meta = STATUS_META.get(summary["overall_status"], STATUS_META["REVIEW REQUIRED"])

    # Deterministic report text re-derived from the filtered issue set -- reused by
    # both the AI Report tab and the Downloads tab so they always agree with the
    # status pill / KPI cards above, and with each other.
    filtered_report_text = None
    if filtering_active:
        filtered_report_text = build_fallback_report_text(
            filtered_diffs,
            os.path.basename(result.source_doc.path),
            os.path.basename(result.target_doc.path),
            result.source_doc.language,
            result.target_doc.language,
            result.target_doc.page_count,
        )

    st.markdown(
        '<div class="status-row">'
        '<p class="status-title">Overall status</p>'
        f'<span class="status-pill {meta["cls"]}">'
        f'<span class="dot"><span>{meta["badge"]}</span></span>{meta["label"]}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="kpi-grid">'
        + _kpi_card("Total issues", summary["total_issues"], "var(--register)")
        + _kpi_card("High severity", summary["by_severity"].get("high", 0), "var(--danger)")
        + _kpi_card("Medium severity", summary["by_severity"].get("medium", 0), "var(--warn)")
        + _kpi_card("Low severity", summary["by_severity"].get("low", 0), "var(--ink-faint)")
        + "</div>",
        unsafe_allow_html=True,
    )
    if filtering_active:
        st.caption(
            f"Filtered view: {summary['total_issues']} of {summary_unfiltered['total_issues']} "
            "total issues counted, based on the criteria checked above."
        )

    if not result.llm_used:
        st.caption(
            "Report text generated with the built-in deterministic template "
            "(no GEMINI_API_KEY configured or Gemini API unavailable)."
        )

    tab_report, tab_issues, tab_visual, tab_download = st.tabs(
        ["\U0001F4CB AI Report", "\U0001F50D Issue Browser", "\U0001F5BC\uFE0F Visual Comparison", "\u2B07\uFE0F Downloads"]
    )

    with tab_report:
        if filtering_active:
            st.caption(
                "Regenerated from your selected criteria (deterministic template) -- "
                "check all criteria above to see the original AI-written narrative."
            )
            st.markdown(filtered_report_text)
        else:
            st.markdown(result.report_text)

    with tab_issues:
        pages = sorted(set(r.page for r in filtered_diffs))
        page_filter = st.selectbox("Filter by page", ["All"] + [str(p) for p in pages])
        type_filter = st.multiselect("Filter by object type", ["paragraph", "image", "table", "page"],
                                      default=["paragraph", "image", "table", "page"])
        rows = filtered_diffs
        if page_filter != "All":
            rows = [r for r in rows if r.page == int(page_filter)]
        rows = [r for r in rows if r.object_type in type_filter]

        st.dataframe(
            [
                {
                    "Page": r.page, "Type": r.object_type, "Status": r.status,
                    "Severity": r.severity, "Confidence": round(r.confidence, 2),
                    "Movement (pt)": r.movement_pt, "Recommendation": r.recommendation,
                }
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_visual:
        st.markdown(_legend_html(), unsafe_allow_html=True)

        if not selected_categories:
            st.warning("No criteria selected -- check at least one box above to see highlights.")
        elif not filtered_diffs and result.differences:
            st.info("No issues match the selected criteria.")

        st.markdown(_section_label_plain("Page-by-page overlay"), unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            max_preview_pages = st.slider(
                "Pages to render", 1, min(20, result.target_doc.page_count) or 1, min(5, result.target_doc.page_count) or 1
            )
        with col_b:
            zoom_pct = st.select_slider("Zoom", options=[50, 75, 100, 150, 200, 300], value=100)
        if st.button("Render visual comparison"):
            from modules.page_renderer import build_comparison_gallery
            dpi_override = max(50, int(config.render_dpi * zoom_pct / 100))
            gallery = build_comparison_gallery(
                st.session_state.source_tmp, st.session_state.target_tmp,
                filtered_diffs, config, max_pages=max_preview_pages,
                dpi_override=dpi_override,
            )
            if not gallery:
                st.warning("Visual rendering requires PyMuPDF + Pillow to be installed.")
            # At 100% zoom, fit images to the column so source/target sit neatly
            # side by side; above 100%, show them at native (larger) pixel size --
            # the page scrolls, which is what makes the extra resolution actually
            # visible instead of being scaled back down to fit the column.
            fit_to_column = zoom_pct <= 100
            for entry in gallery:
                st.markdown(f"**Page {entry['page']}**")
                c1, c2 = st.columns(2)
                if entry["source_image"] is not None:
                    c1.image(entry["source_image"], caption="Source", use_container_width=fit_to_column)
                if entry["target_image"] is not None:
                    c2.image(entry["target_image"], caption="Target (highlighted)", use_container_width=fit_to_column)

        st.markdown(_section_label_plain("Annotated PDF -- native zoom & scroll"), unsafe_allow_html=True)
        if result.annotated_pdf:
            if st.button("\U0001F5BC\uFE0F Open annotated PDF viewer"):
                st.session_state.show_annotated_pdf = not st.session_state.get("show_annotated_pdf", False)

            if st.session_state.get("show_annotated_pdf", False):
                st.caption(
                    "This is the actual annotated target PDF. Use your browser's built-in PDF "
                    "controls (scroll to zoom, pinch on trackpad/touch, or the viewer's own +/- "
                    "buttons) to inspect any marking closely -- every box is a real, clickable "
                    "comment with Issue Type / Expected / Found / Recommendation."
                )
                # Rebuild the annotated PDF against the checked criteria only, so the
                # native viewer reflects the same filter as the overlay above.
                pdf_bytes = result.annotated_pdf
                if filtering_active and is_annotation_available():
                    pdf_bytes = build_annotated_pdf(st.session_state.target_tmp, filtered_diffs)
                pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="780" '
                    'style="border:1px solid var(--line); border-radius:4px;" type="application/pdf">'
                    "</iframe>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "The embedded annotated-PDF viewer needs PyMuPDF installed in this environment. "
                "The page-by-page overlay above still works."
            )

    with tab_download:
        st.write("Download the annotated PDF or the QA report:")
        download_pdf_bytes = result.annotated_pdf
        if filtering_active and result.annotated_pdf and is_annotation_available():
            download_pdf_bytes = build_annotated_pdf(st.session_state.target_tmp, filtered_diffs)
        if download_pdf_bytes:
            st.download_button(
                "\U0001F5BC\uFE0F Download Annotated Target PDF", download_pdf_bytes,
                file_name="dtp_qa_annotated.pdf", mime="application/pdf", type="primary",
            )
            st.caption("Every issue is stamped directly on the target PDF as a color-coded box with a clickable comment.")
            if filtering_active:
                st.caption("Reflects only the criteria checked above.")
        else:
            st.caption("Annotated PDF download is unavailable in this environment (requires PyMuPDF).")
        st.divider()
        st.write("QA report, in your preferred format:")
        download_report_files = result.report_files
        if filtering_active:
            download_report_files = export_all(filtered_report_text, filtered_diffs)
        if "txt" in download_report_files:
            st.download_button("Download TXT", download_report_files["txt"], file_name="dtp_qa_report.txt")
        if "docx" in download_report_files:
            st.download_button("Download DOCX", download_report_files["docx"], file_name="dtp_qa_report.docx")
        if "pdf" in download_report_files:
            st.download_button("Download PDF", download_report_files["pdf"], file_name="dtp_qa_report.pdf")


if __name__ == "__main__":
    main()
