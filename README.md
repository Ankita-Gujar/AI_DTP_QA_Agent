# AI DTP QA Agent

A layout/publishing QA tool for translated documents. It compares a **source PDF**
and a **target (translated) PDF** and produces a professional DTP QA report for
designers -- it checks *layout*, not translation, grammar, or spelling.

## What it checks

Missing paragraphs/images/tables, extra paragraphs/images/tables, text
overflow/underflow, object movement, alignment, margin changes, page flow,
paragraph order/split/merge, object size/rotation changes, cropped/missing
graphics, header/footer and page-number consistency, white space changes,
table split/merge/overflow, table row/column count mismatches (a row or
column dropped, merged, or added), caption alignment, bold/italic emphasis
changes, missing/extra bullet or numbered list items within a text block,
and (optionally) color changes.

### Bullets, bold/italic, and table row/column checks in detail

* **Bullet/numbered list items** -- each detected text block is scanned line
  by line for bullet glyphs (•, ‣, ▪, -, *, ...) or numbered/lettered/roman
  markers ("1.", "1)", "a.", "iv)"). If the number of list-item lines in a
  matched source/target block differs, it's flagged `BULLET_MISMATCH` (a
  bullet was dropped or an extra one was added). A whole bullet that was
  extracted as its own separate text block (rather than a line inside a
  larger block) is instead caught by the normal paragraph `MISSING`/`EXTRA`
  check.
* **Bold/italic** -- each text block's dominant emphasis is read from the
  PDF's font-descriptor flags, with a font-name fallback (e.g. `Arial-Bold`)
  for PDFs that don't set the flag bits reliably. A matched block whose
  bold/italic emphasis differs between source and target is flagged
  `FORMATTING_CHANGED`.
* **Table rows/columns** -- every detected table's row and column count is
  compared between source and target. A count mismatch is flagged
  `STRUCTURE_CHANGED` with the exact counts on each side (e.g. "3 rows in
  source vs 2 in target"), distinct from a plain `RESIZED` (same row/column
  count, just a different cell size) or `OVERFLOW` (content spilling past
  the table frame).

## What it deliberately ignores

Translation quality, grammar, spelling, meaning, font family, and small font
size differences.

## Architecture

```
app.py                     Streamlit UI
modules/
  utils.py                 logging, QAConfig, geometry helpers
  document_object.py        DOM dataclasses (Paragraph/Image/Table/Page/Document) + ObjectStatus enum
  pdf_reader.py             PyMuPDF extraction -> raw blocks/images/tables
  paragraph_detector.py     raw blocks -> ParagraphObject (merge/split, reading order, columns)
  image_detector.py         raw images -> ImageObject (perceptual hash, crop heuristic)
  table_detector.py         raw tables -> TableObject
  layout_analyzer.py        margins, header/footer/page-number, pairwise layout math
  ai_matcher.py             multilingual embeddings for cross-lingual semantic similarity
  paragraph_matcher.py      Hungarian-algorithm optimal assignment (40/30/15/15 weighting)
  image_matcher.py          perceptual-hash + geometry matching
  table_matcher.py          geometry matching + split/merge detection
  difference_builder.py     consolidates all matches -> structured DifferenceRecord list
  prompt_builder.py         builds the LLM prompt (writing only, never detection)
  report_builder.py         deterministic fallback text + TXT/DOCX/PDF export
  ai_agent.py                orchestrator (the pipeline entry point)
  page_renderer.py           side-by-side page rendering with color-coded highlights
```

## Install

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Semantic matching uses `sentence-transformers` (multilingual MiniLM by
default). If it isn't installed, the pipeline still runs -- semantic
similarity falls back to a neutral score and matching leans more on layout
signals, which is safer than doing literal (non-semantic) text comparison
across languages. It's the heaviest dependency in `requirements.txt` (pulls
in `torch`); if you're only trialing the tool or hit install trouble with it
on your platform, you can remove that one line and everything else still
works.

To enable AI-written prose for the final report (instead of the built-in
deterministic template), get a free key at
[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and
set it:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

or as a plain environment variable:

```bash
export GEMINI_API_KEY=AI...
```

The report structure and section list are identical either way -- only the
sentence-level wording differs. The LLM is never used for detection; it only
rewrites the JSON differences the pipeline already produced, and on any
failure (missing key, network error, rate limit, timeout) the app silently
falls back to the deterministic template rather than breaking the run. See
`.env.example` for every environment variable the app reads (report model
choice, page-count limit, log level).

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (defaults to http://localhost:8501).

## Verify the install

Before relying on this for real documents, run the bundled synthetic
pipeline test -- it exercises detection, matching, classification, and all
three report export formats without needing real PDFs or PyMuPDF, so it's a
fast way to confirm the install and any config changes didn't break
anything:

```bash
python tests/test_pipeline_synthetic.py
```

It prints a difference summary and ends with `ALL ASSERTIONS PASSED` on
success. It also writes `tests/sample_report.{txt,docx,pdf}` so you can
sanity-check the export formats directly -- those files are regenerated
each run and are git-ignored.

## Matching approach (why not simple diffing)

Paragraph matching is **not** done via raw x/y/width/height comparison or
`difflib.SequenceMatcher`, because:

* Translated text changes length (German → Chinese can shrink 50%+), so
  literal text comparison is meaningless and layout-only geometry comparison
  produces heavy false positives whenever a paragraph reflows.
* Instead, each candidate pair gets a blended score:
  `0.40 * semantic_similarity + 0.30 * layout_similarity + 0.15 * size_similarity + 0.15 * reading_order_similarity`
* The optimal assignment across all candidates is solved with the Hungarian
  algorithm (`scipy.optimize.linear_sum_assignment`), not greedy nearest-match,
  so one strong match can't "steal" a paragraph another needs more.
* Once matched, layout deltas (movement, alignment, height ratio, page index)
  classify *what* changed -- MOVED / RESIZED / OVERFLOW / UNDERFLOW /
  PAGE_MOVED / ALIGNMENT_CHANGED / MARGIN_CHANGED / MISSING / EXTRA.

Images match on perceptual hash (content) blended with position/size
(layout); CLIP embeddings are a documented optional upgrade for images that
have been re-compressed/re-exported. Tables match the same way paragraphs
do -- content first: extracted cell text is compared with the same
cross-lingual semantic matcher (50% weight), blended with position (30%)
and size (20%), so a table that simply drifted down the page (e.g. because
translated text above it got longer) still matches correctly instead of
being reported as a false missing+extra pair. Position/size alone is not
enough to *find* a table, only to describe what changed about it once
found. Row/column count mismatches are then flagged distinctly
(`STRUCTURE_CHANGED`) from a plain size change (`RESIZED`). Table detection
itself falls back to PyMuPDF's text-alignment strategy when the default
ruling-line strategy finds nothing, so borderless/whitespace-aligned tables
(common in DTP exports) are still detected as tables instead of silently
falling through to be read as ordinary paragraphs.

## Annotated PDF & visual review

Every QA run stamps the target PDF with real, color-coded PDF annotations
(not baked-in pixels) -- a box per issue plus a clickable comment with
Issue Type / Expected / Found / Recommendation, opens natively in Acrobat,
Preview, or any browser. It's available as a direct download (Downloads
tab) and as an embedded, natively zoomable viewer in the Visual Comparison
tab -- since it's a real PDF embedded via the browser's own PDF viewer,
standard zoom/scroll/pinch controls work without any custom UI. A
page-by-page rasterized overlay (with its own DPI-based zoom control) is
also available underneath as a fallback/quick-glance view.

## Performance

* Pages are streamed one at a time from `PDFReader` rather than loading a
  whole document into memory, so multi-hundred-page files don't spike RAM.
* Paragraph matching restricts candidate windows to a neighborhood in
  reading order (configurable), so very long documents don't require an
  all-pairs O(n²) embedding comparison across the entire book.
* `QAConfig.enable_parallel` / `max_workers` are exposed for sharding page
  ranges across processes for 1000+ page jobs; the reference implementation
  here processes sequentially per reader (PyMuPDF page objects aren't safely
  shareable across threads) -- see the note in `ai_agent.py` for how to
  extend this to true multiprocessing by giving each worker its own
  `PDFReader` over a page-range shard.

## Operational limits & safety rails

* **Page limit.** A single run refuses documents over `DTP_QA_MAX_PAGES`
  (default 500) with a clear error rather than tying up the app for minutes
  on an unexpectedly huge upload -- split very long documents first, or
  raise the limit via the environment variable. Reading itself is streamed
  page-by-page and doesn't spike memory, so this is a responsiveness rail,
  not a hard technical ceiling.
* **Invalid/corrupted files.** Uploads are opened and page-checked before
  the full pipeline runs; a corrupted, password-protected, or non-PDF file
  produces a clear in-app error instead of a stack trace.
* **Upload size.** Governed by `.streamlit/config.toml`'s
  `[server] maxUploadSize` (300 MB by default) -- raise it there for larger
  source files.
* **Temp files.** Uploaded PDFs are written to the OS temp directory for
  processing and removed automatically when the next run starts or the app
  process exits -- nothing accumulates on disk across runs.
* **LLM step.** Report-writing calls Gemini with a 60s timeout and a
  bounded prompt (see `prompt_builder._MAX_PROMPT_RECORDS`); any failure
  there falls back to the deterministic template and never fails the run.

## Deployment

The app is a standard Streamlit app -- any environment that can run
`streamlit run app.py` works. Two common paths:

**Docker** (see `Dockerfile` in this repo):

```bash
docker build -t dtp-qa-agent .
docker run -p 8501:8501 --env-file .env dtp-qa-agent
```

**Bare metal / VM**, behind a reverse proxy (nginx/Caddy) for TLS:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Run it under a process supervisor (systemd, supervisord, or the Docker
container's own restart policy) so it comes back up after a crash or
server reboot -- Streamlit itself doesn't daemonize or auto-restart.

For multiple concurrent users, note that Streamlit serves each browser
session as its own set of reruns within one process (not one process per
user) -- the semantic-embedding model is loaded once and shared (see
`ai_matcher.SemanticMatcher`), but two QA runs started at the same moment
will still run one after another, not in parallel, within a single
`streamlit run` process. For heavier concurrent load, run multiple
container/process replicas behind a load balancer rather than relying on
one process to serve many simultaneous QA runs.

## Verification & known constraints

This build was developed and reviewed in a sandboxed environment without
package-index or internet access, so while every module has been checked
for syntax correctness (`python -m py_compile`) and the core
detection/matching/report pipeline has been exercised end-to-end via
`tests/test_pipeline_synthetic.py`, it has **not** been run against real
PDFs with the full dependency set (PyMuPDF, sentence-transformers, Gemini)
installed. Before relying on this for a production localization pipeline:

1. Run `python tests/test_pipeline_synthetic.py` after installing
   dependencies to confirm the install itself is healthy.
2. Run a QA pass against a couple of real source/target PDF pairs typical
   of your own documents.
3. Treat the thresholds in `modules/utils.QAConfig` (movement tolerance,
   resize tolerance, overflow ratios, etc.) as a starting point -- tune
   them against your own documents' typical layout tolerance using the
   "Advanced matching thresholds" panel in the sidebar.

All optional heavy dependencies (PyMuPDF, sentence-transformers, imagehash,
python-docx, reportlab, google-generativeai) degrade gracefully with a
logged warning rather than crashing if missing -- only PyMuPDF is required
for real PDF reading; everything else narrows functionality (no semantic
matching, no annotated-PDF export, no AI-written prose, etc.) without
breaking the app.
