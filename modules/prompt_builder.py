"""
prompt_builder.py
-------------------
Builds the prompt sent to the LLM for report WRITING only. The LLM never
sees the PDFs and never detects anything -- it receives the already-final
structured JSON differences and turns them into prose. This module's job
is to make that boundary explicit and impossible to cross accidentally.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .difference_builder import DifferenceRecord, summarize

# Upper bound on how many individual differences are serialized into the LLM
# prompt. On a very large or very messy document (hundreds of pages, many
# thousand differences) sending every single one would make the prompt slow,
# expensive, and more likely to hit an API-side size/rate limit -- for a
# WRITING step, that's a bad trade against reliability. This caps the JSON
# payload while keeping every finding elsewhere (the deterministic report,
# issue browser, and annotated PDF downloads always include the full,
# untruncated list regardless of this limit).
_MAX_PROMPT_RECORDS = 400

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "none": 3}

SYSTEM_INSTRUCTIONS = """ROLE: You are a senior DTP (desktop publishing) QA engineer writing a report for \
graphic designers. You will be given a JSON list of layout differences that were already detected by a \
deterministic geometry/matching pipeline -- you did not detect them and must not invent, remove, merge, \
downgrade, or upgrade any issue. Your only job is to WRITE the report clearly and professionally.

Rules:
- Do NOT comment on translation quality, grammar, spelling, wording, or meaning. This is a layout-only QA report.
- Do NOT invent issues that are not present in the JSON.
- Use the exact page numbers given, but translate object types and status codes into plain English
  sentences rather than printing raw codes -- e.g. write "the paragraph shifted down and to the right"
  instead of "MOVED", and "the text box is overflowing and cutting off content" instead of "OVERFLOW".
  A reader with no DTP background (e.g. a project manager or client) should be able to understand every
  line without needing to look anything up.
- Group issues by page in the "Page-wise Issues" section, in ascending page order.
- Keep tone professional, concise, and actionable -- plain language, short sentences, no unexplained jargon.
- Follow the exact section structure requested.
"""

REPORT_TEMPLATE_INSTRUCTIONS = """Produce the report with exactly these sections, in this order:

1. Document Information
2. Overall Status
3. QA Summary
4. Page-wise Issues
5. Designer Action Items
6. Final Remark

Use Markdown headings (##) for each section."""


def build_prompt(
    records: List[DifferenceRecord],
    source_name: str,
    target_name: str,
    source_lang: str,
    target_lang: str,
    page_count: int,
) -> Dict[str, str]:
    """Returns {"system": ..., "user": ...} ready to send to an LLM chat API.

    `summary` always reflects the FULL, untruncated difference list -- only
    the per-issue `differences` array sent for prose-writing is capped (see
    `_MAX_PROMPT_RECORDS`), so the "QA Summary" section the LLM writes still
    reports accurate totals even on a document large enough to be truncated.
    """
    summary = summarize(records)

    prompt_records = records
    truncated_count = 0
    if len(records) > _MAX_PROMPT_RECORDS:
        # Keep the most important issues: highest severity first, then page
        # order, so the LLM's narrative covers what a designer would actually
        # triage first rather than an arbitrary prefix of the list.
        ranked = sorted(records, key=lambda r: (_SEVERITY_RANK.get(r.severity, 9), r.page))
        prompt_records = ranked[:_MAX_PROMPT_RECORDS]
        truncated_count = len(records) - _MAX_PROMPT_RECORDS

    payload: Dict[str, Any] = {
        "document_info": {
            "source_file": source_name,
            "target_file": target_name,
            "source_language": source_lang,
            "target_language": target_lang,
            "page_count": page_count,
        },
        "summary": summary,
        "differences": [r.to_dict() for r in prompt_records],
    }
    if truncated_count:
        payload["note"] = (
            f"Only the {_MAX_PROMPT_RECORDS} highest-severity differences (of "
            f"{len(records)} total, per `summary` above) are listed individually below. "
            f"{truncated_count} additional lower-priority differences exist but are omitted "
            "here for brevity -- do not claim the document has only "
            f"{_MAX_PROMPT_RECORDS} issues; use the accurate totals in `summary`, and mention "
            "in the report that the full itemized list is available in the Issue Browser and "
            "downloadable report."
        )

    user_prompt = (
        REPORT_TEMPLATE_INSTRUCTIONS
        + "\n\nStructured QA data (JSON):\n"
        + json.dumps(payload, indent=2, default=str)
    )
    return {"system": SYSTEM_INSTRUCTIONS, "user": user_prompt}
