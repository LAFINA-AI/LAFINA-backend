"""The checks between a student's request, the model, and the file it becomes.

On the way in: a request the chat endpoint would refuse is refused here too.
On the way out: whatever the model described is clipped to sizes a renderer
can build quickly and a student can open, and the few places where text could
become something other than text — a spreadsheet formula, ReportLab's markup —
are closed off. Limits truncate and warn rather than fail, because a long
document cut short is still worth having.
"""

from __future__ import annotations

import re
from typing import Sequence
from xml.sax.saxutils import escape

from app.services.document_spec import DOCUMENT_FORMATS, DocumentSpec

# ── Input ─────────────────────────────────────────────────────────────────

MAX_REQUEST_MESSAGES = 10
MAX_REQUEST_CHARS = 8000


class GuardrailError(Exception):
    """A request the endpoint refuses before calling the model."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def check_request(fmt: str, messages: Sequence) -> None:
    """Holds a file request to the same allowance as a chat turn."""
    if fmt not in DOCUMENT_FORMATS:
        raise GuardrailError("Files can be PDF, Word, Excel or PowerPoint.")
    if not messages or len(messages) > MAX_REQUEST_MESSAGES:
        raise GuardrailError(f"Send between 1 and {MAX_REQUEST_MESSAGES} messages.")
    if sum(len(message.content) for message in messages) > MAX_REQUEST_CHARS:
        raise GuardrailError(
            "Total input messages character count exceeds limit of 8,000 characters."
        )
    last = messages[-1]
    if last.role != "user" or not last.content.strip():
        raise GuardrailError("Say what the file should contain.")


# ── Output limits ─────────────────────────────────────────────────────────

MAX_TITLE_CHARS = 120
MAX_SUMMARY_CHARS = 400
MAX_HEADING_CHARS = 200
MAX_PARAGRAPH_CHARS = 4000
MAX_ITEM_CHARS = 600
MAX_LIST_ITEMS = 40
MAX_BLOCKS = 120
MAX_TABLE_COLUMNS = 10
MAX_TABLE_ROWS = 200
MAX_TABLE_CELL_CHARS = 500
MAX_SHEETS = 5
MAX_SHEET_COLUMNS = 30
MAX_SHEET_ROWS = 1000
MAX_SHEET_CELL_CHARS = 1000
MAX_SHEET_NAME_CHARS = 31
MAX_SLIDES = 30
MAX_SLIDE_BULLETS = 8
MAX_SLIDE_TITLE_CHARS = 120
MAX_BULLET_CHARS = 300
MAX_NOTES_CHARS = 2000

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024

# Characters XML 1.0 cannot hold. One of these in a string makes python-docx,
# python-pptx and openpyxl refuse to write the file at all.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def clip(value: object, limit: int, *, keep_lines: bool = False) -> str:
    """Makes one string safe to write and short enough to be worth reading."""
    text = _XML_ILLEGAL.sub("", str(value if value is not None else ""))
    if keep_lines:
        lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").split("\n")]
        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _clip_cell(value: object, limit: int) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return clip(value, limit)


def _fit_row(row: list, width: int, limit: int) -> list:
    """One value per column: long rows are cut, short ones padded."""
    fitted = [_clip_cell(value, limit) for value in row[:width]]
    return fitted + [None] * (width - len(fitted))


def apply_output_limits(spec: DocumentSpec, fmt: str) -> list[str]:
    """Clips the spec in place to what the renderer builds, and says what was cut."""
    warnings: list[str] = []

    spec.title = clip(spec.title, MAX_TITLE_CHARS) or "LAFINA document"
    spec.summary = clip(spec.summary, MAX_SUMMARY_CHARS)

    if fmt in ("pdf", "docx"):
        if len(spec.blocks) > MAX_BLOCKS:
            warnings.append(f"The document was long, so it stops after {MAX_BLOCKS} sections.")
            spec.blocks = spec.blocks[:MAX_BLOCKS]
        cut_tables = False
        for block in spec.blocks:
            if block.type == "heading":
                block.text = clip(block.text, MAX_HEADING_CHARS)
                block.level = min(3, max(1, block.level))
            elif block.type == "paragraph":
                block.text = clip(block.text, MAX_PARAGRAPH_CHARS, keep_lines=True)
            elif block.type in ("bullets", "numbered"):
                block.items = [clip(item, MAX_ITEM_CHARS) for item in block.items[:MAX_LIST_ITEMS]]
                block.items = [item for item in block.items if item]
            elif block.type == "table":
                if len(block.columns) > MAX_TABLE_COLUMNS or len(block.rows) > MAX_TABLE_ROWS:
                    cut_tables = True
                block.columns = [clip(column, MAX_HEADING_CHARS) for column in block.columns]
                block.columns = block.columns[:MAX_TABLE_COLUMNS]
                width = len(block.columns) or min(
                    MAX_TABLE_COLUMNS, max((len(row) for row in block.rows), default=0)
                )
                block.rows = [
                    _fit_row(row, width, MAX_TABLE_CELL_CHARS)
                    for row in block.rows[:MAX_TABLE_ROWS]
                ]
        if cut_tables:
            warnings.append(
                f"Tables are limited to {MAX_TABLE_COLUMNS} columns and {MAX_TABLE_ROWS} rows."
            )
        spec.blocks = [block for block in spec.blocks if _block_has_content(block)]

    elif fmt == "xlsx":
        if len(spec.sheets) > MAX_SHEETS:
            warnings.append(f"Only the first {MAX_SHEETS} sheets were kept.")
            spec.sheets = spec.sheets[:MAX_SHEETS]
        cut_sheets = False
        for sheet in spec.sheets:
            if len(sheet.columns) > MAX_SHEET_COLUMNS or len(sheet.rows) > MAX_SHEET_ROWS:
                cut_sheets = True
            sheet.name = clip(sheet.name, 200)
            sheet.columns = [clip(column, MAX_HEADING_CHARS) for column in sheet.columns]
            sheet.columns = sheet.columns[:MAX_SHEET_COLUMNS]
            width = len(sheet.columns) or min(
                MAX_SHEET_COLUMNS, max((len(row) for row in sheet.rows), default=0)
            )
            sheet.rows = [
                _fit_row(row, width, MAX_SHEET_CELL_CHARS) for row in sheet.rows[:MAX_SHEET_ROWS]
            ]
        if cut_sheets:
            warnings.append(
                f"Sheets are limited to {MAX_SHEET_COLUMNS} columns and {MAX_SHEET_ROWS} rows."
            )

    elif fmt == "pptx":
        if len(spec.slides) > MAX_SLIDES:
            warnings.append(f"The deck was long, so it stops after {MAX_SLIDES} slides.")
            spec.slides = spec.slides[:MAX_SLIDES]
        for slide in spec.slides:
            slide.title = clip(slide.title, MAX_SLIDE_TITLE_CHARS)
            slide.bullets = [
                clip(bullet, MAX_BULLET_CHARS) for bullet in slide.bullets[:MAX_SLIDE_BULLETS]
            ]
            slide.bullets = [bullet for bullet in slide.bullets if bullet]
            slide.notes = clip(slide.notes, MAX_NOTES_CHARS, keep_lines=True)

    return warnings


def _block_has_content(block) -> bool:
    if block.type in ("heading", "paragraph"):
        return bool(block.text)
    if block.type in ("bullets", "numbered"):
        return bool(block.items)
    return bool(block.columns or block.rows)


# ── Text that could become something else ─────────────────────────────────

# Totals over one range, and nothing else. HYPERLINK, WEBSERVICE, DDE and the
# rest reach outside the workbook, so any other formula is written as text.
_ALLOWED_FORMULA = re.compile(
    r"^=(SUM|AVERAGE|MIN|MAX|COUNT)\([A-Z]{1,3}[1-9]\d{0,6}(:[A-Z]{1,3}[1-9]\d{0,6})?\)$"
)


def is_allowed_formula(text: str) -> bool:
    return bool(_ALLOWED_FORMULA.match(text.strip().upper())) if text else False


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def split_bold(text: str) -> list[tuple[str, bool]]:
    """Cuts text into (run, is_bold) pieces at **bold** markers."""
    runs: list[tuple[str, bool]] = []
    position = 0
    for match in _BOLD.finditer(text):
        if match.start() > position:
            runs.append((text[position : match.start()], False))
        runs.append((match.group(1), True))
        position = match.end()
    if position < len(text):
        runs.append((text[position:], False))
    return [(run.replace("**", ""), bold) for run, bold in runs if run]


def plain_text(text: str) -> str:
    """The text with its bold markers removed."""
    return "".join(run for run, _ in split_bold(text))


def pdf_markup(text: str) -> str:
    """Text for a ReportLab Paragraph, with only bold and line breaks as markup.

    Paragraph reads a mini-HTML of its own, including <img src> — which loads
    a path from the server's disk. Every character the model wrote is escaped
    first, so none of it can be read as a tag.
    """
    pieces = [
        f"<b>{escape(run)}</b>" if bold else escape(run) for run, bold in split_bold(text)
    ]
    return "".join(pieces).replace("\n", "<br/>")


_FILENAME_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{n}" for n in range(1, 10)),
    *(f"lpt{n}" for n in range(1, 10)),
}


def safe_filename(title: str, extension: str) -> str:
    """A filename Windows, macOS and Linux all accept, with the right extension."""
    stem = _FILENAME_UNSAFE.sub(" ", plain_text(title or ""))
    stem = " ".join(stem.split()).strip(" .")[:60].strip(" .")
    if not stem or stem.lower() in _RESERVED_NAMES:
        stem = "LAFINA document"
    return f"{stem}.{extension}"
