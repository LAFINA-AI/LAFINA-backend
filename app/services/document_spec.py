"""What the model is asked for when a student wants a file, and reading it back.

The model never writes code and never touches a file. It describes the
document — headings, paragraphs, tables, sheets, slides — as one JSON object,
and the renderers in document_render.py turn that description into a PDF,
Word, Excel or PowerPoint file. Anything the model returns that does not fit
the shape below is refused here, before a renderer sees it.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.services.json_reply import json_candidates

DocumentFormat = Literal["pdf", "docx", "xlsx", "pptx"]
DOCUMENT_FORMATS: tuple[str, ...] = ("pdf", "docx", "xlsx", "pptx")

FORMAT_LABELS: dict[str, str] = {
    "pdf": "PDF",
    "docx": "Word",
    "xlsx": "Excel",
    "pptx": "PowerPoint",
}

CellValue = Union[bool, int, float, str, None]


class _SpecModel(BaseModel):
    # Extra keys are ignored rather than refused: a model that adds an "author"
    # it was not asked for has still described a perfectly usable document.
    # Numbers are accepted where text is expected, since a list item of 2024 is
    # still an item.
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


def _rows_as_lists(data: Any) -> Any:
    """Accepts rows written as objects keyed by column, the other common shape."""
    if not isinstance(data, dict):
        return data
    columns = data.get("columns")
    rows = data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return data
    if not any(isinstance(row, dict) for row in rows):
        return data
    converted = [
        [row.get(str(column)) for column in columns] if isinstance(row, dict) else row
        for row in rows
    ]
    return {**data, "rows": converted}


class HeadingBlock(_SpecModel):
    type: Literal["heading"]
    text: str
    level: int = 1


class ParagraphBlock(_SpecModel):
    type: Literal["paragraph"]
    text: str


class ListBlock(_SpecModel):
    type: Literal["bullets", "numbered"]
    items: list[str] = Field(default_factory=list)


class TableBlock(_SpecModel):
    type: Literal["table"]
    columns: list[str] = Field(default_factory=list)
    rows: list[list[CellValue]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise_rows(cls, data: Any) -> Any:
        return _rows_as_lists(data)


Block = Annotated[
    Union[HeadingBlock, ParagraphBlock, ListBlock, TableBlock],
    Field(discriminator="type"),
]


class SheetSpec(_SpecModel):
    name: str = "Sheet1"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[CellValue]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise_rows(cls, data: Any) -> Any:
        return _rows_as_lists(data)


class SlideSpec(_SpecModel):
    title: str = ""
    bullets: list[str] = Field(default_factory=list)
    notes: str = ""


class DocumentSpec(_SpecModel):
    title: str = ""
    summary: str = ""
    blocks: list[Block] = Field(default_factory=list)
    sheets: list[SheetSpec] = Field(default_factory=list)
    slides: list[SlideSpec] = Field(default_factory=list)

    def has_content_for(self, fmt: str) -> bool:
        if fmt in ("pdf", "docx"):
            return bool(self.blocks)
        if fmt == "xlsx":
            return any(sheet.columns or sheet.rows for sheet in self.sheets)
        if fmt == "pptx":
            return bool(self.slides)
        return False


class DocumentSpecParseError(Exception):
    """The reply held nothing that could be read as a document of that format."""


class DocumentRefusal(Exception):
    """The model declined the request, and said why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ── The prompt ────────────────────────────────────────────────────────────

_SHAPES: dict[str, str] = {
    "pdf": (
        '"blocks": an array, in reading order, of objects each with a "type" of:\n'
        '    "heading" with "text" and "level" (1, 2 or 3);\n'
        '    "paragraph" with "text";\n'
        '    "bullets" or "numbered" with "items" (an array of strings);\n'
        '    "table" with "columns" (header strings) and "rows" '
        "(arrays with one value per column)."
    ),
    "xlsx": (
        '"sheets": an array of worksheets, each with "name" (at most 31 characters), '
        '"columns" (header strings) and "rows" (arrays with one value per column).\n'
        "  Write numeric values as JSON numbers, not strings. For totals you may use only "
        "these formulas, written as a string over a single cell range: =SUM(B2:B10), "
        "=AVERAGE(B2:B10), =MIN(B2:B10), =MAX(B2:B10), =COUNT(B2:B10). No other formulas."
    ),
    "pptx": (
        '"slides": an array of the slides that follow the title slide, each with '
        '"title", "bullets" (three to six short strings) and optional "notes" '
        "(speaker notes)."
    ),
}
_SHAPES["docx"] = _SHAPES["pdf"]


def build_system_prompt(fmt: str, today: str = "") -> str:
    """The instruction for one format; only the key that format needs is described."""
    label = FORMAT_LABELS[fmt]
    date_line = f"Today's date is {today}.\n" if today else ""
    return (
        f"You are LAFINA's document writer. You turn a student's request, and the "
        f"conversation before it, into the content of a {label} file, and you output "
        "only raw JSON.\n"
        f"{date_line}"
        "Reply with a single JSON object with these keys:\n"
        '  "title": a short title for the document;\n'
        '  "summary": one or two sentences, written to the student, saying what the file '
        "contains;\n"
        f"  {_SHAPES[fmt]}\n"
        "Rules:\n"
        "1. No markdown fences, no commentary, nothing before or after the object. Inside "
        "text you may mark emphasis with **bold** only: no other markdown, HTML or code.\n"
        "2. Write complete, useful content: real headings, full sentences, realistic rows. "
        'Never leave placeholders such as "Lorem ipsum" or "[insert here]".\n'
        '3. When the request refers to the conversation ("the plan above"), build the '
        "document from it.\n"
        "4. Never include code to run, macros, download links, or instructions to run "
        "anything.\n"
        "5. If the request is harmful, sexual, hateful or harassing, or asks for a forged "
        "official document (IDs, certificates, receipts, prescriptions, signatures), reply "
        'only with {"refusal": "<one sentence saying why>"}.\n'
        "6. Do not use emojis."
    )


REPAIR_PROMPT = (
    "That reply could not be used: it was not a single JSON object with the keys described. "
    "Reply again with only the JSON object."
)


# ── Reading the reply ─────────────────────────────────────────────────────


def parse_document_spec(raw: str, fmt: str) -> DocumentSpec:
    """Reads a model reply into a spec for one format, or says why it cannot."""
    if not raw or not raw.strip():
        raise DocumentSpecParseError("The model returned an empty reply.")

    data = None
    for candidate in json_candidates(raw):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if not isinstance(data, dict):
        raise DocumentSpecParseError("The model reply was not a JSON object.")

    refusal = data.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise DocumentRefusal(" ".join(refusal.split())[:300])

    try:
        spec = DocumentSpec.model_validate(data)
    except ValidationError as err:
        raise DocumentSpecParseError(
            f"The model reply did not match the document shape ({err.error_count()} issues)."
        ) from err

    if not spec.has_content_for(fmt):
        raise DocumentSpecParseError(f"The model reply had no content for a {fmt} file.")
    return spec
