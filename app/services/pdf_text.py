"""Turning an uploaded PDF into clean, model-sized text.

DeepSeek takes text, not PDFs, so everything upstream of the model happens
here: pull the digital text out page by page, fall back to OCR for the pages a
scanner produced, strip the furniture that repeats on every page, and cut what
is left into chunks that fit the context window without slicing a concept in
half.

Both readers are optional at runtime. pdfplumber is a pure-Python dependency
and is expected to be installed; OCR additionally needs the Tesseract binary,
which plenty of hosts do not have. Neither is imported at module load, so a
deployment without them starts normally and reports the limitation when someone
actually uploads a file.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("lafina.pdf_text")

# A page with less than this much text is treated as scanned rather than
# digital. Real pages of prose run into the thousands; a scan usually yields a
# stray ligature or two from the image metadata, and an empty page yields none.
MIN_CHARS_FOR_DIGITAL_PAGE = 40

# Fraction of pages a line must appear on to count as a running header or
# footer rather than as content that happens to repeat.
RUNNING_LINE_RATIO = 0.6

DEFAULT_CHUNK_CHARS = 6000
DEFAULT_CHUNK_OVERLAP_CHARS = 400
OCR_RESOLUTION_DPI = 200

_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:page\s*)?\d+\s*(?:/|of|\|)?\s*\d*\s*$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class PdfTextError(Exception):
    """A PDF that cannot be turned into text, with the status to answer with."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class PageText:
    number: int
    text: str
    ocr_used: bool = False


@dataclass
class DocumentText:
    pages: list[PageText] = field(default_factory=list)
    total_pages: int = 0
    pages_read: int = 0
    ocr_page_numbers: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def characters(self) -> int:
        return len(self.text)


# ── Optional readers ──────────────────────────────────────────────────────


def pdf_reader_available() -> bool:
    """True when pdfplumber can be imported."""
    try:
        import pdfplumber  # noqa: F401
    except Exception:
        return False
    return True


def ocr_available() -> bool:
    """True when both pytesseract and the Tesseract binary behind it are usable.

    The import succeeding proves nothing: pytesseract is a wrapper around a
    command-line program that most managed hosts do not ship.
    """
    try:
        import pytesseract
    except Exception:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def _ocr_page(page) -> str:
    """Reads one page as an image. Returns "" when OCR cannot run at all."""
    import pytesseract

    try:
        image = page.to_image(resolution=OCR_RESOLUTION_DPI).original
    except Exception as exc:
        logger.warning("Page %s could not be rendered for OCR: %s", page.page_number, exc)
        return ""
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception as exc:
        logger.warning("OCR failed on page %s: %s", page.page_number, exc)
        return ""


# ── Cleaning ──────────────────────────────────────────────────────────────


def clean_page_text(raw: str) -> str:
    """Collapses the whitespace a PDF extractor leaves behind.

    Layout-driven extraction produces runs of spaces where a column gap was,
    and blank lines where vertical space was. Both cost tokens and teach the
    model nothing.
    """
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    # A hyphen at end of line is a word broken by the typesetter, not a hyphen.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    kept = [line for line in lines if line and not _PAGE_NUMBER_LINE.match(line)]
    return "\n".join(kept)


def find_running_lines(pages: list[str], ratio: float = RUNNING_LINE_RATIO) -> set[str]:
    """Finds the header and footer lines that repeat across the document.

    Only the lines at the edges of a page are candidates, so a phrase
    that genuinely recurs in the body — a defined term, say — is never mistaken
    for furniture. How many edge lines count depends on the page: a long page
    can carry a two-line header, but on a short one that would leave nothing
    to be body text, and the content itself would be stripped.
    """
    if len(pages) < 3:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        lines = [line for line in page.split("\n") if line]
        if len(lines) < 3:
            continue
        edge = 2 if len(lines) >= 6 else 1
        candidates = lines[:edge] + lines[-edge:]
        for line in set(candidates):
            if len(line) > 120:
                continue
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(pages) * ratio))
    return {line for line, count in counts.items() if count >= threshold}


def strip_running_lines(page: str, running: set[str]) -> str:
    if not running:
        return page
    return "\n".join(line for line in page.split("\n") if line not in running)


# ── Chunking ──────────────────────────────────────────────────────────────


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Breaks an oversized paragraph on sentence ends, never mid-sentence."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    # A single sentence longer than the limit (a table flattened into one line,
    # usually) is the one case where a hard cut is the only option.
    final: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            final.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            final.append(piece)
    return final


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Cuts text into model-sized pieces on paragraph boundaries.

    Chunks carry the tail of the one before them, so a definition that begins
    at the end of a chunk is still whole somewhere: without that overlap the
    card for it would be generated from half a sentence, or not at all.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    overlap = max(0, min(overlap_chars, max_chars // 2))

    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            units.extend(_split_long_paragraph(paragraph, max_chars))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the carried-over context at a sentence boundary so the next
            # chunk never opens mid-word.
            if tail:
                cut = tail.find(" ")
                tail = tail[cut + 1 :] if cut != -1 else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ── Extraction ────────────────────────────────────────────────────────────


def extract_pdf_text(
    data: bytes,
    *,
    max_pages: int = 40,
    allow_ocr: bool = True,
    password: str | None = None,
) -> DocumentText:
    """Reads a PDF into cleaned per-page text, using OCR where it must.

    Raises PdfTextError for anything the caller should be told about: no PDF
    reader installed, a file that is not a PDF, an encrypted one, or one whose
    pages hold no recoverable text.
    """
    if not data:
        raise PdfTextError("The uploaded file is empty.")
    if not data[:5].startswith(b"%PDF-"):
        raise PdfTextError("That file is not a PDF.")

    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - depends on the deployment
        logger.error("pdfplumber is unavailable: %s", exc)
        raise PdfTextError(
            "PDF reading is not installed on the server. Ask an administrator to install pdfplumber.",
            status_code=503,
        ) from exc

    document = DocumentText()
    ocr_ready = allow_ocr and ocr_available()
    ocr_wanted = False

    try:
        with pdfplumber.open(io.BytesIO(data), password=password or "") as pdf:
            document.total_pages = len(pdf.pages)
            if document.total_pages == 0:
                raise PdfTextError("That PDF has no pages.")
            if document.total_pages > max_pages:
                document.warnings.append(
                    f"Only the first {max_pages} of {document.total_pages} pages were read."
                )

            for page in pdf.pages[:max_pages]:
                try:
                    raw = page.extract_text() or ""
                except Exception as exc:
                    logger.warning("Page %s could not be read: %s", page.page_number, exc)
                    raw = ""

                used_ocr = False
                if len(raw.strip()) < MIN_CHARS_FOR_DIGITAL_PAGE:
                    ocr_wanted = True
                    if ocr_ready:
                        ocr_text = _ocr_page(page)
                        if len(ocr_text.strip()) > len(raw.strip()):
                            raw = ocr_text
                            used_ocr = True

                document.pages.append(
                    PageText(number=page.page_number, text=clean_page_text(raw), ocr_used=used_ocr)
                )
                if used_ocr:
                    document.ocr_page_numbers.append(page.page_number)
    except PdfTextError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "encrypt" in message:
            raise PdfTextError(
                "That PDF is password protected. Remove the password and try again."
            ) from exc
        logger.warning("PDF could not be opened: %s", exc)
        raise PdfTextError("That PDF could not be read. It may be damaged.") from exc

    document.pages_read = len(document.pages)

    # Furniture is only recognisable across pages, so it comes off at the end.
    running = find_running_lines([page.text for page in document.pages])
    for page in document.pages:
        page.text = strip_running_lines(page.text, running)

    if document.characters < MIN_CHARS_FOR_DIGITAL_PAGE:
        if ocr_wanted and not ocr_ready:
            raise PdfTextError(
                "That PDF looks scanned, and text recognition is not available on the server. "
                "Upload a PDF with selectable text instead.",
                status_code=422,
            )
        raise PdfTextError(
            "No readable text was found in that PDF.",
            status_code=422,
        )

    if ocr_wanted and not ocr_ready:
        document.warnings.append(
            "Some pages look scanned and were skipped: text recognition is not available on the server."
        )

    return document
