"""Reading text out of the three formats course material arrives in.

PDFs go through the pdfplumber path with its OCR fallback. Word and PowerPoint
files are Open XML — a zip of XML parts — so they are read here with the
standard library. python-docx and python-pptx are deployed for *writing* the
files the chat assistant generates (see document_render.py), but reading does
not need them: what they add is styling and document structure, and a summary
needs neither.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from xml.etree import ElementTree

from app.services.pdf_text import (
    DocumentText,
    PageText,
    PdfTextError,
    clean_page_text,
    extract_pdf_text,
    find_running_lines,
    strip_running_lines,
)

logger = logging.getLogger("lafina.document_text")

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx")

# Enough for a long chapter; past this the model would never see the tail
# anyway, and holding a whole book in memory helps nobody.
MAX_EXTRACTED_CHARS = 400_000

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

_SLIDE_NUMBER = re.compile(r"slide(\d+)\.xml$")


def detect_kind(filename: str, data: bytes) -> str:
    """Works out which reader a file needs, from its signature and its name.

    The signature decides first: .docx and .pptx are both zips, and a file
    renamed to the wrong extension is a likelier mistake than a hostile one.
    """
    name = (filename or "").lower()
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:2] == b"PK":
        # What the package holds beats what it is called: a deck saved with the
        # wrong extension is a likelier mistake than a hostile one.
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()
            if any(entry.startswith("ppt/slides/") for entry in names):
                return "pptx"
            if "word/document.xml" in names:
                return "docx"
        except zipfile.BadZipFile:
            # Not a readable zip; the extension is the only clue left, and the
            # reader will report what is actually wrong with it.
            if name.endswith(".pptx"):
                return "pptx"
            if name.endswith(".docx"):
                return "docx"
        raise PdfTextError(
            "That file is a zip, but not a Word or PowerPoint document.",
        )
    if name.endswith(".doc") or name.endswith(".ppt"):
        raise PdfTextError(
            "That is an older Word or PowerPoint file. Save it as .docx or .pptx and try again.",
        )
    raise PdfTextError("That file type is not supported. Upload a PDF, Word or PowerPoint file.")


def _open_package(data: bytes, what: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        # Password-protected Office files are not zips at all, which is the
        # most common way to arrive here.
        raise PdfTextError(
            f"That {what} file could not be opened. If it is password protected, "
            "remove the password and try again."
        ) from exc


def _element_text(element: ElementTree.Element, tag: str) -> str:
    """Joins the text runs under one node, honouring breaks and tabs."""
    parts: list[str] = []
    for node in element.iter():
        if node.tag == tag:
            parts.append(node.text or "")
        elif node.tag in (f"{_W}br", f"{_W}cr", f"{_A}br"):
            parts.append("\n")
        elif node.tag == f"{_W}tab":
            parts.append(" ")
    return "".join(parts)


def extract_docx_text(data: bytes) -> DocumentText:
    """Reads a Word document paragraph by paragraph.

    Table cells hold paragraphs of their own, so a table's contents come out
    too — as lines, which is all a summary needs of them.
    """
    document = DocumentText()
    with _open_package(data, "Word") as archive:
        try:
            body = archive.read("word/document.xml")
        except KeyError as exc:
            raise PdfTextError("That Word file has no readable document body.") from exc

    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise PdfTextError("That Word file is damaged and could not be read.") from exc

    lines: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        text = _element_text(paragraph, f"{_W}t").strip()
        if text:
            lines.append(text)
        elif lines and lines[-1] != "":
            # A blank paragraph is a break between blocks, which is what the
            # chunker later splits on.
            lines.append("")

    joined = clean_page_text("\n".join(lines))[:MAX_EXTRACTED_CHARS]
    document.total_pages = 1
    document.pages_read = 1
    document.pages.append(PageText(number=1, text=joined))
    if not joined.strip():
        raise PdfTextError("No readable text was found in that Word file.", status_code=422)
    return document


def extract_pptx_text(data: bytes) -> DocumentText:
    """Reads a PowerPoint deck slide by slide, in slide order."""
    document = DocumentText()
    with _open_package(data, "PowerPoint") as archive:
        slides = [
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        if not slides:
            raise PdfTextError("That PowerPoint file has no slides.")

        # namelist() order is arbitrary, and slide10 sorts before slide2.
        slides.sort(key=lambda name: int(_SLIDE_NUMBER.search(name).group(1))
                    if _SLIDE_NUMBER.search(name) else 0)
        document.total_pages = len(slides)

        total = 0
        for index, name in enumerate(slides, start=1):
            try:
                root = ElementTree.fromstring(archive.read(name))
            except (ElementTree.ParseError, KeyError):
                logger.warning("Slide %s could not be read", name)
                continue
            pieces = [
                (node.text or "").strip()
                for node in root.iter(f"{_A}t")
                if (node.text or "").strip()
            ]
            text = clean_page_text("\n".join(pieces))
            if total + len(text) > MAX_EXTRACTED_CHARS:
                document.warnings.append(
                    f"The deck was long, so only its first {index - 1} slides were read."
                )
                break
            total += len(text)
            document.pages.append(PageText(number=index, text=text))

    document.pages_read = len(document.pages)

    # A footer or a course code on every slide is noise in every summary.
    running = find_running_lines([page.text for page in document.pages])
    for page in document.pages:
        page.text = strip_running_lines(page.text, running)

    if not document.text.strip():
        raise PdfTextError(
            "No readable text was found in that PowerPoint file. Slides made of images "
            "cannot be read.",
            status_code=422,
        )
    return document


def extract_document_text(
    data: bytes,
    filename: str,
    *,
    max_pages: int = 40,
    allow_ocr: bool = True,
) -> tuple[DocumentText, str]:
    """Reads any supported document, returning its text and which kind it was."""
    if not data:
        raise PdfTextError("The uploaded file is empty.")

    kind = detect_kind(filename, data)
    if kind == "pdf":
        return extract_pdf_text(data, max_pages=max_pages, allow_ocr=allow_ocr), kind
    if kind == "docx":
        return extract_docx_text(data), kind
    return extract_pptx_text(data), kind
