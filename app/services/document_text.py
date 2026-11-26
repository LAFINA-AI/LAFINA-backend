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
import zlib
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
MAX_ZIP_MEMBER_BYTES = 20 * 1024 * 1024  # 20 MiB per XML part
MAX_ZIP_TOTAL_BYTES = 30 * 1024 * 1024   # 30 MiB cumulative across parts
MAX_PPTX_SLIDES_DEFAULT = 40

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


def _safe_read_zip_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    what: str,
    max_bytes: int = MAX_ZIP_MEMBER_BYTES,
) -> bytes:
    """Reads one ZIP entry with streaming byte caps to defeat zip bombs and handle corruption."""
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise KeyError(name) from exc
    if info.flag_bits & 0x1:
        raise PdfTextError(
            f"That {what} file is password protected. Remove the password and try again."
        )
    if info.file_size > max_bytes:
        raise PdfTextError(f"That {what} file is too large to process safely.")
    buf = bytearray()
    try:
        with archive.open(info, "r") as stream:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise PdfTextError(f"That {what} file is too large to process safely.")
    except PdfTextError:
        raise
    except RuntimeError as exc:
        raise PdfTextError(
            f"That {what} file is password protected or unreadable. Remove any password and try again."
        ) from exc
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
        raise PdfTextError(f"That {what} file is damaged and could not be read.") from exc
    return bytes(buf)


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


def _paragraph_own_text(paragraph: ElementTree.Element) -> str:
    """Collects text runs owned directly by ``paragraph`` without re-reading nested ``<w:p>`` text boxes."""
    parts: list[str] = []
    stack = list(reversed(list(paragraph)))
    p_tag = f"{_W}p"
    t_tag = f"{_W}t"
    while stack:
        node = stack.pop()
        if node.tag == p_tag:
            # Nested paragraph (e.g. inside a drawing text box) will be visited on its own.
            continue
        if node.tag == t_tag:
            parts.append(node.text or "")
        elif node.tag in (f"{_W}br", f"{_W}cr", f"{_A}br"):
            parts.append("\n")
        elif node.tag == f"{_W}tab":
            parts.append(" ")
        else:
            children = list(node)
            if children:
                stack.extend(reversed(children))
    return "".join(parts)


def extract_docx_text(data: bytes) -> DocumentText:
    """Reads a Word document paragraph by paragraph.

    Table cells hold paragraphs of their own, so a table's contents come out
    too — as lines, which is all a summary needs of them.
    """
    document = DocumentText()
    with _open_package(data, "Word") as archive:
        try:
            body = _safe_read_zip_member(archive, "word/document.xml", what="Word")
        except KeyError as exc:
            raise PdfTextError("That Word file has no readable document body.") from exc

    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise PdfTextError("That Word file is damaged and could not be read.") from exc

    lines: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        text = _paragraph_own_text(paragraph).strip()
        if text:
            lines.append(text)
        elif lines and lines[-1] != "":
            # A blank paragraph is a break between blocks, which is what the
            # chunker later splits on.
            lines.append("")

    joined = clean_page_text("\n".join(lines), strip_page_numbers=False)[:MAX_EXTRACTED_CHARS]
    document.total_pages = 1
    document.pages_read = 1
    document.pages.append(PageText(number=1, text=joined))
    if not joined.strip():
        raise PdfTextError("No readable text was found in that Word file.", status_code=422)
    return document


def extract_pptx_text(data: bytes, *, max_pages: int = MAX_PPTX_SLIDES_DEFAULT) -> DocumentText:
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
        slides.sort(
            key=lambda name: int(_SLIDE_NUMBER.search(name).group(1))
            if _SLIDE_NUMBER.search(name)
            else 0
        )
        document.total_pages = len(slides)
        if len(slides) > max_pages:
            document.warnings.append(
                f"Only the first {max_pages} of {len(slides)} slides were read."
            )
            slides = slides[:max_pages]

        total = 0
        decompressed_total = 0
        for index, name in enumerate(slides, start=1):
            remaining_bytes = MAX_ZIP_TOTAL_BYTES - decompressed_total
            if remaining_bytes <= 0:
                document.warnings.append(
                    f"The deck was too large, so only its first {index - 1} slides were read."
                )
                break
            try:
                raw_slide = _safe_read_zip_member(
                    archive,
                    name,
                    what="PowerPoint",
                    max_bytes=min(MAX_ZIP_MEMBER_BYTES, remaining_bytes),
                )
                decompressed_total += len(raw_slide)
                root = ElementTree.fromstring(raw_slide)
            except (ElementTree.ParseError, KeyError):
                logger.warning("Slide %s could not be read", name)
                continue
            paragraphs = list(root.iter(f"{_A}p"))
            if paragraphs:
                pieces = [
                    _element_text(p, f"{_A}t").strip()
                    for p in paragraphs
                    if _element_text(p, f"{_A}t").strip()
                ]
            else:
                pieces = [
                    (node.text or "").strip()
                    for node in root.iter(f"{_A}t")
                    if (node.text or "").strip()
                ]
            text = clean_page_text("\n".join(pieces), strip_page_numbers=False)
            if total + len(text) > MAX_EXTRACTED_CHARS:
                if total == 0:
                    # Slide 1 alone exceeds the ceiling: truncate instead of failing as image-only.
                    text = text[:MAX_EXTRACTED_CHARS]
                    document.pages.append(PageText(number=index, text=text))
                    document.warnings.append(
                        "Slide 1 was very long, so its text was shortened."
                    )
                else:
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
    return extract_pptx_text(data, max_pages=max_pages), kind
