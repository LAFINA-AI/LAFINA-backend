"""Reading Word and PowerPoint files, and summarising any document into notes."""

import base64
import io
import zipfile

import pytest
from httpx import AsyncClient

from app.services.document_text import (
    detect_kind,
    extract_docx_text,
    extract_document_text,
    extract_pptx_text,
)
from app.services.pdf_text import PdfTextError
from app.services.study_notes import (
    KeyTerm,
    NoteSection,
    StudyNotesParseError,
    StudySummary,
    merge_summaries,
    parse_study_notes_json,
    to_markdown,
)
from tests.test_flashcards import FakeDeepSeek, build_pdf


# ── Documents to read ─────────────────────────────────────────────────────

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def build_docx(paragraphs: list[str]) -> bytes:
    """A Word package with one paragraph per string."""
    body = "".join(
        f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p>' if text else "<w:p/>" for text in paragraphs
    )
    document = f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def build_pptx(slides: list[list[str]]) -> bytes:
    """A PowerPoint package with one text box per string, per slide."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        for index, lines in enumerate(slides, start=1):
            runs = "".join(f"<a:p><a:r><a:t>{line}</a:t></a:r></a:p>" for line in lines)
            archive.writestr(
                f"ppt/slides/slide{index}.xml",
                f'<?xml version="1.0"?><p:sld xmlns:p="{_P}" xmlns:a="{_A}">'
                f"<p:cSld>{runs}</p:cSld></p:sld>",
            )
    return buffer.getvalue()


# ── Detecting and reading ─────────────────────────────────────────────────


def test_each_supported_file_is_recognised_by_what_it_contains():
    assert detect_kind("notes.pdf", build_pdf(["Text."])) == "pdf"
    assert detect_kind("notes.docx", build_docx(["Text."])) == "docx"
    assert detect_kind("deck.pptx", build_pptx([["Text."]])) == "pptx"

    # A wrong extension loses to the package's own contents.
    assert detect_kind("deck.docx", build_pptx([["Text."]])) == "pptx"

    with pytest.raises(PdfTextError) as legacy:
        detect_kind("old.doc", b"\xd0\xcf\x11\xe0 legacy ole file")
    assert ".docx" in legacy.value.message

    with pytest.raises(PdfTextError) as other:
        detect_kind("photo.png", b"\x89PNG\r\n\x1a\n")
    assert "not supported" in other.value.message


def test_a_word_file_is_read_paragraph_by_paragraph():
    data = build_docx(
        [
            "Critical Reasoning",
            "An argument has premises and a conclusion.",
            "",
            "A fallacy is a mistake in reasoning that makes an argument unsound.",
        ]
    )
    document = extract_docx_text(data)
    assert "premises and a conclusion" in document.text
    assert "A fallacy is a mistake" in document.text

    with pytest.raises(PdfTextError):
        extract_docx_text(build_docx([]))


def test_a_powerpoint_file_is_read_slide_by_slide_in_order():
    data = build_pptx(
        [
            ["IT413", "What is critical reasoning?", "It is the evaluation of arguments."],
            ["IT413", "Premises", "A premise is a claim offered in support of a conclusion."],
            ["IT413", "Fallacies", "A fallacy is a recurring mistake in reasoning."],
        ]
    )
    document = extract_pptx_text(data)
    assert document.total_pages == 3
    assert [page.number for page in document.pages] == [1, 2, 3]
    assert "evaluation of arguments" in document.text
    assert "IT413" not in document.text, "the repeated slide header came off"


def test_slide_ten_does_not_sort_before_slide_two():
    data = build_pptx([[f"Slide {index} content about a topic."] for index in range(1, 12)])
    document = extract_pptx_text(data)
    assert [page.number for page in document.pages] == list(range(1, 12))
    assert document.text.index("Slide 2 ") < document.text.index("Slide 10 ")


def test_an_unopenable_office_file_says_so():
    with pytest.raises(PdfTextError) as excinfo:
        extract_docx_text(b"PK\x03\x04 not really a zip")
    assert "password protected" in excinfo.value.message


def test_the_dispatcher_returns_the_kind_it_read():
    document, kind = extract_document_text(build_docx(["A paragraph of course material."]), "a.docx")
    assert kind == "docx"
    assert "course material" in document.text

    document, kind = extract_document_text(
        build_pdf(["A page of course material with enough words on it to count as text."]),
        "a.pdf",
    )
    assert kind == "pdf"


# ── Reading the model's reply ─────────────────────────────────────────────


def test_a_clean_summary_parses():
    raw = """{
      "title": "Critical Reasoning",
      "overview": "The material covers arguments and how to evaluate them.",
      "sections": [{"heading": "Arguments", "points": ["A premise supports a conclusion."]}],
      "keyTerms": [{"term": "Fallacy", "meaning": "A recurring mistake in reasoning."}]
    }"""
    summary = parse_study_notes_json(raw)
    assert summary.title == "Critical Reasoning"
    assert summary.sections[0].heading == "Arguments"
    assert summary.sections[0].points == ["A premise supports a conclusion."]
    assert summary.key_terms[0].term == "Fallacy"


def test_the_usual_wrappers_and_key_names_survive():
    fenced = '```json\n{"title": "T", "overview": "O", "sections": [], "keyTerms": []}\n```'
    assert parse_study_notes_json(fenced).title == "T"

    chatty = 'Here are the notes:\n{"title": "T2", "overview": "O2"}\nHope that helps.'
    assert parse_study_notes_json(chatty).overview == "O2"

    other_keys = """{
      "name": "T3",
      "summary": "O3",
      "topics": [{"topic": "Heading", "bullets": ["A point."]}],
      "glossary": [{"word": "Term", "definition": "Meaning."}]
    }"""
    summary = parse_study_notes_json(other_keys)
    assert summary.title == "T3"
    assert summary.sections[0].heading == "Heading"
    assert summary.key_terms[0].meaning == "Meaning."

    # Only the sections, as a bare array.
    bare = '[{"heading": "H", "points": ["P"]}]'
    assert parse_study_notes_json(bare).sections[0].heading == "H"


def test_unusable_pieces_are_dropped_rather_than_crashing():
    raw = """{
      "title": "T",
      "sections": [
        {"heading": "Kept", "points": ["A real point.", "", 42]},
        {"heading": "", "points": []},
        "A heading on its own"
      ],
      "keyTerms": [{"term": "Only a term"}, {"term": "Good", "meaning": "Kept."}, "nonsense"]
    }"""
    summary = parse_study_notes_json(raw)
    assert [section.heading for section in summary.sections] == ["Kept", "A heading on its own"]
    assert summary.sections[0].points == ["A real point.", "42"]
    assert [term.term for term in summary.key_terms] == ["Good"]


def test_a_reply_with_no_json_is_reported():
    for bad in ("", "   ", "I cannot help with that."):
        with pytest.raises(StudyNotesParseError):
            parse_study_notes_json(bad)


# ── Merging ───────────────────────────────────────────────────────────────


def test_parts_of_one_document_become_one_set_of_notes():
    first = StudySummary(
        title="Critical Reasoning",
        overview="What the material is about.",
        sections=[NoteSection("Arguments", ["A premise supports a conclusion."])],
        key_terms=[KeyTerm("Premise", "A claim offered in support.")],
    )
    second = StudySummary(
        title="Part two",
        overview="Ignored, the document already introduced itself.",
        sections=[
            NoteSection("Arguments", ["A premise supports a conclusion.", "Conclusions follow."]),
            NoteSection("Fallacies", ["A fallacy is a recurring mistake."]),
        ],
        key_terms=[KeyTerm("premise", "A duplicate."), KeyTerm("Fallacy", "A mistake.")],
    )

    merged = merge_summaries([first, second])
    assert merged.title == "Critical Reasoning"
    assert merged.overview == "What the material is about."
    assert [section.heading for section in merged.sections] == ["Arguments", "Fallacies"]
    assert merged.sections[0].points == [
        "A premise supports a conclusion.",
        "Conclusions follow.",
    ], "the repeated point appears once"
    assert [term.term for term in merged.key_terms] == ["Premise", "Fallacy"]


def test_markdown_reads_as_notes():
    summary = StudySummary(
        title="Critical Reasoning",
        overview="An overview.",
        sections=[NoteSection("Arguments", ["A premise supports a conclusion."])],
        key_terms=[KeyTerm("Fallacy", "A mistake in reasoning.")],
    )
    text = to_markdown(summary)
    assert text.startswith("# Critical Reasoning")
    assert "## Arguments" in text
    assert "- A premise supports a conclusion." in text
    assert "## Key terms" in text
    assert "- **Fallacy** — A mistake in reasoning." in text


# ── The endpoint ──────────────────────────────────────────────────────────

SUMMARY_REPLY = """{
  "title": "Critical Reasoning",
  "overview": "The material introduces arguments and how to judge them.",
  "sections": [{"heading": "Arguments", "points": ["A premise supports a conclusion."]}],
  "keyTerms": [{"term": "Fallacy", "meaning": "A recurring mistake in reasoning."}]
}"""


async def _pro_headers(async_client: AsyncClient, email: str) -> dict[str, str]:
    from sqlalchemy import select

    from app.models.account import Account
    from tests.conftest import TestingSessionLocal

    password = "super-strong-lafina-passphrase-2026"
    await async_client.post("/v1/auth/register", json={"email": email, "password": password})
    async with TestingSessionLocal() as db:
        account = (await db.execute(select(Account).where(Account.email == email))).scalar_one()
        account.role = "student_pro"
        await db.commit()
    login = await async_client.post("/v1/auth/login", json={"email": email, "password": password})
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_study_notes_need_a_pro_subscription(async_client: AsyncClient):
    payload = {
        "filename": "notes.pptx",
        "contentBase64": base64.b64encode(build_pptx([["A slide."]])).decode(),
    }
    anon = await async_client.post("/v1/ai/study-notes", json=payload)
    assert anon.status_code == 401

    password = "super-strong-lafina-passphrase-2026"
    reg = await async_client.post(
        "/v1/auth/register", json={"email": "sn_student@ustp.edu.ph", "password": password}
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    res = await async_client.post("/v1/ai/study-notes", headers=headers, json=payload)
    assert res.status_code == 403
    assert "student_pro" in res.json()["detail"]


@pytest.mark.asyncio
async def test_a_powerpoint_deck_becomes_revision_notes(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "sn_pro@ustp.edu.ph")
    fake = FakeDeepSeek(replies=[SUMMARY_REPLY])
    app.dependency_overrides[get_deepseek_client] = lambda: fake
    try:
        deck = build_pptx(
            [
                ["What is critical reasoning?", "It is the evaluation of arguments."],
                ["Premises", "A premise is a claim offered in support of a conclusion."],
            ]
        )
        res = await async_client.post(
            "/v1/ai/study-notes",
            headers=headers,
            json={
                "filename": "IT413 Week04 Critical Reasoning.pptx",
                "contentBase64": base64.b64encode(deck).decode(),
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["title"] == "Critical Reasoning"
        assert data["sourceKind"] == "pptx"
        assert data["totalPages"] == 2
        assert data["sections"][0]["heading"] == "Arguments"
        assert data["keyTerms"][0]["term"] == "Fallacy"
        assert data["markdown"].startswith("# Critical Reasoning")

        sent = fake.calls[0]["messages"][1]["content"]
        assert "evaluation of arguments" in sent
        assert "PK" not in sent[:50], "the package itself never leaves the server"
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_a_word_file_and_an_unsupported_one(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "sn_docx@ustp.edu.ph")
    fake = FakeDeepSeek(replies=[SUMMARY_REPLY])
    app.dependency_overrides[get_deepseek_client] = lambda: fake
    try:
        docx = build_docx(["An argument has premises and a conclusion worth summarising here."])
        ok = await async_client.post(
            "/v1/ai/study-notes",
            headers=headers,
            json={"filename": "week04.docx", "contentBase64": base64.b64encode(docx).decode()},
        )
        assert ok.status_code == 200
        assert ok.json()["sourceKind"] == "docx"

        bad = await async_client.post(
            "/v1/ai/study-notes",
            headers=headers,
            json={
                "filename": "photo.png",
                "contentBase64": base64.b64encode(b"\x89PNG\r\n\x1a\n and more").decode(),
            },
        )
        assert bad.status_code == 400
        assert "not supported" in bad.json()["detail"]
        assert len(fake.calls) == 1, "the unsupported file never reached the model"
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_a_summary_with_nothing_in_it_is_not_passed_off_as_notes(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "sn_empty@ustp.edu.ph")
    app.dependency_overrides[get_deepseek_client] = lambda: FakeDeepSeek(
        replies=['{"title": "", "overview": "", "sections": [], "keyTerms": []}']
    )
    try:
        res = await async_client.post(
            "/v1/ai/study-notes",
            headers=headers,
            json={
                "filename": "admin.docx",
                "contentBase64": base64.b64encode(
                    build_docx(["Course outline. Grading policy. Consultation hours."])
                ).decode(),
            },
        )
        assert res.status_code == 422
        assert "nothing worth summarising" in res.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)
