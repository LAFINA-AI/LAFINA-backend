"""The PDF → flashcards pipeline, and the endpoint that drives it."""

import base64
import zlib

import pytest
from httpx import AsyncClient

from app.services.flashcards import (
    Flashcard,
    FlashcardParseError,
    escape_anki_field,
    merge_cards,
    parse_flashcard_json,
    to_anki_tsv,
    write_anki_tsv,
)
from app.services import pdf_text
from app.services.pdf_text import (
    PdfTextError,
    chunk_text,
    clean_page_text,
    extract_pdf_text,
    find_running_lines,
    strip_running_lines,
)


# ── A PDF to read ─────────────────────────────────────────────────────────


def build_pdf(pages: list[str]) -> bytes:
    """Assembles a minimal one-font PDF, so the reader is tested on a real file.

    Each string becomes a page of Helvetica text; an empty string becomes a
    page with no text at all, which is what a scan looks like to an extractor.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []

    for text in pages:
        if text:
            lines = text.split("\n")
            escaped = "".join(
                f"({line.replace(chr(92), chr(92) * 2).replace('(', chr(92) + '(').replace(')', chr(92) + ')')}) Tj 0 -16 Td\n"
                for line in lines
            )
            stream = f"BT /F1 12 Tf 54 720 Td\n{escaped}ET".encode("latin-1", "replace")
        else:
            stream = b""
        packed = zlib.compress(stream)
        content_ids.append(
            add(
                b"<< /Length "
                + str(len(packed)).encode()
                + b" /Filter /FlateDecode >>\nstream\n"
                + packed
                + b"\nendstream"
            )
        )

    pages_id = len(objects) + len(pages) + 1
    for content_id in content_ids:
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


# ── Cleaning ──────────────────────────────────────────────────────────────


def test_cleaning_removes_the_furniture_and_keeps_the_words():
    raw = "Cell    Biology\n\n\nThe mito-\nchondrion makes ATP.\n  \n12\n"
    cleaned = clean_page_text(raw)
    assert cleaned == "Cell Biology\nThe mitochondrion makes ATP."
    assert clean_page_text("") == ""
    assert clean_page_text("Page 4 of 12") == ""


def test_repeated_headers_are_stripped_but_repeated_content_is_not():
    body = "Mitochondria are organelles."
    pages = [f"BIO 101 Lecture Notes\n{body}\nDr Reyes" for _ in range(5)]
    running = find_running_lines(pages)
    assert "BIO 101 Lecture Notes" in running
    assert "Dr Reyes" in running
    assert body not in running, "a line in the body is never furniture"

    stripped = strip_running_lines(pages[0], running)
    assert stripped == body

    # Two pages are not enough evidence that anything repeats.
    assert find_running_lines(["A\nbody", "A\nbody"]) == set()


# ── Chunking ──────────────────────────────────────────────────────────────


def test_short_text_is_one_chunk_and_empty_text_is_none():
    assert chunk_text("Short enough.") == ["Short enough."]
    assert chunk_text("") == []
    assert chunk_text("   \n\n ") == []


def test_long_text_splits_on_paragraphs_with_overlap():
    paragraphs = [f"Paragraph {index} about a concept worth learning. " * 6 for index in range(12)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=1000, overlap_chars=200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 + 200 for chunk in chunks)
    for chunk in chunks:
        assert not chunk.startswith(" "), "a chunk never opens mid-word"
    # Every paragraph survives somewhere, so no concept is lost at a boundary.
    joined = " ".join(chunks)
    for index in range(12):
        assert f"Paragraph {index} about" in joined


def test_an_oversized_paragraph_is_split_between_sentences():
    paragraph = " ".join(f"Sentence number {index} carries a fact." for index in range(80))
    chunks = chunk_text(paragraph, max_chars=400, overlap_chars=50)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.rstrip().endswith("."), f"cut mid-sentence: {chunk[-40:]!r}"


def test_a_single_sentence_longer_than_a_chunk_still_terminates():
    monster = "x" * 5000
    chunks = chunk_text(monster, max_chars=500, overlap_chars=50)
    assert len(chunks) >= 10
    assert sum(len(chunk) for chunk in chunks) >= 5000


# ── Reading a PDF ─────────────────────────────────────────────────────────


def test_a_digital_pdf_is_read_page_by_page():
    pdf = build_pdf(
        [
            "BIO 101\nMitochondria produce ATP by oxidative phosphorylation.\nDr Reyes",
            "BIO 101\nRibosomes assemble proteins from messenger RNA.\nDr Reyes",
            "BIO 101\nThe Golgi apparatus packages proteins for transport.\nDr Reyes",
        ]
    )
    document = extract_pdf_text(pdf)
    assert document.total_pages == 3
    assert document.pages_read == 3
    assert document.ocr_page_numbers == []
    assert "oxidative phosphorylation" in document.text
    assert "BIO 101" not in document.text, "the running header came off"


def test_a_file_that_is_not_a_pdf_is_refused_clearly():
    with pytest.raises(PdfTextError) as excinfo:
        extract_pdf_text(b"PK\x03\x04 this is a zip")
    assert "not a PDF" in excinfo.value.message

    with pytest.raises(PdfTextError):
        extract_pdf_text(b"")


def test_a_scanned_pdf_falls_back_to_ocr(monkeypatch):
    scanned = build_pdf(["", ""])

    # Without recognition installed, the answer says so instead of returning
    # an empty deck.
    monkeypatch.setattr(pdf_text, "ocr_available", lambda: False)
    with pytest.raises(PdfTextError) as excinfo:
        extract_pdf_text(scanned)
    assert "scanned" in excinfo.value.message
    assert excinfo.value.status_code == 422

    # With it, the pages come back through OCR.
    monkeypatch.setattr(pdf_text, "ocr_available", lambda: True)
    monkeypatch.setattr(
        pdf_text,
        "_ocr_page",
        lambda page: f"Recognised page {page.page_number} about enzyme kinetics and substrates.",
    )
    document = extract_pdf_text(scanned)
    assert document.ocr_page_numbers == [1, 2]
    assert "enzyme kinetics" in document.text


def test_a_long_pdf_reads_only_its_first_pages():
    pdf = build_pdf([f"Page {index} covers a distinct topic worth examining." for index in range(8)])
    document = extract_pdf_text(pdf, max_pages=3)
    assert document.total_pages == 8
    assert document.pages_read == 3
    assert any("first 3" in warning for warning in document.warnings)


# ── Reading the model's reply ─────────────────────────────────────────────


def test_a_clean_json_array_parses():
    cards = parse_flashcard_json('[{"question": "What is ATP?", "answer": "The cell energy currency."}]')
    assert cards == [Flashcard("What is ATP?", "The cell energy currency.")]


def test_fences_prose_and_wrappers_are_all_survivable():
    fenced = '```json\n[{"question": "Q one?", "answer": "A one."}]\n```'
    chatty = 'Here are your flashcards:\n[{"question": "Q two?", "answer": "A two."}]\nHope this helps!'
    wrapped = '{"flashcards": [{"question": "Q three?", "answer": "A three."}]}'
    single = '{"question": "Q four?", "answer": "A four."}'

    assert parse_flashcard_json(fenced)[0].question == "Q one?"
    assert parse_flashcard_json(chatty)[0].question == "Q two?"
    assert parse_flashcard_json(wrapped)[0].question == "Q three?"
    assert parse_flashcard_json(single)[0].question == "Q four?"


def test_other_key_names_and_unusable_entries():
    raw = """[
      {"front": "Front key?", "back": "Back key."},
      {"term": "Osmosis", "definition": "Water moving down its gradient."},
      "not an object",
      {"question": "", "answer": "No question."},
      {"question": "No answer?", "answer": "  "},
      {"question": "Kept?", "answer": "Yes."}
    ]"""
    cards = parse_flashcard_json(raw)
    assert [card.question for card in cards] == ["Front key?", "Osmosis", "Kept?"]


def test_a_bracket_inside_an_answer_does_not_end_the_array():
    raw = 'Sure!\n[{"question": "What is a list?", "answer": "Written as [1, 2, 3] in code."}]'
    cards = parse_flashcard_json(raw)
    assert len(cards) == 1
    assert "[1, 2, 3]" in cards[0].answer


def test_a_reply_with_no_json_is_reported_not_guessed():
    for bad in ("", "   ", "I cannot help with that.", "[not json at all"):
        with pytest.raises(FlashcardParseError):
            parse_flashcard_json(bad)


def test_long_fields_are_truncated_rather_than_dropped():
    long_answer = "word " * 1000
    cards = parse_flashcard_json(f'[{{"question": "Long?", "answer": "{long_answer}"}}]')
    assert len(cards[0].answer) <= 1501
    assert cards[0].answer.endswith("…")


# ── Merging ───────────────────────────────────────────────────────────────


def test_overlapping_chunks_do_not_produce_the_same_card_twice():
    first = [Flashcard("What is ATP?", "Energy currency."), Flashcard("What is DNA?", "Genetic material.")]
    second = [Flashcard("what is atp?", "Energy currency, restated."), Flashcard("What is RNA?", "Messenger.")]
    merged = merge_cards([first, second])
    assert [card.question for card in merged] == ["What is ATP?", "What is DNA?", "What is RNA?"]


def test_a_deck_stops_at_its_limit():
    batch = [Flashcard(f"Question {index}?", "Answer.") for index in range(50)]
    assert len(merge_cards([batch], limit=10)) == 10


# ── Anki export ───────────────────────────────────────────────────────────


def test_tabs_and_newlines_never_break_a_row():
    field = escape_anki_field("Line one\nLine two\tcolumn")
    assert "\t" not in field
    assert "\n" not in field
    assert field == "Line one<br>Line two column"


def test_the_export_is_two_columns_per_card_under_its_header():
    cards = [Flashcard("Q1\twith tab", "A1\nwith newline"), Flashcard("Q2", "A2")]
    tsv = to_anki_tsv(cards)
    lines = tsv.strip().split("\n")
    assert lines[0] == "#separator:tab"
    assert lines[1] == "#html:true"
    assert lines[2] == "#columns:Front\tBack"
    for line in lines[3:]:
        assert len(line.split("\t")) == 2
    assert to_anki_tsv(cards, include_header=False).startswith("Q1 with tab\t")


def test_writing_the_deck_produces_a_utf8_file(tmp_path):
    target = write_anki_tsv([Flashcard("Qué es ATP?", "Moneda energética.")], tmp_path / "deck.tsv")
    assert target.read_text(encoding="utf-8").endswith("Qué es ATP?\tMoneda energética.\n")


# ── The endpoint ──────────────────────────────────────────────────────────


class FakeDeepSeek:
    """Stands in for the provider: records calls, replies how it is told to."""

    def __init__(self, replies=None, error=None):
        self.replies = replies or ['[{"question": "What is ATP?", "answer": "Energy currency."}]']
        self.error = error
        self.calls = []

    async def json_completion(self, messages, user_id, request_id="", model=None, max_tokens=4096):
        self.calls.append({"messages": messages, "model": model, "request_id": request_id})
        if self.error:
            raise self.error
        reply = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return reply, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


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


def _payload(pdf: bytes, **extra) -> dict:
    body = {"filename": "Cell Biology Notes.pdf", "contentBase64": base64.b64encode(pdf).decode()}
    body.update(extra)
    return body


@pytest.mark.asyncio
async def test_flashcards_need_a_pro_subscription(async_client: AsyncClient):
    pdf = build_pdf(["Mitochondria produce ATP by oxidative phosphorylation in the cristae."])

    anon = await async_client.post("/v1/ai/flashcards", json=_payload(pdf))
    assert anon.status_code == 401

    password = "super-strong-lafina-passphrase-2026"
    reg = await async_client.post(
        "/v1/auth/register", json={"email": "fc_student@ustp.edu.ph", "password": password}
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    res = await async_client.post("/v1/ai/flashcards", headers=headers, json=_payload(pdf))
    assert res.status_code == 403
    assert "student_pro" in res.json()["detail"]


@pytest.mark.asyncio
async def test_a_pdf_becomes_a_deck(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "fc_pro@ustp.edu.ph")
    fake = FakeDeepSeek(
        replies=[
            '```json\n[{"question": "What produces ATP?", "answer": "The mitochondrion."},'
            '{"question": "What assembles proteins?", "answer": "The ribosome."}]\n```'
        ]
    )
    app.dependency_overrides[get_deepseek_client] = lambda: fake
    try:
        pdf = build_pdf(
            [
                "Mitochondria produce ATP by oxidative phosphorylation across the inner membrane.",
                "Ribosomes assemble proteins by translating messenger RNA into polypeptides.",
            ]
        )
        res = await async_client.post("/v1/ai/flashcards", headers=headers, json=_payload(pdf))
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["deckTitle"] == "Cell Biology Notes"
        assert [card["question"] for card in data["cards"]] == [
            "What produces ATP?",
            "What assembles proteins?",
        ]
        assert data["totalPages"] == 2
        assert data["ocrPages"] == []
        assert data["usage"]["total_tokens"] == 15
        # The document text reached the model, and the PDF itself never did.
        sent = fake.calls[0]["messages"][1]["content"]
        assert "oxidative phosphorylation" in sent
        assert "%PDF" not in sent
        assert fake.calls[0]["model"] == "deepseek-chat"
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_unusable_uploads_are_refused_before_the_model_is_called(async_client: AsyncClient):
    from app.api.v1.ai import MAX_FLASHCARD_PDF_BYTES, get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "fc_bad@ustp.edu.ph")
    fake = FakeDeepSeek()
    app.dependency_overrides[get_deepseek_client] = lambda: fake
    try:
        not_a_pdf = await async_client.post(
            "/v1/ai/flashcards", headers=headers, json=_payload(b"just some text, not a document")
        )
        assert not_a_pdf.status_code == 400
        assert "not a PDF" in not_a_pdf.json()["detail"]

        undecodable = await async_client.post(
            "/v1/ai/flashcards",
            headers=headers,
            json={"filename": "x.pdf", "contentBase64": "not base64 at all!!!"},
        )
        assert undecodable.status_code == 400

        oversized = await async_client.post(
            "/v1/ai/flashcards",
            headers=headers,
            json=_payload(b"%PDF-" + b"0" * (MAX_FLASHCARD_PDF_BYTES + 1)),
        )
        assert oversized.status_code == 413

        empty_pdf = await async_client.post(
            "/v1/ai/flashcards", headers=headers, json=_payload(build_pdf([""]))
        )
        assert empty_pdf.status_code == 422

        assert fake.calls == [], "nothing unusable was ever sent upstream"
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_a_provider_failure_is_reported_with_its_own_status(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.clients.deepseek import DeepSeekTimeoutError
    from app.main import app

    headers = await _pro_headers(async_client, "fc_down@ustp.edu.ph")
    app.dependency_overrides[get_deepseek_client] = lambda: FakeDeepSeek(
        error=DeepSeekTimeoutError()
    )
    try:
        pdf = build_pdf(["Enzyme kinetics describe how substrate concentration affects rate."])
        res = await async_client.post("/v1/ai/flashcards", headers=headers, json=_payload(pdf))
        assert res.status_code == 504
        assert "timed out" in res.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_a_reply_with_no_cards_in_it_is_not_passed_off_as_a_deck(async_client: AsyncClient):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "fc_empty@ustp.edu.ph")
    app.dependency_overrides[get_deepseek_client] = lambda: FakeDeepSeek(replies=["[]"])
    try:
        pdf = build_pdf(["Course outline. Grading policy. Consultation hours by appointment."])
        res = await async_client.post("/v1/ai/flashcards", headers=headers, json=_payload(pdf))
        assert res.status_code == 422
        assert "testable" in res.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)


@pytest.mark.asyncio
async def test_a_document_upload_is_not_held_to_the_one_mib_body_limit(async_client: AsyncClient):
    """The global 1 MiB cap would reject almost any real PDF."""
    import os

    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    headers = await _pro_headers(async_client, "fc_big@ustp.edu.ph")
    app.dependency_overrides[get_deepseek_client] = lambda: FakeDeepSeek()
    try:
        # Random bytes so base64 stays large, with the PDF signature in front.
        blob = b"%PDF-" + os.urandom(1_500_000)
        res = await async_client.post("/v1/ai/flashcards", headers=headers, json=_payload(blob))
        assert res.status_code != 413, "a document-sized body reached the endpoint"
        assert res.status_code == 400
        assert "could not be read" in res.json()["detail"]

        # Every other route keeps the tight limit.
        other = await async_client.post(
            "/v1/auth/login",
            json={"email": "someone@ustp.edu.ph", "password": "x" * 1_500_000},
        )
        assert other.status_code == 413
    finally:
        app.dependency_overrides.pop(get_deepseek_client, None)
