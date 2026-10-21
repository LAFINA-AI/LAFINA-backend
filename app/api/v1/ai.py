import asyncio
import base64
import binascii
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.config import get_settings
from app.database import get_db
from app.models.account import Account
from app.models.session import AuthSession
from app.models.ai_usage import AIUsage
from app.security.auth import get_current_user_and_session
from app.services.capabilities import (
    has_online_ai_access,
    has_tts_access,
    is_student_pro,
)
from app.clients.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekMalformedResponseError,
)
from app.clients.gemini_tts import GeminiTtsClient, GeminiTtsError
from app.services.flashcards import (
    FLASHCARD_SYSTEM_PROMPT,
    Flashcard,
    FlashcardParseError,
    build_user_prompt,
    merge_cards,
    parse_flashcard_json,
)
from app.services.pdf_text import PdfTextError, chunk_text, extract_pdf_text
from app.services.document_text import extract_document_text
from app.services.study_notes import (
    STUDY_NOTES_SYSTEM_PROMPT,
    StudySummary,
    merge_summaries,
    parse_study_notes_json,
    to_markdown,
)
from app.services.study_notes import build_user_prompt as build_study_prompt
from app.services.document_spec import (
    REPAIR_PROMPT,
    DocumentFormat,
    DocumentRefusal,
    DocumentSpec,
    DocumentSpecParseError,
    build_system_prompt,
    parse_document_spec,
)
from app.services.document_guardrails import (
    MAX_DOCUMENT_BYTES,
    GuardrailError,
    apply_output_limits,
    check_request,
    safe_filename,
)
from app.services.document_render import MIME_TYPES, RENDERERS
from app.clients.pinecone_index import PineconeIndexClient
from app.services.feature_flags import HANDBOOK_RAG, is_enabled
from app.services.handbook_rag import (
    HandbookPassage,
    HandbookRetriever,
    build_handbook_prompt,
    retrieval_query,
)

router = APIRouter(prefix="/v1/ai", tags=["ai"])
settings = get_settings()
logger = logging.getLogger("lafina.ai")

LAFINA_SYSTEM_INSTRUCTION = (
    "You are LAFINA, an intelligent, warm, voice-first AI academic scheduling assistant for students. "
    "Always introduce yourself as LAFINA when asked, greeted, or introduced. "
    "Your primary function is helping students organize their academic life: scheduling classes, study sessions, midterm exams, "
    "assignments, reminders, and time blocks, while also answering academic and general questions. "
    "Rule 1: NEVER refuse, decline, or lecture a user about their requested schedule times (including 3 AM, late night, or early morning hours). Always accept and confirm any requested time, date, or duration without judgment. "
    "Rule 2: When a user asks you to schedule, remind, or plan a task, confirm warmly as LAFINA that the schedule item has been created with the exact title, date, time, and duration requested."
    "Rule 3: Do not use emojis in the chat"
)


def get_deepseek_client(request: Request) -> DeepSeekClient:
    """Dependency helper providing the application-scoped DeepSeekClient."""
    client: DeepSeekClient | None = getattr(request.app.state, "deepseek_client", None)
    if client is None:
        return DeepSeekClient(settings=settings)
    return client


def get_gemini_tts_client(request: Request) -> GeminiTtsClient:
    """Dependency helper providing the application-scoped GeminiTtsClient."""
    client: GeminiTtsClient | None = getattr(request.app.state, "gemini_tts_client", None)
    if client is None:
        return GeminiTtsClient(settings=settings)
    return client


def get_handbook_retriever(request: Request) -> HandbookRetriever | None:
    """Dependency helper providing the application-scoped Student Handbook retriever."""
    retriever: HandbookRetriever | None = getattr(request.app.state, "handbook_retriever", None)
    if retriever is None:
        return HandbookRetriever(settings, PineconeIndexClient(settings))
    return retriever


async def _handbook_passages(
    retriever: HandbookRetriever | None,
    db: AsyncSession,
    messages: list["ChatMessage"],
    request_id: str,
) -> list[HandbookPassage]:
    """Handbook passages for this question, or none.

    Nothing here can fail the chat: with the flag off, the index unconfigured,
    or Pinecone slow or down, the reply simply goes ahead without handbook
    context.
    """
    if retriever is None or not retriever.configured:
        return []
    if not await is_enabled(db, HANDBOOK_RAG):
        return []
    query = retrieval_query(messages)
    if not query:
        return []
    try:
        return await asyncio.wait_for(
            retriever.retrieve(query), timeout=settings.HANDBOOK_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning(
            f"Student Handbook lookup skipped ({type(exc).__name__}: {getattr(exc, 'message', exc)}) "
            f"[requestId={request_id}]"
        )
        return []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4090)


class AIChatRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=10)
    # The student's own switch for the Student Handbook. It can only turn the
    # handbook off: the admin panel's `handbook_rag` flag still has the final
    # say, and clients that never send it (mobile, older desktops) keep it on.
    useHandbook: bool = True


class HandbookSource(BaseModel):
    page: str
    pageEnd: str
    section: str
    score: float


class AIChatResponse(BaseModel):
    requestId: str
    reply: str
    model: str
    usage: dict
    createdAt: str
    # Student Handbook passages the reply drew on; empty when none were relevant.
    sources: list[HandbookSource] = Field(default_factory=list)


class AITtsRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, max_length=512)


class AITtsResponse(BaseModel):
    requestId: str
    audioBase64: str
    mimeType: str = "audio/wav"
    model: str
    voice: str
    createdAt: str


@router.post("/chat", response_model=AIChatResponse)
async def chat_proxy(
    req: AIChatRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
    handbook: HandbookRetriever | None = Depends(get_handbook_retriever),
):
    account, _ = auth_data
    owner_id = account.id

    # Enforce role-based access for Online AI from live DB Account
    if not has_online_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Online AI requires a student_pro subscription. Please upgrade your account."
        )

    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    # Validate total character count across messages (max 8,000 chars)
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > 8000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Total input messages character count exceeds limit of 8,000 characters."
        )

    # Rate limiting: Max 10 requests per minute for chat
    one_min_ago = now - timedelta(minutes=1)
    min_stmt = select(func.count(AIUsage.id)).where(
        AIUsage.owner_id == owner_id,
        AIUsage.request_type == "chat",
        AIUsage.created_at >= one_min_ago
    )
    min_count = (await db.execute(min_stmt)).scalar() or 0
    if min_count >= settings.MAX_AI_REQUESTS_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Per-minute AI request limit exceeded. Please wait a moment before trying again."
        )

    # Rate limiting: Max 100 requests per 24 hours for chat
    twenty_four_hrs_ago = now - timedelta(hours=24)
    day_stmt = select(func.count(AIUsage.id)).where(
        AIUsage.owner_id == owner_id,
        AIUsage.request_type == "chat",
        AIUsage.created_at >= twenty_four_hrs_ago
    )
    day_count = (await db.execute(day_stmt)).scalar() or 0
    if day_count >= settings.MAX_AI_REQUESTS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily AI request quota reached (100 requests/day)."
        )

    # Student Handbook passages ride along as a second system message, so they
    # are reference material the model weighs, never the student's own words.
    passages = (
        await _handbook_passages(handbook, db, req.messages, req.requestId)
        if req.useHandbook
        else []
    )
    handbook_context = (
        [{"role": "system", "content": build_handbook_prompt(passages, settings.HANDBOOK_TITLE)}]
        if passages
        else []
    )
    formatted_messages = (
        [{"role": "system", "content": LAFINA_SYSTEM_INSTRUCTION}]
        + handbook_context
        + [{"role": m.role, "content": m.content} for m in req.messages]
    )

    try:
        reply_text, usage_data = await deepseek.chat_completion(
            messages=formatted_messages,
            user_id=str(account.id),
            request_id=req.requestId
        )
    except DeepSeekError as err:
        raise HTTPException(
            status_code=err.status_code,
            detail=err.message
        )

    # Record AI usage only after successful completion
    db.add(AIUsage(
        owner_id=owner_id,
        request_type="chat",
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        created_at=now
    ))
    await db.commit()

    return AIChatResponse(
        requestId=req.requestId,
        reply=reply_text,
        model=settings.DEEPSEEK_MODEL,
        usage=usage_data,
        createdAt=now_str,
        sources=[HandbookSource(**passage.as_source()) for passage in passages],
    )


class HandbookStatusResponse(BaseModel):
    # The admin panel's `handbook_rag` switch. Off means the feature is not
    # offered at all, and clients hide everything about it.
    enabled: bool
    # Configured, reachable, and holding the handbook: answers will be grounded.
    functional: bool
    passages: int
    detail: str | None = None


@router.get("/handbook/status", response_model=HandbookStatusResponse)
async def handbook_status(
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    handbook: HandbookRetriever | None = Depends(get_handbook_retriever),
):
    """Whether the Student Handbook is switched on, and whether it works right now.

    Cheap to call: the Pinecone side is checked at most once a minute, and not
    at all while the switch is off.
    """
    if not await is_enabled(db, HANDBOOK_RAG):
        return HandbookStatusResponse(enabled=False, functional=False, passages=0)
    if handbook is None:
        return HandbookStatusResponse(
            enabled=True,
            functional=False,
            passages=0,
            detail="The Student Handbook index is not configured on the server.",
        )
    health = await handbook.health()
    return HandbookStatusResponse(
        enabled=True,
        functional=health.functional,
        passages=health.passages,
        detail=health.detail,
    )


@router.post("/tts", response_model=AITtsResponse)
async def tts_proxy(
    req: AITtsRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    gemini_tts: GeminiTtsClient = Depends(get_gemini_tts_client)
):
    account, _ = auth_data
    owner_id = account.id

    # Enforce access for TTS from live DB Account (student_pro only)
    if not has_tts_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gemini TTS requires a student_pro subscription. Please upgrade your account."
        )

    trimmed_text = req.text.strip()
    if not trimmed_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text field cannot be empty or whitespace only."
        )

    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    # Rate limiting: Max 10 TTS requests per minute
    one_min_ago = now - timedelta(minutes=1)
    min_stmt = select(func.count(AIUsage.id)).where(
        AIUsage.owner_id == owner_id,
        AIUsage.request_type == "tts",
        AIUsage.created_at >= one_min_ago
    )
    min_count = (await db.execute(min_stmt)).scalar() or 0
    if min_count >= settings.MAX_TTS_REQUESTS_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Per-minute TTS request limit exceeded. Please wait a moment before trying again."
        )

    # Rate limiting: Max 100 TTS requests per 24 hours
    twenty_four_hrs_ago = now - timedelta(hours=24)
    day_stmt = select(func.count(AIUsage.id)).where(
        AIUsage.owner_id == owner_id,
        AIUsage.request_type == "tts",
        AIUsage.created_at >= twenty_four_hrs_ago
    )
    day_count = (await db.execute(day_stmt)).scalar() or 0
    if day_count >= settings.MAX_TTS_REQUESTS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily TTS request quota reached (100 requests/day)."
        )

    try:
        audio_base64, usage_data = await gemini_tts.synthesize_speech(
            text=trimmed_text,
            request_id=req.requestId
        )
    except GeminiTtsError as err:
        raise HTTPException(
            status_code=err.status_code,
            detail=err.message
        )

    # Record AI usage only after successful completion
    db.add(AIUsage(
        owner_id=owner_id,
        request_type="tts",
        prompt_tokens=usage_data.get("prompt_tokens", len(trimmed_text)),
        completion_tokens=usage_data.get("completion_tokens", 0),
        created_at=now
    ))
    await db.commit()

    return AITtsResponse(
        requestId=req.requestId,
        audioBase64=audio_base64,
        mimeType="audio/wav",
        model=settings.GEMINI_TTS_MODEL,
        voice=settings.GEMINI_TTS_VOICE,
        createdAt=now_str
    )


# ── Flashcards ────────────────────────────────────────────────────────────
# A PDF is not something DeepSeek accepts, so the document is turned into text
# here and only the text is sent upstream. One upload becomes several model
# calls — one per chunk — which is why this endpoint has a budget of its own
# rather than sharing the chat allowance.

MAX_FLASHCARD_PDF_BYTES = 15 * 1024 * 1024
MAX_FLASHCARD_PAGES = 40
MAX_FLASHCARD_CHUNKS = 6
MAX_FLASHCARD_CONCURRENCY = 3
MAX_FLASHCARD_REQUESTS_PER_MIN = 3
MAX_FLASHCARD_REQUESTS_PER_DAY = 20
MAX_FLASHCARDS_PER_DECK = 200


class FlashcardItem(BaseModel):
    question: str
    answer: str


class AIFlashcardRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = Field(default="document.pdf", max_length=255)
    contentBase64: str = Field(..., min_length=16)
    maxCards: int = Field(default=40, ge=5, le=120)


class AIFlashcardResponse(BaseModel):
    requestId: str
    deckTitle: str
    cards: list[FlashcardItem]
    totalPages: int
    pagesRead: int
    ocrPages: list[int]
    chunkCount: int
    model: str
    usage: dict
    warnings: list[str]
    createdAt: str


async def _enforce_ai_quota(
    db: AsyncSession,
    owner_id,
    now: datetime,
    *,
    request_type: str,
    per_minute: int,
    per_day: int,
    wait_message: str,
    day_message: str,
) -> None:
    """Holds one kind of document job to its own allowance.

    Each upload costs several model calls, so these jobs are counted apart from
    the chat allowance and from each other.
    """
    for window, limit, message in (
        (timedelta(minutes=1), per_minute, wait_message),
        (timedelta(hours=24), per_day, day_message),
    ):
        stmt = select(func.count(AIUsage.id)).where(
            AIUsage.owner_id == owner_id,
            AIUsage.request_type == request_type,
            AIUsage.created_at >= now - window,
        )
        used = (await db.execute(stmt)).scalar() or 0
        if used >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=message,
            )


def _decode_upload(content_base64: str) -> bytes:
    """Turns the uploaded payload back into bytes, refusing what cannot work."""
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file could not be decoded.",
        )
    if len(data) > MAX_FLASHCARD_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"That file is larger than {MAX_FLASHCARD_PDF_BYTES // (1024 * 1024)} MB. "
                "Split it and try again."
            ),
        )
    return data


def _deck_title(filename: str) -> str:
    stem = (filename or "document.pdf").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return (stem[:80] or "Flashcards").strip()


@router.post("/flashcards", response_model=AIFlashcardResponse)
async def flashcards_from_pdf(
    req: AIFlashcardRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """Builds a flashcard deck from an uploaded PDF.

    The PDF arrives base64-encoded because the desktop app's cloud transport
    carries JSON only. Extraction and OCR are blocking work, so they run in a
    worker thread rather than on the event loop.
    """
    account, _ = auth_data

    if not has_online_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Flashcard generation requires a student_pro subscription. "
                "Please upgrade your account."
            ),
        )

    now = datetime.now(timezone.utc)
    await _enforce_ai_quota(
        db,
        account.id,
        now,
        request_type="flashcards",
        per_minute=MAX_FLASHCARD_REQUESTS_PER_MIN,
        per_day=MAX_FLASHCARD_REQUESTS_PER_DAY,
        wait_message="Give the last deck a moment to finish before generating another.",
        day_message=(
            f"Daily flashcard limit reached ({MAX_FLASHCARD_REQUESTS_PER_DAY} documents/day)."
        ),
    )

    pdf_bytes = _decode_upload(req.contentBase64)

    try:
        document = await asyncio.to_thread(
            extract_pdf_text, pdf_bytes, max_pages=MAX_FLASHCARD_PAGES
        )
    except PdfTextError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)

    chunks = chunk_text(document.text)
    warnings = list(document.warnings)
    if len(chunks) > MAX_FLASHCARD_CHUNKS:
        warnings.append(
            f"The document was long, so cards come from the first {MAX_FLASHCARD_CHUNKS} sections."
        )
        chunks = chunks[:MAX_FLASHCARD_CHUNKS]
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text was found in that PDF.",
        )

    deck_title = _deck_title(req.filename)
    per_chunk = max(6, min(40, -(-req.maxCards // len(chunks)) + 2))
    semaphore = asyncio.Semaphore(MAX_FLASHCARD_CONCURRENCY)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def run_chunk(index: int, chunk: str) -> list[Flashcard]:
        async with semaphore:
            reply, usage = await deepseek.json_completion(
                messages=[
                    {"role": "system", "content": FLASHCARD_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(chunk, per_chunk, deck_title)},
                ],
                user_id=str(account.id),
                request_id=f"{req.requestId}#{index}",
                model=settings.DEEPSEEK_FLASHCARD_MODEL,
            )
        for key in totals:
            totals[key] += usage.get(key, 0)
        return parse_flashcard_json(reply)

    results = await asyncio.gather(
        *(run_chunk(index, chunk) for index, chunk in enumerate(chunks)),
        return_exceptions=True,
    )

    batches: list[list[Flashcard]] = []
    failures: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(result)
        else:
            batches.append(result)

    if not batches:
        first = failures[0] if failures else None
        if isinstance(first, DeepSeekError):
            raise HTTPException(status_code=first.status_code, detail=first.message)
        if isinstance(first, FlashcardParseError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The model did not return usable flashcards. Try again.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Flashcards could not be generated from that document.",
        )

    if failures:
        # A partial deck is still worth having; say so rather than pretending
        # the document was fully covered.
        warnings.append(
            f"{len(failures)} of {len(chunks)} sections could not be processed and were skipped."
        )

    cards = merge_cards(batches, limit=min(MAX_FLASHCARDS_PER_DECK, req.maxCards))
    if not cards:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Nothing testable was found in that document.",
        )

    db.add(
        AIUsage(
            owner_id=account.id,
            request_type="flashcards",
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            created_at=now,
        )
    )
    await db.commit()

    return AIFlashcardResponse(
        requestId=req.requestId,
        deckTitle=deck_title,
        cards=[FlashcardItem(**card.as_dict()) for card in cards],
        totalPages=document.total_pages,
        pagesRead=document.pages_read,
        ocrPages=document.ocr_page_numbers,
        chunkCount=len(chunks),
        model=settings.DEEPSEEK_FLASHCARD_MODEL,
        usage=totals,
        warnings=warnings,
        createdAt=now.isoformat(),
    )


# ── Study notes ───────────────────────────────────────────────────────────
# The same shape as flashcards — upload, extract, summarise, merge — over Word
# and PowerPoint as well as PDF, because lecture material arrives in all three.

MAX_STUDY_NOTE_CHUNKS = 5
MAX_STUDY_NOTE_REQUESTS_PER_MIN = 3
MAX_STUDY_NOTE_REQUESTS_PER_DAY = 20


class StudyNoteSection(BaseModel):
    heading: str
    points: list[str]


class StudyNoteTerm(BaseModel):
    term: str
    meaning: str


class AIStudyNotesRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = Field(default="document.pdf", max_length=255)
    contentBase64: str = Field(..., min_length=16)


class AIStudyNotesResponse(BaseModel):
    requestId: str
    title: str
    overview: str
    sections: list[StudyNoteSection]
    keyTerms: list[StudyNoteTerm]
    markdown: str
    sourceKind: str
    totalPages: int
    pagesRead: int
    ocrPages: list[int]
    chunkCount: int
    model: str
    usage: dict
    warnings: list[str]
    createdAt: str


@router.post("/study-notes", response_model=AIStudyNotesResponse)
async def study_notes_from_document(
    req: AIStudyNotesRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """Summarises a PDF, Word or PowerPoint document into revision notes."""
    account, _ = auth_data

    if not has_online_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Study notes require a student_pro subscription. "
                "Please upgrade your account."
            ),
        )

    now = datetime.now(timezone.utc)
    await _enforce_ai_quota(
        db,
        account.id,
        now,
        request_type="study_notes",
        per_minute=MAX_STUDY_NOTE_REQUESTS_PER_MIN,
        per_day=MAX_STUDY_NOTE_REQUESTS_PER_DAY,
        wait_message="Give the last summary a moment to finish before starting another.",
        day_message=(
            f"Daily study-notes limit reached ({MAX_STUDY_NOTE_REQUESTS_PER_DAY} documents/day)."
        ),
    )

    file_bytes = _decode_upload(req.contentBase64)

    try:
        document, kind = await asyncio.to_thread(
            extract_document_text, file_bytes, req.filename, max_pages=MAX_FLASHCARD_PAGES
        )
    except PdfTextError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)

    chunks = chunk_text(document.text)
    warnings = list(document.warnings)
    if len(chunks) > MAX_STUDY_NOTE_CHUNKS:
        warnings.append(
            f"The document was long, so the notes cover its first {MAX_STUDY_NOTE_CHUNKS} sections."
        )
        chunks = chunks[:MAX_STUDY_NOTE_CHUNKS]
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text was found in that document.",
        )

    fallback_title = _deck_title(req.filename)
    semaphore = asyncio.Semaphore(MAX_FLASHCARD_CONCURRENCY)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def summarise(index: int, chunk: str) -> StudySummary:
        part = f"part {index + 1} of {len(chunks)}" if len(chunks) > 1 else ""
        async with semaphore:
            reply, usage = await deepseek.json_completion(
                messages=[
                    {"role": "system", "content": STUDY_NOTES_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_study_prompt(chunk, fallback_title, part),
                    },
                ],
                user_id=str(account.id),
                request_id=f"{req.requestId}#{index}",
                model=settings.DEEPSEEK_FLASHCARD_MODEL,
            )
        for key in totals:
            totals[key] += usage.get(key, 0)
        return parse_study_notes_json(reply)

    results = await asyncio.gather(
        *(summarise(index, chunk) for index, chunk in enumerate(chunks)),
        return_exceptions=True,
    )

    parts: list[StudySummary] = []
    failures: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(result)
        else:
            parts.append(result)

    if not parts:
        first = failures[0] if failures else None
        if isinstance(first, DeepSeekError):
            raise HTTPException(status_code=first.status_code, detail=first.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Study notes could not be generated from that document.",
        )
    if failures:
        warnings.append(
            f"{len(failures)} of {len(chunks)} sections could not be summarised and were skipped."
        )

    summary = merge_summaries(parts)
    if summary.is_empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="There was nothing worth summarising in that document.",
        )
    if not summary.title:
        summary.title = fallback_title

    db.add(
        AIUsage(
            owner_id=account.id,
            request_type="study_notes",
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            created_at=now,
        )
    )
    await db.commit()

    return AIStudyNotesResponse(
        requestId=req.requestId,
        title=summary.title,
        overview=summary.overview,
        sections=[StudyNoteSection(**section.as_dict()) for section in summary.sections],
        keyTerms=[StudyNoteTerm(**term.as_dict()) for term in summary.key_terms],
        markdown=to_markdown(summary),
        sourceKind=kind,
        totalPages=document.total_pages,
        pagesRead=document.pages_read,
        ocrPages=document.ocr_page_numbers,
        chunkCount=len(chunks),
        model=settings.DEEPSEEK_FLASHCARD_MODEL,
        usage=totals,
        warnings=warnings,
        createdAt=now.isoformat(),
    )


# ── Documents ─────────────────────────────────────────────────────────────
# A student asks the chat for a file. DeepSeek describes the document as JSON,
# the guardrails check and clip that description, and a fixed renderer builds
# the PDF, Word, Excel or PowerPoint file. The model never writes code and
# never touches a file itself. Student Pro only, not the wider paid-AI group.

MAX_DOCUMENT_REQUESTS_PER_MIN = 3
MAX_DOCUMENT_REQUESTS_PER_DAY = 20
MAX_DOCUMENT_TOKENS = 8192
MAX_DOCUMENT_ATTEMPTS = 2


class AIDocumentRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    format: DocumentFormat
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=10)


class AIDocumentResponse(BaseModel):
    requestId: str
    format: str
    title: str
    summary: str
    filename: str
    mimeType: str
    sizeBytes: int
    contentBase64: str
    model: str
    usage: dict
    warnings: list[str]
    createdAt: str


@router.post("/documents", response_model=AIDocumentResponse)
async def generate_document(
    req: AIDocumentRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """Writes a downloadable file from the chat conversation."""
    account, _ = auth_data

    if not is_student_pro(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "File generation is exclusive to student_pro accounts. "
                "Please upgrade your account."
            ),
        )

    try:
        check_request(req.format, req.messages)
    except GuardrailError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)

    now = datetime.now(timezone.utc)
    await _enforce_ai_quota(
        db,
        account.id,
        now,
        request_type="document",
        per_minute=MAX_DOCUMENT_REQUESTS_PER_MIN,
        per_day=MAX_DOCUMENT_REQUESTS_PER_DAY,
        wait_message="Give the last file a moment to finish before asking for another.",
        day_message=f"Daily file limit reached ({MAX_DOCUMENT_REQUESTS_PER_DAY} files/day).",
    )

    conversation: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(req.format, now.date().isoformat())},
        *({"role": m.role, "content": m.content} for m in req.messages),
    ]
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    spec: DocumentSpec | None = None
    model_answered = False

    async def record_usage() -> None:
        # Counted whenever the model answered, not only when a file came out: a
        # request that keeps being refused, or keeps failing to parse, still
        # spent tokens and must still use up the allowance.
        db.add(
            AIUsage(
                owner_id=account.id,
                request_type="document",
                prompt_tokens=totals["prompt_tokens"],
                completion_tokens=totals["completion_tokens"],
                created_at=now,
            )
        )
        await db.commit()

    for attempt in range(MAX_DOCUMENT_ATTEMPTS):
        try:
            reply, usage = await deepseek.json_completion(
                messages=conversation,
                user_id=str(account.id),
                request_id=f"{req.requestId}#{attempt}",
                model=settings.DEEPSEEK_MODEL,
                max_tokens=MAX_DOCUMENT_TOKENS,
                json_mode=True,
            )
        except DeepSeekMalformedResponseError:
            # JSON mode occasionally answers with nothing at all; ask again.
            continue
        except DeepSeekError as err:
            if model_answered:
                await record_usage()
            raise HTTPException(status_code=err.status_code, detail=err.message)

        model_answered = True
        for key in totals:
            totals[key] += usage.get(key, 0)
        try:
            spec = parse_document_spec(reply, req.format)
            break
        except DocumentRefusal as refusal:
            await record_usage()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"LAFINA can't create that file. {refusal.reason}",
            )
        except DocumentSpecParseError as err:
            logger.warning(f"Document spec rejected ({err}) [requestId={req.requestId}#{attempt}]")
            conversation = [
                *conversation,
                {"role": "assistant", "content": reply[:4000]},
                {"role": "user", "content": REPAIR_PROMPT},
            ]

    if spec is None:
        if model_answered:
            await record_usage()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model did not describe a usable file. Try again, or rephrase the request.",
        )

    warnings = apply_output_limits(spec, req.format)
    try:
        file_bytes = await asyncio.to_thread(RENDERERS[req.format], spec)
    except Exception:
        logger.exception(f"Rendering a {req.format} file failed [requestId={req.requestId}]")
        await record_usage()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The file could not be built from the model's content. Try again.",
        )

    await record_usage()
    if len(file_bytes) > MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="That file came out too large. Ask for a shorter document.",
        )

    return AIDocumentResponse(
        requestId=req.requestId,
        format=req.format,
        title=spec.title,
        summary=spec.summary or f"Here is your file: {spec.title}.",
        filename=safe_filename(spec.title, req.format),
        mimeType=MIME_TYPES[req.format],
        sizeBytes=len(file_bytes),
        contentBase64=base64.b64encode(file_bytes).decode("ascii"),
        model=settings.DEEPSEEK_MODEL,
        usage=totals,
        warnings=warnings,
        createdAt=now.isoformat(),
    )
