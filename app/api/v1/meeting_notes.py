"""Meeting notes from a locally produced transcript.

Two endpoints, because long meetings are summarised in stages and the desktop
app drives the stages:

* ``/section`` turns one slice of a long transcript (or of earlier section
  notes) into partial notes. Each call is short, so a two-hour meeting never
  runs into a proxy timeout, and the app can show real progress between them.
* ``/generate`` produces the final notes — straight from the transcript for an
  ordinary meeting, or by consolidating section notes for a long one.

No audio is accepted here, only text. The DeepSeek key never leaves the server.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.ai import (
    _enforce_ai_quota,
    get_deepseek_client,
)
from app.clients.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekBillingError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekError,
    DeepSeekInvalidRequestError,
    DeepSeekMalformedResponseError,
    DeepSeekProviderServerError,
    DeepSeekRateLimitError,
    DeepSeekTimeoutError,
    DeepSeekTransportError,
)
from app.config import get_settings
from app.database import get_db
from app.models.account import Account
from app.models.ai_usage import AIUsage
from app.models.session import AuthSession
from app.security.auth import get_current_user_and_session
from app.services.capabilities import has_online_ai_access
from app.services.meeting_notes import (
    CONSOLIDATE_SYSTEM_PROMPT,
    MAX_FINAL_CHARS,
    MAX_SECTION_CHARS,
    MEETING_NOTES_SYSTEM_PROMPT,
    SECTION_SYSTEM_PROMPT,
    MeetingNotes,
    MeetingNotesParseError,
    build_final_prompt,
    build_section_prompt,
    looks_like_context_overflow,
    parse_meeting_notes_json,
)

logger = logging.getLogger("lafina.meeting_notes")
router = APIRouter(prefix="/v1/ai/meeting-notes", tags=["meeting-notes"])
settings = get_settings()

# The final pass is the one that counts as "generating notes" for a meeting;
# sections are the working steps of one generation and get their own, far
# looser allowance so a long meeting is never refused halfway through.
MAX_NOTES_PER_MIN = 5
MAX_NOTES_PER_DAY = 40
MAX_SECTIONS_PER_MIN = 40
MAX_SECTIONS_PER_DAY = 800

# One quiet retry for failures that are usually momentary. A rate limit is not
# retried here: hammering a limited API makes it worse, so the app backs off.
TRANSIENT_ERRORS = (DeepSeekTimeoutError, DeepSeekTransportError, DeepSeekProviderServerError)
RETRY_DELAY_SECONDS = 1.5


class ActionItemOut(BaseModel):
    task: str
    assignee: str
    deadline: str
    status: str


class KeyTopicOut(BaseModel):
    topic: str
    discussion: str


class MeetingNotesOut(BaseModel):
    title: str
    summary: str
    key_topics: list[KeyTopicOut]
    decisions: list[str]
    action_items: list[ActionItemOut]
    important_dates: list[str]
    issues: list[str]
    unresolved_questions: list[str]
    key_points: list[str]


class SectionRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1)
    index: int = Field(..., ge=0)
    count: int = Field(..., ge=1, le=500)
    material: Literal["transcript", "notes"] = "transcript"


class GenerateRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1)
    source: Literal["transcript", "sections"] = "transcript"
    titleHint: str = Field(default="", max_length=160)
    recordedAt: str = Field(default="", max_length=64)


class NotesResponse(BaseModel):
    requestId: str
    notes: MeetingNotesOut
    model: str
    usage: dict
    createdAt: str


def _to_out(notes: MeetingNotes) -> MeetingNotesOut:
    return MeetingNotesOut(**notes.as_dict())


def _http_error(err: DeepSeekError) -> HTTPException:
    """Maps a DeepSeek failure to the status and sentence the app acts on."""
    if isinstance(err, (DeepSeekConfigError, DeepSeekAuthenticationError)):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Meeting notes are not configured on the server: its DeepSeek API key is missing or "
                "invalid. Your transcript is saved, and notes can be generated once the key is fixed."
            ),
        )
    if isinstance(err, DeepSeekBillingError):
        # Not a key problem: telling an administrator to replace a working key sends them the wrong way.
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Meeting notes are paused: the server's DeepSeek account is out of credit. Your "
                "transcript is saved, and notes can be generated once the account is topped up."
            ),
        )
    if isinstance(err, DeepSeekRateLimitError):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="DeepSeek is limiting requests right now. Wait a minute, then try again.",
        )
    if isinstance(err, DeepSeekTimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="DeepSeek took too long to answer. Try again; your transcript is saved.",
        )
    if isinstance(err, DeepSeekInvalidRequestError) and looks_like_context_overflow(err.message):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="This part of the meeting was too long for DeepSeek to read in one go.",
        )
    if isinstance(err, DeepSeekMalformedResponseError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="DeepSeek returned a reply that could not be read. Try again.",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="DeepSeek is unavailable right now. Try again in a few minutes; your transcript is saved.",
    )


async def _complete_with_retry(
    deepseek: DeepSeekClient,
    system_prompt: str,
    user_prompt: str,
    account: Account,
    request_id: str,
) -> tuple[MeetingNotes, dict]:
    """Asks for notes, retrying once on a momentary failure or an unreadable reply."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            reply, usage = await deepseek.json_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                user_id=str(account.id),
                request_id=f"{request_id}#{attempt}",
                model=settings.DEEPSEEK_FLASHCARD_MODEL,
                max_tokens=4096,
            )
            return parse_meeting_notes_json(reply), usage
        except TRANSIENT_ERRORS as err:
            last_error = err
        except MeetingNotesParseError as err:
            # A malformed reply is often a one-off; a second is reported.
            last_error = err
        except DeepSeekError as err:
            raise _http_error(err)
        if attempt == 0:
            logger.warning("Meeting notes attempt failed, retrying once: %s", last_error)
            await asyncio.sleep(RETRY_DELAY_SECONDS)

    if isinstance(last_error, MeetingNotesParseError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="DeepSeek returned notes that could not be read. Try generating them again.",
        )
    assert isinstance(last_error, DeepSeekError)
    raise _http_error(last_error)


def _require_entitlement(account: Account) -> None:
    if not has_online_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "AI meeting notes require a student_pro subscription. "
                "Your recording and transcript stay available on this computer."
            ),
        )


def _record_usage(db: AsyncSession, account: Account, request_type: str, usage: dict, now: datetime) -> None:
    db.add(
        AIUsage(
            owner_id=account.id,
            request_type=request_type,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            created_at=now,
        )
    )


@router.post("/section", response_model=NotesResponse)
async def summarise_section(
    req: SectionRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """Partial notes for one section of a long meeting."""
    account, _ = auth_data
    _require_entitlement(account)

    if len(req.content) > MAX_SECTION_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"A section can be at most {MAX_SECTION_CHARS:,} characters. Split it further.",
        )
    if req.index >= req.count:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Section index is out of range.")

    now = datetime.now(timezone.utc)
    await _enforce_ai_quota(
        db,
        account.id,
        now,
        request_type="meeting_notes_section",
        per_minute=MAX_SECTIONS_PER_MIN,
        per_day=MAX_SECTIONS_PER_DAY,
        wait_message="Too many parts of meetings are being summarised at once. Wait a minute, then try again.",
        day_message="Today's limit for summarising meetings has been reached. Try again tomorrow.",
    )

    notes, usage = await _complete_with_retry(
        deepseek,
        SECTION_SYSTEM_PROMPT,
        build_section_prompt(req.content, req.index, req.count, req.material),
        account,
        req.requestId,
    )
    _record_usage(db, account, "meeting_notes_section", usage, now)
    await db.commit()

    return NotesResponse(
        requestId=req.requestId,
        notes=_to_out(notes),
        model=settings.DEEPSEEK_FLASHCARD_MODEL,
        usage=usage,
        createdAt=now.isoformat(),
    )


@router.post("/generate", response_model=NotesResponse)
async def generate_notes(
    req: GenerateRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """The final notes for a meeting."""
    account, _ = auth_data
    _require_entitlement(account)

    if len(req.content) > MAX_FINAL_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"The meeting material is longer than {MAX_FINAL_CHARS:,} characters. "
                "Summarise it in sections first."
            ),
        )

    now = datetime.now(timezone.utc)
    await _enforce_ai_quota(
        db,
        account.id,
        now,
        request_type="meeting_notes",
        per_minute=MAX_NOTES_PER_MIN,
        per_day=MAX_NOTES_PER_DAY,
        wait_message="Notes are being generated for several meetings at once. Wait a minute, then try again.",
        day_message=f"Today's limit of {MAX_NOTES_PER_DAY} meeting notes has been reached. Try again tomorrow.",
    )

    system_prompt = MEETING_NOTES_SYSTEM_PROMPT if req.source == "transcript" else CONSOLIDATE_SYSTEM_PROMPT
    notes, usage = await _complete_with_retry(
        deepseek,
        system_prompt,
        build_final_prompt(req.content, req.source, req.titleHint, req.recordedAt),
        account,
        req.requestId,
    )
    if notes.is_empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "DeepSeek found nothing to take notes on. The recording may be silent, or the "
                "transcript may be too garbled — check the Transcript tab."
            ),
        )

    _record_usage(db, account, "meeting_notes", usage, now)
    await db.commit()

    return NotesResponse(
        requestId=req.requestId,
        notes=_to_out(notes),
        model=settings.DEEPSEEK_FLASHCARD_MODEL,
        usage=usage,
        createdAt=now.isoformat(),
    )
