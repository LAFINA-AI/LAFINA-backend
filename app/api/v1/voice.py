"""Live voice conversation: ElevenAgents for speech and turns, DeepSeek for answers.

Flow:
1. The signed-in app calls ``POST /v1/ai/voice/sessions``. The server checks the
   account, configuration, voice and daily allowance, ends any earlier live
   session of that account, stores a new ``voice_sessions`` row and returns a
   signed ElevenLabs URL plus a grant naming that row.
2. The app opens the ElevenLabs conversation WebSocket with the signed URL and
   passes the grant as a dynamic variable. ElevenLabs does speech recognition,
   turn taking, interruptions and speech.
3. For each turn ElevenLabs calls ``POST /v1/ai/voice/llm/chat/completions``
   (OpenAI-compatible, SSE) with the shared secret as the bearer token and the
   grant in ``X-Lafina-Voice-Grant``. The route reserves the turn atomically
   on the session row, then streams DeepSeek's reply back.
4. The app ends the session with ``POST /v1/ai/voice/sessions/{id}/end``. A
   session also stops being served when it expires, is superseded or the
   account is disabled; ElevenLabs then gets errors instead of answers.

Cancelling a turn does not refund provider usage; late output is simply not
spoken by ElevenLabs once the student interrupts.
"""

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.ai import (
    ChatMessage,
    _handbook_passages,
    _usage_counts,
    get_deepseek_client,
    get_elevenlabs_tts_client,
    get_handbook_retriever,
    open_fresh_session,
)
from app.services.handbook_rag import HandbookRetriever, build_handbook_prompt
from app.clients.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekBillingError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekError,
    DeepSeekRateLimitError,
    DeepSeekTimeoutError,
)
from app.clients.elevenlabs_agents import ElevenLabsAgentClient, ElevenLabsAgentError
from app.clients.elevenlabs_tts import _FILIPINA_ORDER, ElevenLabsTtsClient, ElevenLabsTtsError
from app.config import get_settings
from app.database import get_db, set_transaction_rls_user
from app.models.account import Account
from app.models.ai_usage import AIUsage
from app.models.session import AuthSession
from app.models.voice_session import VoiceSession
from app.security.auth import get_current_user_and_session
from app.services.capabilities import has_chat_access
from app.services.voice_conversation import (
    CONVERSATION_HEADER,
    GRANT_FIELD,
    GRANT_HEADER,
    VOICE_STYLE_INSTRUCTION,
    VoiceGrantError,
    app_updates_prompt,
    approved_voice_tools,
    completion_chunk,
    conversation_turns,
    issue_grant,
    message_role_counts,
    pending_tool_result,
    read_grant,
    student_context_prompt,
    turn_key,
    voice_tool_continuation,
)
from app.services.voice_conversation import (
    VOICE_SYSTEM_INSTRUCTION as LAFINA_SYSTEM_INSTRUCTION,
)

router = APIRouter(prefix="/v1/ai/voice", tags=["voice"])
settings = get_settings()
logger = logging.getLogger("lafina.voice")

SESSION_USAGE = "voice_session"
TURN_USAGE = "voice_turn"
# A model call that finishes a student turn after a tool result (no new words).
TOOL_TURN_USAGE = "voice_tool_turn"
CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_CONVERSATION_ID_RE = re.compile(CONVERSATION_ID_PATTERN)
NO_STORE = {"Cache-Control": "no-store"}
MAX_VOICE_SESSIONS_PER_MIN = 12
MAX_REDELIVERIES_PER_TURN = 3
_FALLBACK_AGENT_CLIENT: ElevenLabsAgentClient | None = None
_TURN_REDELIVERIES: dict[tuple[str, str], tuple[float, int]] = {}


def get_elevenlabs_agent_client(request: Request) -> ElevenLabsAgentClient:
    global _FALLBACK_AGENT_CLIENT
    client: ElevenLabsAgentClient | None = getattr(
        request.app.state, "elevenlabs_agent_client", None
    )
    if client is None:
        if _FALLBACK_AGENT_CLIENT is None:
            _FALLBACK_AGENT_CLIENT = ElevenLabsAgentClient(settings=settings)
        return _FALLBACK_AGENT_CLIENT
    return client


def _record_redelivery_and_check_limit(session_id: uuid.UUID, key: str) -> bool:
    """Returns True when an unmetered redelivery is still within the per-turn retry cap."""
    now_mono = time.monotonic()
    if len(_TURN_REDELIVERIES) > 4096:
        cutoff = now_mono - 600.0
        for k in [k for k, (ts, _) in _TURN_REDELIVERIES.items() if ts < cutoff]:
            _TURN_REDELIVERIES.pop(k, None)
    cache_key = (str(session_id), key)
    ts, count = _TURN_REDELIVERIES.get(cache_key, (now_mono, 0))
    if now_mono - ts >= 600.0:
        ts, count = now_mono, 0
    if count >= MAX_REDELIVERIES_PER_TURN:
        return False
    _TURN_REDELIVERIES[cache_key] = (ts, count + 1)
    return True


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _require_access(account: Account) -> None:
    if not has_chat_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sign in with an active account to use voice conversation.",
        )


async def _owned_session(db: AsyncSession, account: Account, session_id: uuid.UUID) -> VoiceSession:
    await set_transaction_rls_user(db, account.id)
    row = await db.get(VoiceSession, session_id)
    if row is None or row.owner_id != account.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Voice conversation not found."
        )
    return row


class VoiceCapabilitiesResponse(BaseModel):
    provider: str = "elevenlabs"
    enabled: bool
    maxDurationSeconds: int
    # None means no limit.
    maxTurns: int | None
    sessionsPerDay: int | None
    detail: str | None = None
    # The agent accepts the app's opening line (first_message override).
    openingLine: bool = False
    # The agent and backend allowlist the on-device `search_ustp_sources` tool (ADR-047).
    clientUstpTool: bool = False


class VoiceSessionRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=64)
    voiceId: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )
    useHandbook: bool = True


class VoiceSessionResponse(BaseModel):
    provider: str = "elevenlabs"
    sessionId: str
    signedUrl: str
    grant: str
    voiceId: str
    expiresAt: str
    maxDurationSeconds: int
    maxTurns: int | None


class VoiceSessionConnected(BaseModel):
    conversationId: str = Field(..., pattern=CONVERSATION_ID_PATTERN)


class VoiceSessionEnd(BaseModel):
    conversationId: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)


class VoiceSessionStatus(BaseModel):
    sessionId: str
    status: str
    endedReason: str | None
    turns: int
    maxTurns: int | None
    expiresAt: str
    lastErrorCode: str | None
    lastErrorMessage: str | None


@router.get("/capabilities", response_model=VoiceCapabilitiesResponse)
async def voice_capabilities(
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    agents: ElevenLabsAgentClient = Depends(get_elevenlabs_agent_client),
):
    account, _ = auth_data
    _require_access(account)
    enabled = agents.configured
    return VoiceCapabilitiesResponse(
        enabled=enabled,
        maxDurationSeconds=settings.VOICE_CONVERSATION_MAX_SECONDS,
        maxTurns=settings.VOICE_CONVERSATION_MAX_TURNS,
        sessionsPerDay=settings.VOICE_CONVERSATION_SESSIONS_PER_DAY,
        detail=None if enabled else "Voice conversation is not set up on this server yet.",
        openingLine=enabled and settings.VOICE_OPENING_LINE_OVERRIDE,
        clientUstpTool=enabled,
    )


async def _validate_voice(tts: ElevenLabsTtsClient, voice_id: str | None) -> str:
    """The voice the session will use; an unavailable choice is refused, not replaced."""
    if voice_id is None:
        return settings.ELEVENLABS_VOICE_ID
    if voice_id in _FILIPINA_ORDER:
        return voice_id
    try:
        voices = await tts.list_voices()
    except ElevenLabsTtsError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    if not any(voice["id"] == voice_id for voice in voices):
        raise HTTPException(
            status_code=422,
            detail="Your saved online voice is no longer available. Choose another voice in Profile.",
        )
    return voice_id


@router.post("/sessions", response_model=VoiceSessionResponse)
async def create_voice_session(
    req: VoiceSessionRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    agents: ElevenLabsAgentClient = Depends(get_elevenlabs_agent_client),
    tts: ElevenLabsTtsClient = Depends(get_elevenlabs_tts_client),
):
    account, _ = auth_data
    _require_access(account)
    if not agents.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice conversation is not set up on this server yet.",
        )

    now = datetime.now(timezone.utc)
    daily_sessions = settings.VOICE_CONVERSATION_SESSIONS_PER_DAY
    started_min, started_today = await _usage_counts(db, account.id, SESSION_USAGE, now)
    if started_min >= MAX_VOICE_SESSIONS_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many voice calls started in one minute. Please wait a moment.",
        )
    if daily_sessions is not None and started_today >= daily_sessions:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily voice conversation limit reached ({daily_sessions} a day). "
                "Text chat and offline voice still work."
            ),
        )
    # Release read transaction before awaiting slow ElevenLabs HTTP calls so no
    # database lock or idle-in-transaction connection is held across network I/O.
    await db.commit()

    voice_id = await _validate_voice(tts, req.voiceId)
    try:
        signed_url = await agents.signed_url()
    except ElevenLabsAgentError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)

    # Re-acquire a brief per-account row lock to enforce quotas atomically and
    # supersede any earlier live conversation.
    await db.execute(select(Account.id).where(Account.id == account.id).with_for_update())
    now = datetime.now(timezone.utc)
    started_min, started_today = await _usage_counts(db, account.id, SESSION_USAGE, now)
    if started_min >= MAX_VOICE_SESSIONS_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many voice calls started in one minute. Please wait a moment.",
        )
    if daily_sessions is not None and started_today >= daily_sessions:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily voice conversation limit reached ({daily_sessions} a day). "
                "Text chat and offline voice still work."
            ),
        )

    # One live conversation per account: a new start ends the earlier one, so
    # its grant stops buying answers even if that phone is still connected.
    await set_transaction_rls_user(db, account.id)
    await db.execute(
        update(VoiceSession)
        .where(VoiceSession.owner_id == account.id, VoiceSession.status == "active")
        .values(status="ended", ended_reason="superseded", ended_at=now)
    )
    expires_at = now + timedelta(
        seconds=settings.VOICE_CONVERSATION_MAX_SECONDS
        + settings.VOICE_CONVERSATION_START_GRACE_SECONDS
    )
    row = VoiceSession(
        owner_id=account.id,
        status="active",
        voice_id=voice_id,
        use_handbook=req.useHandbook,
        created_at=now,
        expires_at=expires_at,
    )
    db.add(row)
    db.add(AIUsage(owner_id=account.id, request_type=SESSION_USAGE, created_at=now))
    await db.commit()

    grant = issue_grant(settings, session_id=row.id, owner_id=account.id, expires_at=expires_at)
    body = VoiceSessionResponse(
        sessionId=str(row.id),
        signedUrl=signed_url,
        grant=grant,
        voiceId=voice_id,
        expiresAt=expires_at.isoformat(),
        maxDurationSeconds=settings.VOICE_CONVERSATION_MAX_SECONDS,
        maxTurns=settings.VOICE_CONVERSATION_MAX_TURNS,
    )
    return JSONResponse(body.model_dump(), headers=NO_STORE)


def _bind_conversation(row: VoiceSession, conversation_id: str | None) -> None:
    if not conversation_id:
        return
    if row.conversation_id is None:
        row.conversation_id = conversation_id
    elif row.conversation_id != conversation_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This voice session belongs to a different conversation.",
        )


@router.post("/sessions/{session_id}/connected", response_model=VoiceSessionStatus)
async def voice_session_connected(
    session_id: uuid.UUID,
    req: VoiceSessionConnected,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
):
    account, _ = auth_data
    _require_access(account)
    row = await _owned_session(db, account, session_id)
    if row.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This voice conversation has ended."
        )
    _bind_conversation(row, req.conversationId)
    await db.commit()
    return _status_of(row)


@router.post("/sessions/{session_id}/end", response_model=VoiceSessionStatus)
async def end_voice_session(
    session_id: uuid.UUID,
    req: VoiceSessionEnd,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
):
    """Idempotent. Ending never touches the student's reminders or tasks."""
    account, _ = auth_data
    row = await _owned_session(db, account, session_id)
    if row.status == "active":
        row.status = "ended"
        row.ended_reason = "client"
        row.ended_at = datetime.now(timezone.utc)
        if req.conversationId and row.conversation_id is None:
            row.conversation_id = req.conversationId
        await db.commit()
    return _status_of(row)


@router.get("/sessions/{session_id}", response_model=VoiceSessionStatus)
async def voice_session_status(
    session_id: uuid.UUID,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
):
    account, _ = auth_data
    row = await _owned_session(db, account, session_id)
    return _status_of(row)


def _status_of(row: VoiceSession) -> VoiceSessionStatus:
    return VoiceSessionStatus(
        sessionId=str(row.id),
        status=row.status,
        endedReason=row.ended_reason,
        turns=row.turns,
        maxTurns=settings.VOICE_CONVERSATION_MAX_TURNS,
        expiresAt=_utc(row.expires_at).isoformat(),
        lastErrorCode=row.last_error_code,
        lastErrorMessage=row.last_error_message,
    )


# ── Custom LLM (called by ElevenLabs, not by the app) ───────────────────


def _llm_error(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code}},
        headers=NO_STORE,
    )


def _provider_error_code(err: DeepSeekError) -> str:
    if isinstance(err, DeepSeekBillingError):
        return "provider_billing"
    if isinstance(err, DeepSeekRateLimitError):
        return "provider_rate_limited"
    if isinstance(err, DeepSeekTimeoutError):
        return "provider_timeout"
    if isinstance(err, (DeepSeekConfigError, DeepSeekAuthenticationError)):
        return "provider_config"
    return "provider_unavailable"


_STUDENT_MESSAGES = {
    "provider_billing": "LAFINA's online answers are paused because the response service account needs attention.",
    "provider_rate_limited": "The response service is busy. Wait a moment, then try again.",
    "provider_timeout": "The response service took too long to answer.",
    "provider_config": "The response service is not configured correctly on the server.",
    "provider_unavailable": "The response service is unavailable right now.",
    "turn_limit": "This conversation reached its turn limit. End it and start a new one.",
    "rate_limited": "Too many turns in a short time. Wait a moment.",
    "daily_limit": "Daily voice conversation limit reached. Text chat and offline voice still work.",
}


async def _record_error(db: AsyncSession, row: VoiceSession, code: str) -> None:
    try:
        await set_transaction_rls_user(db, row.owner_id)
        await db.execute(
            update(VoiceSession)
            .where(VoiceSession.id == row.id)
            .values(
                last_error_code=code, last_error_message=_STUDENT_MESSAGES.get(code, code)[:300]
            )
        )
        await db.commit()
    except Exception as exc:  # Never let bookkeeping hide the original failure.
        logger.warning(
            f"Voice error could not be saved ({type(exc).__name__}) [voiceSession={row.id}]"
        )
        await db.rollback()


def _shared_secret_ok(request: Request) -> bool:
    secret = settings.ELEVENLABS_AGENT_LLM_SECRET
    expected = secret.get_secret_value().strip() if secret else ""
    if not expected:
        return False
    supplied = request.headers.get("authorization", "")
    return hmac.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _should_use_voice_handbook(text: str) -> bool:
    """Only policy questions pay the retrieval latency on the voice path."""
    if re.search(
        r"\b(?:add|create|schedule|remind|book|move|rename|delete|cancel|set)\b", text, re.I
    ) and not re.search(
        r"\b(?:policy|policies|handbook|allowed|rules|requirements|grading|attendance|absences|retention|probation|scholarship)\b",
        text,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:handbook|ustp|policy|policies|regulations|dress code|grading|attendance|absences|retention|scholarship|latin honors|dean.s list|uniform|plagiarism|probation|graduation requirements|academic rules)\b",
            text,
            re.I,
        )
    )


@router.post("/llm/chat/completions")
async def voice_llm_turn(
    request: Request,
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
    handbook: HandbookRetriever | None = Depends(get_handbook_retriever),
):
    """OpenAI-compatible streaming completion for one conversation turn."""
    if not _shared_secret_ok(request):
        return _llm_error(401, "Unauthorized.", "unauthorized")
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _llm_error(400, "Request body must be JSON.", "invalid_request")
    if not isinstance(body, dict):
        return _llm_error(400, "Request body must be a JSON object.", "invalid_request")

    extra = body.get("elevenlabs_extra_body")
    grant = request.headers.get(GRANT_HEADER) or (
        extra.get(GRANT_FIELD)
        if isinstance(extra, dict) and isinstance(extra.get(GRANT_FIELD), str)
        else ""
    )
    try:
        session_id, owner_id = read_grant(settings, grant)
    except VoiceGrantError:
        return _llm_error(401, "This conversation is not authorized.", "invalid_grant")

    await set_transaction_rls_user(db, owner_id)
    row = await db.get(VoiceSession, session_id)
    if row is None or row.owner_id != owner_id:
        return _llm_error(401, "This conversation is not authorized.", "invalid_grant")
    account = await db.get(Account, owner_id)
    now = datetime.now(timezone.utc)
    if account is None or not has_chat_access(account):
        if row.status == "active":
            row.status, row.ended_reason, row.ended_at = "ended", "account", now
            await db.commit()
        return _llm_error(403, "This account cannot use voice conversation.", "account_disabled")
    if row.status != "active":
        return _llm_error(409, "This voice conversation has ended.", "session_ended")
    if _utc(row.expires_at) <= now:
        row.status, row.ended_reason, row.ended_at = "ended", "expired", now
        await db.commit()
        return _llm_error(409, "This voice conversation has expired.", "session_expired")

    conversation_id = request.headers.get(CONVERSATION_HEADER) or None
    if conversation_id is not None and not _CONVERSATION_ID_RE.fullmatch(conversation_id):
        return _llm_error(
            400, "Invalid conversation identifier.", "invalid_conversation_id"
        )
    if conversation_id and row.conversation_id and row.conversation_id != conversation_id:
        return _llm_error(
            403, "This voice session belongs to a different conversation.", "conversation_mismatch"
        )

    raw_messages = body.get("messages")
    try:
        approved_tools = approved_voice_tools(body.get("tools"))
    except ValueError:
        return _llm_error(400, "Unsupported voice tool.", "invalid_tools")
    allowed_names = {t["function"]["name"] for t in approved_tools}
    continuation = voice_tool_continuation(
        raw_messages,
        allowed_names=allowed_names,
        max_messages=settings.VOICE_TOOL_EXCHANGE_MAX_MESSAGES,
        max_chars=settings.VOICE_TOOL_EXCHANGE_MAX_CHARS,
    )
    model = settings.DEEPSEEK_MODEL
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def say(text: str) -> StreamingResponse:
        return _spoken_response(completion_id, created, model, text)

    # From here on a live call never gets an error status or an empty answer.
    # ElevenLabs treats either as a failed generation, retries the same
    # request, and ends the whole call when the retries fail too. The student
    # hears one short sentence instead, and the call goes on.
    if not continuation and pending_tool_result(raw_messages, allowed_names):
        # Re-answering the student's words here would run the same tool again.
        logger.warning(
            f"Voice tool result could not be read: {message_role_counts(raw_messages)} "
            f"[voiceSession={row.id}]"
        )
        return say(TOOL_RESULT_FALLBACK)
    if continuation:
        # Preserve prior spoken turns ahead of the active tool exchange so
        # multi-turn tool calls retain context from earlier in the call.
        prior_turns: list[dict] = []
        if isinstance(raw_messages, list):
            last_user_idx = next(
                (
                    i
                    for i in range(len(raw_messages) - 1, -1, -1)
                    if isinstance(raw_messages[i], dict)
                    and raw_messages[i].get("role") == "user"
                ),
                -1,
            )
            if last_user_idx > 0:
                earlier = [
                    m
                    for m in raw_messages[:last_user_idx]
                    if isinstance(m, dict)
                    and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)
                    and m.get("content").strip()
                ]
                room_msgs = max(
                    0, settings.VOICE_TOOL_EXCHANGE_MAX_MESSAGES - len(continuation)
                )
                room_chars = max(
                    0,
                    settings.VOICE_TOOL_EXCHANGE_MAX_CHARS
                    - len(json.dumps(continuation, ensure_ascii=False)),
                )
                used_chars = 0
                for msg in reversed(earlier[-min(room_msgs, settings.VOICE_CONVERSATION_MAX_CONTEXT_MESSAGES) :]):
                    text_val = " ".join(msg["content"].split())
                    if not text_val or used_chars + len(text_val) > room_chars:
                        break
                    prior_turns.append({"role": msg["role"], "content": text_val})
                    used_chars += len(text_val)
                prior_turns.reverse()
                while prior_turns and prior_turns[0]["role"] != "user":
                    prior_turns.pop(0)
        turns = prior_turns + continuation
    else:
        turns = conversation_turns(
            raw_messages,
            max_messages=settings.VOICE_CONVERSATION_MAX_CONTEXT_MESSAGES,
            max_chars=settings.VOICE_CONVERSATION_MAX_CONTEXT_CHARS,
        )
    if not turns:
        # Nothing new to answer: ElevenLabs re-engaging after a silence (the
        # history ends with LAFINA's words), or noise. Nothing is paid.
        return say(SILENCE_CHECK_IN if _lafina_spoke_last(raw_messages) else NOT_HEARD)

    # Key by stable session_id (not mutable conversation_id) so binding
    # conversation_id mid-retry never double-charges the same turn.
    key = turn_key(
        str(row.id),
        len(raw_messages) if isinstance(raw_messages, list) else 0,
    )
    redelivery = row.last_turn_key == key
    usage_row: AIUsage | None = None
    if not redelivery:
        if continuation:
            tool_min, _ = await _usage_counts(db, owner_id, TOOL_TURN_USAGE, now)
            if tool_min >= settings.VOICE_CONVERSATION_TURNS_PER_MIN * 2:
                await _record_error(db, row, "rate_limited")
                return say(_STUDENT_MESSAGES["rate_limited"])
        else:
            minute_turns, day_turns = await _usage_counts(db, owner_id, TURN_USAGE, now)
            if minute_turns >= settings.VOICE_CONVERSATION_TURNS_PER_MIN:
                await _record_error(db, row, "rate_limited")
                return say(_STUDENT_MESSAGES["rate_limited"])
            day_cap = settings.VOICE_CONVERSATION_TURNS_PER_DAY
            if day_cap is not None and day_turns >= day_cap:
                await _record_error(db, row, "daily_limit")
                return say(_STUDENT_MESSAGES["daily_limit"])
        # Reserve the turn atomically before paying for it: within budget,
        # still active, and not already reserved by a parallel request.
        conditions = [
            VoiceSession.id == session_id,
            VoiceSession.status == "active",
            or_(VoiceSession.last_turn_key.is_(None), VoiceSession.last_turn_key != key),
        ]
        max_turns = settings.VOICE_CONVERSATION_MAX_TURNS
        if not continuation and max_turns is not None:
            conditions.append(VoiceSession.turns < max_turns)
        await set_transaction_rls_user(db, owner_id)
        reserved = await db.execute(
            update(VoiceSession)
            .where(*conditions)
            .values(
                turns=VoiceSession.turns + (0 if continuation else 1),
                last_turn_key=key,
                conversation_id=func.coalesce(VoiceSession.conversation_id, conversation_id),
                last_error_code=None,
                last_error_message=None,
            )
            .execution_options(synchronize_session=False)
        )
        if reserved.rowcount == 1:
            usage_row = AIUsage(
                owner_id=owner_id,
                request_type=TOOL_TURN_USAGE if continuation else TURN_USAGE,
                created_at=now,
            )
            db.add(usage_row)
            await db.commit()
        else:
            await db.rollback()
            await set_transaction_rls_user(db, owner_id)
            await db.refresh(row)
            if row.last_turn_key == key:
                redelivery = True  # a parallel request reserved this same turn
            elif row.status != "active":
                return _llm_error(409, "This voice conversation has ended.", "session_ended")
            else:
                await _record_error(db, row, "turn_limit")
                return say(_STUDENT_MESSAGES["turn_limit"])
    if redelivery:
        if not _record_redelivery_and_check_limit(session_id, key):
            await _record_error(db, row, "rate_limited")
            return say(_STUDENT_MESSAGES["rate_limited"])
        logger.info(
            f"Voice turn asked again; answering without a new charge [voiceSession={row.id}]"
        )

    request_id = f"voice-{session_id}-{uuid.uuid4().hex[:12]}"
    last_user = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
    use_handbook = (
        row.use_handbook
        and not continuation
        and _should_use_voice_handbook(last_user)
    )
    # Commit/close the pre-stream DB transaction before streaming starts.
    await db.commit()
    student_note = student_context_prompt(
        extra, settings.VOICE_CONVERSATION_MAX_STUDENT_CONTEXT_CHARS
    )
    updates_note = app_updates_prompt(raw_messages)
    logger.info(f"Voice turn messages: {message_role_counts(raw_messages)} [{request_id}]")

    usage_row_id = usage_row.id if usage_row is not None else None

    async def save_tokens(usage: dict) -> None:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        async with open_fresh_session(db) as bg_db:
            await set_transaction_rls_user(bg_db, owner_id)
            if usage_row_id is not None:
                await bg_db.execute(
                    update(AIUsage)
                    .where(AIUsage.id == usage_row_id, AIUsage.owner_id == owner_id)
                    .values(prompt_tokens=prompt, completion_tokens=completion)
                    .execution_options(synchronize_session=False)
                )
            await bg_db.execute(
                update(VoiceSession)
                .where(VoiceSession.id == session_id)
                .values(
                    prompt_tokens=VoiceSession.prompt_tokens + prompt,
                    completion_tokens=VoiceSession.completion_tokens + completion,
                )
                .execution_options(synchronize_session=False)
            )
            await bg_db.commit()

    async def record_stream_error(code: str) -> None:
        async with open_fresh_session(db) as bg_db:
            await set_transaction_rls_user(bg_db, owner_id)
            await bg_db.execute(
                update(VoiceSession)
                .where(VoiceSession.id == session_id)
                .values(
                    last_error_code=code,
                    last_error_message=_STUDENT_MESSAGES.get(code, _STUDENT_MESSAGES["provider_unavailable"]),
                )
                .execution_options(synchronize_session=False)
            )
            await bg_db.commit()

    async def settle(work) -> None:
        try:
            await asyncio.shield(asyncio.ensure_future(work))
        except Exception as exc:
            logger.warning(
                f"Voice bookkeeping could not be saved ({type(exc).__name__}) [{request_id}]"
            )

    def chunk(**fields) -> str:
        return _sse(completion_chunk(completion_id, created, model, **fields))

    async def stream():
        # The first bytes go out at once, before retrieval and DeepSeek, so
        # ElevenLabs sees the answer start well inside its cascade timeout.
        yield chunk(text="")
        passages = []
        if use_handbook:
            retrieval_at = time.monotonic()
            try:
                async with open_fresh_session(db) as rag_db:
                    passages = await asyncio.wait_for(
                        _handbook_passages(
                            handbook,
                            rag_db,
                            [ChatMessage(role="user", content=last_user[:4090])],
                            request_id,
                        ),
                        timeout=min(settings.HANDBOOK_TIMEOUT_SECONDS, 2.0),
                    )
            except Exception:
                logger.info("Voice handbook budget exceeded or failed [%s]", request_id)
            logger.info(
                "Voice retrieval duration=%dms used=%s [%s]",
                int((time.monotonic() - retrieval_at) * 1000),
                bool(passages),
                request_id,
            )
        formatted = (
            [
                {"role": "system", "content": LAFINA_SYSTEM_INSTRUCTION},
                {"role": "system", "content": VOICE_STYLE_INSTRUCTION},
            ]
            + ([{"role": "system", "content": student_note}] if student_note else [])
            + ([{"role": "system", "content": updates_note}] if updates_note else [])
            + (
                [
                    {
                        "role": "system",
                        "content": build_handbook_prompt(passages, settings.HANDBOOK_TITLE),
                    }
                ]
                if passages
                else []
            )
            + turns
        )
        events = deepseek.chat_completion_stream(
            messages=formatted,
            user_id=str(owner_id),
            request_id=request_id,
            max_tokens=settings.VOICE_CONVERSATION_MAX_REPLY_TOKENS,
            tools=approved_tools or None,
            sanitize_for_speech=True,
        )
        calls_seen = False
        spoken_any = False
        try:
            first_at = time.monotonic()
            first_deadline = first_at + settings.VOICE_FIRST_WORDS_TIMEOUT_SECONDS
            try:
                pending = await asyncio.wait_for(
                    anext(events), timeout=settings.VOICE_FIRST_WORDS_TIMEOUT_SECONDS
                )
            except (DeepSeekError, TimeoutError, StopAsyncIteration) as err:
                failure = (
                    err
                    if isinstance(err, DeepSeekError)
                    else (
                        DeepSeekTimeoutError()
                        if isinstance(err, TimeoutError)
                        else DeepSeekError("Empty model stream.")
                    )
                )
                code = _provider_error_code(failure)
                logger.warning(f"Voice turn failed before the first words ({code}) [{request_id}]")
                await settle(record_stream_error(code))
                yield chunk(
                    text=TRANSIENT_RETRY
                    if code in {"provider_timeout", "provider_unavailable"}
                    else _STUDENT_MESSAGES[code]
                )
                yield chunk(finish="stop")
                yield "data: [DONE]\n\n"
                return
            logger.info(
                "Voice first model event=%dms [%s]",
                int((time.monotonic() - first_at) * 1000),
                request_id,
            )
            while True:
                kind, value = pending
                if kind == "delta":
                    spoken_any = True
                    yield chunk(text=value)
                elif kind == "started":
                    pass
                elif kind == "tool_calls":
                    calls_seen = True
                    piece = completion_chunk(completion_id, created, model)
                    piece["choices"][0]["delta"] = {"tool_calls": value}
                    yield _sse(piece)
                else:
                    yield chunk(finish="tool_calls" if calls_seen else "stop")
                    yield "data: [DONE]\n\n"
                    await settle(save_tokens(value if isinstance(value, dict) else {}))
                    return
                # Before the first audible words or tool call arrive, keep the
                # overall first-words deadline so `("started", None)` cannot
                # double the silence window past ElevenLabs' cascade timeout.
                if not spoken_any and not calls_seen:
                    remaining = max(0.05, first_deadline - time.monotonic())
                    next_timeout = min(12.0, remaining)
                else:
                    next_timeout = 12.0
                pending = await asyncio.wait_for(anext(events), timeout=next_timeout)
        except StopAsyncIteration:
            yield chunk(finish="tool_calls" if calls_seen else "stop")
            yield "data: [DONE]\n\n"
        except (DeepSeekError, TimeoutError) as err:
            if isinstance(err, TimeoutError):
                err = DeepSeekTimeoutError()
            code = _provider_error_code(err)
            logger.warning(f"Voice turn stopped part-way ({code}) [{request_id}]")
            await settle(record_stream_error(code))
            # Words, never an error object: an error ends the call.
            fallback_text = (
                MIDSTREAM_APOLOGY
                if spoken_any
                else (
                    TRANSIENT_RETRY
                    if code in {"provider_timeout", "provider_unavailable"}
                    else _STUDENT_MESSAGES[code]
                )
            )
            yield chunk(text=fallback_text)
            yield chunk(finish="stop")
            yield "data: [DONE]\n\n"
        finally:
            await events.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={**NO_STORE, "X-Accel-Buffering": "no"},
    )


SILENCE_CHECK_IN = "I'm still here whenever you're ready."
NOT_HEARD = "Sorry, I didn't catch that. Could you say it again?"
TOOL_RESULT_FALLBACK = "I couldn't finish that step. Please check your screen, or say it again."
TRANSIENT_RETRY = (
    "I couldn't finish that response. Please try again, or check the card for any action result."
)
MIDSTREAM_APOLOGY = " Sorry, I lost my train of thought. Could you say that again?"


def _lafina_spoke_last(messages: object) -> bool:
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") in ("user", "assistant"):
            return message.get("role") == "assistant"
    return False


def _spoken_response(completion_id: str, created: int, model: str, text: str) -> StreamingResponse:
    """A complete, successful answer of one short sentence."""

    async def body():
        yield _sse(completion_chunk(completion_id, created, model, text=text))
        yield _sse(completion_chunk(completion_id, created, model, finish="stop"))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={**NO_STORE, "X-Accel-Buffering": "no"}
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
