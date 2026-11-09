"""Rules for live voice conversations that do not need the database.

Grants: the server signs a short JWT naming one voice session and its owner.
The app hands it to ElevenLabs as a dynamic variable; ElevenLabs sends it
back on every custom-LLM call as the `X-Lafina-Voice-Grant` header (and, when
the agent forwards it, inside `elevenlabs_extra_body`). The custom-LLM route
trusts nothing else in the request body as identity.

Messages: ElevenLabs sends the whole conversation in OpenAI chat format,
including its own system prompt. Only user and assistant turns are kept,
recent ones first, within the configured message and character bounds, and
LAFINA's own system instructions are put in front.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import jwt

GRANT_AUDIENCE = "lafina-voice-llm"
GRANT_HEADER = "x-lafina-voice-grant"
CONVERSATION_HEADER = "x-lafina-conversation-id"
GRANT_FIELD = "lafina_voice_grant"

VOICE_STYLE_INSTRUCTION = (
    "This is a live spoken conversation: everything you write is read aloud. "
    "Answer in plain spoken sentences. Never use markdown, lists, headings, tables, links, code or emojis. "
    "Keep most replies to one to three short sentences unless the student asks for more detail. "
    "The student's words come from speech recognition: if they look cut off or unclear, ask briefly what they meant. "
    "When the student asks to add, change or delete a task, event, reminder or schedule, say that LAFINA shows "
    "a confirmation card on the screen when it understood the details, and that nothing is saved until they tap Confirm. "
    "Never say that anything was saved, scheduled, changed or deleted. "
    "When Student Handbook passages are provided, say the section and page you used. "
    "When none are provided, do not present university rules or policies as verified; say you could not check the handbook."
)


class VoiceGrantError(Exception):
    pass


def issue_grant(
    settings, *, session_id: uuid.UUID, owner_id: uuid.UUID, expires_at: datetime
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(owner_id),
        "vsid": str(session_id),
        "iss": settings.JWT_ISSUER,
        "aud": GRANT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.JWT_PRIVATE_KEY, algorithm="RS256")


def read_grant(settings, token: str) -> tuple[uuid.UUID, uuid.UUID]:
    """(session id, owner id) from a valid, unexpired grant."""
    if not token or len(token) > 4096:
        raise VoiceGrantError("missing grant")
    try:
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY,
            algorithms=["RS256"],
            issuer=settings.JWT_ISSUER,
            audience=GRANT_AUDIENCE,
        )
        return uuid.UUID(payload["vsid"]), uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise VoiceGrantError("invalid grant") from None


def _text_of(content: object) -> str:
    """OpenAI content is a string or a list of parts; only text is kept."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def conversation_turns(
    messages: object, *, max_messages: int, max_chars: int
) -> list[dict[str, str]]:
    """The recent user/assistant turns, oldest first, within both bounds.

    Ends with the student's latest words. An empty, noise-only or missing
    latest student turn returns an empty list, so no paid request is made.
    """
    if not isinstance(messages, list):
        return []
    turns: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            continue
        text = " ".join(_text_of(message.get("content")).split())
        if text:
            turns.append({"role": message["role"], "content": text})
    # Nothing new from the student (for example ElevenLabs re-engaging after a
    # long silence, when the history ends with LAFINA's own words): answering
    # would repeat the previous reply, so there is nothing to answer.
    if not turns or turns[-1]["role"] != "user":
        return []
    if not any(ch.isalnum() for ch in turns[-1]["content"]):
        return []
    kept: list[dict[str, str]] = []
    used = 0
    for turn in reversed(turns[-max_messages:]):
        room = max_chars - used
        if room <= 0:
            break
        content = turn["content"]
        if len(content) > room:
            if kept:
                break
            # The latest words are the ones that matter: keep their end.
            content = content[-room:]
        kept.append({"role": turn["role"], "content": content})
        used += len(content)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def turn_key(conversation_id: str | None, raw_message_count: int, last_user_text: str) -> str:
    """Same conversation, same history length, same words: the same turn."""
    digest = hashlib.sha256()
    digest.update((conversation_id or "-").encode())
    digest.update(b"\0")
    digest.update(str(raw_message_count).encode())
    digest.update(b"\0")
    digest.update(last_user_text.encode())
    return digest.hexdigest()


def completion_chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    text: str | None = None,
    finish: str | None = None,
) -> dict:
    """One OpenAI `chat.completion.chunk`, the shape ElevenLabs reads."""
    delta: dict[str, str] = {}
    if text is not None:
        delta = {"role": "assistant", "content": text}
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
