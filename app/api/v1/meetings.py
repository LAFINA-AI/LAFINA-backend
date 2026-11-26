import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.ai import finalize_ai_usage, release_ai_usage, reserve_ai_usage
from app.config import get_settings
from app.database import get_db
from app.models.account import Account
from app.models.session import AuthSession
from app.security.auth import get_current_user_and_session
from app.services.capabilities import has_study_ai_access
from app.clients.deepseek import DeepSeekClient, DeepSeekError

router = APIRouter(tags=["meetings"])
settings = get_settings()
logger = logging.getLogger("lafina.meetings")

MAX_MEETING_SUMMARY_PER_MIN = 5
MAX_MEETING_SUMMARY_PER_DAY = 40
MAX_MEETING_TRANSCRIPT_CHARS = 120_000

MEETING_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert meeting transcription analyzer and executive summarizer. "
    "Analyze the provided meeting transcript and extract a clear, concise, structured JSON summary. "
    "You MUST respond ONLY with valid JSON conforming to the following schema without markdown backticks:\n"
    "{\n"
    '  "key_points": ["string", ...],\n'
    '  "decisions": ["string", ...],\n'
    '  "open_questions": ["string", ...],\n'
    '  "action_items": [\n'
    '    {\n'
    '      "task": "string",\n'
    '      "assignee": "string or Unassigned",\n'
    '      "due": "ISO 8601 date string or null",\n'
    '      "context": "string"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Ensure all key technical and business commitments, decisions, and deadlines mentioned in the meeting are accurately captured."
)


def get_deepseek_client(request: Request) -> DeepSeekClient:
    client: DeepSeekClient | None = getattr(request.app.state, "deepseek_client", None)
    if client is None:
        return DeepSeekClient(settings=settings)
    return client


# --- Schemas ---

class MeetingActionItem(BaseModel):
    task: str
    assignee: Optional[str] = "Unassigned"
    due: Optional[str] = None
    context: Optional[str] = None


class MeetingSummaryData(BaseModel):
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    action_items: list[MeetingActionItem] = Field(default_factory=list)


class MeetingSummaryRequest(BaseModel):
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript: str = Field(..., min_length=1, max_length=MAX_MEETING_TRANSCRIPT_CHARS)
    meetingTitle: Optional[str] = Field(default=None, max_length=200)


class MeetingSummaryResponse(BaseModel):
    requestId: str
    summary: MeetingSummaryData
    model: str
    createdAt: str


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()]


def _parse_summary_json(content: str) -> MeetingSummaryData:
    cleaned = (content or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        raw_actions = data.get("action_items")
        action_items: list[MeetingActionItem] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if isinstance(item, dict):
                    action_items.append(
                        MeetingActionItem(
                            task=str(item.get("task") or "Untitled Task"),
                            assignee=str(item.get("assignee") or "Unassigned"),
                            due=str(item["due"]) if item.get("due") is not None else None,
                            context=str(item["context"]) if item.get("context") is not None else None,
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    action_items.append(MeetingActionItem(task=item.strip()))
        return MeetingSummaryData(
            key_points=_string_items(data.get("key_points")),
            decisions=_string_items(data.get("decisions")),
            open_questions=_string_items(data.get("open_questions")),
            action_items=action_items,
        )
    except Exception:
        # Fallback text parsing
        return MeetingSummaryData(
            key_points=[cleaned[:300]] if cleaned else [],
            decisions=[],
            open_questions=[],
            action_items=[],
        )


# --- Endpoints ---

@router.post("/v1/meetings/summary", response_model=MeetingSummaryResponse)
async def generate_meeting_summary(
    req: MeetingSummaryRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    db: AsyncSession = Depends(get_db),
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """
    Generates structured AI summary (key points, decisions, open questions, action items)
    using DeepSeek. Zero audio upload; transcript text sent strictly with user consent.
    No prompt logging or retention on the server.
    """
    account, _ = auth_data
    if not has_study_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sign in with an active account to summarize meetings.",
        )

    trimmed_transcript = req.transcript.strip()
    if not trimmed_transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript cannot be empty or whitespace only.",
        )

    now = datetime.now(timezone.utc)
    usage_row = await reserve_ai_usage(
        db,
        account.id,
        now,
        request_type="meeting_summary",
        per_minute=MAX_MEETING_SUMMARY_PER_MIN,
        per_day=MAX_MEETING_SUMMARY_PER_DAY,
        wait_message="Too many meeting summary requests. Wait a minute, then try again.",
        day_message=f"Daily meeting summary limit reached ({MAX_MEETING_SUMMARY_PER_DAY} summaries/day).",
    )

    user_content = (
        f"Meeting Title: {(req.meetingTitle or 'General Team Sync').strip()}\n\n"
        f"Transcript:\n{trimmed_transcript}"
    )
    messages = [
        {"role": "system", "content": MEETING_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        reply_text, usage_data = await deepseek.chat_completion(
            messages=messages,
            user_id=str(account.id),
            request_id=req.requestId,
        )
    except DeepSeekError as e:
        await release_ai_usage(db, account.id, usage_row.id)
        raise HTTPException(
            status_code=e.status_code,
            detail=f"DeepSeek AI summarization failed: {e.message}",
        )
    except Exception:
        await release_ai_usage(db, account.id, usage_row.id)
        logger.exception("Unexpected error during meeting summarization [requestId=%s]", req.requestId)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Meeting summarization failed unexpectedly. Please try again.",
        )

    await finalize_ai_usage(
        db,
        account.id,
        usage_row.id,
        usage_data.get("prompt_tokens", 0) if isinstance(usage_data, dict) else 0,
        usage_data.get("completion_tokens", 0) if isinstance(usage_data, dict) else 0,
    )
    summary_data = _parse_summary_json(reply_text)
    return MeetingSummaryResponse(
        requestId=req.requestId,
        summary=summary_data,
        model=settings.DEEPSEEK_MODEL,
        createdAt=now.isoformat(),
    )
