import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import get_settings
from app.models.account import Account
from app.models.session import AuthSession
from app.security.auth import get_current_user_and_session
from app.services.capabilities import has_online_ai_access
from app.clients.deepseek import DeepSeekClient, DeepSeekError

router = APIRouter(tags=["meetings"])
settings = get_settings()

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
    transcript: str = Field(..., min_length=1)
    meetingTitle: Optional[str] = None


class MeetingSummaryResponse(BaseModel):
    requestId: str
    summary: MeetingSummaryData
    model: str
    createdAt: str


def _parse_summary_json(content: str) -> MeetingSummaryData:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return MeetingSummaryData(
            key_points=data.get("key_points", []),
            decisions=data.get("decisions", []),
            open_questions=data.get("open_questions", []),
            action_items=[
                MeetingActionItem(
                    task=item.get("task", "Untitled Task"),
                    assignee=item.get("assignee", "Unassigned"),
                    due=item.get("due"),
                    context=item.get("context"),
                )
                for item in data.get("action_items", [])
            ],
        )
    except Exception:
        # Fallback text parsing
        return MeetingSummaryData(
            key_points=[cleaned[:300]],
            decisions=[],
            open_questions=[],
            action_items=[],
        )


# --- Endpoints ---

@router.post("/v1/meetings/summary", response_model=MeetingSummaryResponse)
async def generate_meeting_summary(
    req: MeetingSummaryRequest,
    auth_data: Annotated[tuple[Account, AuthSession], Depends(get_current_user_and_session)],
    deepseek: DeepSeekClient = Depends(get_deepseek_client),
):
    """
    Generates structured AI summary (key points, decisions, open questions, action items)
    using DeepSeek-V4 Flash. Zero audio upload; transcript text sent strictly with user consent.
    No prompt logging or retention on the server.
    """
    account, _ = auth_data
    if not has_online_ai_access(account):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AI meeting summarization requires an active Student Pro plan.",
        )

    user_content = f"Meeting Title: {req.meetingTitle or 'General Team Sync'}\n\nTranscript:\n{req.transcript}"
    messages = [
        {"role": "system", "content": MEETING_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        res = await deepseek.chat_completion(messages=messages, max_tokens=1500)
        summary_data = _parse_summary_json(res.get("reply", "{}"))
        return MeetingSummaryResponse(
            requestId=req.requestId,
            summary=summary_data,
            model=res.get("model", "deepseek-chat"),
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
    except DeepSeekError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"DeepSeek AI summarization failed: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal summarization error: {str(e)}",
        )
