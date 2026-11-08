import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class VoiceSession(Base):
    """One live voice conversation the server agreed to pay for.

    The app gets a signed ElevenLabs URL and a grant naming this row. Every
    turn ElevenLabs asks the custom-LLM route for must carry that grant, and
    the row decides whether the turn is still allowed: it belongs to an active
    account, has not ended or expired, and is within its turn budget.
    """

    __tablename__ = "voice_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # "active" | "ended"
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    # Why it ended: "client", "superseded", "expired", "account".
    ended_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    voice_id: Mapped[str] = mapped_column(String(128), nullable=False)
    use_handbook: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # ElevenLabs conversation this session was bound to, on first sight.
    conversation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    turns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Identifies the last turn so a repeated callback is not paid twice.
    last_turn_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Last provider failure, so the app can explain a silent turn.
    last_error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error_message: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
