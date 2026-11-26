import uuid
from datetime import datetime, timezone
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, declared_attr, mapped_column
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class _SyncEntityMixin:
    @declared_attr.directive
    def __table_args__(cls):
        return (Index(f"ix_{cls.__tablename__}_owner_change_id", "owner_id", "change_id"),)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    client_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[int] = mapped_column(BigInteger, default=1, nullable=False)
    change_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProfileSync(_SyncEntityMixin, Base):
    __tablename__ = "profile_sync"


class TasksSync(_SyncEntityMixin, Base):
    __tablename__ = "tasks_sync"


class EventsSync(_SyncEntityMixin, Base):
    __tablename__ = "events_sync"


class TimeBlocksSync(_SyncEntityMixin, Base):
    __tablename__ = "time_blocks_sync"


class RemindersSync(_SyncEntityMixin, Base):
    __tablename__ = "reminders_sync"


class NotesSync(_SyncEntityMixin, Base):
    __tablename__ = "notes_sync"


class CustomCategoriesSync(_SyncEntityMixin, Base):
    __tablename__ = "custom_categories_sync"


class PomodoroSettingsSync(_SyncEntityMixin, Base):
    __tablename__ = "pomodoro_settings_sync"


class PomodoroSessionsSync(_SyncEntityMixin, Base):
    __tablename__ = "pomodoro_sessions_sync"


class FlashcardDecksSync(_SyncEntityMixin, Base):
    __tablename__ = "flashcard_decks_sync"


class StudySummariesSync(_SyncEntityMixin, Base):
    __tablename__ = "study_summaries_sync"


class RecordedMeetingsSync(_SyncEntityMixin, Base):
    __tablename__ = "recorded_meetings_sync"


class ChatSessionsSync(_SyncEntityMixin, Base):
    """One chat conversation. Its messages live in `chat_messages_sync`."""

    __tablename__ = "chat_sessions_sync"


class ChatMessagesSync(_SyncEntityMixin, Base):
    """One chat message. `payload.session_id` names its `chat_sessions_sync` row."""

    __tablename__ = "chat_messages_sync"
