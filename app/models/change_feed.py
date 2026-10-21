import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, BigInteger, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class ChangeFeed(Base):
    __tablename__ = "change_feed"

    # SQLite only auto-generates values for an INTEGER PRIMARY KEY. The type
    # variant keeps production on a PostgreSQL BIGINT sequence while allowing
    # the test database to exercise the same database-generated ID path.
    change_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
