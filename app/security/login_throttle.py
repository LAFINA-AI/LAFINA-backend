"""Slowing down password guessing at sign-in.

Every failed sign-in is written to ``security_events``, keyed by a hash of the
email it was for, whether or not that account exists: the limit then behaves
the same for real and made-up addresses, so it cannot be used to find out who
has an account. The raw email is never stored.

Once an email has ``MAX_LOGIN_FAILURES_PER_15MIN`` failures inside the window,
sign-in for it is refused until the oldest of them ages out, even with the
right password. Otherwise the limit would only slow a guesser down, not stop
one. The count is kept in the database rather than in memory so it survives a
restart or a redeploy and holds across server instances.
"""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.ai_usage import SecurityEvent

settings = get_settings()

LOGIN_FAILED_EVENT = "login_failed"
WINDOW = timedelta(minutes=15)


def email_key(normalized_email: str) -> str:
    """The key failures are counted under: a hash, so the log holds no addresses."""
    digest = hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()
    return f"email-sha256:{digest}"


def _as_utc(moment: datetime) -> datetime:
    # SQLite, used by the test suite, hands back naive datetimes.
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def seconds_until_allowed(db: AsyncSession, key: str, now: datetime) -> int | None:
    """None while sign-in is allowed; otherwise how long until it is again."""
    limit = settings.MAX_LOGIN_FAILURES_PER_15MIN
    stmt = (
        select(SecurityEvent.created_at)
        .where(
            SecurityEvent.event_type == LOGIN_FAILED_EVENT,
            SecurityEvent.details == key,
            SecurityEvent.created_at >= now - WINDOW,
        )
        .order_by(SecurityEvent.created_at.desc())
        .limit(limit)
    )
    recent = (await db.execute(stmt)).scalars().all()
    if len(recent) < limit:
        return None
    # Allowed again once the limit-th most recent failure leaves the window.
    reopens_at = _as_utc(recent[-1]) + WINDOW
    return max(1, int((reopens_at - now).total_seconds()) + 1)


async def record_failure(
    db: AsyncSession,
    key: str,
    owner_id: uuid.UUID | None,
    ip_address: str | None,
) -> None:
    """Writes the failure and commits it, since the request is about to fail and roll back."""
    db.add(
        SecurityEvent(
            event_type=LOGIN_FAILED_EVENT,
            owner_id=owner_id,
            ip_address=(ip_address or None) and ip_address[:64],
            details=key,
        )
    )
    await db.commit()
