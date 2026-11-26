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
RECOVER_FAILED_EVENT = "recover_failed"
PASSWORD_CHANGE_FAILED_EVENT = "password_change_failed"
REGISTER_IP_EVENT = "register_ip"
WINDOW = timedelta(minutes=15)
REGISTER_WINDOW = timedelta(hours=1)


def email_key(normalized_email: str) -> str:
    """The key failures are counted under: a hash, so the log holds no addresses."""
    digest = hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()
    return f"email-sha256:{digest}"


def ip_key(ip_address: str) -> str:
    """Hashed IP key for registration throttling without storing raw IPs in details."""
    digest = hashlib.sha256(ip_address.strip().encode("utf-8")).hexdigest()
    return f"ip-sha256:{digest}"


def _as_utc(moment: datetime) -> datetime:
    # SQLite, used by the test suite, hands back naive datetimes.
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def _retry_after_for_query(
    db: AsyncSession,
    *,
    event_type: str,
    key: str,
    now: datetime,
    limit: int,
    window: timedelta,
    ip_address: str | None = None,
) -> int | None:
    conditions = [
        SecurityEvent.event_type == event_type,
        SecurityEvent.details == key,
        SecurityEvent.created_at >= now - window,
    ]
    if ip_address is not None:
        conditions.append(SecurityEvent.ip_address == ip_address[:64])
    stmt = (
        select(SecurityEvent.created_at)
        .where(*conditions)
        .order_by(SecurityEvent.created_at.desc())
        .limit(limit)
    )
    recent = (await db.execute(stmt)).scalars().all()
    if len(recent) < limit:
        return None
    reopens_at = _as_utc(recent[-1]) + window
    return max(1, int((reopens_at - now).total_seconds()) + 1)


async def seconds_until_allowed(
    db: AsyncSession,
    key: str,
    now: datetime,
    *,
    event_type: str = LOGIN_FAILED_EVENT,
    limit: int | None = None,
    window: timedelta = WINDOW,
    ip_address: str | None = None,
) -> int | None:
    """None while the action is allowed; otherwise how long until it is again."""
    effective_limit = limit if limit is not None else settings.MAX_LOGIN_FAILURES_PER_15MIN
    if event_type == LOGIN_FAILED_EVENT and ip_address:
        # Per-(email, IP) throttle stops brute force from one host without
        # letting a remote attacker easily lock out the real user's IP.
        ip_wait = await _retry_after_for_query(
            db,
            event_type=event_type,
            key=key,
            now=now,
            limit=effective_limit,
            window=window,
            ip_address=ip_address,
        )
        if ip_wait is not None:
            return ip_wait
        # Higher distributed ceiling across all IPs still stops botnets.
        return await _retry_after_for_query(
            db,
            event_type=event_type,
            key=key,
            now=now,
            limit=effective_limit * 3,
            window=window,
        )
    return await _retry_after_for_query(
        db,
        event_type=event_type,
        key=key,
        now=now,
        limit=effective_limit,
        window=window,
    )


async def record_failure(
    db: AsyncSession,
    key: str,
    owner_id: uuid.UUID | None,
    ip_address: str | None,
    *,
    event_type: str = LOGIN_FAILED_EVENT,
) -> None:
    """Writes the event and commits it, since a failing request is about to roll back."""
    db.add(
        SecurityEvent(
            event_type=event_type,
            owner_id=owner_id,
            ip_address=(ip_address or None) and ip_address[:64],
            details=key,
        )
    )
    await db.commit()
