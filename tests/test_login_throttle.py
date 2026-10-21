"""Sign-in cannot be used to guess passwords, or to find out who has an account.

The phone app signs in to FastAPI to restore an account it does not have yet,
which makes this endpoint the one thing standing between a stolen email
address and an account. These tests hold it to that.
"""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.config import get_settings
from app.models.ai_usage import SecurityEvent
from app.security import login_throttle
from tests.conftest import TestingSessionLocal

settings = get_settings()
LIMIT = settings.MAX_LOGIN_FAILURES_PER_15MIN
PASSWORD = "correct-horse-battery-staple"


async def _register(client: AsyncClient, email: str) -> None:
    res = await client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert res.status_code == 201


async def _login(client: AsyncClient, email: str, password: str):
    return await client.post("/v1/auth/login", json={"email": email, "password": password})


@pytest.mark.asyncio
async def test_repeated_wrong_passwords_lock_the_email_even_for_the_right_one(async_client: AsyncClient):
    await _register(async_client, "target@ustp.edu.ph")

    for attempt in range(LIMIT):
        res = await _login(async_client, "target@ustp.edu.ph", f"wrong-guess-{attempt}")
        assert res.status_code == 401, attempt

    locked = await _login(async_client, "Target@USTP.edu.ph", PASSWORD)
    assert locked.status_code == 429
    assert "Too many sign-in attempts" in locked.json()["detail"]
    retry_after = int(locked.headers["Retry-After"])
    assert 0 < retry_after <= 15 * 60 + 1

    # Another account is not affected by someone guessing at this one.
    await _register(async_client, "bystander@ustp.edu.ph")
    assert (await _login(async_client, "bystander@ustp.edu.ph", PASSWORD)).status_code == 200


@pytest.mark.asyncio
async def test_an_unknown_email_is_answered_exactly_like_a_real_one(async_client: AsyncClient):
    await _register(async_client, "real@ustp.edu.ph")

    real = await _login(async_client, "real@ustp.edu.ph", "not-the-password")
    unknown = await _login(async_client, "nobody@ustp.edu.ph", "not-the-password")
    assert (real.status_code, real.json()) == (unknown.status_code, unknown.json())

    for attempt in range(LIMIT - 1):
        await _login(async_client, "real@ustp.edu.ph", f"guess-{attempt}")
        await _login(async_client, "nobody@ustp.edu.ph", f"guess-{attempt}")

    real_locked = await _login(async_client, "real@ustp.edu.ph", "another-guess")
    unknown_locked = await _login(async_client, "nobody@ustp.edu.ph", "another-guess")
    assert real_locked.status_code == unknown_locked.status_code == 429
    assert real_locked.json() == unknown_locked.json()


@pytest.mark.asyncio
async def test_failures_age_out_of_the_window(async_client: AsyncClient):
    await _register(async_client, "patient@ustp.edu.ph")
    key = login_throttle.email_key("patient@ustp.edu.ph")
    long_ago = datetime.now(timezone.utc) - timedelta(minutes=16)
    async with TestingSessionLocal() as session:
        for _ in range(LIMIT):
            session.add(SecurityEvent(event_type=login_throttle.LOGIN_FAILED_EVENT, details=key, created_at=long_ago))
        await session.commit()

    assert (await _login(async_client, "patient@ustp.edu.ph", PASSWORD)).status_code == 200


@pytest.mark.asyncio
async def test_the_failure_log_holds_no_email_addresses(async_client: AsyncClient):
    await _login(async_client, "private.person@ustp.edu.ph", "some-guess")

    async with TestingSessionLocal() as session:
        events = (await session.execute(select(SecurityEvent))).scalars().all()
    assert len(events) == 1
    assert events[0].event_type == "login_failed"
    assert events[0].owner_id is None
    assert "private.person" not in (events[0].details or "")
    assert events[0].details == login_throttle.email_key("private.person@ustp.edu.ph")


@pytest.mark.asyncio
async def test_a_successful_sign_in_is_not_counted(async_client: AsyncClient):
    await _register(async_client, "regular@ustp.edu.ph")
    for _ in range(LIMIT + 2):
        assert (await _login(async_client, "regular@ustp.edu.ph", PASSWORD)).status_code == 200
