import uuid
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from app.models.account import Account
from app.security.auth import hash_password
from tests.conftest import TestingSessionLocal


@pytest.mark.asyncio
async def test_meeting_summary_generation_entitled(async_client: AsyncClient):
    """Verifies that a Student Pro user can request an AI meeting summary via DeepSeek."""
    user_id = uuid.uuid4()

    async with TestingSessionLocal() as db:
        pro_user = Account(
            id=user_id,
            email="pro_meeting@ustp.edu.ph",
            password_hash=hash_password("Password123!"),
            role="student_pro",
            system_role="user",
            subscription_plan="student_pro",
            is_active=True,
        )
        db.add(pro_user)
        await db.commit()

    login_res = await async_client.post(
        "/v1/auth/login",
        json={
            "email": "pro_meeting@ustp.edu.ph",
            "password": "Password123!",
        },
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    fake_deepseek_reply = {
        "reply": (
            "{\n"
            '  "key_points": ["Discussed IoT firmware v2.1 timeline", "Calibration team ready"],\n'
            '  "decisions": ["Ship firmware by Friday"],\n'
            '  "open_questions": ["Is hardware buffer size sufficient?"],\n'
            '  "action_items": [\n'
            "    {\n"
            '      "task": "Calibrate oscilloscopes",\n'
            '      "assignee": "Alice",\n'
            '      "due": "2026-08-28T17:00:00Z",\n'
            '      "context": "Needs zero jitter"\n'
            "    }\n"
            "  ]\n"
            "}"
        ),
        "model": "deepseek-chat",
    }

    with patch(
        "app.clients.deepseek.DeepSeekClient.chat_completion",
        return_value=fake_deepseek_reply,
    ):
        res = await async_client.post(
            "/v1/meetings/summary",
            json={
                "transcript": "Dr. Vance: Welcome team. Let's calibrate oscilloscopes by Friday.",
                "meetingTitle": "Firmware Sync",
            },
            headers=headers,
        )

        assert res.status_code == 200
        data = res.json()
        assert "summary" in data
        assert len(data["summary"]["key_points"]) == 2
        assert data["summary"]["decisions"] == ["Ship firmware by Friday"]
        assert len(data["summary"]["action_items"]) == 1
        assert data["summary"]["action_items"][0]["assignee"] == "Alice"


@pytest.mark.asyncio
async def test_meeting_summary_rejects_free_student_account(async_client: AsyncClient):
    """Free student accounts must receive 403 when calling `/v1/meetings/summary`."""
    async with TestingSessionLocal() as db:
        student_user = Account(
            id=uuid.uuid4(),
            email="free_meeting@ustp.edu.ph",
            password_hash=hash_password("Password123!"),
            role="student",
            system_role="user",
            subscription_plan="student",
            is_active=True,
        )
        db.add(student_user)
        await db.commit()

    login_res = await async_client.post(
        "/v1/auth/login",
        json={
            "email": "free_meeting@ustp.edu.ph",
            "password": "Password123!",
        },
    )
    assert login_res.status_code == 200
    headers = {"Authorization": f"Bearer {login_res.json()['access_token']}"}

    res = await async_client.post(
        "/v1/meetings/summary",
        json={"transcript": "Lecture check-in notes."},
        headers=headers,
    )
    assert res.status_code == 403
    assert "Student Pro" in res.json()["detail"]
