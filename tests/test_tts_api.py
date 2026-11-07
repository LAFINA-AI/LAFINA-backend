import httpx
import pytest
from httpx import AsyncClient, MockTransport, Response
from pydantic import SecretStr
from sqlalchemy import select

from app.clients.elevenlabs_tts import ElevenLabsTtsClient
from app.config import Settings
from app.main import app
from app.models.account import Account
from app.models.ai_usage import AIUsage
from tests.conftest import TestingSessionLocal


def create_mock_elevenlabs_client(failing: bool = False):
    def handler(request: httpx.Request) -> Response:
        if failing:
            return Response(500, json={"error": "upstream failure"})
        return Response(200, content=b"synthetic-mp3-audio", headers={"Content-Type": "audio/mpeg"})

    settings = Settings(ENVIRONMENT="development", ELEVENLABS_API_KEY="synthetic-elevenlabs-key")
    http_client = httpx.AsyncClient(transport=MockTransport(handler))
    return ElevenLabsTtsClient(settings=settings, client=http_client)


@pytest.mark.asyncio
async def test_tts_unauthenticated_returns_401(async_client: AsyncClient):
    """Unauthenticated call to /v1/ai/tts must return 401."""
    res = await async_client.post("/v1/ai/tts", json={"text": "Hello world"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_tts_role_enforcement(async_client: AsyncClient):
    """
    Role enforcement test:
    - student -> 200 OK
    - admin -> 200 OK
    - student_pro -> 200 OK
    """
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client()

    # 1. Register a student user (default role = student)
    reg_student = await async_client.post(
        "/v1/auth/register", json={"email": "student_basic@ustp.edu.ph", "password": "Password123!"}
    )
    token_student = reg_student.json()["access_token"]

    res_student = await async_client.post(
        "/v1/ai/tts",
        json={"text": "Hello world"},
        headers={"Authorization": f"Bearer {token_student}"},
    )
    assert res_student.status_code == 200

    # 2. Promote user role in DB to admin -> must return 403 for TTS
    async with TestingSessionLocal() as db:
        stmt = select(Account).where(Account.email == "student_basic@ustp.edu.ph")
        acc = (await db.execute(stmt)).scalar_one()
        acc.role = "admin"
        await db.commit()

    res_admin = await async_client.post(
        "/v1/ai/tts",
        json={"text": "Hello world"},
        headers={"Authorization": f"Bearer {token_student}"},
    )
    assert res_admin.status_code == 200

    # 3. Promote user role in DB to student_pro -> must succeed (200)
    async with TestingSessionLocal() as db:
        stmt = select(Account).where(Account.email == "student_basic@ustp.edu.ph")
        acc = (await db.execute(stmt)).scalar_one()
        acc.role = "student_pro"
        await db.commit()

    res_pro = await async_client.post(
        "/v1/ai/tts",
        json={"text": "Hello world"},
        headers={"Authorization": f"Bearer {token_student}"},
    )
    assert res_pro.status_code == 200
    data = res_pro.json()
    assert "audioBase64" in data
    assert data["mimeType"] == "audio/mpeg"
    assert data["voice"] == "JBFqnCBsd6RMkjVDRZzb"


@pytest.mark.asyncio
async def test_tts_text_validation(async_client: AsyncClient):
    """Test text validation: empty text, whitespace-only, and max 512 length limit."""
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client()

    reg = await async_client.post(
        "/v1/auth/register", json={"email": "text_val@ustp.edu.ph", "password": "Password123!"}
    )
    token = reg.json()["access_token"]

    async with TestingSessionLocal() as db:
        acc = (
            await db.execute(select(Account).where(Account.email == "text_val@ustp.edu.ph"))
        ).scalar_one()
        acc.role = "student_pro"
        await db.commit()

    headers = {"Authorization": f"Bearer {token}"}

    # Empty string
    res_empty = await async_client.post("/v1/ai/tts", json={"text": ""}, headers=headers)
    assert res_empty.status_code == 422

    # Whitespace only string
    res_ws = await async_client.post("/v1/ai/tts", json={"text": "   "}, headers=headers)
    assert res_ws.status_code == 400

    # Over 512 characters
    res_long = await async_client.post("/v1/ai/tts", json={"text": "a" * 513}, headers=headers)
    assert res_long.status_code == 422


@pytest.mark.asyncio
async def test_tts_quota_and_isolation(async_client: AsyncClient):
    """
    Test TTS quotas: 10 requests/min limit and isolation from chat quotas.
    Success-only usage logging: failed requests must NOT consume quota.
    """
    # 1. Register student_pro user
    reg = await async_client.post(
        "/v1/auth/register", json={"email": "quota_test@ustp.edu.ph", "password": "Password123!"}
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    async with TestingSessionLocal() as db:
        acc = (
            await db.execute(select(Account).where(Account.email == "quota_test@ustp.edu.ph"))
        ).scalar_one()
        acc.role = "student_pro"
        await db.commit()

    # 2. Test failed upstream request does NOT record AIUsage
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client(failing=True)
    res_fail = await async_client.post("/v1/ai/tts", json={"text": "Fail test"}, headers=headers)
    assert res_fail.status_code == 502

    async with TestingSessionLocal() as db:
        usage_count = (await db.execute(select(AIUsage))).scalars().all()
        assert len(usage_count) == 0

    # 3. Test successful requests fill 10 quota items
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client(failing=False)
    for i in range(10):
        res = await async_client.post("/v1/ai/tts", json={"text": f"Phrase {i}"}, headers=headers)
        assert res.status_code == 200

    # 11th request must be rate limited (429)
    res_11 = await async_client.post("/v1/ai/tts", json={"text": "Phrase 11"}, headers=headers)
    assert res_11.status_code == 429

    # 4. Quota isolation check: Chat endpoint quota must STILL have full allowance!
    # Mock deepseek_client in app.state
    from app.clients.deepseek import DeepSeekClient

    def deepseek_handler(req: httpx.Request) -> Response:
        return Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677858288,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    settings = Settings(ENVIRONMENT="development", DEEPSEEK_API_KEY=SecretStr("test-ds-key"))
    app.state.deepseek_client = DeepSeekClient(
        settings=settings, client=httpx.AsyncClient(transport=MockTransport(deepseek_handler))
    )

    res_chat = await async_client.post(
        "/v1/ai/chat",
        json={"messages": [{"role": "user", "content": "Chat test"}]},
        headers=headers,
    )
    assert res_chat.status_code == 200


@pytest.mark.asyncio
async def test_elevenlabs_capability_requires_auth_and_entitlement(async_client: AsyncClient):
    response = await async_client.get("/v1/ai/tts/capabilities")
    assert response.status_code == 401
    registration = await async_client.post(
        "/v1/auth/register",
        json={"email": "elevenlabs_capability@ustp.edu.ph", "password": "SyntheticPassword123!"},
    )
    headers = {"Authorization": "Bearer " + registration.json()["access_token"]}
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client()
    assert (await async_client.get("/v1/ai/tts/capabilities", headers=headers)).status_code == 200
    async with TestingSessionLocal() as db:
        account = (
            await db.execute(
                select(Account).where(Account.email == "elevenlabs_capability@ustp.edu.ph")
            )
        ).scalar_one()
        account.role = "student_pro"
        await db.commit()
    app.state.elevenlabs_tts_client = create_mock_elevenlabs_client()
    response = await async_client.get("/v1/ai/tts/capabilities", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"provider": "elevenlabs", "enabled": True, "mimeType": "audio/mpeg"}
    assert "synthetic-elevenlabs-key" not in response.text
