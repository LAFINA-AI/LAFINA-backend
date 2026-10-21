import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_healthz(async_client: AsyncClient):
    res = await async_client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_register_and_login(async_client: AsyncClient):
    # Reject passwords shorter than the shared six-character minimum.
    short_res = await async_client.post("/v1/auth/register", json={
        "email": "student@ustp.edu.ph",
        "password": "12345"
    })
    assert short_res.status_code == 400
    assert "at least 6 characters" in short_res.json()["detail"]

    # Accept exactly six characters.
    six_res = await async_client.post("/v1/auth/register", json={
        "email": "six@ustp.edu.ph",
        "password": "abc123"
    })
    assert six_res.status_code == 201

    # Reject passwords longer than 128 characters.
    long_res = await async_client.post("/v1/auth/register", json={
        "email": "long@ustp.edu.ph",
        "password": "x" * 129
    })
    assert long_res.status_code == 400
    assert "no more than 128 characters" in long_res.json()["detail"]

    short_login_res = await async_client.post("/v1/auth/login", json={
        "email": "six@ustp.edu.ph",
        "password": "12345"
    })
    assert short_login_res.status_code == 400

    # Valid registration.
    reg_res = await async_client.post("/v1/auth/register", json={
        "email": "student@ustp.edu.ph",
        "password": "super-strong-lafina-passphrase-2026"
    })
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert "access_token" in reg_data
    assert "refresh_token" in reg_data
    assert len(reg_data["recovery_codes"]) == 4

    duplicate_res = await async_client.post("/v1/auth/register", json={
        "email": "STUDENT@USTP.EDU.PH",
        "password": "another-valid-password"
    })
    assert duplicate_res.status_code == 409

    # Login
    login_res = await async_client.post("/v1/auth/login", json={
        "email": "Student@USTP.EDU.PH",
        "password": "super-strong-lafina-passphrase-2026",
        "device_info": "Pixel 7 Android 14"
    })
    assert login_res.status_code == 200
    login_data = login_res.json()
    refresh_token = login_data["refresh_token"]

    # Refresh Token Rotation
    ref_res = await async_client.post("/v1/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert ref_res.status_code == 200
    new_token_data = ref_res.json()
    assert new_token_data["refresh_token"] != refresh_token

    # Old refresh token is revoked
    old_ref_res = await async_client.post("/v1/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert old_ref_res.status_code == 401

    # Check /me profile with rotated token
    new_access_token = new_token_data["access_token"]
    headers = {"Authorization": f"Bearer {new_access_token}"}
    me_res = await async_client.get("/v1/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "student@ustp.edu.ph"
    assert me_res.json()["role"] == "student"


@pytest.mark.asyncio
async def test_recovery_password_policy(async_client: AsyncClient):
    short_res = await async_client.post("/v1/auth/recover", json={
        "email": "missing@ustp.edu.ph",
        "recovery_code": "MISSING",
        "new_password": "12345"
    })
    assert short_res.status_code == 400
    assert "at least 6 characters" in short_res.json()["detail"]

    long_res = await async_client.post("/v1/auth/recover", json={
        "email": "missing@ustp.edu.ph",
        "recovery_code": "MISSING",
        "new_password": "x" * 129
    })
    assert long_res.status_code == 400
    assert "no more than 128 characters" in long_res.json()["detail"]

    six_res = await async_client.post("/v1/auth/recover", json={
        "email": "missing@ustp.edu.ph",
        "recovery_code": "MISSING",
        "new_password": "abc123"
    })
    assert six_res.status_code == 404


@pytest.mark.asyncio
async def test_sync_batch_and_tampering_defense(async_client: AsyncClient):
    # Register & get token
    reg_res = await async_client.post("/v1/auth/register", json={
        "email": "sync_user@ustp.edu.ph",
        "password": "super-strong-lafina-passphrase-2026"
    })
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Attempt tampering: injection of owner_id or role in payload should be rejected or stripped
    tamper_mutation = {
        "mutationId": "mut-tamper-001",
        "entityType": "task",
        "entityId": "task-001",
        "operation": "create",
        "clientUpdatedAt": "2026-07-22T10:00:00Z",
        "payload": {
            "title": "Hack Exam",
            "owner_id": "00000000-0000-0000-0000-000000000000",
            "role": "admin"
        }
    }
    sync_res = await async_client.post("/v1/sync/batch", headers=headers, json={
        "mutations": [tamper_mutation],
        "cursor": 0
    })
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert len(sync_data["rejected"]) == 1
    assert "Forbidden" in sync_data["rejected"][0]["reason"]

    # Valid mutation
    valid_mutation = {
        "mutationId": "mut-valid-001",
        "entityType": "task",
        "entityId": "task-001",
        "operation": "create",
        "clientUpdatedAt": "2026-07-22T10:00:00Z",
        "payload": {
            "title": "Study Mobile AI Architecture",
            "due_date": "2026-07-25",
            "due_time": "14:00",
            "is_completed": False,
            "priority": "high",
            "category": "Academics"
        }
    }
    sync_valid_res = await async_client.post("/v1/sync/batch", headers=headers, json={
        "mutations": [valid_mutation],
        "cursor": 0
    })
    assert sync_valid_res.status_code == 200
    res_data = sync_valid_res.json()
    assert len(res_data["accepted"]) == 1
    assert res_data["accepted"][0]["status"] == "accepted"
    assert len(res_data["changes"]) == 1
    assert res_data["changes"][0]["entityId"] == "task-001"

@pytest.mark.asyncio
async def test_online_ai_proxy(async_client: AsyncClient):
    from pydantic import SecretStr
    from httpx import AsyncClient as HttpxAsyncClient, MockTransport, Response
    from sqlalchemy import select, func
    from tests.conftest import TestingSessionLocal
    from app.models.account import Account
    from app.models.ai_usage import AIUsage
    from app.config import Settings
    from app.clients.deepseek import DeepSeekClient
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    # 1. Unauthenticated attempt should be rejected with 401
    unauth_res = await async_client.post("/v1/ai/chat", json={
        "messages": [{"role": "user", "content": "Hello?"}]
    })
    assert unauth_res.status_code == 401

    # Register user
    password = "super-strong-lafina-passphrase-2026"
    reg_res = await async_client.post("/v1/auth/register", json={
        "email": "ai_user@ustp.edu.ph",
        "password": password
    })
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Student role attempt should be rejected with 403 Forbidden before provider call
    student_chat_res = await async_client.post("/v1/ai/chat", headers=headers, json={
        "requestId": "req-001",
        "messages": [
            {"role": "user", "content": "How do I organize my study schedule for midterms?"}
        ]
    })
    assert student_chat_res.status_code == 403
    assert "student_pro" in student_chat_res.json()["detail"]

    # 3. Promote account role to student_pro in DB and log in to get updated JWT claim
    async with TestingSessionLocal() as db:
        stmt = select(Account).where(Account.email == "ai_user@ustp.edu.ph")
        acc = (await db.execute(stmt)).scalar_one()
        acc.role = "student_pro"
        await db.commit()

    login_res = await async_client.post("/v1/auth/login", json={
        "email": "ai_user@ustp.edu.ph",
        "password": password
    })
    pro_token = login_res.json()["access_token"]
    pro_headers = {"Authorization": f"Bearer {pro_token}"}

    # 4. Without configured DEEPSEEK_API_KEY in dev environment, should return 503 Service Unavailable
    unconfig_settings = Settings(ENVIRONMENT="development", DEEPSEEK_API_KEY=None)
    unconfig_client = DeepSeekClient(settings=unconfig_settings)
    app.dependency_overrides[get_deepseek_client] = lambda: unconfig_client

    dev_chat_res = await async_client.post("/v1/ai/chat", headers=pro_headers, json={
        "requestId": "req-unconfigured",
        "messages": [{"role": "user", "content": "Test unconfigured key"}]
    })
    assert dev_chat_res.status_code == 503
    assert "not configured" in dev_chat_res.json()["detail"]

    # 5. Inject a mock DeepSeekClient returning successful response
    mock_response_json = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Here is your study plan: 1. Prioritize weak subjects."},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
    }
    mock_transport = MockTransport(lambda req: Response(200, json=mock_response_json))
    mock_httpx = HttpxAsyncClient(transport=mock_transport)
    test_settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-test-valid-key")
    )
    mock_client = DeepSeekClient(settings=test_settings, client=mock_httpx)

    app.dependency_overrides[get_deepseek_client] = lambda: mock_client

    pro_chat_res = await async_client.post("/v1/ai/chat", headers=pro_headers, json={
        "requestId": "req-002",
        "messages": [
            {"role": "user", "content": "How do I organize my study schedule for midterms?"}
        ]
    })
    assert pro_chat_res.status_code == 200
    chat_data = pro_chat_res.json()
    assert chat_data["requestId"] == "req-002"
    assert "Prioritize weak subjects" in chat_data["reply"]
    assert chat_data["model"] == "deepseek-v4-flash"
    assert chat_data["usage"]["total_tokens"] == 25

    # Verify AIUsage record created in DB
    async with TestingSessionLocal() as db:
        usage_stmt = select(func.count(AIUsage.id))
        usage_count = (await db.execute(usage_stmt)).scalar()
        assert usage_count == 1

    # 6. Provider failure (e.g. 503 provider error) should NOT record AIUsage in DB
    fail_transport = MockTransport(lambda req: Response(503, json={"error": "provider outage"}))
    fail_httpx = HttpxAsyncClient(transport=fail_transport)
    fail_client = DeepSeekClient(settings=test_settings, client=fail_httpx)

    app.dependency_overrides[get_deepseek_client] = lambda: fail_client

    fail_res = await async_client.post("/v1/ai/chat", headers=pro_headers, json={
        "requestId": "req-fail",
        "messages": [{"role": "user", "content": "Fail test"}]
    })
    assert fail_res.status_code == 503

    # Confirm usage count remains 1 (no new record written on failure)
    async with TestingSessionLocal() as db:
        usage_count_after = (await db.execute(select(func.count(AIUsage.id)))).scalar()
        assert usage_count_after == 1

    app.dependency_overrides.pop(get_deepseek_client, None)



@pytest.mark.asyncio
async def test_change_password(async_client: AsyncClient):
    reg_res = await async_client.post("/v1/auth/register", json={
        "email": "changer@ustp.edu.ph",
        "password": "super-strong-lafina-passphrase-2026"
    })
    assert reg_res.status_code == 201
    headers = {"Authorization": f"Bearer {reg_res.json()['access_token']}"}

    # A second device, to prove it is signed out by the change.
    other_login = await async_client.post("/v1/auth/login", json={
        "email": "changer@ustp.edu.ph",
        "password": "super-strong-lafina-passphrase-2026"
    })
    other_refresh = other_login.json()["refresh_token"]

    # A token alone is not enough: the current password is required.
    wrong_res = await async_client.post("/v1/auth/password", headers=headers, json={
        "current_password": "not-the-password",
        "new_password": "a-brand-new-passphrase-2026"
    })
    # 400, not 401: a typo must not look like an expired session to the client.
    assert wrong_res.status_code == 400
    assert "Current password is incorrect" in wrong_res.json()["detail"]

    # The same policy as registration applies to the replacement.
    short_res = await async_client.post("/v1/auth/password", headers=headers, json={
        "current_password": "super-strong-lafina-passphrase-2026",
        "new_password": "12345"
    })
    assert short_res.status_code == 400
    assert "at least 6 characters" in short_res.json()["detail"]

    same_res = await async_client.post("/v1/auth/password", headers=headers, json={
        "current_password": "super-strong-lafina-passphrase-2026",
        "new_password": "super-strong-lafina-passphrase-2026"
    })
    assert same_res.status_code == 400
    assert "different" in same_res.json()["detail"]

    anon_res = await async_client.post("/v1/auth/password", json={
        "current_password": "super-strong-lafina-passphrase-2026",
        "new_password": "a-brand-new-passphrase-2026"
    })
    assert anon_res.status_code in (401, 403)

    ok_res = await async_client.post("/v1/auth/password", headers=headers, json={
        "current_password": "super-strong-lafina-passphrase-2026",
        "new_password": "a-brand-new-passphrase-2026"
    })
    assert ok_res.status_code == 200

    old_login = await async_client.post("/v1/auth/login", json={
        "email": "changer@ustp.edu.ph",
        "password": "super-strong-lafina-passphrase-2026"
    })
    assert old_login.status_code == 401

    new_login = await async_client.post("/v1/auth/login", json={
        "email": "changer@ustp.edu.ph",
        "password": "a-brand-new-passphrase-2026"
    })
    assert new_login.status_code == 200

    # The session that made the change stays usable...
    me_res = await async_client.get("/v1/me", headers=headers)
    assert me_res.status_code == 200

    # ...while every other device is signed out.
    stale_res = await async_client.post("/v1/auth/refresh", json={"refresh_token": other_refresh})
    assert stale_res.status_code == 401
