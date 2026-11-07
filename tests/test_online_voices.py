"""Signed-in chat/speech contract, with synthetic providers and real auth/SQLite."""
import json

import httpx
import pytest
from sqlalchemy import select

from app.api.v1.ai import get_deepseek_client
from app.clients.deepseek import DeepSeekClient
from app.clients.elevenlabs_tts import ElevenLabsTtsClient, ElevenLabsTtsError
from app.config import Settings
from app.main import app
from app.models.account import Account
from app.services.capabilities import has_online_ai_access
from tests.conftest import TestingSessionLocal


@pytest.fixture
def providers():
    captured = []
    def transport(request):
        captured.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"voices": [{"voice_id": "voice_a", "name": "Alice"}, {"voice_id": "voice_b", "name": "Bella"}], "has_more": False})
        if request.url.host == "api.deepseek.com":
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Try short study sessions."}}], "usage": {"total_tokens": 12}})
        return httpx.Response(200, content=b"synthetic audio", headers={"content-type": "audio/mpeg"})
    config = Settings(_env_file=None, ENVIRONMENT="development", DEEPSEEK_API_KEY="synthetic-key", ELEVENLABS_API_KEY="synthetic-key")
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    eleven = ElevenLabsTtsClient(config, http)
    deepseek = DeepSeekClient(config, http)
    previous = app.dependency_overrides.get(get_deepseek_client)
    previous_tts = getattr(app.state, "elevenlabs_tts_client", None)
    app.dependency_overrides[get_deepseek_client] = lambda: deepseek
    app.state.elevenlabs_tts_client = eleven
    yield captured, eleven
    if previous is None:
        app.dependency_overrides.pop(get_deepseek_client, None)
    else:
        app.dependency_overrides[get_deepseek_client] = previous
    app.state.elevenlabs_tts_client = previous_tts


async def signed_in(client, role="student", active=True):
    response = await client.post("/v1/auth/register", json={"email": "online-test@example.org", "password": "SyntheticPassword123!"})
    assert response.status_code == 201
    async with TestingSessionLocal() as db:
        account = (await db.execute(select(Account).where(Account.email == "online-test@example.org"))).scalar_one()
        account.role = role
        account.is_active = active
        # A basic student still cannot use unrelated paid generation features.
        if role == "student":
            assert not has_online_ai_access(account)
        await db.commit()
    return {"Authorization": "Bearer " + response.json()["access_token"]}


@pytest.mark.parametrize("role", ["student", "student_pro", "admin"])
async def test_all_active_signed_in_roles_can_choose_voice_and_chat(async_client, providers, role):
    captured, _ = providers
    headers = await signed_in(async_client, role)
    catalogue = await async_client.get("/v1/ai/tts/voices", headers=headers)
    assert catalogue.status_code == 200
    assert catalogue.json()["voices"] == [{"id": "voice_a", "name": "Alice"}, {"id": "voice_b", "name": "Bella"}]
    result = await async_client.post("/v1/ai/tts", headers=headers, json={"text": "Hello", "voiceId": "voice_b"})
    assert result.status_code == 200
    assert result.json()["voice"] == "voice_b"
    assert captured[-1].url.path.endswith("/voice_b")
    result = await async_client.post("/v1/ai/chat", headers=headers, json={"messages": [{"role": "user", "content": "Study advice?"}], "useHandbook": False})
    assert result.status_code == 200
    assert result.json()["provider"] == "deepseek"
    assert result.json()["usedHandbook"] is False
    assert result.json()["reply"] == "Try short study sessions."
    payload = json.loads(captured[-1].content)
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert not any(m["role"] == "system" and "has been created" in m["content"] for m in payload["messages"])
    assert sum(r.method == "GET" for r in captured) == 1  # catalogue cache reused by synthesis


async def test_guests_and_disabled_accounts_never_reach_providers(async_client, providers):
    captured, _ = providers
    for path in ["/v1/ai/tts/voices", "/v1/ai/tts/capabilities"]:
        assert (await async_client.get(path)).status_code == 401
    headers = await signed_in(async_client, active=False)
    assert (await async_client.get("/v1/ai/tts/voices", headers=headers)).status_code == 401
    assert (await async_client.post("/v1/ai/chat", headers=headers, json={"messages": [{"role": "user", "content": "Hello"}]})).status_code == 401
    assert (await async_client.post("/v1/ai/tts", headers=headers, json={"text": "Hello"})).status_code == 401
    assert captured == []


async def test_unknown_and_malformed_voice_never_generate_audio(async_client, providers):
    captured, eleven = providers
    headers = await signed_in(async_client)
    for voice, status in [("missing", 422), ("../path", 422)]:
        response = await async_client.post("/v1/ai/tts", headers=headers, json={"text": "Hello", "voiceId": voice})
        assert response.status_code == status
    assert all(request.method == "GET" for request in captured)
    assert eleven.used_characters == 0


@pytest.mark.parametrize("status,expected", [(403, 503), (429, 429), (500, 502)])
async def test_catalogue_errors_are_actionable_and_redacted(status, expected):
    config = Settings(_env_file=None, ELEVENLABS_API_KEY="synthetic-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, text="private provider body"))) as http:
        with pytest.raises(ElevenLabsTtsError) as error:
            await ElevenLabsTtsClient(config, http).list_voices()
        assert error.value.status_code == expected
        assert "private provider body" not in error.value.message


def test_deepseek_legacy_alias_and_explicit_model():
    assert Settings(_env_file=None, DEEPSEEK_MODEL="deepseek-v4-flash").DEEPSEEK_MODEL == "deepseek-flash"
    assert Settings(_env_file=None, DEEPSEEK_MODEL="deepseek-v4-pro").DEEPSEEK_MODEL == "deepseek-v4-pro"


def test_persona_prompt_is_conversational_and_honest():
    from app.api.v1.ai import LAFINA_SYSTEM_INSTRUCTION as prompt

    assert "Always introduce yourself" not in prompt
    assert "do not repeat the introduction" in prompt
    assert "Never claim a task, reminder or event was saved" in prompt
