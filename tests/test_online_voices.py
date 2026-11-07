"""Signed-in chat/speech contract, with synthetic providers and real auth/SQLite."""
import json

import httpx
import pytest
from sqlalchemy import select

from app.api.v1.ai import get_deepseek_client
from app.clients.deepseek import DeepSeekClient
from app.clients.elevenlabs_tts import FILIPINA_VOICES, ElevenLabsTtsClient, ElevenLabsTtsError
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
    filipina = [{"id": voice_id, "name": name, "language": "fil", "gender": "female"} for voice_id, name in FILIPINA_VOICES]
    assert catalogue.json()["voices"] == filipina + [{"id": "voice_a", "name": "Alice"}, {"id": "voice_b", "name": "Bella"}]
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


async def test_filipina_voices_come_first_with_labels_and_need_no_catalogue_lookup():
    seen = []

    def transport(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"voices": [
                {"voice_id": "zz_male", "name": "Adam", "labels": {"language": "en", "gender": "male"}},
                {"voice_id": "fil_extra", "name": "Ate Lyn", "labels": {"language": "fil", "gender": "female"}},
                {"voice_id": "BtcCEANLn920ZIoN9grW", "name": "Camille - Warm & Conversational, long account name", "labels": {"language": "fil"}},
                {"voice_id": "odd", "name": "Odd", "labels": {"language": "<script>", "gender": 7}},
            ], "has_more": False})
        return httpx.Response(200, content=b"synthetic audio", headers={"content-type": "audio/mpeg"})

    config = Settings(_env_file=None, ELEVENLABS_API_KEY="synthetic-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client = ElevenLabsTtsClient(config, http)
        await client.synthesize_speech("Kumusta!", "req-1", voice_id="n6WaB3rOlZSC9y8yEPEz")
        assert [r.method for r in seen] == ["POST"]  # featured voice: no catalogue call
        assert json.loads(seen[0].content)["model_id"] == "eleven_v4"
        voices = await client.list_voices()
    names = [v["name"] for v in voices]
    assert names[: len(FILIPINA_VOICES)] == [name for _, name in FILIPINA_VOICES]
    assert names[len(FILIPINA_VOICES):] == ["Ate Lyn", "Adam", "Odd"]
    assert voices[-1] == {"id": "odd", "name": "Odd"}  # unusable labels are dropped
    assert {"id": "zz_male", "name": "Adam", "language": "en", "gender": "male"} in voices


def test_default_voice_is_filipina_and_model_is_v4():
    config = Settings(_env_file=None)
    assert config.ELEVENLABS_MODEL_ID == "eleven_v4"
    assert config.ELEVENLABS_VOICE_ID == FILIPINA_VOICES[0][0]
