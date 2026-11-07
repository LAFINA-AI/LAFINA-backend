from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.clients.elevenlabs_tts import (
    ElevenLabsTtsClient,
    ElevenLabsTtsError,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(key="synthetic-server-key", budget=20000):
    return SimpleNamespace(
        ELEVENLABS_API_KEY=SecretStr(key),
        ELEVENLABS_MODEL_ID="eleven_flash_v2_5",
        ELEVENLABS_VOICE_ID="test_voice",
        ELEVENLABS_MAX_CHARACTERS_PER_PROCESS=budget,
    )


@pytest.mark.anyio
async def test_contract_and_duplicate_budget():
    captured = []

    def transport(request):
        captured.append(request)
        return httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client = ElevenLabsTtsClient(settings(), http)
        audio, usage = await client.synthesize_speech("Hello", "r1")
        assert audio == "YXVkaW8="
        assert usage["prompt_tokens"] == 5
        assert captured[0].headers["xi-api-key"] == "synthetic-server-key"
        assert captured[0].url.host == "api.elevenlabs.io"
        with pytest.raises(ElevenLabsTtsError) as duplicate:
            await client.synthesize_speech("Hello", "r1")
        assert duplicate.value.status_code == 409
        assert len(captured) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("status,expected", [(401, 503), (403, 503), (429, 429), (500, 502)])
async def test_sanitized_errors(status, expected):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="private-provider-body")
        )
    ) as http:
        client = ElevenLabsTtsClient(settings(), http)
        with pytest.raises(ElevenLabsTtsError) as failure:
            await client.synthesize_speech("Hello", "r1")
        assert failure.value.status_code == expected
        assert "private-provider-body" not in failure.value.message


@pytest.mark.anyio
async def test_timeout_and_missing_configuration():
    def transport(_):
        raise httpx.ReadTimeout("private error details")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(ElevenLabsTtsError) as timeout:
            await ElevenLabsTtsClient(settings(), http).synthesize_speech("Hello", "r1")
        assert timeout.value.status_code == 504
        with pytest.raises(ElevenLabsTtsError) as missing:
            await ElevenLabsTtsClient(settings(key=""), http).synthesize_speech("Hello", "r1")
        assert missing.value.status_code == 503


@pytest.mark.anyio
async def test_budget_and_input_validation_before_network():
    def forbidden(_):
        raise AssertionError("Network should not be called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as http:
        client = ElevenLabsTtsClient(settings(budget=1), http)
        with pytest.raises(ElevenLabsTtsError) as budget:
            await client.synthesize_speech("Hello", "r1")
        assert budget.value.status_code == 429
        with pytest.raises(ElevenLabsTtsError) as invalid:
            await client.synthesize_speech("x" * 513, "r2")
        assert invalid.value.status_code == 400


def test_fast_model_only_for_low_latency_and_only_when_set():
    fast = SimpleNamespace(**vars(settings()), ELEVENLABS_FAST_MODEL_ID="eleven_v4_turbo")
    assert ElevenLabsTtsClient(fast).model_for("low") == "eleven_v4_turbo"
    assert ElevenLabsTtsClient(fast).model_for("standard") == "eleven_flash_v2_5"
    assert ElevenLabsTtsClient(settings()).model_for("low") == "eleven_flash_v2_5"
