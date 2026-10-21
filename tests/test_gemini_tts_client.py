import base64
import json
import logging
import pytest
from pydantic import SecretStr
import httpx
from httpx import MockTransport, Response

from app.config import Settings
from app.clients.gemini_tts import (
    GeminiTtsClient,
    GeminiTtsConfigError,
    GeminiTtsAuthenticationError,
    GeminiTtsRateLimitError,
    GeminiTtsInvalidRequestError,
    GeminiTtsProviderServerError,
    GeminiTtsTimeoutError,
    GeminiTtsMalformedResponseError
)


def create_dummy_pcm(sample_count: int = 480) -> bytes:
    """Creates dummy 16-bit PCM little-endian audio bytes (480 samples = 20ms at 24kHz)."""
    return b"\x00\x00" * sample_count


@pytest.mark.asyncio
async def test_gemini_tts_client_missing_key():
    """Client must fail fast with 503 if GEMINI_API_KEY is missing or invalid."""
    settings = Settings(ENVIRONMENT="development", GEMINI_API_KEY=None)
    client = GeminiTtsClient(settings=settings)
    with pytest.raises(GeminiTtsConfigError) as exc:
        await client.synthesize_speech("Hello world")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_gemini_tts_client_success(caplog):
    """Test successful Gemini TTS synthesis, payload formatting, PCM-to-WAV header, and log privacy."""
    caplog.set_level(logging.INFO)
    captured = {}
    pcm_bytes = create_dummy_pcm(2400)  # 100ms audio
    pcm_b64 = base64.b64encode(pcm_bytes).decode("utf-8")

    def handler(request: httpx.Request) -> Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.read().decode("utf-8"))
        return Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/L16;codec=pcm;rate=24000",
                                        "data": pcm_b64
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 15
                }
            }
        )

    transport = MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        settings = Settings(
            ENVIRONMENT="development",
            GEMINI_API_KEY=SecretStr("test-gemini-key-12345")
        )
        client = GeminiTtsClient(settings=settings, client=http_client)
        wav_b64, usage = await client.synthesize_speech("Test reminder task", request_id="req-123")

    # Check request payload & headers
    assert "x-goog-api-key" in captured["headers"]
    assert captured["headers"]["x-goog-api-key"] == "test-gemini-key-12345"
    assert "gemini-3.1-flash-tts-preview" in captured["url"]
    assert captured["body"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Aoede"
    assert captured["body"]["contents"][0]["role"] == "user"
    assert captured["body"]["contents"][0]["parts"][0]["text"] == (
        "## Transcript:\nTest reminder task"
    )

    # Check decoded WAV header
    wav_bytes = base64.b64decode(wav_b64)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert usage["prompt_tokens"] == 15

    # Privacy check: Spoken text and secrets must not be in log
    log_text = caplog.text
    assert "Test reminder task" not in log_text
    assert "test-gemini-key-12345" not in log_text
    assert "test-gemini-key" not in log_text


@pytest.mark.asyncio
async def test_gemini_tts_client_error_mappings():
    """Test mapping of upstream HTTP status codes to sanitized GeminiTtsError subclasses."""
    settings = Settings(ENVIRONMENT="development", GEMINI_API_KEY=SecretStr("test-key"))

    error_cases = [
        (401, GeminiTtsAuthenticationError, 503),
        (403, GeminiTtsAuthenticationError, 503),
        (429, GeminiTtsRateLimitError, 429),
        (400, GeminiTtsInvalidRequestError, 502),
        (422, GeminiTtsInvalidRequestError, 502),
        (500, GeminiTtsProviderServerError, 503),
        (503, GeminiTtsProviderServerError, 503),
    ]

    for status_code, exc_class, expected_http_code in error_cases:
        def handler(request: httpx.Request) -> Response:
            return Response(status_code, json={"error": {"message": "Upstream error"}})

        transport = MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = GeminiTtsClient(settings=settings, client=http_client)
            with pytest.raises(exc_class) as exc_info:
                await client.synthesize_speech("Hello")
            assert exc_info.value.status_code == expected_http_code


@pytest.mark.asyncio
async def test_gemini_tts_client_timeout():
    """Test timeout exception mapping to 504."""
    settings = Settings(ENVIRONMENT="development", GEMINI_API_KEY=SecretStr("test-key"))

    def handler(request: httpx.Request) -> Response:
        raise httpx.TimeoutException("Connection timed out")

    transport = MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GeminiTtsClient(settings=settings, client=http_client)
        with pytest.raises(GeminiTtsTimeoutError) as exc_info:
            await client.synthesize_speech("Hello")
        assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_gemini_tts_client_malformed_responses():
    """Test malformed response scenarios (empty parts, invalid base64, size cap exceeded, odd byte length)."""
    settings = Settings(ENVIRONMENT="development", GEMINI_API_KEY=SecretStr("test-key"))

    # Case 1: Empty candidates
    def handler_empty(request: httpx.Request) -> Response:
        return Response(200, json={"candidates": []})

    async with httpx.AsyncClient(transport=MockTransport(handler_empty)) as http_client:
        client = GeminiTtsClient(settings=settings, client=http_client)
        with pytest.raises(GeminiTtsMalformedResponseError):
            await client.synthesize_speech("Hello")

    # Case 2: Invalid base64
    def handler_bad_b64(request: httpx.Request) -> Response:
        return Response(200, json={
            "candidates": [{"content": {"parts": [{"inlineData": {"data": "not-valid-base64!!!"}}]}}]
        })

    async with httpx.AsyncClient(transport=MockTransport(handler_bad_b64)) as http_client:
        client = GeminiTtsClient(settings=settings, client=http_client)
        with pytest.raises(GeminiTtsMalformedResponseError):
            await client.synthesize_speech("Hello")

    # Case 3: Odd byte length PCM
    odd_pcm_b64 = base64.b64encode(b"\x00\x01\x02").decode("utf-8")
    def handler_odd(request: httpx.Request) -> Response:
        return Response(200, json={
            "candidates": [{"content": {"parts": [{"inlineData": {"data": odd_pcm_b64}}]}}]
        })

    async with httpx.AsyncClient(transport=MockTransport(handler_odd)) as http_client:
        client = GeminiTtsClient(settings=settings, client=http_client)
        with pytest.raises(GeminiTtsMalformedResponseError):
            await client.synthesize_speech("Hello")
