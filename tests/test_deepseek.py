import json
import logging
import pytest
from pydantic import SecretStr
import httpx
from httpx import AsyncClient, MockTransport, Response

from app.config import Settings
from app.clients.deepseek import (
    DeepSeekClient,
    DeepSeekAuthenticationError,
    DeepSeekBillingError,
    DeepSeekRateLimitError,
    DeepSeekInvalidRequestError,
    DeepSeekProviderServerError,
    DeepSeekTimeoutError,
    DeepSeekMalformedResponseError,
    DeepSeekTransportError
)


# --- Configuration Tests ---

def test_production_rejects_missing_or_placeholder_keys():
    """Production environment must fail fast if DEEPSEEK_API_KEY is missing or placeholder."""
    for invalid_key in [None, "", "mock-deepseek-key-for-dev", "your-deepseek-api-key", "placeholder"]:
        with pytest.raises(ValueError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                DEEPSEEK_API_KEY=SecretStr(invalid_key) if invalid_key is not None else None
            )
        assert "DEEPSEEK_API_KEY secret environment variable must be configured" in str(exc_info.value)


def test_development_allows_missing_key():
    """Development environment permits startup without valid key."""
    settings = Settings(ENVIRONMENT="development", DEEPSEEK_API_KEY=None)
    assert not settings.is_deepseek_key_valid()


def test_secret_redaction_in_settings_representation():
    """DEEPSEEK_API_KEY secret value must be redacted in string and repr output."""
    raw_secret = "sk-deepseek-super-secret-key-12345"
    settings = Settings(ENVIRONMENT="development", DEEPSEEK_API_KEY=SecretStr(raw_secret))
    assert raw_secret not in str(settings)
    assert raw_secret not in repr(settings)
    assert "**********" in repr(settings.DEEPSEEK_API_KEY)


# --- Client Provider Tests (Using httpx.MockTransport) ---

@pytest.mark.asyncio
async def test_client_successful_chat_completion(caplog):
    """Test successful chat completion, header, path, non-thinking mode, pseudonymous user, and log safety."""
    caplog.set_level(logging.INFO)
    captured_request = {}

    def handler(request: httpx.Request) -> Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = json.loads(request.read().decode("utf-8"))
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
                        "message": {"role": "assistant", "content": "Hello! I am LAFINA Online AI."},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20
                }
            }
        )

    transport = MockTransport(handler)
    mock_httpx = AsyncClient(transport=transport)
    settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-valid-test-key-999")
    )
    client = DeepSeekClient(settings=settings, client=mock_httpx)

    user_uuid = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
    reply, usage = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        user_id=user_uuid,
        request_id="req-test-100"
    )

    assert reply == "Hello! I am LAFINA Online AI."
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 20

    # Verify request parameters
    assert captured_request["url"] == "https://api.deepseek.com/chat/completions"
    assert captured_request["headers"]["authorization"] == "Bearer sk-valid-test-key-999"
    assert captured_request["body"]["model"] == "deepseek-v4-flash"
    assert captured_request["body"]["stream"] is False
    assert captured_request["body"]["thinking"] == {"type": "disabled"}
    assert captured_request["body"]["user"] == user_uuid

    # Assert secret never appears in logs
    for record in caplog.records:
        assert "sk-valid-test-key-999" not in record.getMessage()
        assert "Authorization" not in record.getMessage()


@pytest.mark.asyncio
async def test_client_error_mappings():
    """Verify HTTP status mappings: 401->503, 402->503, 429->429, 400/422->502, 500/503->503."""
    status_exception_map = [
        (401, DeepSeekAuthenticationError, 503),
        (402, DeepSeekBillingError, 503),
        (429, DeepSeekRateLimitError, 429),
        (400, DeepSeekInvalidRequestError, 502),
        (422, DeepSeekInvalidRequestError, 502),
        (500, DeepSeekProviderServerError, 503),
        (503, DeepSeekProviderServerError, 503),
    ]

    settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-test-key")
    )

    for status_code, expected_exc, expected_status in status_exception_map:
        transport = MockTransport(lambda req, sc=status_code: Response(sc, json={"error": "upstream error"}))
        mock_httpx = AsyncClient(transport=transport)
        client = DeepSeekClient(settings=settings, client=mock_httpx)

        with pytest.raises(expected_exc) as exc_info:
            await client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                user_id="user-123"
            )
        assert exc_info.value.status_code == expected_status
        # Ensure secret is not in exception message
        assert "sk-test-key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_timeout_handling():
    """Verify httpx.TimeoutException maps to 504 DeepSeekTimeoutError."""
    def handler(request: httpx.Request):
        raise httpx.TimeoutException("Connection timed out", request=request)

    transport = MockTransport(handler)
    mock_httpx = AsyncClient(transport=transport)
    settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-test-key")
    )
    client = DeepSeekClient(settings=settings, client=mock_httpx)

    with pytest.raises(DeepSeekTimeoutError) as exc_info:
        await client.chat_completion(messages=[{"role": "user", "content": "hi"}], user_id="u1")
    assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_client_transport_failure():
    """Verify network connection/transport error maps to 503 DeepSeekTransportError."""
    def handler(request: httpx.Request):
        raise httpx.ConnectError("Failed to connect", request=request)

    transport = MockTransport(handler)
    mock_httpx = AsyncClient(transport=transport)
    settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-test-key")
    )
    client = DeepSeekClient(settings=settings, client=mock_httpx)

    with pytest.raises(DeepSeekTransportError) as exc_info:
        await client.chat_completion(messages=[{"role": "user", "content": "hi"}], user_id="u1")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_client_malformed_and_empty_responses():
    """Verify malformed JSON, empty choices, or null content maps to 502 DeepSeekMalformedResponseError."""
    malformed_responses = [
        Response(200, text="not valid json"),
        Response(200, json={"choices": []}),
        Response(200, json={"choices": [{"message": {"content": None}}]}),
        Response(200, json={"choices": [{"message": {"content": ""}}]}),
    ]

    settings = Settings(
        ENVIRONMENT="development",
        DEEPSEEK_API_KEY=SecretStr("sk-test-key")
    )

    for resp in malformed_responses:
        transport = MockTransport(lambda req, r=resp: r)
        mock_httpx = AsyncClient(transport=transport)
        client = DeepSeekClient(settings=settings, client=mock_httpx)

        with pytest.raises(DeepSeekMalformedResponseError) as exc_info:
            await client.chat_completion(messages=[{"role": "user", "content": "hi"}], user_id="u1")
        assert exc_info.value.status_code == 502
