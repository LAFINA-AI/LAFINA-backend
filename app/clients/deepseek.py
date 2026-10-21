import logging
import time
from typing import Optional
from pydantic import BaseModel, Field
import httpx

from app.config import Settings

logger = logging.getLogger("lafina.deepseek")

# Upstream error bodies are provider text, never our key or the user's prompt,
# so they are safe to log and to pass back to the caller.
_MAX_UPSTREAM_DETAIL_CHARS = 300


def extract_upstream_error(body_text: str) -> str:
    """Pull the provider's own explanation out of a non-200 response.

    DeepSeek answers with ``{"error": {"message": ...}}``; a gateway in front of
    it may answer with plain text or HTML. Without this the caller only sees a
    status code, which is not enough to tell "no credit" from "wrong plan" from
    "model not enabled" — all of which arrive as 402.
    """
    text = (body_text or "").strip()
    if not text:
        return "(empty response body)"
    try:
        import json

        data = json.loads(text)
    except ValueError:
        return text[:_MAX_UPSTREAM_DETAIL_CHARS]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or error.get("code")
            if message:
                return str(message)[:_MAX_UPSTREAM_DETAIL_CHARS]
        elif isinstance(error, str) and error:
            return error[:_MAX_UPSTREAM_DETAIL_CHARS]
        if data.get("message"):
            return str(data["message"])[:_MAX_UPSTREAM_DETAIL_CHARS]
    return text[:_MAX_UPSTREAM_DETAIL_CHARS]


class DeepSeekError(Exception):
    """Base exception for DeepSeek client errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DeepSeekConfigError(DeepSeekError):
    """Missing or invalid API key configuration (503 Service Unavailable)."""
    def __init__(self, message: str = "DeepSeek API key is not configured or invalid."):
        super().__init__(message, status_code=503)


class DeepSeekAuthenticationError(DeepSeekError):
    """Upstream 401 Authentication Error (503 Service Unavailable)."""
    def __init__(self, message: str = "DeepSeek provider authentication failed."):
        super().__init__(message, status_code=503)


class DeepSeekBillingError(DeepSeekError):
    """Upstream 402 Insufficient Balance / Billing Error (503 Service Unavailable)."""
    def __init__(self, message: str = "DeepSeek provider billing is unavailable."):
        super().__init__(message, status_code=503)


class DeepSeekRateLimitError(DeepSeekError):
    """Upstream 429 Rate Limit Error (429 Too Many Requests)."""
    def __init__(self, message: str = "DeepSeek rate limit exceeded. Please try again later."):
        super().__init__(message, status_code=429)


class DeepSeekInvalidRequestError(DeepSeekError):
    """Upstream 400 or 422 Invalid Request Error (502 Bad Gateway)."""
    def __init__(self, message: str = "Invalid request sent to DeepSeek provider."):
        super().__init__(message, status_code=502)


class DeepSeekProviderServerError(DeepSeekError):
    """Upstream 500 or 503 Server Error (503 Service Unavailable)."""
    def __init__(self, message: str = "DeepSeek provider temporary outage."):
        super().__init__(message, status_code=503)


class DeepSeekTimeoutError(DeepSeekError):
    """Request timeout (504 Gateway Timeout)."""
    def __init__(self, message: str = "DeepSeek AI assistant request timed out."):
        super().__init__(message, status_code=504)


class DeepSeekMalformedResponseError(DeepSeekError):
    """Upstream payload malformed or missing choices/content (502 Bad Gateway)."""
    def __init__(self, message: str = "Received malformed response from DeepSeek provider."):
        super().__init__(message, status_code=502)


class DeepSeekTransportError(DeepSeekError):
    """Network connection or transport failure (503 Service Unavailable)."""
    def __init__(self, message: str = "Failed to communicate with DeepSeek AI proxy."):
        super().__init__(message, status_code=503)


class DeepSeekMessage(BaseModel):
    role: str
    content: Optional[str] = None


class DeepSeekChoice(BaseModel):
    index: int = 0
    message: DeepSeekMessage
    finish_reason: Optional[str] = None


class DeepSeekUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class DeepSeekCompletionResponse(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: list[DeepSeekChoice]
    usage: Optional[DeepSeekUsage] = None


class DeepSeekClient:
    """
    Production DeepSeek API client wrapping httpx.AsyncClient connection pool.
    Handles authentication, payload formatting (non-thinking mode), typed response
    parsing, sanitization, and safe error status code mapping.
    """

    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None):
        self.settings = settings
        self._custom_client = client
        self._client: Optional[httpx.AsyncClient] = client

    async def start(self) -> None:
        if self._custom_client is not None:
            self._client = self._custom_client
        elif self._client is None:
            timeout = httpx.Timeout(self.settings.DEEPSEEK_TIMEOUT_SECONDS)
            self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        if self._custom_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        request_id: str = ""
    ) -> tuple[str, dict[str, int]]:
        """
        Executes chat completion request to DeepSeek API with non-thinking mode disabled.
        Returns (reply_content, usage_dict).
        """
        return await self._completion(
            messages=messages,
            user_id=user_id,
            request_id=request_id,
            max_tokens=1024,
            temperature=0.7,
        )

    async def json_completion(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        request_id: str = "",
        model: Optional[str] = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Completion tuned for callers that need machine-readable output.

        Same OpenAI-compatible endpoint as the chat proxy, with a near-zero
        temperature and room for a long reply, because a deck of flashcards is
        both longer than a chat turn and worthless if the model gets creative
        with the shape of it. ``json_mode`` also asks the provider to return a
        JSON object; the prompt must say "JSON" for DeepSeek to accept it.
        """
        return await self._completion(
            messages=messages,
            user_id=user_id,
            request_id=request_id,
            max_tokens=max_tokens,
            temperature=0.1,
            model=model,
            json_mode=json_mode,
        )

    async def _completion(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        request_id: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, int]]:
        reason = self.settings.get_deepseek_key_invalid_reason()
        if reason is not None:
            logger.warning(
                f"DeepSeek call attempted without valid API key ({reason}) [requestId={request_id}]"
            )
            raise DeepSeekConfigError("DeepSeek API key is not configured.")

        if self._client is None:
            await self.start()

        assert self._client is not None, "AsyncClient must be initialized"

        raw_key = self.settings.DEEPSEEK_API_KEY.get_secret_value().strip().strip("'\"") if self.settings.DEEPSEEK_API_KEY else ""
        headers = {
            "Authorization": f"Bearer {raw_key}",
            "Content-Type": "application/json"
        }

        chosen_model = model or self.settings.DEEPSEEK_MODEL
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "user": user_id
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
        start_time = time.monotonic()

        try:
            res = await self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(f"DeepSeek request timed out after {duration_ms}ms [requestId={request_id}]")
            raise DeepSeekTimeoutError()
        except httpx.RequestError as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                f"DeepSeek transport failure ({type(exc).__name__}) after {duration_ms}ms [requestId={request_id}]"
            )
            raise DeepSeekTransportError()

        duration_ms = int((time.monotonic() - start_time) * 1000)

        if res.status_code != 200:
            status_code = res.status_code
            # The provider's own message is the only thing that distinguishes,
            # say, an unfunded account from an expired grant or a model the
            # account cannot use — all of which DeepSeek reports as 402.
            detail = extract_upstream_error(res.text)
            logger.warning(
                f"DeepSeek upstream error HTTP {status_code} after {duration_ms}ms "
                f"[requestId={request_id}] model={chosen_model} "
                f"upstream_detail={detail!r}"
            )
            if status_code == 401:
                raise DeepSeekAuthenticationError(
                    f"DeepSeek provider authentication failed: {detail}"
                )
            elif status_code == 402:
                raise DeepSeekBillingError(
                    f"DeepSeek provider billing is unavailable: {detail}"
                )
            elif status_code == 429:
                raise DeepSeekRateLimitError(
                    f"DeepSeek rate limit exceeded. Please try again later: {detail}"
                )
            elif status_code in (400, 422):
                raise DeepSeekInvalidRequestError(
                    f"Invalid request sent to DeepSeek provider: {detail}"
                )
            elif status_code in (500, 503):
                raise DeepSeekProviderServerError(
                    f"DeepSeek provider temporary outage: {detail}"
                )
            else:
                raise DeepSeekError(
                    message=f"DeepSeek provider returned error status {status_code}: {detail}",
                    status_code=502
                )

        try:
            data = res.json()
            parsed = DeepSeekCompletionResponse.model_validate(data)
        except Exception as parse_err:
            logger.error(
                f"DeepSeek malformed JSON response ({type(parse_err).__name__}) [requestId={request_id}]"
            )
            raise DeepSeekMalformedResponseError()

        if not parsed.choices:
            logger.error(f"DeepSeek choices array empty [requestId={request_id}]")
            raise DeepSeekMalformedResponseError("Received empty choice array from DeepSeek provider.")

        reply_content = parsed.choices[0].message.content
        if reply_content is None or reply_content.strip() == "":
            logger.error(f"DeepSeek message content null or empty [requestId={request_id}]")
            raise DeepSeekMalformedResponseError("Received null or empty content from DeepSeek provider.")

        usage_dict = {
            "prompt_tokens": parsed.usage.prompt_tokens if parsed.usage else 0,
            "completion_tokens": parsed.usage.completion_tokens if parsed.usage else 0,
            "total_tokens": parsed.usage.total_tokens if parsed.usage else 0
        }

        logger.info(
            f"DeepSeek request success: status=200, duration={duration_ms}ms, "
            f"model={chosen_model}, prompt_tokens={usage_dict['prompt_tokens']}, "
            f"completion_tokens={usage_dict['completion_tokens']} [requestId={request_id}]"
        )

        return reply_content, usage_dict
