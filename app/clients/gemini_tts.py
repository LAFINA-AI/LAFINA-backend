import base64
import io
import logging
import time
import wave
from typing import Optional
import httpx

from app.config import Settings

logger = logging.getLogger("lafina.gemini_tts")

MAX_DECODED_PCM_SIZE_BYTES = 5 * 1024 * 1024  # 5 MiB cap


class GeminiTtsError(Exception):
    """Base exception for Gemini TTS client errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GeminiTtsConfigError(GeminiTtsError):
    """Missing or invalid Gemini API key configuration (503 Service Unavailable)."""
    def __init__(self, message: str = "Gemini API key is not configured or invalid."):
        super().__init__(message, status_code=503)


class GeminiTtsAuthenticationError(GeminiTtsError):
    """Upstream 401/403 Authentication Error (503 Service Unavailable)."""
    def __init__(self, message: str = "Gemini provider authentication failed."):
        super().__init__(message, status_code=503)


class GeminiTtsRateLimitError(GeminiTtsError):
    """Upstream 429 Rate Limit Error (429 Too Many Requests)."""
    def __init__(self, message: str = "Gemini TTS rate limit exceeded. Please try again later."):
        super().__init__(message, status_code=429)


class GeminiTtsInvalidRequestError(GeminiTtsError):
    """Upstream 400 or 422 Invalid Request Error (502 Bad Gateway)."""
    def __init__(self, message: str = "Invalid request sent to Gemini TTS provider."):
        super().__init__(message, status_code=502)


class GeminiTtsProviderServerError(GeminiTtsError):
    """Upstream 500 or 503 Server Error (503 Service Unavailable)."""
    def __init__(self, message: str = "Gemini TTS provider temporary outage."):
        super().__init__(message, status_code=503)


class GeminiTtsTimeoutError(GeminiTtsError):
    """Request timeout (504 Gateway Timeout)."""
    def __init__(self, message: str = "Gemini TTS request timed out."):
        super().__init__(message, status_code=504)


class GeminiTtsMalformedResponseError(GeminiTtsError):
    """Upstream payload malformed, missing audio, or invalid PCM (502 Bad Gateway)."""
    def __init__(self, message: str = "Received malformed audio response from Gemini TTS provider."):
        super().__init__(message, status_code=502)


class GeminiTtsTransportError(GeminiTtsError):
    """Network connection or transport failure (503 Service Unavailable)."""
    def __init__(self, message: str = "Failed to communicate with Gemini TTS proxy."):
        super().__init__(message, status_code=503)


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Wraps raw 16-bit PCM bytes in a standard RIFF/WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


class GeminiTtsClient:
    """
    Production Gemini TTS API client wrapping httpx.AsyncClient connection pool.
    Proxies requests to gemini-3.1-flash-tts-preview with Aoede voice, validates
    PCM response metadata strictly, converts PCM to WAV, and maps errors cleanly.
    
    Privacy Contract: Spoken text, Base64 payloads, and API keys are strictly excluded
    from all log statements.
    """

    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None):
        self.settings = settings
        self._custom_client = client
        self._client: Optional[httpx.AsyncClient] = client

    async def start(self) -> None:
        if self._custom_client is not None:
            self._client = self._custom_client
        elif self._client is None:
            timeout = httpx.Timeout(self.settings.GEMINI_TTS_TIMEOUT_SECONDS)
            self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        if self._custom_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def synthesize_speech(
        self,
        text: str,
        request_id: str = ""
    ) -> tuple[str, dict[str, int]]:
        """
        Synthesizes text to WAV base64 string via Gemini API.
        Returns (audio_base64_wav, usage_dict).
        """
        reason = self.settings.get_gemini_key_invalid_reason()
        if reason is not None:
            logger.warning(
                f"Gemini TTS call attempted without valid API key ({reason}) [requestId={request_id}]"
            )
            raise GeminiTtsConfigError("Gemini API key is not configured.")

        if self._client is None:
            await self.start()

        assert self._client is not None, "AsyncClient must be initialized"

        raw_key = (
            self.settings.GEMINI_API_KEY.get_secret_value().strip().strip("'\"")
            if self.settings.GEMINI_API_KEY
            else ""
        )
        headers = {
            "x-goog-api-key": raw_key,
            "Content-Type": "application/json"
        }

        # Keep the prompt compact to reduce request overhead and match Gemini's TTS format.
        prompt_text = f"## Transcript:\n{text}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.settings.GEMINI_TTS_VOICE
                        }
                    }
                },
                "temperature": 1.0
            }
        }

        base_url = self.settings.GEMINI_BASE_URL.rstrip('/')
        model = self.settings.GEMINI_TTS_MODEL
        url = f"{base_url}/v1beta/models/{model}:generateContent"

        start_time = time.monotonic()

        try:
            res = await self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(f"Gemini TTS request timed out after {duration_ms}ms [requestId={request_id}]")
            raise GeminiTtsTimeoutError()
        except httpx.RequestError as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                f"Gemini TTS transport failure ({type(exc).__name__}) after {duration_ms}ms [requestId={request_id}]"
            )
            raise GeminiTtsTransportError()

        duration_ms = int((time.monotonic() - start_time) * 1000)

        if res.status_code != 200:
            status_code = res.status_code
            logger.warning(
                f"Gemini TTS upstream error HTTP {status_code} after {duration_ms}ms [requestId={request_id}]"
            )
            if status_code in (401, 403):
                raise GeminiTtsAuthenticationError()
            elif status_code == 429:
                raise GeminiTtsRateLimitError()
            elif status_code in (400, 422):
                raise GeminiTtsInvalidRequestError()
            elif status_code in (500, 502, 503):
                raise GeminiTtsProviderServerError()
            else:
                raise GeminiTtsError(
                    message=f"Gemini TTS provider returned error status {status_code}",
                    status_code=502
                )

        try:
            data = res.json()
        except Exception as parse_err:
            logger.error(
                f"Gemini TTS malformed JSON response ({type(parse_err).__name__}) [requestId={request_id}]"
            )
            raise GeminiTtsMalformedResponseError()

        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            logger.error(f"Gemini TTS candidates missing or not array [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Missing candidates in Gemini TTS response.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or not isinstance(parts, list):
            logger.error(f"Gemini TTS content parts missing or empty [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Missing content parts in Gemini TTS response.")

        audio_parts = [p for p in parts if "inlineData" in p]
        if len(audio_parts) != 1:
            logger.error(
                f"Gemini TTS response contained {len(audio_parts)} inlineData parts (expected exactly 1) [requestId={request_id}]"
            )
            raise GeminiTtsMalformedResponseError("Response must contain exactly one audio data part.")

        inline_data = audio_parts[0]["inlineData"]
        raw_b64 = inline_data.get("data")
        if not raw_b64 or not isinstance(raw_b64, str):
            logger.error(f"Gemini TTS inlineData missing data field [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Missing inline audio data.")

        # Gemini 3.1 TTS returns audio/L16;codec=pcm;rate=24000.
        mime_type = inline_data.get("mimeType", "")
        normalized_mime_type = mime_type.lower()
        is_supported_pcm = (
            normalized_mime_type.startswith("audio/l16")
            or normalized_mime_type.startswith("audio/pcm")
        )
        if mime_type and not is_supported_pcm:
            logger.error(f"Gemini TTS unexpected MIME type: {mime_type} [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError(f"Unsupported audio MIME type: {mime_type}")

        # Strict Base64 decoding
        try:
            pcm_bytes = base64.b64decode(raw_b64, validate=True)
        except Exception:
            logger.error(f"Gemini TTS invalid base64 encoding [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Invalid Base64 audio content.")

        # Validate non-empty and PCM size limit (max 5 MiB)
        if len(pcm_bytes) == 0:
            logger.error(f"Gemini TTS returned 0-byte PCM audio [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Decoded audio stream is empty.")

        if len(pcm_bytes) > MAX_DECODED_PCM_SIZE_BYTES:
            logger.error(
                f"Gemini TTS PCM size ({len(pcm_bytes)} bytes) exceeds limit ({MAX_DECODED_PCM_SIZE_BYTES} bytes) [requestId={request_id}]"
            )
            raise GeminiTtsMalformedResponseError("Decoded audio stream exceeds size limit of 5 MiB.")

        # Validate 16-bit sample alignment (byte length must be even)
        if len(pcm_bytes) % 2 != 0:
            logger.error(
                f"Gemini TTS PCM size ({len(pcm_bytes)} bytes) is not aligned to 16-bit samples [requestId={request_id}]"
            )
            raise GeminiTtsMalformedResponseError("Decoded audio stream has unaligned sample length.")

        # Convert 24 kHz, 16-bit mono PCM to WAV format
        try:
            wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=24000, channels=1, sample_width=2)
            wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")
        except Exception as wav_err:
            logger.error(f"Gemini TTS WAV header creation failed: {wav_err} [requestId={request_id}]")
            raise GeminiTtsMalformedResponseError("Failed to encode WAV audio stream.")

        usage_meta = data.get("usageMetadata", {})
        usage_dict = {
            "prompt_tokens": usage_meta.get("promptTokenCount", len(text)),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", len(text))
        }

        logger.info(
            f"Gemini TTS request success: status=200, duration={duration_ms}ms, "
            f"model={model}, voice={self.settings.GEMINI_TTS_VOICE}, "
            f"wav_bytes={len(wav_bytes)} [requestId={request_id}]"
        )

        return wav_base64, usage_dict
