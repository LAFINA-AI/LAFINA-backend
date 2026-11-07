"""Server-held ElevenLabs provider, compatible with the existing TTS proxy lifecycle."""

import asyncio
import base64
import time
from urllib.parse import quote

import httpx


class ElevenLabsTtsError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ElevenLabsTtsClient:
    def __init__(self, settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client
        self.owns_client = client is None
        self.used_characters = 0
        self.requests: dict[str, float] = {}

    @property
    def configured(self):
        value = self.settings.ELEVENLABS_API_KEY
        return bool(value and value.get_secret_value().strip())

    async def start(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=12, follow_redirects=False)

    async def close(self):
        if self.client is not None and self.owns_client:
            await self.client.aclose()
            self.client = None

    async def synthesize_speech(self, text: str, request_id: str):
        if not self.configured:
            raise ElevenLabsTtsError("ElevenLabs speech is not configured.", 503)
        if not text.strip() or len(text) > 512:
            raise ElevenLabsTtsError("Speech text must contain 1 to 512 characters.", 400)
        now = time.monotonic()
        self.requests = {
            key: created for key, created in self.requests.items() if now - created < 600
        }
        if request_id in self.requests:
            raise ElevenLabsTtsError("Speech request already submitted; no automatic retry.", 409)
        if self.used_characters + len(text) > self.settings.ELEVENLABS_MAX_CHARACTERS_PER_PROCESS:
            raise ElevenLabsTtsError("Online speech character budget reached.", 429)
        # Reserve before awaiting. Provider failures may still incur charges.
        self.used_characters += len(text)
        self.requests[request_id] = now
        await self.start()
        audio = bytearray()
        try:
            async with asyncio.timeout(14):
                async with self.client.stream(
                    "POST",
                    "https://api.elevenlabs.io/v1/text-to-speech/"
                    + quote(self.settings.ELEVENLABS_VOICE_ID, safe="")
                    + "?output_format=mp3_44100_128",
                    headers={
                        "xi-api-key": self.settings.ELEVENLABS_API_KEY.get_secret_value().strip()
                    },
                    json={"text": text, "model_id": self.settings.ELEVENLABS_MODEL_ID},
                    timeout=12,
                ) as response:
                    if response.status_code == 429:
                        raise ElevenLabsTtsError("ElevenLabs speech usage limit reached.", 429)
                    if response.status_code in (401, 403):
                        raise ElevenLabsTtsError(
                            "ElevenLabs speech configuration needs attention.", 503
                        )
                    if response.status_code != 200:
                        raise ElevenLabsTtsError("ElevenLabs speech is unavailable.")
                    if not response.headers.get("content-type", "").startswith("audio/"):
                        raise ElevenLabsTtsError("Invalid speech audio response.")
                    async for chunk in response.aiter_bytes():
                        audio.extend(chunk)
                        if len(audio) > 2_000_000:
                            raise ElevenLabsTtsError("Speech audio response is too large.")
            if not audio:
                raise ElevenLabsTtsError("Speech audio response was empty.")
            return base64.b64encode(audio).decode("ascii"), {
                "prompt_tokens": len(text),
                "completion_tokens": 0,
            }
        except (TimeoutError, httpx.TimeoutException):
            raise ElevenLabsTtsError("ElevenLabs speech timed out.", 504) from None
        except httpx.HTTPError:
            raise ElevenLabsTtsError("ElevenLabs speech is unavailable.") from None
