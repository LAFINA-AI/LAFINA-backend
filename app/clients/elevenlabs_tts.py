"""Server-held ElevenLabs provider, compatible with the existing TTS proxy lifecycle."""

import asyncio
import base64
import re
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
        self._voices: list[dict[str, str]] = []
        self._voices_at = 0.0
        self._voices_lock = asyncio.Lock()

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

    async def list_voices(self) -> list[dict[str, str]]:
        """Bounded, shared five-minute catalogue; no audio or arbitrary URLs."""
        if not self.configured:
            raise ElevenLabsTtsError("ElevenLabs speech is not configured.", 503)
        async with self._voices_lock:
            if self._voices_at and time.monotonic() - self._voices_at < 300:
                return list(self._voices)
            await self.start()
            voices: dict[str, dict[str, str]] = {}
            cursor = None
            try:
                async with asyncio.timeout(12):
                    for _ in range(5):
                        params = {"page_size": 100}
                        if cursor:
                            params["next_page_token"] = cursor
                        response = await self.client.get(
                            "https://api.elevenlabs.io/v2/voices",
                            params=params,
                            headers={"xi-api-key": self.settings.ELEVENLABS_API_KEY.get_secret_value().strip()},
                            timeout=10,
                        )
                        if response.status_code == 429:
                            raise ElevenLabsTtsError("Voice catalogue usage limit reached.", 429)
                        if response.status_code in (401, 403):
                            raise ElevenLabsTtsError("The server key needs Voices read permission.", 503)
                        if response.status_code != 200:
                            raise ElevenLabsTtsError("Voice catalogue is unavailable. Try again.")
                        if len(response.content) > 2_000_000:
                            raise ElevenLabsTtsError("Voice catalogue response is too large.")
                        data = response.json()
                        if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
                            raise ElevenLabsTtsError("Invalid voice catalogue response.")
                        for voice in data["voices"][:100]:
                            if not isinstance(voice, dict):
                                continue
                            voice_id, name = voice.get("voice_id"), voice.get("name")
                            if isinstance(voice_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", voice_id) and isinstance(name, str) and name.strip():
                                voices[voice_id] = {"id": voice_id, "name": name.strip()[:160]}
                        cursor = data.get("next_page_token")
                        if not data.get("has_more") or not isinstance(cursor, str) or not cursor:
                            break
                self._voices = sorted(voices.values(), key=lambda v: v["name"].casefold())
                self._voices_at = time.monotonic()
                return list(self._voices)
            except (TimeoutError, httpx.TimeoutException):
                raise ElevenLabsTtsError("Voice catalogue timed out. Try again.", 504) from None
            except (httpx.HTTPError, ValueError):
                raise ElevenLabsTtsError("Voice catalogue is unavailable. Try again.") from None

    async def synthesize_speech(self, text: str, request_id: str, voice_id: str | None = None):
        if not self.configured:
            raise ElevenLabsTtsError("ElevenLabs speech is not configured.", 503)
        if not text.strip() or len(text) > 512:
            raise ElevenLabsTtsError("Speech text must contain 1 to 512 characters.", 400)
        selected_voice = voice_id or self.settings.ELEVENLABS_VOICE_ID
        if voice_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", voice_id):
                raise ElevenLabsTtsError("Invalid online voice selection.", 400)
            voices = await self.list_voices()
            if not any(voice["id"] == voice_id for voice in voices):
                raise ElevenLabsTtsError("That online voice is no longer available. Choose another voice in Profile.", 422)
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
                    + quote(selected_voice, safe="")
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
