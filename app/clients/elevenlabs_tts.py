"""Server-held ElevenLabs provider, compatible with the existing TTS proxy lifecycle."""

import asyncio
import base64
import re
import time
from urllib.parse import quote

import httpx


VOICE_ID_PATTERN = r"[A-Za-z0-9_-]{1,128}"
LABEL_PATTERN = r"[a-z][a-z_-]{0,23}"

# Filipina voices from the ElevenLabs voice library (language "fil"), always
# offered and listed first, with short display names. They work by voice ID
# even before the server account has used them; first use adds them to it.
FILIPINA_VOICES: tuple[tuple[str, str], ...] = (
    ("BtcCEANLn920ZIoN9grW", "Camille"),
    ("n6WaB3rOlZSC9y8yEPEz", "Ate Daniella"),
    ("Lnvev73Hs1s5LVjBfOr9", "Thea Santos"),
    ("6LXO12QQcaZd6xGp9UIZ", "Cielle"),
    ("1y13AZVBVdi3UwMhDhaI", "Amihan Reyes"),
    ("OPouXJLzlYeaXfgM6Q7O", "Ate Weng"),
)
_FILIPINA_ORDER = {voice_id: index for index, (voice_id, _) in enumerate(FILIPINA_VOICES)}


def _label(labels: object, key: str) -> str | None:
    value = labels.get(key) if isinstance(labels, dict) else None
    if isinstance(value, str) and re.fullmatch(LABEL_PATTERN, value.strip().lower()):
        return value.strip().lower()
    return None


def _catalogue_order(voice: dict) -> tuple:
    """Featured Filipina voices first, then other Filipino voices, then by name."""
    if voice["id"] in _FILIPINA_ORDER:
        return (0, _FILIPINA_ORDER[voice["id"]], "")
    return (1 if voice.get("language") == "fil" else 2, 0, voice["name"].casefold())


class ElevenLabsTtsError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ElevenLabsTtsClient:
    MAX_REQUEST_CACHE = 4096
    BUDGET_WINDOW_SECONDS = 86_400.0

    def __init__(self, settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client
        self.owns_client = client is None
        self._used_by_owner: dict[str, tuple[float, int]] = {}
        self.requests: dict[tuple[str, str], float] = {}
        self._voices: list[dict[str, str]] = []
        self._voices_at = 0.0
        self._voices_lock = asyncio.Lock()

    @property
    def used_characters(self) -> int:
        now = time.monotonic()
        return sum(
            count
            for started, count in self._used_by_owner.values()
            if now - started < self.BUDGET_WINDOW_SECONDS
        )

    @used_characters.setter
    def used_characters(self, value: int) -> None:
        if value <= 0:
            self._used_by_owner.clear()
        else:
            self._used_by_owner["__default__"] = (time.monotonic(), int(value))

    def _owner_used(self, owner_key: str, now: float) -> int:
        entry = self._used_by_owner.get(owner_key)
        if not entry:
            return 0
        started, count = entry
        if now - started >= self.BUDGET_WINDOW_SECONDS:
            self._used_by_owner.pop(owner_key, None)
            return 0
        return count

    def _add_owner_usage(self, owner_key: str, delta: int, now: float) -> None:
        current = self._owner_used(owner_key, now)
        started = self._used_by_owner[owner_key][0] if owner_key in self._used_by_owner else now
        new_val = max(0, current + delta)
        if new_val == 0:
            self._used_by_owner.pop(owner_key, None)
        else:
            self._used_by_owner[owner_key] = (started, new_val)

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
                        raw_body = bytearray()
                        async with self.client.stream(
                            "GET",
                            "https://api.elevenlabs.io/v2/voices",
                            params=params,
                            headers={"xi-api-key": self.settings.ELEVENLABS_API_KEY.get_secret_value().strip()},
                            timeout=10,
                        ) as response:
                            if response.status_code == 429:
                                raise ElevenLabsTtsError("Voice catalogue usage limit reached.", 429)
                            if response.status_code in (401, 403):
                                raise ElevenLabsTtsError("The server key needs Voices read permission.", 503)
                            if response.status_code != 200:
                                raise ElevenLabsTtsError("Voice catalogue is unavailable. Try again.")
                            async for chunk in response.aiter_bytes():
                                raw_body.extend(chunk)
                                if len(raw_body) > 2_000_000:
                                    raise ElevenLabsTtsError("Voice catalogue response is too large.")
                        import json
                        data = json.loads(raw_body.decode("utf-8"))
                        if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
                            raise ElevenLabsTtsError("Invalid voice catalogue response.")
                        for voice in data["voices"][:100]:
                            if not isinstance(voice, dict):
                                continue
                            voice_id, name = voice.get("voice_id"), voice.get("name")
                            if isinstance(voice_id, str) and re.fullmatch(VOICE_ID_PATTERN, voice_id) and isinstance(name, str) and name.strip():
                                entry = {"id": voice_id, "name": name.strip()[:160]}
                                for key in ("language", "gender"):
                                    value = _label(voice.get("labels"), key)
                                    if value:
                                        entry[key] = value
                                voices[voice_id] = entry
                        cursor = data.get("next_page_token")
                        if not data.get("has_more") or not isinstance(cursor, str) or not cursor:
                            break
                for voice_id, name in FILIPINA_VOICES:
                    voices[voice_id] = {"id": voice_id, "name": name, "language": "fil", "gender": "female"}
                self._voices = sorted(voices.values(), key=_catalogue_order)
                self._voices_at = time.monotonic()
                return list(self._voices)
            except ElevenLabsTtsError:
                raise
            except (TimeoutError, httpx.TimeoutException):
                raise ElevenLabsTtsError("Voice catalogue timed out. Try again.", 504) from None
            except (httpx.HTTPError, ValueError, UnicodeDecodeError):
                raise ElevenLabsTtsError("Voice catalogue is unavailable. Try again.") from None

    def model_for(self, latency: str = "standard") -> str:
        """`low` (a spoken conversation) uses the fast model when one is set."""
        if latency == "low":
            fast = getattr(self.settings, "ELEVENLABS_FAST_MODEL_ID", None)
            if fast:
                return fast
        return self.settings.ELEVENLABS_MODEL_ID

    async def synthesize_speech(
        self,
        text: str,
        request_id: str,
        voice_id: str | None = None,
        latency: str = "standard",
        owner_id: str | None = None,
    ):
        if not self.configured:
            raise ElevenLabsTtsError("ElevenLabs speech is not configured.", 503)
        if not text.strip() or len(text) > 512:
            raise ElevenLabsTtsError("Speech text must contain 1 to 512 characters.", 400)
        selected_voice = voice_id or self.settings.ELEVENLABS_VOICE_ID
        if voice_id is not None:
            if not re.fullmatch(VOICE_ID_PATTERN, voice_id):
                raise ElevenLabsTtsError("Invalid online voice selection.", 400)
            # Featured voices need no catalogue lookup.
            if voice_id not in _FILIPINA_ORDER and not any(
                voice["id"] == voice_id for voice in await self.list_voices()
            ):
                raise ElevenLabsTtsError("That online voice is no longer available. Choose another voice in Profile.", 422)
        now = time.monotonic()
        owner_key = str(owner_id) if owner_id else "__default__"
        req_key = (owner_key, str(request_id)[:128])
        self.requests = {
            key: created for key, created in self.requests.items() if now - created < 600
        }
        if len(self.requests) >= self.MAX_REQUEST_CACHE:
            oldest = sorted(self.requests.items(), key=lambda item: item[1])[
                : len(self.requests) - self.MAX_REQUEST_CACHE + 1
            ]
            for old_key, _ in oldest:
                self.requests.pop(old_key, None)
        if req_key in self.requests:
            raise ElevenLabsTtsError("Speech request already submitted; no automatic retry.", 409)
        if self._owner_used(owner_key, now) + len(text) > self.settings.ELEVENLABS_MAX_CHARACTERS_PER_PROCESS:
            raise ElevenLabsTtsError("Online speech character budget reached.", 429)
        # Reserve per-owner budget before awaiting; refund on pre-bill auth/outage failures.
        self._add_owner_usage(owner_key, len(text), now)
        self.requests[req_key] = now
        await self.start()
        model = self.model_for(latency)
        audio = bytearray()
        refund = False
        try:
            async with asyncio.timeout(14):
                async with self.client.stream(
                    "POST",
                    "https://api.elevenlabs.io/v1/text-to-speech/"
                    + quote(selected_voice, safe="")
                    + "?output_format="
                    # Half the bytes for conversation; speech still sounds clear.
                    + ("mp3_44100_64" if latency == "low" else "mp3_44100_128"),
                    headers={
                        "xi-api-key": self.settings.ELEVENLABS_API_KEY.get_secret_value().strip()
                    },
                    json={"text": text, "model_id": model},
                    timeout=12,
                ) as response:
                    if response.status_code == 429:
                        refund = True
                        raise ElevenLabsTtsError("ElevenLabs speech usage limit reached.", 429)
                    if response.status_code in (401, 403):
                        refund = True
                        raise ElevenLabsTtsError(
                            "ElevenLabs speech configuration needs attention.", 503
                        )
                    if response.status_code != 200:
                        if response.status_code >= 500:
                            refund = True
                        raise ElevenLabsTtsError("ElevenLabs speech is unavailable.")
                    if not response.headers.get("content-type", "").lower().startswith("audio/"):
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
                "model": model,
            }
        except (TimeoutError, httpx.TimeoutException):
            raise ElevenLabsTtsError("ElevenLabs speech timed out.", 504) from None
        except httpx.HTTPError:
            refund = True
            raise ElevenLabsTtsError("ElevenLabs speech is unavailable.") from None
        finally:
            if refund:
                self._add_owner_usage(owner_key, -len(text), time.monotonic())
                self.requests.pop(req_key, None)
