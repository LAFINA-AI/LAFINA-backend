"""Server-side half of a live ElevenAgents conversation: minting signed URLs.

The phone never sees the ElevenLabs API key. It asks this server for a
session, and the server asks ElevenLabs for a signed WebSocket URL for the
configured private agent (`enable_auth: true`). That URL is a bearer
credential for one conversation start; it is returned once and never logged.
"""

import asyncio
from urllib.parse import urlparse

import httpx

SIGNED_URL_ENDPOINT = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"


class ElevenLabsAgentError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _is_elevenlabs_socket(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "wss" and (host == "elevenlabs.io" or host.endswith(".elevenlabs.io"))


class ElevenLabsAgentClient:
    def __init__(self, settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client
        self.owns_client = client is None

    @property
    def configured(self) -> bool:
        key = self.settings.ELEVENLABS_API_KEY
        secret = self.settings.ELEVENLABS_AGENT_LLM_SECRET
        return bool(
            key
            and key.get_secret_value().strip()
            and (self.settings.ELEVENLABS_AGENT_ID or "").strip()
            and secret
            and secret.get_secret_value().strip()
        )

    def missing_configuration(self) -> str | None:
        """Names what is missing, for operators; never a secret value."""
        missing = []
        key = self.settings.ELEVENLABS_API_KEY
        if not (key and key.get_secret_value().strip()):
            missing.append("ELEVENLABS_API_KEY")
        if not (self.settings.ELEVENLABS_AGENT_ID or "").strip():
            missing.append("ELEVENLABS_AGENT_ID")
        secret = self.settings.ELEVENLABS_AGENT_LLM_SECRET
        if not (secret and secret.get_secret_value().strip()):
            missing.append("ELEVENLABS_AGENT_LLM_SECRET")
        return ", ".join(missing) or None

    async def start(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=10, follow_redirects=False)

    async def close(self):
        if self.client is not None and self.owns_client:
            await self.client.aclose()
            self.client = None

    async def signed_url(self) -> str:
        """A fresh signed URL (valid about 15 minutes to start one conversation)."""
        if not self.configured:
            raise ElevenLabsAgentError("Voice conversation is not configured on the server.", 503)
        await self.start()
        try:
            async with asyncio.timeout(10):
                response = await self.client.get(
                    SIGNED_URL_ENDPOINT,
                    params={"agent_id": self.settings.ELEVENLABS_AGENT_ID.strip()},
                    headers={
                        "xi-api-key": self.settings.ELEVENLABS_API_KEY.get_secret_value().strip()
                    },
                    timeout=8,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise ElevenLabsAgentError(
                "ElevenLabs did not answer in time. Try again.", 504
            ) from None
        except httpx.HTTPError:
            raise ElevenLabsAgentError("ElevenLabs could not be reached. Try again.", 503) from None
        if response.status_code == 429:
            raise ElevenLabsAgentError(
                "ElevenLabs is busy or out of conversation capacity. Try later.", 429
            )
        if response.status_code in (401, 403):
            raise ElevenLabsAgentError(
                "The server's ElevenLabs key needs the ElevenAgents (Conversational AI) permission.",
                503,
            )
        if response.status_code == 404:
            raise ElevenLabsAgentError(
                "The configured voice agent was not found on ElevenLabs.", 503
            )
        if response.status_code != 200:
            raise ElevenLabsAgentError("ElevenLabs could not start a conversation. Try again.", 502)
        try:
            data = response.json()
        except ValueError:
            raise ElevenLabsAgentError("ElevenLabs sent an unreadable answer.", 502) from None
        url = data.get("signed_url") if isinstance(data, dict) else None
        if not isinstance(url, str) or len(url) > 4096 or not _is_elevenlabs_socket(url):
            raise ElevenLabsAgentError("ElevenLabs sent an invalid conversation address.", 502)
        return url
