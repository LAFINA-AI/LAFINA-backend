"""The Pinecone index that holds the Student Handbook, over its REST API.

Pinecone does both jobs: its hosted embedding model (Pinecone Inference)
turns text into vectors, and the index stores and searches them. Passages and
questions are embedded with different input types — the model places a
question nearer the passage that answers it than one that merely repeats it.

Plain httpx rather than the Pinecone SDK: the backend needs five calls —
embed, query, upsert, delete a namespace, count what is in one — and the
deployment stays free of another dependency tree. The text being embedded and
the API key never reach a log line.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote, urlsplit

import httpx

from app.config import Settings

logger = logging.getLogger("lafina.pinecone")

# Pinecone takes at most 1,000 vectors or 2 MB per upsert; a handbook chunk
# with its text in the metadata is a few kilobytes, so 100 stays well inside.
UPSERT_BATCH = 100
# llama-text-embed-v2 takes at most 96 texts per request.
EMBED_BATCH = 96

INPUT_PASSAGE = "passage"
INPUT_QUERY = "query"


class PineconeError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class PineconeMatch:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PineconeIndexClient:
    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None):
        self.settings = settings
        self._custom_client = client
        self._client: Optional[httpx.AsyncClient] = client
        self._host: str | None = (settings.PINECONE_INDEX_HOST or "").strip() or None

    async def start(self) -> None:
        if self._custom_client is not None:
            self._client = self._custom_client
        elif self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    async def close(self) -> None:
        if self._custom_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def configured(self) -> bool:
        return self.settings.get_pinecone_invalid_reason() is None

    def _headers(self) -> dict[str, str]:
        key = self.settings.PINECONE_API_KEY.get_secret_value().strip().strip("'\"")
        return {
            "Api-Key": key,
            "X-Pinecone-API-Version": self.settings.PINECONE_API_VERSION,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, url: str, **kwargs) -> dict:
        if not self.configured:
            raise PineconeError("Pinecone is not configured.", status_code=503)
        if self._client is None:
            await self.start()
        assert self._client is not None
        try:
            response = await self._client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.TimeoutException:
            raise PineconeError("Pinecone request timed out.", status_code=504)
        except httpx.RequestError as exc:
            logger.warning(f"Pinecone transport failure ({type(exc).__name__})")
            raise PineconeError("Could not reach Pinecone.", status_code=503)
        if response.status_code >= 400:
            # Pinecone's error bodies are its own wording, never our key.
            detail = response.text[:300]
            logger.warning(f"Pinecone {method} {url.split('?')[0]} -> {response.status_code}: {detail}")
            raise PineconeError(
                f"Pinecone returned HTTP {response.status_code}.",
                status_code=429 if response.status_code == 429 else 503,
            )
        if not response.content:
            return {}
        try:
            parsed = response.json()
        except ValueError:
            raise PineconeError("Pinecone returned a malformed response.")
        if not isinstance(parsed, dict):
            raise PineconeError("Pinecone returned an unexpected response shape.")
        return parsed

    @staticmethod
    def _normalize_host(raw_host: str) -> str:
        candidate = raw_host.strip().rstrip("/")
        if not candidate:
            raise PineconeError("Pinecone did not report a host for the index.")
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        parts = urlsplit(candidate)
        if parts.scheme != "https" or not parts.netloc:
            raise PineconeError("Pinecone index host must be a valid HTTPS URL.", status_code=503)
        return f"https://{parts.netloc}{parts.path.rstrip('/')}"

    async def host(self) -> str:
        """The index's data-plane host, looked up once and remembered."""
        if self._host is None:
            name = quote(self.settings.PINECONE_INDEX_NAME.strip(), safe="")
            described = await self._request(
                "GET", f"{self.settings.PINECONE_CONTROL_URL.rstrip('/')}/indexes/{name}"
            )
            host = str(described.get("host") or "").strip()
            normalized = self._normalize_host(host)
            dimension = described.get("dimension")
            if dimension is not None:
                try:
                    dim_int = int(dimension)
                except (TypeError, ValueError):
                    raise PineconeError("Pinecone reported a malformed index dimension.")
                if dim_int and dim_int != self.settings.HANDBOOK_EMBED_DIMENSIONS:
                    raise PineconeError(
                        f"The Pinecone index is {dim_int}-dimensional but embeddings are "
                        f"{self.settings.HANDBOOK_EMBED_DIMENSIONS}-dimensional.",
                        status_code=503,
                    )
            self._host = normalized
            return normalized
        return self._normalize_host(self._host)

    async def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        """One vector per text, in order, from Pinecone's hosted embedding model."""
        dimensions = self.settings.HANDBOOK_EMBED_DIMENSIONS
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            data = await self._request(
                "POST",
                f"{self.settings.PINECONE_CONTROL_URL.rstrip('/')}/embed",
                json={
                    "model": self.settings.HANDBOOK_EMBED_MODEL,
                    "parameters": {
                        "input_type": input_type,
                        "truncate": "END",
                        "dimension": dimensions,
                    },
                    "inputs": [{"text": text or " "} for text in batch],
                },
            )
            try:
                embedded = [list(map(float, item["values"])) for item in data["data"]]
            except (KeyError, TypeError, ValueError):
                raise PineconeError("Pinecone returned a malformed embedding response.")
            if len(embedded) != len(batch) or any(len(v) != dimensions for v in embedded):
                raise PineconeError("Pinecone returned embeddings of the wrong shape.")
            vectors.extend(embedded)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text], input_type=INPUT_QUERY))[0]

    async def query(self, vector: list[float], *, top_k: int, namespace: str) -> list[PineconeMatch]:
        data = await self._request(
            "POST",
            f"{await self.host()}/query",
            json={
                "vector": vector,
                "topK": top_k,
                "namespace": namespace,
                "includeMetadata": True,
                "includeValues": False,
            },
        )
        raw_matches = data.get("matches")
        if raw_matches is None:
            raw_matches = []
        if not isinstance(raw_matches, list):
            raise PineconeError("Pinecone returned a malformed matches list.")
        results: list[PineconeMatch] = []
        for match in raw_matches:
            if not isinstance(match, dict):
                continue
            try:
                score = float(match.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            meta = match.get("metadata")
            results.append(
                PineconeMatch(
                    id=str(match.get("id", "")),
                    score=score,
                    metadata=dict(meta) if isinstance(meta, dict) else {},
                )
            )
        return results

    async def upsert(self, vectors: list[dict[str, Any]], *, namespace: str) -> int:
        written = 0
        for start in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[start : start + UPSERT_BATCH]
            data = await self._request(
                "POST",
                f"{await self.host()}/vectors/upsert",
                json={"vectors": batch, "namespace": namespace},
            )
            raw_count = data.get("upsertedCount")
            try:
                written += int(raw_count) if raw_count is not None else len(batch)
            except (TypeError, ValueError):
                written += len(batch)
        return written

    async def delete_namespace(self, namespace: str) -> None:
        """Deletes every vector in one namespace. Other namespaces are untouched."""
        try:
            await self._request(
                "POST",
                f"{await self.host()}/vectors/delete",
                json={"deleteAll": True, "namespace": namespace},
            )
        except PineconeError as err:
            # Deleting a namespace that does not exist yet is not a failure.
            if "404" not in err.message:
                raise

    async def namespace_count(self, namespace: str) -> int:
        stats = await self._request("POST", f"{await self.host()}/describe_index_stats", json={})
        namespaces = stats.get("namespaces")
        if not isinstance(namespaces, dict):
            return 0
        entry = namespaces.get(namespace)
        if not isinstance(entry, dict):
            return 0
        try:
            return int(entry.get("vectorCount") or 0)
        except (TypeError, ValueError):
            raise PineconeError("Pinecone returned a malformed namespace count.")
