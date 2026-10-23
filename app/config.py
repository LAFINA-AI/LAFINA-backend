import functools
import os
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Generate default RSA key pair for development/testing if not provided via env
def _generate_dev_rsa_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")
    return private_pem, public_pem


_dev_priv_pem, _dev_pub_pem = _generate_dev_rsa_pair()

INVALID_PLACEHOLDER_KEYS = {
    "",
    "mock-deepseek-key-for-dev",
    "your-deepseek-api-key",
    "sk-your-deepseek-api-key",
    "placeholder",
    "your_deepseek_api_key",
    "change-me",
}

# libpq-only query parameters that Neon / Vercel connection strings include
# (`?sslmode=require&channel_binding=require`) which `asyncpg.connect()` does
# not accept as keyword arguments.
_LIBPQ_ONLY_PARAMS = {
    "channel_binding",
    "pgbouncer",
    "connect_timeout",
    "options",
    "sslnegotiation",
    "application_name",
}

_DEFAULT_LOCAL_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/lafina"


def normalize_database_url(raw_url: str) -> str:
    """Normalize a Postgres or SQLite URL for SQLAlchemy's async drivers.

    Neon and Vercel provide `postgres://` or `postgresql://` URLs with libpq
    query parameters such as `?sslmode=require&channel_binding=require`.
    SQLAlchemy's `asyncpg` dialect needs the `postgresql+asyncpg://` scheme,
    `ssl=` instead of `sslmode=`, and no libpq-only query parameters.
    SQLite URLs (`sqlite://` or `sqlite+aiosqlite://`) are preserved for local
    and unit-test databases.
    """
    cleaned = raw_url.strip().strip("'\"")
    if not cleaned:
        return _DEFAULT_LOCAL_DB_URL

    if cleaned.startswith("sqlite://") and not cleaned.startswith("sqlite+aiosqlite://"):
        return "sqlite+aiosqlite://" + cleaned[len("sqlite://"):]
    if cleaned.startswith("sqlite+aiosqlite://"):
        return cleaned

    for prefix in (
        "postgres://",
        "postgresql://",
        "postgresql+psycopg://",
        "postgresql+psycopg2://",
    ):
        if cleaned.startswith(prefix):
            cleaned = "postgresql+asyncpg://" + cleaned[len(prefix):]
            break

    if not cleaned.startswith("postgresql+asyncpg://"):
        return cleaned

    parts = urlsplit(cleaned)
    if not parts.query:
        return cleaned

    raw_params = parse_qsl(parts.query, keep_blank_values=True)
    has_explicit_ssl = any(k == "ssl" for k, _ in raw_params)
    filtered_params: list[tuple[str, str]] = []

    for key, value in raw_params:
        if key == "sslmode":
            if not has_explicit_ssl and value:
                filtered_params.append(("ssl", value))
            continue
        if key in _LIBPQ_ONLY_PARAMS:
            continue
        filtered_params.append((key, value))

    new_query = urlencode(filtered_params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def is_pooled_database_url(url: str) -> bool:
    """True when the URL points at a Neon PgBouncer `-pooler` endpoint."""
    try:
        hostname = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return "-pooler." in hostname or hostname.endswith("-pooler")


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    API_BASE_URL: str = "http://localhost:8000"

    # Neon / Postgres connections:
    # - DATABASE_URL: pooled connection (`-pooler`) for Vercel serverless & web traffic
    # - DATABASE_URL_UNPOOLED: direct connection (no `-pooler`) for Alembic migrations
    # Vercel's Neon integration also injects POSTGRES_URL and POSTGRES_URL_NON_POOLING.
    DATABASE_URL: str = _DEFAULT_LOCAL_DB_URL
    DATABASE_URL_UNPOOLED: Optional[str] = None
    POSTGRES_URL: Optional[str] = None
    POSTGRES_URL_NON_POOLING: Optional[str] = None

    # Optional legacy Render Postgres URL used by `scripts/migrate_render_to_neon.py`.
    RENDER_DATABASE_URL: Optional[str] = None

    # Set automatically on Vercel serverless deployments.
    VERCEL: Optional[str] = None
    VERCEL_URL: Optional[str] = None

    # Optional Admin Account Seeding from Environment Variables
    ADMIN_EMAIL: Optional[str] = None
    ADMIN_PASSWORD: Optional[str] = None

    # JWT RS256 Configuration
    JWT_PRIVATE_KEY: str = _dev_priv_pem
    JWT_PUBLIC_KEY: str = _dev_pub_pem
    JWT_ISSUER: str = "lafina-auth"
    JWT_AUDIENCE: str = "lafina-app"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Argon2id Configuration (NIST SP 800-63B & OWASP: >= 19MiB memory, 2 iterations, 1 parallelism)
    ARGON2_MEMORY_COST_KIB: int = 19456  # 19 MiB
    ARGON2_TIME_COST: int = 2
    ARGON2_PARALLELISM: int = 1

    # Security & Rate Limiting
    MAX_BODY_SIZE_BYTES: int = 1048576  # 1 MiB
    # A document upload is the one body that is legitimately large. It gets its
    # own ceiling so the 1 MiB rule can stay tight for everything else; the
    # flashcards endpoint checks the decoded PDF against its own limit again.
    MAX_UPLOAD_BODY_SIZE_BYTES: int = 22020096  # 21 MiB: 15 MB of PDF, base64
    MAX_LOGIN_FAILURES_PER_15MIN: int = 10
    MAX_REGISTRATIONS_PER_IP_PER_HOUR: int = 100
    MAX_AI_REQUESTS_PER_MIN: int = 10
    MAX_AI_REQUESTS_PER_DAY: int = 100
    MAX_TTS_REQUESTS_PER_MIN: int = 10
    MAX_TTS_REQUESTS_PER_DAY: int = 100

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    # Flashcards need a model that follows a strict output shape over a long
    # reply, which is a different job from the conversational assistant.
    DEEPSEEK_FLASHCARD_MODEL: str = "deepseek-chat"
    DEEPSEEK_TIMEOUT_SECONDS: float = 120.0

    # Gemini TTS Configuration
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"
    GEMINI_TTS_MODEL: str = "gemini-3.1-flash-tts-preview"
    GEMINI_TTS_VOICE: str = "Aoede"
    GEMINI_TTS_TIMEOUT_SECONDS: float = 20.0

    # Student Handbook RAG. The handbook lives in a Pinecone index, embedded by
    # Pinecone's own hosted model, and the online assistant quotes it when a
    # question is about the university. Switched on and off at runtime from the
    # admin panel (the `handbook_rag` feature flag) — these settings only say
    # where it lives.
    PINECONE_API_KEY: SecretStr | None = None
    PINECONE_INDEX_NAME: Optional[str] = None
    # Skips the control-plane lookup of the index host when set.
    PINECONE_INDEX_HOST: Optional[str] = None
    PINECONE_CONTROL_URL: str = "https://api.pinecone.io"
    PINECONE_API_VERSION: str = "2025-04"
    HANDBOOK_NAMESPACE: str = "ustp-handbook-2023"
    HANDBOOK_TITLE: str = "USTP Student Handbook 2023"
    # Pinecone Inference. Must match the index: `lafina-rag` is 768-dimensional,
    # cosine, and llama-text-embed-v2 can produce 768 dimensions.
    HANDBOOK_EMBED_MODEL: str = "llama-text-embed-v2"
    HANDBOOK_EMBED_DIMENSIONS: int = 768
    HANDBOOK_TOP_K: int = 5
    # Passages scoring below this are not about the question and are dropped.
    # Measured against llama-text-embed-v2 on the 2023 handbook: questions
    # about the university score 0.30-0.57, unrelated ones 0.27 at most.
    HANDBOOK_MIN_SCORE: float = 0.30
    # Passages this far below the best match are left out too: they are what
    # the index returns to fill the top-k, not what the question is about.
    HANDBOOK_RELATIVE_MARGIN: float = 0.15
    # The chat never waits longer than this for the handbook; past it the
    # reply goes ahead without handbook context.
    HANDBOOK_TIMEOUT_SECONDS: float = 8.0

    # Password blocklist (common passwords to reject)
    COMMON_PASSWORDS: set[str] = {
        "password", "password123", "1234567890", "12345678", "qwertyuiop",
        "administrator", "letmein123", "welcome123", "changeme123", "lafina12345"
    }

    def _is_placeholder_url(self, url: str | None) -> bool:
        if not url:
            return True
        u = url.lower()
        return any(
            placeholder in u
            for placeholder in (
                "neon_user",
                "neon_password",
                "ep-example",
                "render_db_user",
                "render_internal_db_host",
            )
        )

    def get_async_database_url(self) -> str:
        """Return the normalized application DATABASE_URL (pooled on Neon)."""
        raw = self.DATABASE_URL
        if (
            not raw
            or raw == _DEFAULT_LOCAL_DB_URL
            or self._is_placeholder_url(raw)
        ) and self.POSTGRES_URL and not self._is_placeholder_url(self.POSTGRES_URL):
            raw = self.POSTGRES_URL
        return normalize_database_url(raw)

    def get_migration_database_url(self) -> str:
        """Return the direct (unpooled) database URL for Alembic migrations and dumps.

        Per Neon best practices, schema migrations must run over the direct
        connection (`DATABASE_URL_UNPOOLED` / `POSTGRES_URL_NON_POOLING`) rather
        than the PgBouncer `-pooler` endpoint.
        """
        raw = self.DATABASE_URL_UNPOOLED
        if self._is_placeholder_url(raw):
            raw = None
        if (
            not raw
            and self.POSTGRES_URL_NON_POOLING
            and not self._is_placeholder_url(self.POSTGRES_URL_NON_POOLING)
        ):
            raw = self.POSTGRES_URL_NON_POOLING
        if raw:
            return normalize_database_url(raw)
        return self.get_async_database_url()

    def is_vercel_runtime(self) -> bool:
        """True when executing inside a Vercel serverless function."""
        return bool(self.VERCEL or os.environ.get("VERCEL"))

    def get_deepseek_key_invalid_reason(self) -> str | None:
        if self.DEEPSEEK_API_KEY is None:
            return "DEEPSEEK_API_KEY environment variable is not set (None)"
        raw_key = self.DEEPSEEK_API_KEY.get_secret_value().strip().strip("'\"")
        if not raw_key:
            return "DEEPSEEK_API_KEY is blank or empty"
        if raw_key.lower() in INVALID_PLACEHOLDER_KEYS:
            return f"DEEPSEEK_API_KEY is set to a placeholder value ('{raw_key}')"
        return None

    def is_deepseek_key_valid(self) -> bool:
        return self.get_deepseek_key_invalid_reason() is None

    def get_gemini_key_invalid_reason(self) -> str | None:
        if self.GEMINI_API_KEY is None:
            return "GEMINI_API_KEY environment variable is not set (None)"
        raw_key = self.GEMINI_API_KEY.get_secret_value().strip().strip("'\"")
        if not raw_key:
            return "GEMINI_API_KEY is blank or empty"
        if raw_key.lower() in INVALID_PLACEHOLDER_KEYS:
            return f"GEMINI_API_KEY is set to a placeholder value ('{raw_key}')"
        return None

    def is_gemini_key_valid(self) -> bool:
        return self.get_gemini_key_invalid_reason() is None

    def get_pinecone_invalid_reason(self) -> str | None:
        if self.PINECONE_API_KEY is None:
            return "PINECONE_API_KEY environment variable is not set (None)"
        raw_key = self.PINECONE_API_KEY.get_secret_value().strip().strip("'\"")
        if not raw_key or raw_key.lower() in INVALID_PLACEHOLDER_KEYS:
            return "PINECONE_API_KEY is blank or a placeholder"
        if not (self.PINECONE_INDEX_NAME or "").strip():
            return "PINECONE_INDEX_NAME environment variable is not set"
        return None

    def is_handbook_configured(self) -> bool:
        """Pinecone both embeds and stores the handbook, so its key and index are all it needs."""
        return self.get_pinecone_invalid_reason() is None

    @model_validator(mode="after")
    def validate_deepseek_config(self) -> "Settings":
        reason = self.get_deepseek_key_invalid_reason()
        if self.ENVIRONMENT == "production" and reason is not None:
            raise ValueError(
                f"DEEPSEEK_API_KEY secret environment variable must be configured with a valid key in production ({reason})."
            )
        return self

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@functools.lru_cache()
def get_settings() -> Settings:
    return Settings()
