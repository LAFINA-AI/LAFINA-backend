from sqlalchemy.pool import NullPool

from app.config import Settings, is_pooled_database_url, normalize_database_url
from app.database import build_engine_kwargs


def test_normalize_database_url_converts_neon_pooled_and_direct_urls() -> None:
    """Neon `postgresql://` URLs with libpq params must be normalized for asyncpg."""
    neon_pooled = (
        "postgresql://neondb_owner:secret@ep-cool-fog-123456-pooler.ap-southeast-1.aws.neon.tech"
        "/neondb?sslmode=require&channel_binding=require"
    )
    normalized = normalize_database_url(neon_pooled)
    assert normalized == (
        "postgresql+asyncpg://neondb_owner:secret@ep-cool-fog-123456-pooler.ap-southeast-1.aws.neon.tech"
        "/neondb?ssl=require"
    )
    assert is_pooled_database_url(normalized) is True

    neon_direct = (
        "postgres://neondb_owner:secret@ep-cool-fog-123456.ap-southeast-1.aws.neon.tech"
        "/neondb?sslmode=require"
    )
    normalized_direct = normalize_database_url(neon_direct)
    assert normalized_direct == (
        "postgresql+asyncpg://neondb_owner:secret@ep-cool-fog-123456.ap-southeast-1.aws.neon.tech"
        "/neondb?ssl=require"
    )
    assert is_pooled_database_url(normalized_direct) is False


def test_normalize_database_url_preserves_sqlite() -> None:
    """SQLite URLs used for local/offline-sync tests must map to aiosqlite."""
    assert normalize_database_url("sqlite:///./local.db") == "sqlite+aiosqlite:///./local.db"
    mem_url = "sqlite+aiosqlite:///file:memdb1?mode=memory&cache=shared&uri=true"
    assert normalize_database_url(mem_url) == mem_url


def test_migration_url_prefers_unpooled_neon_connection() -> None:
    """Alembic migrations must use DATABASE_URL_UNPOOLED (direct) over pooled DATABASE_URL."""
    settings = Settings(
        ENVIRONMENT="development",
        DATABASE_URL="postgresql://u:p@ep-abc-pooler.aws.neon.tech/neondb?sslmode=require",
        DATABASE_URL_UNPOOLED="postgresql://u:p@ep-abc.aws.neon.tech/neondb?sslmode=require",
        _env_file=None,
    )
    assert "-pooler." in settings.get_async_database_url()
    assert "-pooler." not in settings.get_migration_database_url()


def test_vercel_postgres_env_aliases_are_supported() -> None:
    """Vercel's Neon integration variables (POSTGRES_URL / POSTGRES_URL_NON_POOLING) work automatically."""
    settings = Settings(
        ENVIRONMENT="development",
        POSTGRES_URL="postgres://u:p@ep-xyz-pooler.aws.neon.tech/neondb?sslmode=require",
        POSTGRES_URL_NON_POOLING="postgres://u:p@ep-xyz.aws.neon.tech/neondb?sslmode=require",
        _env_file=None,
    )
    assert settings.get_async_database_url().startswith("postgresql+asyncpg://u:p@ep-xyz-pooler.")
    assert settings.get_migration_database_url().startswith("postgresql+asyncpg://u:p@ep-xyz.")


def test_build_engine_kwargs_disables_statement_cache_on_neon_pooler() -> None:
    """PgBouncer transaction pooling on `-pooler` endpoints requires statement_cache_size=0 and NullPool."""
    url, kwargs = build_engine_kwargs(
        "postgresql://u:p@ep-abc-pooler.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    assert url.startswith("postgresql+asyncpg://")
    assert kwargs["connect_args"] == {"statement_cache_size": 0}
    assert kwargs["poolclass"] is NullPool
