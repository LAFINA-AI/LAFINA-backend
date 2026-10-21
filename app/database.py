import uuid
from typing import Any, AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings, is_pooled_database_url, normalize_database_url

settings = get_settings()


def build_engine_kwargs(raw_url: str, *, is_serverless: bool = False) -> tuple[str, dict[str, Any]]:
    """Build normalized URL and SQLAlchemy async engine kwargs for Neon, Postgres, or SQLite.

    - Neon `-pooler` endpoints run PgBouncer in transaction mode, so asyncpg's
      prepared statement cache must be disabled (`statement_cache_size=0`) and
      client-side pooling is delegated to PgBouncer (`NullPool`).
    - Vercel serverless functions also use `NullPool` so connections are not
      held across short-lived function invocations.
    """
    normalized_url = normalize_database_url(raw_url)
    kwargs: dict[str, Any] = {
        "echo": False,
        "future": True,
    }

    if normalized_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        return normalized_url, kwargs

    pooled_endpoint = is_pooled_database_url(normalized_url)
    if pooled_endpoint:
        kwargs["connect_args"] = {"statement_cache_size": 0}

    if is_serverless or pooled_endpoint:
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_pre_ping"] = True

    return normalized_url, kwargs


_db_url, _engine_kwargs = build_engine_kwargs(
    settings.get_async_database_url(),
    is_serverless=settings.is_vercel_runtime(),
)

engine = create_async_engine(_db_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def set_transaction_rls_user(
    session: AsyncSession,
    account_id: uuid.UUID | str,
) -> None:
    """Bind PostgreSQL row-level security policies to the authenticated account.

    ``set_config(..., true)`` is transaction-local, so pooled connections (including
    Neon's PgBouncer `-pooler` in transaction mode) cannot leak one request's
    identity into another. SQLite is intentionally a no-op because it is used by
    the local/unit-test suite and does not support RLS.
    """
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return

    await session.execute(
        text("SELECT set_config('app.current_user_id', :account_id, true)"),
        {"account_id": str(account_id)},
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
