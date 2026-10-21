import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.admin import set_admin_session_maker
from app.main import app
from app.database import Base, get_db
from app.models.account import Account  # noqa: F401
from app.models.session import AuthSession  # noqa: F401
from app.models.ai_usage import AIUsage  # noqa: F401
from app.models.recovery import RecoveryCode  # noqa: F401
from app.models.synchronized_content import (  # noqa: F401
    ProfileSync, TasksSync, EventsSync, TimeBlocksSync,
    RemindersSync, NotesSync, CustomCategoriesSync
)
from app.models.mutations import IdempotentMutation  # noqa: F401
from app.models.change_feed import ChangeFeed  # noqa: F401

# Use in-memory SQLite with aiosqlite (shared cache) for fast backend unit tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///file:memdb1?mode=memory&cache=shared&uri=true"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

set_admin_session_maker(TestingSessionLocal)

def rebind_admin_session_maker():
    set_admin_session_maker(TestingSessionLocal)
    if hasattr(app.state, "admin") and app.state.admin:
        app.state.admin.engine = test_engine
        app.state.admin.session_maker = TestingSessionLocal
        for view in getattr(app.state.admin, "views", []):
            view.session_maker = TestingSessionLocal

rebind_admin_session_maker()

@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_test_db():
    rebind_admin_session_maker()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

app.dependency_overrides[get_db] = override_get_db

# Settings read backend/.env, which may hold a real Pinecone key: no test
# reaches the Student Handbook index unless it installs its own retriever.
# setdefault, because tests also import this module as tests.conftest,
# which runs it a second time and must not undo an override a test has set.
from app.api.v1.ai import get_handbook_retriever  # noqa: E402

app.dependency_overrides.setdefault(get_handbook_retriever, lambda: None)

@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
