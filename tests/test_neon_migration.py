"""Tests for the Render -> Vercel Neon database migration and business-removal guarantees."""

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import (
    AIUsage,
    Account,
    ChangeFeed,
    FeatureFlag,
    FlashcardDecksSync,
    NotesSync,
    PomodoroSessionsSync,
    PomodoroSettingsSync,
    ProfileSync,
    RecordedMeetingsSync,
    StudySummariesSync,
    SyncHead,
    TasksSync,
)
from app.security.auth import hash_password
from scripts.migrate_render_to_neon import migrate_database


@pytest.mark.asyncio
async def test_business_and_gmail_endpoints_are_removed(async_client: AsyncClient):
    """All legacy `/v1/businesses/*` and `/v1/email/gmail/*` endpoints must be absent (404)."""
    for removed_path in (
        "/v1/businesses",
        f"/v1/businesses/{uuid.uuid4()}",
        f"/v1/businesses/{uuid.uuid4()}/sync",
        f"/v1/businesses/{uuid.uuid4()}/chat/channels",
        f"/v1/businesses/{uuid.uuid4()}/meetings",
        "/v1/email/gmail/status",
        "/v1/email/gmail/connect/start",
    ):
        res = await async_client.get(removed_path)
        assert res.status_code == 404, f"Expected 404 for {removed_path}, got {res.status_code}"


@pytest.mark.asyncio
async def test_me_profile_has_no_business_session(async_client: AsyncClient):
    """`/v1/me` and `/v1/auth/me` must return personal profile fields without `business_session`."""
    reg = await async_client.post(
        "/v1/auth/register",
        json={
            "email": "neon_student@ustp.edu.ph",
            "password": "StrongPassword123!",
        },
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    for endpoint in ("/v1/me", "/v1/auth/me"):
        res = await async_client.get(endpoint, headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body["email"] == "neon_student@ustp.edu.ph"
        assert body["role"] == "student"
        assert body["subscription_plan"] == "student"
        assert body["effective_subscription_plan"] == "student"
        assert "business_session" not in body


@pytest.mark.asyncio
async def test_render_to_neon_migration_copies_personal_data_and_remaps_business_accounts():
    """End-to-end test of `migrate_database`: copies all personal tables, remaps business -> student_pro, and is idempotent."""
    source_engine = create_async_engine(
        "sqlite+aiosqlite:///file:render_src?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    target_engine = create_async_engine(
        "sqlite+aiosqlite:///file:neon_dst?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    try:
        async with source_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with target_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        src_maker = async_sessionmaker(bind=source_engine, class_=AsyncSession, expire_on_commit=False)
        dst_maker = async_sessionmaker(bind=target_engine, class_=AsyncSession, expire_on_commit=False)

        now = datetime.now(timezone.utc)
        legacy_biz_user_id = uuid.uuid4()
        student_user_id = uuid.uuid4()

        async with src_maker() as src:
            src.add_all(
                [
                    Account(
                        id=legacy_biz_user_id,
                        email="legacy_biz@ustp.edu.ph",
                        password_hash=hash_password("Password123!"),
                        role="business",
                        system_role="user",
                        subscription_plan="business",
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    ),
                    Account(
                        id=student_user_id,
                        email="student@ustp.edu.ph",
                        password_hash=hash_password("Password123!"),
                        role="student",
                        system_role="user",
                        subscription_plan="student",
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    ),
                    FeatureFlag(
                        key="handbook_rag",
                        enabled=True,
                        description="USTP handbook RAG",
                        updated_at=now,
                    ),
                    ProfileSync(
                        owner_id=legacy_biz_user_id,
                        client_id="profile-default",
                        version=1,
                        change_id=1,
                        payload={"displayName": "Legacy Pro"},
                        updated_at=now,
                    ),
                    TasksSync(
                        owner_id=legacy_biz_user_id,
                        client_id="task-1",
                        version=2,
                        change_id=2,
                        payload={"title": "Migrate to Neon", "status": "todo"},
                        updated_at=now,
                    ),
                    NotesSync(
                        owner_id=student_user_id,
                        client_id="note-1",
                        version=1,
                        change_id=3,
                        payload={"title": "Algorithms lecture"},
                        updated_at=now,
                    ),
                    PomodoroSettingsSync(
                        owner_id=student_user_id,
                        client_id="pomo-settings",
                        version=1,
                        change_id=4,
                        payload={"focusMinutes": 25, "breakMinutes": 5},
                        updated_at=now,
                    ),
                    PomodoroSessionsSync(
                        owner_id=student_user_id,
                        client_id="pomo-session-1",
                        version=1,
                        change_id=5,
                        payload={"durationMinutes": 25},
                        updated_at=now,
                    ),
                    FlashcardDecksSync(
                        owner_id=legacy_biz_user_id,
                        client_id="deck-1",
                        version=1,
                        change_id=6,
                        payload={"title": "OS Deck", "cards": []},
                        updated_at=now,
                    ),
                    StudySummariesSync(
                        owner_id=legacy_biz_user_id,
                        client_id="summary-1",
                        version=1,
                        change_id=7,
                        payload={"title": "OS Summary"},
                        updated_at=now,
                    ),
                    RecordedMeetingsSync(
                        owner_id=legacy_biz_user_id,
                        client_id="rec-1",
                        version=1,
                        change_id=8,
                        payload={"title": "Capstone Standup", "transcript": "All green"},
                        updated_at=now,
                    ),
                    ChangeFeed(
                        change_id=8,
                        owner_id=legacy_biz_user_id,
                        entity_type="recorded_meetings",
                        entity_id="rec-1",
                        operation="UPSERT",
                        version=1,
                        payload={"title": "Capstone Standup"},
                        created_at=now,
                    ),
                    SyncHead(
                        owner_id=legacy_biz_user_id,
                        latest_change_id=8,
                        updated_at=now,
                    ),
                    AIUsage(
                        id=uuid.uuid4(),
                        owner_id=legacy_biz_user_id,
                        request_type="chat",
                        prompt_tokens=120,
                        completion_tokens=80,
                        created_at=now,
                    ),
                ]
            )
            await src.commit()

        # 1. Dry-run reads counts and remaps without writing to target
        dry_report = await migrate_database(source_engine, target_engine, dry_run=True)
        assert dry_report.table_counts["accounts"] == 2
        assert dry_report.remapped_business_accounts == 1
        async with dst_maker() as dst:
            assert (await dst.execute(select(Account))).scalars().all() == []

        # 2. Live migration copies all records and converts business -> student_pro
        live_report = await migrate_database(source_engine, target_engine, dry_run=False)
        assert live_report.table_counts["accounts"] == 2
        assert live_report.table_counts["tasks_sync"] == 1
        assert live_report.table_counts["flashcard_decks_sync"] == 1
        assert live_report.table_counts["recorded_meetings_sync"] == 1
        assert live_report.remapped_business_accounts == 1

        # 3. Re-running migration is idempotent (no duplicate key errors)
        second_report = await migrate_database(source_engine, target_engine, dry_run=False)
        assert second_report.table_counts["accounts"] == 2

        async with dst_maker() as dst:
            migrated_pro = await dst.get(Account, legacy_biz_user_id)
            assert migrated_pro is not None
            assert migrated_pro.role == "student_pro"
            assert migrated_pro.subscription_plan == "student_pro"

            migrated_student = await dst.get(Account, student_user_id)
            assert migrated_student is not None
            assert migrated_student.role == "student"
            assert migrated_student.subscription_plan == "student"

            migrated_task = await dst.get(TasksSync, (legacy_biz_user_id, "task-1"))
            assert migrated_task is not None
            assert migrated_task.payload["title"] == "Migrate to Neon"

            migrated_rec = await dst.get(RecordedMeetingsSync, (legacy_biz_user_id, "rec-1"))
            assert migrated_rec is not None
            assert migrated_rec.payload["title"] == "Capstone Standup"

            sync_head = await dst.get(SyncHead, legacy_biz_user_id)
            assert sync_head is not None
            assert sync_head.latest_change_id == 8
    finally:
        await source_engine.dispose()
        await target_engine.dispose()
