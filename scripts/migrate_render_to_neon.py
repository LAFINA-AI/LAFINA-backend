"""Migrate personal LAFINA data from Render PostgreSQL to Vercel Neon Postgres.

Copies all 21 personal tables in foreign-key order while:
1. Skipping all retired business and Gmail tables (and dropping them on the
   target if present).
2. Remapping legacy `role='business'` and `subscription_plan='business'`
   accounts to `'student_pro'` so paid online AI features remain active.
3. Handling PostgreSQL `FORCE ROW LEVEL SECURITY` on owner-scoped sync tables
   during both source reads and target writes.
4. Synchronizing the `change_feed_change_id_seq` sequence on Neon after copy.

Usage (from repository root):
    python -m scripts.migrate_render_to_neon --dry-run
    python -m scripts.migrate_render_to_neon
    python -m scripts.migrate_render_to_neon --source-url postgresql://... --target-url postgresql://...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings, normalize_database_url
from app.database import Base
from app.models import (
    AIUsage,
    Account,
    AuthSession,
    ChangeFeed,
    CustomCategoriesSync,
    EventsSync,
    FeatureFlag,
    FlashcardDecksSync,
    IdempotentMutation,
    NotesSync,
    PomodoroSessionsSync,
    PomodoroSettingsSync,
    ProfileSync,
    RecordedMeetingsSync,
    RecoveryCode,
    RemindersSync,
    SecurityEvent,
    StudySummariesSync,
    SyncHead,
    TasksSync,
    TimeBlocksSync,
)

# Ordered so parent rows (`accounts`) are always inserted before child rows.
PERSONAL_MODELS_IN_ORDER = (
    Account,
    AuthSession,
    RecoveryCode,
    FeatureFlag,
    ProfileSync,
    TasksSync,
    EventsSync,
    TimeBlocksSync,
    RemindersSync,
    NotesSync,
    CustomCategoriesSync,
    PomodoroSettingsSync,
    PomodoroSessionsSync,
    FlashcardDecksSync,
    StudySummariesSync,
    RecordedMeetingsSync,
    IdempotentMutation,
    ChangeFeed,
    SyncHead,
    AIUsage,
    SecurityEvent,
)

RLS_FORCED_TABLES = (
    "profile_sync",
    "tasks_sync",
    "events_sync",
    "time_blocks_sync",
    "reminders_sync",
    "notes_sync",
    "custom_categories_sync",
    "pomodoro_settings_sync",
    "pomodoro_sessions_sync",
    "flashcard_decks_sync",
    "study_summaries_sync",
    "recorded_meetings_sync",
    "idempotent_mutations",
    "change_feed",
    "sync_heads",
)

LEGACY_BUSINESS_AND_GMAIL_TABLES = (
    "gmail_send_audits",
    "gmail_connections",
    "gmail_oauth_states",
    "business_meeting_recipients",
    "business_meetings",
    "business_task_comments",
    "business_chat_messages",
    "business_chat_channels",
    "business_idempotent_mutations",
    "business_change_feed",
    "business_work_blocks",
    "business_task_assignments",
    "business_tasks",
    "business_invitations",
    "business_memberships",
    "businesses",
)


@dataclass
class MigrationReport:
    """Summary of rows read, written, and remapped during a migration run."""

    table_counts: dict[str, int] = field(default_factory=dict)
    remapped_business_accounts: int = 0
    skipped_missing_source_tables: list[str] = field(default_factory=list)
    dry_run: bool = False


def remap_account_dict(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Convert legacy `'business'` role/subscription_plan values to `'student_pro'`."""
    updated = dict(row)
    remapped = False
    if updated.get("role") == "business":
        updated["role"] = "student_pro"
        remapped = True
    if updated.get("subscription_plan") == "business":
        updated["subscription_plan"] = "student_pro"
        remapped = True
    return updated, remapped


async def _get_existing_tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))


async def _set_rls_force(session: AsyncSession, tables: set[str], *, force: bool) -> None:
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    directive = "FORCE" if force else "NO FORCE"
    for table in RLS_FORCED_TABLES:
        if table in tables:
            await session.execute(text(f"ALTER TABLE {table} {directive} ROW LEVEL SECURITY"))


async def migrate_database(
    source_engine: AsyncEngine,
    target_engine: AsyncEngine,
    *,
    dry_run: bool = False,
    ensure_target_schema: bool = True,
) -> MigrationReport:
    """Copy all personal tables from `source_engine` to `target_engine`."""
    report = MigrationReport(dry_run=dry_run)
    source_tables = await _get_existing_tables(source_engine)

    if ensure_target_schema and not dry_run:
        async with target_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            if conn.dialect.name == "postgresql":
                for legacy_table in LEGACY_BUSINESS_AND_GMAIL_TABLES:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {legacy_table} CASCADE"))

    target_tables = await _get_existing_tables(target_engine)

    source_maker = async_sessionmaker(bind=source_engine, class_=AsyncSession, expire_on_commit=False)
    target_maker = async_sessionmaker(bind=target_engine, class_=AsyncSession, expire_on_commit=False)

    # Read all personal tables from source. On PostgreSQL, temporarily relax
    # FORCE ROW LEVEL SECURITY inside a read transaction that rolls back so the
    # source database is never mutated.
    extracted_rows: dict[str, list[dict[str, Any]]] = {}
    async with source_maker() as src_session:
        try:
            await _set_rls_force(src_session, source_tables, force=False)
            for model in PERSONAL_MODELS_IN_ORDER:
                table_name = model.__tablename__
                if table_name not in source_tables:
                    report.skipped_missing_source_tables.append(table_name)
                    report.table_counts[table_name] = 0
                    extracted_rows[table_name] = []
                    continue

                columns = [col.name for col in model.__table__.columns]
                res = await src_session.execute(select(model))
                records = res.scalars().all()
                rows: list[dict[str, Any]] = []
                for rec in records:
                    row_data = {col: getattr(rec, col) for col in columns}
                    if model is Account:
                        row_data, was_remapped = remap_account_dict(row_data)
                        if was_remapped:
                            report.remapped_business_accounts += 1
                    rows.append(row_data)

                extracted_rows[table_name] = rows
                report.table_counts[table_name] = len(rows)
        finally:
            await src_session.rollback()

    if dry_run:
        return report

    # Write extracted rows to target idempotently in foreign-key order.
    async with target_maker() as dst_session:
        await _set_rls_force(dst_session, target_tables, force=False)
        try:
            for model in PERSONAL_MODELS_IN_ORDER:
                table_name = model.__tablename__
                rows = extracted_rows.get(table_name, [])
                for row_data in rows:
                    await dst_session.merge(model(**row_data))
                await dst_session.flush()

            # Advance PostgreSQL sequence for change_feed.change_id if applicable.
            if dst_session.get_bind().dialect.name == "postgresql" and "change_feed" in target_tables:
                await dst_session.execute(
                    text(
                        """
                        SELECT setval(
                            pg_get_serial_sequence('change_feed', 'change_id'),
                            GREATEST(COALESCE((SELECT MAX(change_id) FROM change_feed), 1), 1),
                            (SELECT COUNT(*) > 0 FROM change_feed)
                        )
                        WHERE pg_get_serial_sequence('change_feed', 'change_id') IS NOT NULL
                        """
                    )
                )
        finally:
            await _set_rls_force(dst_session, target_tables, force=True)

        await dst_session.commit()

    return report


async def _run_cli(args: argparse.Namespace) -> int:
    settings = get_settings()
    raw_source = args.source_url or settings.RENDER_DATABASE_URL
    raw_target = args.target_url or settings.get_migration_database_url()

    if not raw_source:
        print(
            "[Error] Source Render database URL is required. Pass --source-url or set RENDER_DATABASE_URL in .env."
        )
        return 1
    if not raw_target:
        print(
            "[Error] Target Neon database URL is required. Pass --target-url or set DATABASE_URL_UNPOOLED / DATABASE_URL in .env."
        )
        return 1

    source_url = normalize_database_url(raw_source)
    target_url = normalize_database_url(raw_target)

    if source_url == target_url:
        print("[Error] Source and target database URLs are identical; aborting.")
        return 1

    source_engine = create_async_engine(source_url, poolclass=NullPool)
    target_engine = create_async_engine(target_url, poolclass=NullPool)
    try:
        mode = "DRY RUN" if args.dry_run else "LIVE MIGRATION"
        print(f"[{mode}] Starting Render -> Neon personal data migration...")
        report = await migrate_database(
            source_engine,
            target_engine,
            dry_run=args.dry_run,
        )
        print("\nTable row counts:")
        for table, count in report.table_counts.items():
            print(f"  - {table}: {count}")
        print(f"\nBusiness accounts remapped to student_pro: {report.remapped_business_accounts}")
        if report.skipped_missing_source_tables:
            print(f"Skipped missing source tables: {', '.join(report.skipped_missing_source_tables)}")
        print(f"\n[{mode}] Completed successfully.")
        return 0
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-url",
        help="Source Render Postgres URL (defaults to RENDER_DATABASE_URL env var)",
    )
    parser.add_argument(
        "--target-url",
        help="Target Neon direct Postgres URL (defaults to DATABASE_URL_UNPOOLED / DATABASE_URL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and count rows from source without writing to target",
    )
    sys.exit(asyncio.run(_run_cli(parser.parse_args())))


if __name__ == "__main__":
    main()
