"""015 Audit hardening: composite indexes, ai_usage RLS, and owner_id FKs.

Revision ID: 015_audit_hardening
Revises: 014_chat_history_sync
Create Date: 2026-11-26

Hardens personal sync and AI usage tables:
- Adds composite `(owner_id, change_id)` indexes across `change_feed` and all 14
  `*_sync` tables, plus `(owner_id, entity_type, entity_id, change_id)` on
  `change_feed` and partial indexes on live chat rows.
- Enables and forces PostgreSQL owner-isolation RLS on `ai_usage`.
- Adds `FOREIGN KEY (owner_id) REFERENCES accounts(id) ON DELETE CASCADE NOT VALID`
  across all 14 `*_sync` tables so future deletes cascade cleanly without failing
  migration on any pre-existing orphaned rows.
"""
from alembic import op

revision = "015_audit_hardening"
down_revision = "014_chat_history_sync"
branch_labels = None
depends_on = None

SYNC_TABLES = [
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
    "chat_sessions_sync",
    "chat_messages_sync",
]


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_change_feed_owner_change_id "
        "ON change_feed (owner_id, change_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_change_feed_owner_entity_change "
        "ON change_feed (owner_id, entity_type, entity_id, change_id)"
    )

    for table in SYNC_TABLES:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{table}_owner_change_id "
            f"ON {table} (owner_id, change_id)"
        )
        op.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'fk_{table}_owner_id'
                ) THEN
                    ALTER TABLE {table}
                    ADD CONSTRAINT fk_{table}_owner_id
                    FOREIGN KEY (owner_id) REFERENCES accounts(id) ON DELETE CASCADE
                    NOT VALID;
                END IF;
            END $$;
            """
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_live_owner "
        "ON chat_sessions_sync (owner_id) WHERE deleted_at IS NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chat_messages_live_session "
        "ON chat_messages_sync (owner_id, (payload->>'session_id')) WHERE deleted_at IS NULL"
    )

    # Enable and force owner-scoped RLS on ai_usage (matching voice_sessions & sync tables).
    op.execute("ALTER TABLE ai_usage ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE ai_usage FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS ai_usage_owner_policy ON ai_usage")
    op.execute(
        """
        CREATE POLICY ai_usage_owner_policy ON ai_usage
        FOR ALL
        USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS ai_usage_owner_policy ON ai_usage")
    op.execute("ALTER TABLE ai_usage NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE ai_usage DISABLE ROW LEVEL SECURITY")

    op.execute("DROP INDEX IF EXISTS ix_chat_messages_live_session")
    op.execute("DROP INDEX IF EXISTS ix_chat_sessions_live_owner")

    for table in reversed(SYNC_TABLES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS fk_{table}_owner_id")
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_owner_change_id")

    op.execute("DROP INDEX IF EXISTS ix_change_feed_owner_entity_change")
    op.execute("DROP INDEX IF EXISTS ix_change_feed_owner_change_id")
