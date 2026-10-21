"""011 Study tools sync

Revision ID: 011_study_tools_sync
Revises: 010_feature_flags
Create Date: 2026-09-19

Personal sync tables for the study tools the desktop and mobile apps share:
Pomodoro settings and sessions, flashcard decks, study-note summaries and the
text of recorded meetings. Same shape and owner-isolation policy as the
tables from 001.

`IF NOT EXISTS` throughout: the API runs `Base.metadata.create_all` at
startup, so a deploy that boots before this migration runs will already have
created the tables (without RLS, which this migration then enables).
"""
from alembic import op

revision = "011_study_tools_sync"
down_revision = "010_feature_flags"
branch_labels = None
depends_on = None

TABLES = [
    "pomodoro_settings_sync",
    "pomodoro_sessions_sync",
    "flashcard_decks_sync",
    "study_summaries_sync",
    "recorded_meetings_sync",
]


def upgrade() -> None:
    for table in TABLES:
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                owner_id UUID NOT NULL,
                client_id VARCHAR(128) NOT NULL,
                version BIGINT NOT NULL,
                change_id BIGINT NOT NULL,
                payload JSON NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                deleted_at TIMESTAMP WITH TIME ZONE,
                CONSTRAINT {table}_pkey PRIMARY KEY (owner_id, client_id)
            )
        """)
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{table}_change_id ON {table} (change_id)"
        )
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {table}_owner_policy ON {table}")
        op.execute(f"""
            CREATE POLICY {table}_owner_policy ON {table}
            FOR ALL
            USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
            WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        """)


def downgrade() -> None:
    for table in reversed(TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table}")
