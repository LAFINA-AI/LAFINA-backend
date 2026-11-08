"""013 Live voice conversation sessions

Revision ID: 013_voice_sessions
Revises: 012_remove_business_and_gmail
Create Date: 2026-11-08

One row per live voice conversation (ElevenAgents + DeepSeek). The custom-LLM
route checks the row on every turn: owner, status, expiry and turn budget.
Owner-scoped like the synchronized tables, with the same RLS policy.
"""

from alembic import op

revision = "013_voice_sessions"
down_revision = "012_remove_business_and_gmail"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS voice_sessions (
            id UUID PRIMARY KEY,
            owner_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            status VARCHAR(16) NOT NULL DEFAULT 'active',
            ended_reason VARCHAR(32),
            voice_id VARCHAR(128) NOT NULL,
            use_handbook BOOLEAN NOT NULL DEFAULT true,
            conversation_id VARCHAR(128),
            turns INTEGER NOT NULL DEFAULT 0,
            last_turn_key VARCHAR(64),
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            last_error_code VARCHAR(32),
            last_error_message VARCHAR(300),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            ended_at TIMESTAMP WITH TIME ZONE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_voice_sessions_owner_id ON voice_sessions (owner_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_voice_sessions_created_at ON voice_sessions (created_at)"
    )
    op.execute("ALTER TABLE voice_sessions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE voice_sessions FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS voice_sessions_owner_policy ON voice_sessions")
    op.execute("""
        CREATE POLICY voice_sessions_owner_policy ON voice_sessions
        FOR ALL
        USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS voice_sessions")
