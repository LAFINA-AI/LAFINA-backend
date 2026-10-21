"""Create the initial personal-account schema and PostgreSQL RLS policies.

Revision ID: 001_initial_schema_and_rls
Revises:
Create Date: 2026-07-22

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial_schema_and_rls'
down_revision = None
branch_labels = None
depends_on = None

SYNC_TABLES = [
    "profile_sync", "tasks_sync", "events_sync", "time_blocks_sync",
    "reminders_sync", "notes_sync", "custom_categories_sync"
]


def _create_sync_table(table: str) -> None:
    """Create one owner-scoped synchronized-content table."""
    op.create_table(
        table,
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("change_id", sa.BigInteger(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint(
            "owner_id",
            "client_id",
            name=f"{table}_pkey",
        ),
    )
    op.create_index(f"ix_{table}_change_id", table, ["change_id"])


def upgrade() -> None:
    """Create the baseline tables before enabling owner-isolation policies."""
    op.create_table(
        "accounts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="accounts_pkey"),
    )
    op.create_index("ix_accounts_email", "accounts", ["email"], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("refresh_token_hash", sa.String(length=256), nullable=False),
        sa.Column("device_info", sa.String(length=256), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            name="auth_sessions_owner_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="auth_sessions_pkey"),
    )
    op.create_index("ix_auth_sessions_owner_id", "auth_sessions", ["owner_id"])
    op.create_index(
        "ix_auth_sessions_refresh_token_hash",
        "auth_sessions",
        ["refresh_token_hash"],
    )

    op.create_table(
        "recovery_codes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("code_hash", sa.String(length=256), nullable=False),
        sa.Column("is_used", sa.Boolean(), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            name="recovery_codes_owner_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="recovery_codes_pkey"),
    )
    op.create_index("ix_recovery_codes_owner_id", "recovery_codes", ["owner_id"])

    for table in SYNC_TABLES:
        _create_sync_table(table)

    op.create_table(
        "idempotent_mutations",
        sa.Column("mutation_id", sa.String(length=128), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("response_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            name="idempotent_mutations_owner_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "mutation_id",
            name="idempotent_mutations_pkey",
        ),
    )
    op.create_index(
        "ix_idempotent_mutations_owner_id",
        "idempotent_mutations",
        ["owner_id"],
    )

    op.create_table(
        "change_feed",
        sa.Column("change_id", sa.BigInteger(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            name="change_feed_owner_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("change_id", name="change_feed_pkey"),
    )
    op.create_index("ix_change_feed_owner_id", "change_feed", ["owner_id"])
    op.create_index("ix_change_feed_created_at", "change_feed", ["created_at"])

    op.create_table(
        "ai_usage",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("request_type", sa.String(length=64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            name="ai_usage_owner_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="ai_usage_pkey"),
    )
    op.create_index("ix_ai_usage_owner_id", "ai_usage", ["owner_id"])
    op.create_index("ix_ai_usage_created_at", "ai_usage", ["created_at"])

    op.create_table(
        "security_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("details", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="security_events_pkey"),
    )
    op.create_index("ix_security_events_owner_id", "security_events", ["owner_id"])
    op.create_index("ix_security_events_created_at", "security_events", ["created_at"])

    # Enable RLS on all synchronized tables and enforce default-deny ownership policies.
    for table in SYNC_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")
        op.execute(f"""
            CREATE POLICY {table}_owner_policy ON {table}
            FOR ALL
            USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
            WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);
        """)

def downgrade() -> None:
    """Remove the complete baseline schema in dependency-safe order."""
    for table in SYNC_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_owner_policy ON {table};")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    op.drop_table("security_events")
    op.drop_table("ai_usage")
    op.drop_table("change_feed")
    op.drop_table("idempotent_mutations")
    for table in reversed(SYNC_TABLES):
        op.drop_table(table)
    op.drop_table("recovery_codes")
    op.drop_table("auth_sessions")
    op.drop_table("accounts")
