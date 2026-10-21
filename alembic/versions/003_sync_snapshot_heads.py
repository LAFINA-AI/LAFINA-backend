"""Add durable account-scoped sync heads for snapshot resets.

Revision ID: 003_sync_snapshot_heads
Revises: 002_personal_sync_hardening
Create Date: 2026-08-25

"""
from alembic import op
import sqlalchemy as sa


revision = "003_sync_snapshot_heads"
down_revision = "002_personal_sync_hardening"
branch_labels = None
depends_on = None

BACKFILL_SOURCE_TABLES = (
    "change_feed",
    "profile_sync",
    "tasks_sync",
    "events_sync",
    "time_blocks_sync",
    "reminders_sync",
    "notes_sync",
    "custom_categories_sync",
)


def upgrade() -> None:
    """Create and backfill durable owner-scoped sync high-water marks."""
    op.create_table(
        "sync_heads",
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column(
            "latest_change_id",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["accounts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("owner_id"),
    )
    # Migration sessions have no request-scoped RLS identity. Temporarily stop
    # forcing the owner policies so the table owner can see every account while
    # building the per-owner high-water marks; the transaction restores FORCE
    # before it commits.
    for table in BACKFILL_SOURCE_TABLES:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")

    op.execute(
        """
        INSERT INTO sync_heads (owner_id, latest_change_id, updated_at)
        SELECT owner_id, MAX(change_id), now()
        FROM (
            SELECT owner_id, change_id FROM change_feed
            UNION ALL SELECT owner_id, change_id FROM profile_sync
            UNION ALL SELECT owner_id, change_id FROM tasks_sync
            UNION ALL SELECT owner_id, change_id FROM events_sync
            UNION ALL SELECT owner_id, change_id FROM time_blocks_sync
            UNION ALL SELECT owner_id, change_id FROM reminders_sync
            UNION ALL SELECT owner_id, change_id FROM notes_sync
            UNION ALL SELECT owner_id, change_id FROM custom_categories_sync
        ) AS owner_changes
        GROUP BY owner_id
        """
    )
    for table in BACKFILL_SOURCE_TABLES:
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

    op.execute("ALTER TABLE sync_heads ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE sync_heads FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY sync_heads_owner_policy ON sync_heads
        FOR ALL
        USING (
            owner_id = NULLIF(
                current_setting('app.current_user_id', true), ''
            )::uuid
        )
        WITH CHECK (
            owner_id = NULLIF(
                current_setting('app.current_user_id', true), ''
            )::uuid
        )
        """
    )


def downgrade() -> None:
    """Remove the durable snapshot high-water marks."""
    op.execute("DROP POLICY IF EXISTS sync_heads_owner_policy ON sync_heads")
    op.execute("ALTER TABLE sync_heads DISABLE ROW LEVEL SECURITY")
    op.drop_table("sync_heads")
