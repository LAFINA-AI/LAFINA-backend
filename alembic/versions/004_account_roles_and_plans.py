"""004 Account system_role and subscription_plan columns

Revision ID: 004_account_roles_and_plans
Revises: 003_sync_snapshot_heads
Create Date: 2026-08-25

Adds personal account role and subscription_plan columns (`student` / `student_pro`)
without any business tables.
"""
from alembic import op

revision = "004_account_roles_and_plans"
down_revision = "003_sync_snapshot_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Expand alembic_version column width for descriptive revision IDs
    op.execute("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(64)")

    # Add system_role and subscription_plan columns to accounts
    op.execute(
        "ALTER TABLE accounts ADD COLUMN IF NOT EXISTS system_role VARCHAR(32) NOT NULL DEFAULT 'user'"
    )
    op.execute(
        "ALTER TABLE accounts ADD COLUMN IF NOT EXISTS subscription_plan VARCHAR(32) NOT NULL DEFAULT 'student'"
    )


def downgrade() -> None:
    op.drop_column("accounts", "subscription_plan")
    op.drop_column("accounts", "system_role")
