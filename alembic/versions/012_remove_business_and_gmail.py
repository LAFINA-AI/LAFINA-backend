"""012 Remove legacy business and Gmail tables and remap business accounts

Revision ID: 012_remove_business_and_gmail
Revises: 011_study_tools_sync
Create Date: 2026-10-21

Ensures any database migrated or restored from the legacy Render deployment has:
1. All 15 business and Gmail tables dropped (`IF EXISTS ... CASCADE`).
2. Any accounts with `role = 'business'` or `subscription_plan = 'business'`
   converted to `'student_pro'` so paid AI features continue working.
"""
from alembic import op

revision = "012_remove_business_and_gmail"
down_revision = "011_study_tools_sync"
branch_labels = None
depends_on = None

LEGACY_TABLES_TO_DROP = (
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


def upgrade() -> None:
    for table in LEGACY_TABLES_TO_DROP:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    op.execute(
        "UPDATE accounts SET role = 'student_pro' WHERE role = 'business'"
    )
    op.execute(
        "UPDATE accounts SET subscription_plan = 'student_pro' WHERE subscription_plan = 'business'"
    )


def downgrade() -> None:
    # Business and Gmail schemas are intentionally retired in this version.
    pass
