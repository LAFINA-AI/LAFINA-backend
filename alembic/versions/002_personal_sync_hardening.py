"""Harden personal sync ordering, idempotency, and RLS.

Revision ID: 002_personal_sync_hardening
Revises: 001_initial_schema_and_rls
Create Date: 2026-08-24

"""
from alembic import op
import sqlalchemy as sa


revision = "002_personal_sync_hardening"
down_revision = "001_initial_schema_and_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Install database-owned change IDs and account-scoped mutation keys."""
    op.execute("LOCK TABLE change_feed IN ACCESS EXCLUSIVE MODE")
    op.execute("CREATE SEQUENCE IF NOT EXISTS change_feed_change_id_seq")
    op.execute(
        """
        SELECT setval(
            'change_feed_change_id_seq',
            GREATEST(
                COALESCE(MAX(change_id), 1),
                (SELECT last_value FROM change_feed_change_id_seq)
            ),
            CASE
                WHEN MAX(change_id) IS NOT NULL THEN true
                ELSE (SELECT is_called FROM change_feed_change_id_seq)
            END
        )
        FROM change_feed
        """
    )
    op.execute(
        """
        ALTER TABLE change_feed
        ALTER COLUMN change_id
        SET DEFAULT nextval('change_feed_change_id_seq'::regclass)
        """
    )
    op.execute(
        "ALTER SEQUENCE change_feed_change_id_seq OWNED BY change_feed.change_id"
    )

    op.add_column(
        "idempotent_mutations",
        sa.Column("client_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(
        "idempotent_mutations_pkey",
        "idempotent_mutations",
        type_="primary",
    )
    op.create_primary_key(
        "idempotent_mutations_pkey",
        "idempotent_mutations",
        ["owner_id", "mutation_id"],
    )

    for table in ("change_feed", "idempotent_mutations"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_owner_policy ON {table}
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
    """Restore the legacy global mutation key while retaining auto-increment IDs."""
    for table in ("change_feed", "idempotent_mutations"):
        op.execute(f"DROP POLICY IF EXISTS {table}_owner_policy ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    op.drop_constraint(
        "idempotent_mutations_pkey",
        "idempotent_mutations",
        type_="primary",
    )
    op.create_primary_key(
        "idempotent_mutations_pkey",
        "idempotent_mutations",
        ["mutation_id"],
    )
    op.drop_column("idempotent_mutations", "client_updated_at")
