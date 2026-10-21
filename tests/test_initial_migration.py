"""Regression tests for bootstrapping a fresh Neon PostgreSQL database and Alembic chain."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "alembic" / "versions"


def _load_migration(filename: str) -> ModuleType:
    path = VERSIONS_DIR / filename
    spec = spec_from_file_location(f"local_alembic_versions.{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_migration_creates_sync_tables_before_enabling_rls(monkeypatch):
    """Every RLS target must exist before its first ALTER TABLE statement."""
    migration = _load_migration("001_initial_schema_and_rls.py")
    created_tables: set[str] = set()

    def record_create_table(table: str, *args, **kwargs) -> None:
        del args, kwargs
        created_tables.add(table)

    def assert_rls_target_exists(statement: str) -> None:
        normalized = " ".join(statement.split())
        if not normalized.startswith("ALTER TABLE"):
            return
        table = normalized.split()[2]
        assert table in created_tables

    monkeypatch.setattr(migration.op, "create_table", record_create_table)
    monkeypatch.setattr(migration.op, "create_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(migration.op, "execute", assert_rls_target_exists)

    migration.upgrade()

    assert set(migration.SYNC_TABLES).issubset(created_tables)
    assert {
        "accounts",
        "auth_sessions",
        "recovery_codes",
        "idempotent_mutations",
        "change_feed",
        "ai_usage",
        "security_events",
    }.issubset(created_tables)


def test_alembic_revision_chain_is_linear_and_free_of_business_schema_creation():
    """The migration chain must walk cleanly from 001 to 012 without creating business tables."""
    modules = [
        _load_migration(path.name)
        for path in sorted(VERSIONS_DIR.glob("*.py"))
    ]

    by_down_rev = {m.down_revision: m for m in modules}
    assert None in by_down_rev, "Missing root migration (down_revision=None)"

    ordered: list[str] = []
    cur = by_down_rev[None]
    while cur is not None:
        ordered.append(cur.revision)
        cur = by_down_rev.get(cur.revision)

    assert len(ordered) == len(modules)
    assert ordered == [
        "001_initial_schema_and_rls",
        "002_personal_sync_hardening",
        "003_sync_snapshot_heads",
        "004_account_roles_and_plans",
        "010_feature_flags",
        "011_study_tools_sync",
        "012_remove_business_and_gmail",
    ]


def test_migration_012_drops_legacy_tables_and_remaps_business_accounts(monkeypatch):
    """Migration 012 must drop retired tables and convert business accounts to student_pro."""
    migration = _load_migration("012_remove_business_and_gmail.py")
    executed_sql: list[str] = []

    monkeypatch.setattr(migration.op, "execute", lambda sql: executed_sql.append(sql.strip()))
    migration.upgrade()

    for table in migration.LEGACY_TABLES_TO_DROP:
        assert any(f"DROP TABLE IF EXISTS {table} CASCADE" in stmt for stmt in executed_sql)

    assert any(
        "UPDATE accounts SET role = 'student_pro' WHERE role = 'business'" in stmt
        for stmt in executed_sql
    )
    assert any(
        "UPDATE accounts SET subscription_plan = 'student_pro' WHERE subscription_plan = 'business'"
        in stmt
        for stmt in executed_sql
    )
