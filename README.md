# LAFINA Backend — Vercel + Neon Postgres

FastAPI cloud sync and online AI backend for **LAFINA** (Offline-First Voice Academic Scheduler).

## Architecture Overview

- **Local First (`SQLite`) ↔ Cloud Sync (`Neon Postgres`)**: Mobile and desktop clients store all primary scheduling and study data locally in SQLite and synchronize personal entities (`profile`, `tasks`, `events`, `time_blocks`, `reminders`, `notes`, `custom_categories`, `pomodoro_settings`, `pomodoro_sessions`, `flashcard_decks`, `study_summaries`, `recorded_meetings`) with Neon Postgres via `/v1/sync/push`, `/v1/sync/pull`, and `/v1/sync/full` using Last-Write-Wins (`updated_at`) and PostgreSQL Row-Level Security (`app.current_user_id`).
- **Personal Plans Only (`student` / `student_pro` / `admin`)**: All business roles, business workspaces, business chat, business task/work-block sync, business meetings, and Gmail OAuth/API integrations have been removed.
- **Neon Connection Pooling & Direct Migrations**:
  - **Runtime (`DATABASE_URL`)**: Uses Neon's pooled `-pooler` endpoint (PgBouncer transaction mode) with `NullPool` and `statement_cache_size=0` so Vercel serverless functions scale without exhausting connections or colliding on prepared statements.
  - **Migrations & Data Copy (`DATABASE_URL_UNPOOLED`)**: Uses Neon's direct (non-pooled) endpoint for Alembic schema migrations (`LOCK TABLE`, `ALTER TABLE`, RLS policies) and bulk data migration.
  - Standard `postgres://` and `postgresql://` URLs with `?sslmode=require&channel_binding=require` are automatically normalized for SQLAlchemy's `postgresql+asyncpg` driver.

---

## Repository Structure

```text
LAFINA-backend/
├── alembic/
│   ├── env.py                                     # Uses DATABASE_URL_UNPOOLED (direct) for migrations
│   └── versions/
│       ├── 001_initial_schema_and_rls.py          # Core personal tables & RLS policies
│       ├── 002_personal_sync_hardening.py         # Sequence-backed change_feed & composite mutation PK
│       ├── 003_sync_snapshot_heads.py             # Per-owner sync_heads high-water marks
│       ├── 004_account_roles_and_plans.py         # system_role & subscription_plan on accounts
│       ├── 010_feature_flags.py                   # Runtime feature flags (handbook_rag)
│       ├── 011_study_tools_sync.py                # Study tools personal sync tables & RLS
│       └── 012_remove_business_and_gmail.py       # Drops legacy business/Gmail tables & remaps business -> student_pro
├── app/
│   ├── api/v1/                                    # auth, sync, ai, meetings (/v1/meetings/summary), meeting_notes
│   ├── clients/                                   # DeepSeek, Gemini TTS, Pinecone clients
│   ├── models/                                    # Personal SQLAlchemy ORM models
│   ├── security/                                  # Argon2id, RS256 JWT, login throttle
│   ├── services/                                  # Entitlements, flashcards, study notes, meeting notes, RAG, docgen
│   ├── admin.py                                   # SQLAdmin studio at /admin
│   ├── config.py                                  # Settings + Neon URL normalization
│   ├── database.py                                # Async engine + transaction-local RLS helper
│   └── main.py                                    # FastAPI entrypoint (auto-detected by Vercel)
├── scripts/
│   ├── create_admin.py                            # Create or promote an admin account
│   ├── ingest_handbook.py                         # Ingest USTP Student Handbook PDF into Pinecone
│   ├── migrate_render_to_neon.py                  # Copy personal data from Render Postgres to Neon Postgres
│   └── seed_demo_accounts.py                      # Seed demo student & student_pro accounts
├── tests/                                         # Pytest unit & migration test suite (SQLite in-memory)
├── .env.example                                   # Environment variable template for Vercel + Neon
├── alembic.ini
├── pyproject.toml
├── requirements.txt
└── vercel.json                                    # Vercel Python function config (maxDuration: 60s)
```

---

## What To Do Next (Step-by-Step Migration Guide)

### 1. Create / Link Your Neon Postgres Database on Vercel

1. Open your Vercel dashboard (or the Neon console) and create a **Neon Postgres** project in the region closest to your users (e.g., Singapore `ap-southeast-1`).
2. Connect the Neon database to your Vercel project via the **Vercel Storage / Marketplace Neon Integration** (this automatically injects `DATABASE_URL` and `DATABASE_URL_UNPOOLED` into your Vercel environment), or pull them locally with:
   ```bash
   vercel env pull .env
   # or with the Neon CLI:
   neon env pull
   ```
3. Copy `.env.example` to `.env` if setting up manually and fill in:
   - `DATABASE_URL`: Neon **pooled** connection string (hostname contains `-pooler`).
   - `DATABASE_URL_UNPOOLED`: Neon **direct** connection string (hostname has **no** `-pooler`).
   - `RENDER_DATABASE_URL`: Your existing Render Postgres **External Database URL** (from the Render dashboard -> Postgres -> Connections -> External Database URL) for the one-time data copy.
   - `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `JWT_PRIVATE_KEY`, `JWT_PUBLIC_KEY`.

---

### 2. Apply Schema Migrations to Neon

Always run Alembic migrations against the **direct (unpooled)** Neon connection (`alembic/env.py` automatically selects `DATABASE_URL_UNPOOLED` when set):

```bash
python -m alembic upgrade head
```

This creates all 21 personal tables (`accounts`, `auth_sessions`, `recovery_codes`, `feature_flags`, 12 `*_sync` tables, `idempotent_mutations`, `change_feed`, `sync_heads`, `ai_usage`, `security_events`) and enables PostgreSQL Row-Level Security (`FORCE ROW LEVEL SECURITY`) without creating any business or Gmail tables.

---

### 3. Migrate Existing Data from Render to Neon

#### Option A: Recommended Python Data Migration Script (`scripts/migrate_render_to_neon.py`)

This script copies only the 21 personal tables in foreign-key order, automatically converts any legacy `role='business'` or `subscription_plan='business'` accounts to `'student_pro'`, preserves all sync records and change IDs, and advances the `change_feed_change_id_seq` sequence on Neon:

1. **Preview the migration (Dry Run)** — reads row counts from Render without writing to Neon:
   ```bash
   python -m scripts.migrate_render_to_neon --dry-run
   ```
   *(Or pass URLs explicitly: `python -m scripts.migrate_render_to_neon --source-url "postgresql://..." --target-url "postgresql://..." --dry-run`)*

2. **Run the live migration**:
   ```bash
   python -m scripts.migrate_render_to_neon
   ```
   The script is idempotent (`merge`-based) and safe to re-run if interrupted.

#### Option B: `pg_dump` / `pg_restore` + Alembic Cleanup Migration (`012`)

If you prefer a raw PostgreSQL dump from Render into Neon using direct connections:

```bash
pg_dump --no-owner --no-acl "$RENDER_DATABASE_URL" | psql "$DATABASE_URL_UNPOOLED"
python -m alembic upgrade head
```

Running `python -m alembic upgrade head` executes revision `012_remove_business_and_gmail`, which drops all 15 legacy business and Gmail tables (`CASCADE`) and updates all `business` accounts to `student_pro`.

---

### 4. Verify with Neon Diagnostics (Optional)

With the Neon CLI installed, inspect your migrated Neon database:

```bash
neon inspect db table-sizes
neon inspect db index-sizes
```

---

### 5. Deploy to Vercel

1. Install and sign in to the Vercel CLI (or connect this repository in the Vercel dashboard):
   ```bash
   vercel
   vercel --prod
   ```
2. Verify all environment variables are set in **Vercel Project Settings -> Environment Variables**:
   - `ENVIRONMENT=production`
   - `API_BASE_URL=https://<your-project>.vercel.app`
   - `DATABASE_URL` (pooled `-pooler` Neon URL)
   - `DATABASE_URL_UNPOOLED` (direct Neon URL)
   - `JWT_PRIVATE_KEY` & `JWT_PUBLIC_KEY` (generate once and keep fixed across serverless instances!)
   - `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
3. Bootstrap your admin account against Neon (if not already migrated from Render):
   ```bash
   python -m scripts.create_admin admin@lafina.app "YourStrongAdminPassword123!"
   ```
4. Check the live health endpoint:
   ```bash
   curl https://<your-project>.vercel.app/healthz
   ```
5. Update the mobile/desktop app's cloud API base URL to `https://<your-project>.vercel.app`.

---

## Running Locally & Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite (uses in-memory SQLite via aiosqlite)
python -m pytest

# Run the linter
python -m ruff check app scripts tests alembic

# Start local development server
uvicorn app.main:app --reload --port 8000
```
