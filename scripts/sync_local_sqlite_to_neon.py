"""Migrate and synchronize a local LAFINA client SQLite database into Neon Postgres.

This script:
1. Reads all personal data for the linked local user from `lafina.sqlite`
   (excluding legacy business & Gmail tables).
2. Validates every active entity payload against the strict Pydantic schemas
   in `app.api.v1.sync`.
3. Bulk-loads the `Account`, all 12 `*_sync` tables, `change_feed`, and
   `sync_heads` into Neon Postgres in a single atomic transaction.
4. Updates `lafina.sqlite` (`sync_metadata`, `sync_state`, `sync_outbox`,
   `sync_conflicts`, and legacy demo business rows) so the local SQLite database
   and Neon Postgres are 100% synchronized at the exact same cursor and entity
   versions.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.v1.sync import (
    MODEL_MAP,
    SUPPORTED_ENTITY_TYPES,
    CustomCategoryPayload,
    EventPayload,
    FlashcardDeckPayload,
    NotePayload,
    PomodoroSessionPayload,
    PomodoroSettingsPayload,
    ProfilePayload,
    RecordedMeetingPayload,
    ReminderPayload,
    StudySummaryPayload,
    TaskPayload,
    TimeBlockPayload,
)
from app.config import get_settings
from app.database import build_engine_kwargs
from app.models.account import Account
from app.models.change_feed import ChangeFeed
from app.models.mutations import IdempotentMutation
from app.models.sync_head import SyncHead

DEFAULT_SQLITE_PATH = Path(r"c:\Users\bnoel\AppData\Roaming\lafina-personal\lafina.sqlite")
TARGET_LOCAL_USER_ID = "user_mtt5uztbrgy7p4db3"

RLS_SYNC_TABLES = [
    "idempotent_mutations",
    "change_feed",
    "sync_heads",
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
]

LEGACY_BUSINESS_AND_GMAIL_TABLES = [
    "business_action_candidates",
    "business_meeting_recipients",
    "business_meeting_segments",
    "business_meetings",
    "business_chat_messages",
    "business_chat_channels",
    "business_work_blocks",
    "business_task_comments",
    "business_task_assignments",
    "business_tasks",
    "business_invitations",
    "business_memberships",
    "business_capabilities_cache",
    "businesses",
    "gmail_local_drafts",
    "gmail_messages_cache",
    "gmail_threads_cache",
    "gmail_connections",
]

DEMO_USER_IDS = (
    "11111111-1111-4111-a111-111111111111",
    "22222222-2222-4222-a222-222222222222",
    "33333333-3333-4333-a333-333333333333",
)


@dataclass
class PreparedSyncItem:
    entity_type: str
    entity_id: str
    version: int
    change_id: int
    payload: dict[str, Any]
    updated_at: datetime
    updated_at_iso: str
    deleted_at: datetime | None


def parse_iso_datetime(value: str | None, fallback: datetime) -> datetime:
    if not value or not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return fallback


def normalize_tags_json(raw_tags: str | None) -> str:
    tags_str = (raw_tags or "[]").strip()
    if not tags_str:
        return "[]"
    try:
        parsed = json.loads(tags_str)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass
    return json.dumps(
        [s.strip() for s in tags_str.split(",") if s.strip()],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def extract_personal_items(conn: sqlite3.Connection, local_user_id: str) -> tuple[dict[str, Any], list[PreparedSyncItem]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    now = datetime.now(timezone.utc)

    user_row = cur.execute("SELECT * FROM users WHERE id = ?", (local_user_id,)).fetchone()
    if not user_row:
        raise RuntimeError(f"Local user {local_user_id!r} not found in SQLite.")
    user = dict(user_row)

    existing_versions: dict[tuple[str, str], int] = {}
    for row in cur.execute(
        "SELECT entity_type, entity_id, version FROM sync_metadata WHERE user_id = ? AND scope_type = 'account'",
        (local_user_id,),
    ):
        existing_versions[(row["entity_type"], row["entity_id"])] = int(row["version"] or 1)

    outbox_entities: set[tuple[str, str]] = set()
    for row in cur.execute(
        "SELECT entity_type, entity_id FROM sync_outbox WHERE user_id = ? AND scope_type = 'account'",
        (local_user_id,),
    ):
        outbox_entities.add((row["entity_type"], row["entity_id"]))

    def resolve_version(entity_type: str, entity_id: str) -> int:
        base_v = existing_versions.get((entity_type, entity_id), 0)
        if (entity_type, entity_id) in outbox_entities:
            return max(1, base_v + 1)
        return max(1, base_v)

    raw_items: list[tuple[datetime, str, str, int, dict[str, Any], datetime | None]] = []

    # 1. Profile singleton
    pref_row = cur.execute("SELECT * FROM user_preferences WHERE user_id = ?", (local_user_id,)).fetchone()
    pref = dict(pref_row) if pref_row else {}
    profile_payload = ProfilePayload(
        username=(user.get("username") or "User")[:128],
        wake_time=pref.get("wake_time") or "07:00",
        sleep_time=pref.get("sleep_time") or "23:00",
        study_peak_hours=pref.get("study_peak_hours") or "[]",
        busiest_day=pref.get("busiest_day") or "Monday",
        reminder_lead_minutes=int(pref.get("reminder_lead_minutes") or 15),
        snooze_tendency=pref.get("snooze_tendency") or "snooze_once",
        weekly_class_count=pref.get("weekly_class_count") or "4-6",
        longest_class_gap=pref.get("longest_class_gap") or "1 hour",
        time_format_24h=bool(user.get("time_format_24h") == 1),
        week_starts_monday=bool(user.get("week_starts_monday") == 1),
        dark_mode=bool(user.get("dark_mode") == 1),
    ).model_dump()
    profile_updated = parse_iso_datetime(pref.get("updated_at") or user.get("updated_at"), now)
    raw_items.append((profile_updated, "profile", "profile", resolve_version("profile", "profile"), profile_payload, None))

    # 2. Pomodoro settings singleton
    ps_row = cur.execute("SELECT * FROM pomodoro_settings WHERE user_id = ?", (local_user_id,)).fetchone()
    if ps_row:
        ps = dict(ps_row)
        ps_payload = PomodoroSettingsPayload(
            focus_minutes=int(ps["focus_minutes"]),
            short_break_minutes=int(ps["short_break_minutes"]),
            long_break_minutes=int(ps["long_break_minutes"]),
            long_break_interval=int(ps["long_break_interval"]),
            auto_start_breaks=bool(ps["auto_start_breaks"] == 1),
            auto_start_focus=bool(ps["auto_start_focus"] == 1),
            sound_enabled=bool(ps["sound_enabled"] == 1),
            volume=float(ps["volume"]),
            ring_seconds=int(ps["ring_seconds"]),
            notifications_enabled=bool(ps["notifications_enabled"] == 1),
        ).model_dump()
        ps_updated = parse_iso_datetime(ps.get("updated_at"), now)
        raw_items.append((
            ps_updated,
            "pomodoro_settings",
            "pomodoro_settings",
            resolve_version("pomodoro_settings", "pomodoro_settings"),
            ps_payload,
            None,
        ))

    # 3. Tasks
    for row in cur.execute("SELECT * FROM tasks WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else TaskPayload(
                title=(r.get("title") or "Task")[:256],
                due_date=r.get("due_date"),
                due_time=r.get("due_time"),
                is_completed=bool(r.get("is_completed") == 1),
                priority=r.get("priority") or "Medium",
                category=(r.get("category") or "General")[:64],
                notes=r.get("notes"),
                recurrence_rule=r.get("recurrence_rule"),
            ).model_dump()
        )
        raw_items.append((updated_at, "task", eid, resolve_version("task", eid), payload, deleted_at))

    # 4. Events
    for row in cur.execute("SELECT * FROM events WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else EventPayload(
                title=(r.get("title") or "Event")[:256],
                date=str(r["date"]),
                start_time=str(r["start_time"]),
                end_time=str(r["end_time"]),
                location=r.get("location"),
                linked_calendar_block=r.get("linked_calendar_block"),
                recurrence_rule=r.get("recurrence_rule"),
            ).model_dump()
        )
        raw_items.append((updated_at, "event", eid, resolve_version("event", eid), payload, deleted_at))

    # 5. Time blocks
    for row in cur.execute("SELECT * FROM time_blocks WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else TimeBlockPayload(
                title=(r.get("title") or "Time Block")[:256],
                date=str(r["date"]),
                start_time=str(r["start_time"]),
                end_time=str(r["end_time"]),
                color=str(r.get("color") or "#4F46E5"),
                category=str(r.get("category") or "General"),
                notes=r.get("notes"),
                recurrence_rule=r.get("recurrence_rule"),
            ).model_dump()
        )
        raw_items.append((updated_at, "time_block", eid, resolve_version("time_block", eid), payload, deleted_at))

    # 6. Reminders
    for row in cur.execute("SELECT * FROM reminders WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else ReminderPayload(
                task=(r.get("task") or "Reminder")[:256],
                description=r.get("description"),
                scheduled_at=str(r["scheduled_at"]),
                trigger_at=str(r["trigger_at"]),
                status=str(r.get("status") or "pending"),
                snooze_count=int(r.get("snooze_count") or 0),
            ).model_dump()
        )
        raw_items.append((updated_at, "reminder", eid, resolve_version("reminder", eid), payload, deleted_at))

    # 7. Notes
    for row in cur.execute("SELECT * FROM notes WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else NotePayload(
                title=(r.get("title") or "Untitled")[:256],
                body=str(r.get("body") or ""),
                is_pinned=bool(r.get("is_pinned") == 1),
                tags=normalize_tags_json(r.get("tags")),
                category=str(r.get("category") or "General"),
                is_voice_transcribed=bool(r.get("is_voice_transcribed") == 1),
                sort_order=int(r.get("sort_order") or 0),
            ).model_dump()
        )
        raw_items.append((updated_at, "note", eid, resolve_version("note", eid), payload, deleted_at))

    # 8. Custom categories
    for row in cur.execute("SELECT * FROM custom_categories WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else CustomCategoryPayload(
                name=(r.get("name") or "Category")[:128],
                color=(r.get("color") or "#4F46E5")[:32],
            ).model_dump()
        )
        raw_items.append((updated_at, "custom_category", eid, resolve_version("custom_category", eid), payload, deleted_at))

    # 9. Pomodoro sessions
    for row in cur.execute("SELECT * FROM pomodoro_sessions WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("finished_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        payload = (
            {}
            if deleted_at
            else PomodoroSessionPayload(
                phase=r["phase"],
                task=(r.get("task") or "")[:120],
                duration_ms=max(0, min(10_800_000, int(round(float(r.get("duration_ms") or 0))))),
                started_at=r.get("started_at"),
                finished_at=str(r["finished_at"]),
            ).model_dump()
        )
        raw_items.append((updated_at, "pomodoro_session", eid, resolve_version("pomodoro_session", eid), payload, deleted_at))

    # 10. Flashcard decks
    for row in cur.execute("SELECT * FROM flashcard_decks WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        if deleted_at:
            payload = {}
        else:
            cards = json.loads(r.get("cards_json") or "[]")
            warnings = json.loads(r.get("warnings_json") or "[]")
            payload = FlashcardDeckPayload(
                title=(r.get("title") or "Deck")[:160],
                source_name=r["source_name"][:255] if r.get("source_name") else None,
                cards=[{"question": c["question"][:500], "answer": c["answer"][:2000]} for c in cards[:200]],
                page_count=max(0, int(r.get("page_count") or 0)),
                ocr_page_count=max(0, int(r.get("ocr_page_count") or 0)),
                warnings=[str(w)[:1000] for w in warnings[:50]],
                created_at=str(r.get("created_at") or updated_at.isoformat()),
            ).model_dump()
        raw_items.append((updated_at, "flashcard_deck", eid, resolve_version("flashcard_deck", eid), payload, deleted_at))

    # 11. Study summaries
    for row in cur.execute("SELECT * FROM study_summaries WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        if deleted_at:
            payload = {}
        else:
            sections = json.loads(r.get("sections_json") or "[]")
            key_terms = json.loads(r.get("key_terms_json") or "[]")
            warnings = json.loads(r.get("warnings_json") or "[]")
            payload = StudySummaryPayload(
                title=(r.get("title") or "Summary")[:160],
                source_name=r["source_name"][:255] if r.get("source_name") else None,
                source_kind=r["source_kind"] if r.get("source_kind") in ("pdf", "docx", "pptx", "") else "pdf",
                overview=(r.get("overview") or "")[:10_000],
                sections=[
                    {
                        "heading": s["heading"][:300],
                        "points": [str(pt)[:2000] for pt in s.get("points", [])[:50]],
                    }
                    for s in sections[:60]
                ],
                key_terms=[
                    {"term": kt["term"][:200], "meaning": kt["meaning"][:2000]}
                    for kt in key_terms[:100]
                ],
                markdown=(r.get("markdown") or "")[:200_000],
                page_count=max(0, int(r.get("page_count") or 0)),
                warnings=[str(w)[:1000] for w in warnings[:50]],
                created_at=str(r.get("created_at") or updated_at.isoformat()),
            ).model_dump()
        raw_items.append((updated_at, "study_summary", eid, resolve_version("study_summary", eid), payload, deleted_at))

    # 12. Recorded meetings
    for row in cur.execute("SELECT * FROM recorded_meetings WHERE user_id = ?", (local_user_id,)):
        r = dict(row)
        if r.get("status") not in ("transcription_complete", "completed") and not r.get("deleted_at"):
            continue
        eid = str(r["id"])
        updated_at = parse_iso_datetime(r.get("updated_at") or r.get("created_at"), now)
        deleted_at = parse_iso_datetime(r["deleted_at"], updated_at) if r.get("deleted_at") else None
        if deleted_at:
            payload = {}
        else:
            transcript_raw = json.loads(r["transcript_json"]) if r.get("transcript_json") else None
            transcript = None
            if transcript_raw is not None:
                transcript = [
                    {
                        "start_ms": max(0, int(round(float(seg.get("startMs", seg.get("start_ms", 0)))))),
                        "end_ms": max(0, int(round(float(seg.get("endMs", seg.get("end_ms", 0)))))),
                        "text": str(seg.get("text", ""))[:10_000],
                    }
                    for seg in transcript_raw[:20_000]
                    if seg.get("text")
                ]
            notes_raw = json.loads(r["notes_json"]) if r.get("notes_json") else None
            notes = None
            if notes_raw is not None:
                notes = {
                    "title": str(notes_raw.get("title", ""))[:200],
                    "summary": str(notes_raw.get("summary", ""))[:20_000],
                    "key_topics": [
                        {
                            "topic": str(kt.get("topic", ""))[:5000],
                            "discussion": str(kt.get("discussion", ""))[:5000],
                        }
                        for kt in notes_raw.get("key_topics", [])[:200]
                    ],
                    "decisions": [str(x)[:5000] for x in notes_raw.get("decisions", [])[:200]],
                    "action_items": [
                        {
                            "task": str(ai.get("task", ""))[:5000],
                            "assignee": str(ai.get("assignee", ""))[:500],
                            "deadline": str(ai.get("deadline", ""))[:500],
                            "status": "done" if ai.get("status") == "done" else "pending",
                        }
                        for ai in notes_raw.get("action_items", [])[:200]
                    ],
                    "important_dates": [str(x)[:5000] for x in notes_raw.get("important_dates", [])[:200]],
                    "issues": [str(x)[:5000] for x in notes_raw.get("issues", [])[:200]],
                    "unresolved_questions": [str(x)[:5000] for x in notes_raw.get("unresolved_questions", [])[:200]],
                    "key_points": [str(x)[:5000] for x in notes_raw.get("key_points", [])[:200]],
                }
            payload = RecordedMeetingPayload(
                title=(r.get("title") or "Meeting")[:200],
                started_at=str(r["started_at"]),
                duration_seconds=max(0.0, float(r.get("duration_seconds") or 0)),
                source="upload" if r.get("source") == "upload" else "recording",
                language=r["language"][:16] if r.get("language") else None,
                status=r["status"],
                transcript=transcript,
                notes=notes,
                notes_edited=bool(r.get("notes_edited") == 1),
            ).model_dump()
        raw_items.append((updated_at, "recorded_meeting", eid, resolve_version("recorded_meeting", eid), payload, deleted_at))

    # Order deterministically by (updated_at, entity_type, entity_id) and assign sequential change_ids 1..N
    raw_items.sort(key=lambda item: (item[0], item[1], item[2]))
    prepared: list[PreparedSyncItem] = []
    for idx, (updated_at, entity_type, entity_id, version, payload, deleted_at) in enumerate(raw_items, start=1):
        prepared.append(
            PreparedSyncItem(
                entity_type=entity_type,
                entity_id=entity_id,
                version=version,
                change_id=idx,
                payload=payload,
                updated_at=updated_at,
                updated_at_iso=updated_at.astimezone(timezone.utc).isoformat(),
                deleted_at=deleted_at,
            )
        )
    return user, prepared


async def push_to_neon(user: dict[str, Any], items: list[PreparedSyncItem]) -> int:
    settings = get_settings()
    target_url, engine_kwargs = build_engine_kwargs(
        settings.get_migration_database_url(),
        is_serverless=True,
    )
    engine = create_async_engine(target_url, **engine_kwargs)
    owner_id = uuid.UUID(str(user["cloud_account_id"]))
    now = datetime.now(timezone.utc)
    created_at = parse_iso_datetime(user.get("created_at"), now)
    max_change_id = items[-1].change_id if items else 0

    try:
        async with engine.begin() as conn:
            for tbl in RLS_SYNC_TABLES:
                await conn.execute(text(f"ALTER TABLE {tbl} NO FORCE ROW LEVEL SECURITY"))
            await conn.execute(
                text("SELECT set_config('app.current_user_id', :uid, true)"),
                {"uid": str(owner_id)},
            )

            # 1. Upsert Account
            existing_account = (
                await conn.execute(select(Account).where(Account.id == owner_id))
            ).scalar_one_or_none()
            migrated_pw_hash = f"sha256${user['password_hash']}"
            if existing_account is None:
                await conn.execute(
                    insert(Account).values(
                        id=owner_id,
                        email=str(user["email"]).strip().lower(),
                        password_hash=migrated_pw_hash,
                        role="student_pro",
                        system_role="admin",
                        subscription_plan="student_pro",
                        is_active=True,
                        created_at=created_at,
                        updated_at=now,
                    )
                )
            else:
                await conn.execute(
                    text(
                        """
                        UPDATE accounts
                        SET email = :email,
                            role = 'student_pro',
                            system_role = 'admin',
                            subscription_plan = 'student_pro',
                            is_active = TRUE,
                            updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": owner_id,
                        "email": str(user["email"]).strip().lower(),
                        "updated_at": now,
                    },
                )

            # 2. Clear existing sync state for this owner so the bulk load is idempotent
            for model_cls in MODEL_MAP.values():
                await conn.execute(delete(model_cls).where(model_cls.owner_id == owner_id))
            await conn.execute(delete(ChangeFeed).where(ChangeFeed.owner_id == owner_id))
            await conn.execute(delete(SyncHead).where(SyncHead.owner_id == owner_id))
            await conn.execute(delete(IdempotentMutation).where(IdempotentMutation.owner_id == owner_id))

            # 3. Group items by entity_type and bulk insert into *_sync tables
            by_type: dict[str, list[dict[str, Any]]] = {}
            change_feed_rows: list[dict[str, Any]] = []
            for item in items:
                by_type.setdefault(item.entity_type, []).append(
                    {
                        "owner_id": owner_id,
                        "client_id": item.entity_id,
                        "version": item.version,
                        "change_id": item.change_id,
                        "payload": item.payload,
                        "updated_at": item.updated_at,
                        "deleted_at": item.deleted_at,
                    }
                )
                change_feed_rows.append(
                    {
                        "change_id": item.change_id,
                        "owner_id": owner_id,
                        "entity_type": item.entity_type,
                        "entity_id": item.entity_id,
                        "operation": "delete" if item.deleted_at is not None else "update",
                        "version": item.version,
                        "payload": item.payload,
                        "created_at": item.updated_at,
                    }
                )

            batch_size = 1000
            for entity_type, rows in by_type.items():
                model_cls = MODEL_MAP[entity_type]
                for i in range(0, len(rows), batch_size):
                    await conn.execute(insert(model_cls), rows[i : i + batch_size])
                print(f"  [Neon] Inserted {len(rows)} rows into {model_cls.__tablename__}")

            for i in range(0, len(change_feed_rows), batch_size):
                await conn.execute(insert(ChangeFeed), change_feed_rows[i : i + batch_size])
            print(f"  [Neon] Inserted {len(change_feed_rows)} rows into change_feed")

            await conn.execute(
                insert(SyncHead).values(
                    owner_id=owner_id,
                    latest_change_id=max_change_id,
                    updated_at=now,
                )
            )
            print(f"  [Neon] Updated sync_heads latest_change_id = {max_change_id}")

            seq_res = await conn.execute(
                text("SELECT pg_get_serial_sequence('change_feed', 'change_id')")
            )
            seq_name = seq_res.scalar_one_or_none()
            if seq_name and max_change_id > 0:
                await conn.execute(
                    text("SELECT setval(:seq_name, :max_id, true)"),
                    {"seq_name": seq_name, "max_id": max_change_id},
                )

            for tbl in RLS_SYNC_TABLES:
                await conn.execute(text(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY"))
    finally:
        await engine.dispose()

    return max_change_id


def align_local_sqlite(sqlite_path: Path, local_user_id: str, items: list[PreparedSyncItem], max_change_id: int) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    now_iso = datetime.now(timezone.utc).isoformat()
    all_types_json = json.dumps(sorted(SUPPORTED_ENTITY_TYPES), separators=(",", ":"))

    try:
        with conn:
            cur = conn.cursor()

            # 1. Replace sync_metadata with exact Neon versions and change_ids
            cur.execute("DELETE FROM sync_metadata")
            cur.executemany(
                """
                INSERT INTO sync_metadata (
                    user_id, scope_type, scope_id, entity_type, entity_id,
                    version, change_id, created_at, updated_at
                ) VALUES (?, 'account', ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        local_user_id,
                        local_user_id,
                        item.entity_type,
                        item.entity_id,
                        item.version,
                        item.change_id,
                        item.updated_at_iso,
                        item.updated_at_iso,
                    )
                    for item in items
                ],
            )

            # 2. Clear sync_outbox and sync_conflicts (all personal changes are now in Neon)
            cur.execute("DELETE FROM sync_outbox")
            cur.execute("DELETE FROM sync_conflicts")

            # 3. Update sync_state to match Neon cursor and negotiated entity types
            cur.execute("DELETE FROM sync_state WHERE user_id != ?", (local_user_id,))
            cur.execute(
                """
                INSERT INTO sync_state (
                    user_id, scope_type, scope_id, cursor, last_synced_at,
                    status, error_message, server_entity_types, snapshot_entity_types,
                    created_at, updated_at
                ) VALUES (?, 'account', ?, ?, ?, 'Synced', NULL, ?, ?, ?, ?)
                ON CONFLICT(user_id, scope_type, scope_id) DO UPDATE SET
                    cursor = excluded.cursor,
                    last_synced_at = excluded.last_synced_at,
                    status = 'Synced',
                    error_message = NULL,
                    server_entity_types = excluded.server_entity_types,
                    snapshot_entity_types = excluded.snapshot_entity_types,
                    updated_at = excluded.updated_at
                """,
                (
                    local_user_id,
                    local_user_id,
                    max_change_id,
                    now_iso,
                    all_types_json,
                    all_types_json,
                    now_iso,
                    now_iso,
                ),
            )

            # 4. Empty legacy business & Gmail tables and demo business users
            for tbl in LEGACY_BUSINESS_AND_GMAIL_TABLES:
                exists = cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,),
                ).fetchone()
                if exists:
                    cur.execute(f"DELETE FROM {tbl}")

            cur.execute(
                f"DELETE FROM users WHERE id IN ({','.join('?' for _ in DEMO_USER_IDS)})",
                DEMO_USER_IDS,
            )
    finally:
        conn.close()

    # Keep lafina.sqlite.bak aligned if present
    bak_path = sqlite_path.with_suffix(".sqlite.bak")
    if bak_path.exists():
        shutil.copy2(sqlite_path, bak_path)


async def main() -> None:
    sqlite_path = DEFAULT_SQLITE_PATH
    backup_path = Path(str(sqlite_path) + ".backup-20261023")
    if not backup_path.exists():
        shutil.copy2(sqlite_path, backup_path)
        print(f"Created safety backup at {backup_path}")
    else:
        print(f"Verified safety backup at {backup_path} ({backup_path.stat().st_size} bytes)")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        user, items = extract_personal_items(conn, TARGET_LOCAL_USER_ID)
    finally:
        conn.close()

    print(
        f"Prepared {len(items)} personal entities for {user['username']} "
        f"({user['email']}, cloud_account_id={user['cloud_account_id']})"
    )

    max_change_id = await push_to_neon(user, items)
    align_local_sqlite(sqlite_path, TARGET_LOCAL_USER_ID, items, max_change_id)
    print(f"Local SQLite and Neon Postgres are now synchronized at cursor {max_change_id}!")


if __name__ == "__main__":
    asyncio.run(main())
