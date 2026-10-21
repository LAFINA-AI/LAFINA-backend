import uuid
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1 import sync as sync_api
from app.database import set_transaction_rls_user
from app.models.change_feed import ChangeFeed
from app.models.mutations import IdempotentMutation
from app.models.synchronized_content import ProfileSync, TasksSync
from tests.conftest import TestingSessionLocal


async def _register(async_client: AsyncClient, email: str) -> dict[str, str]:
    response = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "sync-test-password"},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _task_mutation(
    mutation_id: str,
    entity_id: str,
    title: str,
    base_version: int | None = None,
) -> dict[str, object]:
    mutation: dict[str, object] = {
        "mutationId": mutation_id,
        "entityType": "task",
        "entityId": entity_id,
        "operation": "create" if base_version in (None, 0) else "update",
        "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
        "payload": {"title": title},
    }
    if base_version is not None:
        mutation["baseVersion"] = base_version
    return mutation


@pytest.mark.asyncio
async def test_change_ids_are_generated_and_monotonic(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "sequence@ustp.edu.ph")
    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("seq-1", "task-seq-1", "First", 0),
                _task_mutation("seq-2", "task-seq-2", "Second", 0),
            ],
        },
    )

    assert response.status_code == 200
    change_ids = [item["changeId"] for item in response.json()["changes"]]
    assert len(change_ids) == 2
    assert change_ids == sorted(change_ids)
    assert len(set(change_ids)) == 2

    async with TestingSessionLocal() as db:
        direct_change = ChangeFeed(
            owner_id=(
                await db.execute(select(TasksSync.owner_id).limit(1))
            ).scalar_one(),
            entity_type="task",
            entity_id="direct-sequence-check",
            operation="create",
            version=1,
            payload={"title": "Direct"},
        )
        db.add(direct_change)
        await db.flush()
        assert direct_change.change_id > change_ids[-1]
        await db.rollback()


@pytest.mark.asyncio
async def test_idempotency_and_change_feed_are_owner_scoped(
    async_client: AsyncClient,
) -> None:
    first_headers = await _register(async_client, "owner-one@ustp.edu.ph")
    second_headers = await _register(async_client, "owner-two@ustp.edu.ph")
    shared_mutation_id = "shared-device-mutation"
    shared_entity_id = "shared-client-task-id"

    first_response = await async_client.post(
        "/v1/sync/batch",
        headers=first_headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation(
                    shared_mutation_id,
                    shared_entity_id,
                    "First owner's task",
                    0,
                )
            ],
        },
    )
    second_response = await async_client.post(
        "/v1/sync/batch",
        headers=second_headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation(
                    shared_mutation_id,
                    shared_entity_id,
                    "Second owner's task",
                    0,
                )
            ],
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["changes"][0]["payload"]["title"] == "First owner's task"
    assert second_response.json()["changes"][0]["payload"]["title"] == "Second owner's task"

    first_cursor = first_response.json()["nextCursor"]
    duplicate_response = await async_client.post(
        "/v1/sync/batch",
        headers=first_headers,
        json={
            "cursor": first_cursor,
            "mutations": [
                _task_mutation(
                    shared_mutation_id,
                    shared_entity_id,
                    "This must not overwrite",
                    1,
                )
            ],
        },
    )
    duplicate_data = duplicate_response.json()
    assert duplicate_response.status_code == 200
    assert duplicate_data["accepted"][0]["reason"] == "idempotent_duplicate"
    assert duplicate_data["changes"] == []

    async with TestingSessionLocal() as db:
        idempotency_count = (
            await db.execute(
                select(func.count(IdempotentMutation.mutation_id)).where(
                    IdempotentMutation.mutation_id == shared_mutation_id
                )
            )
        ).scalar_one()
        assert idempotency_count == 2
        audited_mutations = list(
            (
                await db.execute(
                    select(IdempotentMutation).where(
                        IdempotentMutation.mutation_id == shared_mutation_id
                    )
                )
            ).scalars()
        )
        assert all(record.client_updated_at is not None for record in audited_mutations)

        tasks = list(
            (
                await db.execute(
                    select(TasksSync).where(
                        TasksSync.client_id == shared_entity_id
                    )
                )
            ).scalars()
        )
        assert sorted(task.payload["title"] for task in tasks) == [
            "First owner's task",
            "Second owner's task",
        ]


@pytest.mark.asyncio
async def test_version_conflict_rejects_stale_write_but_legacy_write_still_works(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "conflict@ustp.edu.ph")
    entity_id = "conflict-task"

    create_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("conflict-create", entity_id, "Version one", 0)
            ],
        },
    )
    assert create_response.status_code == 200
    assert create_response.json()["accepted"][0]["serverVersion"] == 1

    update_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": create_response.json()["nextCursor"],
            "mutations": [
                _task_mutation("conflict-update", entity_id, "Version two", 1)
            ],
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["accepted"][0]["serverVersion"] == 2

    stale_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": update_response.json()["nextCursor"],
            "mutations": [
                _task_mutation("conflict-stale", entity_id, "Stale overwrite", 1)
            ],
        },
    )
    stale_data = stale_response.json()
    assert stale_response.status_code == 200
    assert stale_data["accepted"] == []
    assert stale_data["rejected"][0]["reason"] == "version_conflict"
    assert stale_data["rejected"][0]["serverVersion"] == 2
    assert stale_data["rejected"][0]["serverPayload"]["title"] == "Version two"
    assert stale_data["changes"] == []

    stale_retry = _task_mutation(
        "conflict-stale",
        entity_id,
        "A reused mutation ID must remain rejected",
        2,
    )
    retry_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": update_response.json()["nextCursor"],
            "mutations": [stale_retry],
        },
    )
    assert retry_response.status_code == 200
    assert retry_response.json()["rejected"][0]["reason"] == "version_conflict"
    assert retry_response.json()["changes"] == []

    legacy_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": update_response.json()["nextCursor"],
            "mutations": [
                _task_mutation("conflict-legacy", entity_id, "Legacy last write")
            ],
        },
    )
    assert legacy_response.status_code == 200
    assert legacy_response.json()["accepted"][0]["serverVersion"] == 3

    async with TestingSessionLocal() as db:
        task = (
            await db.execute(
                select(TasksSync).where(TasksSync.client_id == entity_id)
            )
        ).scalar_one()
        assert task.version == 3
        assert task.payload["title"] == "Legacy last write"


@pytest.mark.asyncio
async def test_create_base_version_zero_rejects_existing_entity_collision(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "create-collision@ustp.edu.ph")
    entity_id = "create-collision-task"

    first_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("collision-first", entity_id, "Original", 0)
            ],
        },
    )
    collision_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": first_response.json()["nextCursor"],
            "mutations": [
                _task_mutation(
                    "collision-second",
                    entity_id,
                    "Must not overwrite",
                    0,
                )
            ],
        },
    )

    assert first_response.status_code == 200
    assert collision_response.status_code == 200
    collision = collision_response.json()
    assert collision["accepted"] == []
    assert collision["rejected"][0]["reason"] == "version_conflict"
    assert collision["rejected"][0]["serverVersion"] == 1
    assert collision["rejected"][0]["serverPayload"]["title"] == "Original"
    assert collision["changes"] == []

    async with TestingSessionLocal() as db:
        task = (
            await db.execute(
                select(TasksSync).where(TasksSync.client_id == entity_id)
            )
        ).scalar_one()
        assert task.version == 1
        assert task.payload["title"] == "Original"


@pytest.mark.asyncio
async def test_task_priority_and_note_tags_use_canonical_payloads(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "canonical-payloads@ustp.edu.ph")

    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("priority-default", "task-default", "Default", 0),
                {
                    **_task_mutation(
                        "priority-low",
                        "task-low",
                        "Lowercase input",
                        0,
                    ),
                    "payload": {"title": "Lowercase input", "priority": "low"},
                },
                {
                    **_task_mutation(
                        "priority-invalid",
                        "task-invalid",
                        "Invalid priority",
                        0,
                    ),
                    "payload": {"title": "Invalid priority", "priority": "urgent"},
                },
                {
                    "mutationId": "note-default-tags",
                    "entityType": "note",
                    "entityId": "note-default",
                    "operation": "create",
                    "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
                    "baseVersion": 0,
                    "payload": {"title": "No tags", "body": "Body"},
                },
                {
                    "mutationId": "note-canonical-tags",
                    "entityType": "note",
                    "entityId": "note-canonical",
                    "operation": "create",
                    "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
                    "baseVersion": 0,
                    "payload": {
                        "title": "Tags",
                        "body": "Body",
                        "tags": '[ "sales", "urgent" ]',
                    },
                },
                {
                    "mutationId": "note-invalid-tags",
                    "entityType": "note",
                    "entityId": "note-invalid",
                    "operation": "create",
                    "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
                    "baseVersion": 0,
                    "payload": {
                        "title": "Bad tags",
                        "body": "Body",
                        "tags": "not-json",
                    },
                },
                {
                    "mutationId": "note-non-array-tags",
                    "entityType": "note",
                    "entityId": "note-non-array",
                    "operation": "create",
                    "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
                    "baseVersion": 0,
                    "payload": {
                        "title": "Object tags",
                        "body": "Body",
                        "tags": '{"department":"sales"}',
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    payload_by_id = {
        change["entityId"]: change["payload"] for change in data["changes"]
    }
    rejected_by_id = {
        result["entityId"]: result for result in data["rejected"]
    }
    assert payload_by_id["task-default"]["priority"] == "Medium"
    assert payload_by_id["task-low"]["priority"] == "Low"
    assert payload_by_id["note-default"]["tags"] == "[]"
    assert payload_by_id["note-canonical"]["tags"] == '["sales","urgent"]'
    assert "priority must be High, Medium, or Low" in rejected_by_id[
        "task-invalid"
    ]["reason"]
    assert "tags must be a JSON array of strings" in rejected_by_id[
        "note-invalid"
    ]["reason"]
    assert "tags must be a JSON array of strings" in rejected_by_id[
        "note-non-array"
    ]["reason"]


@pytest.mark.asyncio
async def test_client_updated_at_requires_timezone(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "timestamp@ustp.edu.ph")
    mutation = _task_mutation("naive-time", "naive-time-task", "Invalid", 0)
    mutation["clientUpdatedAt"] = "2026-08-24T10:00:00"

    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [mutation]},
    )

    assert response.status_code == 422
    assert "timezone offset" in response.text


@pytest.mark.asyncio
async def test_missing_owner_cursor_requires_reset(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "missing-cursor@ustp.edu.ph")

    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 999_999, "mutations": []},
    )

    assert response.status_code == 200
    assert response.json()["resetRequired"] is True


@pytest.mark.asyncio
async def test_pruned_deltas_return_authoritative_live_and_tombstone_snapshot(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "snapshot-pruned@ustp.edu.ph")
    create_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("snapshot-live", "live-task", "Still live", 0),
                _task_mutation(
                    "snapshot-delete-create",
                    "deleted-task",
                    "Delete me",
                    0,
                ),
            ],
        },
    )
    delete_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": create_response.json()["nextCursor"],
            "mutations": [
                {
                    "mutationId": "snapshot-delete",
                    "entityType": "task",
                    "entityId": "deleted-task",
                    "operation": "delete",
                    "clientUpdatedAt": "2026-08-24T10:01:00+08:00",
                    "baseVersion": 1,
                    "payload": {},
                }
            ],
        },
    )
    assert create_response.status_code == 200
    assert delete_response.status_code == 200

    async with TestingSessionLocal() as db:
        owner_id = (
            await db.execute(
                select(TasksSync.owner_id).where(
                    TasksSync.client_id == "live-task"
                )
            )
        ).scalar_one()
        await db.execute(
            delete(ChangeFeed).where(ChangeFeed.owner_id == owner_id)
        )
        await db.commit()

    reset_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": []},
    )
    assert reset_response.status_code == 200
    reset_data = reset_response.json()
    assert reset_data["resetRequired"] is True
    assert reset_data["changes"] == []
    assert reset_data["snapshot"] is None

    snapshot_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [], "snapshot": {}},
    )
    assert snapshot_response.status_code == 200
    snapshot_data = snapshot_response.json()
    snapshot = snapshot_data["snapshot"]
    assert snapshot_data["resetRequired"] is False
    assert snapshot_data["changes"] == []
    assert snapshot["complete"] is True
    assert snapshot["hasMore"] is False
    assert snapshot["nextAfter"] is None
    assert snapshot["boundaryCursor"] == delete_response.json()["nextCursor"]
    assert snapshot["authoritativeEntityTypes"] == [
        "task",
        "event",
        "time_block",
        "reminder",
        "note",
        "custom_category",
    ]
    assert snapshot["prunePolicy"] == {
        "preserveOutboxStatuses": ["pending", "in_progress", "failed"],
        "requireExistingSyncMetadata": True,
    }

    items_by_id = {item["entityId"]: item for item in snapshot["items"]}
    assert items_by_id["live-task"]["operation"] == "update"
    assert items_by_id["live-task"]["payload"]["title"] == "Still live"
    assert items_by_id["live-task"]["payload"]["priority"] == "Medium"
    assert items_by_id["deleted-task"]["operation"] == "delete"
    assert items_by_id["deleted-task"]["payload"] == {}
    assert items_by_id["deleted-task"]["deletedAt"] is not None

    # Once the retained tombstone itself expires, snapshot completeness (rather
    # than a delete delta) tells clients to prune a previously synchronized row.
    async with TestingSessionLocal() as db:
        await db.execute(
            delete(TasksSync).where(TasksSync.client_id == "deleted-task")
        )
        await db.commit()

    pruned_tombstone_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [], "snapshot": {}},
    )
    pruned_snapshot = pruned_tombstone_response.json()["snapshot"]
    assert pruned_snapshot["boundaryCursor"] == snapshot["boundaryCursor"]
    assert [item["entityId"] for item in pruned_snapshot["items"]] == [
        "live-task"
    ]

    boundary_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": snapshot["boundaryCursor"],
            "mutations": [],
        },
    )
    assert boundary_response.status_code == 200
    assert boundary_response.json()["resetRequired"] is False
    assert boundary_response.json()["changes"] == []


@pytest.mark.asyncio
async def test_snapshot_boundary_is_stable_and_concurrent_write_is_a_delta(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sync_api, "SNAPSHOT_PAGE_SIZE", 1)
    headers = await _register(async_client, "snapshot-pages@ustp.edu.ph")
    create_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _task_mutation("page-a", "a-task", "A", 0),
                _task_mutation("page-b", "b-task", "B", 0),
            ],
        },
    )
    assert create_response.status_code == 200

    first_page_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [], "snapshot": {}},
    )
    first_page = first_page_response.json()["snapshot"]
    assert first_page["hasMore"] is True
    assert first_page["complete"] is False
    assert [item["entityId"] for item in first_page["items"]] == ["a-task"]
    boundary = first_page["boundaryCursor"]

    concurrent_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": boundary,
            "mutations": [
                _task_mutation("page-c", "c-task", "Concurrent", 0)
            ],
        },
    )
    assert concurrent_response.status_code == 200
    concurrent_change = concurrent_response.json()["changes"][0]
    assert concurrent_change["entityId"] == "c-task"
    assert concurrent_change["changeId"] > boundary

    second_page_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [],
            "snapshot": {
                "boundaryCursor": boundary,
                "after": first_page["nextAfter"],
            },
        },
    )
    second_page = second_page_response.json()["snapshot"]
    assert second_page["boundaryCursor"] == boundary
    assert second_page["complete"] is True
    assert [item["entityId"] for item in second_page["items"]] == ["b-task"]

    delta_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": boundary, "mutations": []},
    )
    assert delta_response.status_code == 200
    assert [item["entityId"] for item in delta_response.json()["changes"]] == [
        "c-task"
    ]


@pytest.mark.asyncio
async def test_profile_delete_is_terminal_and_idempotent(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "profile-delete@ustp.edu.ph")
    mutation = {
        "mutationId": "delete-profile",
        "entityType": "profile",
        "entityId": "local-profile",
        "operation": "delete",
        "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
        "payload": {},
    }

    first_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [mutation]},
    )
    duplicate_response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": [mutation]},
    )

    assert first_response.status_code == 200
    assert first_response.json()["accepted"] == []
    assert first_response.json()["rejected"][0]["reason"] == (
        "profile_delete_not_allowed"
    )
    assert first_response.json()["changes"] == []
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["rejected"] == first_response.json()["rejected"]

    async with TestingSessionLocal() as db:
        profile_count = (
            await db.execute(select(func.count(ProfileSync.client_id)))
        ).scalar_one()
        terminal_count = (
            await db.execute(
                select(func.count(IdempotentMutation.mutation_id)).where(
                    IdempotentMutation.mutation_id == "delete-profile"
                )
            )
        ).scalar_one()
        assert profile_count == 0
        assert terminal_count == 1


@pytest.mark.asyncio
async def test_incomplete_profile_uses_local_preference_defaults(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "profile-defaults@ustp.edu.ph")

    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                {
                    "mutationId": "create-minimal-profile",
                    "entityType": "profile",
                    "entityId": "minimal-profile",
                    "operation": "create",
                    "clientUpdatedAt": "2026-08-24T10:00:00+08:00",
                    "baseVersion": 0,
                    "payload": {"username": "Minimal User"},
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()["changes"][0]["payload"]
    assert payload["study_peak_hours"] == "[]"
    assert payload["snooze_tendency"] == "snooze_once"
    assert payload["weekly_class_count"] == "4-6"
    assert payload["longest_class_gap"] == "1 hour"


@pytest.mark.asyncio
async def test_postgresql_rls_context_is_transaction_local() -> None:
    execute = AsyncMock()
    fake_session = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(
            dialect=SimpleNamespace(name="postgresql")
        ),
        execute=execute,
    )
    account_id = uuid.uuid4()

    await set_transaction_rls_user(
        cast(AsyncSession, fake_session),
        account_id,
    )

    statement, parameters = execute.await_args.args
    assert "set_config('app.current_user_id'" in str(statement)
    assert ", true)" in str(statement)
    assert parameters == {"account_id": str(account_id)}
