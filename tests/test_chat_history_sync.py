"""Chat history sync: sessions and messages follow the account across phones.

Chat edits are last-write-wins (no baseVersion), deleting a chat deletes its
messages, deleted text does not stay in the change feed, a message written
into a chat another phone deleted brings the chat back, and the purge route
removes the account's chat history and nothing else.
"""
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.api.v1 import sync as sync_api
from app.models.change_feed import ChangeFeed
from tests.conftest import TestingSessionLocal

CHAT_TYPES = ["profile", "task", "chat_session", "chat_message"]
STAMP = "2026-11-21T10:00:00+08:00"
SESSION = "chat_session:chat_abc"


async def _register(async_client: AsyncClient, email: str) -> dict[str, str]:
    response = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "chat-sync-password"},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _mutation(
    mutation_id: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, object],
    operation: str = "create",
) -> dict[str, object]:
    # Chat is last-write-wins: the app never sends a baseVersion for it.
    return {
        "mutationId": mutation_id,
        "entityType": entity_type,
        "entityId": entity_id,
        "operation": operation,
        "clientUpdatedAt": STAMP,
        "payload": payload,
    }


def _session(title: str = "Thesis help") -> dict[str, object]:
    return {"title": title, "kind": "general", "created_at": "2026-11-21T02:00:00.000Z"}


def _message(
    content: str = "When is my thesis due?",
    session_id: str = SESSION,
    sender: str = "user",
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "sender": sender,
        "content": content,
        "created_at": "2026-11-21T02:00:01.000Z",
    }


async def _batch(
    async_client: AsyncClient,
    headers: dict[str, str],
    mutations: list[dict[str, object]],
    cursor: int = 0,
    entity_types: list[str] | None = CHAT_TYPES,
) -> dict:
    body: dict[str, object] = {"cursor": cursor, "mutations": mutations}
    if entity_types is not None:
        body["entityTypes"] = entity_types
    response = await async_client.post("/v1/sync/batch", headers=headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def _feed_payloads(entity_id: str) -> list[dict]:
    async with TestingSessionLocal() as session:
        rows = (
            await session.execute(
                select(ChangeFeed)
                .where(ChangeFeed.entity_id == entity_id)
                .order_by(ChangeFeed.change_id)
            )
        ).scalars().all()
        return [dict(row.payload) for row in rows]


@pytest.mark.asyncio
async def test_chat_round_trips_with_canonical_payloads(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "chat-roundtrip@ustp.edu.ph")
    data = await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation(
                "m-1",
                "chat_message",
                "msg_1",
                {
                    **_message("Here is the file", sender="assistant"),
                    "handbook_note": "Handbook p. 12",
                    "local_reply": True,
                    "review_resolution": {"status": "confirmed", "summary": "Added"},
                    "attachment": {
                        "file_name": "plan.pdf",
                        "format": "pdf",
                        "mime_type": "application/pdf",
                        "size_bytes": 2048,
                    },
                },
            ),
        ],
    )
    assert data["rejected"] == []
    assert "chat_session" in data["supportedEntityTypes"]
    assert "chat_message" in data["supportedEntityTypes"]
    by_type = {item["entityType"]: item for item in data["changes"]}
    assert by_type["chat_session"]["payload"] == _session()
    message = by_type["chat_message"]["payload"]
    assert message["voice_interrupted"] is False
    assert message["review_resolution"] == {
        "status": "confirmed",
        "summary": "Added",
        "resolved_at": None,
    }
    assert message["attachment"]["file_name"] == "plan.pdf"


@pytest.mark.asyncio
async def test_clients_that_do_not_declare_chat_never_receive_it(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "chat-legacy@ustp.edu.ph")
    await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message()),
            _mutation("t-1", "task", "task-1", {"title": "Read"}),
        ],
    )
    legacy = await _batch(async_client, headers, [], entity_types=None)
    assert [item["entityType"] for item in legacy["changes"]] == ["task"]
    planner_only = await _batch(
        async_client, headers, [], entity_types=["profile", "task"]
    )
    assert [item["entityType"] for item in planner_only["changes"]] == ["task"]


@pytest.mark.asyncio
async def test_chat_edits_are_last_write_wins(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "chat-lww@ustp.edu.ph")
    await _batch(
        async_client, headers, [_mutation("s-1", "chat_session", SESSION, _session())]
    )
    # Two phones rename the same chat without knowing each other's version.
    first = await _batch(
        async_client,
        headers,
        [_mutation("s-2", "chat_session", SESSION, _session("From phone A"), "update")],
    )
    second = await _batch(
        async_client,
        headers,
        [_mutation("s-3", "chat_session", SESSION, _session("From phone B"), "update")],
    )
    assert first["rejected"] == [] and second["rejected"] == []
    assert second["accepted"][0]["serverVersion"] == 3
    snapshot = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={"cursor": 0, "mutations": [], "entityTypes": CHAT_TYPES, "snapshot": {}},
        )
    ).json()["snapshot"]
    titles = [
        item["payload"]["title"]
        for item in snapshot["items"]
        if item["entityType"] == "chat_session"
    ]
    assert titles == ["From phone B"]


@pytest.mark.asyncio
async def test_chat_payload_rejections(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "chat-reject@ustp.edu.ph")
    data = await _batch(
        async_client,
        headers,
        [
            _mutation(
                "too-long",
                "chat_message",
                "msg_long",
                _message("x" * (sync_api.MAX_CHAT_MESSAGE_CHARS + 1)),
            ),
            _mutation("bad-sender", "chat_message", "msg_bad", _message(sender="system")),
            _mutation(
                "file-path",
                "chat_message",
                "msg_path",
                {
                    **_message(),
                    "attachment": {"file_name": "a.pdf", "uri": "file:///data/a.pdf"},
                },
            ),
            _mutation(
                "memory",
                "chat_message",
                "msg_memory",
                {**_message(), "memory_changes": {"added": []}},
            ),
            _mutation("bad-kind", "chat_session", "chat_session:x", {**_session(), "kind": "work"}),
        ],
    )
    assert data["accepted"] == []
    reasons = {item["mutationId"]: item["reason"] for item in data["rejected"]}
    assert set(reasons) == {"too-long", "bad-sender", "file-path", "memory", "bad-kind"}
    assert all(reason.startswith("Payload validation failed") for reason in reasons.values())


@pytest.mark.asyncio
async def test_deleting_a_chat_deletes_its_messages_and_their_text(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "chat-cascade@ustp.edu.ph")
    other = "chat_session:chat_other"
    created = await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message("secret one")),
            _mutation("m-2", "chat_message", "msg_2", _message("secret two", sender="assistant")),
            _mutation("s-2", "chat_session", other, _session("Keep me")),
            _mutation("m-3", "chat_message", "msg_3", _message("kept", session_id=other)),
        ],
    )
    cursor = created["nextCursor"]

    deleted = await _batch(
        async_client,
        headers,
        [_mutation("s-del", "chat_session", SESSION, {}, "delete")],
        cursor=cursor,
    )
    assert deleted["rejected"] == []
    deletes = {
        (item["entityType"], item["entityId"])
        for item in deleted["changes"]
        if item["operation"] == "delete"
    }
    assert deletes == {
        ("chat_message", "msg_1"),
        ("chat_message", "msg_2"),
        ("chat_session", SESSION),
    }
    # Messages are deleted before their session in the feed.
    order = [item["entityId"] for item in deleted["changes"]]
    assert order.index(SESSION) > order.index("msg_1")

    # No copy of the deleted text is left in the change feed.
    for entity_id in ("msg_1", "msg_2", SESSION):
        assert await _feed_payloads(entity_id) == [{}]
    assert await _feed_payloads("msg_3") != [{}]

    # A phone restoring from scratch sees tombstones without text, and the other chat.
    snapshot = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={"cursor": 0, "mutations": [], "entityTypes": CHAT_TYPES, "snapshot": {}},
        )
    ).json()["snapshot"]
    items = {item["entityId"]: item for item in snapshot["items"]}
    assert items["msg_1"]["operation"] == "delete" and items["msg_1"]["payload"] == {}
    assert items[SESSION]["operation"] == "delete"
    assert items["msg_3"]["payload"]["content"] == "kept"
    assert snapshot["authoritativeEntityTypes"][-2:] == ["chat_session", "chat_message"]
    session_ranks = [i for i, item in enumerate(snapshot["items"]) if item["entityType"] == "chat_session"]
    message_ranks = [i for i, item in enumerate(snapshot["items"]) if item["entityType"] == "chat_message"]
    assert max(session_ranks) < min(message_ranks)

    # A phone that was behind the delete still converges from its old cursor.
    replay = await _batch(async_client, headers, [], cursor=0)
    final = {item["entityId"]: item["operation"] for item in replay["changes"]}
    assert final["msg_1"] == "delete" and final[SESSION] == "delete"
    assert final["msg_3"] == "create"


@pytest.mark.asyncio
async def test_stale_edit_does_not_restore_a_deleted_message(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "chat-stale@ustp.edu.ph")
    await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message()),
        ],
    )
    await _batch(
        async_client, headers, [_mutation("m-del", "chat_message", "msg_1", {}, "delete")]
    )
    stale = await _batch(
        async_client,
        headers,
        [_mutation("m-edit", "chat_message", "msg_1", _message("edited"), "update")],
    )
    assert stale["accepted"][0]["reason"] == "already_deleted"
    snapshot = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={"cursor": 0, "mutations": [], "entityTypes": CHAT_TYPES, "snapshot": {}},
        )
    ).json()["snapshot"]
    items = {item["entityId"]: item for item in snapshot["items"]}
    assert items["msg_1"]["operation"] == "delete"


@pytest.mark.asyncio
async def test_new_message_into_a_deleted_chat_brings_the_chat_back(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "chat-revive@ustp.edu.ph")
    await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message()),
        ],
    )
    deleted = await _batch(
        async_client, headers, [_mutation("s-del", "chat_session", SESSION, {}, "delete")]
    )
    # Written offline on a phone that had not heard of the delete yet.
    revived = await _batch(
        async_client,
        headers,
        [_mutation("m-2", "chat_message", "msg_2", _message("written offline"))],
        cursor=deleted["nextCursor"],
    )
    sessions = [item for item in revived["changes"] if item["entityType"] == "chat_session"]
    assert len(sessions) == 1
    assert sessions[0]["operation"] == "update"
    assert sessions[0]["payload"] == {
        "title": "",
        "kind": "general",
        "created_at": "2026-11-21T02:00:01.000Z",
    }

    # A message into a chat the server never saw creates that chat too.
    ustp = await _batch(
        async_client,
        headers,
        [
            _mutation(
                "m-3",
                "chat_message",
                "msg_3",
                _message("Ask USTP", session_id="chat_session:ustp_chat_1"),
            )
        ],
        cursor=revived["nextCursor"],
    )
    created = [item for item in ustp["changes"] if item["entityType"] == "chat_session"]
    assert created[0]["operation"] == "create"
    assert created[0]["payload"]["kind"] == "ustp"


@pytest.mark.asyncio
async def test_purge_removes_only_chat_history(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "chat-purge@ustp.edu.ph")
    created = await _batch(
        async_client,
        headers,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message("private")),
            _mutation("m-2", "chat_message", "msg_2", _message("also private")),
            _mutation("t-1", "task", "task-1", {"title": "Keep this task"}),
        ],
    )

    unauthenticated = await async_client.post("/v1/sync/chat/purge")
    assert unauthenticated.status_code == 401

    purge = await async_client.post("/v1/sync/chat/purge", headers=headers)
    assert purge.status_code == 200
    assert purge.json()["deletedSessions"] == 1
    assert purge.json()["deletedMessages"] == 2
    again = await async_client.post("/v1/sync/chat/purge", headers=headers)
    assert again.json()["deletedSessions"] == 0
    assert again.json()["deletedMessages"] == 0

    after = await _batch(async_client, headers, [], cursor=created["nextCursor"])
    assert {
        (item["entityType"], item["operation"]) for item in after["changes"]
    } == {("chat_session", "delete"), ("chat_message", "delete")}
    assert await _feed_payloads("msg_1") == [{}]
    assert await _feed_payloads("task-1") != [{}]


@pytest.mark.asyncio
async def test_chat_is_isolated_between_accounts(async_client: AsyncClient) -> None:
    alice = await _register(async_client, "chat-alice@ustp.edu.ph")
    bob = await _register(async_client, "chat-bob@ustp.edu.ph")
    await _batch(
        async_client,
        alice,
        [
            _mutation("s-1", "chat_session", SESSION, _session()),
            _mutation("m-1", "chat_message", "msg_1", _message("alice only")),
        ],
    )
    bob_view = await _batch(async_client, bob, [])
    assert bob_view["changes"] == []
    await async_client.post("/v1/sync/chat/purge", headers=bob)
    alice_view = await _batch(async_client, alice, [])
    assert {item["operation"] for item in alice_view["changes"]} == {"create"}
