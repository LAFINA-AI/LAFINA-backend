"""Entity-type negotiation and the study-tool sync entities.

Clients from before negotiation reject unknown entity types and require the
snapshot's authoritative list to match theirs exactly, so a request that
declares nothing must look exactly like the old protocol.
"""
import pytest
from httpx import AsyncClient

from app.api.v1 import sync as sync_api

LEGACY_AUTHORITATIVE = [
    "task",
    "event",
    "time_block",
    "reminder",
    "note",
    "custom_category",
]
ALL_TYPES = [
    "profile",
    "task",
    "event",
    "time_block",
    "reminder",
    "note",
    "custom_category",
    "pomodoro_settings",
    "pomodoro_session",
    "flashcard_deck",
    "study_summary",
    "recorded_meeting",
]
STAMP = "2026-09-19T10:00:00+08:00"


async def _register(async_client: AsyncClient, email: str) -> dict[str, str]:
    response = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "sync-test-password"},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _mutation(
    mutation_id: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, object],
    operation: str = "create",
    base_version: int | None = 0,
) -> dict[str, object]:
    mutation: dict[str, object] = {
        "mutationId": mutation_id,
        "entityType": entity_type,
        "entityId": entity_id,
        "operation": operation,
        "clientUpdatedAt": STAMP,
        "payload": payload,
    }
    if base_version is not None:
        mutation["baseVersion"] = base_version
    return mutation


def _deck(card_count: int = 2) -> dict[str, object]:
    return {
        "title": "Cell Biology",
        "source_name": "bio101.pdf",
        "cards": [
            {"question": f"Q{index}", "answer": f"A{index}"}
            for index in range(card_count)
        ],
        "page_count": 12,
        "ocr_page_count": 1,
        "warnings": ["Page 7 was scanned."],
        "created_at": "2026-09-19T02:00:00.000Z",
    }


def _meeting(status: str = "completed") -> dict[str, object]:
    return {
        "title": "Capstone sync",
        "started_at": "2026-09-19T01:00:00.000Z",
        "duration_seconds": 1834.5,
        "source": "recording",
        "language": "en",
        "status": status,
        "transcript": [
            {"start_ms": 0, "end_ms": 4200, "text": "Let's review the sprint."},
        ],
        "notes": {
            "title": "Capstone sync",
            "summary": "Reviewed the sprint.",
            "key_topics": [{"topic": "Sprint", "discussion": "On track."}],
            "decisions": ["Ship Friday"],
            "action_items": [
                {
                    "task": "Write tests",
                    "assignee": "Noel",
                    "deadline": "Friday",
                    "status": "pending",
                }
            ],
            "important_dates": [],
            "issues": [],
            "unresolved_questions": [],
            "key_points": ["Demo on Monday"],
        },
        "notes_edited": False,
    }


@pytest.mark.asyncio
async def test_legacy_requests_never_receive_study_tool_changes(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "negotiation-legacy@ustp.edu.ph")

    pushed = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("deck-1", "flashcard_deck", "deck-1", _deck()),
                _mutation("task-1", "task", "task-1", {"title": "Read ch. 3"}),
            ],
        },
    )
    assert pushed.status_code == 200
    assert len(pushed.json()["accepted"]) == 2

    legacy = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "mutations": []},
    )
    assert legacy.status_code == 200
    legacy_data = legacy.json()
    assert [item["entityType"] for item in legacy_data["changes"]] == ["task"]
    # The cursor stops at the last change actually sent, as old clients require.
    assert legacy_data["nextCursor"] == legacy_data["changes"][-1]["changeId"]

    # Only undeclared rows after the cursor: an empty page that doesn't move.
    deck_only = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("deck-2", "flashcard_deck", "deck-2", _deck()),
            ],
        },
    )
    assert deck_only.status_code == 200
    lagging = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": legacy_data["nextCursor"], "mutations": []},
    )
    lagging_data = lagging.json()
    assert lagging_data["changes"] == []
    assert lagging_data["nextCursor"] == legacy_data["nextCursor"]
    assert lagging_data["hasMore"] is False
    assert lagging_data["resetRequired"] is False

    # A later legacy change carries the cursor past the skipped deck.
    await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "mutations": [
                _mutation("task-2", "task", "task-2", {"title": "Quiz"}),
            ],
        },
    )
    caught_up = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={"cursor": legacy_data["nextCursor"], "mutations": []},
        )
    ).json()
    skipped_deck_change_id = deck_only.json()["changes"][-1]["changeId"]
    assert deck_only.json()["changes"][-1]["entityId"] == "deck-2"
    assert [item["entityId"] for item in caught_up["changes"]] == ["task-2"]
    assert caught_up["nextCursor"] > skipped_deck_change_id


@pytest.mark.asyncio
async def test_declared_requests_receive_new_types_and_supported_list(
    async_client: AsyncClient,
) -> None:
    headers = await _register(async_client, "negotiation-declared@ustp.edu.ph")
    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("deck-1", "flashcard_deck", "deck-1", _deck()),
                _mutation(
                    "session-1",
                    "pomodoro_session",
                    "session-1",
                    {
                        "phase": "focus",
                        "task": "Essay",
                        "duration_ms": 1_500_000,
                        "started_at": "2026-09-19T01:00:00.000Z",
                        "finished_at": "2026-09-19T01:25:00.000Z",
                    },
                ),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["supportedEntityTypes"] == ALL_TYPES
    assert [item["entityType"] for item in data["changes"]] == [
        "flashcard_deck",
        "pomodoro_session",
    ]
    assert data["changes"][0]["payload"]["cards"][1] == {
        "question": "Q1",
        "answer": "A1",
    }


@pytest.mark.asyncio
async def test_unknown_declared_types_are_ignored(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "negotiation-unknown@ustp.edu.ph")
    await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("deck-1", "flashcard_deck", "deck-1", _deck()),
                _mutation("task-1", "task", "task-1", {"title": "Read"}),
            ],
        },
    )
    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={"cursor": 0, "entityTypes": ["task", "hologram"], "mutations": []},
    )
    assert response.status_code == 200
    assert [item["entityType"] for item in response.json()["changes"]] == ["task"]


@pytest.mark.asyncio
async def test_snapshot_only_covers_declared_types(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "negotiation-snapshot@ustp.edu.ph")
    await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("deck-1", "flashcard_deck", "deck-1", _deck()),
                _mutation("task-1", "task", "task-1", {"title": "Read"}),
                _mutation(
                    "settings-1",
                    "pomodoro_settings",
                    "pomodoro_settings",
                    {"focus_minutes": 50},
                ),
            ],
        },
    )

    legacy = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={"cursor": 0, "mutations": [], "snapshot": {}},
        )
    ).json()["snapshot"]
    assert legacy["authoritativeEntityTypes"] == LEGACY_AUTHORITATIVE
    assert {item["entityType"] for item in legacy["items"]} <= {"profile", "task"}

    declared = (
        await async_client.post(
            "/v1/sync/batch",
            headers=headers,
            json={
                "cursor": 0,
                "entityTypes": ALL_TYPES,
                "mutations": [],
                "snapshot": {},
            },
        )
    ).json()["snapshot"]
    assert declared["authoritativeEntityTypes"] == [
        *LEGACY_AUTHORITATIVE,
        "pomodoro_session",
        "flashcard_deck",
        "study_summary",
        "recorded_meeting",
    ]
    types = {item["entityType"] for item in declared["items"]}
    assert {"task", "flashcard_deck", "pomodoro_settings"} <= types
    assert declared["boundaryCursor"] == legacy["boundaryCursor"]


@pytest.mark.asyncio
async def test_study_tool_payloads_are_canonicalized(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "negotiation-payloads@ustp.edu.ph")
    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation(
                    "settings-1",
                    "pomodoro_settings",
                    "pomodoro_settings",
                    {"focus_minutes": 50, "auto_start_focus": True},
                ),
                _mutation("meeting-1", "recorded_meeting", "meeting-1", _meeting()),
                _mutation(
                    "summary-1",
                    "study_summary",
                    "summary-1",
                    {
                        "title": "Thermodynamics",
                        "source_kind": "pptx",
                        "sections": [{"heading": "Laws", "points": ["Energy is conserved."]}],
                        "key_terms": [{"term": "Entropy", "meaning": "Disorder."}],
                        "markdown": "# Thermodynamics",
                        "created_at": "2026-09-19T02:00:00.000Z",
                    },
                ),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rejected"] == []
    payloads = {item["entityType"]: item["payload"] for item in data["changes"]}
    assert payloads["pomodoro_settings"] == {
        "focus_minutes": 50,
        "short_break_minutes": 5,
        "long_break_minutes": 15,
        "long_break_interval": 4,
        "auto_start_breaks": True,
        "auto_start_focus": True,
        "sound_enabled": True,
        "volume": 0.7,
        "ring_seconds": 10,
        "notifications_enabled": True,
    }
    assert payloads["recorded_meeting"]["notes"]["action_items"][0]["status"] == "pending"
    assert payloads["study_summary"]["overview"] == ""
    assert payloads["study_summary"]["warnings"] == []


@pytest.mark.asyncio
async def test_study_tool_payload_rejections(async_client: AsyncClient) -> None:
    headers = await _register(async_client, "negotiation-reject@ustp.edu.ph")
    settings = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation(
                    "settings-1",
                    "pomodoro_settings",
                    "pomodoro_settings",
                    {"focus_minutes": 30},
                ),
            ],
        },
    )
    assert settings.json()["accepted"][0]["serverVersion"] == 1

    response = await async_client.post(
        "/v1/sync/batch",
        headers=headers,
        json={
            "cursor": 0,
            "entityTypes": ALL_TYPES,
            "mutations": [
                _mutation("too-many", "flashcard_deck", "deck-big", _deck(201)),
                _mutation(
                    "bad-phase",
                    "pomodoro_session",
                    "session-bad",
                    {
                        "phase": "nap",
                        "duration_ms": 1000,
                        "finished_at": "2026-09-19T01:25:00.000Z",
                    },
                ),
                _mutation(
                    "in-progress",
                    "recorded_meeting",
                    "meeting-live",
                    _meeting(status="recording"),
                ),
                _mutation(
                    "forbidden",
                    "study_summary",
                    "summary-owner",
                    {
                        "title": "x",
                        "created_at": "2026-09-19T02:00:00.000Z",
                        "owner_id": "someone-else",
                    },
                ),
                _mutation(
                    "settings-delete",
                    "pomodoro_settings",
                    "pomodoro_settings",
                    {},
                    operation="delete",
                    base_version=1,
                ),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] == []
    reasons = {item["mutationId"]: item["reason"] for item in data["rejected"]}
    assert reasons["too-many"].startswith("Payload validation failed")
    assert reasons["bad-phase"].startswith("Payload validation failed")
    assert reasons["in-progress"].startswith("Payload validation failed")
    assert reasons["forbidden"] == "Forbidden untrusted client payload fields detected"
    assert reasons["settings-delete"] == "pomodoro_settings_delete_not_allowed"


def test_transcript_text_is_capped_as_a_whole() -> None:
    segment = {"start_ms": 0, "end_ms": 1, "text": "x" * 10_000}
    payload = _meeting()
    payload["transcript"] = [segment] * 61
    with pytest.raises(ValueError):
        sync_api.RecordedMeetingPayload(**payload)
