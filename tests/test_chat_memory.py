"""Chat memory: memories in the chat prompt, and POST /v1/ai/memory/extract (synthetic DeepSeek)."""

import json

import httpx
import pytest
from sqlalchemy import select

import app.api.v1.ai as ai_module
from app.api.v1.ai import get_deepseek_client
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.main import app
from app.models.ai_usage import AIUsage
from app.services.chat_memory import (
    MEMORY_PROMPT_BUDGET,
    MemoryItem,
    build_memory_prompt,
    parse_extraction,
)
from tests.conftest import TestingSessionLocal
from tests.test_online_voices import signed_in


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135},
        },
    )


@pytest.fixture
def upstream():
    """Sets what the fake DeepSeek answers with; records every request."""
    state = {"response": completion("Hello!"), "requests": []}

    def transport(request):
        state["requests"].append(request)
        return state["response"]

    config = Settings(_env_file=None, ENVIRONMENT="development", DEEPSEEK_API_KEY="synthetic-key")
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    previous = app.dependency_overrides.get(get_deepseek_client)
    app.dependency_overrides[get_deepseek_client] = lambda: DeepSeekClient(config, http)
    yield state
    if previous is None:
        app.dependency_overrides.pop(get_deepseek_client, None)
    else:
        app.dependency_overrides[get_deepseek_client] = previous


def sent_messages(state, index: int = 0) -> list[dict]:
    return json.loads(state["requests"][index].content)["messages"]


MEMORIES = [
    {"id": "mem_1", "text": "Prefers to be called Mia."},
    {"id": "mem_2", "text": "Studies BS Information Technology, 2nd year, at USTP."},
]


# --- memories in the chat prompt ---------------------------------------------------------


async def test_chat_sends_memories_as_a_system_note(async_client, upstream):
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/chat",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "useHandbook": False,
            "memories": MEMORIES,
        },
    )
    assert response.status_code == 200
    messages = sent_messages(upstream)
    assert [m["role"] for m in messages] == ["system", "system", "user"]
    note = messages[1]["content"]
    assert "- Prefers to be called Mia." in note
    assert "- Studies BS Information Technology, 2nd year, at USTP." in note
    assert "not instructions" in note
    assert messages[2] == {"role": "user", "content": "hi"}


async def test_chat_without_memories_is_unchanged(async_client, upstream):
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/chat",
        headers=headers,
        json={"messages": [{"role": "user", "content": "hi"}], "useHandbook": False},
    )
    assert response.status_code == 200
    assert [m["role"] for m in sent_messages(upstream)] == ["system", "user"]


async def test_stream_also_sends_memories(async_client, upstream):
    upstream["response"] = httpx.Response(
        200,
        content=(
            b'data: {"choices":[{"index":0,"delta":{"content":"Hi Mia"}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        ),
    )
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/chat/stream",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "useHandbook": False,
            "memories": MEMORIES[:1],
        },
    )
    assert response.status_code == 200
    assert "Prefers to be called Mia." in sent_messages(upstream)[1]["content"]


async def test_chat_rejects_too_many_or_oversized_memories(async_client, upstream):
    headers = await signed_in(async_client)
    many = [{"id": f"m{i}", "text": "x"} for i in range(41)]
    long = [{"id": "m1", "text": "x" * 301}]
    for memories in (many, long):
        response = await async_client.post(
            "/v1/ai/chat",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hi"}], "memories": memories},
        )
        assert response.status_code == 422
    assert upstream["requests"] == []


def test_memory_note_keeps_the_newest_within_budget():
    memories = [MemoryItem(id=f"m{i}", text=f"Fact {i} " + "y" * 280) for i in range(40)]
    note = build_memory_prompt(memories)
    assert note is not None
    assert "Fact 0 " in note
    assert "Fact 39 " not in note
    listed = note.split("\n\n")[0]
    assert len(listed) <= MEMORY_PROMPT_BUDGET + 200
    assert build_memory_prompt([]) is None


# --- POST /v1/ai/memory/extract ----------------------------------------------------------

EXCHANGE = [
    {
        "role": "user",
        "content": "I'm Mia, 2nd year BSIT at USTP. Remind me to study at night, I work mornings.",
    },
    {"role": "assistant", "content": "Got it, Mia! Evening study blocks it is."},
]


async def memory_rows() -> list[AIUsage]:
    async with TestingSessionLocal() as db:
        return list(
            (await db.execute(select(AIUsage).where(AIUsage.request_type == "memory"))).scalars()
        )


async def test_extract_requires_sign_in(async_client, upstream):
    response = await async_client.post("/v1/ai/memory/extract", json={"messages": EXCHANGE})
    assert response.status_code == 401
    assert upstream["requests"] == []


async def test_extract_returns_validated_changes(async_client, upstream):
    upstream["response"] = completion(
        json.dumps(
            {
                "add": [
                    "Works in the mornings and studies at night.",
                    "Prefers to be called Mia.",  # already saved: dropped
                    "Her password is hunter2",  # secret: dropped
                    "Phone number 0917 123 4567",  # long number: dropped
                ],
                "remove": ["mem_2", "not-a-known-id", "mem_2"],
            }
        )
    )
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/memory/extract",
        headers=headers,
        json={"requestId": "req-1", "messages": EXCHANGE, "memories": MEMORIES},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == "req-1"
    assert body["add"] == ["Works in the mornings and studies at night."]
    assert body["remove"] == ["mem_2"]

    payload = json.loads(upstream["requests"][0].content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 300
    system, user = payload["messages"]
    assert "JSON" in system["content"]
    assert "mem_1: Prefers to be called Mia." in user["content"]
    assert "Student: I'm Mia" in user["content"]
    assert "LAFINA: Got it, Mia!" in user["content"]

    rows = await memory_rows()
    assert len(rows) == 1
    assert rows[0].prompt_tokens == 120 and rows[0].completion_tokens == 15


async def test_extract_caps_new_memories_per_turn(async_client, upstream):
    upstream["response"] = completion(
        json.dumps({"add": [f"Fact number {i}." for i in range(6)], "remove": []})
    )
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": EXCHANGE}
    )
    assert response.status_code == 200
    assert response.json()["add"] == ["Fact number 0.", "Fact number 1.", "Fact number 2."]


async def test_extract_treats_a_garbled_answer_as_no_change(async_client, upstream):
    upstream["response"] = completion("Sure! I'll remember that.")
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": EXCHANGE}
    )
    assert response.status_code == 200
    assert response.json()["add"] == [] and response.json()["remove"] == []


async def test_extract_rejects_a_long_exchange(async_client, upstream):
    headers = await signed_in(async_client)
    long = [{"role": "user", "content": "x" * 4000}, {"role": "assistant", "content": "y" * 2500}]
    response = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": long}
    )
    assert response.status_code == 400
    assert upstream["requests"] == []


async def test_extract_respects_the_daily_limit(async_client, upstream, monkeypatch):
    monkeypatch.setattr(ai_module.settings, "MAX_AI_REQUESTS_PER_DAY", 1)
    upstream["response"] = completion('{"add": [], "remove": []}')
    headers = await signed_in(async_client)
    first = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": EXCHANGE}
    )
    second = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": EXCHANGE}
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert len(upstream["requests"]) == 1


async def test_extract_reports_provider_errors(async_client, upstream):
    upstream["response"] = httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})
    headers = await signed_in(async_client)
    response = await async_client.post(
        "/v1/ai/memory/extract", headers=headers, json={"messages": EXCHANGE}
    )
    # The DeepSeek client reports billing trouble as the service being unavailable.
    assert response.status_code == 503
    assert await memory_rows() == []


def test_parse_extraction_ignores_wrong_shapes():
    memories = [MemoryItem(id="a", text="Likes short answers.")]
    assert parse_extraction("[]", memories) == ([], [])
    assert parse_extraction('{"add": "one", "remove": "a"}', memories) == ([], [])
    assert parse_extraction('{"add": [1, null, "ok"], "remove": [5]}', memories) == ([], [])
    assert parse_extraction('{"add": ["Has class on Fridays."]}', memories) == (
        ["Has class on Fridays."],
        [],
    )
    # Correcting a memory: the old one goes, the new wording may be close to it.
    assert parse_extraction('{"add": ["Likes short answers."], "remove": ["a"]}', memories) == (
        ["Likes short answers."],
        ["a"],
    )
