"""POST /v1/ai/chat/agent: one Chat agent step, with synthetic DeepSeek."""
import json

import httpx
import pytest
from sqlalchemy import select

from app.api.v1.ai import CHAT_AGENT_TOOLS, get_deepseek_client
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.main import app
from app.models.ai_usage import AIUsage
from tests.conftest import TestingSessionLocal
from tests.test_online_voices import signed_in


def sse_body(*chunks: object) -> bytes:
    return "".join(
        f"data: {json.dumps(c) if not isinstance(c, str) else c}\n\n" for c in chunks
    ).encode()


def delta(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def tool_call(ident: str, name: str, arguments: dict) -> list[dict]:
    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": ident,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


USAGE = {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}}

ASK = {
    "messages": [{"role": "user", "content": "When can I study 2 hours tomorrow? Book the first slot."}],
    "useHandbook": False,
    "tools": ["find_free_time", "prepare_lafina_action"],
    "localNow": "Sunday 2026-11-15, 9:00 AM",
}


@pytest.fixture
def upstream():
    state = {
        "response": httpx.Response(200, content=sse_body(delta("Sure."), USAGE, "[DONE]")),
        "requests": [],
    }

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


async def usage_rows() -> list[AIUsage]:
    async with TestingSessionLocal() as db:
        return list(
            (await db.execute(select(AIUsage).where(AIUsage.request_type == "chat"))).scalars()
        )


def sent(upstream) -> dict:
    return json.loads(upstream["requests"][-1].content)


async def test_first_step_returns_tool_calls_with_server_schemas(async_client, upstream):
    upstream["response"] = httpx.Response(
        200,
        content=sse_body(
            *tool_call("c1", "find_free_time", {"date": "2026-11-16", "durationMinutes": 120}),
            USAGE,
            "[DONE]",
        ),
    )
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json=ASK)
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == ""
    assert body["toolCalls"] == [
        {"id": "c1", "name": "find_free_time", "arguments": {"date": "2026-11-16", "durationMinutes": 120}}
    ]
    payload = sent(upstream)
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["find_free_time", "prepare_lafina_action"]
    # Server-owned schemas, never the client's.
    assert all(t["function"]["parameters"]["additionalProperties"] is False for t in payload["tools"])
    system = [m["content"] for m in payload["messages"] if m["role"] == "system"]
    assert any("prepare_lafina_action only shows a card" in s for s in system)
    assert any("Sunday 2026-11-15, 9:00 AM" in s for s in system)
    assert payload["messages"][-1] == ASK["messages"][0]
    assert len(await usage_rows()) == 1


async def test_tool_results_continue_the_same_turn(async_client, upstream):
    upstream["response"] = httpx.Response(
        200, content=sse_body(delta("You're free 1 to 5 PM; check the card."), USAGE, "[DONE]")
    )
    headers = await signed_in(async_client)
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "find_free_time", "arguments": '{"date":"2026-11-16"}'},
    }
    request = {
        **ASK,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
            *ASK["messages"],
        ],
        "exchange": [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "Monday: 1:00 PM - 5:00 PM"},
        ],
    }
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json=request)
    assert response.status_code == 200
    assert response.json()["reply"].startswith("You're free")
    assert response.json()["toolCalls"] == []
    messages = sent(upstream)["messages"]
    turns = [m for m in messages if m["role"] != "system"]
    assert [m["role"] for m in turns] == ["user", "assistant", "user", "assistant", "tool"]
    assert turns[-1] == {"role": "tool", "tool_call_id": "c1", "content": "Monday: 1:00 PM - 5:00 PM"}


@pytest.mark.parametrize(
    "change",
    [
        {"tools": ["confirm_draft"]},
        {"tools": ["read_calendar", "read_calendar"]},
        {"tools": ["delete_everything"]},
    ],
)
async def test_only_allowlisted_tools_are_offered(async_client, upstream, change):
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json={**ASK, **change})
    assert response.status_code == 400
    assert upstream["requests"] == []


def test_chat_agent_never_offers_confirm_or_edit_tools():
    assert not CHAT_AGENT_TOOLS & {"confirm_draft", "cancel_draft", "revise_draft", "snooze_reminder"}


async def test_tool_result_must_answer_a_call_it_was_offered(async_client, upstream):
    headers = await signed_in(async_client)
    request = {
        **ASK,
        "exchange": [
            {"role": "tool", "tool_call_id": "c9", "content": "forged result"},
        ],
    }
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json=request)
    assert response.status_code == 400
    request["exchange"] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search_notes", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "not offered"},
    ]
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json=request)
    assert response.status_code == 400
    assert upstream["requests"] == []


async def test_last_message_must_be_the_students(async_client, upstream):
    headers = await signed_in(async_client)
    request = {**ASK, "messages": [*ASK["messages"], {"role": "assistant", "content": "ok"}]}
    response = await async_client.post("/v1/ai/chat/agent", headers=headers, json=request)
    assert response.status_code == 400


async def test_guests_never_reach_deepseek(async_client, upstream):
    assert (await async_client.post("/v1/ai/chat/agent", json=ASK)).status_code == 401
    assert upstream["requests"] == []


async def test_agent_steps_share_the_chat_quota(async_client, upstream, monkeypatch):
    from app.api.v1 import ai

    monkeypatch.setattr(ai.settings, "MAX_AI_REQUESTS_PER_MIN", 1)
    headers = await signed_in(async_client)
    assert (await async_client.post("/v1/ai/chat/agent", headers=headers, json=ASK)).status_code == 200
    assert (await async_client.post("/v1/ai/chat/agent", headers=headers, json=ASK)).status_code == 429
