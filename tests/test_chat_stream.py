"""POST /v1/ai/chat/stream: the chat reply piece by piece, with synthetic DeepSeek."""
import json

import httpx
import pytest
from sqlalchemy import select

from app.api.v1.ai import get_deepseek_client
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.main import app
from app.models.ai_usage import AIUsage
from tests.conftest import TestingSessionLocal
from tests.test_online_voices import signed_in


def sse_body(*chunks: object) -> bytes:
    # Strings starting with ":" are SSE comments (keep-alives), sent as-is.
    lines = [
        f"{c}\n\n" if isinstance(c, str) and c.startswith(":")
        else f"data: {json.dumps(c) if not isinstance(c, str) else c}\n\n"
        for c in chunks
    ]
    return "".join(lines).encode()


def delta(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46}}


@pytest.fixture
def upstream():
    """Sets what the fake DeepSeek answers with; records every request."""
    state = {"response": httpx.Response(200, content=sse_body(delta("Hi"), USAGE, "[DONE]")), "requests": []}

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


def events(response: httpx.Response) -> list[dict]:
    return [
        json.loads(block[len("data: "):])
        for block in response.text.split("\n\n")
        if block.startswith("data: ")
    ]


async def usage_rows() -> list[AIUsage]:
    async with TestingSessionLocal() as db:
        return list((await db.execute(select(AIUsage).where(AIUsage.request_type == "chat"))).scalars())


CHAT = {"messages": [{"role": "user", "content": "hi"}], "useHandbook": False}


async def test_reply_arrives_in_pieces_then_done(async_client, upstream):
    upstream["response"] = httpx.Response(
        200,
        content=sse_body(": keep-alive", delta("Hey"), delta(" there,"), delta(" Maria!"), USAGE, "[DONE]"),
    )
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response)
    assert [e["text"] for e in got if e["type"] == "delta"] == ["Hey", " there,", " Maria!"]
    done = got[-1]
    assert done["type"] == "done"
    assert done["provider"] == "deepseek"
    assert done["usedHandbook"] is False
    assert done["sources"] == []
    assert done["usage"]["completion_tokens"] == 6
    payload = json.loads(upstream["requests"][0].content)
    assert payload["stream"] is True
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["messages"][0]["role"] == "system"
    rows = await usage_rows()
    assert len(rows) == 1
    assert (rows[0].prompt_tokens, rows[0].completion_tokens) == (40, 6)


@pytest.mark.parametrize("status,expected", [(402, 503), (429, 429), (500, 503)])
async def test_provider_errors_before_any_text_are_plain_http_errors(async_client, upstream, status, expected):
    upstream["response"] = httpx.Response(status, json={"error": {"message": "private provider body"}})
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)
    assert response.status_code == expected
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert await usage_rows() == []


async def test_empty_reply_is_an_error_not_a_blank_bubble(async_client, upstream):
    upstream["response"] = httpx.Response(200, content=sse_body(USAGE, "[DONE]"))
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)
    assert response.status_code == 502
    assert await usage_rows() == []


async def test_failure_part_way_keeps_text_sent_and_reports_error(async_client, upstream):
    upstream["response"] = httpx.Response(200, content=sse_body(delta("Sure, here"), "{not json"))
    headers = await signed_in(async_client)
    response = await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)
    assert response.status_code == 200
    got = events(response)
    assert got[0] == {"type": "delta", "text": "Sure, here"}
    assert got[-1]["type"] == "error"
    assert got[-1]["status"] == 502
    # The provider was used, so the request still counts toward the quota.
    assert len(await usage_rows()) == 1


async def test_guests_and_disabled_accounts_never_reach_deepseek(async_client, upstream):
    assert (await async_client.post("/v1/ai/chat/stream", json=CHAT)).status_code == 401
    headers = await signed_in(async_client, active=False)
    assert (await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)).status_code == 401
    assert upstream["requests"] == []


async def test_stream_shares_the_chat_quota(async_client, upstream, monkeypatch):
    from app.api.v1 import ai

    monkeypatch.setattr(ai.settings, "MAX_AI_REQUESTS_PER_MIN", 1)
    headers = await signed_in(async_client)
    assert (await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)).status_code == 200
    assert (await async_client.post("/v1/ai/chat/stream", headers=headers, json=CHAT)).status_code == 429
    assert len(upstream["requests"]) == 1
