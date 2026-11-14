import pytest

from app.services.voice_conversation import approved_voice_tools, voice_tool_continuation


def call(name="search_notes", ident="c1", arguments='{"query":"physics"}'):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_allowlist_uses_server_schema_and_rejects_unknown_names():
    tools = approved_voice_tools(
        [{"type": "function", "function": {"name": "search_notes", "parameters": {"evil": True}}}]
    )
    assert tools[0]["function"]["parameters"]["type"] == "object"
    with pytest.raises(ValueError):
        approved_voice_tools([{"type": "function", "function": {"name": "delete_everything"}}])


def test_continuation_requires_matching_unique_bounded_tool_results():
    messages = [
        {"role": "user", "content": "find notes"},
        {"role": "assistant", "tool_calls": [call()]},
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
    ]
    result = voice_tool_continuation(
        messages, allowed_names={"search_notes"}, max_messages=10, max_chars=5000
    )
    assert result[-1] == {"role": "tool", "tool_call_id": "c1", "content": "found"}
    assert (
        voice_tool_continuation(
            messages[:-1] + [{"role": "tool", "tool_call_id": "wrong", "content": "found"}],
            allowed_names={"search_notes"},
            max_messages=10,
            max_chars=5000,
        )
        == []
    )
    assert (
        voice_tool_continuation(
            messages, allowed_names={"search_notes"}, max_messages=10, max_chars=10
        )
        == []
    )
    messages[1]["tool_calls"].append(call(ident="c1"))
    assert (
        voice_tool_continuation(
            messages, allowed_names={"search_notes"}, max_messages=10, max_chars=5000
        )
        == []
    )


def test_agent_config_declares_waiting_client_tools():
    from types import SimpleNamespace
    from app.config import Settings
    from scripts.provision_voice_agent import agent_config

    settings = Settings(_env_file=None)
    args = SimpleNamespace(
        name="test",
        first_message="Hi",
        language="en",
        tts_model="test",
        turn_eagerness="normal",
        turn_timeout=10,
    )
    config = agent_config(settings, "https://example.invalid", "secret-ref", args)
    tools = config["conversation_config"]["agent"]["prompt"]["tools"]
    assert len(tools) == 12
    assert all(t["type"] == "client" and t["expects_response"] for t in tools)
    assert "client_tool_call" in config["conversation_config"]["conversation"]["client_events"]


def test_spoken_stream_drops_split_and_unclosed_reasoning_tags():
    from app.clients.deepseek import SpeechReasoningFilter

    filt = SpeechReasoningFilter()
    parts = ["Hello <thi", "nk>secret plan</th", "ink> there."]
    assert "".join(filt.feed(part) for part in parts) + filt.feed("", final=True) == "Hello  there."
    filt = SpeechReasoningFilter()
    assert filt.feed("Hi <analysis>private", final=True) == "Hi "
    filt = SpeechReasoningFilter()
    assert filt.feed("Safe <thi", final=True) == "Safe "


def test_contract_matches_current_mobile_tools():
    """The server contract must follow the app's registry (mobile `REGISTERED_TOOLS`)."""
    from app.services.voice_conversation import voice_tool_definitions

    tools = {t["function"]["name"]: t["function"] for t in voice_tool_definitions()}
    read = tools["read_lafina_data"]["parameters"]["properties"]["kind"]["enum"]
    assert read == ["calendar", "todos", "notes", "memory"]
    assert all(t["parameters"].get("additionalProperties") is False for t in tools.values())
