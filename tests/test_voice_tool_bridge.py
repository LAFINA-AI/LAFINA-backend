import json

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


def test_oversized_results_are_shortened_not_refused():
    messages = [
        {"role": "user", "content": "what is on my calendar this week?"},
        {"role": "assistant", "tool_calls": [call(ident="a"), call(ident="b")]},
        {"role": "tool", "tool_call_id": "a", "content": '{"title": "Math"}' * 400},
        {"role": "tool", "tool_call_id": "b", "content": "short"},
    ]
    result = voice_tool_continuation(
        messages, allowed_names={"search_notes"}, max_messages=10, max_chars=3000
    )
    assert [m["role"] for m in result] == ["user", "assistant", "tool", "tool"]
    assert result[2]["content"].endswith("(result shortened)")
    assert result[3]["content"] == "short"
    assert len(json.dumps(result, ensure_ascii=False)) <= 3000


def test_contextual_updates_and_spoken_words_do_not_break_the_exchange():
    update = {
        "id": "u1",
        "type": "function",
        "function": {"name": "contextual_update", "arguments": "{}"},
    }
    messages = [
        {"role": "user", "content": "find notes"},
        {"role": "system", "content": "app update"},
        {"role": "assistant", "content": "Let me look."},
        {"role": "assistant", "tool_calls": [call(), update]},
        {"role": "tool", "tool_call_id": "u1", "content": "app update as a tool"},
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
        {"role": "system", "content": "a later app update"},
    ]
    result = voice_tool_continuation(
        messages, allowed_names={"search_notes"}, max_messages=10, max_chars=5000
    )
    assert result == [
        {"role": "user", "content": "find notes"},
        {"role": "assistant", "content": "Let me look.", "tool_calls": [call()]},
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
    ]


def test_unanswered_or_unoffered_calls_are_still_refused():
    base = [{"role": "user", "content": "find notes"}]
    # A call with no result.
    assert (
        voice_tool_continuation(
            base
            + [
                {"role": "assistant", "tool_calls": [call(), call(ident="c2")]},
                {"role": "tool", "tool_call_id": "c1", "content": "found"},
            ],
            allowed_names={"search_notes"},
            max_messages=10,
            max_chars=5000,
        )
        == []
    )
    # Only a call we never offered: nothing of ours to continue.
    assert (
        voice_tool_continuation(
            base
            + [
                {"role": "assistant", "tool_calls": [call(name="delete_everything")]},
                {"role": "tool", "tool_call_id": "c1", "content": "done"},
            ],
            allowed_names={"search_notes"},
            max_messages=10,
            max_chars=5000,
        )
        == []
    )


def test_pending_tool_result_ignores_only_unoffered_calls():
    from app.services.voice_conversation import pending_tool_result

    user = {"role": "user", "content": "find notes"}
    ours = [
        user,
        {"role": "assistant", "tool_calls": [call()]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    assert pending_tool_result(ours, {"search_notes"})
    forged = [user, {"role": "tool", "tool_call_id": "nobody", "content": "x"}]
    assert pending_tool_result(forged, {"search_notes"})
    theirs = [
        user,
        {"role": "assistant", "tool_calls": [call(name="contextual_update", ident="u")]},
        {"role": "tool", "tool_call_id": "u", "content": "x"},
    ]
    assert not pending_tool_result(theirs, {"search_notes"})
    assert not pending_tool_result([user], {"search_notes"})


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
    assert len(tools) == 16
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
    assert read == ["calendar", "todos", "notes", "alarms", "memory"]
    prepare = tools["prepare_lafina_action"]["parameters"]["properties"]
    assert prepare["location"]["type"] == "string"
    assert prepare["category"]["type"] == "string"
    assert "create_alarm" in prepare["action"]["enum"]
    assert all(t["parameters"].get("additionalProperties") is False for t in tools.values())
