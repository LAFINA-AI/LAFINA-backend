"""Tests for the on-device USTP sources tool (`search_ustp_sources`, ADR-047)."""

from types import SimpleNamespace

from app.api.v1.ai import CHAT_AGENT_TOOLS
from app.services.voice_conversation import (
    VOICE_STYLE_INSTRUCTION,
    VOICE_SYSTEM_INSTRUCTION,
    approved_voice_tools,
    voice_tool_definitions,
)
from scripts.provision_voice_agent import agent_config


def test_search_ustp_sources_is_in_voice_and_chat_allowlists():
    defs = {t["function"]["name"]: t for t in voice_tool_definitions()}
    assert "search_ustp_sources" in defs
    params = defs["search_ustp_sources"]["function"]["parameters"]
    assert params["required"] == ["query"]
    assert set(params["properties"].keys()) == {"query", "collection"}

    approved = approved_voice_tools(
        [{"type": "function", "function": {"name": "search_ustp_sources"}}]
    )
    assert [t["function"]["name"] for t in approved] == ["search_ustp_sources"]
    assert "search_ustp_sources" in CHAT_AGENT_TOOLS


def test_voice_instructions_require_search_ustp_sources():
    assert "search_ustp_sources" in VOICE_SYSTEM_INSTRUCTION
    assert "search_ustp_sources" in VOICE_STYLE_INSTRUCTION


def test_provision_agent_config_includes_search_ustp_sources():
    settings = SimpleNamespace(
        VOICE_CONVERSATION_MAX_REPLY_TOKENS=220,
        ELEVENLABS_VOICE_ID="test-voice", VOICE_CONVERSATION_MAX_SECONDS=300,
    )
    args = SimpleNamespace(
        name="LAFINA Voice",
        first_message="Hi!",
        language="en",
        tts_model="eleven_turbo_v2_5",
        turn_timeout=7.0, turn_eagerness="normal",
    )
    cfg = agent_config(settings, "https://lafina-backend.vercel.app", "sec-1", args)
    tool_names = [t["name"] for t in cfg["conversation_config"]["agent"]["prompt"]["tools"]]
    assert "search_ustp_sources" in tool_names
