"""Call Lafina's opening line: the app may choose it only when the agent
allows the first_message override and the server says so."""

import json
import sys

import httpx
import pytest

from app.api.v1 import voice
from app.config import Settings
from tests.test_online_voices import signed_in
from tests.test_voice_conversation import world  # noqa: F401  (fixture)


@pytest.mark.usefixtures("world")
async def test_capabilities_do_not_offer_the_opening_line_by_default(async_client):
    headers = await signed_in(async_client)
    caps = (await async_client.get("/v1/ai/voice/capabilities", headers=headers)).json()
    assert caps["enabled"] is True
    assert caps["openingLine"] is False


@pytest.mark.usefixtures("world")
async def test_capabilities_offer_the_opening_line_when_turned_on(async_client, monkeypatch):
    monkeypatch.setattr(voice.settings, "VOICE_OPENING_LINE_OVERRIDE", True)
    headers = await signed_in(async_client)
    caps = (await async_client.get("/v1/ai/voice/capabilities", headers=headers)).json()
    assert caps["openingLine"] is True


@pytest.mark.usefixtures("world")
async def test_an_unconfigured_server_never_offers_the_opening_line(async_client, monkeypatch):
    from app.main import app

    monkeypatch.setattr(voice.settings, "VOICE_OPENING_LINE_OVERRIDE", True)
    monkeypatch.setattr(app.state.elevenlabs_agent_client.settings, "ELEVENLABS_AGENT_ID", None)
    headers = await signed_in(async_client)
    caps = (await async_client.get("/v1/ai/voice/capabilities", headers=headers)).json()
    assert caps["enabled"] is False and caps["openingLine"] is False


def test_new_agents_allow_only_the_voice_and_the_opening_line():
    from scripts.provision_voice_agent import agent_overrides

    assert agent_overrides() == {
        "conversation_config_override": {
            "tts": {"voice_id": True},
            "agent": {"first_message": True},
        },
        "custom_llm_extra_body": True,
    }


def test_allow_opening_line_patches_only_the_override_permissions(monkeypatch, capsys):
    from scripts import provision_voice_agent as provision

    config = Settings(_env_file=None, ENVIRONMENT="development", ELEVENLABS_VOICE_ID="test_voice")
    monkeypatch.setattr(provision, "Settings", lambda: config)
    monkeypatch.setattr(sys, "argv", ["provision", "--allow-opening-line"])
    assert provision.main() == 0
    patch = json.loads(capsys.readouterr().out)
    assert patch == {"platform_settings": {"overrides": provision.agent_overrides()}}


def test_check_flags_the_server_flag_on_while_the_agent_refuses(monkeypatch, capsys):
    from scripts import provision_voice_agent as provision

    config = Settings(_env_file=None, ENVIRONMENT="development", ELEVENLABS_VOICE_ID="test_voice")
    monkeypatch.setattr(provision, "Settings", lambda: config)
    monkeypatch.setattr(
        sys, "argv", ["provision", "--server-url", "https://lafina-backend.vercel.app"]
    )
    assert provision.main() == 0
    agent = json.loads(capsys.readouterr().out)

    def checked(flag_on: bool, agent_allows: bool) -> int:
        config.VOICE_OPENING_LINE_OVERRIDE = flag_on
        overrides = agent["platform_settings"]["overrides"]["conversation_config_override"]
        overrides["agent"] = {"first_message": agent_allows}
        with httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=agent))
        ) as client:
            return provision.check(config, "synthetic-agent", client)

    # The live agent today: override refused, flag off. Calls keep working.
    assert checked(flag_on=False, agent_allows=False) == 0
    capsys.readouterr()
    assert checked(flag_on=True, agent_allows=True) == 0
    capsys.readouterr()
    # The dangerous order: flag on before the agent allows it.
    assert checked(flag_on=True, agent_allows=False) == 1
    assert "every app call would be refused" in capsys.readouterr().out
