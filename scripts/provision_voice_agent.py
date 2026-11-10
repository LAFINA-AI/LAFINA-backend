"""Create, update or check the private ElevenAgents agent for live voice conversation.

The agent does speech recognition, turn taking, interruptions and speech.
Its "LLM" is this backend's `/v1/ai/voice/llm/chat/completions`, which answers
with DeepSeek, so LAFINA's own prompt, handbook and limits apply.

Run from the repository root with the private .env (never print it):

    # Show the agent configuration that would be sent (no network, no secrets):
    python -m scripts.provision_voice_agent --server-url https://lafina-backend.vercel.app

    # Read-only: does the configured agent look right?
    python -m scripts.provision_voice_agent --check

    # Create the agent (uploads ELEVENLABS_AGENT_LLM_SECRET to the ElevenLabs
    # secret store, then prints the new agent ID for ELEVENLABS_AGENT_ID):
    python -m scripts.provision_voice_agent --server-url https://lafina-backend.vercel.app --apply

    # Update the existing agent in place (ELEVENLABS_AGENT_ID or --agent-id):
    python -m scripts.provision_voice_agent --server-url https://... --apply --update

The ElevenLabs key needs ElevenAgents (Conversational AI) write access for
--apply and read access for --check; the running server needs ElevenAgents
read access (signed URLs) and the existing Text to Speech / Voices access.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from app.config import Settings
from app.services.voice_conversation import voice_tool_definitions

API = "https://api.elevenlabs.io"
SECRET_NAME = "lafina_voice_llm_secret"
CLIENT_EVENTS = [
    "conversation_initiation_metadata",
    "ping",
    "audio",
    "interruption",
    "user_transcript",
    "tentative_user_transcript",
    "agent_response",
    "agent_response_correction",
    "agent_chat_response_part",
    "agent_response_complete",
    "vad_score",
    "client_error",
    "client_tool_call",
]
PLACEHOLDER_PROMPT = (
    "You are LAFINA, a friendly academic assistant for students. "
    "(LAFINA's server supplies the full instructions for every turn.)"
)


def agent_config(settings: Settings, server_url: str, secret_id: str, args) -> dict:
    return {
        "name": args.name,
        "conversation_config": {
            "agent": {
                "first_message": args.first_message,
                "language": args.language,
                "prompt": {
                    "prompt": PLACEHOLDER_PROMPT,
                    "llm": "custom-llm",
                    "tools": [
                        {
                            "type": "client",
                            "name": tool["function"]["name"],
                            "description": tool["function"]["description"],
                            "parameters": tool["function"]["parameters"],
                            "expects_response": True,
                            "response_timeout_secs": 30,
                        }
                        for tool in voice_tool_definitions()
                    ],
                    "custom_llm": {
                        # ElevenLabs appends /chat/completions.
                        "url": server_url.rstrip("/") + "/v1/ai/voice/llm",
                        "model_id": "lafina-deepseek",
                        "api_key": {"secret_id": secret_id},
                        "request_headers": {
                            "X-Lafina-Voice-Grant": {"variable_name": "lafina_voice_grant"},
                            "X-Lafina-Conversation-Id": {
                                "variable_name": "system__conversation_id"
                            },
                        },
                        "api_type": "chat_completions",
                    },
                    "temperature": 0.7,
                    "max_tokens": settings.VOICE_CONVERSATION_MAX_REPLY_TOKENS,
                },
            },
            "tts": {
                "model_id": args.tts_model,
                "voice_id": settings.ELEVENLABS_VOICE_ID,
                "agent_output_audio_format": "pcm_16000",
            },
            "asr": {"user_input_audio_format": "pcm_16000", "quality": "high"},
            "turn": {
                # Students pause to think: wait longer before taking the turn.
                "turn_eagerness": args.turn_eagerness,
                # Seconds of silence before LAFINA speaks up again (1-30).
                "turn_timeout": args.turn_timeout,
            },
            "conversation": {
                "max_duration_seconds": settings.VOICE_CONVERSATION_MAX_SECONDS,
                "client_events": CLIENT_EVENTS,
            },
        },
        "platform_settings": {
            # Signed URLs only; the agent ID alone cannot start a conversation.
            "auth": {"enable_auth": True},
            "overrides": {
                # The app may choose the voice, and pass the grant; nothing else.
                "conversation_config_override": {"tts": {"voice_id": True}},
                "custom_llm_extra_body": True,
            },
        },
    }


def _key(settings: Settings) -> str:
    key = settings.ELEVENLABS_API_KEY
    if not key or not key.get_secret_value().strip():
        sys.exit("ELEVENLABS_API_KEY is not set in the environment or .env.")
    return key.get_secret_value().strip()


def _fail(response: httpx.Response, action: str) -> None:
    if response.status_code in (401, 403):
        sys.exit(
            f"{action}: the ElevenLabs key lacks ElevenAgents permission (HTTP {response.status_code})."
        )
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    # Provider detail text never contains our key; it can name a bad field.
    sys.exit(
        f"{action} failed: HTTP {response.status_code} {json.dumps(detail)[:400] if detail else ''}"
    )


def check(settings: Settings, agent_id: str, client: httpx.Client) -> int:
    response = client.get(f"{API}/v1/convai/agents/{agent_id}")
    if response.status_code != 200:
        _fail(response, "Reading the agent")
    data = response.json()
    conv = data.get("conversation_config", {})
    prompt = conv.get("agent", {}).get("prompt", {})
    custom = prompt.get("custom_llm") or {}
    platform = data.get("platform_settings", {})
    summary = {
        "llm": prompt.get("llm"),
        "custom_llm_url": custom.get("url"),
        "custom_llm_has_secret": bool((custom.get("api_key") or {}).get("secret_id")),
        "grant_header_mapped": "X-Lafina-Voice-Grant" in (custom.get("request_headers") or {}),
        "enable_auth": (platform.get("auth") or {}).get("enable_auth"),
        "voice_override_allowed": (
            ((platform.get("overrides") or {}).get("conversation_config_override") or {}).get("tts")
            or {}
        ).get("voice_id"),
        "tts_model": conv.get("tts", {}).get("model_id"),
        "default_voice": conv.get("tts", {}).get("voice_id"),
        "output_format": conv.get("tts", {}).get("agent_output_audio_format"),
        "input_format": conv.get("asr", {}).get("user_input_audio_format"),
        "turn_eagerness": conv.get("turn", {}).get("turn_eagerness"),
        "max_duration_seconds": conv.get("conversation", {}).get("max_duration_seconds"),
        "server_max_seconds": settings.VOICE_CONVERSATION_MAX_SECONDS,
    }
    print(json.dumps(summary, indent=2))
    problems = []
    expected_tools = {tool["function"]["name"] for tool in voice_tool_definitions()}
    actual_tools = {
        tool.get("name")
        for tool in (prompt.get("tools") or [])
        if isinstance(tool, dict)
        and tool.get("type") == "client"
        and tool.get("expects_response") is True
    }
    if not expected_tools.issubset(actual_tools):
        problems.append("required client tools with response waiting are missing or not visible")
    if "client_tool_call" not in (conv.get("conversation", {}).get("client_events") or []):
        problems.append("client_tool_call event is not enabled")
    if summary["llm"] != "custom-llm":
        problems.append("agent does not use the custom LLM")
    if not summary["enable_auth"]:
        problems.append(
            "enable_auth is off: anyone with the agent ID could start paid conversations"
        )
    if not summary["grant_header_mapped"]:
        problems.append("X-Lafina-Voice-Grant header is not mapped")
    if summary["output_format"] != "pcm_16000" or summary["input_format"] != "pcm_16000":
        problems.append("the app streams pcm_16000 in and plays pcm_16000 out")
    if summary["max_duration_seconds"] != settings.VOICE_CONVERSATION_MAX_SECONDS:
        problems.append("max_duration_seconds differs from VOICE_CONVERSATION_MAX_SECONDS")
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server-url", help="Public backend URL ElevenLabs will call (https).")
    parser.add_argument("--agent-id", help="Existing agent; defaults to ELEVENLABS_AGENT_ID.")
    parser.add_argument("--apply", action="store_true", help="Actually create or update the agent.")
    parser.add_argument(
        "--update", action="store_true", help="Update the existing agent instead of creating one."
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only check of the existing agent."
    )
    parser.add_argument("--name", default="LAFINA voice conversation")
    parser.add_argument("--language", default="en", help='Agent language, e.g. "en" or "fil".')
    parser.add_argument("--tts-model", default="eleven_v4_turbo")
    parser.add_argument(
        "--turn-eagerness", default="patient", choices=["patient", "normal", "eager"]
    )
    parser.add_argument("--turn-timeout", type=float, default=15.0)
    parser.add_argument("--first-message", default="Hi, I'm listening. What's on your mind?")
    args = parser.parse_args()

    settings = Settings()
    agent_id = (args.agent_id or settings.ELEVENLABS_AGENT_ID or "").strip()

    if args.check:
        if not agent_id:
            sys.exit("No agent to check: set ELEVENLABS_AGENT_ID or pass --agent-id.")
        with httpx.Client(headers={"xi-api-key": _key(settings)}, timeout=20) as client:
            return check(settings, agent_id, client)

    if not args.server_url or not args.server_url.startswith("https://"):
        sys.exit("--server-url must be the public https URL of the backend.")
    if not args.apply:
        print(json.dumps(agent_config(settings, args.server_url, "<secret_id>", args), indent=2))
        print("\nDry run only. Add --apply to create the agent.", file=sys.stderr)
        return 0

    llm_secret = settings.ELEVENLABS_AGENT_LLM_SECRET
    if not llm_secret or len(llm_secret.get_secret_value().strip()) < 24:
        sys.exit(
            "Set ELEVENLABS_AGENT_LLM_SECRET (24+ random characters) in the private .env and in "
            "the Vercel project first; the same value is uploaded to ElevenLabs here."
        )
    if args.update and not agent_id:
        sys.exit("--update needs ELEVENLABS_AGENT_ID or --agent-id.")

    with httpx.Client(headers={"xi-api-key": _key(settings)}, timeout=30) as client:
        created = client.post(
            f"{API}/v1/convai/secrets",
            json={
                "type": "new",
                "name": SECRET_NAME,
                "value": llm_secret.get_secret_value().strip(),
            },
        )
        if created.status_code != 200:
            _fail(created, "Storing the custom-LLM secret")
        secret_id = created.json().get("secret_id")
        config = agent_config(settings, args.server_url, secret_id, args)
        if args.update:
            response = client.patch(f"{API}/v1/convai/agents/{agent_id}", json=config)
            if response.status_code != 200:
                _fail(response, "Updating the agent")
            print(f"Updated agent {agent_id}.")
        else:
            response = client.post(f"{API}/v1/convai/agents/create", json=config)
            if response.status_code != 200:
                _fail(response, "Creating the agent")
            agent_id = response.json().get("agent_id")
            print(
                f"Created agent {agent_id}. Set ELEVENLABS_AGENT_ID={agent_id} in the Vercel project."
            )
        return check(settings, agent_id, client)


if __name__ == "__main__":
    raise SystemExit(main())
