"""Meeting notes: the parser that refuses to guess, and the endpoints around it."""

import json

import pytest
from httpx import AsyncClient

from app.clients.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekBillingError,
    DeepSeekConfigError,
    DeepSeekInvalidRequestError,
    DeepSeekRateLimitError,
    DeepSeekTimeoutError,
    DeepSeekTransportError,
)
from app.services.meeting_notes import (
    CONSOLIDATE_SYSTEM_PROMPT,
    MAX_FINAL_CHARS,
    MAX_SECTION_CHARS,
    MEETING_NOTES_SYSTEM_PROMPT,
    SECTION_SYSTEM_PROMPT,
    MeetingNotesParseError,
    looks_like_context_overflow,
    parse_meeting_notes_json,
)
from tests.test_flashcards import _pro_headers


GOOD_NOTES = {
    "title": "Release planning",
    "summary": "The team agreed to ship on Friday after the migration is tested.",
    "key_topics": [{"topic": "Database migration", "discussion": "Needs another test run."}],
    "decisions": ["The deployment will occur on Friday."],
    "action_items": [
        {"task": "Complete database migration", "assignee": "John", "deadline": "Friday", "status": "pending"},
        {"task": "Update the runbook", "assignee": "Unassigned", "deadline": "TBD", "status": "pending"},
    ],
    "important_dates": ["Friday"],
    "issues": ["Database migration may require additional testing."],
    "unresolved_questions": ["Which hosting configuration will be used?"],
    "key_points": ["Staging must be frozen on Thursday."],
}


# ── The parser ────────────────────────────────────────────────────────────


def test_complete_notes_parse_into_every_section():
    notes = parse_meeting_notes_json(json.dumps(GOOD_NOTES))
    assert notes.title == "Release planning"
    assert notes.decisions == ["The deployment will occur on Friday."]
    assert notes.key_topics[0].topic == "Database migration"
    assert notes.action_items[0].assignee == "John"
    assert notes.action_items[0].deadline == "Friday"
    assert notes.important_dates == ["Friday"]
    assert notes.unresolved_questions == ["Which hosting configuration will be used?"]
    assert notes.key_points == ["Staging must be frozen on Thursday."]


def test_an_owner_or_deadline_the_model_could_only_guess_is_stored_as_empty():
    notes = parse_meeting_notes_json(json.dumps(GOOD_NOTES))
    runbook = notes.action_items[1]
    assert runbook.task == "Update the runbook"
    assert runbook.assignee == "", "'Unassigned' is not a person"
    assert runbook.deadline == "", "'TBD' is not a deadline"

    for placeholder in ("N/A", "none", "Not specified", "unknown", "-", "No deadline."):
        parsed = parse_meeting_notes_json(
            json.dumps({"action_items": [{"task": "Task", "assignee": placeholder, "deadline": placeholder}]})
        )
        assert parsed.action_items[0].assignee == ""
        assert parsed.action_items[0].deadline == ""


def test_missing_fields_are_empty_rather_than_invented():
    notes = parse_meeting_notes_json('{"summary": "A short call about lunch."}')
    assert notes.title == ""
    assert notes.decisions == []
    assert notes.action_items == []
    assert notes.is_empty is False

    assert parse_meeting_notes_json("{}").is_empty is True


def test_the_usual_wrappers_and_other_key_names_are_read():
    fenced = "```json\n" + json.dumps(GOOD_NOTES) + "\n```"
    assert parse_meeting_notes_json(fenced).title == "Release planning"

    chatty = "Here are the notes:\n" + json.dumps(GOOD_NOTES) + "\nLet me know!"
    assert parse_meeting_notes_json(chatty).summary.startswith("The team agreed")

    nested = json.dumps({"meeting_notes": GOOD_NOTES})
    assert parse_meeting_notes_json(nested).decisions == ["The deployment will occur on Friday."]

    other = parse_meeting_notes_json(
        json.dumps(
            {
                "meeting_title": "Sync",
                "executive_summary": "Summary.",
                "topics": ["Budget"],
                "decisions_made": ["Approve"],
                "tasks": [{"description": "Send invoice", "owner": "Ana", "due": "Monday", "status": "completed"}],
                "open_questions": ["Who signs?"],
                "problems": ["Late invoices"],
            }
        )
    )
    assert other.title == "Sync"
    assert other.key_topics[0].topic == "Budget"
    assert other.action_items[0].assignee == "Ana"
    assert other.action_items[0].status == "done"
    assert other.unresolved_questions == ["Who signs?"]
    assert other.issues == ["Late invoices"]


def test_repeats_and_junk_in_lists_are_dropped():
    notes = parse_meeting_notes_json(
        json.dumps({"decisions": ["Ship Friday", "ship friday", "", None, 42, "Freeze staging"]})
    )
    assert notes.decisions == ["Ship Friday", "42", "Freeze staging"]


def test_a_reply_with_no_notes_in_it_is_an_error_not_a_guess():
    for bad in ("", "   ", "I could not find a meeting.", "[1, 2, 3]"):
        with pytest.raises(MeetingNotesParseError):
            parse_meeting_notes_json(bad)


def test_every_prompt_carries_the_rules_against_inventing():
    for prompt in (MEETING_NOTES_SYSTEM_PROMPT, SECTION_SYSTEM_PROMPT, CONSOLIDATE_SYSTEM_PROMPT):
        assert "Do not invent" in prompt
        assert "only if the material explicitly names" in prompt
        assert "Do not infer one" in prompt
        assert "empty string or an empty array rather than guessing" in prompt
        assert "A suggestion nobody agreed to is not a decision" in prompt


def test_a_context_overflow_is_recognised_from_the_provider_message():
    assert looks_like_context_overflow("This model's maximum context length is 65536 tokens")
    assert looks_like_context_overflow("Request exceeds token limit")
    assert not looks_like_context_overflow("Invalid parameter: temperature")


# ── The endpoints ─────────────────────────────────────────────────────────


class ScriptedDeepSeek:
    """Plays a fixed sequence of outcomes: a reply string, or an exception to raise."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def json_completion(self, messages, user_id, request_id="", model=None, max_tokens=4096):
        self.calls.append({"messages": messages, "request_id": request_id})
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


@pytest.fixture
def no_retry_delay(monkeypatch):
    from app.api.v1 import meeting_notes

    monkeypatch.setattr(meeting_notes, "RETRY_DELAY_SECONDS", 0)


def _override(fake):
    from app.api.v1.ai import get_deepseek_client
    from app.main import app

    app.dependency_overrides[get_deepseek_client] = lambda: fake
    return lambda: app.dependency_overrides.pop(get_deepseek_client, None)


TRANSCRIPT = (
    "[00:00] John: Let's plan the release. [00:12] Maria: The migration still needs a test run. "
    "[00:31] John: Then we deploy Friday. I'll finish the migration by then."
)


@pytest.mark.asyncio
async def test_notes_are_generated_from_the_transcript(async_client: AsyncClient, no_retry_delay):
    headers = await _pro_headers(async_client, "mn_pro@ustp.edu.ph")
    fake = ScriptedDeepSeek(json.dumps(GOOD_NOTES))
    restore = _override(fake)
    try:
        res = await async_client.post(
            "/v1/ai/meeting-notes/generate",
            headers=headers,
            json={"content": TRANSCRIPT, "source": "transcript", "titleHint": "Release", "recordedAt": "2026-09-13"},
        )
        assert res.status_code == 200, res.text
        notes = res.json()["notes"]
        assert notes["decisions"] == ["The deployment will occur on Friday."]
        assert notes["action_items"][1]["assignee"] == "", "a guessed owner never reaches the app"

        sent = fake.calls[0]["messages"]
        assert sent[0]["content"] == MEETING_NOTES_SYSTEM_PROMPT
        assert "Then we deploy Friday" in sent[1]["content"], "the transcript text is what is sent"
        assert "2026-09-13" in sent[1]["content"]

        # Consolidating section notes uses the consolidation instruction.
        fake.calls.clear()
        res = await async_client.post(
            "/v1/ai/meeting-notes/generate",
            headers=headers,
            json={"content": json.dumps([GOOD_NOTES, GOOD_NOTES]), "source": "sections"},
        )
        assert res.status_code == 200
        assert fake.calls[0]["messages"][0]["content"] == CONSOLIDATE_SYSTEM_PROMPT
    finally:
        restore()


@pytest.mark.asyncio
async def test_a_section_of_a_long_meeting_becomes_partial_notes(async_client: AsyncClient, no_retry_delay):
    headers = await _pro_headers(async_client, "mn_section@ustp.edu.ph")
    fake = ScriptedDeepSeek(json.dumps({**GOOD_NOTES, "title": ""}))
    restore = _override(fake)
    try:
        res = await async_client.post(
            "/v1/ai/meeting-notes/section",
            headers=headers,
            json={"content": TRANSCRIPT, "index": 2, "count": 5, "material": "transcript"},
        )
        assert res.status_code == 200, res.text
        assert fake.calls[0]["messages"][0]["content"] == SECTION_SYSTEM_PROMPT
        assert "part 3 of 5" in fake.calls[0]["messages"][1]["content"]

        bad = await async_client.post(
            "/v1/ai/meeting-notes/section",
            headers=headers,
            json={"content": TRANSCRIPT, "index": 5, "count": 5},
        )
        assert bad.status_code == 400
    finally:
        restore()


@pytest.mark.asyncio
async def test_meeting_notes_need_a_pro_plan_but_say_the_transcript_is_safe(async_client: AsyncClient):
    anon = await async_client.post("/v1/ai/meeting-notes/generate", json={"content": TRANSCRIPT})
    assert anon.status_code == 401

    password = "super-strong-lafina-passphrase-2026"
    reg = await async_client.post("/v1/auth/register", json={"email": "mn_free@ustp.edu.ph", "password": password})
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
    assert res.status_code == 403
    assert "transcript stay available" in res.json()["detail"]


@pytest.mark.asyncio
async def test_material_too_long_for_one_request_is_refused_before_the_model_sees_it(async_client: AsyncClient):
    headers = await _pro_headers(async_client, "mn_long@ustp.edu.ph")
    fake = ScriptedDeepSeek(json.dumps(GOOD_NOTES))
    restore = _override(fake)
    try:
        section = await async_client.post(
            "/v1/ai/meeting-notes/section",
            headers=headers,
            json={"content": "x" * (MAX_SECTION_CHARS + 1), "index": 0, "count": 1},
        )
        assert section.status_code == 413
        final = await async_client.post(
            "/v1/ai/meeting-notes/generate",
            headers=headers,
            json={"content": "x" * (MAX_FINAL_CHARS + 1)},
        )
        assert final.status_code == 413
        assert "sections" in final.json()["detail"]
        assert fake.calls == [], "nothing oversized reached DeepSeek, and nothing was cut short"
    finally:
        restore()


@pytest.mark.asyncio
async def test_each_deepseek_failure_gets_its_own_status_and_explanation(async_client: AsyncClient, no_retry_delay):
    headers = await _pro_headers(async_client, "mn_errors@ustp.edu.ph")
    cases = [
        (DeepSeekConfigError(), 503, "not configured"),
        (DeepSeekAuthenticationError(), 503, "API key is missing or invalid"),
        # What DeepSeek actually returned during live testing: 402 Insufficient Balance.
        (DeepSeekBillingError(), 503, "out of credit"),
        (DeepSeekRateLimitError(), 429, "Wait a minute"),
        (DeepSeekTimeoutError(), 504, "took too long"),
        (DeepSeekTransportError(), 503, "unavailable right now"),
        (
            DeepSeekInvalidRequestError("Invalid request: This model's maximum context length is 65536 tokens"),
            413,
            "too long",
        ),
    ]
    for error, expected_status, phrase in cases:
        fake = ScriptedDeepSeek(error)
        restore = _override(fake)
        try:
            res = await async_client.post(
                "/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT}
            )
            assert res.status_code == expected_status, f"{type(error).__name__}: {res.text}"
            assert phrase in res.json()["detail"], f"{type(error).__name__}: {res.json()['detail']}"
        finally:
            restore()


@pytest.mark.asyncio
async def test_a_momentary_failure_is_retried_once_and_a_limit_is_not(async_client: AsyncClient, no_retry_delay):
    headers = await _pro_headers(async_client, "mn_retry@ustp.edu.ph")

    # A timeout, then an answer: the person never sees the timeout.
    fake = ScriptedDeepSeek(DeepSeekTimeoutError(), json.dumps(GOOD_NOTES))
    restore = _override(fake)
    try:
        res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
        assert res.status_code == 200
        assert len(fake.calls) == 2
    finally:
        restore()

    # An unreadable reply, then a good one.
    fake = ScriptedDeepSeek("Sorry, here you go: {not json", json.dumps(GOOD_NOTES))
    restore = _override(fake)
    try:
        res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
        assert res.status_code == 200
    finally:
        restore()

    # Unreadable twice is reported, not papered over with a made-up summary.
    fake = ScriptedDeepSeek("no json here", "still none")
    restore = _override(fake)
    try:
        res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
        assert res.status_code == 502
        assert "could not be read" in res.json()["detail"]
    finally:
        restore()

    # A rate limit is handed straight back for the app to back off from.
    fake = ScriptedDeepSeek(DeepSeekRateLimitError(), json.dumps(GOOD_NOTES))
    restore = _override(fake)
    try:
        res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
        assert res.status_code == 429
        assert len(fake.calls) == 1, "a limited API is not hit again straight away"
    finally:
        restore()


@pytest.mark.asyncio
async def test_notes_with_nothing_in_them_are_not_passed_off_as_a_result(async_client: AsyncClient, no_retry_delay):
    headers = await _pro_headers(async_client, "mn_empty@ustp.edu.ph")
    restore = _override(ScriptedDeepSeek("{}"))
    try:
        res = await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": "..."})
        assert res.status_code == 422
        assert "Transcript tab" in res.json()["detail"]
    finally:
        restore()


@pytest.mark.asyncio
async def test_usage_is_recorded_so_quotas_count_real_generations(async_client: AsyncClient, no_retry_delay):
    from sqlalchemy import func, select

    from app.models.ai_usage import AIUsage
    from tests.conftest import TestingSessionLocal

    headers = await _pro_headers(async_client, "mn_usage@ustp.edu.ph")
    restore = _override(ScriptedDeepSeek(json.dumps(GOOD_NOTES)))
    try:
        await async_client.post(
            "/v1/ai/meeting-notes/section", headers=headers, json={"content": TRANSCRIPT, "index": 0, "count": 2}
        )
        await async_client.post("/v1/ai/meeting-notes/generate", headers=headers, json={"content": TRANSCRIPT})
    finally:
        restore()

    async with TestingSessionLocal() as db:
        kinds = dict(
            (await db.execute(select(AIUsage.request_type, func.count()).group_by(AIUsage.request_type))).all()
        )
    assert kinds.get("meeting_notes_section") == 1
    assert kinds.get("meeting_notes") == 1
