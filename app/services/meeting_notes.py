"""Turning a meeting transcript into structured meeting notes.

The transcript is produced on the user's laptop by Whisper; only its text ever
reaches this server, and from here only the text reaches DeepSeek.

Long meetings are summarised hierarchically. The desktop app cuts the transcript
into sections, each section is turned into partial notes, and the partial notes
are consolidated into the final set. Keeping the intermediate step as
*structured notes* rather than free prose is deliberate: a decision or an
action item stays a decision or an action item on its way up, instead of being
paraphrased into a sentence the final pass has to rediscover.

The prompts say, repeatedly, not to invent. The parser backs that up: an owner
or a deadline the model could only have guessed — "Unassigned", "TBD", "N/A" —
is stored as empty, which is what the transcript actually said.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.services.json_reply import clean_text, json_candidates

logger = logging.getLogger("lafina.meeting_notes")

# A section is a slice of transcript small enough to summarise well; the final
# pass gets everything, so its ceiling is far higher. Both sit comfortably
# inside DeepSeek's context window with room for the reply.
MAX_SECTION_CHARS = 24_000
MAX_FINAL_CHARS = 90_000

MAX_TITLE = 120
MAX_SUMMARY = 2_000
MAX_TOPIC = 160
MAX_DISCUSSION = 1_200
MAX_ITEM = 500
MAX_LIST = 40

_NOTES_SCHEMA = (
    "{\n"
    '  "title": string,\n'
    '  "summary": string,\n'
    '  "key_topics": [{"topic": string, "discussion": string}],\n'
    '  "decisions": [string],\n'
    '  "action_items": [{"task": string, "assignee": string, "deadline": string, "status": "pending"}],\n'
    '  "important_dates": [string],\n'
    '  "issues": [string],\n'
    '  "unresolved_questions": [string],\n'
    '  "key_points": [string]\n'
    "}"
)

_FIDELITY_RULES = (
    "Rules:\n"
    "1. Use only what is in the material. Do not invent names, numbers, dates, owners or outcomes.\n"
    "2. Keep confirmed decisions, proposed ideas, unresolved issues and action items distinct. "
    "A suggestion nobody agreed to is not a decision.\n"
    "3. For an action item, set \"assignee\" only if the material explicitly names who will do it; "
    "otherwise use an empty string. Never write \"Unassigned\" or guess from context.\n"
    "4. Set \"deadline\" only if a deadline was explicitly stated; otherwise use an empty string. "
    "Do not infer one.\n"
    "5. If a field has nothing, return an empty string or an empty array rather than guessing.\n"
    "6. Transcripts come from speech recognition and contain errors. Do not quote obvious "
    "mis-hearings as fact; leave out what cannot be understood.\n"
    "7. Reply with the JSON object only: no markdown, no code fences, no commentary."
)

MEETING_NOTES_SYSTEM_PROMPT = (
    "You are an AI meeting assistant. Analyze the meeting material and produce accurate, "
    "structured meeting notes.\n"
    "Extract: the meeting title; an executive summary; key discussion topics; decisions made; "
    "action items with the person responsible and the deadline where explicitly stated; "
    "important dates; problems or issues discussed; questions that remain unresolved; and "
    "important points that should be remembered.\n"
    f"Return a single JSON object with exactly this shape:\n{_NOTES_SCHEMA}\n{_FIDELITY_RULES}"
)

SECTION_SYSTEM_PROMPT = (
    "You are an AI meeting assistant working through a long meeting one part at a time. "
    "Extract structured notes from this part only. Another pass will combine the parts, so be "
    "complete about what this part contains — every decision, action item, date, issue and open "
    "question — and do not write an overall conclusion.\n"
    "The \"title\" may be empty for a part. The \"summary\" is a short account of this part.\n"
    f"Return a single JSON object with exactly this shape:\n{_NOTES_SCHEMA}\n{_FIDELITY_RULES}"
)

CONSOLIDATE_SYSTEM_PROMPT = (
    "You are an AI meeting assistant. You are given structured notes that were extracted, in "
    "order, from consecutive parts of one long meeting. Combine them into the final notes for "
    "the whole meeting.\n"
    "Merge duplicates. When a later part changes or reverses something from an earlier part, "
    "the later one stands, and an earlier proposal that was later settled is a decision, not an "
    "open question. Write one title and one executive summary for the whole meeting.\n"
    f"Return a single JSON object with exactly this shape:\n{_NOTES_SCHEMA}\n{_FIDELITY_RULES}"
)


@dataclass
class ActionItem:
    task: str
    assignee: str = ""
    deadline: str = ""
    status: str = "pending"

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "assignee": self.assignee,
            "deadline": self.deadline,
            "status": self.status,
        }


@dataclass
class KeyTopic:
    topic: str
    discussion: str = ""

    def as_dict(self) -> dict:
        return {"topic": self.topic, "discussion": self.discussion}


@dataclass
class MeetingNotes:
    title: str = ""
    summary: str = ""
    key_topics: list[KeyTopic] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    important_dates: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.summary,
                self.key_topics,
                self.decisions,
                self.action_items,
                self.important_dates,
                self.issues,
                self.unresolved_questions,
                self.key_points,
            ]
        )

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "key_topics": [topic.as_dict() for topic in self.key_topics],
            "decisions": list(self.decisions),
            "action_items": [item.as_dict() for item in self.action_items],
            "important_dates": list(self.important_dates),
            "issues": list(self.issues),
            "unresolved_questions": list(self.unresolved_questions),
            "key_points": list(self.key_points),
        }


class MeetingNotesParseError(Exception):
    """The model's reply held no JSON object that could be read as meeting notes."""


# What a model writes when the honest answer is "not stated".
_PLACEHOLDERS = {
    "",
    "unassigned",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "not specified",
    "not stated",
    "not mentioned",
    "unspecified",
    "tbd",
    "to be determined",
    "no deadline",
    "-",
}


def _placeholder_to_empty(value: str) -> str:
    return "" if value.strip().strip(".").lower() in _PLACEHOLDERS else value


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _string_list(value: object, limit: int = MAX_ITEM) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for entry in _as_list(value):
        if isinstance(entry, dict):
            entry = next((v for v in entry.values() if isinstance(v, str) and v.strip()), "")
        text = clean_text(entry, limit) if entry is not None else ""
        key = text.lower()
        if len(text) > 1 and key not in seen:
            seen.add(key)
            items.append(text)
        if len(items) >= MAX_LIST:
            break
    return items


def _first(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value)
    return ""


def parse_meeting_notes_json(raw: str) -> MeetingNotes:
    """Reads a model reply into meeting notes, forgiving wrappers but not guesses."""
    if not raw or not raw.strip():
        raise MeetingNotesParseError("The model returned an empty reply.")

    data = None
    for candidate in json_candidates(raw):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if not isinstance(data, dict):
        raise MeetingNotesParseError("The model reply was not a JSON object of meeting notes.")

    # A reply nested under a single key ("notes", "meeting_notes") is unwrapped.
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict):
            data = only

    notes = MeetingNotes(
        title=clean_text(_first(data, ("title", "meeting_title", "name")), MAX_TITLE),
        summary=clean_text(_first(data, ("summary", "executive_summary", "overview")), MAX_SUMMARY),
    )

    for entry in _as_list(data.get("key_topics") or data.get("topics") or data.get("key_discussion_topics")):
        if isinstance(entry, str):
            topic = clean_text(entry, MAX_TOPIC)
            if topic:
                notes.key_topics.append(KeyTopic(topic=topic))
            continue
        if not isinstance(entry, dict):
            continue
        topic = clean_text(_first(entry, ("topic", "title", "name", "heading")), MAX_TOPIC)
        discussion = clean_text(_first(entry, ("discussion", "details", "summary", "notes")), MAX_DISCUSSION)
        if topic or discussion:
            notes.key_topics.append(KeyTopic(topic=topic or "Discussion", discussion=discussion))
        if len(notes.key_topics) >= MAX_LIST:
            break

    for entry in _as_list(data.get("action_items") or data.get("actions") or data.get("tasks")):
        if isinstance(entry, str):
            task = clean_text(entry, MAX_ITEM)
            if task:
                notes.action_items.append(ActionItem(task=task))
            continue
        if not isinstance(entry, dict):
            continue
        task = clean_text(_first(entry, ("task", "action", "item", "description", "title")), MAX_ITEM)
        if not task:
            continue
        assignee = _placeholder_to_empty(
            clean_text(_first(entry, ("assignee", "owner", "responsible", "person")), 120)
        )
        deadline = _placeholder_to_empty(
            clean_text(_first(entry, ("deadline", "due", "due_date", "by")), 120)
        )
        status = _first(entry, ("status",)).strip().lower()
        notes.action_items.append(
            ActionItem(
                task=task,
                assignee=assignee,
                deadline=deadline,
                status="done" if status in ("done", "completed", "complete") else "pending",
            )
        )
        if len(notes.action_items) >= MAX_LIST:
            break

    notes.decisions = _string_list(data.get("decisions") or data.get("decisions_made"))
    notes.important_dates = _string_list(data.get("important_dates") or data.get("dates"))
    notes.issues = _string_list(data.get("issues") or data.get("problems"))
    notes.unresolved_questions = _string_list(
        data.get("unresolved_questions") or data.get("open_questions") or data.get("questions")
    )
    notes.key_points = _string_list(
        data.get("key_points") or data.get("important_points") or data.get("remember")
    )
    return notes


def build_section_prompt(content: str, index: int, count: int, material: str) -> str:
    """The instruction for one section of a long meeting."""
    what = "transcript" if material == "transcript" else "notes from earlier parts of the meeting"
    return (
        f"This is part {index + 1} of {count} of the meeting. The material below is {what}.\n"
        "Return only the JSON object.\n\n"
        "--- MATERIAL START ---\n"
        f"{content}\n"
        "--- MATERIAL END ---"
    )


def build_final_prompt(content: str, source: str, title_hint: str = "", recorded_at: str = "") -> str:
    """The instruction for the whole meeting, from its transcript or its section notes."""
    context = []
    if title_hint:
        context.append(f"The user named this meeting: {title_hint}")
    if recorded_at:
        context.append(f"It was recorded on {recorded_at}.")
    lead = ("\n".join(context) + "\n\n") if context else ""
    what = "Meeting transcript" if source == "transcript" else "Notes from each part of the meeting, in order"
    return (
        f"{lead}Return only the JSON object.\n\n"
        f"{what}:\n"
        "--- MATERIAL START ---\n"
        f"{content}\n"
        "--- MATERIAL END ---"
    )


def looks_like_context_overflow(detail: str) -> bool:
    """True when an upstream error says the request was too long for the model."""
    return bool(
        re.search(r"context length|maximum context|too many tokens|token limit|context window", detail or "", re.I)
    )
