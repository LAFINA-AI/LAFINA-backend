"""Turning document text into a deck of flashcards.

The model is asked for a bare JSON array and told, in as many words, not to
wrap it in anything. It sometimes does anyway — a fenced code block, a
"Here are your flashcards:" line, an object with the array inside it — so the
parser here treats the reply as untrusted text that probably contains JSON
rather than as JSON. A deck is worth salvaging; a 502 over a stray fence is
not.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.json_reply import clean_text, json_candidates

logger = logging.getLogger("lafina.flashcards")

MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 1500
MIN_FIELD_CHARS = 2

# Keys seen in the wild when a model ignores the schema it was given.
_QUESTION_KEYS = ("question", "front", "q", "term", "prompt")
_ANSWER_KEYS = ("answer", "back", "a", "definition", "response")

FLASHCARD_SYSTEM_PROMPT = (
    "You are a study-material generator that outputs only raw JSON.\n"
    "From the supplied course text, extract high-yield, testable knowledge: definitions, "
    "mechanisms, processes, classifications, formulas, causes and effects, and factual "
    "details a student could be examined on.\n"
    "Rules:\n"
    "1. Reply with a JSON array of objects and nothing else.\n"
    '2. Every object has exactly two string keys: "question" and "answer".\n'
    "3. No markdown, no code fences, no headings, no commentary, no explanation, "
    "no text before or after the array.\n"
    "4. Each question must stand on its own: never write 'according to the text', "
    "'in this document', or refer to a page, figure or section number.\n"
    "5. Each answer must be self-contained and factual, one to three sentences.\n"
    "6. Skip administrative material — title pages, author names, timetables, "
    "grading policies, references — and anything that is not worth memorising.\n"
    "7. If the text contains nothing testable, reply with exactly []."
)


@dataclass(frozen=True)
class Flashcard:
    question: str
    answer: str

    def as_dict(self) -> dict[str, str]:
        return {"question": self.question, "answer": self.answer}


class FlashcardParseError(Exception):
    """The reply held no JSON array that could be read as flashcards."""


def build_user_prompt(chunk: str, max_cards: int, source_hint: str = "") -> str:
    """Wraps one chunk of course text in its instruction."""
    heading = f"Source: {source_hint}\n\n" if source_hint else ""
    return (
        f"{heading}Create up to {max_cards} flashcards from the course text below. "
        "Return only the JSON array.\n\n"
        "--- COURSE TEXT START ---\n"
        f"{chunk}\n"
        "--- COURSE TEXT END ---"
    )


def _first_value(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        for candidate in (key, key.upper(), key.capitalize()):
            value = entry.get(candidate)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, (int, float)):
                return str(value)
    return ""


def parse_flashcard_json(raw: str) -> list[Flashcard]:
    """Reads a model reply into flashcards, forgiving everything but nonsense.

    Accepts a bare array, an array inside a code fence, an array with prose
    around it, and the ``{"flashcards": [...]}`` shape that JSON-mode replies
    tend to take.
    """
    if not raw or not raw.strip():
        raise FlashcardParseError("The model returned an empty reply.")

    data = None
    for candidate in json_candidates(raw):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if data is None:
        raise FlashcardParseError("The model reply was not valid JSON.")

    if isinstance(data, dict):
        # An object wrapping the array under some key of its own choosing.
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data]

    if not isinstance(data, list):
        raise FlashcardParseError("The model reply was not a list of flashcards.")

    cards: list[Flashcard] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        question = clean_text(_first_value(entry, _QUESTION_KEYS), MAX_QUESTION_CHARS)
        answer = clean_text(_first_value(entry, _ANSWER_KEYS), MAX_ANSWER_CHARS)
        if len(question) < MIN_FIELD_CHARS or len(answer) < MIN_FIELD_CHARS:
            continue
        cards.append(Flashcard(question=question, answer=answer))
    return cards


def _dedupe_key(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def merge_cards(batches: list[list[Flashcard]], limit: int = 200) -> list[Flashcard]:
    """Joins the decks from every chunk, dropping repeats.

    Chunks overlap on purpose, so the same definition often appears twice; the
    first wording wins and order is preserved, which keeps the deck in the
    order of the document.
    """
    seen: set[str] = set()
    merged: list[Flashcard] = []
    for batch in batches:
        for card in batch:
            key = _dedupe_key(card.question)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(card)
            if len(merged) >= limit:
                return merged
    return merged


# ── Anki export ───────────────────────────────────────────────────────────

ANKI_HEADER_LINES = ("#separator:tab", "#html:true", "#columns:Front\tBack")


def escape_anki_field(value: str) -> str:
    """Makes one field safe for a tab-delimited import.

    A tab would start a new column and a newline a new note, so both have to
    go: the newline becomes a line break Anki renders, which is why the export
    declares ``#html:true``.
    """
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = text.replace("\n", "<br>")
    return re.sub(r"[ ]{2,}", " ", text).strip()


def to_anki_tsv(cards: list[Flashcard], include_header: bool = True) -> str:
    """Renders a deck as the Front/Back TSV Anki imports directly."""
    lines: list[str] = list(ANKI_HEADER_LINES) if include_header else []
    for card in cards:
        front = escape_anki_field(card.question)
        back = escape_anki_field(card.answer)
        if not front or not back:
            continue
        lines.append(f"{front}\t{back}")
    return "\n".join(lines) + "\n"


def write_anki_tsv(cards: list[Flashcard], path: str | Path, include_header: bool = True) -> Path:
    """Writes the deck to disk as UTF-8, which is what Anki expects."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_anki_tsv(cards, include_header), encoding="utf-8")
    return target
