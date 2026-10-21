"""Turning a document into a summary worth revising from.

Where flashcards cut the material into questions, this keeps its shape: what
the document is about, the points under each heading, and the terms a student
has to know. The model returns one JSON object per chunk and they are merged,
so a long document still reads as one set of notes rather than five.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.services.json_reply import clean_text, json_candidates

logger = logging.getLogger("lafina.study_notes")

MAX_TITLE_CHARS = 120
MAX_OVERVIEW_CHARS = 900
MAX_HEADING_CHARS = 120
MAX_POINT_CHARS = 400
MAX_TERM_CHARS = 80
MAX_MEANING_CHARS = 400
MAX_SECTIONS = 14
MAX_POINTS_PER_SECTION = 10
MAX_TERMS = 30

STUDY_NOTES_SYSTEM_PROMPT = (
    "You are a study-notes generator that outputs only raw JSON.\n"
    "Summarise the supplied course material into revision notes a student can read "
    "instead of the original.\n"
    "Reply with a single JSON object with exactly these keys:\n"
    '  "title": a short name for the material;\n'
    '  "overview": two to four sentences saying what it covers and why it matters;\n'
    '  "sections": an array of objects, each with "heading" and "points" '
    "(an array of short, self-contained sentences);\n"
    '  "keyTerms": an array of objects, each with "term" and "meaning".\n'
    "Rules:\n"
    "1. No markdown, no code fences, no commentary, nothing before or after the object.\n"
    "2. Follow the order of the material, and keep its own headings where it has them.\n"
    "3. Every point must stand on its own: never write 'as shown above', 'this slide', "
    "or refer to a page or figure number.\n"
    "4. Explain, do not merely list: a point should teach the idea, not name it.\n"
    "5. Skip administrative material such as title pages, timetables and grading policies.\n"
    "6. Use only what the material says. Do not add facts of your own.\n"
    '7. If there is nothing worth summarising, reply with {"title": "", "overview": "", '
    '"sections": [], "keyTerms": []}.'
)


@dataclass
class NoteSection:
    heading: str
    points: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"heading": self.heading, "points": list(self.points)}


@dataclass
class KeyTerm:
    term: str
    meaning: str

    def as_dict(self) -> dict[str, str]:
        return {"term": self.term, "meaning": self.meaning}


@dataclass
class StudySummary:
    title: str = ""
    overview: str = ""
    sections: list[NoteSection] = field(default_factory=list)
    key_terms: list[KeyTerm] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.overview and not self.sections and not self.key_terms

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "overview": self.overview,
            "sections": [section.as_dict() for section in self.sections],
            "keyTerms": [term.as_dict() for term in self.key_terms],
        }


class StudyNotesParseError(Exception):
    """The reply held no JSON object that could be read as notes."""


def build_user_prompt(chunk: str, source_hint: str = "", part: str = "") -> str:
    """Wraps one chunk of material in its instruction."""
    heading = f"Source: {source_hint}\n" if source_hint else ""
    section = f"This is {part} of the material.\n" if part else ""
    return (
        f"{heading}{section}Summarise the course material below into revision notes. "
        "Return only the JSON object.\n\n"
        "--- MATERIAL START ---\n"
        f"{chunk}\n"
        "--- MATERIAL END ---"
    )


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _first_string(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        for candidate in (key, key.upper(), key.capitalize()):
            value = entry.get(candidate)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def parse_study_notes_json(raw: str) -> StudySummary:
    """Reads a model reply into a summary, forgiving the usual wrappers."""
    if not raw or not raw.strip():
        raise StudyNotesParseError("The model returned an empty reply.")

    data = None
    for candidate in json_candidates(raw):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if data is None:
        raise StudyNotesParseError("The model reply was not valid JSON.")

    # A reply that is only the array of sections is still usable.
    if isinstance(data, list):
        data = {"sections": data}
    if not isinstance(data, dict):
        raise StudyNotesParseError("The model reply was not a set of notes.")

    summary = StudySummary(
        title=clean_text(_first_string(data, ("title", "name")), MAX_TITLE_CHARS),
        overview=clean_text(
            _first_string(data, ("overview", "summary", "abstract")), MAX_OVERVIEW_CHARS
        ),
    )

    for entry in _as_list(data.get("sections") or data.get("topics")):
        if not isinstance(entry, dict):
            # A bare string is a heading with nothing under it; keep it as one.
            text = clean_text(entry, MAX_HEADING_CHARS)
            if text:
                summary.sections.append(NoteSection(heading=text))
            continue
        heading = clean_text(
            _first_string(entry, ("heading", "title", "section", "topic")), MAX_HEADING_CHARS
        )
        points = [
            clean_text(point, MAX_POINT_CHARS)
            for point in _as_list(entry.get("points") or entry.get("bullets") or entry.get("items"))
        ]
        points = [point for point in points if len(point) > 1][:MAX_POINTS_PER_SECTION]
        if heading or points:
            summary.sections.append(NoteSection(heading=heading or "Notes", points=points))

    for entry in _as_list(data.get("keyTerms") or data.get("key_terms") or data.get("glossary")):
        if not isinstance(entry, dict):
            continue
        term = clean_text(_first_string(entry, ("term", "name", "word")), MAX_TERM_CHARS)
        meaning = clean_text(
            _first_string(entry, ("meaning", "definition", "description")), MAX_MEANING_CHARS
        )
        if term and meaning:
            summary.key_terms.append(KeyTerm(term=term, meaning=meaning))

    return summary


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def merge_summaries(parts: list[StudySummary]) -> StudySummary:
    """Joins the notes from every chunk into one set.

    The first part supplies the title and the opening paragraph, because a
    document says what it is about at the start. Sections keep the order they
    were written in; a heading that comes round again — chunks overlap — takes
    the new points rather than being repeated.
    """
    merged = StudySummary()
    section_at: dict[str, NoteSection] = {}
    seen_terms: set[str] = set()
    seen_points: set[str] = set()

    for part in parts:
        if not merged.title and part.title:
            merged.title = part.title
        if not merged.overview and part.overview:
            merged.overview = part.overview

        for section in part.sections:
            key = _key(section.heading)
            target = section_at.get(key)
            if target is None:
                if len(merged.sections) >= MAX_SECTIONS:
                    continue
                target = NoteSection(heading=section.heading)
                section_at[key] = target
                merged.sections.append(target)
            for point in section.points:
                point_key = _key(point)
                if not point_key or point_key in seen_points:
                    continue
                if len(target.points) >= MAX_POINTS_PER_SECTION:
                    break
                seen_points.add(point_key)
                target.points.append(point)

        for term in part.key_terms:
            key = _key(term.term)
            if not key or key in seen_terms or len(merged.key_terms) >= MAX_TERMS:
                continue
            seen_terms.add(key)
            merged.key_terms.append(term)

    # A heading that ended up with nothing under it teaches nobody anything.
    merged.sections = [
        section for section in merged.sections if section.points or section.heading != "Notes"
    ]
    return merged


def to_markdown(summary: StudySummary) -> str:
    """Renders the notes as markdown, for export and for saving into Notes."""
    lines: list[str] = []
    if summary.title:
        lines.append(f"# {summary.title}")
        lines.append("")
    if summary.overview:
        lines.append(summary.overview)
        lines.append("")
    for section in summary.sections:
        lines.append(f"## {section.heading}")
        lines.extend(f"- {point}" for point in section.points)
        lines.append("")
    if summary.key_terms:
        lines.append("## Key terms")
        lines.extend(f"- **{term.term}** — {term.meaning}" for term in summary.key_terms)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
