"""Reading JSON out of a model reply that was asked for JSON only.

Models mostly comply and occasionally do not: a fenced code block, a line of
introduction, an object wrapped around the array. Everything that has to cope
with that lives here so the flashcard and study-note parsers behave the same
way rather than each forgiving a different set of mistakes.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def strip_fences(text: str) -> str:
    """Removes a wrapping markdown code fence, if there is one."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped.strip()


def extract_json_span(text: str) -> str | None:
    """Finds the outermost JSON array or object in a reply with prose around it.

    Brackets inside string literals are skipped, so a sentence containing "["
    of its own does not end the span early. Whichever bracket opens first is
    the outermost value: trying arrays first would pull the first list out of
    the middle of an object reply.
    """
    openers = sorted(
        ((text.find(opener), opener, closer) for opener, closer in (("[", "]"), ("{", "}"))),
        key=lambda entry: (entry[0] == -1, entry[0]),
    )
    for start, opener, closer in openers:
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def json_candidates(raw: str) -> list[str]:
    """The strings worth trying to parse, best first."""
    stripped = strip_fences(raw)
    candidates = [stripped]
    span = extract_json_span(stripped)
    if span and span != stripped:
        candidates.append(span)
    return candidates


def clean_text(value: object, limit: int) -> str:
    """Collapses a model's whitespace and drops the bold it was told not to use."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text
