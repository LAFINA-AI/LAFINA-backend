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


def iter_json_spans(text: str) -> list[str]:
    """Yields top-level balanced JSON array/object spans found inside prose.

    Unlike stopping at the very first bracket (which fails on prose like
    ``Note [1]: {"title": ...}``), this scans forward so valid JSON after
    bracketed labels is still discovered.
    """
    spans: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        next_arr = text.find("[", pos)
        next_obj = text.find("{", pos)
        if next_arr == -1 and next_obj == -1:
            break
        if next_arr == -1 or (next_obj != -1 and next_obj < next_arr):
            start, opener, closer = next_obj, "{", "}"
        else:
            start, opener, closer = next_arr, "[", "]"
        depth = 0
        in_string = False
        escaped = False
        matched_end = -1
        for index in range(start, length):
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
                    matched_end = index
                    break
        if matched_end != -1:
            spans.append(text[start : matched_end + 1])
            pos = matched_end + 1
        else:
            pos = start + 1
    return spans


def extract_json_span(text: str) -> str | None:
    """Finds the best outermost JSON array or object in a reply with prose around it."""
    import json

    spans = iter_json_spans(text)
    if not spans:
        return None
    # Prefer the first span that parses as a dict or a list containing dicts.
    fallback_valid: str | None = None
    for span in spans:
        try:
            parsed = json.loads(span)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return span
        if isinstance(parsed, list) and (not parsed or any(isinstance(x, dict) for x in parsed)):
            return span
        if fallback_valid is None:
            fallback_valid = span
    return fallback_valid or spans[0]


def json_candidates(raw: str) -> list[str]:
    """The strings worth trying to parse, best first."""
    stripped = strip_fences(raw)
    candidates = [stripped]
    best = extract_json_span(stripped)
    if best and best not in candidates:
        candidates.append(best)
    for span in iter_json_spans(stripped):
        if span not in candidates:
            candidates.append(span)
    return candidates


def clean_text(value: object, limit: int) -> str:
    """Collapses a model's whitespace and drops the bold it was told not to use."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    if len(text) > limit:
        text = text[: max(1, limit - 1)].rstrip() + "…"
    return text
