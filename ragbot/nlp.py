"""Cleaning, stemming, query expansion, and follow-up rewriting."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a an the is are was were be been being of to and or for in on with how what when where who
    whom why does do did doing as at by from that this these those it its they their them he she
    his her we our you your i me my can could should would about into over after before than then
    also just too very not no nor if so such only own same other each both few more most some any
    all per via within during without across because while onto off up down out have has had
    having will shall may might must many much am
    """.split()
)

# General English expansions. Keys and values are already stemmed.
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "vacation": ("paid", "time"),
    "pto": ("paid", "time"),
    "price": ("cost",),
    "pricing": ("cost",),
    "monthly": ("month",),
    "hire": ("employee",),
    "guarantee": ("warranty",),
}

IRREGULAR = {
    "extend": "extend",
    "extends": "extend",
    "extended": "extend",
    "extending": "extend",
    "extension": "extend",
    "extensions": "extend",
}

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_FOLLOW_UP = re.compile(
    r"^(and|also|what about|how about)\b|\b(it|its|they|their|them|that one|the same)\b",
    re.IGNORECASE,
)


def stem(token: str) -> str:
    """Strip regular plurals, and collapse the extend/extended family."""
    mapped = IRREGULAR.get(token)
    if mapped:
        return mapped
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if any(character.isalpha() for character in raw):
            raw = stem(raw)
            if raw in STOPWORDS or len(raw) < 2:
                continue
        tokens.append(raw)
    return tokens


def analyze_query(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stemmed question terms and the synonym expansions added for search."""
    base = tuple(dict.fromkeys(tokenize(text)))
    expanded = tokenize_query(text)
    extras = tuple(token for token in expanded if token not in base)
    return base, extras


def tokenize_query(text: str) -> list[str]:
    """Tokenize a question and append synonym expansions for retrieval."""
    expanded: list[str] = []
    seen: set[str] = set()
    for token in dict.fromkeys(tokenize(text)):
        if token not in seen:
            expanded.append(token)
            seen.add(token)
        for alternative in EXPANSIONS.get(token, ()):
            if alternative not in seen:
                expanded.append(alternative)
                seen.add(alternative)
    return expanded


def covers(token: str, haystack: set[str]) -> bool:
    if token in haystack:
        return True
    alternatives = EXPANSIONS.get(token, ())
    return bool(alternatives) and all(alternative in haystack for alternative in alternatives)


def overlap_weight(token: str, haystack: set[str], idf: dict[str, float]) -> float:
    if token in haystack:
        return float(idf.get(token, 0.0))
    alternatives = EXPANSIONS.get(token, ())
    if alternatives and all(alternative in haystack for alternative in alternatives):
        return max((float(idf.get(alternative, 0.0)) for alternative in alternatives), default=0.0)
    return 0.0


def lexical_match(
    question: str,
    text: str,
    idf: dict[str, float],
    *,
    min_idf: float,
    min_coverage: float,
    sparse_score: float,
) -> bool:
    """True when enough rare question terms, or their expansions, appear in the passage."""
    if sparse_score <= 0:
        return False
    tokens = list(dict.fromkeys(tokenize(question)))
    if not tokens:
        return False
    haystack = set(tokenize(text))
    distinctive = [token for token in tokens if idf.get(token, 10.0) >= min_idf]
    if not distinctive:
        return False
    matched = [token for token in distinctive if covers(token, haystack)]
    return len(matched) / len(distinctive) >= min_coverage


def clean_prose(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = line.replace("**", "").replace("__", "")
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        lines.append(line.rstrip())
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in re.split(r"\n+", text):
        line = " ".join(line.split())
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def markdown_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        cleaned = clean_prose(text)
        return [("", cleaned)] if cleaned else []
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    preamble = text[: matches[0].start()]
    if preamble.strip():
        sections.append(("", clean_prose(preamble)))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = " ".join(clean_prose(match.group(2)).split())
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = clean_prose(text[match.end() : end])
        if not body:
            continue
        path = " › ".join(item for _, item in stack)
        sections.append((path, body))
    return sections


def standalone_query(question: str, history: list[dict] | None) -> str:
    """Attach the previous user question when this one is a pronoun follow-up."""
    question = " ".join(question.split())
    if not question or not history:
        return question
    previous = ""
    for item in reversed(history):
        if item.get("role") == "user" and item.get("content"):
            previous = " ".join(str(item["content"]).split())
            break
    if not previous or previous.lower() in question.lower():
        return question
    if _FOLLOW_UP.search(question):
        return f"{previous.rstrip('?.!')} — {question}"
    return question
