"""Text normalisation, hashing and lexical helpers (no heavy dependencies)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines; keep paragraph structure."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = _WHITESPACE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_words(text: str) -> set[str]:
    """Lower-cased tokens minus stopwords and 1-character noise."""
    return {word for word in tokenize(text) if word not in STOPWORDS and len(word) > 2}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(subset: set[str], reference: set[str]) -> float:
    """Fraction of `subset` that also appears in `reference`."""
    if not subset:
        return 0.0
    return len(subset & reference) / len(subset)


def truncate(text: str, limit: int, suffix: str = " …") -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > limit * 0.6 else cut).rstrip() + suffix
