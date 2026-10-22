"""Deciding whether a message needs its own search query.

The failure this module exists to prevent is the most common one in
conversational retrieval: a follow-up like "and the other one?" carries the
reference, not the topic. Searching those words retrieves passages that
merely contain them, and the answer then repeats or refuses (spec §9).

The rule here is deliberately cheap and conservative. Only a message that is
short or points back at something is rewritten; a long, self-contained
question is searched as written, with no extra model call. The alternative
signal in the spec -- "shares almost no content words with the retrieved
documents" -- needs a retrieval first to know what it shares with, which
would make every question pay for the check it was meant to avoid.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

#: Messages at or below this many words are treated as possible follow-ups
#: when there is history. Eight covers "explain in simple manner" and
#: "and what about it in celsius?" while leaving full questions alone.
FOLLOW_UP_MAX_WORDS = 8

#: Words that point back at an earlier turn. A message containing one of
#: these is a follow-up however long it is.
_REFERENCE_WORDS = frozenset(
    {
        "it",
        "its",
        "it's",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "one",
        "ones",
        "other",
        "same",
        "former",
        "latter",
    }
)

_WORD = re.compile(r"[a-z0-9']+")


class FollowUpRewriter(Protocol):
    """Turns a follow-up into a query that can stand on its own.

    Implemented by the model-backed providers; a fake stands behind it in
    tests. A rewriter must return the question unchanged rather than invent
    a topic it cannot support.
    """

    def rewrite_follow_up(
        self, *, question: str, history: list[dict[str, str]]
    ) -> str:
        ...


def looks_like_follow_up(
    question: str, history: Sequence[dict[str, str]]
) -> bool:
    """Whether ``question`` is worth rewriting before it is searched.

    No history means no conversation to resolve against: the first question
    is always searched as written.
    """
    if not history:
        return False

    words = _WORD.findall(question.lower())
    if not words:
        return False

    if len(words) <= FOLLOW_UP_MAX_WORDS:
        return True
    return any(word in _REFERENCE_WORDS for word in words)
