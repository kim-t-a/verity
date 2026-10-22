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

#: Stems of requests that ask for the *same* answer in different words.
#: Deliberately narrow: "summarise the contract" is a new task, not a
#: restatement, so summar* is not here.
_RESTYLE_STEMS = ("simpl", "rephras", "shorten", "eli5")

#: Whole phrases that mean "say the last answer differently". The
#: "I don't understand" family is here because it is the same request in the
#: user's own words: the answer was not an explanation, and they want one.
_RESTYLE_PHRASES = (
    "in other words",
    "say that again",
    "plain english",
    "plain words",
    "break that down",
    "explain like",
    "not understanding",
    "don't understand",
    "don’t understand",
    "do not understand",
    "didn't understand",
    "didn’t understand",
    "makes no sense",
    "doesn't make sense",
    "doesn’t make sense",
    "not following",
    "lost me",
)


def looks_like_restyle(question: str) -> bool:
    """Whether the message asks for the previous answer, explained differently.

    The distinction matters because a restyle is the one follow-up where the
    model must *not* just re-extract the same sentence: the evidence is the
    same, and the work is the explanation. "Explain in simple manner" and "I
    don't understand" are the same request.
    """
    text = question.lower()
    if any(phrase in text for phrase in _RESTYLE_PHRASES):
        return True
    return any(
        word.startswith(_RESTYLE_STEMS) for word in _WORD.findall(text)
    )


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
