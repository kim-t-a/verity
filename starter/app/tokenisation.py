"""Turning text into the terms retrieval matches on.

**This is not normalisation, and the difference is the whole point of the
module.** :func:`normalisation.normalise` deletes *all* whitespace, so running it
over a sentence fuses the sentence into one token. It answers "is this the
answer?" -- a comparison between two whole values. Retrieval needs "which words
are here?", which is a different question with a different tool.

**But the values in this corpus are spelled with separators.** The answers are
things like ``4.3.2``, ``ORR-FAN-2214-B``, ``94``, ``Q3 FY27``, ``REV-C2``. A
tokenizer that splits on every non-alphanumeric character turns ``4.3.2`` into
``4``, ``3``, ``2`` -- three extremely common terms, none of which identifies the
value. A query for the firmware version would then be matching noise, and would
retrieve the whole corpus about equally.

So a tokenizer has to satisfy two pulls at once:

* ``4.3.2`` must survive as **one** term, or the value cannot be matched.
* ``field-replaceable`` must be reachable from a query for ``field replaceable``,
  because the corpus writes it *both* ways -- ``support/rma_parts.xlsx`` says
  "Fan assembly, dual-rotor, field replaceable" while a question is far more
  likely to hyphenate it. Fusing on the hyphen alone would make those two
  spellings mutually invisible.

The rule below satisfies both, and the tie-breaker is **digits**:

1. Split the text into *runs* of alphanumerics joined by separators --
   ``4.3.2``, ``orr-fan-2214-b``, ``field-replaceable``, ``94``, ``r2.pdf``.
2. Emit each separator-delimited **part** of two or more characters. This is
   what makes ``field-replaceable`` and ``field replaceable`` meet, and it is
   why a query for ``r2`` still finds ``r2.pdf``.
3. **If the run contains a digit**, also emit the run with its separators
   removed -- ``4.3.2`` -> ``432``, ``ORR-FAN-2214-B`` -> ``ORRFAN2214B``. A
   value spelled with separators is matched as the value.

The digit test is what keeps the two halves apart. It fires on ``4.3.2`` and on
``ORR-FAN-2214-B``, where the joined form is the identifier someone would quote
as an answer, and stays silent on ``field-replaceable``, where joining would only
manufacture a word that neither the corpus nor the question actually uses.

Note the separator set here is deliberately the same one
:mod:`normalisation` deletes, so a run's joined form is exactly that run's
normalised spelling -- one spelling of a value across the whole system.
"""

from __future__ import annotations

import re

from config import MIN_TOKEN_LENGTH

#: A run of alphanumerics joined by separators. ``[^\W_]`` is "a word character
#: that is not an underscore", which under Python's default Unicode matching
#: means letters and digits in any script -- so ``naïve`` stays one token
#: instead of splitting at the ``ï``. The pattern requires a run to *begin and
#: end* with an alphanumeric, which is what drops the full stop in ``94.``
#: without any cleanup pass afterwards.
#:
#: The separator class is ``._·-``: the same characters :mod:`normalisation`
#: deletes, so the joined form below matches the normalised spelling exactly.
_RUN = re.compile(r"[^\W_]+(?:[._·-][^\W_]+)*")

#: The separator-delimited pieces of a run -- ``orr``, ``fan``, ``2214``, ``b``.
_PART = re.compile(r"[^\W_]+")

_SEPARATOR = re.compile(r"[._·-]")


def tokenise(text: str) -> list[str]:
    """The terms retrieval indexes ``text`` under.

    Order is preserved and repeats are kept -- a term occurring three times is
    emitted three times, because term frequency is what BM25 scores on. Only the
    *within-run* duplicates are collapsed: ``94`` would otherwise arrive once as
    a part and again as a joined form, and counting a single occurrence twice
    would inflate both its frequency and the chunk's length.
    """
    tokens: list[str] = []
    for run in _RUN.findall(text.lower()):
        parts = _PART.findall(run)

        candidates = list(parts)
        if any(character.isdigit() for character in run):
            # Joined by removing the separators, which is exactly what the parts
            # already are once concatenated -- so no second spelling rule is
            # needed, and none can drift from this one.
            candidates.append(_SEPARATOR.sub("", run))

        for token in dict.fromkeys(candidates):
            if len(token) >= MIN_TOKEN_LENGTH:
                tokens.append(token)
    return tokens


def query_terms(query: str) -> tuple[str, ...]:
    """The distinct terms of a query, in first-seen order.

    Distinct because a term repeated in a question carries no more meaning than
    one mention -- BM25 scores a query term once against a chunk, and scoring
    "the thermal throttle incident thermal" as if ``thermal`` mattered twice
    would let repetition in the question distort the ranking.
    """
    return tuple(dict.fromkeys(tokenise(query)))


def joined_form(run: str) -> str:
    """The separator-free spelling of a run -- ``4.3.2`` -> ``432``.

    Exposed because later stages have to recognise a value in a chunk that was
    written with different separators than the question used, and they must do
    it by the same rule the index was built with rather than by a second
    implementation that can drift from it.
    """
    return _SEPARATOR.sub("", run.lower())
