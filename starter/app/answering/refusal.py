"""Refusal: recognising the question the corpus does not answer.

**The problem.** One sample question asks for the unit price of the TQ-40. The
price is real and it is in the corpus -- inside ``supplier_agreement_ENCRYPTED.pdf``,
which cannot be opened, so it is not a source. The honest submission is an empty
answer with no citations. Anything else means the answer was cracked out of the
file or invented, and both are wrong.

This is not the same as "the answer is hard to find". Every other question is
answerable from the readable corpus; this one is answerable from nothing. The
module's whole job is to tell those apart, and the difficulty is that it must do
so without refusing anything that *is* answerable -- an answer refused is scored
exactly like an answer given wrongly, and refusing a question that could have
been answered spends a point to gain nothing.

**What does not work, measured.** The obvious signals were tried against all ten
sample questions and they are refuted rather than merely doubted:

* *Confidence, or how much of the question the winner explains.* Backwards. The
  refusal question's winner scores 0.60 -- third highest of the ten -- while the
  two-hop question that is answered correctly scores 0.27. Any threshold that
  refuses the one also refuses the other, and six more besides.
* *The question's subject being absent from the readable corpus.* The word
  ``price`` never appears in it, which looks decisive until the same test is run
  on the questions that *are* answered: the two-hop question has four of its
  content words absent (``log``, ``shows``, ``underlying``, ``defect``) and the
  refusal question has one. Paraphrase is normal in this corpus, so absence of a
  word is not absence of a subject.
* *Rarity of the matched terms.* This does separate the ten -- every answered
  question matches at least one term occurring in three or fewer chunks, and the
  refusal question's rarest match occurs in five. But the margin is one chunk,
  it is calibrated on a single negative example, and it is a statistic rather
  than a fact about the text. It was rejected for that reason: on a different and
  larger corpus the document-frequency distribution moves and the threshold would
  silently start refusing answerable questions.

**What does work, and why it is not a threshold.** The refusal question's winner
is ``lead_time_days=7``: the corpus does not merely fail to contain a price, it
*states* that this value is a lead time. The question asked for a price. A value
the corpus explicitly labels as something else cannot be the thing asked for, and
that is a fact about the text rather than a number fitted to a sample:

    Refuse when the winning value is presented as a field's value, and the
    field's name shares no word with the question.

The rule fires on four of the ten sample questions: three answered correctly
(``part_number`` for a question about the part number, ``fixed_in`` for one about
what fixed a ticket, ``DEFAULT_BATCH_TIMEOUT_S`` for one about a batch timeout)
and the refusal question, whose ``lead_time_days`` matches nothing. It is silent
on the other six, including the two image questions -- and silence is the point:
it can only refuse a value the corpus has explicitly labelled as being something
else, so it never refuses on a hunch.

**Where it could be wrong.** The rule assumes field names are written in words a
question would also use. Every one of the fifteen field names in the readable
corpus is -- ``fixed_in``, ``unit_cost_usd``, ``lead_time_days``, none of them
cryptic -- but a corpus using ``pn`` for part number would make a correct answer
unreachable. That is the risk this rule takes, stated plainly because it is the
one that cannot be measured here.

**What the rule deliberately does not do.** It does not refuse a value that is
merely *mentioned* in prose. ``ERROR E7731: thermal throttle engaged`` has a
colon, and a colon in this corpus introduces both a field and an ordinary
sentence, so treating it as a label would read ``ERROR`` as the field name of a
log line. Only ``=`` identifies a field here, and a value with no label gets no
verdict from this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from answering.candidates import Candidate
from answering.shapes import content_terms, value_spans
from models import Chunk
from tokenisation import joined_form

#: What ends a field's name: the row separator, a line break, or a statement end.
#:
#: A field name is read backwards from its ``=`` to the nearest of these. Without
#: the boundary a ``key=value | key=value`` row would read the *first* field's
#: name as the label of the *second* field's value, which is the one mistake this
#: must not make -- the sample's bug database is exactly that shape.
_FIELD_BOUNDARY = re.compile(r"[|\n;]")

#: How far back to look for the ``=`` that assigns this value.
#:
#: The same short lookback the candidate's ``follows_equals`` uses, and for the
#: same reason: a few characters distinguish ``fixed_in=4.3.2`` from a version
#: named inside a sentence, and a longer window would reach across field
#: boundaries.
_ASSIGNMENT_LOOKBACK = 40


def refusal(winner: Candidate, query: str, chunks: Sequence[Chunk]) -> str | None:
    """Why ``query`` should be refused, or ``None`` when it should be answered.

    A reason rather than a boolean, because a refusal is the one outcome that
    discards an answer the system was otherwise ready to submit -- so when it
    happens, the diagnostic has to say which label contradicted the question
    rather than leaving it to be re-derived by hand.

    ``chunks`` is the whole corpus, as in :func:`~answering.citations.cited_paths`,
    and it is used here to find the chunk the winner came from. Every uncertain
    step abstains: if the chunk cannot be found, or holds no span matching the
    value, or the question has no content words to compare against, the answer
    stands. The rule's job is to refuse what it can *prove* is mislabelled, and
    an abstention is always the safer failure.
    """
    asked = _content_forms(query)
    if not asked:
        return None

    labelled = _field_label(winner, chunks)
    if labelled is None:
        return None

    label, chunk = labelled
    named = _content_forms(label)

    # A field name with no discriminating word of its own -- one made entirely of
    # stopwords -- cannot be compared against anything, so it is not evidence of
    # a mismatch. Abstaining is always the safer failure for this rule.
    if not named:
        return None

    if named & asked:
        return None

    return (
        f"the winning value {winner.value!r} is the value of the field {label!r} "
        f"in {chunk.rel_path}, and that field shares no word with the question"
    )


def _content_forms(text: str) -> frozenset[str]:
    """The discriminating words of ``text``, each in the index's joined spelling.

    Both sides of the comparison go through this, so that a question and a field
    name are read the same way -- and so that ``ORR-1847`` in a question matches
    ``orr_1847`` in a field name rather than only its exact spelling.
    """
    return frozenset(joined_form(term) for term in content_terms(text))


def _field_label(
    winner: Candidate, chunks: Sequence[Chunk]
) -> tuple[str, Chunk] | None:
    """The name of the field ``winner``'s value is assigned to, if it is one.

    ``None`` covers every case where the question is not the corpus's to answer
    or not: the chunk is missing, the value is not in it, or the value is written
    into a sentence rather than assigned to a field.
    """
    chunk = next(
        (
            c
            for c in chunks
            if c.rel_path == winner.rel_path and c.location == winner.location
        ),
        None,
    )
    if chunk is None:
        return None

    value = joined_form(winner.value)
    for span in value_spans(chunk.text):
        if joined_form(span.text) != value:
            continue
        label = _label_ending_at(chunk.text, span.start)
        if label is not None:
            return label, chunk

    return None


def _label_ending_at(text: str, position: int) -> str | None:
    """The field name assigned by the ``=`` immediately before ``position``.

    ``None`` when the value is not assigned by an ``=`` at all -- which is the
    common case, because most values in this corpus are written into sentences.
    """
    window = text[max(0, position - _ASSIGNMENT_LOOKBACK) : position]
    if not window.rstrip().endswith("="):
        return None

    head = window.rstrip()[:-1]  # drop the '='
    cut = max((match.end() for match in _FIELD_BOUNDARY.finditer(head)), default=0)
    label = head[cut:].strip()
    return label or None
