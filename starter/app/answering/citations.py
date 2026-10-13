"""Citations: which files were *necessary*, not which ones were used.

The contract does not ask which files the answer was read out of. It asks which
files were **necessary to produce it**, and it scores the answer as an exact set:
a citation too many costs exactly what a citation too few costs. So the rule
cannot be "cite what looks relevant" -- it has to be the definition itself.

    A file is necessary when removing it would make the answer impossible.

**The single-document case needs no work, and that is worth saying plainly.** If
the winning value was read out of file ``A``, then removing ``A`` removes the
value, so ``A`` is necessary -- and nothing else is, because every other file
leaves a value that is still sitting in ``A``. Seven of the ten sample questions
are this case, and for them the correct citation set is exactly the winner's
file. :func:`cited_paths` returns that without searching anything.

**The multi-hop case is the one that needs a rule.** One sample question is
answerable only by joining two files:

* the production log records the incident and, in its last line, the ticket it
  was logged against -- ``incident logged against ORR-1847``;
* the bug database maps that ticket to the release that closed it --
  ``ticket=ORR-1847 ... fixed_in=4.3.2``.

Neither file alone yields the answer, and the question names neither the ticket
nor the release. The value lives in the database; the database alone does not say
*which row* the question is about. What identifies the row is its **key**, and
what supplies that key is the log. So the log is necessary, and the shape of the
rule is:

    If a value co-occurs with a key in its own chunk, and the question does not
    supply that key, then whichever file *does* supply it was load-bearing.

One wrinkle in that illustration is worth flagging, because the implementation
does not follow the tidy version: extraction actually settles on the database's
*decoy* row, which mentions ``4.3.2`` in prose while its own ``fixed_in`` is
empty, and the ``ORR-1847`` that makes the log necessary sits in the row next to
it. :func:`_keys_around` handles that by gathering keys from every chunk of the
winner's file that spells the value, and the measurement behind that choice is in
its docstring.

**Only identifiers and codes can be keys.** A key is something you look up, and a
bare number is not: it names no particular thing. Measured, that restriction is
load-bearing rather than tidy. The log line holding the sample's error code also
says "clocks reduced to 60%", and ``60`` appears in a comment in the ingest
service and again in the roadmap -- so a rule that let numbers bridge would cite
three files for a question answerable from one, and score zero for it.

**Keys are matched by their joined spelling, never by a fragment.** ``ORR-1847``
tokenises to ``orr``, ``1847`` and ``orr1847``, and matching on any of those
would match the shared ``orr`` in every ``ORR-FAN-*``, ``ORR-HSK-*`` and
``ORR-CBL-*`` part number in the parts spreadsheet. The joined form is the value
as somebody would quote it as an answer, which is the same rule the index is
built on -- :func:`tokenisation.joined_form` exists for exactly this.

**The search is over the whole corpus, not over the retrieved chunks.** That was
measured too, and it is not a matter of taste: the log line naming the ticket is
*not* among the ten chunks the sample question retrieves. A rule that only looked
at what retrieval happened to fetch would find no bridge here and cite one file
of the two that are necessary. Necessity is a fact about the corpus -- the
bridging statement lives wherever it lives, whether or not retrieval found it.

**A file that merely mentions the key is not necessary.** Naming a ticket is not
the same as connecting it to anything, and a technical corpus names tickets in
several places -- a status report, a parts list, a changelog. So a file is cited
only when the passage using the key is *about the question*: the chunk holding
the key must account for at least one of the question's own words. The log line
that says "incident logged against ORR-1847" is talking about the incident the
question describes, which is precisely what makes it the file that identifies the
row. A parts spreadsheet that lists the ticket number and nothing else identifies
nothing, and removing it leaves the answer exactly as reachable as before.

**What this rule does not do yet.** It has no notion of a question that cannot be
answered at all: it always names at least the winner's file, so a question whose
answer is genuinely absent from the readable corpus still gets a citation. That
judgement is :mod:`answering.refusal`, and it is the other half of Stage 6b.
"""

from __future__ import annotations

from collections.abc import Sequence

from answering.candidates import Candidate
from answering.shapes import AnswerShape, content_terms, term_forms, value_spans
from models import Chunk
from tokenisation import joined_form

#: The shapes that can act as a key: things one looks *up*.
#:
#: A number and a version are quantities, not names, and a quantity identifies
#: nothing -- see the module docstring for the measurement that shows a numeric
#: key citing three files where one is necessary.
_KEY_SHAPES = (AnswerShape.IDENTIFIER, AnswerShape.CODE)


def cited_paths(winner: Candidate, query: str, chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Every file necessary for ``winner``'s value, sorted for a stable output.

    ``chunks`` is the whole corpus, not the retrieved subset -- see the module
    docstring for why that distinction changes the answer on the sample set.

    The winner's own file is always included, because the value was read out of
    it and removing it removes the value. Any further file is included only when
    it supplies a key that the winner's chunk uses and the question does not.
    """
    keys = _keys_around(winner, chunks) - _forms(query)

    # No key to look up means no chain to follow, and the search below is the
    # expensive half -- it tokenises every chunk in the corpus. Most questions
    # have no key at all, so the common case skips it entirely.
    if not keys:
        return (winner.rel_path,)

    asked = frozenset(content_terms(query))
    paths = {winner.rel_path}
    for chunk in chunks:
        if chunk.rel_path != winner.rel_path and _uses_a_key(chunk, keys, asked):
            paths.add(chunk.rel_path)

    return tuple(sorted(paths))


def _uses_a_key(chunk: Chunk, keys: frozenset[str], asked: frozenset[str]) -> bool:
    """Whether ``chunk`` uses one of ``keys`` in a passage that is about the question.

    Both halves are required. The key alone would cite any file that happens to
    name the ticket, and the question's words alone would cite any file about the
    incident -- neither identifies the value's row on its own, and it takes a
    passage doing both at once to be the one that connects them.
    """
    return bool(keys & _forms(chunk.text)) and bool(asked & set(content_terms(chunk.text)))


def _keys_around(winner: Candidate, chunks: Sequence[Chunk]) -> frozenset[str]:
    """The keys presented alongside the winning value, anywhere in its file.

    "Alongside" means in a chunk that spells this value out, and the search runs
    over every chunk of the winner's *file* -- not only the chunk the winner came
    from. That is measured rather than incidental: restricting it to the winner's
    own chunk scores 6/10 on the sample citation sets where this scores 7/10, and
    the question it loses is the two-hop one.

    The reason is worth spelling out, because it is not the tidy story. The
    sample's bug database holds two rows naming ``4.3.2`` -- the row that closed
    the incident (``ORR-1847``, reaching the version through its ``fixed_in``
    field) and a decoy row whose *summary* mentions the version while its
    ``fixed_in`` is empty. Extraction picks the decoy, because the decoy's prose
    accounts for three of the question's words and the real row's fields account
    for one. Both rows still present the same value, so both contribute keys, and
    the ``ORR-1847`` that makes the log necessary comes from the row the
    extractor did *not* pick. A rule reading only the winning chunk would miss it.

    Co-occurrence in one chunk is still the requirement, and it carries real
    weight: the datasheet's title line names ``TQ-40`` and its specification line
    holds ``94``, but no single chunk holds both, so a question about the
    temperature grows no keys and cites one file -- where a rule that took keys
    from the whole *file* regardless of the value would bridge the model number
    into three other documents.

    Only the winner's own file is examined, because a key in some other file is
    not the key to this value.
    """
    value = joined_form(winner.value)
    keys: set[str] = set()

    for chunk in chunks:
        if chunk.rel_path != winner.rel_path:
            continue
        spans = value_spans(chunk.text)
        if not any(joined_form(span.text) == value for span in spans):
            continue
        for span in spans:
            key = joined_form(span.text)
            if span.shape in _KEY_SHAPES and key != value:
                keys.add(key)

    return frozenset(keys)


def _forms(text: str) -> frozenset[str]:
    """Shorthand for the shared spelling helper, used all over this module."""
    return term_forms(text)
