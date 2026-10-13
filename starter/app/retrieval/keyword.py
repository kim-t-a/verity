"""Keyword retrieval, scored with BM25 over two fields.

**Why BM25 and not something simpler.** The obvious lexical scorer is "how many
query words does this chunk contain", and it fails in two ways that this corpus
punishes. It ignores *rarity*: a chunk containing ``the`` scores as well as one
containing ``E7731``, so a question's common words drown its distinctive ones.
And it ignores *length*: a long chunk contains more of everything, so it wins by
being long.

BM25 fixes both, and it does so with two ideas worth knowing by name.

*Inverse document frequency* weights a term by how rare it is. A term in every
chunk carries almost no information; a term in two chunks carries a great deal.
This is what makes ``E7731`` count for far more than ``error``, and it is the
mechanism that lets a question be matched on its one distinctive term.

*Term-frequency saturation with length normalisation* stops repetition from
paying. A term occurring ten times is worth more than one occurrence, but the
curve flattens, so a chunk cannot win by repeating itself; and the score is
divided by the chunk's length relative to average, so a long chunk has to earn
its size. The two constants controlling this are ``BM25_K1`` and ``BM25_B``, and
they are in :mod:`config` with their reasoning.

**Two fields, not one.** A chunk is scored twice: once on its text and once on
its file path, and the second contributes at ``PATH_FIELD_WEIGHT``. The reason is
in :mod:`config`, but the short version is that questions name documents by kind
-- "the production log" -- and a log file's lines never contain the word "log".
Treated as one concatenated document this would be worse than useless, because
paths differ in length and would distort the length statistics that BM25 depends
on; as a separate field it is just the same scorer applied to a second, shorter
document with its own averages.

**What this deliberately does not do.** No stemming, so ``batch`` does not match
``batches``. No synonym expansion. No phrase matching. Every one of those trades
precision for recall, and in this challenge the wrong answer and the missing
answer cost exactly the same -- so a change that finds more chunks but ranks the
right one lower is a loss, not a gain. The mechanism is built plain here and can
be tuned against the sample scorer in Stage 5, where the effect of a change can
actually be measured rather than assumed.

**One thing it knowingly gets wrong.** It has no notion of a superseded
document. ``specs/tq40_datasheet_r1_WITHDRAWN.pdf`` answers question 1 just as
convincingly as the current revision, and both are returned. That is correct for
retrieval -- both chunks genuinely match, and a retriever that quietly dropped
one would be hiding a judgement it is not equipped to make -- and it is why the
answering stage has to demote withdrawn revisions explicitly rather than trusting
the ranking to have done it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from chunkstore import ChunkIndex
from config import BM25_B, BM25_K1, PATH_FIELD_WEIGHT, RETRIEVAL_LIMIT
from models import Chunk
from retrieval import RetrievedChunk, rank
from tokenisation import query_terms, tokenise

#: Marks a matched term that came from the file path rather than the text.
#:
#: A ``path:`` prefix rather than a separate field on the result, because the
#: distinction is exactly what a person debugging a ranking needs to see at a
#: glance: a chunk that scored on ``path:logs`` alone is one that was found by
#: its filename and says nothing about the question, while the same score from
#: content terms means the chunk really is about it.
PATH_TERM_PREFIX = "path:"


@dataclass(frozen=True)
class _Field:
    """One searchable field: term postings, lengths, and a weight.

    A field is a complete little BM25 index over its own documents. Keeping it
    self-contained is what lets the file path be scored with statistics of its
    own instead of borrowing the text's.
    """

    name: str
    weight: float
    postings: dict[str, dict[int, int]]
    lengths: list[int]
    average_length: float

    @classmethod
    def build(cls, name: str, weight: float, documents: Sequence[str]) -> "_Field":
        postings: dict[str, dict[int, int]] = {}
        lengths: list[int] = []
        for position, document in enumerate(documents):
            tokens = tokenise(document)
            lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, {})[position] = frequency
        average = sum(lengths) / len(lengths) if lengths else 0.0
        return cls(
            name=name, weight=weight, postings=postings, lengths=lengths, average_length=average
        )

    @property
    def is_usable(self) -> bool:
        """False for a field with nothing in it, or with no length to divide by."""
        return bool(self.postings) and self.average_length > 0.0

    def _idf(self, term: str, total: int) -> float:
        """Inverse document frequency, in the smoothed BM25 form.

        The ``+0.5`` terms are smoothing, and they are not cosmetic: without
        them a term appearing in *every* chunk would score an idf of exactly
        zero. The ``+1`` inside the logarithm is what keeps a term's weight
        positive rather than letting it go negative as it approaches ubiquity --
        a negative weight would mean a chunk is *penalised* for containing a
        query word, which is never what a ranking should do.
        """
        document_frequency = len(self.postings.get(term, ()))
        return math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    def score_into(
        self, term: str, total: int, scores: dict[int, float], matched: dict[int, list[str]]
    ) -> None:
        """Add this field's contribution for ``term`` to the running totals."""
        postings = self.postings.get(term)
        if not postings or not self.is_usable:
            return

        idf = self._idf(term, total)
        for position, frequency in postings.items():
            length = self.lengths[position]
            # k1 bounds how much repetition can pay; b decides how hard a long
            # chunk is penalised for its length.
            normalisation = 1.0 - BM25_B + BM25_B * (length / self.average_length)
            weight = idf * (frequency * (BM25_K1 + 1.0)) / (frequency + BM25_K1 * normalisation)

            scores[position] = scores.get(position, 0.0) + self.weight * weight
            matched.setdefault(position, []).append(
                term if self.name == "text" else f"{PATH_TERM_PREFIX}{term}"
            )


class KeywordRetriever:
    """BM25 over an index's chunks, scoring text and file path as separate fields.

    Built once per index and searched once per process. Construction is where
    the corpus-wide statistics are computed -- which chunks contain each term,
    and how long each chunk is -- because those are properties of the *corpus*,
    not of the question, and recomputing them per query would be paying
    repeatedly for the same answer.
    """

    def __init__(self, index: ChunkIndex, *, path_weight: float = PATH_FIELD_WEIGHT) -> None:
        self._chunks: tuple[Chunk, ...] = index.chunks

        # An inverted index per field rather than a bag of words per chunk,
        # because the query needs the opposite direction: given a term, which
        # chunks hold it. Scored this way only the chunks containing at least
        # one query term are ever looked at, so a query costs what its matches
        # cost rather than what the corpus costs.
        # A field weighted zero is dropped rather than searched-and-multiplied.
        # Multiplying by zero still registers the chunk as *matched*, so the
        # result would come back ranked last with a score of exactly nothing --
        # a chunk returned for a query it does not match, occupying a slot in
        # the top ten that a real match could have used.
        self._fields: tuple[_Field, ...] = tuple(
            field
            for field in (
                _Field.build("text", 1.0, [chunk.text for chunk in self._chunks]),
                _Field.build("path", path_weight, [chunk.rel_path for chunk in self._chunks]),
            )
            if field.weight > 0.0
        )

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vocabulary_size(self) -> int:
        """Distinct terms indexed across both fields.

        Diagnostic: a collapse towards zero means the tokenizer has stopped
        seeing the corpus, which would show up as every query returning nothing.
        """
        return len({term for field in self._fields for term in field.postings})

    def search(self, query: str, *, limit: int = RETRIEVAL_LIMIT) -> list[RetrievedChunk]:
        """The best ``limit`` chunks for ``query``, best first.

        Returns an empty list when the query has no terms, when the index holds
        no chunks, or when nothing matches. None of those is an error -- the
        last is exactly what a question about the encrypted file looks like --
        and the honest response downstream is to refuse, not to raise.
        """
        terms = query_terms(query)
        if not terms or not self._chunks:
            return []

        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}

        for term in terms:
            for field in self._fields:
                field.score_into(term, len(self._chunks), scores, matched)

        return rank(
            self._chunks,
            (
                (position, score, tuple(sorted(matched[position])))
                for position, score in scores.items()
            ),
        )[:limit]
