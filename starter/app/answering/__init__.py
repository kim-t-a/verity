"""Answering: turning retrieved chunks into the value the question asked for.

Retrieval decides *what to look at*. This package decides **what the answer is**
-- and the two are scored very differently. A retrieval miss is invisible in the
graded output; a wrong answer is the whole score.

**The shape of the problem.** Every graded answer is a short literal that already
exists as text in the corpus. Nothing is composed, paraphrased or inferred, and
the corpus is fictional, so anything recalled from a model's weights rather than
read from a file is wrong by construction. So there is no generation here: the
task is to *find* the one span that is the value the question wants, and to say
which file it came from.

**Six steps, one per module, in the order a candidate travels.**

1. :mod:`answering.shapes` -- what kind of value does the question want, and
   what does that kind look like in text? ``94`` is a number, ``4.3.2`` is a
   version, ``ORR-FAN-2214-B`` is an identifier.
2. :mod:`answering.supersession` -- is the file this value came from still the
   current revision? A withdrawn datasheet holds a real number that is no longer
   the answer, and it is the one question in the sample set that retrieval alone
   measurably cannot settle.
3. :mod:`answering.candidates` -- a value plus everything needed to judge it:
   the file, the position in the ranking, the surrounding words, whether it
   follows an ``=`` sign.
4. :mod:`answering.extract` -- the lexicographic ordering that picks between
   candidates, with no invented weights.
5. :mod:`answering.refusal` -- is this a question the readable corpus answers at
   all? One sample question is not, because its answer sits inside a file that
   cannot be opened.
6. :mod:`answering.citations` -- which files were *necessary* to produce that
   value, as an exact set. This is a different question from where the value was
   read, and the difference is the whole of Stage 6b: one sample question cannot
   be answered without a file the answer never appears in.

**Why the split is this way.** Each module answers exactly one question and can
be tested without the corpus. That matters more here than elsewhere: the graded
corpus is different and larger, so a rule that only holds on the ten sample
questions is worthless, and the way to notice is to be able to assert a rule's
*mechanism* on synthetic input rather than its output on the corpus.

**The stage's real lesson, worth carrying forward.** Both :mod:`~answering.refusal`
and :mod:`~answering.citations` were built only after the obvious designs had been
measured and *refuted* -- a confidence threshold that scored backwards, a
document-frequency threshold fitted to a single example, a citation rule that
searched only what retrieval had fetched. Each refutation is recorded in the
module that survived it, because the next person to reach for the obvious design
should find out why it does not work without having to re-measure it.
"""

from __future__ import annotations

from answering.candidates import Candidate, candidates_from
from answering.citations import cited_paths
from answering.extract import Extraction, extract
from answering.refusal import refusal
from answering.shapes import AnswerShape, ValueSpan, content_terms, question_shapes, value_spans
from answering.supersession import is_superseded, superseded_paths

__all__ = [
    "AnswerShape",
    "Candidate",
    "Extraction",
    "ValueSpan",
    "candidates_from",
    "cited_paths",
    "content_terms",
    "extract",
    "is_superseded",
    "question_shapes",
    "refusal",
    "superseded_paths",
    "value_spans",
]
