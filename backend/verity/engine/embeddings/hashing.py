"""A deterministic, CPU-only stand-in for a real embedding model.

**This is not a retrieval model and is not a candidate for submission.** It has
no notion of meaning; it is the hashing trick over word tokens, which is a
decades-old lexical technique wearing a vector's clothes. It exists so that the
index, the store, the vector search and the whole CLI can be built and tested
end to end with no model download, no network and no GPU -- and so that the
real backend, when it lands, has something to be compared against.

Two details are load-bearing rather than incidental.

**The hash must be stable across processes.** Python's built-in ``hash()`` is
randomised per interpreter for strings, so a backend using it would produce
different vectors in the ``--index`` process and in each of the ten ``--query``
processes. Nothing would fail: the vectors would simply be unrelated, and
retrieval would return results that look arbitrary. ``blake2b`` is used
instead, and a test runs the embedder in two subprocesses with different
``PYTHONHASHSEED`` values and compares their output.

**Tokenisation is not normalisation.** :func:`normalisation.normalise` is for
comparing whole values -- it deletes separators and *all* whitespace, so running
it over a sentence would fuse it into a single token and destroy exactly the
word boundaries this needs. It is the right tool for "is this the answer?" and
the wrong tool for "which words are here?".
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

#: Word-ish runs. Deliberately loose: this backend only has to be stable and
#: sensible, and a stricter rule would be modelling language, which is the real
#: backend's job. Digits are kept, because values in this corpus are full of
#: them -- ``94``, ``2214``, ``1847``.
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words vectors, hashed into a fixed number of buckets.

    Satisfies :class:`embeddings.EmbeddingProvider` structurally. Sub-linear
    term weighting and L2 normalisation are included so that cosine similarity
    between two of these vectors behaves like a usable lexical score, which is
    what makes the stand-in useful for developing retrieval rather than merely
    for exercising the plumbing.
    """

    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"hashing-bow-v1-d{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str) -> int:
        """Map a token to a bucket, identically in every process.

        ``hashlib`` rather than ``hash()``: see the module docstring. The digest
        is truncated to 8 bytes, which is far more than a 256-way split needs
        and keeps the conversion cheap.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def tokens(self, text: str) -> list[str]:
        """The words this backend sees in ``text``.

        Exposed because retrieval diagnostics need to explain *why* a chunk
        scored as it did, and "these are the terms" is the first thing to look
        at when a score is surprising.
        """
        return _TOKEN.findall(text.lower())

    def embed_one(self, text: str) -> list[float]:
        counts = [0.0] * self._dimension
        for token in self.tokens(text):
            counts[self._bucket(token)] += 1.0

        # Sub-linear term weighting: a word occurring ten times is worth more
        # than one occurring once, but not ten times more. Without this, a long
        # chunk repeating a word would dominate every similarity.
        weighted = [1.0 + math.log(count) if count > 0 else 0.0 for count in counts]

        norm = math.sqrt(sum(value * value for value in weighted))
        if norm == 0.0:
            # Text with no tokens at all. A zero vector is the honest answer;
            # dividing by the norm would raise.
            return weighted
        return [value / norm for value in weighted]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]
