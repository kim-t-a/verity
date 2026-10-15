"""Turning read results into the chunks the index will store.

Chunking is where the retrieval system's recall is decided, before any
retrieval code exists. A value that ends up split across two chunks can never
be matched, and a value buried in the middle of a thousand-word chunk is
matched so weakly that it loses to everything else. Both failures are invisible
at query time -- the search simply returns the wrong files -- so the boundaries
are drawn here, deliberately, and tested.

The shape of the work is set by the readers: they already emit one span per
row, line or paragraph, which is the unit a human uses to talk about a
document ("row 2", "line 42"). This module mostly preserves that, and does two
things to it:

* drops spans that cannot be matched by anything, and
* splits spans too large to be matched well.
"""

from __future__ import annotations

from collections.abc import Iterable

from verity.engine.config import CHUNK_BREAK_LOOKBACK, MAX_CHUNK_CHARS
from verity.engine.models import Chunk, ReadResult, TextSpan


def _split_points(text: str) -> list[str]:
    """Cut ``text`` into pieces no longer than ``MAX_CHUNK_CHARS``.

    Each cut is pulled back to the nearest space within
    ``CHUNK_BREAK_LOOKBACK``, so a piece does not begin or end mid-token. The
    lookback is bounded: on text with no spaces at all -- a base64 blob, a
    minified file -- searching further would scan the whole prefix for a break
    that is not there, so the cut falls where the limit says and the piece is
    simply a hard slice.

    The final piece is whatever is left, including a remainder shorter than the
    lookback. Callers check for content; this function only cuts.
    """
    pieces: list[str] = []
    remaining = text

    while len(remaining) > MAX_CHUNK_CHARS:
        window = remaining[:MAX_CHUNK_CHARS]
        break_at = window.rfind(" ", MAX_CHUNK_CHARS - CHUNK_BREAK_LOOKBACK)
        if break_at <= 0:
            break_at = MAX_CHUNK_CHARS
        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:].lstrip()

    pieces.append(remaining)
    return pieces


def _span_chunks(result: ReadResult, span: TextSpan) -> Iterable[Chunk]:
    """Yield the chunks one span becomes, one per piece after splitting."""
    pieces = _split_points(span.text)
    for index, piece in enumerate(pieces, start=1):
        location = span.location
        if len(pieces) > 1:
            # The location must still identify *which* piece, or a diagnostic
            # pointing at "line 42" would be ambiguous about which part of it
            # held the value.
            location = f"{span.location} part {index} of {len(pieces)}"

        chunk = Chunk(
            rel_path=result.rel_path,
            media_type=result.media_type,
            location=location,
            ordinal=span.ordinal,
            text=piece,
        )
        if chunk.has_searchable_content:
            yield chunk


def chunks_for(result: ReadResult) -> tuple[Chunk, ...]:
    """Every indexable chunk from one file's read result.

    Returns an empty tuple for any file that was not successfully read. That
    gate is the whole reason this function takes a :class:`ReadResult` rather
    than its spans: ``is_usable`` is the single test for "may this file be
    treated as evidence", and routing it through here means an encrypted or
    unreadable file cannot reach the index by a caller forgetting to check.
    """
    if not result.is_usable:
        return ()

    chunks: list[Chunk] = []
    for span in result.spans:
        chunks.extend(_span_chunks(result, span))
    return tuple(chunks)


def chunks_for_all(results: Iterable[ReadResult]) -> tuple[Chunk, ...]:
    """Every indexable chunk from a whole corpus, in the order given.

    Order is preserved rather than sorted so that a caller can pair the result
    back against its input, and so that ``ordinal`` stays comparable within a
    file.
    """
    chunks: list[Chunk] = []
    for result in results:
        chunks.extend(chunks_for(result))
    return tuple(chunks)
