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

Additionally, consecutive spans that fit together within the maximum chunk
size are *grouped* into a single chunk, so related information (a table
header and its first row, adjacent paragraphs in a section) is retrieved as
one unit rather than fragmented. Each span's source location is preserved in
the chunk's location string, separated by ``; ``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from verity.engine.config import CHUNK_BREAK_LOOKBACK, MAX_CHUNK_CHARS
from verity.engine.models import Chunk, ReadResult, TextSpan
from verity.engine.normalisation import normalise

#: Sentence-ending punctuation (``.``, ``?``, ``!``) followed by whitespace,
#: or sentence-ending punctuation at the end of the text. A digit immediately
#: before the punctuation (as in ``94.5``) prevents a match, so decimal numbers
#: are not treated as sentence boundaries. No assumption is made about the
#: script or case of the text that follows, so non-English sentences split just
#: as English ones do. Used to find clean split points inside oversized spans.
#: Abbreviations like ``e.g.`` may match, which is acceptable: the whitespace
#: and hard-split fallbacks still keep values intact, and sentence splitting is
#: only a preference, not a correctness requirement.
_SENTENCE_END = re.compile(r"(?<!\d)[.!?]\s+|(?<!\d)[.!?]\s*$")


def _find_break_at(window: str) -> int:
    """Find the best place to cut ``window`` (already capped at MAX_CHUNK_CHARS).

    Preference order: sentence boundary, then nearest space, then hard cap.
    The lookback region is the final ``CHUNK_BREAK_LOOKBACK`` characters of
    the window, so a cut never scans the whole prefix for a break that is not
    there.
    """
    lookback_start = len(window) - CHUNK_BREAK_LOOKBACK

    # 1. Sentence boundary within the lookback region: prefer ending at a
    #    full stop so a value is not torn mid-sentence.
    best = 0
    for match in _SENTENCE_END.finditer(window):
        end = match.end()
        if end >= lookback_start:
            best = end
    if best > 0:
        return best

    # 2. Nearest space within the lookback region: avoids cutting mid-token.
    break_at = window.rfind(" ", lookback_start)
    if break_at > 0:
        return break_at

    # 3. Hard split at the cap: text with no breaks at all (base64, minified).
    return len(window)


def _split_points(text: str) -> list[str]:
    """Cut ``text`` into pieces no longer than ``MAX_CHUNK_CHARS``.

    Each cut is pulled back to the nearest sentence boundary, then the
    nearest space, within ``CHUNK_BREAK_LOOKBACK`` characters before the cap.
    If neither exists, the cut falls at the cap (hard slice).

    The final piece is whatever is left, including a remainder shorter than the
    lookback. Callers check for content; this function only cuts.
    """
    pieces: list[str] = []
    remaining = text

    while len(remaining) > MAX_CHUNK_CHARS:
        window = remaining[:MAX_CHUNK_CHARS]
        break_at = _find_break_at(window)
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


def _make_group_chunk(result: ReadResult, spans: list[TextSpan]) -> list[Chunk]:
    """Build one chunk from a group of consecutive spans.

    A single-span group keeps that span's own location unchanged. A
    multi-span group joins the spans' locations with ``; `` so every source
    location is named exactly as the reader produced it -- no implied range,
    no gaps, no guesswork about lines that were dropped between spans.
    """
    if len(spans) == 1:
        span = spans[0]
        chunk = Chunk(
            rel_path=result.rel_path,
            media_type=result.media_type,
            location=span.location,
            ordinal=span.ordinal,
            text=span.text,
        )
    else:
        combined_text = "\n".join(span.text for span in spans)
        location = "; ".join(span.location for span in spans)
        chunk = Chunk(
            rel_path=result.rel_path,
            media_type=result.media_type,
            location=location,
            ordinal=spans[0].ordinal,
            text=combined_text,
        )

    return [chunk] if chunk.has_searchable_content else []


def chunks_for(result: ReadResult) -> tuple[Chunk, ...]:
    """Every indexable chunk from one file's read result.

    Returns an empty tuple for any file that was not successfully read. That
    gate is the whole reason this function takes a :class:`ReadResult` rather
    than its spans: ``is_usable`` is the single test for "may this file be
    treated as evidence", and routing it through here means an encrypted or
    unreadable file cannot reach the index by a caller forgetting to check.

    Consecutive spans that fit together within ``MAX_CHUNK_CHARS`` are grouped
    into one chunk, joined by a newline. An oversized span is never grouped
    with its neighbours; it is split on its own by :func:`_span_chunks`.
    """
    if not result.is_usable:
        return ()

    # Drop spans that cannot be matched by anything, matching the behaviour
    # of _span_chunks so punctuation-only text never reaches the index.
    spans = [
        span for span in result.spans
        if any(character.isalnum() for character in normalise(span.text))
    ]

    chunks: list[Chunk] = []
    current_group: list[TextSpan] = []
    current_text = ""

    for span in spans:
        # An oversized span is never grouped with its neighbours. Flush the
        # current group, then split this span on its own.
        if len(span.text) > MAX_CHUNK_CHARS:
            if current_group:
                chunks.extend(_make_group_chunk(result, current_group))
                current_group = []
                current_text = ""
            chunks.extend(_span_chunks(result, span))
            continue

        # Try to add this span to the current group, joined by a newline.
        candidate = current_text + "\n" + span.text if current_text else span.text
        if len(candidate) <= MAX_CHUNK_CHARS:
            current_group.append(span)
            current_text = candidate
        else:
            # This span would overflow the group. Flush the group, then start
            # a new group with this span.
            if current_group:
                chunks.extend(_make_group_chunk(result, current_group))
            current_group = [span]
            current_text = span.text

    if current_group:
        chunks.extend(_make_group_chunk(result, current_group))

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
