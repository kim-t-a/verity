"""Reader for comma-separated (and similarly delimited) files.

The delimiter is sniffed rather than assumed. Files named ``.csv`` are
frequently semicolon- or tab-separated in practice, and guessing wrong turns a
whole table into a single column of unparsed text, which retrieves badly and
yields no usable values.
"""

from __future__ import annotations

import csv
import io

from models import DiscoveredFile, ReadResult, ReadStatus

from . import source, tabular

#: Delimiters tried, in order, when deciding how a file is separated. Comma
#: first because it is by far the most common; the others exist because a
#: mis-sniffed file degrades silently into one useless column.
CANDIDATE_DELIMITERS = (",", ";", "\t", "|")


def _sniff_delimiter(sample: str) -> str:
    """Guess the delimiter, falling back to a comma.

    ``csv.Sniffer`` is used when it can decide, because it weighs the whole
    sample rather than counting a single character. It raises on input too
    uniform to judge -- a one-column file, for instance -- which is not an
    error worth propagating, just a case where the default is fine.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        return ","


def read(discovered: DiscoveredFile) -> ReadResult:
    try:
        text, note = source.read_text_file(discovered.path)
    except source.NotTextError as exc:
        return ReadResult.failure(
            discovered.rel_path, discovered.media_type, ReadStatus.UNREADABLE, str(exc)
        )

    sample = text[:4096]
    delimiter = _sniff_delimiter(sample)

    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        # A malformed file costs us this file only; the caller records it.
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            f"csv parse failed ({delimiter!r} delimiter): {exc}",
        )

    spans = tabular.build_spans(rows, location_prefix="")
    if delimiter != ",":
        note = "; ".join(filter(None, [note, f"delimiter {delimiter!r}"]))
    return ReadResult.of(discovered.rel_path, discovered.media_type, spans, detail=note)
