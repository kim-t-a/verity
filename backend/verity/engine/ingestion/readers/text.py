"""Reader for plain text: ``.txt``, ``.log`` and source code.

This is the simplest reader and the baseline every other one is measured
against. Its unit of provenance is the **line**, because that is the unit a
human already uses to talk about a text file ("line 42"), and because line
numbers survive into diagnostics without any interpretation.
"""

from __future__ import annotations

from verity.engine.models import DiscoveredFile, ReadResult, ReadStatus, TextSpan

from . import source


def read(discovered: DiscoveredFile) -> ReadResult:
    """Recover one span per non-blank line.

    Blank lines are dropped rather than emitted: they carry no tokens, so they
    add nothing to retrieval and would only pad the chunk count.

    ``str.splitlines`` is used instead of ``str.split("\\n")`` because it
    understands every line terminator a text file might use -- ``\\r\\n`` from
    Windows, a lone ``\\r`` from classic Mac, and the Unicode separators -- so
    a CRLF file does not come back as one enormous line with a stray ``\\r``
    glued to every value.
    """
    try:
        text, note = source.read_text_file(discovered.path)
    except source.NotTextError as exc:
        # A binary file that happens to be named ``.txt``. Indexing its mojibake
        # would put noise into retrieval that no query could ever match.
        return ReadResult.failure(
            discovered.rel_path, discovered.media_type, ReadStatus.UNREADABLE, str(exc)
        )

    spans = [
        TextSpan(text=line.strip(), location=f"line {number}", ordinal=number)
        for number, line in enumerate(text.splitlines(), start=1)
    ]
    return ReadResult.of(discovered.rel_path, discovered.media_type, spans, detail=note)
