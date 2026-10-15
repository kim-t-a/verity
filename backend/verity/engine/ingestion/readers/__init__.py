"""Readers: turning one discovered file into text with provenance.

Each format gets exactly one module, and every module exposes the same
``read(discovered) -> ReadResult``. Adding a format therefore means adding a
module and one line in the table below, not editing a chain of conditionals.

**The invariant every reader must hold: it never raises.** A reader that
throws would abort the indexing pass and cost us the entire corpus over one
malformed file. Failures are returned as values, which is why ``ReadResult``
carries a status rather than the readers returning bare text.
"""

from __future__ import annotations

from verity.engine.models import DiscoveredFile, MediaType, ReadResult, ReadStatus

from . import delimited, docx, image, pdf, text, xlsx

#: Which reader handles which media type. A type absent from this table is
#: reported ``UNSUPPORTED`` rather than guessed at -- the honest answer for a
#: format nothing here understands, such as the corpus's unknown binary.
#:
#: Images are in the table precisely because they are *not* expected to be
#: understood here. Leaving them out would make an image indistinguishable
#: from an unknown binary in the output; the image reader reports
#: ``VISION_UNAVAILABLE``, which names the gap instead of hiding it.
_READERS = {
    MediaType.TEXT: text.read,
    MediaType.LOG: text.read,
    MediaType.CODE: text.read,
    MediaType.CSV: delimited.read,
    MediaType.DOCX: docx.read,
    MediaType.XLSX: xlsx.read,
    MediaType.PDF: pdf.read,
    MediaType.PNG: image.read,
    MediaType.JPG: image.read,
}


def reader_for(media_type: MediaType):
    """The reader for a media type, or ``None`` when there is not one yet."""
    return _READERS.get(media_type)


def read_file(discovered: DiscoveredFile) -> ReadResult:
    """Read one discovered file, converting any failure into a result value.

    The ``except Exception`` is deliberate and load-bearing. Parsers for
    complex binary formats fail in ways that are impractical to enumerate --
    a truncated stream, an unexpected integer width, a decompressor raising
    something undocumented. Catching narrowly here would mean a new crash
    class every time the corpus contained something new, and a crash during
    indexing costs every file rather than one. The failure is recorded with
    its exception type, so nothing is hidden.
    """
    reader = reader_for(discovered.media_type)
    if reader is None:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNSUPPORTED,
            f"no reader for media type {discovered.media_type!r}",
        )

    try:
        return reader(discovered)
    except Exception as exc:  # noqa: BLE001 - see the docstring above
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            f"{type(exc).__name__}: {exc}",
        )
