"""Text files whose encoding is not UTF-8.

The candidate-encoding list in ``config`` ends with latin-1, which maps all
256 byte values, so decoding never fails -- it just sometimes succeeds at
producing something that is not text. A UTF-16 file is the case that matters:
every byte decodes as cp1252, so the text comes back interleaved with NULs and
is then rejected wholesale as "not text".
"""

from __future__ import annotations

import codecs

from verity.engine.ingestion.readers import read_file
from verity.engine.models import ReadStatus

BODY = "Section 4.2 Termination\nNotice period is 30 days.\n"


def test_a_utf16_le_file_is_read(build_corpus) -> None:
    """What Notepad writes when saving as "Unicode"."""
    files = build_corpus({"notes.txt": BODY.encode("utf-16")})  # includes a BOM

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "30 days" in result.text


def test_a_utf16_be_file_is_read(build_corpus) -> None:
    """Big-endian, with an explicit BOM: str.encode("utf-16-be") adds none."""
    raw = codecs.BOM_UTF16_BE + BODY.encode("utf-16-be")
    files = build_corpus({"notes.txt": raw})

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "30 days" in result.text


def test_a_cp1252_file_is_still_read_as_cp1252(build_corpus) -> None:
    """The guard on the fix: BOM detection must not swallow ordinary files.

    0x93 and 0x94 are curly quotes in cp1252. This file has no BOM, so it must
    still take the candidate-encoding path rather than being forced through a
    UTF-16 decoder that would happily produce garbage of the right length.
    """
    files = build_corpus({"notes.txt": b"\x93Termination\x94 notice is 30 days.\n"})

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "Termination" in result.text
    assert "30 days" in result.text


def test_a_truncated_utf16_file_keeps_the_text_it_has(build_corpus) -> None:
    """A half-written UTF-16 file is salvaged, not discarded.

    Cutting the last unit in half leaves a byte sequence an unbending UTF-16
    decoder refuses. Refusing it here would mean a real user's file -- one
    whose tail was never written -- reads as *unreadable* even though every
    line before the cut is intact. Decoding with ``errors="replace"`` keeps
    ``decode_text`` total, which is its documented contract, and matches how
    the PDF reader treats a truncated flate stream: a prefix is worth more
    than nothing.
    """
    raw = codecs.BOM_UTF16_LE + BODY.encode("utf-16-le") + b"A"  # trailing odd byte
    files = build_corpus({"truncated.txt": raw})

    result = read_file(files["truncated.txt"])

    assert result.status is ReadStatus.OK
    assert "30 days" in result.text
