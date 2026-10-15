"""Text files whose encoding is not UTF-8.

The candidate-encoding list in ``config`` ends with latin-1, which maps all
256 byte values, so decoding never fails -- it just sometimes succeeds at
producing something that is not text. A UTF-16 file is the case that matters:
every byte decodes as cp1252, so the text comes back interleaved with NULs and
is then rejected wholesale as "not text".
"""

from __future__ import annotations

import codecs
import random

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


def test_a_binary_file_wearing_a_utf16_bom_is_still_not_text(build_corpus) -> None:
    """The BOM must not become a way past the not-text check.

    Honouring a byte-order mark means trusting two bytes to choose the decoder,
    and a decoder that never raises will decode anything. Arbitrary bytes
    starting ``FF FE`` decode as UTF-16 into plausible-looking code points, so a
    binary file named ``.txt`` would be indexed as a document -- worse than
    rejecting it, because its "text" then gets embedded and retrieved.

    Before the BOM path existed this file was rejected for containing control
    characters. This pins that it still is.
    """
    rng = random.Random(20261015)  # fixed seed: a flaky rejection test is useless
    raw = codecs.BOM_UTF16_LE + bytes(rng.randrange(256) for _ in range(20_000))
    files = build_corpus({"capture.txt": raw})

    result = read_file(files["capture.txt"])

    assert result.status is ReadStatus.UNREADABLE
    assert result.is_usable is False


def test_structured_binary_wearing_a_utf16_bom_is_not_text(build_corpus) -> None:
    """The same, with bytes chosen so no random draw can make it flaky.

    ``D8 00`` is a lone high surrogate -- unpaired by construction, and never
    produced by encoding any character. It is the shape an arbitrary byte
    stream takes far more often than not, and it is exactly what the strict
    decode on the BOM path is there to refuse.
    """
    raw = codecs.BOM_UTF16_LE + b"\x00\xd8" * 400
    files = build_corpus({"dump.txt": raw})

    result = read_file(files["dump.txt"])

    assert result.status is ReadStatus.UNREADABLE


def test_valid_utf16_text_is_read_whatever_script_it_is_in(build_corpus) -> None:
    """The boundary the test above stops at, pinned on purpose.

    ``bytes(range(0, 32))`` repeated decodes as UTF-16LE into well-formed code
    points -- Latin Extended, Syriac, Devanagari, Tibetan, Hangul -- and a
    strict decoder accepts them, so this file is read rather than refused.
    That is the deliberate side of the line: these bytes *are* valid UTF-16,
    and the only way to reject them is a heuristic about which scripts may
    appear together, which would equally reject a genuine document written in
    any of them. Narrowing this would trade a rare file for a common one.

    The test exists so that behaviour is a decision on the record rather than
    an accident someone later "fixes" into a regression.
    """
    raw = codecs.BOM_UTF16_LE + bytes(range(0, 32)) * 400
    files = build_corpus({"dump.txt": raw})

    result = read_file(files["dump.txt"])

    assert result.status is ReadStatus.OK
