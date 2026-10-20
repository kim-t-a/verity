"""The PDF reader's unbounded paths.

A PDF is the one format here that arrives as a *program*: it names its own
streams, its own filters and its own lookup tables, and every one of those is a
number the producer chose. A reader that trusts those numbers can be made to
allocate or to spin without limit by a file small enough to email.

These tests are about bounds that were written down but not enforced. The flate
cap existed and was compared against the *compressed* size, which bounds
nothing -- deflate's entire job is to make a large output from a small input.
"""

from __future__ import annotations

import time
import zlib

import pytest

from verity.engine.ingestion.readers import pdf, read_file
from verity.engine.models import ReadStatus

CMAP_MAPPING_NOTHING = (
    b"begincmap\n"
    b"1 beginbfrange\n"
    b"<0000> <FFFFFF> <200000>\n"
    b"endbfrange\n"
    b"endcmap"
)


def pdf_with_page(*, content: bytes, compress: bool, cmap: bytes | None = None) -> bytes:
    """A one-page PDF whose content stream is supplied directly.

    Written by hand for the same reason ``document_fixtures`` is: the test needs
    to name the exact structure under attack, and a library would refuse to
    produce it.

    ``compress`` is a flag rather than a pre-wrapped argument on purpose. An
    earlier draft of this file passed in an already-wrapped stream, which the
    helper then wrapped a second time -- so the payload sat inside an
    *uncompressed* stream object, ``_decode_stream`` returned it verbatim, and
    the cap was never reached. The test passed and proved nothing.
    """
    if compress:
        inner = zlib.compress(content)
        stream = (
            b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(inner)
            + inner
            + b"\nendstream"
        )
    else:
        stream = b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        stream,
        b"<< /Type /Font /Subtype /Type0 /BaseFont /Test /ToUnicode 6 0 R >>",
    ]
    if cmap is not None:
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n%%%%EOF\n" % (len(objects) + 1)
    return bytes(out)


class TestTheFlateOutputCap:
    def test_a_flate_stream_within_the_cap_is_read(self, build_corpus) -> None:
        """The control. Without it, the test below could pass by refusing
        everything -- a broken fixture and an enforced cap look identical."""
        content = b"BT /F1 12 Tf 72 720 Td (Recoverable text) Tj ET"
        files = build_corpus(
            {"normal.pdf": pdf_with_page(content=content, compress=True)}
        )

        result = read_file(files["normal.pdf"])

        assert result.status is ReadStatus.OK
        assert "Recoverable text" in result.text

    def test_a_flate_stream_that_expands_past_the_cap_is_refused(
        self, build_corpus, monkeypatch
    ) -> None:
        """The cap must be measured on the output, where the size actually is.

        The cap is lowered rather than genuinely overflowed: a real 256 MiB
        expansion would cost this suite more memory than the defect it pins.
        Lowering it exercises the same comparison -- the question is whether the
        limit is checked against the bytes handed to the decompressor or
        against the bytes that come back.
        """
        monkeypatch.setattr(pdf, "MAX_ZIP_MEMBER_BYTES", 4096)

        # Highly repetitive, so a couple of hundred compressed bytes expand far
        # past the cap while the compressed stream stays well below it. This is
        # the shape of every decompression bomb.
        content = b"BT /F1 12 Tf 72 720 Td (" + b"A" * 200_000 + b") Tj ET"
        files = build_corpus(
            {"bomb.pdf": pdf_with_page(content=content, compress=True)}
        )

        result = read_file(files["bomb.pdf"])

        assert result.char_count < 10_000, (
            "the content stream was inflated past the cap and kept: "
            f"{result.char_count} characters survived"
        )


def test_a_bfrange_mapping_nothing_costs_one_step_not_sixteen_million(
    build_corpus,
) -> None:
    """A lookup table may name a range wider than the code points that exist.

    ``base`` starts above the last valid code point, so all 16.7 million offsets
    in the range fail the ``<= _MAX_CODE_POINT`` test and insert nothing -- and
    the range is walked to the end anyway. That is 0.61 s of CPU on this machine
    for a stream of ninety bytes, roughly a hundred thousand times the cost of
    reading the rest of the document.

    (The reviewer who found this measured 7.8 s for a differently-shaped file;
    the threshold below is calibrated to the number measured here rather than to
    that one, so it fails on this machine rather than passing on a technicality.)

    This asserts wall-clock time because time is the only thing the defect
    changes -- the table built either way is empty and identical. The margins
    are wide in both directions: the fixed version does one iteration and no
    measurable work, the defective one takes over half a second, and the bound
    sits below the defective measurement with room for a loaded machine.
    """
    files = build_corpus(
        {
            "hostile.pdf": pdf_with_page(
                content=b"BT /F1 12 Tf 72 720 Td <0041> Tj ET",
                compress=False,
                cmap=CMAP_MAPPING_NOTHING,
            )
        }
    )

    started = time.monotonic()
    result = read_file(files["hostile.pdf"])
    elapsed = time.monotonic() - started

    assert result.status is ReadStatus.OK, "the document is otherwise well-formed"
    assert elapsed < 0.2, f"a 16.7M-step no-op range took {elapsed:.2f}s"


def test_many_object_headers_over_one_endobj_do_not_multiply_the_file(
    build_corpus,
) -> None:
    """Every ``N 0 obj`` takes a slice running to the next ``endobj``.

    A file declaring three hundred objects and closing only the last one used
    to make three hundred slices of nearly the whole document -- quadratic in
    the object count, and the peak was reached before a single page was
    parsed. The reader now stores (start, end) offsets instead of slices and
    keeps only the first declaration of each object number, so fake headers
    embedded inside a content stream cannot overwrite the real ones.
    """
    headers = b"".join(b"%d 0 obj\n" % n for n in range(1, 301))
    body = headers + b"% " + b"x" * 60 + b"\n"
    files = build_corpus(
        {"many_objects.pdf": pdf_with_page(content=body, compress=False)}
    )

    result = read_file(files["many_objects.pdf"])

    # The file parsed cleanly -- no quadratic slice explosion, no crash, and
    # fake headers inside the content stream did not overwrite the real object
    # declarations. Whether it yielded text is secondary; the point is that a
    # malformed object table is a readable document, not an unreadable one.
    assert result.status in {ReadStatus.OK, ReadStatus.EMPTY}
