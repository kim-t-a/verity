"""Tests for the image reader's own job: the header, and the outcome it picks.

This reader's contract has two halves. The first is that it recognises an image
and states plainly what it found, and the second -- unchanged from before there
was any engine -- is that it never invents a span. A fabricated caption would
flow into the index, be retrieved, and be cited, and nothing downstream could
tell it apart from text that was really in the file. So the assertions below
check ``spans == ()`` on every path that does not end in a successful read.

**The engine is forced absent throughout this file.** Everything here is about
what the reader does *without* one, and that must not depend on whether the
machine running the tests happens to have Tesseract installed -- otherwise the
suite would assert one thing on a laptop and another in the container, and a
test that means different things in different places is worse than no test. The
fixture below pins it. The engine-present path, and the recognition itself, are
tested in ``test_readers_ocr_engine.py``, which supplies a stand-in engine.
"""

from __future__ import annotations

import struct

import pytest
from config import OCR_ENGINE_ENV
from document_fixtures import jpeg_bytes, png_bytes
from ingestion.readers import image, read_file
from models import MediaType, ReadStatus


@pytest.fixture(autouse=True)
def engine_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the engine setting at nothing, whatever the host has installed.

    A path that does not resolve makes ``find_engine`` return ``None`` rather
    than falling back to the name on ``PATH``, which is exactly the "this build
    has no engine" state these tests are about.
    """
    monkeypatch.setenv(OCR_ENGINE_ENV, "/nonexistent/tesseract-for-tests")


class TestPngHeader:
    def test_dimensions_are_read(self) -> None:
        info = image.probe_from_bytes(png_bytes(width=1400, height=900))

        assert info is not None
        assert info.media_type is MediaType.PNG
        assert (info.width, info.height) == (1400, 900)

    def test_a_signature_without_ihdr_is_not_trusted(self) -> None:
        """A valid PNG signature is eight bytes and proves very little.

        Reading dimensions from a fixed offset without checking that IHDR is
        actually there would report whatever bytes happen to sit at those
        offsets -- numbers that look like measurements and are not.
        """
        assert image.probe_from_bytes(png_bytes()[:12] + b"XXXX" + png_bytes()[16:]) is None

    def test_a_truncated_header_is_not_trusted(self) -> None:
        assert image.probe_from_bytes(png_bytes()[:23]) is None

    def test_a_non_png_is_not_mistaken_for_one(self) -> None:
        assert image.probe_from_bytes(b"not an image at all") is None


class TestJpegHeader:
    def test_dimensions_are_read_from_a_bare_frame(self) -> None:
        info = image.probe_from_bytes(jpeg_bytes(width=640, height=480))

        assert info is not None
        assert info.media_type is MediaType.JPG
        assert (info.width, info.height) == (640, 480)

    def test_dimensions_are_found_after_metadata(self) -> None:
        """The case a fixed-offset probe gets wrong.

        Camera and editor output carries APP1 segments -- EXIF, colour
        profiles -- ahead of the frame header, so a reader that assumes the
        frame is at a known offset works on a minimal file and fails on every
        real photograph.
        """
        info = image.probe_from_bytes(jpeg_bytes(width=800, height=600, metadata_bytes=4096))

        assert info is not None
        assert (info.width, info.height) == (800, 600)

    @pytest.mark.parametrize("frame_kind", [0xC0, 0xC1, 0xC2])
    def test_every_frame_kind_carries_dimensions_the_same_way(self, frame_kind: int) -> None:
        """Baseline, extended sequential and progressive all place them alike."""
        info = image.probe_from_bytes(jpeg_bytes(width=320, height=240, frame_kind=frame_kind))

        assert info is not None
        assert (info.width, info.height) == (320, 240)

    def test_a_marker_that_is_not_a_frame_header_is_skipped(self) -> None:
        """``0xC4`` is a Huffman table, not a frame.

        Its payload has the same two-byte shape at those offsets, so a walker
        that matched any ``0xFFCx`` would read a table as a picture size.
        """
        payload = bytes(20)
        raw = b"\xff\xd8\xff\xc4" + struct.pack(">H", len(payload) + 2) + payload
        raw += jpeg_bytes(width=100, height=50)[2:]

        info = image.probe_from_bytes(raw)

        assert info is not None
        assert (info.width, info.height) == (100, 50)

    def test_a_truncated_segment_chain_is_not_walked_forever(self) -> None:
        """A hostile file can present segments that never reach a frame."""
        raw = b"\xff\xd8" + (b"\xff\xe1\xff\xff" + b"\x00" * 4) * 50

        assert image.probe_from_bytes(raw) is None

    def test_a_zero_length_segment_does_not_stall_the_walk(self) -> None:
        """A declared length below 2 would move the cursor backwards."""
        raw = b"\xff\xd8\xff\xe1\x00\x00" + b"\x00" * 32

        assert image.probe_from_bytes(raw) is None


class TestReaderContract:
    def test_a_recognised_image_reports_vision_unavailable(self, build_corpus) -> None:
        files = build_corpus({"specs/backplane_pinout.png": png_bytes(width=1400, height=900)})

        result = read_file(files["specs/backplane_pinout.png"])

        assert result.status is ReadStatus.VISION_UNAVAILABLE
        assert result.is_usable is False

    def test_the_detail_states_what_was_recognised(self, build_corpus) -> None:
        """The report has to be specific enough to be actionable.

        "Unsupported file" would leave a reader unable to tell whether the
        image was opened and understood, or skipped without being looked at.
        The detail has to name the *build* as the reason, too: the file is
        fine, and blaming it would send someone to inspect a healthy image.
        """
        files = build_corpus({"specs/pinout.png": png_bytes(width=1400, height=900)})

        result = read_file(files["specs/pinout.png"])

        assert "1400x900" in result.detail
        assert "PNG" in result.detail
        assert "no OCR engine is installed in this build" in result.detail

    def test_a_missing_engine_is_what_causes_vision_unavailable(self, build_corpus) -> None:
        """The status is attributable to the build, not to the file.

        Pinned because the two are easy to conflate: a reader that reported
        every image as unreadable would look identical from the outside, and
        would blame a corpus of perfectly good images.
        """
        files = build_corpus({"specs/pinout.png": png_bytes(width=1400, height=900)})

        result = read_file(files["specs/pinout.png"])

        assert result.status is ReadStatus.VISION_UNAVAILABLE

    def test_a_jpeg_is_recognised_too(self, build_corpus) -> None:
        files = build_corpus({"support/asset_label.jpg": jpeg_bytes(width=1200, height=800)})

        result = read_file(files["support/asset_label.jpg"])

        assert result.status is ReadStatus.VISION_UNAVAILABLE
        assert "1200x800" in result.detail

    def test_a_corrupt_file_wearing_an_image_extension_is_unreadable(
        self, build_corpus
    ) -> None:
        """This is a different finding from a vision gap, and says so.

        The name claims a PNG and the bytes are not one. Reporting
        ``VISION_UNAVAILABLE`` here would blame the missing model for a file
        that no model could read.
        """
        files = build_corpus({"specs/broken.png": b"this is plain text, not a picture"})

        result = read_file(files["specs/broken.png"])

        assert result.status is ReadStatus.UNREADABLE
        assert result.spans == ()

    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8",
            b"\xff\xd8\xff\xd9",
            bytes(range(256)),
        ],
    )
    def test_no_input_ever_produces_a_span(self, build_corpus, payload: bytes) -> None:
        """The invariant. Whatever the bytes, no text may be invented."""
        files = build_corpus({"specs/mystery.png": payload})

        result = read_file(files["specs/mystery.png"])

        assert result.spans == ()
        assert not result.is_usable

    def test_an_empty_file_is_reported_as_empty_not_as_an_image(self, build_corpus) -> None:
        """A zero-byte ``.png`` is a different finding again.

        It is not an image with unreadable contents; it is nothing at all.
        """
        files = build_corpus({"specs/nothing.png": b""})

        result = read_file(files["specs/nothing.png"])

        assert result.status is ReadStatus.UNREADABLE
        assert result.spans == ()
