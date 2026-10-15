"""Reader for images: the file's own header, plus what an OCR engine saw.

The challenge guarantees images in the corpus and requires them to be handled.
This module is the seam where that happens, and it is deliberately thin: the
work is done by two pure functions it calls, and what is left here is the part
that cannot be pure -- opening a file, and starting a process.

**Rule 8 is why the unavailable path still exists.** Never silently skip image
support: if the vision path is not ready it must report itself as unavailable,
not return invented text. An image that produced no output is indistinguishable,
from the outside, from an image that was never looked at, whereas an image
reported as ``VISION_UNAVAILABLE`` is a statement about *this build* -- and that
is still the true statement when the engine has not been installed. So the
engine's absence is one of the outcomes below rather than an oversight, and it
is the outcome a development machine reaches by default.

**What this module genuinely does on its own.** It parses the image header and
reports the real format and pixel dimensions. Those are facts read out of the
file, not guesses, and they are worth having: they distinguish "we opened the
image and it is a valid 1400x900 PNG we cannot read" from "this file claims to
be a PNG and is not one". Only the first is a vision gap. The width is not
merely reported, either -- the geometry needs it to reject a heading too wide to
be a label, which is why it is read from the header rather than from the engine.

**What it deliberately does not do.** The dimensions are reported in ``detail``
and are *not* emitted as spans. Spans are what the index stores and the answer
extractor searches, and ``1400`` is not text the corpus contains -- it is
metadata about a file. Emitting it would put a number into retrieval that no
question is asking about, and could let an unrelated question match this file
and cite it. A file may only be cited once we have actually read what it says.

**How the pieces fit.** :func:`~ingestion.readers.ocr_engine.recognise` runs the
engine and hands back its bytes; :func:`~ingestion.readers.ocr_tsv.words_from_tsv`
turns those bytes into ``Detection``s; and
:func:`~ingestion.readers.image_geometry.layout_from` turns the detections into
lines and the relationships between them. Each of those is a pure function
tested without an engine and without an image, so the only decisions this module
makes are about outcomes -- which of the four ways the chain can end maps to
which status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import MAX_IMAGE_HEADER_BYTES
from models import DiscoveredFile, MediaType, ReadResult, ReadStatus, TextSpan

from . import ocr_engine, source
from .image_geometry import Layout, layout_from
from .ocr_tsv import words_from_tsv

#: PNG stores its dimensions in the IHDR chunk, which the specification
#: requires to be first. Offsets, from the start of the file:
#: 0-7 signature, 8-11 chunk length, 12-15 the literal ``IHDR``,
#: 16-19 width, 20-23 height -- all integers, big-endian.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_OFFSET = 12
_PNG_WIDTH_OFFSET = 16
_PNG_HEIGHT_OFFSET = 20
_PNG_HEADER_BYTES = 24

#: JPEG is a sequence of markers rather than a fixed layout, so the dimensions
#: have to be found by walking it. Every marker starts with 0xFF; the byte
#: after it is the kind. ``SOFn`` markers carry the frame dimensions.
_JPEG_MARKER = 0xFF
_JPEG_SOF_KINDS = frozenset(
    # SOF0-SOF3, SOF5-SOF7, SOF9-SOF11, SOF13-SOF15. The gaps (SOF4, SOF8,
    # SOF12) are not frame headers: they are the arithmetic-coding tables and
    # the JPEG extension, and treating one as a frame header would read the
    # dimensions out of the wrong bytes entirely.
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
_JPEG_STANDALONE = frozenset({0x01, 0xD8, 0xD9}) | set(range(0xD0, 0xD8))

#: Offsets into a frame header, measured from the start of its **length**
#: field -- which is where the segment walk leaves the cursor. The payload
#: itself begins two bytes later:
#:
#: ============  ==========================================
#: +0 .. +1      segment length, including these two bytes
#: +2            sample precision
#: +3 .. +4      height, big-endian
#: +5 .. +6      width, big-endian
#: +7            component count
#: ============  ==========================================
#:
#: Reading height at +5 instead of +3 yields the width and reads width out of
#: the component bytes. The result is a plausible-looking pair of swapped
#: numbers, which is worth spelling out because it is not a crash and the
#: only symptom is a summary line that is quietly wrong.
_JPEG_SOF_HEIGHT_OFFSET = 3
_JPEG_SOF_WIDTH_OFFSET = 5

#: The shortest length field that still covers precision, height, width and the
#: component count: 2 + 1 + 2 + 2 + 1. Anything below this cannot be a frame
#: header, whatever marker introduced it.
_JPEG_SOF_MINIMUM_LENGTH = 8
_JPEG_MAX_SEGMENTS = 512


@dataclass(frozen=True)
class ImageInfo:
    """What the file header states about an image, independent of its content."""

    media_type: MediaType
    width: int
    height: int

    @property
    def summary(self) -> str:
        return f"{self.media_type.value.upper()} image, {self.width}x{self.height} pixels"


def _read_integer(raw: bytes, offset: int, width: int) -> int:
    return int.from_bytes(raw[offset : offset + width], "big")


def _png_info(head: bytes) -> ImageInfo | None:
    if len(head) < _PNG_HEADER_BYTES or not head.startswith(_PNG_SIGNATURE):
        return None
    if head[_PNG_IHDR_OFFSET : _PNG_IHDR_OFFSET + 4] != b"IHDR":
        # A structurally valid signature with no IHDR first is not a PNG we can
        # reason about. Reporting no dimensions is better than reporting zero.
        return None
    return ImageInfo(
        media_type=MediaType.PNG,
        width=_read_integer(head, _PNG_WIDTH_OFFSET, 4),
        height=_read_integer(head, _PNG_HEIGHT_OFFSET, 4),
    )


def _jpeg_info(head: bytes) -> ImageInfo | None:
    """Walk the JPEG marker segments to the frame header.

    The loop is bounded. A malformed or hostile file can present a chain of
    segments that never reaches a frame header, and an unbounded walk would
    read off the end of the buffer looking for one.
    """
    if not head.startswith(b"\xff\xd8"):
        return None

    position = 2
    for _ in range(_JPEG_MAX_SEGMENTS):
        if position + 1 >= len(head) or head[position] != _JPEG_MARKER:
            return None
        kind = head[position + 1]
        position += 2

        if kind in _JPEG_STANDALONE:
            continue
        if position + 2 > len(head):
            return None

        length = _read_integer(head, position, 2)
        if length < 2:
            # A segment length below 2 would move the cursor backwards.
            return None

        if kind in _JPEG_SOF_KINDS:
            if length < _JPEG_SOF_MINIMUM_LENGTH or position + length > len(head):
                return None
            return ImageInfo(
                media_type=MediaType.JPG,
                height=_read_integer(head, position + _JPEG_SOF_HEIGHT_OFFSET, 2),
                width=_read_integer(head, position + _JPEG_SOF_WIDTH_OFFSET, 2),
            )
        position += length

    return None


def probe_from_bytes(head: bytes) -> ImageInfo | None:
    """Identify the image ``head`` begins, without touching the filesystem.

    Split out from :func:`probe` so the header parsers can be tested by handing
    them bytes -- including bytes no real file would contain -- rather than by
    writing a file for every case.

    Returns:
        The facts the header states, or ``None`` when ``head`` is not a PNG or
        JPEG whose header can be trusted. ``None`` is a real answer: it means
        the file's name claims an image and its contents do not support that
        claim.
    """
    return _png_info(head) or _jpeg_info(head)


def probe(path: Path) -> ImageInfo | None:
    """Read an image's format and dimensions from its header.

    Only the header is read; the pixels never are. This build has no use for
    them, and a large image is an expensive thing to pull off disk for facts
    that live in its first few hundred bytes.
    """
    try:
        head, _ = source.load_bytes(path, limit=MAX_IMAGE_HEADER_BYTES)
    except OSError:
        return None

    return probe_from_bytes(head)


def _spans_from(layout: Layout) -> list[TextSpan]:
    """The lines, then the relationships, as spans.

    Lines come first and keep the reading order the geometry assigned them, so
    a diagnostic listing the spans reads down the image. Associations follow,
    numbered on from the last line, because they are derived facts rather than
    things printed on the page -- and because a *line* ordinal has to stay the
    line's place in the image for the two to be told apart at all.

    Both are emitted. Withholding a line because nothing associated with it
    would throw away the text of every flat label, and withholding the
    association would leave a designator with no signal and a signal with no
    pin; neither half answers the question on its own.
    """
    spans = [
        TextSpan(text=line.text, location=line.location, ordinal=line.ordinal)
        for line in layout.lines
    ]
    last = max((line.ordinal for line in layout.lines), default=0)
    spans += [
        TextSpan(text=association.text, location=association.location, ordinal=last + index)
        for index, association in enumerate(layout.associations, start=1)
    ]
    return spans


def _recognition_detail(info: ImageInfo, layout: Layout) -> str:
    """What was recognised, in enough detail to debug a wrong answer.

    The counts are the useful part. "0 associations" against a page full of
    lines says the geometry found no diagram and refused nothing; a non-zero
    refusal count says the opposite -- there was a diagram and it was too
    ambiguous to read -- and those two call for looking at different things.
    """
    detail = (
        f"recognised as a {info.summary}; "
        f"{len(layout.lines)} text lines and {len(layout.associations)} associations"
    )
    if layout.rejections:
        # Named rather than counted, because the reason is what says whether
        # the image is ambiguous or the tolerances are wrong.
        reasons = sorted({rejection.reason.value for rejection in layout.rejections})
        detail += f"; refused {len(layout.rejections)} ambiguous pairings ({', '.join(reasons)})"
    if layout.dropped_detections:
        detail += f"; discarded {layout.dropped_detections} detections that were not text"
    if layout.duplicate_detections:
        detail += f"; dropped {layout.duplicate_detections} duplicate detections"
    return detail


def read(discovered: DiscoveredFile) -> ReadResult:
    """Recover the text of an image, or say precisely why it could not.

    Four outcomes, and the distinctions between them are the point:

    ``UNREADABLE``
        The name claims an image and the bytes are not one. No engine could
        read it, so blaming the build would be wrong.
    ``VISION_UNAVAILABLE``
        A real image and no engine in this build. A statement about the build,
        not about the file -- the one case where "we could not read it" is not
        a fault in the file.
    ``UNREADABLE``
        The engine was found and failed, or returned something that is not TSV.
        Now it *is* about the file, or about the engine's installation.
    ``OK`` / ``EMPTY``
        The engine read the image. ``EMPTY`` when it found no words, which is a
        legitimate result for a blank photograph and not a failure.
    """
    info = probe(discovered.path)

    if info is None:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            "named as an image, but the header is not a readable PNG or JPEG",
        )

    engine = ocr_engine.find_engine()
    if engine is None:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.VISION_UNAVAILABLE,
            f"recognised as a {info.summary}; no OCR engine is installed in this build, "
            "so the image's contents have not been read",
        )

    try:
        raw = ocr_engine.recognise(engine, discovered.path)
    except ocr_engine.EngineFailed as exc:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            f"recognised as a {info.summary}; the OCR engine could not read it: {exc}",
        )

    words = words_from_tsv(raw)
    if not words.header_recognised:
        # The engine ran and exited zero without producing TSV. Reporting this
        # as EMPTY would make a broken invocation look like a blank image.
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            f"recognised as a {info.summary}; the OCR engine returned no TSV to parse",
        )

    layout = layout_from(words.detections, image_width=info.width)

    return ReadResult.of(
        discovered.rel_path,
        discovered.media_type,
        _spans_from(layout),
        detail=_recognition_detail(info, layout),
    )
