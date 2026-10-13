"""Reader for images: recognition now, vision later.

The challenge guarantees images in the corpus and requires them to be handled.
This module does not yet read them -- there is no vision model in the build --
and its whole job is to say so *precisely*, rather than to be quietly absent.

That distinction is the point. Rule 8: never silently skip image support. If
the vision path is not ready it must report itself as unavailable, not return
invented text. An image that produced no output is indistinguishable, from the
outside, from an image that was never looked at; an image reported as
``VISION_UNAVAILABLE`` is a statement about this build, and it is the statement
that has to be visible while the requirement is still unmet.

**What this module genuinely does.** It parses the image header and reports the
real format and pixel dimensions. Those are facts read out of the file, not
guesses, and they are worth having: they distinguish "we opened the image and
it is a valid 1400x900 PNG we cannot read" from "this file claims to be a PNG
and is not one". Only the first is a vision gap.

**What it deliberately does not do.** The dimensions are reported in ``detail``
and are *not* emitted as spans. Spans are what the index stores and the answer
extractor searches, and ``1400`` is not text the corpus contains -- it is
metadata about a file. Emitting it would put a number into retrieval that no
question is asking about, and could let an unrelated question match this file
and cite it. A file may only be cited once we have actually read what it says.

**Where the vision path attaches.** Stage 7 replaces the body of :func:`read`,
after the header probe and before the return. There is no backend interface
here yet on purpose: the requirements that would shape one -- which model,
which image resolutions, whether a caption is enough or the pixels must be
reasoned over -- are not known, and an interface designed without them would be
replaced rather than implemented. The single seam is marked below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import MAX_IMAGE_HEADER_BYTES
from models import DiscoveredFile, MediaType, ReadResult, ReadStatus

from . import source

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


def read(discovered: DiscoveredFile) -> ReadResult:
    """Report a recognised image as unreadable *by this build*, with evidence.

    Never returns text. There is no code path here that can produce a span, so
    there is no way for an image to reach retrieval or to be cited as a source.
    """
    info = probe(discovered.path)

    if info is None:
        return ReadResult.failure(
            discovered.rel_path,
            discovered.media_type,
            ReadStatus.UNREADABLE,
            "named as an image, but the header is not a readable PNG or JPEG",
        )

    # --- The vision path attaches here. ---------------------------------
    # A backend would consume the pixels and return spans with locations such
    # as "region (120, 240)-(410, 300)". Until one exists, the honest result
    # is that the image was recognised and could not be read.
    return ReadResult.failure(
        discovered.rel_path,
        discovered.media_type,
        ReadStatus.VISION_UNAVAILABLE,
        f"recognised as a {info.summary}; no vision backend is configured in this build, "
        "so the image's contents have not been read",
    )
