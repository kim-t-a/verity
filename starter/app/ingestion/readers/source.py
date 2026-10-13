"""Reading bytes off disk safely, and turning them into text.

Every reader that handles a text-shaped format starts by calling into this
module rather than touching the filesystem itself, so that the three hazards
that apply to *all* of them -- an unbounded read, an undecodable byte
sequence, and a binary file wearing a text extension -- are handled once, in
one place, instead of six times with six slightly different behaviours.
"""

from __future__ import annotations

from pathlib import Path

from config import (
    MAX_CONTROL_CHARACTER_RATIO,
    MAX_TEXT_BYTES,
    TEXT_ENCODINGS,
    UNREMARKABLE_ENCODINGS,
)


class SourceTooLarge(Exception):
    """Raised when a file exceeds the configured size cap.

    A dedicated exception rather than a generic ``OSError`` so the caller can
    report the real reason instead of a misleading "unreadable".
    """


class NotTextError(Exception):
    """Raised when bytes decode but are plainly not text.

    Decoding cannot decide this on its own: once Latin-1 is among the candidate
    encodings, every byte sequence decodes successfully. The control-character
    check in :func:`looks_like_text` is what actually separates a document from
    a binary file with a misleading name.
    """


def _is_control(character: str) -> bool:
    """Whether a character is an invisible control code rather than writing."""
    if character in "\t\n\r":
        return False
    code = ord(character)
    return code < 32 or 127 <= code <= 159


def looks_like_text(text: str) -> bool:
    """Whether decoded content is plausibly a document."""
    if not text:
        return True
    controls = sum(1 for character in text if _is_control(character))
    return controls / len(text) <= MAX_CONTROL_CHARACTER_RATIO


def load_bytes(path: Path, limit: int | None = None) -> tuple[bytes, str]:
    """Read at most ``limit`` bytes from ``path``.

    Returns the bytes and a note that is empty when the whole file was read.
    The size is checked from ``stat`` *before* reading, so a file that claims
    to be enormous costs us a stat and nothing more.

    ``limit`` is resolved inside the function rather than as a default argument
    so that the cap is read from ``config`` at call time. A default argument
    would be bound once, at import, which would make the constant quietly
    impossible to adjust.
    """
    cap = MAX_TEXT_BYTES if limit is None else limit
    size = path.stat().st_size
    if size > cap:
        with path.open("rb") as handle:
            return handle.read(cap), f"truncated at {cap} bytes; the file is {size} bytes"
    return path.read_bytes(), ""


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode bytes to text, and say which encoding won.

    Returns ``(text, encoding)``. This cannot fail: Latin-1 maps all 256 byte
    values, so the final candidate always succeeds. An undecodable file
    therefore yields mojibake rather than an error, which is the right
    trade-off at this layer -- mojibake still tokenises and can still be
    matched against a query. Whether the result is *text at all* is a separate
    question, answered by :func:`looks_like_text`.
    """
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # Unreachable while Latin-1 is in TEXT_ENCODINGS; kept so the function is
    # total even if that list is ever trimmed.
    return raw.decode("utf-8", "replace"), "utf-8 (with replacement)"


def read_text_file(path: Path, limit: int | None = None) -> tuple[str, str]:
    """Read and decode a text-shaped file, returning ``(text, note)``.

    ``note`` combines truncation and encoding remarks, and is empty when the
    read was entirely routine.

    Raises :class:`NotTextError` when the content decodes but is not text --
    the caller records that as a failure rather than indexing the mojibake.
    """
    raw, note = load_bytes(path, limit)
    text, encoding = decode_text(raw)

    if not looks_like_text(text):
        raise NotTextError(
            f"content is not text: more than "
            f"{MAX_CONTROL_CHARACTER_RATIO:.0%} of the decoded bytes are control characters"
        )

    notes = [note] if note else []
    if encoding not in UNREMARKABLE_ENCODINGS:
        notes.append(f"decoded as {encoding}")
    return text, "; ".join(notes)
