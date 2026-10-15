"""Identify a file's true format.

**Why this exists instead of just trusting the extension.** The extension is a
claim made by whoever named the file. The leading bytes are evidence. For the
formats that carry a signature -- PDF, PNG, JPEG, and the ZIP container behind
DOCX and XLSX -- the bytes are decisive, and a file that claims one thing and
contains another is a fact worth knowing about rather than silently obeying.

The reverse is also true: plain text, CSV, logs and source code have no
signature at all, so for those the name is the only evidence there is, and we
use it.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.engine.models import MediaType

#: What a file's name claims it is. A hint, never the final word.
EXTENSION_TYPES: dict[str, MediaType] = {
    ".pdf": MediaType.PDF,
    ".docx": MediaType.DOCX,
    ".xlsx": MediaType.XLSX,
    ".csv": MediaType.CSV,
    ".txt": MediaType.TEXT,
    ".log": MediaType.LOG,
    ".py": MediaType.CODE,
    ".png": MediaType.PNG,
    ".jpg": MediaType.JPG,
    ".jpeg": MediaType.JPG,
}

#: Leading byte sequences that identify a format. First match wins; no two of
#: these overlap, so the order carries no meaning beyond that.
MAGIC_SIGNATURES: tuple[tuple[bytes, MediaType], ...] = (
    (b"%PDF-", MediaType.PDF),
    (b"\x89PNG\r\n\x1a\n", MediaType.PNG),
    (b"\xff\xd8\xff", MediaType.JPG),
    # DOCX and XLSX are both ZIP archives. The signature says "ZIP"; only the
    # extension distinguishes the two, which sniff() handles below.
    (b"PK\x03\x04", MediaType.ZIP),
)

#: The two ZIP-based office formats, which need the extension to tell apart.
_OOXML_TYPES = frozenset({MediaType.DOCX, MediaType.XLSX})


@dataclass(frozen=True)
class TypeVerdict:
    """What we decided a file is, and what each source of evidence said."""

    media_type: MediaType
    """The format we will treat the file as."""

    extension_type: MediaType
    """What the filename claims."""

    magic_type: MediaType | None
    """What the leading bytes say, or None when they match nothing known."""

    agrees: bool
    """True when the bytes and the name point at the same format.

    A ``None`` magic (plain text has none) counts as agreement: absence of
    evidence is not contradiction.
    """


def magic_type(head: bytes) -> MediaType | None:
    """Return the format implied by ``head``, or None if unrecognised."""
    for signature, media_type in MAGIC_SIGNATURES:
        if head.startswith(signature):
            return media_type
    return None


def type_from_extension(extension: str) -> MediaType:
    """Return the format implied by a filename extension, or UNKNOWN."""
    return EXTENSION_TYPES.get(extension.lower(), MediaType.UNKNOWN)


def sniff(head: bytes, extension: str) -> TypeVerdict:
    """Decide the format of a file from its leading bytes and its name.

    Args:
        head: The first few bytes of the file (see ``config.PROBE_BYTES``).
        extension: The filename suffix, including the dot.

    Returns:
        A :class:`TypeVerdict` recording the decision and the evidence for it.
    """
    named = type_from_extension(extension)
    signature = magic_type(head)

    if signature is None:
        # Nothing recognised. Text formats have no signature, so the name is
        # the best evidence available; a binary with no signature stays
        # UNKNOWN, which is the honest answer rather than a guess.
        return TypeVerdict(named, named, None, True)

    if signature is MediaType.ZIP:
        # docx and xlsx are indistinguishable from their ZIP header alone.
        # Trust the name when it claims one of them; otherwise report the
        # container for what it is.
        resolved = named if named in _OOXML_TYPES else MediaType.ZIP
        return TypeVerdict(resolved, named, signature, resolved == named)

    return TypeVerdict(signature, named, signature, signature == named)
