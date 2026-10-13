"""Shared data structures for ingestion.

These are deliberately small and mostly immutable. A discovered file is a
*fact about the corpus* at the moment we walked it, not a place to accumulate
state -- so it is frozen, and anything mutable lives in the result object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class MediaType(StrEnum):
    """The formats the challenge guarantees, plus catch-alls.

    ``UNKNOWN`` is not an error: the corpus deliberately contains a file of an
    unknown binary type, and discovering it correctly means reporting it as
    unknown rather than guessing or crashing.
    """

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    TEXT = "text"
    LOG = "log"
    CODE = "code"
    PNG = "png"
    JPG = "jpg"
    ZIP = "zip"
    UNKNOWN = "unknown"


class SkipReason(StrEnum):
    """Why a file or directory was not usable.

    Kept as a closed set rather than free text so the ledger can be grouped and
    asserted on in tests. The free-text ``detail`` field carries the rest.
    """

    PERMISSION_DENIED = "permission_denied"
    UNREADABLE = "unreadable"
    NOT_A_REGULAR_FILE = "not_a_regular_file"
    STAT_FAILED = "stat_failed"
    DIRECTORY_UNREADABLE = "directory_unreadable"
    SYMLINKED_DIRECTORY = "symlinked_directory"


@dataclass(frozen=True)
class DiscoveredFile:
    """A file we can read, with everything downstream needs to know about it."""

    path: Path
    """Absolute location on disk. Used to open the file; never cited."""

    rel_path: str
    """Path relative to the corpus root, with forward slashes.

    **This is the citation string.** The contract grades citations as an exact
    set of corpus-relative paths, so this field -- not ``path``, not the bare
    filename -- is what ends up in the output JSON.
    """

    media_type: MediaType
    extension: str
    size_bytes: int

    magic_agrees: bool
    """False when the file's leading bytes contradict its extension.

    Worth recording rather than resolving silently: a PDF named ``.txt`` is a
    finding about the corpus, and later stages may want to know that the name
    is not trustworthy for this file.
    """


@dataclass(frozen=True)
class SkippedEntry:
    """One thing we could not use, and why.

    The walk records these instead of raising, so that a single bad file costs
    us that file and nothing else.
    """

    rel_path: str
    reason: SkipReason
    detail: str


@dataclass
class DiscoveryResult:
    """Everything one traversal of the corpus learned."""

    root: Path
    files: list[DiscoveredFile] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)
    empty_directories: list[str] = field(default_factory=list)
    directory_count: int = 0

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped)

    def files_of_type(self, media_type: MediaType) -> list[DiscoveredFile]:
        return [f for f in self.files if f.media_type is media_type]

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for discovered in self.files:
            counts[str(discovered.media_type)] = counts.get(str(discovered.media_type), 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        """A one-line human-readable digest, used in logs and diagnostics."""
        parts = ", ".join(f"{kind}={count}" for kind, count in self.counts_by_type().items())
        return (
            f"{self.total_files} readable file(s) [{parts or 'none'}], "
            f"{self.total_skipped} skipped, "
            f"{self.directory_count} director(y/ies), "
            f"{len(self.empty_directories)} empty"
        )


class ReadStatus(StrEnum):
    """How far a reader got with one file.

    The distinction that matters most is ``OK`` versus ``EMPTY``. Both mean the
    reader understood the format and finished without error, but only ``OK``
    means it recovered text. A file that yielded nothing is not evidence for
    anything, so it must never be cited -- and if the two cases were collapsed
    into one, an empty read would look exactly like a successful one.
    """

    OK = "ok"
    """Parsed and produced at least one non-blank span."""

    EMPTY = "empty"
    """Parsed cleanly, but the document genuinely holds no text."""

    ENCRYPTED = "encrypted"
    """Password-protected. We never attempt to crack it; the file is not a source."""

    UNREADABLE = "unreadable"
    """The format was recognised but the bytes could not be parsed."""

    UNSUPPORTED = "unsupported"
    """No reader exists for this media type (images, until the vision path is ready)."""


@dataclass(frozen=True)
class TextSpan:
    """A contiguous run of text that came from one identifiable place.

    Provenance is carried per span rather than per file because the answer has
    to be traceable to a location: "the value is somewhere in this 40-page
    datasheet" is not good enough to debug a wrong answer, and not good enough
    to prove a citation was necessary.
    """

    text: str
    """The recovered text, exactly as it appeared in the source."""

    location: str
    """Human-readable provenance -- ``page 3``, ``sheet 'RMA Parts' row 7``,
    ``line 42``. Shown in diagnostics; never cited in the output JSON."""

    ordinal: int
    """Position within the file, for ordering. Only comparable within one file."""


@dataclass(frozen=True)
class ReadResult:
    """Everything one reader recovered from one file.

    Readers never raise. A failure is a value, so that a single malformed file
    costs us that file and nothing else -- the same rule the walker follows.
    """

    rel_path: str
    media_type: MediaType
    status: ReadStatus
    spans: tuple[TextSpan, ...] = ()
    detail: str = ""
    """Why the read failed, or what was unusual about it. Empty when routine."""

    @property
    def is_usable(self) -> bool:
        """True only when this file may be treated as evidence.

        ``Never cite a file we did not successfully read`` -- this property is
        the single gate for that rule, so later stages ask it rather than
        re-deriving the answer from ``status`` themselves.
        """
        return self.status is ReadStatus.OK

    @property
    def text(self) -> str:
        """All recovered text, one span per line."""
        return "\n".join(span.text for span in self.spans)

    @property
    def char_count(self) -> int:
        return sum(len(span.text) for span in self.spans)

    @classmethod
    def failure(
        cls,
        rel_path: str,
        media_type: MediaType,
        status: ReadStatus,
        detail: str,
    ) -> "ReadResult":
        return cls(rel_path=rel_path, media_type=media_type, status=status, detail=detail)

    @classmethod
    def of(
        cls,
        rel_path: str,
        media_type: MediaType,
        spans: list[TextSpan],
        detail: str = "",
    ) -> "ReadResult":
        """Build a result, downgrading to ``EMPTY`` when nothing was recovered.

        Centralised so no reader can accidentally report ``OK`` for a file it
        got no text out of.
        """
        kept = tuple(span for span in spans if span.text.strip())
        if not kept:
            return cls.failure(
                rel_path,
                media_type,
                ReadStatus.EMPTY,
                detail or "the document contains no extractable text",
            )
        return cls(
            rel_path=rel_path, media_type=media_type, status=ReadStatus.OK, spans=kept, detail=detail
        )
