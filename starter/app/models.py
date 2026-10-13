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
