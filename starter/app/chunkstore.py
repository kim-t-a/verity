"""Persisting the index, and reading it back in a later process.

This module exists because of one fact about the challenge harness: ``--index``
runs **once**, and each of the ten questions runs in a **fresh process**. The
index is therefore not an optimisation -- it is the only channel through which
anything survives from indexing to answering. Anything held in memory during
``--index`` is gone by the time the first question is asked.

Read that way, the requirements write themselves:

* **Loading must be cheap**, because it happens once per question, ten times,
  inside a 30-second budget that also has to cover retrieval and answering.
* **A damaged index must be detected, not tolerated.** If the store cannot be
  read correctly, the alternative to refusing is answering from misread chunks
  -- confident wrong answers citing files that do not support them. That is the
  worst possible output, and it is indistinguishable from success without the
  checks below.
* **A stale index must be detected too.** The corpus root is passed again at
  query time, so the fingerprint recorded at build time can be compared against
  the corpus actually in front of us.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CHUNKS_FILENAME,
    INDEX_FORMAT_VERSION,
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
)
from models import Chunk, DiscoveredFile, MediaType


class IndexUnavailable(Exception):
    """No usable index at this path. Build one with ``--index``."""


class IndexIncompatible(Exception):
    """An index exists but cannot be trusted for this build."""


class StaleIndexError(IndexIncompatible):
    """The index was built from a different corpus than the one in front of us.

    A subclass rather than a plain error because it is the one failure the
    caller may want to treat differently: it says the *content* is wrong, not
    that the file is damaged, and the two lead to different diagnostics.
    """


@dataclass(frozen=True)
class IndexManifest:
    """Metadata about an index, small enough to read and judge on its own."""

    format_version: int
    created_utc: str
    corpus_fingerprint: str
    chunk_count: int
    file_count: int
    provider_id: str | None
    """Which embedding backend built the vectors, or ``None`` for a
    lexical-only index. Recorded so that a reader can tell the two apart
    without inspecting the vectors, and so that a stand-in index cannot be
    mistaken for a real one."""

    dimension: int | None
    status_counts: dict[str, int]
    skipped_count: int

    @property
    def has_vectors(self) -> bool:
        """Three fields have to agree before vectors can be indexed into.

        Any one of them alone would let a caller walk off the end of the
        buffer: a dimension with no count, a count with no provider, a provider
        with no file on disk.
        """
        return (
            self.provider_id is not None
            and self.dimension is not None
            and self.chunk_count > 0
        )

    def to_json(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "created_utc": self.created_utc,
            "corpus_fingerprint": self.corpus_fingerprint,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "provider_id": self.provider_id,
            "dimension": self.dimension,
            "status_counts": self.status_counts,
            "skipped_count": self.skipped_count,
        }

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "IndexManifest":
        return cls(
            format_version=int(raw["format_version"]),
            created_utc=str(raw["created_utc"]),
            corpus_fingerprint=str(raw["corpus_fingerprint"]),
            chunk_count=int(raw["chunk_count"]),
            file_count=int(raw["file_count"]),
            provider_id=raw["provider_id"] if raw["provider_id"] is None else str(raw["provider_id"]),
            dimension=None if raw["dimension"] is None else int(raw["dimension"]),
            status_counts={str(k): int(v) for k, v in dict(raw["status_counts"]).items()},
            skipped_count=int(raw["skipped_count"]),
        )


@dataclass(frozen=True)
class ChunkIndex:
    """Everything the index holds: the metadata, the chunks, and any vectors.

    ``vectors`` is positional against ``chunks`` -- entry *i* is the vector for
    chunk *i*. That pairing is the reason the two are one object rather than
    two files a caller reads separately: read apart, they can be paired wrongly,
    and a vector matched to the wrong chunk yields a citation that looks
    entirely plausible and is false.
    """

    manifest: IndexManifest
    chunks: tuple[Chunk, ...]
    vectors: tuple[tuple[float, ...], ...] = ()

    def __len__(self) -> int:
        return len(self.chunks)


def corpus_fingerprint(files: Iterable[DiscoveredFile]) -> str:
    """A short, stable digest of which files a corpus holds and how big they are.

    Deliberately built from *names and sizes*, not contents. This has to be
    computable at query time in milliseconds, before any file is read, and
    reading every file again just to verify the index would spend the
    per-question budget re-doing the work ``--index`` already did. Names and
    sizes are enough to catch what this is for: an index built over one corpus
    and queried against another.

    Sorted, so the digest does not depend on directory listing order -- which
    the harness's own documentation warns varies between filesystems.
    """
    digest = hashlib.sha256()
    for rel_path, size_bytes in sorted((f.rel_path, f.size_bytes) for f in files):
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size_bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    *,
    fingerprint: str,
    chunks: Sequence[Chunk],
    file_count: int,
    provider_id: str | None,
    dimension: int | None,
    status_counts: dict[str, int],
    skipped_count: int,
) -> IndexManifest:
    """Assemble a manifest for a freshly built index.

    ``created_utc`` is stamped here rather than passed in so there is one place
    that decides how a build time is spelled, and it is recorded in UTC so that
    the value in a diagnostic cannot be misread as local time by whoever is
    reading it.
    """
    return IndexManifest(
        format_version=INDEX_FORMAT_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        corpus_fingerprint=fingerprint,
        chunk_count=len(chunks),
        file_count=file_count,
        provider_id=provider_id,
        dimension=dimension,
        status_counts=dict(sorted(status_counts.items())),
        skipped_count=skipped_count,
    )


def _pack_vectors(vectors: Sequence[Sequence[float]]) -> bytes:
    """Flatten vectors into a little-endian float32 buffer, row by row."""
    flat = array("f")
    for vector in vectors:
        flat.extend(vector)
    if flat.itemsize != 4:
        # Not reachable on any platform we target, but a silent wrong-size
        # float would corrupt every vector rather than fail.
        raise IndexIncompatible(f"expected 4-byte floats, this build has {flat.itemsize}")
    if not _little_endian():
        flat.byteswap()
    return flat.tobytes()


def _unpack_vectors(raw: bytes, count: int, dimension: int) -> tuple[tuple[float, ...], ...]:
    """Rebuild vectors from a float32 buffer, checking the size first.

    The size check is the important part. Without it a truncated buffer yields
    short rows or a misaligned walk, and every vector after the truncation
    point belongs to the wrong chunk -- a matching bug that reports citations
    for files that do not contain the answer.
    """
    expected = count * dimension * 4
    if len(raw) != expected:
        raise IndexIncompatible(
            f"vectors file is {len(raw)} bytes; {count} chunks at dimension "
            f"{dimension} need {expected}"
        )

    flat = array("f")
    flat.frombytes(raw)
    if not _little_endian():
        flat.byteswap()
    return tuple(
        tuple(flat[start : start + dimension])
        for start in range(0, len(flat), dimension)
    )


def _little_endian() -> bool:
    return array("H", b"\x01\x00")[0] == 1


def _chunk_to_json(chunk: Chunk) -> dict[str, object]:
    """One chunk as a JSON-ready object.

    ``text`` is written raw and read back raw -- never the normalised form.
    The index stores what the corpus says; normalisation is applied when a
    comparison is made, so that a change to the normaliser does not require
    rebuilding the index and cannot silently alter what an answer quotes.
    """
    return {
        "rel_path": chunk.rel_path,
        "media_type": str(chunk.media_type),
        "location": chunk.location,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
    }


def _chunk_from_json(raw: dict[str, object]) -> Chunk:
    try:
        return Chunk(
            rel_path=str(raw["rel_path"]),
            media_type=MediaType(str(raw["media_type"])),
            location=str(raw["location"]),
            ordinal=int(raw["ordinal"]),
            text=str(raw["text"]),
        )
    except (KeyError, ValueError) as exc:
        raise IndexIncompatible(f"chunk record is malformed: {exc}") from exc


def save(directory: Path, index: ChunkIndex) -> None:
    """Write an index to ``directory``, manifest last.

    The write order is the point. ``chunks.jsonl`` and ``vectors.bin`` land
    first and the manifest lands last, so an interrupted build leaves a
    directory that :func:`load` rejects as having no index -- rather than one
    that looks complete and is missing half its chunks.

    Vectors are only written when the manifest says there are some, so an index
    whose provider is ``None`` never has a stray vectors file that a later
    reader might try to use.
    """
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / CHUNKS_FILENAME).open("w", encoding="utf-8") as handle:
        for chunk in index.chunks:
            # ensure_ascii=False keeps non-ASCII corpus text readable on disk
            # instead of expanding it to escape sequences.
            handle.write(json.dumps(_chunk_to_json(chunk), ensure_ascii=False) + "\n")

    vectors_path = directory / VECTORS_FILENAME
    if index.vectors:
        vectors_path.write_bytes(_pack_vectors(index.vectors))
    elif vectors_path.exists():
        # A rebuild without vectors must not leave the previous build's
        # vectors behind: they would be paired against a different chunk list.
        vectors_path.unlink()

    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(index.manifest.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load(directory: Path) -> ChunkIndex:
    """Read an index, refusing anything that cannot be trusted.

    Raises:
        IndexUnavailable: nothing usable at ``directory``.
        IndexIncompatible: present but damaged, or written by a different
            layout version.
    """
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise IndexUnavailable(
            f"no index at {directory} (no {MANIFEST_FILENAME}); build one with --index"
        )

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = IndexManifest.from_json(raw_manifest)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise IndexIncompatible(f"{MANIFEST_FILENAME} is malformed: {exc}") from exc

    if manifest.format_version != INDEX_FORMAT_VERSION:
        raise IndexIncompatible(
            f"index was written in format version {manifest.format_version}; "
            f"this build reads version {INDEX_FORMAT_VERSION}. Rebuild with --index."
        )

    chunks_path = directory / CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise IndexIncompatible(f"index has a manifest but no {CHUNKS_FILENAME}")

    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(_chunk_from_json(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise IndexIncompatible(
                    f"{CHUNKS_FILENAME} line {number} is not valid JSON: {exc}"
                ) from exc

    if len(chunks) != manifest.chunk_count:
        raise IndexIncompatible(
            f"manifest declares {manifest.chunk_count} chunks, "
            f"{CHUNKS_FILENAME} holds {len(chunks)}"
        )

    vectors: tuple[tuple[float, ...], ...] = ()
    if manifest.has_vectors:
        vectors_path = directory / VECTORS_FILENAME
        if not vectors_path.is_file():
            raise IndexIncompatible(f"manifest declares vectors but {VECTORS_FILENAME} is missing")
        vectors = _unpack_vectors(
            vectors_path.read_bytes(), manifest.chunk_count, manifest.dimension or 0
        )

    return ChunkIndex(manifest=manifest, chunks=tuple(chunks), vectors=vectors)


def verify_against(directory: Path, corpus: Path) -> None:
    """Check a built index still describes the corpus in front of us.

    Raises:
        StaleIndexError: the corpus has changed since the index was built.

    The corpus is fingerprinted from names and sizes only, so this costs a stat
    per file and reads none of them -- cheap enough to run inside every
    question's budget.

    Refusing is a deliberate choice over warning and continuing. A mismatch
    means the index describes a different set of documents, so the answers it
    produces would be drawn from text that is not in the corpus being asked
    about -- and they would come with citations that look right. Failing loudly
    is recoverable; a confident wrong answer is not.
    """
    # Imported here rather than at module scope: the store itself has no need
    # for corpus traversal, and importing it at the top would tie reading an
    # index to being able to walk a filesystem.
    from ingestion import walker

    index = load(directory)
    current = corpus_fingerprint(walker.discover(corpus).files)
    if current != index.manifest.corpus_fingerprint:
        raise StaleIndexError(
            f"the index at {directory} was built from a different corpus than "
            f"{corpus}; rebuild with --index"
        )
