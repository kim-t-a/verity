"""The persisted index: what ``--index`` writes and every ``--query`` reads.

Two things are being tested here, and the second matters more than the first.

The ordinary half is the round trip: chunks and vectors in, the same chunks and
vectors out, across a real write to disk.

The half that earns its keep is the *refusal* paths. The index is the only
channel from the indexing pass to the answering passes, and a store that reads
back a damaged or mismatched index without complaining produces answers drawn
from the wrong text -- with citations that look entirely plausible. Every one of
those cases is therefore tested as an expected exception rather than tolerated
behaviour. An index is cheap to rebuild (``--index`` is ungraded), so refusing
costs a startup pass and no score; answering from a misread index costs the
question.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import CORPUS_DIR

from verity.engine.index_types import (
    ChunkIndex,
    IndexIncompatible,
    IndexManifest,
    IndexUnavailable,
    StaleIndexError,
    build_manifest,
    corpus_fingerprint,
    load,
    save,
    verify_against,
)
from verity.engine.config import (
    CHUNKS_FILENAME,
    INDEX_FORMAT_VERSION,
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
)
from verity.engine.ingestion import walker
from verity.engine.models import Chunk, DiscoveredFile, MediaType


def make_chunk(
    rel_path: str = "specs/datasheet.pdf",
    ordinal: int = 0,
    location: str = "page 1 line 1",
    text: str = "maximum junction temperature 94 C",
    media_type: MediaType = MediaType.PDF,
) -> Chunk:
    return Chunk(
        rel_path=rel_path,
        media_type=media_type,
        location=location,
        ordinal=ordinal,
        text=text,
    )


def make_index(
    chunks: list[Chunk] | None = None,
    vectors: tuple[tuple[float, ...], ...] = (),
    provider_id: str | None = None,
    dimension: int | None = None,
    fingerprint: str = "fingerprint",
) -> ChunkIndex:
    chunks = [make_chunk()] if chunks is None else chunks
    manifest = build_manifest(
        fingerprint=fingerprint,
        chunks=chunks,
        file_count=1,
        provider_id=provider_id,
        dimension=dimension,
        status_counts={"ok": 1},
        skipped_count=0,
    )
    return ChunkIndex(manifest=manifest, chunks=tuple(chunks), vectors=vectors)


# --- The manifest -----------------------------------------------------------


def test_a_fresh_manifest_records_the_format_version() -> None:
    """Written by the builder, checked by the reader: the version is the seam."""
    assert make_index().manifest.format_version == INDEX_FORMAT_VERSION


def test_manifest_survives_a_json_round_trip() -> None:
    original = make_index(provider_id="hashing-bow-v1-d256", dimension=256).manifest

    restored = IndexManifest.from_json(json.loads(json.dumps(original.to_json())))

    assert restored == original


def test_a_manifest_without_a_provider_has_no_vectors() -> None:
    """``has_vectors`` guards the buffer, so all three fields must agree.

    Any one of them alone would let a caller walk off the end of an empty or
    short vector buffer.
    """
    assert make_index(provider_id=None, dimension=256).manifest.has_vectors is False
    assert make_index(provider_id="x", dimension=None).manifest.has_vectors is False
    assert make_index(chunks=[], provider_id="x", dimension=256).manifest.has_vectors is False
    assert make_index(provider_id="x", dimension=256).manifest.has_vectors is True


def test_status_counts_are_stored_in_a_stable_order() -> None:
    """Sorted at build time so two manifests for the same corpus compare equal."""
    manifest = build_manifest(
        fingerprint="f",
        chunks=[make_chunk()],
        file_count=3,
        provider_id=None,
        dimension=None,
        status_counts={"unsupported": 1, "ok": 8, "encrypted": 1},
        skipped_count=0,
    )

    assert list(manifest.status_counts) == ["encrypted", "ok", "unsupported"]


# --- Fingerprinting ---------------------------------------------------------


def test_fingerprint_ignores_file_order() -> None:
    """Directory listing order varies between filesystems; the digest must not.

    The harness documentation warns that walk order follows the directory
    listing, so a fingerprint that depended on it would report a stale index for
    a corpus that had not changed.
    """
    files = [
        DiscoveredFile(Path("/x/a.pdf"), "a.pdf", MediaType.PDF, ".pdf", 10, True),
        DiscoveredFile(Path("/x/b.txt"), "b.txt", MediaType.TEXT, ".txt", 20, True),
    ]

    assert corpus_fingerprint(files) == corpus_fingerprint(list(reversed(files)))


def test_fingerprint_changes_when_a_file_changes_size() -> None:
    before = [DiscoveredFile(Path("/x/a.pdf"), "a.pdf", MediaType.PDF, ".pdf", 10, True)]
    after = [DiscoveredFile(Path("/x/a.pdf"), "a.pdf", MediaType.PDF, ".pdf", 11, True)]

    assert corpus_fingerprint(before) != corpus_fingerprint(after)


def test_fingerprint_changes_when_a_path_changes() -> None:
    before = [DiscoveredFile(Path("/x/a.pdf"), "a.pdf", MediaType.PDF, ".pdf", 10, True)]
    after = [DiscoveredFile(Path("/x/b.pdf"), "b.pdf", MediaType.PDF, ".pdf", 10, True)]

    assert corpus_fingerprint(before) != corpus_fingerprint(after)


def test_fingerprint_is_stable_across_calls() -> None:
    """It is compared between two processes, so it may not vary at all."""
    files = [DiscoveredFile(Path("/x/a.pdf"), "a.pdf", MediaType.PDF, ".pdf", 10, True)]

    assert corpus_fingerprint(files) == corpus_fingerprint(files)


def test_fingerprint_separates_concatenation_ambiguity() -> None:
    """``("ab", "c")`` and ``("a", "bc")`` must not collide.

    Without a separator between fields, the digest would be over the same byte
    stream for both, and a corpus could change shape without the fingerprint
    noticing.
    """
    first = [
        DiscoveredFile(Path("/x/ab"), "ab", MediaType.TEXT, "", 1, True),
        DiscoveredFile(Path("/x/c"), "c", MediaType.TEXT, "", 1, True),
    ]
    second = [
        DiscoveredFile(Path("/x/a"), "a", MediaType.TEXT, "", 1, True),
        DiscoveredFile(Path("/x/bc"), "bc", MediaType.TEXT, "", 1, True),
    ]

    assert corpus_fingerprint(first) != corpus_fingerprint(second)


# --- Round trips ------------------------------------------------------------


def test_a_lexical_only_index_round_trips(tmp_path: Path) -> None:
    index = make_index()

    save(tmp_path, index)
    restored = load(tmp_path)

    assert restored.chunks == index.chunks
    assert restored.manifest == index.manifest
    assert restored.vectors == ()


def test_an_index_with_vectors_round_trips(tmp_path: Path) -> None:
    chunks = [make_chunk(ordinal=0), make_chunk(ordinal=1, text="second chunk")]
    vectors = ((1.0, 0.0, 0.5), (0.0, 1.0, 0.25))
    index = make_index(chunks=chunks, vectors=vectors, provider_id="test-v1", dimension=3)

    save(tmp_path, index)
    restored = load(tmp_path)

    assert restored.vectors == vectors
    assert len(restored) == 2


def test_float32_precision_is_the_only_loss(tmp_path: Path) -> None:
    """Vectors are stored as float32, so the round trip is lossy but bounded.

    Worth stating rather than discovering: a test asserting exact equality on a
    value like 0.1 would fail, and the reason has to be the storage width rather
    than a bug.
    """
    index = make_index(vectors=((0.1, 0.2),), provider_id="test-v1", dimension=2)

    save(tmp_path, index)
    (vector,) = load(tmp_path).vectors

    assert vector[0] == pytest.approx(0.1, abs=1e-6)
    assert vector[1] == pytest.approx(0.2, abs=1e-6)


def test_non_ascii_text_round_trips_unchanged(tmp_path: Path) -> None:
    """Corpus text is written raw, not escaped and not normalised.

    An answer has to be quoted exactly as the corpus spells it, so the store may
    not apply the comparison normaliser on the way through.
    """
    text = "température maximale — 94 °C · naïve"
    index = make_index(chunks=[make_chunk(text=text)])

    save(tmp_path, index)

    assert load(tmp_path).chunks[0].text == text
    # And the JSON on disk is readable rather than a wall of \u escapes.
    assert "température" in (tmp_path / CHUNKS_FILENAME).read_text(encoding="utf-8")


def test_chunk_normalisation_is_applied_on_read_not_on_write(tmp_path: Path) -> None:
    """``Chunk.normalised`` is derived, so a change to the normaliser needs no rebuild."""
    index = make_index(chunks=[make_chunk(text="ORR-FAN-2214-B")])

    save(tmp_path, index)
    (chunk,) = load(tmp_path).chunks

    assert chunk.text == "ORR-FAN-2214-B"
    assert chunk.normalised == "ORRFAN2214B"


def test_saving_twice_overwrites_rather_than_appends(tmp_path: Path) -> None:
    """A rebuild replaces the store; appending would silently double every chunk."""
    save(tmp_path, make_index(chunks=[make_chunk(text="first")]))
    save(tmp_path, make_index(chunks=[make_chunk(text="second")]))

    restored = load(tmp_path)

    assert len(restored) == 1
    assert restored.chunks[0].text == "second"


def test_a_rebuild_without_vectors_removes_the_old_vector_file(tmp_path: Path) -> None:
    """Otherwise the stale buffer would be paired against the new chunk list.

    That is the worst available failure: correct-looking chunks, vectors from a
    different build, and citations that come out plausible and false.
    """
    save(tmp_path, make_index(vectors=((1.0,),), provider_id="test-v1", dimension=1))
    assert (tmp_path / VECTORS_FILENAME).exists()

    save(tmp_path, make_index())

    assert not (tmp_path / VECTORS_FILENAME).exists()


def test_save_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    save(tmp_path, make_index())

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        CHUNKS_FILENAME,
        MANIFEST_FILENAME,
    ]


# --- Refusals: nothing there ------------------------------------------------


def test_loading_a_directory_with_no_index_refuses(tmp_path: Path) -> None:
    with pytest.raises(IndexUnavailable) as raised:
        load(tmp_path)

    # The message has to say how to fix it -- this is the error a fresh
    # container would hit if --index never ran.
    assert "--index" in str(raised.value)


def test_loading_a_missing_directory_refuses(tmp_path: Path) -> None:
    with pytest.raises(IndexUnavailable):
        load(tmp_path / "does-not-exist")


def test_a_manifest_without_chunks_is_a_half_built_index(tmp_path: Path) -> None:
    """The write order makes this the signature of an interrupted build.

    The manifest is written last precisely so that a crash mid-write leaves no
    manifest -- but if the two ever disagree, the index cannot be trusted.
    """
    save(tmp_path, make_index())
    (tmp_path / CHUNKS_FILENAME).unlink()

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert CHUNKS_FILENAME in str(raised.value)


# --- Refusals: damaged ------------------------------------------------------


def test_malformed_manifest_json_is_refused(tmp_path: Path) -> None:
    save(tmp_path, make_index())
    (tmp_path / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert MANIFEST_FILENAME in str(raised.value)


def test_a_manifest_missing_a_required_key_is_refused(tmp_path: Path) -> None:
    save(tmp_path, make_index())
    raw = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    del raw["chunk_count"]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatible):
        load(tmp_path)


def test_a_future_format_version_is_refused_with_both_versions_named(
    tmp_path: Path,
) -> None:
    """A reader must never guess at a layout it does not know.

    Misreading a chunk store produces confident wrong answers citing files that
    do not support them, which is the one output worse than refusing.
    """
    save(tmp_path, make_index())
    raw = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    raw["format_version"] = INDEX_FORMAT_VERSION + 1
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert str(INDEX_FORMAT_VERSION + 1) in str(raised.value)
    assert str(INDEX_FORMAT_VERSION) in str(raised.value)


def test_a_malformed_chunk_line_is_refused_and_the_line_number_is_named(
    tmp_path: Path,
) -> None:
    """A damaged record must cost the index, not one chunk.

    Silently skipping the line would leave the store short of a chunk while the
    manifest still claimed it -- and the missing chunk could be the one holding
    the answer.
    """
    save(tmp_path, make_index(chunks=[make_chunk(text="first"), make_chunk(text="second")]))
    chunks_path = tmp_path / CHUNKS_FILENAME
    chunks_path.write_text(
        chunks_path.read_text(encoding="utf-8").replace('"second"', "oops"),
        encoding="utf-8",
    )

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert "line 2" in str(raised.value)


def test_a_chunk_record_with_a_wrong_ordinal_type_is_refused(tmp_path: Path) -> None:
    save(tmp_path, make_index())
    (tmp_path / CHUNKS_FILENAME).write_text(
        json.dumps(
            {
                "rel_path": "a.txt",
                "media_type": "text",
                "location": "line 1",
                "ordinal": "not a number",
                "text": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(IndexIncompatible):
        load(tmp_path)


def test_a_chunk_count_that_disagrees_with_the_manifest_is_refused(tmp_path: Path) -> None:
    """The count is the cheap cross-check that the whole file was read."""
    save(tmp_path, make_index(chunks=[make_chunk(text="first"), make_chunk(text="second")]))
    chunks_path = tmp_path / CHUNKS_FILENAME
    kept = chunks_path.read_text(encoding="utf-8").splitlines()[0]
    chunks_path.write_text(kept + "\n", encoding="utf-8")

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert "2" in str(raised.value) and "1" in str(raised.value)


def test_a_truncated_vector_buffer_is_refused(tmp_path: Path) -> None:
    """Without the size check a short buffer misaligns every vector after the cut.

    Each one then belongs to the wrong chunk, and the failure is invisible:
    retrieval returns plausible chunks with the wrong scores.
    """
    save(tmp_path, make_index(vectors=((1.0, 2.0), (3.0, 4.0)), provider_id="t", dimension=2))
    vectors_path = tmp_path / VECTORS_FILENAME
    vectors_path.write_bytes(vectors_path.read_bytes()[:-1])

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert "bytes" in str(raised.value)


def test_a_missing_vector_file_is_refused(tmp_path: Path) -> None:
    """The manifest declaring vectors that are not there is not a lexical index.

    Treating it as one would silently drop vector retrieval for every question
    while appearing to work.
    """
    save(tmp_path, make_index(vectors=((1.0,),), provider_id="t", dimension=1))
    (tmp_path / VECTORS_FILENAME).unlink()

    with pytest.raises(IndexIncompatible) as raised:
        load(tmp_path)

    assert VECTORS_FILENAME in str(raised.value)


# --- Refusals: stale --------------------------------------------------------


def test_verify_accepts_an_index_built_from_the_corpus_at_hand(tmp_path: Path) -> None:
    """The happy path, against the real sample corpus: no exception is the result."""
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"sample corpus not present at {CORPUS_DIR}")

    discovered = walker.discover(CORPUS_DIR)
    index = make_index(fingerprint=corpus_fingerprint(discovered.files))
    index_dir = tmp_path / "index"
    save(index_dir, index)

    verify_against(index_dir, CORPUS_DIR)  # must not raise


def test_verify_refuses_an_index_built_from_a_different_corpus(tmp_path: Path) -> None:
    """A stale index yields answers drawn from text that is not in this corpus.

    Refusing is a deliberate choice over warning and continuing: the wrong
    answers would arrive with citations that look right, and a loud failure is
    recoverable while a confident wrong answer is not.
    """
    index_dir = tmp_path / "index"
    save(index_dir, make_index(fingerprint="a fingerprint from some other corpus"))

    with pytest.raises(StaleIndexError) as raised:
        verify_against(index_dir, tmp_path / "corpus")

    assert "--index" in str(raised.value)


def test_a_stale_index_is_also_an_incompatible_one(tmp_path: Path) -> None:
    """Callers that only handle ``IndexIncompatible`` still catch staleness.

    The subclass exists so the two can be told apart, not so one can escape.
    """
    assert issubclass(StaleIndexError, IndexIncompatible)


def test_verify_refuses_when_there_is_no_index_at_all(tmp_path: Path) -> None:
    with pytest.raises(IndexUnavailable):
        verify_against(tmp_path / "nothing-here", tmp_path)
