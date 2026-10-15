"""Behaviour on the inputs a real user's documents will contain.

Every test here asserts something the engine already does. They exist because
the corpus the tests were written against is small, tidy and shaped like the
challenge, whereas uploaded documents are not: they are huge, malformed,
misnamed, and occasionally hostile. None of these may raise -- ``read_file``
converts any exception to a failure value, and that guarantee is what stops
one bad document from costing a whole ingest batch.
"""

from __future__ import annotations

import pytest

from verity.engine.ingestion.readers import read_file, reader_for
from verity.engine.models import MediaType, ReadStatus


@pytest.mark.parametrize(
    "rel_path,media_type",
    [
        ("empty.txt", MediaType.TEXT),
        ("empty.csv", MediaType.CSV),
        ("empty.pdf", MediaType.PDF),
        ("empty.docx", MediaType.DOCX),
        ("empty.xlsx", MediaType.XLSX),
    ],
)
def test_a_zero_byte_file_is_a_value_not_an_exception(
    build_corpus, rel_path: str, media_type: MediaType
) -> None:
    """A 0-byte document is malformed, not empty, and hand-written parsers raise on it."""
    files = build_corpus({rel_path: b""})
    result = read_file(files[rel_path])

    assert result.status in {ReadStatus.EMPTY, ReadStatus.UNREADABLE}
    assert result.is_usable is False


def test_a_file_that_vanishes_before_it_is_read_is_a_failure_value(build_corpus, tmp_path) -> None:
    """The delete-while-ingesting case, and the only reason read_file catches broadly."""
    files = build_corpus({"gone.txt": b"text that will not be there\n"})
    discovered = files["gone.txt"]
    discovered.path.unlink()

    result = read_file(discovered)

    assert result.status is ReadStatus.UNREADABLE
    assert "FileNotFoundError" in result.detail


def test_a_document_claiming_a_huge_size_is_capped_before_reading(build_corpus) -> None:
    """MAX_TEXT_BYTES is checked from stat before the read, so this costs a stat."""
    files = build_corpus({"big.txt": b"x" * (40 * 1024 * 1024)})

    result = read_file(files["big.txt"])

    assert result.status is ReadStatus.OK
    assert "truncated" in result.detail


def test_a_zip_member_larger_than_the_cap_is_refused(build_corpus) -> None:
    """A DOCX is a zip; a member past MAX_ZIP_MEMBER_BYTES must not be inflated."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Highly compressible, so the archive stays small while the member does not.
        archive.writestr("word/document.xml", b"<w:t>" + b"a" * (300 * 1024 * 1024) + b"</w:t>")
    files = build_corpus({"bomb.docx": buffer.getvalue()})

    result = read_file(files["bomb.docx"])

    assert result.is_usable is False
    assert result.status in {ReadStatus.UNREADABLE, ReadStatus.EMPTY}


def test_every_media_type_with_a_reader_survives_garbage_bytes(build_corpus) -> None:
    """Nothing in the reader table may raise, whatever bytes it is handed."""
    payload = bytes(range(256)) * 64
    cases = {
        "junk.txt": MediaType.TEXT,
        "junk.csv": MediaType.CSV,
        "junk.pdf": MediaType.PDF,
        "junk.docx": MediaType.DOCX,
        "junk.xlsx": MediaType.XLSX,
    }
    files = build_corpus({name: payload for name in cases})

    for name, media_type in cases.items():
        assert reader_for(media_type) is not None, media_type
        result = read_file(files[name])  # must not raise
        assert result.rel_path == name
