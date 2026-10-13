"""Tests for defensive corpus traversal.

The behaviour under test is not "can we list files" -- that is trivial -- but
"does one bad file cost us the rest of the corpus". Every test here is written
with that failure mode in mind.
"""

from __future__ import annotations

import pytest
from conftest import CORPUS_DIR, HazardCorpus
from ingestion import walker
from models import MediaType, SkipReason


# --------------------------------------------------------------------------
# Synthetic corpus: every hazard, including the one that only appears on a
# filesystem that enforces permissions.
# --------------------------------------------------------------------------


def test_discovers_every_readable_file_nested_to_any_depth(hazard_corpus: HazardCorpus) -> None:
    result = walker.discover(hazard_corpus.root)

    assert sorted(f.rel_path for f in result.files) == hazard_corpus.readable


def test_unreadable_file_does_not_abort_the_walk(hazard_corpus: HazardCorpus) -> None:
    """The failure this whole module exists to prevent.

    ``vendor/a_unreadable.txt`` sorts *first* in its directory, so a walk that
    raised on it would lose every file after it -- including the two other
    vendor files, which is exactly the silent, order-dependent loss the brief
    warns about.
    """
    if not hazard_corpus.permissions_enforced:
        pytest.skip(
            "this filesystem does not enforce chmod 000 (drvfs, or running as root), "
            "so the unreadable-file case cannot be exercised here"
        )

    result = walker.discover(hazard_corpus.root)
    found = {f.rel_path for f in result.files}

    assert hazard_corpus.unreadable_rel not in found
    # Everything sorting after the bad file must still be present.
    assert "vendor/b_encrypted.pdf" in found
    assert "vendor/c_unknown.dat" in found
    assert "support/parts.csv" in found


def test_ledger_names_the_file_and_the_reason(hazard_corpus: HazardCorpus) -> None:
    if not hazard_corpus.permissions_enforced:
        pytest.skip("filesystem does not enforce chmod 000")

    result = walker.discover(hazard_corpus.root)
    ledger = {entry.rel_path: entry for entry in result.skipped}

    assert hazard_corpus.unreadable_rel in ledger
    assert ledger[hazard_corpus.unreadable_rel].reason is SkipReason.PERMISSION_DENIED
    assert ledger[hazard_corpus.unreadable_rel].detail


def test_empty_directory_is_recorded_not_an_error(hazard_corpus: HazardCorpus) -> None:
    result = walker.discover(hazard_corpus.root)

    assert result.empty_directories == ["archive"]


def test_unknown_binary_type_is_reported_as_unknown(hazard_corpus: HazardCorpus) -> None:
    result = walker.discover(hazard_corpus.root)
    types = {f.rel_path: f.media_type for f in result.files}

    assert types["vendor/c_unknown.dat"] is MediaType.UNKNOWN


def test_encrypted_pdf_is_discovered_as_a_pdf(hazard_corpus: HazardCorpus) -> None:
    """Discovery identifies the format; deciding it is unreadable is the reader's job."""
    result = walker.discover(hazard_corpus.root)
    types = {f.rel_path: f.media_type for f in result.files}

    assert types["vendor/b_encrypted.pdf"] is MediaType.PDF


def test_citation_paths_are_relative_and_use_forward_slashes(
    hazard_corpus: HazardCorpus,
) -> None:
    """The citation contract, asserted at the point where the path is created."""
    result = walker.discover(hazard_corpus.root)

    for discovered in result.files:
        assert not discovered.rel_path.startswith("/"), discovered.rel_path
        assert "\\" not in discovered.rel_path, discovered.rel_path
        assert discovered.path.is_absolute(), discovered.path


def test_missing_corpus_root_is_recorded_rather_than_silently_empty(tmp_path) -> None:
    """An absent corpus must not look like a corpus that happened to be empty."""
    result = walker.discover(tmp_path / "does-not-exist")

    assert result.total_files == 0
    assert result.total_skipped == 1
    assert result.skipped[0].reason is SkipReason.DIRECTORY_UNREADABLE


def test_results_are_sorted_so_listing_order_cannot_change_them(
    hazard_corpus: HazardCorpus,
) -> None:
    result = walker.discover(hazard_corpus.root)

    assert [f.rel_path for f in result.files] == sorted(f.rel_path for f in result.files)
    assert [s.rel_path for s in result.skipped] == sorted(s.rel_path for s in result.skipped)


def test_walking_twice_gives_the_same_answer(hazard_corpus: HazardCorpus) -> None:
    first = walker.discover(hazard_corpus.root)
    second = walker.discover(hazard_corpus.root)

    assert [f.rel_path for f in first.files] == [f.rel_path for f in second.files]
    assert [s.rel_path for s in first.skipped] == [s.rel_path for s in second.skipped]


# --------------------------------------------------------------------------
# The real sample corpus.
# --------------------------------------------------------------------------


class TestRealSampleCorpus:
    """These run against the corpus shipped with the kit, never modified."""

    def test_every_file_is_discovered(self) -> None:
        result = walker.discover(CORPUS_DIR)

        assert result.total_files == 13
        assert result.total_skipped == 0

    def test_the_known_answers_files_are_present(self) -> None:
        found = {f.rel_path for f in walker.discover(CORPUS_DIR).files}

        assert found == {
            "engineering/ingest_service.py",
            "engineering/meridian_release_notes.txt",
            "logs/prod_inference_2026-09-02.log",
            "planning/roadmap_fy27.docx",
            "specs/backplane_pinout.png",
            "specs/tq40_datasheet_r1_WITHDRAWN.pdf",
            "specs/tq40_datasheet_r2.pdf",
            "support/asset_label.jpg",
            "support/bug_database.csv",
            "support/rma_parts.xlsx",
            "vendor/internal_audit.txt",
            "vendor/supplier_agreement_ENCRYPTED.pdf",
            "vendor/telemetry_capture.dat",
        }

    def test_every_known_format_is_identified(self) -> None:
        types = {f.rel_path: f.media_type for f in walker.discover(CORPUS_DIR).files}

        assert types["specs/tq40_datasheet_r2.pdf"] is MediaType.PDF
        assert types["specs/tq40_datasheet_r1_WITHDRAWN.pdf"] is MediaType.PDF
        assert types["vendor/supplier_agreement_ENCRYPTED.pdf"] is MediaType.PDF
        assert types["planning/roadmap_fy27.docx"] is MediaType.DOCX
        assert types["support/rma_parts.xlsx"] is MediaType.XLSX
        assert types["support/bug_database.csv"] is MediaType.CSV
        assert types["engineering/meridian_release_notes.txt"] is MediaType.TEXT
        assert types["logs/prod_inference_2026-09-02.log"] is MediaType.LOG
        assert types["engineering/ingest_service.py"] is MediaType.CODE
        assert types["specs/backplane_pinout.png"] is MediaType.PNG
        assert types["support/asset_label.jpg"] is MediaType.JPG
        assert types["vendor/telemetry_capture.dat"] is MediaType.UNKNOWN

    def test_bytes_and_extensions_agree_where_they_should(self) -> None:
        """Every identified file's contents match its name.

        A failure here means a file in the corpus is not what it claims to be,
        which would make extension-based routing unsafe.
        """
        result = walker.discover(CORPUS_DIR)

        mismatched = [
            f.rel_path
            for f in result.files
            if f.media_type is not MediaType.UNKNOWN and not f.magic_agrees
        ]
        assert mismatched == []

    def test_empty_archive_directory_is_reported(self) -> None:
        assert walker.discover(CORPUS_DIR).empty_directories == ["archive"]

    def test_corpus_is_never_modified_by_walking_it(self) -> None:
        """The sample corpus is reference data; a walk must be read-only."""
        before = sorted(
            (p.relative_to(CORPUS_DIR).as_posix(), p.stat().st_size)
            for p in CORPUS_DIR.rglob("*")
            if p.is_file()
        )

        walker.discover(CORPUS_DIR)

        after = sorted(
            (p.relative_to(CORPUS_DIR).as_posix(), p.stat().st_size)
            for p in CORPUS_DIR.rglob("*")
            if p.is_file()
        )
        assert before == after

    def test_summary_is_human_readable(self) -> None:
        summary = walker.discover(CORPUS_DIR).summary()

        assert "13 readable file(s)" in summary
        assert "0 skipped" in summary
        assert "1 empty" in summary
