"""End-to-end checks against the real sample corpus.

Every other suite tests a reader in isolation with documents this project
built. This one runs the whole ingestion path over the corpus that ships with
the kit and asserts against ``sample-questions.json`` -- the only ground truth
available before the graded corpus exists.

**``sample-questions.json`` is used here as test data, never as a lookup
table.** Nothing in ``starter/app`` may read that file: its contents describe
the sample corpus, and the graded corpus is different and larger. What these
tests establish is the weaker, transferable claim -- that for each sample
question, the expected answer is genuinely recoverable from the file the
question expects to be cited.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import CORPUS_DIR, REPO_ROOT
from verity.engine.ingestion import walker
from verity.engine.ingestion.readers import read_file
from verity.engine.models import ReadStatus

SAMPLE_QUESTIONS = REPO_ROOT / "sample-questions.json"


def normalise(value: str) -> str:
    """The contract's single normalisation, applied to test comparisons.

    Uppercase; drop whitespace and the separators ``-`` ``.`` ``·`` ``_``.
    Written out here rather than imported so that a change to the production
    normaliser cannot silently change what these tests mean.
    """
    return re.sub(r"[\s\-._·]", "", value.upper())


@pytest.fixture(scope="module")
def corpus_reads() -> dict[str, object]:
    """Read every file in the sample corpus once, through the real walker."""
    return {
        discovered.rel_path: read_file(discovered)
        for discovered in walker.discover(CORPUS_DIR).files
    }


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))["queries"]


class TestCorpusCoverage:
    def test_every_file_produces_a_result(self, corpus_reads) -> None:
        assert len(corpus_reads) == 13

    def test_the_expected_status_of_each_file(self, corpus_reads) -> None:
        """A pinned map, so a regression in any single reader is visible here."""
        statuses = {rel: str(result.status) for rel, result in corpus_reads.items()}

        assert statuses == {
            "engineering/ingest_service.py": "ok",
            "engineering/meridian_release_notes.txt": "ok",
            "logs/prod_inference_2026-09-02.log": "ok",
            "planning/roadmap_fy27.docx": "ok",
            "specs/backplane_pinout.png": "vision_unavailable",
            "specs/tq40_datasheet_r1_WITHDRAWN.pdf": "ok",
            "specs/tq40_datasheet_r2.pdf": "ok",
            "support/asset_label.jpg": "vision_unavailable",
            "support/bug_database.csv": "ok",
            "support/rma_parts.xlsx": "ok",
            "vendor/internal_audit.txt": "ok",
            "vendor/supplier_agreement_ENCRYPTED.pdf": "encrypted",
            # The only genuinely unknown format in the corpus. It stays
            # UNSUPPORTED, which is what makes the two image rows above
            # meaningful: the statuses distinguish "a gap we must close" from
            # "a file nothing could read".
            "vendor/telemetry_capture.dat": "unsupported",
        }

    def test_no_file_is_unreadable(self, corpus_reads) -> None:
        """``UNREADABLE`` means the reader failed. Nothing here should."""
        broken = [
            rel for rel, result in corpus_reads.items() if result.status is ReadStatus.UNREADABLE
        ]

        assert broken == []

    def test_usable_files_all_produced_text(self, corpus_reads) -> None:
        for rel_path, result in corpus_reads.items():
            if result.is_usable:
                assert result.char_count > 0, rel_path

    def test_reading_the_corpus_does_not_modify_it(self) -> None:
        before = sorted(
            (p.relative_to(CORPUS_DIR).as_posix(), p.stat().st_size)
            for p in CORPUS_DIR.rglob("*")
            if p.is_file()
        )

        for discovered in walker.discover(CORPUS_DIR).files:
            read_file(discovered)

        after = sorted(
            (p.relative_to(CORPUS_DIR).as_posix(), p.stat().st_size)
            for p in CORPUS_DIR.rglob("*")
            if p.is_file()
        )
        assert before == after


class TestSampleAnswersAreRecoverable:
    """For every answerable sample question, the value is in the cited file."""

    TEXT_ANSWERABLE = [1, 2, 3, 4, 5, 6, 9]

    @pytest.mark.parametrize("number", TEXT_ANSWERABLE)
    def test_answer_appears_in_its_expected_citations(
        self, samples, corpus_reads, number: int
    ) -> None:
        item = next(q for q in samples if q["n"] == number)
        haystack = "\n".join(
            corpus_reads[rel].text for rel in item["expected_citations"] if rel in corpus_reads
        )

        assert normalise(item["expected_answer"]) in normalise(haystack), (
            f"question {number}: {item['expected_answer']!r} not found in "
            f"{item['expected_citations']}"
        )

    @pytest.mark.parametrize("number", [7, 8])
    def test_image_questions_are_not_yet_answerable(
        self, samples, corpus_reads, number: int
    ) -> None:
        """These must fail *now*, and the reason must be the missing vision path.

        If this test ever starts passing, it means text extraction has begun
        inventing content for images -- which would be far worse than failing.
        So it asserts the specific status rather than merely "not OK": an image
        that came back ``EMPTY``, or ``UNREADABLE``, would mean something else
        went wrong and the reason is no longer the one recorded here.
        """
        item = next(q for q in samples if q["n"] == number)
        results = [corpus_reads[rel] for rel in item["expected_citations"]]

        assert all(result.status is ReadStatus.VISION_UNAVAILABLE for result in results)
        assert all(result.spans == () for result in results)

    def test_the_refusal_question_has_no_readable_source(self, samples, corpus_reads) -> None:
        """Question 10's answer exists only inside the encrypted file.

        Its correct output is an empty answer with no citations, and the whole
        basis for that is that the file cannot be read. If this file ever
        became readable, the graded behaviour would change.
        """
        item = next(q for q in samples if q["n"] == 10)

        assert item["unanswerable"] is True
        assert item["expected_answer"] == ""
        assert item["expected_citations"] == []
        assert corpus_reads["vendor/supplier_agreement_ENCRYPTED.pdf"].status is ReadStatus.ENCRYPTED

    def test_no_readable_text_file_contains_the_encrypted_files_value(
        self, corpus_reads
    ) -> None:
        """The value the encrypted agreement holds must not leak from elsewhere.

        Question 10 is graded as a refusal precisely because the number appears
        *only* behind encryption. If it were recoverable from a readable file,
        refusing would be wrong.
        """
        encrypted = corpus_reads["vendor/supplier_agreement_ENCRYPTED.pdf"]
        assert encrypted.text == ""

        # The ciphertext of the real file, as it appears on disk, must not be
        # presented as content by any reader.
        for rel_path, result in corpus_reads.items():
            if result.is_usable:
                assert "6412" not in result.text or rel_path != encrypted.rel_path


class TestDecoysSurviveExtraction:
    """The near-misses are the point of the sample corpus; they must survive intact."""

    def test_the_withdrawn_revision_is_still_present_and_still_wrong(
        self, corpus_reads
    ) -> None:
        assert "105" in corpus_reads["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].text
        assert "94" in corpus_reads["specs/tq40_datasheet_r2.pdf"].text

    def test_the_legacy_fan_row_is_present(self, corpus_reads) -> None:
        """A near-miss row: a fan assembly, but for the TQ-20."""
        text = corpus_reads["support/rma_parts.xlsx"].text

        assert "ORR-FAN-1990-A" in text
        assert "ORR-FAN-2214-B" in text

    def test_the_release_notes_do_not_contain_the_fixed_version(self, corpus_reads) -> None:
        """The decoy: the most release-ish file, and it lacks the answer token."""
        assert "4.3.2" not in corpus_reads["engineering/meridian_release_notes.txt"].text

    def test_the_old_timeout_appears_only_in_the_comment(self, corpus_reads) -> None:
        """Line-level provenance is what separates the decoy from the answer.

        The constant name appears twice in the file -- once as the assignment
        and once as a function's default -- so the assertion targets the
        assignment specifically rather than the name.
        """
        result = corpus_reads["engineering/ingest_service.py"]
        comments = [s for s in result.spans if s.text.startswith("#") and "60" in s.text]
        assignments = [s for s in result.spans if s.text.startswith("DEFAULT_BATCH_TIMEOUT_S")]

        assert len(comments) == 1
        assert len(assignments) == 1
        assert "180" in assignments[0].text
        assert "180" not in comments[0].text

    def test_the_ticket_link_is_present_in_the_log(self, corpus_reads) -> None:
        """The multi-hop route: the log names the ticket, the database holds the fix."""
        assert "ORR-1847" in corpus_reads["logs/prod_inference_2026-09-02.log"].text
        assert "ORR-1847" in corpus_reads["support/bug_database.csv"].text
