"""Tests for the engine seam: finding Tesseract, running it, and reading an image.

The seam is the one part of the vision path that cannot be a pure function, so
it is the part with the least to assert and the most to get wrong. Two things
are tested here that could not otherwise be:

**The command line.** ``recognition_argv`` is separated from ``recognise`` for
exactly this. Tesseract takes the image and the output base as its first two
positional arguments and *then* its options, so an option placed one slot early
is read as the output base -- a run that writes nothing, reports success, and
returns an empty file. Asserting the vector is the only cheap way to catch that.

**A real process, without a real engine.** The stand-in engine is this same
Python interpreter running a short script, passed to the module as a command
*with arguments*. That is why the engine setting is split like a shell word
rather than treated as one path: it needs no executable bit, no compiled
binary, and no Tesseract, so the whole chain from ``find_engine`` through to
``TextSpan``s runs in the suite on any machine.

The words are invented. The sample pinout pairs ``B14`` with ``THERM_ALERT#``,
and a suite built around those two strings would pass against a lookup table
keyed on the filename.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from config import OCR_ENGINE_ENV, OCR_LANGUAGE, OCR_PAGE_SEGMENTATION_MODE
from document_fixtures import png_bytes
from ingestion.readers import image, ocr_engine
from ingestion.readers.ocr_tsv import TSV_HEADER
from models import ReadStatus

_HEADER = "\t".join(TSV_HEADER)


def _row(
    text: str,
    *,
    left: float = 100,
    top: float = 200,
    width: float = 40,
    height: float = 16,
    conf: float = 96.154,
    level: int | float = 5,
) -> str:
    fields = [level, 1, 1, 1, 1, 1, left, top, width, height, conf, text]
    return "\t".join(str(value) for value in fields)


def _tsv(*rows: str) -> str:
    return "\n".join([_HEADER, *rows])


def _stand_in_engine(tmp_path: Path, body: str, *, exit_code: int = 0) -> list[str]:
    """A command prefix that runs ``body`` as a Python program.

    ``sys.executable`` is used because it is certain to exist and to be
    executable, and the script is passed as a file rather than inline so the
    body stays readable. Nothing here needs a chmod: the interpreter is the
    program being run, and the script is only an argument to it.
    """
    script = tmp_path / "stand_in_engine.py"
    script.write_text(
        "import sys\n"
        f"sys.stdout.write({_tsv(*[row for row in body.splitlines()])!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


class TestFindingTheEngine:
    """Absence, which must be a value rather than an error."""

    def test_nothing_configured_and_nothing_installed_means_no_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OCR_ENGINE_ENV, "/nonexistent/tesseract")
        assert ocr_engine.find_engine() is None

    def test_an_empty_setting_falls_back_to_the_name_on_the_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The container installs the engine and sets nothing, so an empty
        # setting has to mean "look for it", not "there is none".
        monkeypatch.setenv(OCR_ENGINE_ENV, "")
        monkeypatch.setenv("PATH", "")
        assert ocr_engine.find_engine() is None

    def test_a_command_that_resolves_is_returned_as_a_vector(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(OCR_ENGINE_ENV, f"{sys.executable} --some-flag")
        found = ocr_engine.find_engine()
        assert found is not None
        assert found[0] == sys.executable
        assert found[1:] == ["--some-flag"]

    def test_an_unbalanced_quote_is_absence_and_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A typo in a shell profile must not abort the indexing pass.
        monkeypatch.setenv(OCR_ENGINE_ENV, "'unclosed")
        assert ocr_engine.find_engine() is None

    def test_a_whitespace_only_setting_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OCR_ENGINE_ENV, "   ")
        monkeypatch.setenv("PATH", "")
        assert ocr_engine.find_engine() is None


class TestTheCommandLine:
    """What is actually executed -- asserted without executing it."""

    def test_the_image_comes_before_the_output_base_and_the_options(self) -> None:
        argv = ocr_engine.recognition_argv(["tesseract"], Path("/corpus/a.png"))
        assert argv[1] == "/corpus/a.png"
        assert argv[2] == "stdout"

    def test_the_configured_segmentation_mode_is_passed(self) -> None:
        argv = ocr_engine.recognition_argv(["tesseract"], Path("/corpus/a.png"))
        assert argv[argv.index("--psm") + 1] == str(OCR_PAGE_SEGMENTATION_MODE)

    def test_the_configured_language_is_passed(self) -> None:
        argv = ocr_engine.recognition_argv(["tesseract"], Path("/corpus/a.png"))
        assert argv[argv.index("-l") + 1] == OCR_LANGUAGE

    def test_tsv_is_requested(self) -> None:
        argv = ocr_engine.recognition_argv(["tesseract"], Path("/corpus/a.png"))
        assert argv[-1] == "tsv"

    def test_the_engine_prefix_is_kept_whole(self) -> None:
        argv = ocr_engine.recognition_argv(["sh", "/opt/wrapper"], Path("/corpus/a.png"))
        assert argv[:2] == ["sh", "/opt/wrapper"]

    def test_a_path_with_shell_metacharacters_stays_one_argument(self) -> None:
        # The path comes from the corpus, which is untrusted. It is an element
        # of a vector, never text spliced into a command line.
        hostile = Path("/corpus/; rm -rf / #.png")
        argv = ocr_engine.recognition_argv(["tesseract"], hostile)
        assert argv[1] == str(hostile)
        assert len(argv) == 8


class TestRunningTheEngine:
    """The process boundary: what comes back, and what is raised."""

    def test_the_bytes_the_engine_wrote_are_returned(self, tmp_path: Path) -> None:
        engine = _stand_in_engine(tmp_path, _row("SENTINEL"))
        raw = ocr_engine.recognise(engine, tmp_path / "unused.png")
        assert b"SENTINEL" in raw

    def test_a_nonzero_exit_is_an_engine_failure(self, tmp_path: Path) -> None:
        engine = _stand_in_engine(tmp_path, _row("SENTINEL"), exit_code=1)
        with pytest.raises(ocr_engine.EngineFailed):
            ocr_engine.recognise(engine, tmp_path / "unused.png")

    def test_a_command_that_cannot_be_started_is_an_engine_failure(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ocr_engine.EngineFailed):
            ocr_engine.recognise(["/nonexistent/program"], tmp_path / "unused.png")

    def test_a_hang_is_killed_rather_than_allowed_to_run(self, tmp_path: Path) -> None:
        """A wedged child would otherwise eat the whole run's time budget."""
        script = tmp_path / "hang.py"
        script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        with pytest.raises(ocr_engine.EngineFailed, match="did not finish"):
            ocr_engine.recognise([sys.executable, str(script)], tmp_path / "u.png", timeout=0.5)

    def test_the_failure_says_which_kind_it_was(self, tmp_path: Path) -> None:
        # A missing shared library and a corrupt image are fixed by different
        # people, so the message has to distinguish them.
        engine = _stand_in_engine(tmp_path, _row("SENTINEL"), exit_code=3)
        with pytest.raises(ocr_engine.EngineFailed, match="exited 3"):
            ocr_engine.recognise(engine, tmp_path / "unused.png")


class TestReadingAnImage:
    """The whole chain, from a file on disk to spans, with a stand-in engine."""

    @pytest.fixture
    def corpus(self, build_corpus, tmp_path: Path):
        """Build a corpus whose image is a real PNG, and wire up the engine."""

        def build(tsv: str, *, width: int = 1400, height: int = 900):
            files = build_corpus({"specs/pinout.png": png_bytes(width=width, height=height)})
            engine = _stand_in_engine(tmp_path, tsv)
            return files["specs/pinout.png"], engine

        return build

    def test_a_recognised_image_yields_spans(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discovered, engine = corpus(_row("SENTINEL"))
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert result.status is ReadStatus.OK
        assert result.is_usable is True
        assert [span.text for span in result.spans] == ["SENTINEL"]

    def test_the_span_carries_where_on_the_image_it_came_from(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discovered, engine = corpus(_row("SENTINEL", left=63, top=150))
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert result.spans[0].location == "text line 1 at (63, 150)"

    def test_the_dimensions_are_not_emitted_as_text(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Metadata is not corpus content and must not become searchable.

        ``1400`` is a fact about a file, not something the corpus says. Putting
        it in a span would let a question it has nothing to do with match this
        file and cite it -- and every extra citation scores zero. This is the
        version of the claim that can actually fail: it needs a read that
        *succeeded*, since a reader that returned nothing at all would satisfy
        it for the wrong reason.
        """
        discovered, engine = corpus(_row("SENTINEL"), width=1400, height=900)
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert result.is_usable is True
        assert "1400" not in result.text
        assert "900" not in result.text

    def test_a_pairing_becomes_a_span_of_its_own(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The relationship is the finding, and it has to reach the index.

        Neither half answers the question alone: a designator with no signal
        and a signal with no pin are both useless, and the fact that they go
        together appears nowhere in the image as text.
        """
        tsv = _tsv(
            _row("P3", left=150, top=150, width=40, height=16),
            _row("TDI", left=140, top=222, width=60, height=10),
        )
        discovered, engine = corpus(tsv)
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert "P3 | TDI" in [span.text for span in result.spans]

    def test_the_relationship_records_both_lines_it_came_from(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reader checking the pairing needs to be able to see the two boxes
        # it was inferred from, not just the joined text.
        tsv = _tsv(
            _row("P3", left=150, top=150, width=40, height=16),
            _row("TDI", left=140, top=222, width=60, height=10),
        )
        discovered, engine = corpus(tsv)
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        pairing = next(s for s in result.spans if "|" in s.text)
        assert "text lines" in pairing.location
        assert "region" in pairing.location

    def test_a_blank_image_is_empty_rather_than_unreadable(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A photograph with no words is a legitimate result, not a failure."""
        discovered, engine = corpus("")
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert result.status is ReadStatus.EMPTY
        assert result.spans == ()

    def test_an_engine_that_returns_no_tsv_is_unreadable_not_empty(
        self, corpus, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The distinction ``header_recognised`` exists for.

        An error message on standard output is not a blank image, and treating
        it as one would let a broken invocation pass for a photograph of
        nothing.
        """
        script = tmp_path / "erroring_engine.py"
        script.write_text(
            "import sys\nsys.stdout.write('Error, cannot read input file')\n",
            encoding="utf-8",
        )
        discovered, _ = corpus(_row("SENTINEL"))
        monkeypatch.setenv(OCR_ENGINE_ENV, f"{sys.executable} {script}")

        result = image.read(discovered)

        assert result.status is ReadStatus.UNREADABLE
        assert "no TSV" in result.detail

    def test_an_engine_failure_is_unreadable_and_says_why(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discovered, engine = corpus(_row("SENTINEL"))
        failing = _stand_in_engine(Path(engine[1]).parent, _row("X"), exit_code=2)
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(failing))

        result = image.read(discovered)

        assert result.status is ReadStatus.UNREADABLE
        assert "exited 2" in result.detail

    def test_the_detail_counts_what_was_found(
        self, corpus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tsv = _tsv(
            _row("P3", left=150, top=150, width=40, height=16),
            _row("TDI", left=140, top=222, width=60, height=10),
        )
        discovered, engine = corpus(tsv)
        monkeypatch.setenv(OCR_ENGINE_ENV, " ".join(engine))

        result = image.read(discovered)

        assert "2 text lines" in result.detail
        assert "1 associations" in result.detail
