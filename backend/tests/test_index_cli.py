"""The command line the harness actually runs, as a subprocess.

Every other test in this suite imports the application and calls functions. This
one does not: it runs ``python3 app.py --index ...`` and ``python3 app.py
--corpus ... --query-id ... --query ...`` as separate processes, exactly as the
harness does, and then reads the JSON off disk.

That distinction is the point. The harness starts a **new process** for each of
the ten questions, so the only thing that crosses from indexing to answering is
what ``--index`` wrote to disk. A suite that called ``index()`` and then
``answer()`` in one interpreter would pass while the container failed, because
it would keep the in-memory state the real run loses.

The output contract is checked on every run, including the ones that go wrong: a
response missing ``answer`` or ``citations`` is malformed and scores zero, on
purpose, so a crashed writer and a considered refusal cannot look alike.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import APP_DIR, CORPUS_DIR

APP = APP_DIR / "app.py"

REQUIRED_KEYS = {"answer", "citations", "confidence"}


def run_cli(*arguments: str, index_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a fresh interpreter with its directories redirected.

    ``MC3_INDEX_DIR`` and ``MC2_OUTPUT_DIR`` exist so this is possible at all:
    the defaults are ``/app/index`` and ``/app/output``, which do not exist on a
    development machine.

    **No ``-I``.** Isolated mode implies ``-P``, which drops the *script's own
    directory* from ``sys.path`` -- and because the Dockerfile copies
    ``starter/app/`` to ``/app/``, the application imports its own modules by
    bare name (``import config``). Running the app under ``-I`` therefore fails
    with ``ModuleNotFoundError: No module named 'chunking'`` before it reads
    anything. The harness runs ``python3 /app/app.py``, so the test must too.

    The environment is replaced wholesale rather than inherited, which is the
    isolation ``-I`` was being reached for: nothing from the developer's shell
    reaches the child, and the application is invoked by absolute path.
    """
    return subprocess.run(
        [sys.executable, str(APP), *arguments],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "MC3_INDEX_DIR": str(index_dir),
            "MC2_OUTPUT_DIR": str(output_dir),
        },
    )


@pytest.fixture
def directories(tmp_path: Path) -> tuple[Path, Path]:
    index_dir = tmp_path / "index"
    output_dir = tmp_path / "output"
    return index_dir, output_dir


@pytest.fixture
def sample_corpus() -> Path:
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"sample corpus not present at {CORPUS_DIR}")
    return CORPUS_DIR


def read_output(output_dir: Path, query_id: str) -> dict[str, object]:
    path = output_dir / f"{query_id}_output.json"
    assert path.is_file(), f"the harness reads {path}; it was not written"
    return json.loads(path.read_text(encoding="utf-8"))


# --- The interface ----------------------------------------------------------


def test_help_still_documents_both_entry_points() -> None:
    """Both invocations come from the starter's contract and may not be renamed."""
    completed = subprocess.run(
        [sys.executable, str(APP), "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    for flag in ("--index", "--corpus", "--query-id", "--query"):
        assert flag in completed.stdout


def test_indexing_the_sample_corpus_succeeds(directories, sample_corpus: Path) -> None:
    index_dir, output_dir = directories

    completed = run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    assert completed.returncode == 0, completed.stderr
    assert (index_dir / "manifest.json").is_file()
    assert (index_dir / "chunks.jsonl").is_file()


def test_indexing_writes_nothing_to_the_output_directory(directories, sample_corpus: Path) -> None:
    """``--index`` produces an index, not an answer.

    The harness charges this pass to the startup budget and reads no JSON from
    it, so a file appearing here would mean the two entry points have been
    conflated.
    """
    index_dir, output_dir = directories

    run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_indexing_reports_what_it_found_on_stderr(directories, sample_corpus: Path) -> None:
    """The index pass is otherwise invisible in the graded result.

    If a format silently stops being read, the only symptom is later questions
    answering wrongly, so the pass states its own numbers.
    """
    index_dir, output_dir = directories

    completed = run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    assert "indexed" in completed.stderr
    assert "files" in completed.stderr
    assert "chunks" in completed.stderr
    # And the report goes to stderr, so stdout stays free of anything that
    # could be mistaken for program output.
    assert completed.stdout == ""


def test_the_index_records_the_files_it_could_not_read(directories, sample_corpus: Path) -> None:
    """The corpus's hazards have to be *reported*, not merely survived.

    The encrypted PDF and the unknown binary are expected here; the point of the
    assertion is that they are named rather than quietly absent.
    """
    index_dir, output_dir = directories

    completed = run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    status_counts = manifest["status_counts"]

    assert status_counts.get("ok", 0) > 0
    assert "encrypted" in status_counts
    assert "unsupported" in status_counts
    assert "not indexed" in completed.stderr


# --- The output contract ----------------------------------------------------


def test_a_query_writes_the_three_required_keys(directories, sample_corpus: Path) -> None:
    index_dir, output_dir = directories
    run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    completed = run_cli(
        "--corpus",
        str(sample_corpus),
        "--query-id",
        "query_01",
        "--query",
        "What is the maximum junction temperature?",
        index_dir=index_dir,
        output_dir=output_dir,
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_output(output_dir, "query_01")
    assert set(payload) == REQUIRED_KEYS
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["confidence"], float)


def test_the_output_filename_uses_the_given_query_id(directories, sample_corpus: Path) -> None:
    """The id is GIVEN, never derived from the question text.

    A question has no filename to take a stem from, so the stem comes off the
    command line -- and for a query id that appears nowhere in the question.
    """
    index_dir, output_dir = directories
    run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    run_cli(
        "--corpus",
        str(sample_corpus),
        "--query-id",
        "query_07",
        "--query",
        "What is the maximum junction temperature?",
        index_dir=index_dir,
        output_dir=output_dir,
    )

    assert (output_dir / "query_07_output.json").is_file()


def test_the_contract_holds_when_there_is_no_index(directories, sample_corpus: Path) -> None:
    """A missing index is an error path, and the contract still governs it.

    ``--query`` is run without ``--index`` having been run first. The writer must
    still emit a well-formed file: a crashed writer and a considered refusal
    must not look the same, so an error is reported as an empty answer with an
    empty citation list rather than as a stack trace and no file.
    """
    index_dir, output_dir = directories

    completed = run_cli(
        "--corpus",
        str(sample_corpus),
        "--query-id",
        "query_02",
        "--query",
        "anything at all",
        index_dir=index_dir,
        output_dir=output_dir,
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_output(output_dir, "query_02")
    assert set(payload) == REQUIRED_KEYS
    assert payload["answer"] == ""
    assert payload["citations"] == []


def test_a_stale_index_is_refused_rather_than_answered_from(
    directories, sample_corpus: Path, tmp_path: Path
) -> None:
    """A corpus that changed since indexing must not be answered from the old one.

    The answers would be drawn from text that is not in the corpus being asked
    about, and they would come with citations that look right.
    """
    index_dir, output_dir = directories
    run_cli("--index", str(sample_corpus), index_dir=index_dir, output_dir=output_dir)

    # A different corpus at a different path, so the fingerprints cannot match.
    other = tmp_path / "other-corpus"
    other.mkdir()
    (other / "notes.txt").write_text("nothing relevant\n", encoding="utf-8")

    completed = run_cli(
        "--corpus",
        str(other),
        "--query-id",
        "query_03",
        "--query",
        "anything at all",
        index_dir=index_dir,
        output_dir=output_dir,
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_output(output_dir, "query_03")
    assert set(payload) == REQUIRED_KEYS
    assert payload["answer"] == ""
    assert "--index" in completed.stderr


def test_a_query_without_a_query_id_is_a_usage_error(directories, sample_corpus: Path) -> None:
    """Without an id there is no filename to write, so failing loudly is correct."""
    index_dir, output_dir = directories

    completed = run_cli(
        "--corpus", str(sample_corpus), "--query", "anything", index_dir=index_dir, output_dir=output_dir
    )

    assert completed.returncode != 0


def test_indexing_the_hazard_corpus_does_not_abort(directories, hazard_corpus, tmp_path: Path) -> None:
    """One bad file must cost that file and nothing else.

    The hazard corpus puts an unreadable file that sorts *first* within its
    directory, so a walk that raises on it would lose every file after it --
    and which files are lost would depend on filename order.
    """
    index_dir, output_dir = directories

    completed = run_cli("--index", str(hazard_corpus.root), index_dir=index_dir, output_dir=output_dir)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    # Every file was accounted for, whether read or refused.
    counted = sum(manifest["status_counts"].values())
    assert counted == manifest["file_count"]
