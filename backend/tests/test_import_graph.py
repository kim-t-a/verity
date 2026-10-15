"""Guard the import graph that runs inside the container.

**Why this test exists.** The Dockerfile copies ``starter/app/`` to ``/app/``, and
the harness runs ``python3 /app/app.py``. That places ``/app`` on ``sys.path``, so
the application's modules import each other by *bare name* -- ``import config``,
``from ingestion import walker`` -- and there is no ``starter`` package at runtime.

A local test suite can easily hide a break here. If the tests imported the code as
``starter.app.ingestion.walker``, everything would pass locally while the container
died on the first ``import config``. These tests import through the same path the
container uses, so that failure mode shows up now rather than on the notebook.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from conftest import APP_DIR


def test_app_dir_is_on_sys_path() -> None:
    """The container relies on this; the test suite must reproduce it."""
    assert str(APP_DIR) in sys.path


@pytest.mark.parametrize(
    "module_name",
    [
        "config",
        "models",
        "normalisation",
        "tokenisation",
        "chunking",
        "chunkstore",
        "embeddings",
        "embeddings.hashing",
        "embeddings.local_model",
        "retrieval",
        "retrieval.hybrid",
        "retrieval.keyword",
        "retrieval.vector",
        "answering",
        "answering.candidates",
        "answering.citations",
        "answering.extract",
        "answering.refusal",
        "answering.shapes",
        "answering.supersession",
        "app",
        "ingestion",
        "ingestion.filetypes",
        "ingestion.walker",
        "ingestion.readers",
        "ingestion.readers.image",
        "ingestion.readers.image_geometry",
        "ingestion.readers.ocr_engine",
        "ingestion.readers.ocr_tsv",
        "ingestion.readers.pdf",
    ],
)
def test_module_imports_by_bare_name(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_no_module_imports_through_a_starter_package_prefix() -> None:
    """Nothing may depend on a ``starter`` package that does not exist at runtime."""
    sources = sorted(APP_DIR.rglob("*.py"))

    offenders: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        if "import starter" in text or "from starter" in text:
            offenders.append(str(source.relative_to(APP_DIR)))

    assert offenders == [], f"modules importing via a 'starter.' prefix: {offenders}"


def test_cli_entry_point_keeps_its_contract() -> None:
    """The three names the harness depends on must survive every refactor.

    The contract is the command line, the output path and the JSON keys -- not
    the internal structure. This asserts the outward surface only.
    """
    app = importlib.import_module("app")

    assert callable(app.index)
    assert callable(app.answer)
    assert callable(app.main)

    # answer() must return the documented triple even while it is still a stub.
    result = app.answer(Path("/nonexistent"), "any question")
    assert isinstance(result, tuple) and len(result) == 3
    answer_text, citations, confidence = result
    assert isinstance(answer_text, str)
    assert isinstance(citations, list)
    assert isinstance(confidence, float)


def test_output_and_index_dirs_are_overridable_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness sets nothing, but local runs and tests need to redirect these."""
    monkeypatch.setenv("MC3_INDEX_DIR", "/tmp/mc3-test-index")
    monkeypatch.setenv("MC2_OUTPUT_DIR", "/tmp/mc3-test-output")

    app = importlib.reload(importlib.import_module("app"))

    assert app.INDEX_DIR == Path("/tmp/mc3-test-index")
    assert app.OUTPUT_DIR == Path("/tmp/mc3-test-output")

    # Restore the module to its import-time state so later tests are unaffected.
    monkeypatch.undo()
    importlib.reload(app)
