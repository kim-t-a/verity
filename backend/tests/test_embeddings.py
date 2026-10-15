"""Choosing an embedding backend -- and the rule-7 guard around the CPU one.

The challenge rejects a run that never touches the GPU, so the local stand-in
must be *impossible to reach by accident*. "Impossible" is a claim about the
code, so it gets a test rather than a comment: these assert the guard fires, and
that the default is the real backend.

The cross-process test at the bottom is the other half. A stand-in embedder that
produced different vectors in ``--index`` and in each ``--query`` process would
not raise, not warn, and not look broken -- it would simply make every vector
comparison meaningless. That is exactly the silent-wrong-output shape this
project has already been bitten by three times, so it is tested directly.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import APP_DIR

from verity.engine.config import (
    ALLOW_LOCAL_BACKEND_ENV,
    EMBEDDING_PROVIDER_ENV,
    HASHING_DIMENSION,
)
from verity.engine.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalBackendNotAllowed,
    UnknownProvider,
    provider_from_environment,
)
from verity.engine.embeddings.hashing import HashingEmbedder


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No embedding environment at all -- which is how the harness runs us."""
    monkeypatch.delenv(EMBEDDING_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(ALLOW_LOCAL_BACKEND_ENV, raising=False)


# --- The rule-7 guard -------------------------------------------------------


def test_a_bare_environment_never_selects_the_local_backend(clean_environment: None) -> None:
    """The whole safety property: no environment must not mean "use the CPU".

    Stage 2 could assert this by requiring ``None``, because no GPU backend
    existed. Now one does, so the assertion has to be written as the property
    rather than as the absence: a bare environment reaches the *real* backend,
    which may legitimately fail to start -- missing weights, no GPU -- but must
    never quietly resolve to the hashing stand-in.

    Both branches are accepted, and the second is why this stays meaningful:
    if the default were ever changed to the stand-in, ``provider_from_environment``
    would raise ``LocalBackendNotAllowed`` (its second switch is unset here),
    which is not caught below, so the test fails loudly rather than passing.
    """
    try:
        provider = provider_from_environment()
    except EmbeddingError:
        # The real backend was chosen and could not start. That is a fault to
        # fix, not a silent fallback -- and emphatically not the stand-in.
        provider = None

    assert not isinstance(provider, HashingEmbedder)


def test_naming_the_local_backend_alone_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """One stray environment variable must not be enough to reach the CPU path."""
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, "hashing")
    monkeypatch.delenv(ALLOW_LOCAL_BACKEND_ENV, raising=False)

    with pytest.raises(LocalBackendNotAllowed) as raised:
        provider_from_environment()

    # The message has to name both variables: this error is read by whoever
    # finds a CPU-only run rejected, and "not allowed" without the fix is a
    # dead end.
    assert ALLOW_LOCAL_BACKEND_ENV in str(raised.value)
    assert EMBEDDING_PROVIDER_ENV in str(raised.value)


@pytest.mark.parametrize("second_switch", ["", "0", "true", "yes", "2"])
def test_only_the_exact_opt_in_value_works(
    monkeypatch: pytest.MonkeyPatch, second_switch: str
) -> None:
    """Anything other than ``1`` is not consent.

    ``"true"`` and ``"yes"`` are the values a person would reach for next, so
    they are the ones worth pinning: a truthiness check would accept them and
    quietly widen the guard.
    """
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, "hashing")
    monkeypatch.setenv(ALLOW_LOCAL_BACKEND_ENV, second_switch)

    with pytest.raises(LocalBackendNotAllowed):
        provider_from_environment()


def test_both_switches_together_reach_the_local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, "hashing")
    monkeypatch.setenv(ALLOW_LOCAL_BACKEND_ENV, "1")

    provider = provider_from_environment()

    assert isinstance(provider, HashingEmbedder)
    assert provider.dimension == HASHING_DIMENSION


def test_an_unknown_backend_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typos must fail loudly rather than fall back to something usable.

    A silent fallback here is the dangerous case: a misspelt provider would
    build an index with the wrong vectors and never say so.
    """
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, "cuda")
    monkeypatch.setenv(ALLOW_LOCAL_BACKEND_ENV, "1")

    with pytest.raises(UnknownProvider) as raised:
        provider_from_environment()

    assert "cuda" in str(raised.value)


# --- The provider contract --------------------------------------------------


def test_hashing_embedder_satisfies_the_protocol() -> None:
    assert isinstance(HashingEmbedder(dimension=64), EmbeddingProvider)


def test_model_id_changes_with_dimension() -> None:
    """The id is what the manifest records, so it must distinguish vector sets.

    Two embedders that produce incomparable vectors must not share an id --
    otherwise an index built at one dimension would look compatible with a
    query embedder at another, and every score would be nonsense.
    """
    assert HashingEmbedder(dimension=64).model_id != HashingEmbedder(dimension=256).model_id


def test_vectors_are_unit_length() -> None:
    """L2-normalised, so cosine similarity is a plain dot product."""
    vector = HashingEmbedder(dimension=128).embed_one("batch timeout 180 seconds")

    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0)


def test_a_zero_vector_is_returned_rather_than_dividing_by_zero() -> None:
    """Text with no tokens is a real input: punctuation-only chunks exist."""
    vector = HashingEmbedder(dimension=16).embed_one("-- ... ---")

    assert len(vector) == 16
    assert all(value == 0.0 for value in vector)


def test_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)


def test_embed_returns_one_vector_per_input_in_order() -> None:
    """Order is the contract: result *i* is the vector for input *i*.

    A backend that reordered would attach vectors to the wrong chunks, and the
    resulting citations would look plausible and be false.
    """
    embedder = HashingEmbedder(dimension=64)
    texts = ["alpha beta", "gamma", "delta epsilon zeta"]

    vectors = embedder.embed(texts)

    assert len(vectors) == 3
    for index, text in enumerate(texts):
        assert vectors[index] == embedder.embed_one(text)


def test_similar_text_scores_higher_than_unrelated_text() -> None:
    """The stand-in has to be *usable* for developing retrieval, not just stable.

    Without this, a broken similarity could survive every plumbing test and only
    show up as inexplicable retrieval results in Stage 3.
    """
    embedder = HashingEmbedder(dimension=256)
    query = embedder.embed_one("maximum junction temperature")

    related = embedder.embed_one("the maximum junction temperature is 94 C")
    unrelated = embedder.embed_one("quarterly revenue recognition policy")

    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(query, related) > dot(query, unrelated)


# --- Cross-process determinism ----------------------------------------------

#: Run in a fresh interpreter; prints the vector for a fixed string as JSON.
#: ``sys.path`` is set from the argument rather than inherited, so the probe
#: imports the application exactly the way the container does.
_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from verity.engine.embeddings.hashing import HashingEmbedder

vector = HashingEmbedder(dimension=32).embed_one("junction temperature 94")
print(json.dumps(vector))
"""


@pytest.mark.parametrize("python_hash_seed", ["0", "1", "12345"])
def test_vectors_are_identical_across_processes(python_hash_seed: str) -> None:
    """The trap this test exists for: ``hash()`` is randomised per process.

    Python salts string hashing per interpreter, so an embedder built on the
    built-in ``hash()`` would produce *different* vectors in the ``--index``
    process and in each of the ten ``--query`` processes. Nothing would raise.
    The vectors would simply be unrelated, and retrieval would return results
    that look arbitrary -- the same silent-wrong-output shape as the PDF ``Tj``
    regex and the JPEG offsets.

    The seeded run is compared against a fixed baseline run, so the pair differs
    in ``PYTHONHASHSEED`` and nothing else. With ``hash()`` these two disagree;
    with ``blake2b`` they must not.
    """
    baseline = _embed_in_subprocess("0")
    seeded = _embed_in_subprocess(python_hash_seed)

    assert seeded == baseline, (
        f"vectors differ between PYTHONHASHSEED={python_hash_seed} and "
        "PYTHONHASHSEED=0; the embedder is using the randomised built-in hash()"
    )

    # And the value is a real vector, not two runs agreeing on an empty list.
    vector = json.loads(baseline)
    assert len(vector) == 32
    assert any(value != 0.0 for value in vector)


def _embed_in_subprocess(python_hash_seed: str) -> str:
    """The probe's stdout, run in a fresh interpreter under a given hash seed."""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE, str(APP_DIR)],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": python_hash_seed, "PATH": "/usr/bin:/bin"},
    )
    return completed.stdout.strip()
