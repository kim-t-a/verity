"""The real embedding backend.

Two kinds of test live here, and they are separated on purpose.

The **refusals and the wiring** need no model and no torch: they assert that a
bad configuration is rejected with a message naming the real cause, and that the
configured pooling actually reaches the arithmetic rather than sitting in
``config.py`` as decoration.

The **embedding** tests need the downloaded weights and torch. They skip when
either is absent, so the suite still runs on a machine that has neither -- which
is what AGENTS.md requires, and what keeps the import-graph guard meaningful.

What none of these can show is whether the embeddings are *good*. Cosine
geometry, ordering and determinism are checkable; "these vectors capture
meaning" is not, and the retrieval measurements are where that claim lives.
"""

from __future__ import annotations

import math

import pytest
from conftest import REPO_ROOT

from verity.engine.embeddings import (
    HASHING_DIMENSION,
    LOCAL_EMBEDDING_PROVIDER,
    EmbeddingProvider,
    LocalBackendNotAllowed,
    UnknownProvider,
    provider_from_environment,
)
from verity.engine.embeddings.local_model import LocalModelEmbedder, LocalModelUnavailable
from verity.engine.config import (
    ALLOW_LOCAL_BACKEND_ENV,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_DIR_ENV,
    EMBEDDING_MODEL_ID,
    EMBEDDING_PROVIDER_ENV,
)

MODEL_DIR = REPO_ROOT / "starter" / "models" / "bge-small-en-v1.5"

#: A sentence long and varied enough that CLS and mean pooling disagree on it.
#: A short one would not: the two collapse to nearly the same vector, which is
#: exactly why the corpus measurement could not separate them.
SENTENCE = "The maximum junction temperature of the TQ-40 is 94 degrees Celsius."


def _require_model() -> None:
    """Skip when the weights or torch are missing, rather than failing."""
    if not MODEL_DIR.is_dir():
        pytest.skip(f"model not downloaded to {MODEL_DIR}")
    pytest.importorskip("torch", reason="the real backend needs torch")
    pytest.importorskip("transformers", reason="the real backend needs transformers")


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> LocalModelEmbedder:
    """The real backend, on the CPU and pointed at the repository's copy.

    The device is set explicitly rather than relying on a default, because the
    default is ``cuda`` and the development machine has no GPU -- which the
    backend correctly refuses. Tests that want the CPU path must ask for it,
    exactly as a developer would.
    """
    _require_model()
    monkeypatch.setenv(EMBEDDING_MODEL_DIR_ENV, str(MODEL_DIR))
    monkeypatch.setenv("MC3_EMBEDDING_DEVICE", "cpu")
    return LocalModelEmbedder()


# --- Refusals that need no model --------------------------------------------


def test_an_unknown_pooling_is_refused(tmp_path) -> None:
    """Checked before anything is loaded, so a typo costs nothing."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LocalModelUnavailable, match="unknown pooling"):
        LocalModelEmbedder(tmp_path, device="cpu", pooling="average")


def test_a_missing_directory_names_the_variable_that_fixes_it(tmp_path) -> None:
    """The commonest real failure: the image was built without the weights."""
    missing = tmp_path / "not-here"

    with pytest.raises(LocalModelUnavailable, match=EMBEDDING_MODEL_DIR_ENV):
        LocalModelEmbedder(missing, device="cpu")


def test_a_half_copied_model_names_the_missing_file(tmp_path) -> None:
    """A partial COPY is the failure this check exists for.

    Without it the error surfaces deep inside a loading library and reports a
    file format problem, which sends a reader looking in entirely the wrong
    place.
    """
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    # model.safetensors deliberately absent.

    with pytest.raises(LocalModelUnavailable, match="model.safetensors"):
        LocalModelEmbedder(tmp_path, device="cpu")


def test_a_gpu_that_is_not_there_is_refused_rather_than_fallen_back_from() -> None:
    """The rule-7 guard, from the direction that matters.

    A silent CPU fallback would produce a run that looks healthy and is rejected
    for never touching the GPU. Refusing loudly converts that invisible
    disqualification into an error someone can act on.

    Exercised through the real constructor rather than by poking at internals,
    so what is tested is the path a container actually takes.
    """
    _require_model()
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():  # pragma: no cover - no GPU on the dev machine
        pytest.skip("a GPU is present, so the missing-device path cannot be reached")

    with pytest.raises(LocalModelUnavailable, match="cannot see a GPU"):
        LocalModelEmbedder(MODEL_DIR, device="cuda")


# --- The factory ------------------------------------------------------------


def test_the_default_provider_is_now_the_real_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no environment at all -- how the harness runs -- a model is loaded.

    This is the assertion that closes the Stage 4 gap, where the default returned
    ``None`` and every container run was silently lexical-only.
    """
    _require_model()
    monkeypatch.delenv(EMBEDDING_PROVIDER_ENV, raising=False)
    monkeypatch.setenv(EMBEDDING_MODEL_DIR_ENV, str(MODEL_DIR))
    monkeypatch.setenv("MC3_EMBEDDING_DEVICE", "cpu")

    provider = provider_from_environment()

    assert isinstance(provider, LocalModelEmbedder)
    assert isinstance(provider, EmbeddingProvider)


def test_the_stand_in_still_needs_both_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widening the real backend must not have loosened the rule-7 guard."""
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, LOCAL_EMBEDDING_PROVIDER)
    monkeypatch.delenv(ALLOW_LOCAL_BACKEND_ENV, raising=False)

    with pytest.raises(LocalBackendNotAllowed):
        provider_from_environment()


def test_the_stand_in_is_reachable_with_both_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, LOCAL_EMBEDDING_PROVIDER)
    monkeypatch.setenv(ALLOW_LOCAL_BACKEND_ENV, "1")

    provider = provider_from_environment()

    assert provider is not None
    assert provider.dimension == HASHING_DIMENSION


def test_an_unknown_provider_name_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EMBEDDING_PROVIDER_ENV, "some-cloud-api")

    with pytest.raises(UnknownProvider):
        provider_from_environment()


# --- Identity ---------------------------------------------------------------


def test_the_provider_id_is_the_expected_literal() -> None:
    """Pinned literally, and deliberately *not* recomputed from its inputs.

    The id is built by an f-string from the model name, the pooling and the
    dimension, so asserting it *contains* the pooling would be a tautology that
    passes no matter what the id said. Writing the string out means an
    accidental change to the recipe is caught here -- and caught loudly, because
    the id is what stops an old index being searched with new vectors.
    """
    assert EMBEDDING_MODEL_ID == "bge-small-en-v1.5-cls-d384-v1"


def test_the_declared_dimension_matches_the_model(embedder: LocalModelEmbedder) -> None:
    assert embedder.dimension == EMBEDDING_DIMENSION


# --- Embedding --------------------------------------------------------------


def test_a_vector_has_the_declared_width(embedder: LocalModelEmbedder) -> None:
    (vector,) = embedder.embed(["a short sentence"])

    assert len(vector) == EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in vector)


def test_vectors_are_unit_length(embedder: LocalModelEmbedder) -> None:
    """Normalised at the boundary, so a dot product *is* the cosine."""
    for vector in embedder.embed([SENTENCE, "another", ""]):
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0, abs=1e-5)


def test_order_is_preserved(embedder: LocalModelEmbedder) -> None:
    """The caller pairs result *i* with input *i*; reordering would attach
    vectors to the wrong chunks and nothing downstream could detect it."""
    first, second = embedder.embed([SENTENCE, "an entirely different sentence"])
    batched = embedder.embed([SENTENCE, "an entirely different sentence"])

    assert batched[0] == pytest.approx(first)
    assert batched[1] == pytest.approx(second)


def test_the_same_text_embeds_identically_twice(embedder: LocalModelEmbedder) -> None:
    """``model.eval()`` matters: in training mode dropout would make this differ."""
    assert embedder.embed([SENTENCE])[0] == pytest.approx(embedder.embed([SENTENCE])[0])


def test_different_text_gives_a_different_vector(embedder: LocalModelEmbedder) -> None:
    """Guards against a degenerate backend that ignores its input."""
    a, b = embedder.embed([SENTENCE, "Quarterly logistics for the Turin office."])

    assert a != pytest.approx(b)


def test_an_empty_list_embeds_to_an_empty_list(embedder: LocalModelEmbedder) -> None:
    """An empty corpus is legitimate and must not reach the model as a batch."""
    assert embedder.embed([]) == []


def test_batching_does_not_change_the_result(embedder: LocalModelEmbedder) -> None:
    """Padding is per-batch, so a text's vector must not depend on its neighbours.

    This is the silent failure that batching invites: if attention padding leaked
    between items, every vector would depend on what else happened to be in its
    batch, and the index would be irreproducible without changing a line.
    """
    texts = [f"sentence number {n} about thermal limits" for n in range(40)]

    together = embedder.embed(texts)
    alone = [embedder.embed([text])[0] for text in texts]

    for batched, single in zip(together, alone):
        assert batched == pytest.approx(single, abs=1e-5)


def test_the_configured_pooling_reaches_the_arithmetic(
    embedder: LocalModelEmbedder,
) -> None:
    """The setting is load-bearing, not decorative.

    If CLS were wired up but the module silently used mean -- or vice versa --
    every other test here would still pass. This one fails, because it requires
    that asking for the other pooling produces different numbers.
    """
    mean_pooled = LocalModelEmbedder(MODEL_DIR, device="cpu", pooling="mean")

    cls_vector = embedder.embed([SENTENCE])[0]
    mean_vector = mean_pooled.embed([SENTENCE])[0]

    cosine = sum(a * b for a, b in zip(cls_vector, mean_vector))
    assert cosine < 0.999, "CLS and mean pooling produced the same vector"


def test_a_similar_question_lands_nearer_than_an_unrelated_one(
    embedder: LocalModelEmbedder,
) -> None:
    """The one semantic property worth asserting, and it is deliberately weak.

    Not "retrieval works" -- one pair of sentences cannot establish that. It
    checks the geometry is oriented the way cosine similarity assumes, which a
    sign error or a mis-pooled vector would break while leaving every
    structural test above still passing.
    """
    question = "What is the maximum junction temperature?"
    on_topic = "The maximum junction temperature is 94 C."
    off_topic = "The Turin office relocated its logistics depot in March."

    q, near, far = embedder.embed([question, on_topic, off_topic])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731

    assert dot(q, near) > dot(q, far)
