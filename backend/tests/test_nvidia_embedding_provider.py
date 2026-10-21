"""Tests for the NVIDIA embedding provider.

These tests never call the real NVIDIA service. They use a hand-written fake
that satisfies the same :class:`~verity.engine.embedding_provider.EmbeddingProvider`
Protocol, and a fake HTTP handler for the success path. The real provider is
verified against NVIDIA's endpoint only when you set ``NVIDIA_API_KEY`` and
run against the live service -- that is your call, and it belongs to Phase 0
evaluation, not to the regression suite.
"""

from __future__ import annotations

import json
from typing import Sequence
from unittest.mock import patch

import pytest

from verity.engine.embedding_provider import EmbeddingProvider
from verity.engine.nvidia_embedding_provider import (
    NvidiaEmbeddingError,
    NvidiaEmbeddingProvider,
)


class FakeProvider:
    """Deterministic fake: returns vectors the test chose.

    Satisfies the same Protocol as NvidiaEmbeddingProvider, so retrieval and
    hybrid code can be tested without any network or credentials.
    """

    def __init__(self, vectors: dict[str, list[float]], model_id: str = "fake-v1") -> None:
        self._vectors = vectors
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return len(next(iter(self._vectors.values())))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


def test_provider_satisfies_the_protocol() -> None:
    """The NVIDIA provider goes through the same door as every other backend."""
    provider = NvidiaEmbeddingProvider(api_key="test-key", base_url="https://example.com")
    assert isinstance(provider, EmbeddingProvider)


def test_model_id_and_dimension_are_reported() -> None:
    """model_id and dimension are contract -- retrieval validates against them."""
    provider = NvidiaEmbeddingProvider(
        api_key="k", base_url="https://example.com", model_id="my-model", dimension=128
    )
    assert provider.model_id == "my-model"
    assert provider.dimension == 128


def test_missing_api_key_raises_instead_of_fabricating() -> None:
    """No key means every call refuses, never guesses."""
    provider = NvidiaEmbeddingProvider(api_key="", base_url="https://example.com")
    with pytest.raises(NvidiaEmbeddingError, match="NVIDIA_API_KEY is not set"):
        provider.embed(["hello world"])


def test_missing_base_url_raises() -> None:
    """No endpoint means every call refuses, never guesses."""
    provider = NvidiaEmbeddingProvider(api_key="k", base_url="")
    with pytest.raises(NvidiaEmbeddingError, match="NVIDIA_API_BASE_URL is not set"):
        provider.embed(["hello world"])


def test_bad_http_response_raises() -> None:
    """A non-2xx from NVIDIA is an error, not a silent wrong answer."""

    import urllib.error

    provider = NvidiaEmbeddingProvider(api_key="k", base_url="https://example.com")

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://example.com/embeddings", 401, "Unauthorized", {}, None
        ),
    ):
        with pytest.raises(NvidiaEmbeddingError, match="HTTP 401"):
            provider.embed(["hello world"])


def test_dimension_mismatch_raises() -> None:
    """NVIDIA returning the wrong width is refused, never silently scored."""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"data": [{"embedding": [0.1, 0.2]}]}
            ).encode("utf-8")

    provider = NvidiaEmbeddingProvider(api_key="k", base_url="https://example.com", dimension=4)

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        with pytest.raises(NvidiaEmbeddingError, match="expected 4"):
            provider.embed(["hello world"])


def test_mismatched_count_raises() -> None:
    """Fewer vectors back than texts sent means pairing would be wrong."""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}).encode("utf-8")

    provider = NvidiaEmbeddingProvider(api_key="k", base_url="https://example.com", dimension=4)

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        with pytest.raises(NvidiaEmbeddingError, match="1 embeddings for 2 inputs"):
            provider.embed(["one", "two"])


def test_successful_batch_returns_vectors_in_order() -> None:
    """One vector per input, paired by position."""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"embedding": [1.0, 0.0, 0.0, 0.0]},
                        {"embedding": [0.0, 1.0, 0.0, 0.0]},
                    ]
                }
            ).encode("utf-8")

    provider = NvidiaEmbeddingProvider(api_key="k", base_url="https://example.com", dimension=4)

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        vectors = provider.embed(["alpha", "beta"])

    assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def test_empty_input_returns_empty() -> None:
    """Embedding nothing returns nothing -- no request is made."""
    provider = NvidiaEmbeddingProvider(api_key="k", base_url="https://example.com")
    assert provider.embed([]) == []


def test_fake_provider_satisfies_the_protocol() -> None:
    """The test fake goes through the same door as the real provider."""
    fake = FakeProvider({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]})
    assert isinstance(fake, EmbeddingProvider)
    assert fake.model_id == "fake-v1"
    assert fake.dimension == 2
