"""NVIDIA embedding provider.

Turns text into vectors by calling NVIDIA's OpenAI-compatible embedding
endpoint, so that :class:`verity.engine.retrieval.vector.VectorRetriever` can
do semantic (meaning-based) search alongside keyword search.

This is the product-layer integration: the engine (``verity.engine``) defines
the :class:`~verity.engine.embedding_provider.EmbeddingProvider` Protocol and knows
nothing about HTTP or NVIDIA. This module is the bridge that satisfies the
Protocol using the real service. It uses only the standard library
(``urllib.request``) so it adds no third-party dependencies.

No real API calls are made in tests -- tests use a hand-written fake provider
that satisfies the same Protocol. Set ``NVIDIA_API_KEY`` and
``NVIDIA_API_BASE_URL`` in ``.env`` (see ``.env.example``) when you are ready
to call the real service. The provider never silently fabricates vectors: if
credentials or the endpoint are missing, every call raises a clear error.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Sequence

from verity.engine.embedding_provider import EmbeddingProvider

#: How many texts are sent in one HTTP request. NVIDIA's endpoint accepts a
#: list of inputs per call; batching keeps each request bounded and amortises
#: the per-request overhead.
_BATCH_SIZE = 64


class NvidiaEmbeddingError(Exception):
    """The NVIDIA embedding provider could not complete a request.

    Raised for missing credentials, network failures, bad responses, or
    dimension mismatches. Never silently falls back to fabricating vectors --
    a wrong vector is worse than no vector.
    """


class NvidiaEmbeddingProvider:
    """Embed text using NVIDIA's embedding endpoint.

    Reads ``NVIDIA_API_KEY`` and ``NVIDIA_API_BASE_URL`` from the environment
    by default. Both can also be passed explicitly, which is useful in tests
    and for pointing at a local mock.

    Construction always succeeds. A call to :meth:`embed` raises
    :class:`NvidiaEmbeddingError` if the credentials or endpoint are absent --
    the provider refuses rather than guessing.
    """

    def __init__(
        self,
        *,
        model_id: str = "nvidia/nemotron-3-embed-1b",
        dimension: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self._api_key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        # An explicitly passed empty string means empty, not "fall back to the
        # environment": tests rely on being able to pin a missing setting.
        self._base_url = (
            base_url if base_url is not None else os.environ.get("NVIDIA_API_BASE_URL", "")
        ).rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        """Identifier recorded in the index manifest. Must change if vectors change."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Length of every vector this backend produces."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input, in the same order.

        Texts are sent to NVIDIA in batches of :data:`_BATCH_SIZE`. Each batch
        is one HTTP POST; results are reassembled in input order so the caller
        can pair result *i* with input *i*.

        Raises:
            NvidiaEmbeddingError: credentials missing, network failure, bad
                response, or dimension mismatch.
        """
        if not self._api_key:
            raise NvidiaEmbeddingError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or pass api_key explicitly."
            )
        if not self._base_url:
            raise NvidiaEmbeddingError(
                "NVIDIA_API_BASE_URL is not set. Set it in .env or pass "
                "base_url explicitly."
            )

        url = f"{self._base_url}/embeddings"
        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[batch_start : batch_start + _BATCH_SIZE])
            body = json.dumps(
                {
                    "input": batch,
                    "model": self._model_id,
                    "encoding_format": "float",
                }
            ).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise NvidiaEmbeddingError(
                    f"NVIDIA embedding request failed with HTTP {exc.code}"
                ) from exc
            except urllib.error.URLError as exc:
                raise NvidiaEmbeddingError(
                    f"NVIDIA embedding request could not reach the endpoint: {exc}"
                ) from exc

            data = payload.get("data", [])
            if len(data) != len(batch):
                raise NvidiaEmbeddingError(
                    f"NVIDIA returned {len(data)} embeddings for {len(batch)} "
                    "inputs in one batch; cannot pair vectors to texts."
                )

            for item in data:
                vector = item.get("embedding")
                if not isinstance(vector, list) or len(vector) != self._dimension:
                    actual = len(vector) if isinstance(vector, list) else "non-list"
                    raise NvidiaEmbeddingError(
                        f"NVIDIA returned a {actual}-dimension vector, "
                        f"expected {self._dimension}"
                    )
                all_vectors.append([float(value) for value in vector])

        return all_vectors
