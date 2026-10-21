"""Which backend the app is configured with, decided in exactly one place.

Selection is by environment variable, so tests can pin the deterministic
fakes with one line and never touch the network:

    VERITY_LLM_PROVIDER      = "nvidia" (default) | "fake"
    VERITY_EMBEDDING_PROVIDER = "nvidia" (default) | "fake"

The fake embedding provider returns all-zero vectors. Those can never score
under cosine similarity (a zero vector has no direction), so a fake-built
index degrades to keyword-only retrieval rather than to plausible-looking
wrong matches -- degraded, and visible in ``HybridRetriever.components``.
"""

from __future__ import annotations

import os

from verity.engine.embeddings import EmbeddingProvider
from verity.engine.nvidia_embeddings import NvidiaEmbeddingProvider
from verity.llm.nvidia_provider import NvidiaLLMProvider
from verity.llm.provider import FakeEmbeddingProvider, FakeProvider, LLMProvider


def default_llm_provider() -> LLMProvider:
    """The LLM backend :meth:`decide` calls this process is configured with."""
    selection = os.environ.get("VERITY_LLM_PROVIDER", "nvidia").strip().lower()
    if selection == "fake":
        return FakeProvider()
    return NvidiaLLMProvider()


def default_embedding_provider() -> EmbeddingProvider:
    """The embedding backend ingestion indexes with."""
    selection = os.environ.get("VERITY_EMBEDDING_PROVIDER", "nvidia").strip().lower()
    if selection == "fake":
        return FakeEmbeddingProvider()
    return NvidiaEmbeddingProvider()
