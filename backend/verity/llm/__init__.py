"""LLM provider package for Verity.

Defines the :class:`~verity.llm.provider.LLMProvider` Protocol that the agent
loop is written against, plus a hand-written :class:`~verity.llm.provider.FakeProvider`
for tests. A real NVIDIA-backed provider can be swapped in later.
"""

from verity.llm.provider import LLMAction, LLMProvider, FakeProvider, FakeEmbeddingProvider

__all__ = ["LLMAction", "LLMProvider", "FakeProvider", "FakeEmbeddingProvider"]
