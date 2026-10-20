"""Agent package for Verity.

The agent loop turns a user question into an evidence-backed answer (or
clarification / refusal) by running a bounded loop:

    retrieve → model decides → (search again | answer | clarify | refuse)

See :mod:`verity.agent.loop` for the entry point.
"""

from verity.agent.loop import AgentDecision, AgentLoop, MAX_ITERATIONS, make_retriever_from_index

__all__ = ["AgentDecision", "AgentLoop", "MAX_ITERATIONS", "make_retriever_from_index"]
