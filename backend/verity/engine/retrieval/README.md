# `verity.engine.retrieval` — three ways to find evidence

| Module | Method | Best at | Weak at |
|---|---|---|---|
| `keyword.py` | BM25 over chunk text and file path | IDs, part numbers, filenames, exact phrases | paraphrases and synonyms |
| `vector.py` | Cosine similarity between embeddings | meaning, "how hot may it get" ≈ "maximum temperature" | exact rare tokens; depends on a good embedder |
| `hybrid.py` | Reciprocal Rank Fusion (RRF) of the two | covering both failure modes at once | adds latency and two indexes to maintain |

## The flow

```
query ──┬── KeywordRetriever.search ──┐
        └── VectorRetriever.search  ──┤
                                        ▼
                         reciprocal_rank_fusion
                                        ▼
                              list[RetrievedChunk]
```

`HybridRetriever` is what the agent uses. If one component is unavailable, the
hybrid retriever can still run the other — but it says so in its
`components` property rather than pretending both ran.

## Why RRF instead of adding scores

BM25 scores and cosine similarities are not on the same scale; summing them
would let whichever produces larger numbers dominate. RRF ignores raw scores
and merges by **rank**:

```
score(chunk) = Σ  1 / (k + rank_in_list)
```

`k` (default from `config.py`) damps the influence of the very top of each
list. A chunk ranked highly by both retrievers wins; one ranked highly by only
one still surfaces.

## `RetrievedChunk`

A result carries the chunk, its score, and which retriever(s) produced it.
`describe_sources()` turns that into the one-line explanation shown in the UI
and in the agent trace.

## What is deliberately absent

There is no reranker yet. A cross-encoder reranker is a common next step: it
re-scores the top ~50 candidates with a model that reads query and chunk
together. It is not needed to make the current corpus work, and adding one is
a spec decision, not a retrieval-module detail.
