# `verity.llm` — the language-model seam

| Module | What it is |
|---|---|
| `provider.py` | The `LLMProvider` Protocol, `LLMAction`, and `FakeProvider` (deterministic, for tests). |
| `nvidia_provider.py` | The real client: calls NVIDIA's OpenAI-compatible chat endpoint. |

## The interface

```python
class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def decide(
        self,
        *,
        messages: list[dict[str, str]],
        chunks: list[Chunk],
        tools: list[str],
    ) -> LLMAction: ...
```

The agent loop is written against this Protocol, not against NVIDIA. Tests
swap in `FakeProvider` with one line; nothing else changes.

## Why `decide` instead of free text

The model returns **structured JSON** describing one action:

```json
{"type": "answer", "text": "150 °C", "chunk_ids": [3]}
```

That makes the loop's rules enforceable in code:

- `type` is validated against the four actions.
- `chunk_ids` are checked against the chunks actually shown.
- A malformed response is a retryable error, not an answer with no source.

Tool calling was considered; structured JSON was chosen because it is easier to
validate strictly and works with any chat-completions model.

## `FakeProvider`

A hand-written fake for tests. It returns scripted actions and never touches
the network. Embedding fakes return all-zero vectors, which cannot score under
cosine similarity — a fake index therefore degrades to keyword-only rather than
producing plausible-looking wrong matches.

## Choosing a backend

`verity/provider_factory.py` reads:

- `VERITY_LLM_PROVIDER` — `fake` or `nvidia` (default `nvidia`)
- `VERITY_EMBEDDING_PROVIDER` — `fake` or `nvidia` (default `nvidia`)

In production both are `nvidia`. In tests both are `fake`.

## Security

`NvidiaProvider` reads the API key from the environment (loaded from `.env` at
the repository root). The key never reaches the browser: the frontend calls the
backend, and the backend makes the model call. That proxy is a security
requirement, not a performance choice.
