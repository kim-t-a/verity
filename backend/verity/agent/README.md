# `verity.agent` — turning a question into a checked answer

`loop.py` runs the bounded loop that produces one of three outcomes:

```
retrieve → model decides → (search again | answer | clarify | refuse)
```

## The decision types

| Action | What it means | What the user sees |
|---|---|---|
| `search` | Rewrite the query and retrieve again. | nothing yet; the loop continues |
| `answer` | Answer using cited chunk IDs. | the answer + citations |
| `clarification` | Ask the user a question. | a question back |
| `refusal` | Evidence is genuinely absent. | "not in the documents" |

`AgentDecision.kind` is one of `"answer"`, `"clarification"`, `"refusal"`.

## The rules the loop enforces

- **I1 — retrieve ≠ cite.** The model may cite only chunks that were actually
  put in the prompt this turn.
- **I2 — no uncitable text.** The provider receives only retrieved chunks;
  nothing is injected silently as "background".
- **I3 — clarification is not avoidance.** The loop is capped at
  `MAX_ITERATIONS = 5`. At the cap it produces a best-effort answer, or
  refuses only when evidence truly does not exist.
- **I4 — memory shapes the question, never the evidence.** Session history may
  inform a follow-up query, but every turn re-retrieves; the model cannot cite
  a document from an earlier turn that was not retrieved now.

## Citation verification

After an `answer`, the loop checks every cited chunk ID against the chunks
shown in this turn. An ID that was never shown is an invented citation, and
the answer is rejected rather than passed to the user.

## The trace

`AgentDecision.trace` records what actually happened: which retriever
components ran, each query and result count, timings, and the fused ranks and
scores of the chunks shown. The frontend's trace panel reads this. Nothing in
it is reconstructed after the fact.

## Tests

`tests/test_agent_loop.py` uses `FakeProvider`, so it needs no network and no
API key. The fake returns scripted actions; the tests assert the loop's
behaviour, not a model's output.
