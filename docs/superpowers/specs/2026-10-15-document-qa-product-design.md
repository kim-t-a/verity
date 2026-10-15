# Document Q&A Product — Design

**Date:** 2026-10-15
**Status:** Approved for implementation planning
**Supersedes:** the MC3 challenge submission architecture in `CONTRACT.md` / `starter/`

---

## 1. Why this document exists

This repository was built to satisfy the AMD AI League Mini-Challenge 3 contract: a
containerised CLI that answered ten questions about a fixed corpus and named the exact
files necessary to produce each answer. The submission deadline (2026-10-13) passed
without a submission.

The code is good and the work should not be wasted. This spec redirects it: the same
retrieval engine, rebuilt as a **web product that anyone can use to ask questions about
their own documents**.

Because the challenge constraints no longer apply, most of what shaped the original
architecture — the mandated ROCm base image, the VRAM floor, the 60 GiB image budget,
the 30 s/600 s timeouts, the offline-at-evaluation rule — is now *gone*. This is a
simplification, not a complication, and it is the main reason the pivot is tractable.

---

## 2. Goals and non-goals

### Goals

1. Anyone can upload documents and ask questions about them in plain language.
2. Answers cite the exact source **file and location**, and the citation is clickable.
3. The system asks a clarifying question when a question is genuinely ambiguous,
   instead of guessing.
4. It remembers the conversation, so follow-up questions work.
5. The retrieval and agent behaviour are **visible**, not a black box.
6. The developer learns the fundamentals while building it — the architecture must not
   hide the mechanisms behind a framework.

### Non-goals (explicitly out of scope)

- Multi-user accounts. Deferred; the schema is shaped to accept them later without
  restructuring (see §9).
- Public hosting, HTTPS, secrets management. Local-first via Docker Compose.
- Non-English documents. Deferred until a non-English file actually appears; at that
  point it needs language-aware segmentation *and* a multilingual embedder, in that
  order.
- Sentence-level (span-precise) citations. The evidence says paragraph/record
  granularity is better, not worse (see §8).

---

## 3. The central insight driving the design

The existing answering layer (`answering/shapes.py`, `candidates.py`, `extract.py`) was
built to find a **literal value that already exists as text** in the corpus — `94`,
`REV-C2`, `Q3 FY27` — with **no language model at query time**. It classifies the
question by expected answer *shape* and ranks candidates lexicographically. That
machinery exists because the machine had to find the answer without understanding
anything.

Adding a language model makes most of it obsolete. Real users ask *"what are the
termination clauses?"* — questions with no literal answer, on which regex shape-matching
has no purchase. **The regex extraction layer is deleted, not adapted.**

This is the single most important structural consequence of the pivot.

---

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Transplant** — keep parsers, chunking, retrieval; rebuild storage, answering, agent loop, API, frontend | Parsers are the highest-effort, least-interesting, best-tested code. The answering layer must change anyway. |
| D2 | Product shape: **multi-user document QA**, each workspace owns its documents | What "a product people can use" means here. |
| D3 | **Defer auth**; one shared workspace | Removes a large non-RAG workstream from the critical path. Nothing is thrown away when auth lands (§9). |
| D4 | Agentic: **build the loop for tier 3, ship tier 2** | Same plumbing either way; ships predictable behaviour, dialled up after observation. |
| D5 | **NVIDIA NIM** (`build.nvidia.com`) for both generation and embeddings | OpenAI-compatible endpoint; one provider. See §13 for the risks this carries. |
| D6 | **Separate worker process**, Postgres as the job queue | Ingestion is minutes-long and cannot run in a request. `FOR UPDATE SKIP LOCKED` avoids adding Redis. |
| D7 | **Chunking: apply the evidence-based default**, no experiment first | See §7. Measure the token ceiling before trusting the existing chunker. |
| D8 | Frontend: **React**, explained as we go | Developer knows basics but is not fluent. |
| D9 | **Fresh design system in Stitch** | Existing navy/emerald system was designed for dense financial tables; a reading-and-conversation interface needs different proportions. |
| D10 | **Show retrieval and agent steps** in the UI | Trust feature for users, debugging window and primary learning tool for the developer. |
| D11 | Memory: **rewrite likely follow-ups only**; **summarise old turns** | See §9. |
| D12 | Duplicate uploads: **skip by sha256, reuse existing chunks** | Saves embedding cost; makes accidental double-uploads harmless. |
| D13 | Clarification trigger: **retrieval rule signals, model decides** | Testable detection combined with judgement; loggable. |
| D14 | Evaluation against **real documents supplied by the developer** | The MC3 corpus is engineered around literal-value answers and would flatter the system. |
| D15 | The product is named **Verity**; Python package `verity`, database `verity` | Short, pronounceable, and states the promise — answers you can verify. |
| D16 | **`AGENTS.md` is rewritten for the product**; challenge material moves to `legacy/` | It currently declares the challenge contract as this repo's source of truth, which is now false and actively misleading. |

---

## 5. Architecture

Four processes under Docker Compose:

```
Browser — React (Vite)
  Upload · Documents · Chat with clarification turns · Trace panel
        │ HTTP/JSON + multipart
api — FastAPI
  routes · validation · agent loop · citation verification
        │ SQL                        │ enqueue job
        ▼                            ▼
db — Postgres + pgvector        worker — Python
  documents, chunks,              parse · chunk · embed
  jobs, sessions, messages        (shares code with api)
        ▲
        │  api → NVIDIA API (generation + embeddings)
```

### Invariants

These are rules, not accidents. Violating any of them is a bug:

- **I1 — Retrieval unit ≠ citation unit.** Retrieve at chunk level; the model may cite
  only chunks it was actually shown. Small-to-big context is permitted; citing a parent
  that was not shown is not.
- **I2 — The model never sees text it cannot cite.** Every chunk in the prompt carries
  an ID the model may reference. No silently injected context.
- **I3 — Clarifying is never a way to avoid answering.** The loop is capped; at the cap
  it produces a best-effort answer, or refuses only when evidence genuinely does not
  exist.
- **I4 — Memory shapes the question, never the evidence.** Every turn re-retrieves.
  The model may not answer from remembered context and cite a document not retrieved
  this turn.
- **I5 — A document is `ready` only when every chunk is embedded.** Partial embedding
  must fail loudly; a document that looks ready with holes in its index is the worst
  available bug.

---

## 6. Data model

```
workspaces   id · name · created_at
                one default row now; becomes one-per-user when auth lands

documents    id · workspace_id · filename · storage_path · content_type
             size_bytes · sha256 · status · error · created_at · processed_at
                status: pending → processing → ready | failed
                sha256: duplicate detection (D12)

chunks       id · document_id · ordinal · text · location · token_count
             embedding vector(N) · tsv tsvector
                the hybrid index: tsv = BM25-equivalent FTS, embedding = dense
                N = the chosen embedder's dimension, fixed once in the migration.
                NOT assumed: the current local model is 384-dim, NVIDIA models
                differ from each other. Phase 0 reads it from the API and pins it.

sessions     id · workspace_id · created_at · state jsonb · summary text
                state holds an in-flight clarification
                summary holds the rolling compression of old turns (§9)

messages     id · session_id · role · content · kind · created_at
                kind: answer | clarification | refusal

citations    message_id · chunk_id · document_id
                audit trail; makes citation precision computable in SQL

jobs         id · kind · payload jsonb · status · attempts · run_at
             locked_at · error
                claimed with SELECT … FOR UPDATE SKIP LOCKED
```

**Why `chunks` carries both columns.** One row holds both halves of hybrid retrieval
(I1's substrate). `tsv` is Postgres full-text — the BM25-equivalent. `embedding` is the
vector. A single query can hit both and fuse with RRF. No Elasticsearch, no separate
vector database.

**Why `location` survives.** Readers already emit a human-readable position —
`page 3 line 42` (PDF), `paragraph 12` (DOCX), `line 42` (text), `row 4` (tabular) —
verified in the existing readers. Keeping it lets a citation point at the right part of
the document, at no cost, because ingestion already computes it.

**Why `citations` is a table, not a JSON blob.** "Which documents have ever been cited?"
and "which answers relied on this contract?" become one-line SQL, and citation precision
becomes computable rather than trusted.

**Indexes:** HNSW on `chunks.embedding`; GIN on `chunks.tsv`; btree on
`chunks.document_id`, `documents.workspace_id`, `jobs.status`.

---

## 7. Ingestion pipeline

```
file → sniff type → reader → spans with locations → chunk → embed (batched) → rows
       reuse          reuse        reuse            modify     new
```

Almost entirely reused existing code:

- `filetypes.py` — content-first format identification by magic bytes.
- `readers/` — PDF, DOCX, XLSX, CSV, text, images with OCR. Each emits spans carrying a
  `location` and an `ordinal`.
- `tabular.py` — already renders rows as `label=value` with a separate header span. This
  is the recommended table serialisation; do not change it.

### Chunking (D7)

**Kept from the existing implementation:**
- Span-based, structure-first chunking. This is the best-evidenced strategy in the
  literature (NVIDIA: page-level won; structure-aware tabular chunking raised MRR
  0.36→0.59) and it already exists at zero extra cost.
- `MAX_CHUNK_CHARS = 2000` as the ceiling — roughly the 256–512 token band the factoid
  evidence supports.
- The `"<location> part i of n"` relabelling on split.
- The `is_usable` gate.

**Changed:**

1. **Add a merge step.** The current chunker only ever splits; it never merges. A file of
   short lines therefore becomes a scatter of fragments, and the evidence says fragments
   hurt the generator. Pack adjacent spans up to the target size — but **never across a
   structural boundary the citation rule depends on**.
2. **Do not add overlap.** A 2026 industrial study (SPLADE + Mistral-8B) found no
   measurable benefit and doubled indexing cost; Chroma found overlap increases
   redundancy and penalises precision. Half this system's retrieval is BM25, which sees
   duplicated text as redundancy rather than evidence.
3. **Verify the ceiling against the tokenizer, not the character count.** `MAX_CHUNK_CHARS
   = 2000` is ~500 tokens *of prose*. Code, part numbers and OCR output are far denser in
   tokens per character. If a 2000-character chunk exceeds the embedder's 512-token
   limit, **the embedder silently truncates** — storing a vector for text that cannot
   fully be retrieved, with no error. Measure this in Phase 0.

**Explicitly deferred as premature optimisation:** semantic chunking (cost not justified
by consistent gains, per Vectara), late chunking (requires a mean-pooling long-context
encoder; incompatible with the current CLS pooling), contextual retrieval (real effect in
Anthropic's own numbers, weakly replicated elsewhere — a *second* thing to try if
citations still fail), proposition-based chunking, AST/code-aware chunking.

### Embedding client

- **Batching is the design, not an optimisation.** OpenAI-compatible embedding endpoints
  accept a list of inputs per request; the client sends batches.
- **Partial failure must be handled explicitly.** If a batch fails, the system must know
  which chunks embedded and which did not (I5). A document is not `ready` until all
  chunks are embedded.
- **The embedding model id is recorded in index metadata.** Changing models requires
  re-embedding everything; the only defence is knowing which model produced which vector.
  The existing `IndexManifest.provider_id` pattern carries over.

---

## 8. Query pipeline, agent loop, and citations

```
                    question (+ session history)
                              │
                              ▼
              rewrite into a standalone query   (only if it looks like a follow-up, §9)
                              │
                              ▼
              retrieve: tsv + vector → RRF      (reuse retrieval/hybrid.py)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  model decides                │◄──────────┐
              │  STRUCTURED output only       │           │
              └───────────────┬───────────────┘           │
                              │                           │
     ┌──────────┬─────────────┼───────────┬──────────┐    │
     ▼          ▼             ▼           ▼          ▼    │
  search     read       list_docs    ask_user    answer   │
     │          │             │           │          │    │
     └──────────┴─────────────┘           │          │    │
                │                         │          │    │
                └──── tool result ────────┼──────────┘    │
                                          │               │
                              max 5 iterations ───────────┘
                              │
                              ▼
                 verify citations → resolve to documents → respond
```

### The tools

| Tool | Purpose |
|---|---|
| `search(query, k)` | Hybrid retrieval over the workspace |
| `read_document(document_id, offset, limit)` | Fetch more of a document (small-to-big context) |
| `list_documents()` | What exists — lets the model judge whether a question is answerable at all |
| `ask_user(question, options?)` | Clarification; terminates the turn |
| `answer(text, chunk_ids)` | Terminal |
| `refuse(reason)` | Terminal |

### Rules

1. **Structured output, never prose parsing.** The model returns a JSON action. A
   documented failure mode is systems that scan the model's *prose* for a question mark
   and mistake a rhetorical question for a clarification request. Parse a field.
2. **The iteration cap is a hard stop.** Five tool calls, then the loop must answer from
   what it has. A documented failure mode is unbounded clarification loops (one project
   froze at iteration 2 having asked 8 questions).
3. **Citations are verified, not trusted.** The model returns chunk IDs; every ID is
   checked against the context actually sent. An ID that was never shown is an invented
   citation and the answer is rejected. This is the direct descendant of the existing
   `answering/citations.py` necessity rule.
4. **Clarifying never replaces answering** (I3).

### Clarification trigger (D13)

A deterministic signal is computed at retrieval time — concretely, the top two retrieval
scores falling within 15% of each other, or one query term matching several documents —
and is **stated in the model's prompt** as an observation ("retrieval found three
documents with similar relevance"). The model then judges whether that warrants asking.
This combines testable detection with judgement, and makes "why did it ask?" loggable.
The threshold is a starting value to be tuned against the evaluation set (§12), not a
derived constant.

### Citation granularity

Citations resolve to `{document_id, filename, location}` — the uploaded file plus where
in it — and are clickable, opening the source at that location.

**Granularity is deliberately paragraph/record level, not sentence level.** "Are Finer
Citations Always Better?" (Johns Hopkins, arXiv 2604.01432) found attribution quality
**lowest at sentence level and highest at paragraph level**, with forced fine-grained
citation costing 16–276% attribution quality for negligible correctness gain.

**Note the open research gap:** no published study measures chunk size or overlap under
an exact-set citation requirement with a co-occurrence necessity rule. The 6/10 → 7/10
measurement recorded in the existing `citations.py` is, as far as the research reached,
ahead of the literature on this specific question.

---

## 9. Memory

The `sessions` and `messages` tables store history. Pasting all of it into every prompt
is not a design: it grows without bound, eventually exceeds the window, and — critically
— **does not fix the actual problem**.

### The problem: follow-ups retrieve garbage

```
User:   What's the termination notice period?
System: 30 days, per section 4.2. [contract.pdf p.7]
User:   And for the other contract?
```

The retriever sees `"And for the other contract?"` and searches for those words. There is
nothing in the string to retrieve on. This is the most common failure in conversational
RAG and is invisible when testing with one-shot questions.

### The fix: conditional query rewriting (D11)

If the message looks like a follow-up — short, containing pronouns or references
(`it`, `that one`, `the other`), or sharing almost no content words with retrieved
documents — rewrite it into a standalone question before retrieval:

```
"And for the other contract?"
      ↓
"What is the termination notice period in the Acme master services agreement?"
```

Both the original and the rewritten query are stored, and the rewritten query is shown in
the trace panel. One-shot questions skip the extra model call entirely.

### Bounding history (D11)

- **Token budget** on recent turns, included verbatim.
- **Rolling summary** on the session row once the conversation outgrows the budget;
  older turns are compressed into it and it travels with every request thereafter.

### Scope

Because auth is deferred (D3), there is **one shared workspace and no user identity**.
Memory is therefore **per-conversation, not per-person**. The schema is shaped so that
adding `user_id` later turns session memory into user memory without restructuring.

### The governing invariant

**Memory shapes the question, never the evidence** (I4). Every turn re-retrieves from the
corpus. Answering from remembered context and citing a document not retrieved this turn
is a bug, not a feature.

---

## 10. API

```
POST   /api/workspaces/{wid}/documents        upload → 202 {document_id}
GET    /api/workspaces/{wid}/documents        list + status
GET    /api/documents/{did}                   detail
DELETE /api/documents/{did}                   remove, cascading chunks
GET    /api/documents/{did}/file              stream original (citation clicks)
POST   /api/sessions                          start a conversation
POST   /api/sessions/{sid}/messages           ask → answer | clarification | refusal
GET    /api/sessions/{sid}/messages           history
GET    /api/health
```

- Upload returns **202 Accepted** — honest for "accepted, not yet done" — enabling real
  progress reporting.
- The messages endpoint returns a **discriminated union**: a `type` field of `answer` /
  `clarification` / `refusal`. The frontend switches on one field; the model's three
  behaviours become three states with no shape-guessing.

---

## 11. Frontend

### Screens

1. **Documents** — drop zone; list rows showing status (`processing` / `ready` /
   `failed`), chunk count, delete. Polls while ingestion runs. This screen is the visible
   proof the worker works.
2. **Chat** — the conversation, with three distinctly rendered message types: answer with
   citation chips, clarification with input, refusal with an explanation of what was
   searched.
3. **Citation panel** — clicking a chip opens the original document and jumps to the
   cited `location` (e.g. PDF page 3), using the `location` column preserved through
   ingestion.
4. **Trace panel** — collapsible; shows retrieved chunks and scores, the rewritten query,
   each agent step, and timings (D10). Primary trust feature, primary debugging window,
   primary learning tool.

### State

React Query for server state (polling document status is exactly its use case); plain
`useState` for the composer. This teaches the server-state / client-state distinction.

### Design

A fresh design system built in Stitch (D9), designed for reading and conversation rather
than dense tabular data. Screens are designed in Stitch and reviewed *before*
implementation.

---

## 12. Testing and evaluation

**Tests assert code correctness. Evaluations measure answer quality. They are different
and both are required.**

### The enabling requirement

The LLM must sit behind an interface:

```python
class LLMProvider(Protocol):
    def complete(self, messages, tools) -> Response: ...
```

With `NvidiaProvider` (real) and `FakeProvider` (tests). Without this the agent loop
cannot be tested — it would be slow, nondeterministic and cost money per run. Same shape
as the existing `EmbeddingProvider` Protocol.

The loop's decision logic must be testable with a **scripted fake**, including the paths
where the model misbehaves, invents a citation, or loops.

### Layers

| Layer | Scope | LLM | DB |
|---|---|---|---|
| Unit | chunking, citation verification, loop state machine, tool dispatch | `FakeProvider` | none |
| Integration | upload → ingest → ask → answer, real Postgres, real files | `FakeProvider` | real |
| Eval | answer quality on labelled questions | real NVIDIA | real |

### On the existing test suite

Of the 681 existing tests (verified by `pytest --collect-only`), the parser, chunking,
retrieval and normalisation tests are portable and valuable. `test_index_cli.py` (CLI
contract), the container-path tests and the GPU-guard tests test a contract being
deliberately deleted. **Losing them is correct, not a regression, and each deletion will
be stated explicitly in its commit.**

### Evaluation

Against **real documents supplied by the developer** (D14), with a labelled set of
question → expected answer → expected source documents.

Metrics, in order of decision value:
1. **Citation precision and recall** — the all-or-nothing measure that matters most.
2. **Answer correctness** — judged, or exact-match where the answer is a literal.
3. **Refusal correctness** — refuses when it should, does not refuse when it should not.

Runs offline, on demand, not in CI.

---

## 13. Risks and unverified assumptions

Ranked by how much damage they do if wrong.

| Risk | Impact | Mitigation |
|---|---|---|
| **NVIDIA free-tier terms** are reported to scope use to *evaluation*, not production, and the primary ToS PDF could not be fetched to confirm | The provider may not be usable for a real product as-is | Model layer is swappable behind `LLMProvider` from day one. **Read the ToS before treating this as a production provider.** |
| **Free-tier limits are contradictory across sources** — some say ~40 req/min with no per-token billing, others a finite credit allowance that can be exhausted mid-batch | Determines whether the project has a compute budget at all | **Phase 0 measures it**: hit `/v1/models`, run requests, observe `429` (rate limit) vs `402` (credit exhausted) |
| **Embedding batching unverified.** Claimed likely; not confirmed | If absent, ingestion becomes very slow and D5 should be revisited in favour of local embeddings | Phase 0 confirms against the real endpoint |
| **`embedqa` models may require `input_type: "query"` vs `"passage"`** | Wrong values degrade retrieval quality in a way that is easy to miss | Phase 0 confirms; encode in the embedding client and cover with a test |
| **Silent embedding truncation** — 2000 chars of dense text may exceed the 512-token limit | Vectors stored for text that cannot be retrieved; no error raised | Phase 0 measures; fix by splitting on tokens rather than characters |
| **No CORS on the NVIDIA endpoint** | Key cannot live in browser code; a long-lived unscoped key placed client-side is published permanently | **The backend proxy is a security requirement, not a performance choice.** Never call NVIDIA from the frontend. |
| Chunking evidence is from single-retriever studies; none measures exact-set citations under hybrid retrieval | The transfer is an assumption, not a finding | Accept for now. If citations underperform in evaluation, the fallback is a controlled 4-arm comparison (fixed-size / recursive / structure-aware / semantic, at 256/512/1024 tokens) on our own labelled questions — our corpus, our metric, so no transfer assumption is needed |

### A note on evidence quality

During research, a widely-circulated claim that recursive ~512-token chunking scores 69%
versus semantic chunking at 54% was traced to its source and found to be unsupported: the
cited URL returns 404, no authors or venue are given, and every retelling conflates three
different vendor names. It is almost certainly content-farm material. **It is recorded
here so it is not rediscovered and repeated.** Verify numbers before building on them.

---

## 14. Build order

Each phase ends with something runnable and verifiable. Do not start the next until the
previous works. (This preserves the "small, verified steps" rule from `AGENTS.md`, which
survives the pivot.)

**Phase 0 — Verify before building.** Nothing is designed around an unverified
assumption.
- Docker Compose skeleton
- NVIDIA key checks: `/v1/models`; `429` vs `402` under load; embedding **batching**;
  `input_type` for `embedqa` models
- Measure the 512-token ceiling on dense chunks (§7)

**Phase 1 — Extract the engine into a package.** Name it; kill bare-name imports (`import
config` only works because Docker flattens the tree); delete the CLI, GPU guard, local
model and container paths. Keep parsers, chunking, normalisation, BM25, RRF. Prune tests
and state each deletion.

**Phase 2 — Storage.** Postgres schema and migrations. Replace `chunkstore.py` with
pgvector + `tsvector`. Write the hybrid query fusing them with the existing RRF.

**Phase 3 — Ingestion worker.** Postgres job queue; ingest pipeline wired end to end;
batched embedding client with retry and partial-failure handling (I5); document status
lifecycle.

**Phase 4 — Agent loop and API.** `LLMProvider`, `NvidiaProvider`, `FakeProvider`; tools;
capped loop with structured output; citation verification; FastAPI routes.

**Phase 5 — Frontend.** Stitch design system first, then React: Documents, Chat, Citation
panel, Trace panel.

**Phase 6 — Evaluation.** Labelled questions from real documents; metrics harness.

### Repository layout

One installable package, `verity`, with one subpackage per process:

```
backend/
  pyproject.toml
  verity/
    engine/     parsers, chunking, retrieval   ← ported from starter/app
    api/        FastAPI
    worker/     ingestion
    db/         schema + migrations
  tests/
frontend/       React
docker-compose.yml
docs/
legacy/         CONTRACT.md, brief, mc3-corpus — kept for reference
```

The engine has **no third-party dependencies** — PDF, DOCX and XLSX parsing are
hand-written against `zipfile` and `xml.etree` (verified). Only the deleted embedding
layer needed `transformers`. Extraction is therefore low-risk: there is no dependency
graph to untangle alongside the imports.

Restructure **in place**, keeping git history and the engine. Challenge material moves to
`legacy/` rather than being deleted: it documents why the code looks as it does, and the
corpus hazards (encrypted PDF, unreadable file, superseded revision, OCR images) remain
excellent ingestion robustness tests.

---

## 15. Open questions

To resolve before or during Phase 1:

1. **The agent's system prompt.** A real piece of writing, to be done deliberately rather
   than bolted on. It encodes the citation rules, the clarification policy, and the
   refusal policy — it is where I1–I4 are actually enforced against the model.
2. **The clarification signal threshold.** Which retrieval signal, and what value.
   Tunable; start with top-two scores within 15% and observe.
3. **Which NVIDIA model** for generation and which for embeddings. Determined by what
   Phase 0 finds available and its measured limits.

Resolved since drafting: the product name (D15, **Verity**) and the fate of `AGENTS.md`
(D16, rewritten for the product).

---

## 16. What is deleted, and why

| Component | Disposition |
|---|---|
| `app.py` CLI (`--index` / `--corpus` / `--query-id`), output JSON contract | **Deleted** — replaced by the API |
| `answering/shapes.py`, `candidates.py`, `extract.py`, `refusal.py` | **Deleted** — no-LLM regex extraction and a keyword-based refusal rule; both obsolete with a model in the loop (§3). Refusal becomes the model's `refuse` action (§8) |
| `answering/citations.py` | **Reimplemented** — the necessity idea is kept; the mechanism changes |
| `answering/supersession.py` | **Kept as an idea**, generalised — real corpora have superseded documents too |
| `embeddings/local_model.py`, `hashing.py` | **Deleted** — replaced by the NVIDIA embedding endpoint |
| `embeddings/__init__.py` Protocol | **Kept** — good interface, implementation swapped |
| `chunkstore.py` (JSONL + binary index) | **Replaced** by Postgres + pgvector |
| `ingestion/` (walker, filetypes, readers) | **Kept** — highest-value reuse |
| `chunking.py` | **Modified** — merge step added, ceiling verified |
| `retrieval/` (BM25, vector, RRF) | **Kept** — matches best practice |
| Dockerfile, ROCm base, GPU guard, tesseract-in-image | **Deleted** — challenge-only |
| `mc3-corpus/`, `CONTRACT.md`, brief | **Moved to `legacy/`** — reference and robustness tests |
