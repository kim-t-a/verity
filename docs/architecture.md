# Architecture

What the system is, how a question travels through it, and why each stage is
shaped the way it is. Design *decisions* with their evidence live in
`../IMPLEMENTATION_PROGRESS.md`; this file describes the shape that resulted.

## The problem, stated precisely

Ten questions about a corpus of technical documents. For each, the submission
must give:

- `answer` — a short literal value that **already exists as text in the corpus**
  (`94`, `Q3 FY27`, `ORR-FAN-2214-B`, `REV-C2`, `4.3.2`), graded on value only;
- `citations` — the **exact set of files necessary** to produce it, graded as a
  set;
- `confidence` — recorded, never scored.

The difficulty is not in the answering. It is in the two constraints that
surround it:

1. **Extra citations score the same as missing ones: zero.** A file that helped
   but was not *load-bearing* is as damaging as a file omitted. This is what
   makes citations a necessity problem rather than a relevance problem.
2. **A refusal and a wrong guess score identically.** So abstaining is only ever
   correct, never a hedge — hedging is pure downside.

Both constraints push in the same direction: the system must be able to say *why*
a file was needed, and must not answer when it cannot.

## The two-phase shape

```
python3 /app/app.py --index /app/corpus          # once; 600 s budget
python3 /app/app.py --corpus /app/corpus \
        --query-id query_01 --query "..."        # ten times; 30 s each
```

**Every question is a fresh process.** That is the contract, not an
implementation detail, and it dictates the whole architecture: nothing survives
in memory between questions, so all expensive work — walking, decoding,
extracting, chunking, embedding — is paid once at index time and persisted.
The per-question path is a load and a lookup, which is what makes a 30-second
budget achievable at all.

The index is three files under `MC3_INDEX_DIR` (`config.py`):
`manifest.json` (`MANIFEST_FILENAME`), `chunks.jsonl` (`CHUNKS_FILENAME`),
`vectors.bin` (`VECTORS_FILENAME`).

The manifest carries a `INDEX_FORMAT_VERSION` and a fingerprint of the corpus it
was built from. Both are checked on load, and a mismatch is refused rather than
answered from — a stale index produces confident answers to questions about
documents that have changed. See `IndexUnavailable` / `IndexIncompatible` in
`chunkstore.py`.

## Pipeline

```
                  INDEX TIME (once, 600 s)
corpus
  │
  ├─ walker.discover ......... defensive traversal; per-file catch
  ├─ read_file ............... format dispatch → ReadResult
  ├─ chunks_for_all .......... ReadResult → Chunk[]
  └─ provider.embed .......... Chunk[] → vectors   (skippable; see below)
        │
        └─► manifest.json + chunks.jsonl + vectors.bin

                  QUERY TIME (per question, 30 s)
question
  │
  ├─ load(index) ............. verify version + fingerprint
  ├─ hybrid search ........... BM25 ∥ dense, fused by RRF → top 10 chunks
  ├─ supersession ............ drop revisions that are no longer live
  ├─ extract ................. shape the question → candidate values → winner
  ├─ citations ............... which files were *necessary* to that winner
  └─ refusal ................. is this a field's value under an unrelated label?
        │
        └─► {"answer": ..., "citations": [...], "confidence": ...}
```

### Ingestion — `ingestion/`

`walker.py` ("defensive corpus traversal") enumerates the tree and survives the
hazards the graded corpus is known to contain: an empty directory, a file with no
read permission, an AES-256 encrypted PDF, an unknown binary type. The rule is
that **one bad file must never abort the walk** — each is caught, recorded with a
status, and the traversal continues.

Whether a file is readable is decided by *reading it*, not by checking
permissions: mode bits lie under root and on mounted Windows drives. The walker
reads `PROBE_BYTES` of each file, which simultaneously proves readability and
captures the leading bytes used to identify the true format.

The readers dispatch on content, not extension (`ingestion/filetypes.py`). Every
format holds at least one graded answer, and two questions are answerable only
from an image, so no reader may fail silently.

**The image path** is the most involved. `image.py` drives it; `ocr_engine.py`
locates and runs the engine; `ocr_tsv.py` parses Tesseract's TSV word output;
`image_geometry.py` reconstructs lines and pairs labels with values spatially.
When the engine is absent the reader reports `VISION_UNAVAILABLE` — it never
returns invented text. When the engine is present but broken it raises, because a
silently degraded vision path would cost exactly the questions that need it.

### Chunking and tokenisation

`chunking.py` turns read results into what the index stores, breaking on
`MAX_CHUNK_CHARS` with `CHUNK_BREAK_LOOKBACK` of context so a break does not
sever a value from its label. `tokenisation.py` produces the terms retrieval
matches on — one implementation, used by both BM25 and the path field, so the
two cannot drift apart.

### Retrieval — `retrieval/`

Two retrievers over the same chunks, deliberately different in kind:

- `keyword.py` — **BM25** (`BM25_K1`, `BM25_B`) over two fields, text and file
  path, with the path weighted by `PATH_FIELD_WEIGHT`. The path carries real
  signal: a value in `specs/tq40_datasheet_r2.pdf` is evidence about `tq40`.
- `vector.py` — **dense** cosine similarity against a local BGE model
  (`bge-small-en-v1.5`, `EMBEDDING_POOLING = "cls"`, `EMBEDDING_DIMENSION = 384`).
  This is what finds a paraphrase whose wording shares no term with the question.

`hybrid.py` ("one ranking from two that disagree") fuses them with Reciprocal
Rank Fusion (`RRF_K`), drawing `FUSION_CANDIDATE_DEPTH` deep from each list and
returning `RETRIEVAL_LIMIT` chunks. Fusion rather than a choice between them:
the two disagree on exactly one sample question, and the disagreement is a dead
heat, so picking a winner would have been a claim the measurement does not
support.

**The embedding backend is allowed to fail, loudly.** `embeddings/local_model.py`
is the real path — torch on the GPU, weights from `EMBEDDING_MODEL_DIR`.
`embeddings/hashing.py` is a deterministic CPU-only stand-in for local testing.

The failure mode is the interesting part. If the real backend cannot load, the
index pass logs `no embeddings, indexing lexical-only` and continues — a
degraded run scores *something* rather than crashing. But the degradation is
never silent: it is reported on **stderr**, and the manifest records which
provider built the index, so a lexical-only index is identifiable afterwards.
That distinction matters: lexical-only scores 9/10 on the sample set where hybrid
scores 10/10, and the difference is invisible in an answer-only view.

`hashing.py` carries a guard so it cannot be reached accidentally in the
container (rule 7: a CPU-only path may exist for testing, but must be impossible
to select by accident). See `ALLOW_LOCAL_BACKEND_ENV`.

### Answering — `answering/`

This is where the two hard constraints are actually satisfied.

- `candidates.py` — "a value span, plus everything needed to judge it." A
  `ValueSpan` is a fact about a piece of text (these characters are a version
  number); a `Candidate` is a *claim about an answer* — this value, in this
  chunk, in this file, at this position in the ranking. Keeping the two distinct
  is what lets later stages reason about a candidate's provenance rather than
  re-deriving it.
- `shapes.py` — "what a question asks for, and what that looks like in text."
  Derives the expected form of the answer (an identifier, a revision code, a
  number) and uses it to filter candidate values.
- `supersession.py` — "deciding which revision is the live one." A `WITHDRAWN`
  file containing a *different* answer is a trap that will be present in the
  graded corpus. The verdict is computed over the **whole corpus**, not over the
  retrieved results, because a revision's status can rest on text held in another
  file — `r2` says what it replaced. It is computed once, so every question
  agrees on it.
- `extract.py` — shapes plus retrieval produce candidate values; one wins.
- `citations.py` — **"which files were *necessary*, not which ones were used."**
  This is the answer to constraint 1. A file is cited only if removing it costs
  the answer. Files that merely contributed are not cited, because citing one
  scores the same as citing none.
- `refusal.py` — "recognising the question the corpus does not answer." The rule
  is narrow and stated in the module: refuse when the winning value is presented
  as a field's value and the field's name shares no word with the question. Two
  plausible alternatives were measured and rejected, and the measurements are
  recorded in the module docstring so they are not re-litigated.

### Normalisation — `normalisation.py`

One function, used everywhere: uppercase, strip whitespace, delete `-` `.` `·`
`_`, ignore units and degree signs. Used for answer comparison, evidence checks
and superseded demotion, so `REV-C2` and `rev c2` cannot be two different things
in two different places. `scripts/score_sample.py` **imports** it rather than
copying it, for the same reason.

## The output contract

Written once, whole, at the end of `main()`:

```json
{"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

Two properties are non-negotiable:

- **Both keys are present on every response, including refusals and errors.** A
  missing key is malformed and scores zero — which is why `answer()` catches its
  own expected failures (`IndexUnavailable`, `IndexIncompatible`,
  `HybridUnavailable`) and returns `("", [], 0.0)` rather than propagating.
- **The filename comes from `--query-id`**, never from the question text.

Note the asymmetry that `main()` preserves: an *expected* failure becomes a
well-formed refusal, but an *unexpected* exception is left to crash and write
nothing. That is deliberate. The contract is explicit that "a crashed writer and
a considered refusal must not look the same", and papering over an unforeseen bug
with an empty answer would dress a crash up as a judgement — and on the question
that has no answer in the corpus, would quietly claim credit for one.

## Budgets

Every one of these is a hard limit; violating any scores zero.

| Limit | Value | Where it bites |
|---|---|---|
| Startup + `--index` | 600 s | `MAX_TEXT_BYTES`, chunk sizes, batch size |
| Per question | 30 s | model load counts; nothing expensive may happen here |
| Whole run | 600 s | ten questions, ten processes |
| Image | ≤ 60 GiB uncompressed | what ships under `starter/app/` + `/models/` |
| VRAM peak | 1–48 GiB | below 1 GiB is judged as having run on CPU |

Measured on CPU: index 8.0 s over the sample corpus; 43.5 s over 130 files /
950 chunks, roughly linear at ~0.33 s/file and ~0.043 s/chunk — so the 600 s
index budget covers on the order of 1,800 files / 14,000 chunks. The per-question
budget is the tighter one on a GPU machine, because model load is inside it.

## Repository layout, and one rule about it

`starter/` is the Docker build context; `starter/app/` is the image payload —
the Dockerfile does `COPY app/ /app/`. Everything under it ships inside the
60 GiB budget and adds to build and startup time.

> **Never put tests, docs, fixtures or scratch files under `starter/app/`.**

Hence: tests in `tests/`, developer tooling in `scripts/`, prose in `docs/`.
