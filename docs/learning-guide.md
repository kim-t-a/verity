# Learning guide

A guided path from "I have not built a retrieval system" to "I can read this
repository and know why each part is there." Every section names the file it
lives in, so the reading is grounded rather than abstract.

Read `architecture.md` first for the shape; this file is the *why* underneath it.

---

## 1. Why this is not a chatbot

The instinct is to reach for an LLM and ask it the question. Resist it here, and
understand why, because the reason shapes everything else.

The answer to every question **already exists as literal text in the corpus** —
`94`, `Q3 FY27`, `ORR-FAN-2214-B`. Grading compares the value only. A generative
model asked "what is the revision?" will happily produce a plausible `REV-C3` that
appears nowhere in the documents, and it will produce it with the same fluency it
would produce the right answer. There is no partial credit and no way to tell the
two apart from the output alone.

So the task is **extraction, not generation**. The system's job is to find the
text that is already there and quote it. That reframing is what makes the problem
tractable without a GPU-heavy model, and it is why `answering/extract.py` exists
in the form it does: it derives candidate values from retrieved text rather than
composing an answer.

> **Takeaway.** Before reaching for generation, ask whether the answer is already
> written down. If it is, retrieval plus extraction is more accurate, cheaper, and
> auditable.

---

## 2. Index once, answer many — and why that is forced

```
--index   → walk, read, chunk, embed, persist      (once, 600 s)
--query   → load, retrieve, extract, cite          (×10, 30 s each)
```

This split is not an optimisation; it is a response to a hard constraint.
**Each question runs in a fresh process.** Nothing survives in memory between
questions. So anything expensive — reading 130 documents, running OCR on images,
computing embeddings — would be paid ten times over inside a 30-second budget if
it happened at query time.

The consequence is a design rule worth internalising: *anything you want at query
time must be on disk before the first question.* That is what `chunkstore.py`
does — `manifest.json`, `chunks.jsonl`, `vectors.bin`.

It also introduces a class of bug that a single-process design would never have:
the index can be **stale**. If the corpus changed since indexing, every answer is
about documents that no longer exist, and nothing in the output would look wrong.
Hence the corpus fingerprint and `INDEX_FORMAT_VERSION` in the manifest, verified
on load. See `test_index_cli.py::test_a_stale_index_is_refused_rather_than_answered_from`.

> **Takeaway.** When state crosses a process boundary, you need a way to tell
> whether it is still valid. A fingerprint is cheaper than a correctness bug.

---

## 3. Reading a corpus that is trying to break you

Before any retrieval, you have to get text out of files. `ingestion/walker.py` is
written against a specific threat model, and the threat model is the lesson.

The graded corpus is *guaranteed* to contain: an empty directory, a file with no
read permission, an AES-256 encrypted PDF, an unknown binary type. Each of these
is a chance for a naive walker to crash — and a crash during `--index` means every
question scores zero, not just the one about that file.

Two design choices follow:

**Catch per file, keep going.** One bad file must never abort the walk. Each
failure is recorded with a status and the traversal continues. The result is a
corpus that is *partially* readable, and a system that knows which parts.

**Decide readability by reading.** The temptation is `os.access()` or checking
mode bits. Both lie — mode bits are bypassed under root, and the Windows drive
this project lives on fabricates POSIX modes without enforcing them. The walker
reads `PROBE_BYTES` of each file instead, which proves readability and captures
the format signature in one operation.

That second point generalises: **when a check and the operation can disagree,
perform the operation and catch the failure.**

> **Takeaway.** Assume the input is hostile in exactly the ways the spec warns
> about, and make failure partial rather than fatal.

---

## 4. Retrieval, and why one method is not enough

Given a question, find the chunks most likely to contain the answer. Two
fundamentally different approaches, in `retrieval/`:

### BM25 — `retrieval/keyword.py`

The classic keyword ranking. Its intuition: a term is informative if it is **rare
across documents** but **frequent within the one you are looking at**. `BM25_K1`
controls how quickly repeated terms stop adding score (saturation — the tenth
occurrence of a word tells you much less than the second), and `BM25_B` controls
how much long documents are penalised.

The interesting detail here is the **two-field** setup: the same BM25 runs over
chunk text *and* over the file path, weighted by `PATH_FIELD_WEIGHT`. Paths carry
real evidence — a value found in `specs/tq40_datasheet_r2.pdf` is stronger
evidence about `tq40` than the same value in an unrelated file.

### Dense retrieval — `retrieval/vector.py`

BM25 fails when the question and the document share no vocabulary: "the maximum
current rating" versus a table headed "I max (A)". A dense retriever maps both
into a vector space where meaning, not wording, determines closeness.

This system uses `bge-small-en-v1.5` — 384 dimensions (`EMBEDDING_DIMENSION`), CLS
pooling (`EMBEDDING_POOLING`). Small is deliberate: model load happens *inside*
the 30-second per-question budget.

### Fusion — `retrieval/hybrid.py`

Two rankings that disagree. You could pick one, but picking a winner requires
evidence that one is better, and the measurement here showed a dead heat on the
single question where they differed. So they are fused with **Reciprocal Rank
Fusion**: each document scores `1/(RRF_K + rank)` in each list, summed. Rank
position, not raw score, which sidesteps the fact that BM25 scores and cosine
similarities are not on comparable scales.

> **Takeaway.** Rank fusion lets you combine retrievers without having to
> calibrate their scores against each other — which is usually the hard part.

---

## 5. The citation problem: necessity, not relevance

This is the heart of the challenge, and the part most worth understanding.

The obvious approach: cite every file that contributed to the answer. That scores
**zero**, because extra citations cost the same as missing ones.

So the question is not "which files were relevant?" but **"which files were
necessary?"** A file is necessary if removing it loses the answer. That is a
counterfactual test, and `answering/citations.py` implements it: hold the answer
fixed, ask what each source is load-bearing for, and cite only those.

The contrast is worth holding onto:

| | Relevance | Necessity |
|---|---|---|
| Question | Did this file help? | Would the answer survive without it? |
| Cites a supporting file | yes | no |
| Correct when scoring punishes extras | no | yes |

A subtler trap sits alongside it: **superseded revisions**
(`answering/supersession.py`). A `WITHDRAWN` file containing a *different* answer
will be in the graded corpus. Both files look relevant; only one is live. The
verdict is computed over the whole corpus rather than the retrieved results,
because a revision's status can rest on text held in another file — `r2` says what
it replaced.

> **Takeaway.** "Which sources support this?" and "which sources are required for
> this?" are different questions. When extras are penalised, only the second one
> is being asked.

---

## 6. Knowing when not to answer — `answering/refusal.py`

A refusal and a wrong guess score identically. So the second-worst thing
(guessing) is exactly as bad as refusing *when the answer exists* — and refusing
correctly scores full marks on the question that has no answer in the corpus.

Which means: **abstain only when evidence genuinely fails, never as a hedge.**
Hedging is pure downside.

The implemented rule is narrow: refuse when the winning value is presented as a
field's value, and the field's name shares no word with the question. That is a
*specific* structural signal, not a low-confidence threshold. Two plausible
alternatives were measured and rejected, and both measurements are recorded in
the module docstring — including the counter-intuitive one that confidence
**runs backwards**: the refusal question's winner scores 0.60, third highest of
the ten, while a question answered *correctly* scores 0.27. Any threshold that
refuses the first also refuses the second, and six more besides.

> **Takeaway.** A refusal rule fitted to intuition will be wrong. Measure the
> alternatives, and write down the ones you rejected so they are not re-litigated.

---

## 7. OCR: when the document is an image

Two sample questions are answerable only from an image and its geometry. The path
is `ocr_engine.py` → `ocr_tsv.py` → `image_geometry.py`.

The conceptual step is the one worth learning. Tesseract returns words with
bounding boxes — a flat bag of text fragments with coordinates, in no particular
order and with no notion of rows or columns. A table header and its value are
just two words near each other. `image_geometry.py` reconstructs the layout:
grouping words into lines, then pairing a label with a value by horizontal or
vertical proximity.

The rules are all in `config.py` and all explained: `LINE_VERTICAL_OVERLAP` for
what counts as the same line, `ASSOCIATION_MIN_GAP_FACTOR` /
`ASSOCIATION_MAX_GAP_FACTOR` for a plausible label–value gap,
`ASSOCIATION_AMBIGUITY_FACTOR` for when two candidates are too close to call. When
ambiguous, the pairing is **dropped** — no association is better than a wrong one.

And the failure mode is designed explicitly. Engine absent → report
`VISION_UNAVAILABLE`. Engine present but broken → raise. Never return invented
text. This is the same principle as the walker: silence about a missing capability
is worse than a loud failure, because a silent degradation costs exactly the
questions that needed it and looks like an ordinary wrong answer.

> **Takeaway.** Layout information is lost by OCR and must be reconstructed from
> coordinates. Ambiguity should produce no answer, not a guessed one.

---

## 8. The container and the GPU

The submission runs as a container built on a **mandated base image**,
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, checked by
**layer identity** — renaming does not help. This is an AMD competition, which is
why the base is ROCm rather than CUDA.

Two constraints surprise people:

**VRAM has a floor, not just a ceiling.** Peak must be between 1 and 48 GiB.
Below 1 GiB the submission is judged to have run on CPU and rejected. The obvious
hack — allocating a dummy buffer to clear the floor — is detected separately. The
system must *genuinely* use the GPU, which is why the embedding model runs on it.

**Image size is the uncompressed size**, not what `docker images` shows. The 60 GiB
budget is real disk footprint. Understanding the storage driver matters here: with
containerd's image store only the *unpacked* image is retained, so the base costs
its unpacked size rather than unpacked-plus-compressed. Getting this wrong makes a
feasible build look impossible.

A practical consequence of the layout rule — `COPY app/ /app/` means everything
under `starter/app/` ships inside the budget. Tests, docs and fixtures go
elsewhere.

That `COPY` has a second, less obvious consequence, and `test_import_graph.py`
exists to guard it. Putting `starter/app/` at `/app/` means the application's
modules import each other by **bare name** (`import config`), with no `starter`
package at runtime. A local suite importing `starter.app.…` would pass while the
container died on its first import — a whole class of failure that only a test
importing the same way the container does can catch.

> **Takeaway.** Read the constraint's *definition*, not its name. "Image size" and
> "what `docker images` prints" are different numbers, and the difference decides
> whether the build fits.

---

## 9. Reading the repository in order

If you are new to this codebase, this order builds understanding fastest:

1. `CONTRACT.md` — what is graded, and how. Everything else serves this.
2. `docs/architecture.md` — the pipeline end to end.
3. `normalisation.py` — short, and used everywhere. Read it before anything that
   compares values.
4. `ingestion/walker.py` — the hazard model that shapes ingestion.
5. `retrieval/keyword.py` then `retrieval/vector.py` then `retrieval/hybrid.py` —
   in that order, because hybrid only makes sense once both halves are clear.
6. `answering/citations.py` — the necessity idea, the core of the challenge.
7. `answering/refusal.py` — including the rejected alternatives.
8. `tests/test_index_cli.py` — how the real command line behaves, and the output
   contract's failure modes.
9. `scripts/score_sample.py` — how the system is measured end to end.

`IMPLEMENTATION_PROGRESS.md` holds the design decisions with their measurements,
stage by stage. When you want to know *why* something is the way it is and this
guide only says *what*, that is where the evidence lives.
