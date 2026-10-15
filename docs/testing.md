# Testing

How to run the suite, what it covers, and — more usefully — where its guarantees
stop.

## Running it

```bash
cd /mnt/d/rag
TMPDIR=/tmp ~/venvs/mc3/bin/python -m pytest tests/ -q
```

Baseline on 2026-10-15: **681 passed, 0 skipped, ~62 s.**

### `TMPDIR=/tmp` is not optional, for two separate reasons

`~/.zshrc` exports `TMPDIR=/mnt/d/mc3-disk/tmp` to keep temp files off C:. Both
halves of that setting are wrong for pytest:

1. **pytest's capture plugin dies.** It does not merely create temp files; it
   `truncate()`s a reopenable handle, and drvfs/9p does not reliably support
   that. The failure is in *teardown* and reports as
   `FileNotFoundError` from `_pytest/capture.py` — with the summary line
   **"no tests ran"**, which reads like a collection problem and sends you
   looking in the wrong place entirely.
2. **The permission tests silently stop testing anything.** `tmp_path` derives
   from `TMPDIR`, and `tests/conftest.py` probes whether `chmod 000` genuinely
   denies reads before running those tests. Measured directly:

   | location | filesystem | `chmod 000` enforced |
   |---|---|---|
   | `/tmp` | ext4 | **yes** |
   | `/mnt/d/mc3-disk/tmp` | drvfs/9p | no |
   | `/mnt/d/rag` | drvfs/9p | no |

   That probe is the reason those tests *skip with a stated reason* rather than
   pass vacuously — a test that cannot fail is worse than no test, because it
   reports safety that does not exist. But a skip is still not coverage, and
   with all 681 tests passing and zero skipping, the unreadable-file path is
   genuinely exercised here.

## The three "without" rules

`AGENTS.md`: tests must be runnable **without a GPU, without network, and
without the real corpus.** Concretely:

- **No GPU.** The real embedding backend is exercised through a stubbed device
  path; `tests/test_embeddings_local.py` skips one test *because* a GPU exists on
  the target machine, since that branch is unreachable there.
- **No network.** No test downloads anything. The 129 MB BGE checkpoint is
  expected at `starter/models/bge-small-en-v1.5`; `tests/test_embeddings_local.py`
  skips — naming the path in the reason — when it or `torch` is absent, rather
  than fetching it. The real backend is still exercised here, on CPU, by the
  eight tests that use its fixture.
- **No real corpus.** `tests/conftest.py` builds a miniature corpus per test
  containing every hazard the brief warns about. The handful of tests that do
  use `mc3-corpus` skip if it is missing, and never write to it.

## What each file covers

Grouped by the stage they belong to.

| Area | Files |
|---|---|
| Ingestion | `test_walker`, `test_filetypes`, `test_readers_{common,text,pdf,docx,tabular,image,real_corpus}` |
| OCR / vision | `test_readers_{ocr_engine,ocr_tsv,image_geometry,image}` (152 tests between them) |
| Index | `test_chunking`, `test_chunkstore`, `test_index_cli` |
| Embeddings | `test_embeddings`, `test_embeddings_local` |
| Retrieval | `test_retrieval_{keyword,vector,hybrid}`, `test_tokenisation` |
| Answering | `test_answering_{shapes,candidates,extract,supersession,citations,refusal}` |
| Cross-cutting | `test_normalisation`, `test_import_graph` |

`test_import_graph.py` is worth knowing about: it guards the import graph that
runs **inside the container**, which is a different graph from the one the local
suite sees. Because the Dockerfile places `starter/app/` at `/app/` and the
harness runs `python3 /app/app.py`, the application's modules import each other by
**bare name** (`import config`, `from ingestion import walker`) and there is no
`starter` package at runtime. A local suite that imported everything as
`starter.app.…` would pass while the container died on its first `import config`.
These tests import through the same path the container uses, so that failure
surfaces here. It also pins the CLI entry point's contract and the
`MC3_INDEX_DIR` / `MC2_OUTPUT_DIR` environment overrides.

## The two tests that are not like the others

### `test_index_cli.py` — the only test that runs the real command line

Every other test imports a function. This one runs
`python3 app.py --index ...` and
`python3 app.py --corpus ... --query-id ... --query ...` as **separate
processes**, exactly as the harness does, and reads the JSON off disk.

That distinction is the point. Because each question is a new process, the only
thing crossing from indexing to answering is what `--index` wrote to disk. A
suite that called `index()` then `answer()` in one interpreter would pass while
the container failed, because it would keep in-memory state the real run loses.

It covers the output contract on the paths that go wrong as well as the ones that
go right: three required keys, filename taken from `--query-id`, a missing index,
a stale index, and a missing `--query-id`. All eleven of these check that a
response missing `answer` or `citations` cannot be produced — because a missing
key is malformed and scores zero, and a crashed writer must not look like a
considered refusal.

### `scripts/score_sample.py` — end-to-end, outside the suite

```bash
~/venvs/mc3/bin/python scripts/score_sample.py \
    --corpus mc3-corpus \
    --model-dir starter/models/bge-small-en-v1.5 \
    --device cpu
```

Not a pytest test and deliberately not in `tests/`, because it is the one piece
of code with the answers in front of it. It hardcodes none of them: every query,
answer, alias, citation set and refusal is read from `sample-questions.json` at
run time.

Three design points matter when reading its output:

- **It imports the application's own normaliser** rather than copying it. A
  second implementation that drifted would score answers differently from the
  pipeline that produced them.
- **Answers are compared exactly**, against the expected value or a declared
  alias. A substring match would let `194` pass as `94`. Where an answer is only
  a substring, the row is annotated *close, but not equal* — visible, never
  credited.
- **Citations are compared as exact sets.** A run citing the right file plus one
  more is a failure and is scored as one.

Two behaviours exist because of specific mistakes made while building it, and
should not be "simplified" away:

- **Both stdout and stderr of the index pass are echoed.** The pipeline reports
  on **stderr**, and that report is the only place a degradation is announced —
  `no embeddings, indexing lexical-only`. A scorer echoing only stdout hid the
  exact warning it existed to surface, and produced a plausible 9/10 that was not
  the number the shipped pipeline produces.
- **All path arguments are resolved to absolute before use.** The app is invoked
  with its own directory as cwd, so `--model-dir starter/models/...` resolved
  against the wrong place — and missed *quietly*, yielding a lexical-only run
  that still scored.

`--reuse` scores JSON files already in `--output-dir` without running anything.
That is how container output is scored: the container writes the files, they are
copied out, and the same comparison runs over them — one implementation, so a
local pass and a container pass mean the same thing.

## Measured results

All on CPU, sample corpus, `--device cpu`, one process per query (so model load
is inside each time).

| Run | Result | Index | Per query |
|---|---|---|---|
| Hybrid, OCR available | **10/10 answers, 10/10 citations** | 7.7 s, 95 chunks | 6.7–10.9 s |
| Hybrid, **no OCR engine** | 8/10 — queries 7 and 8 fail | 9.0 s, 67 chunks | 6.7–10.4 s |
| Lexical-only | 9/10 — query_10 answers `212` instead of refusing | — | — |
| Scale: 130 files / 950 chunks | — | 43.5 s | — |

Two of these rows are the point of scoring at all.

**The no-OCR row** fails exactly the two vision questions, and the index report
names why without being asked: `vision_unavailable=2`, listing
`specs/backplane_pinout.png` and `support/asset_label.jpg` as not indexed. Chunk
count drops 95 → 67. A system that silently skipped images would land here and
look like an ordinary 8/10 with two wrong answers.

**The lexical-only row** differs from the hybrid row by a single question, and the
difference is invisible in an answer-only view — which is why that degradation is
reported loudly on stderr.

Per-query times of 6.7–10.9 s on CPU are comfortably inside the 30 s budget *with
model load included*, which is the number that matters: each query is a fresh
process, so the checkpoint is loaded ten times over the run.

### A trap in the local OCR setup

The local OCR engine is a hand-unpacked tesseract 5.3.4 at
`/mnt/d/mc3-disk/tesseract/tesseract` (a wrapper setting `TESSDATA_PREFIX`), and
`MC3_TESSERACT` must point at it:

```bash
MC3_TESSERACT=/mnt/d/mc3-disk/tesseract/tesseract \
TMPDIR=/tmp ~/venvs/mc3/bin/python scripts/score_sample.py \
    --corpus mc3-corpus \
    --model-dir starter/models/bge-small-en-v1.5 \
    --device cpu
```

**A fresh shell does not have that variable, and the run then scores 8/10 without
erroring.** The failure is quiet in exactly the way `docs/architecture.md` warns
about: two plausible-looking wrong answers rather than a named missing capability.
`tesseract` is not on `PATH` on this machine either, so `which tesseract` will not
find it and looks like a broken install rather than a missing export.

This is a **development-machine** trap only. In the container tesseract is
installed system-wide by `apt-get` and the reader finds it with no environment
variable set at all — which is precisely the property the Dockerfile asserts with
`tesseract --version` at build time.

One more gotcha in the same area: `scripts/score_sample.py` defaults
`--output-dir` and `--index-dir` to `tempfile.gettempdir()`, which **follows
`TMPDIR`**. Running it with `TMPDIR=/tmp` writes to `/tmp/mc3-score-output`, not
to `/mnt/d/mc3-disk/tmp/mc3-score-output`. Files left in the old location look
current when they are not; pass the flags explicitly when it matters.

## Where the suite's guarantees stop

These cannot be verified on this machine. They need the container, and on the
GPU and VRAM checks, the notebook.

- **The `apt-get install tesseract-ocr` step.** A Python 3.14 / Ubuntu 26.04 base
  with package names that may differ from what a 3.12 / 24.04 machine assumes.
  This is why the Dockerfile runs `tesseract --version` as an **assertion**: a
  non-executable install must fail the build, not silently produce
  `VISION_UNAVAILABLE` in the container.
- **The 60 GiB uncompressed image budget**, which is not what `docker images`
  prints.
- **`--network none` startup**, the 600 s startup + index budget, and the 30 s
  per-question budget with real ROCm model load inside it.
- **VRAM floor and ceiling (1–48 GiB).** The floor is the interesting one: below
  1 GiB the submission is judged to have run on CPU and rejected.
- **`--cap-drop DAC_OVERRIDE`**, the harness's own unreadable-file test. As root,
  mode bits are bypassed and the test is meaningless, which is exactly why that
  flag is used.

The equivalent of the last one *is* covered locally, on a native filesystem where
permissions are enforced: a corpus copy at `/tmp/mc3-hazard` with
`vendor/internal_audit.txt` at mode `000`. Verified that reads genuinely fail as
uid 1000, that the walk skips the file and continues (12 files, `ok=10`), and
that scoring is unaffected — still 10/10.

`selfcheck.py` in the repository root is the harness's own check sequence, in the
order the harness runs it. It is the authority for the container phase.
