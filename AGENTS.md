# AGENTS.md — persistent instructions for this repository

This file holds **durable rules**, not progress notes. Progress lives in
`IMPLEMENTATION_PROGRESS.md`. Update this file only when a project rule changes.

## What this project is

A retrieval-augmented question-answering system for **AMD AI League Mini-Challenge 3**.
The system answers ten questions about a corpus of technical documents and, for each
answer, must name the **exact set of files** that were necessary to produce it.

It is **not** a chatbot and **not** a summariser. Every answer is a short literal value
(`94`, `Q3 FY27`, `ORR-FAN-2214-B`, `REV-C2`, `4.3.2`) that already exists as text in
the corpus. The hard part is not generating the value — it is (a) finding it, and
(b) proving which files were load-bearing, because **extra citations score the same as
missing ones: zero.**

## Source of truth

When this file, a comment, or an agent's memory disagrees with these documents, the
documents win:

1. `CONTRACT.md` — the submission contract
2. `README.txt` — notes on the sample kit and its traps
3. `AMD AI League — Match 3 Challenge Brief.pdf` — the full brief
4. `selfcheck.py` — the harness checks, in the order the harness runs them
5. `sample-questions.json` — ten worked examples with expected answers/citations

Note: the brief and `CONTRACT.md` disagree on one point — the brief says absolute
citation paths (`/app/corpus/specs/x.pdf`) are accepted, while `CONTRACT.md`, the
starter docstring and `selfcheck.py` all steer toward **corpus-relative**
(`specs/x.pdf`). **We emit corpus-relative paths.**

## Hard constraints (violating any of these scores zero)

| Constraint | Value |
|---|---|
| Base image | `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` — checked by **layer identity**, renaming does not help |
| Image size | ≤ **60 GiB uncompressed** (not what `docker images` prints) |
| VRAM | peak **1–48 GiB**. Below 1 GiB ⇒ judged to have run on CPU ⇒ rejected |
| Startup | container start + model load + the whole `--index` pass ≤ **600 s** |
| Per question | ≤ **30 s**, from exec to a written JSON file |
| Whole run | ≤ **600 s** for all ten questions |
| Network | **none** at evaluation time; weights ship inside the image |
| GPU | must genuinely use it; a dummy buffer to clear the VRAM floor is detected separately |

## The interface — do not change it

```bash
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
```

Output: `/app/output/<query-id>_output.json`, where `<query-id>` is **the id passed on
the command line**, never derived from question text.

```json
{"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

## Rules that must never be broken

1. **Never hardcode a sample answer, a sample question id, or an expected citation
   set.** `sample-questions.json` is test data, not a lookup table. The graded corpus
   is different and larger.
2. **Never fabricate.** No invented document contents, no invented citations, no
   claimed test results, no claimed model capabilities.
3. **Never change the output contract** to make internal design easier. `answer` and
   `citations` must be present on *every* response, including refusals and errors. A
   missing key is malformed and scores zero.
4. **Never cite a file we did not successfully read.** An unreadable or encrypted file
   is not a source.
5. **Never replace the mandated base image.**
6. **Never make evaluation depend on network access.**
7. **Never use a CPU-only path as the final evaluation path.** (A CPU-only
   `NullBackend` *is* legitimate for local testing — it must be impossible to
   select accidentally in the container.)
8. **Never silently skip image support.** If the vision path is not ready, it must
   report itself as unavailable — not return invented text.
9. **Never let one bad file abort the corpus walk.** Catch per file, record it, continue.
10. **Never download large model weights or start a large build without asking first.**
11. **Never refuse defensively.** A refusal and a wrong guess score identically, so
    abstain only when evidence genuinely fails — never as a hedge.
12. **Never implement a whole stage in one uncontrolled change.** Small, verified steps.

## Corpus hazards (the graded corpus contains all of these)

- an **empty directory**
- a file with **no read permission** (test with `--cap-drop DAC_OVERRIDE`; as root,
  mode bits are bypassed and the test is meaningless)
- an **AES-256 encrypted PDF** — must be skipped, never cracked
- an **unknown binary type**
- **superseded revisions** (a `WITHDRAWN` file with a *different* answer)
- **near-miss rows** and **comments naming old values** next to the real one

## Layout — and one rule about it

`starter/` is the **Docker build context**, and `starter/app/` is the **image payload**:
the Dockerfile does `COPY app/ /app/`, so everything under it ships inside the 60 GiB
budget and adds to build and startup time.

> **Therefore: never put tests, docs, fixtures or scratch files under `starter/app/`.**

```
starter/app/     shipped code only (CLI, ingestion, indexing, retrieval, answering)
starter/Dockerfile
mc3-corpus/      the sample corpus — never modified by tests
tests/           unit + integration tests, incl. fixtures/
scripts/         developer tooling (e.g. score_sample.py)
docs/            architecture, learning guide, testing notes
```

## Local environment facts (verified)

- Project lives at `/mnt/d/rag`, which is a **9p/drvfs mount of Windows `D:`**.
  - **`chmod` is a no-op here** — verified experimentally. The unreadable-file case
    therefore **cannot** be tested in-place; copy the corpus to a native filesystem
    (e.g. `/tmp`) for that test.
  - File I/O is metadata-slow. Keep the venv and Docker build contexts off this mount.
- Laptop Python is **3.12.3**; the image is **Python 3.14**. A package that installs
  locally may have no 3.14 wheel. Verify in the image before committing to a dependency.
- Working venv: `~/venvs/mc3` (native ext4). Activate with
  `source ~/venvs/mc3/bin/activate`.
- Docker is Docker Desktop for Windows, invoked from WSL. There is no AMD GPU on the
  laptop, so the container can be built but not fully self-checked locally.

## Conventions

- Python with type hints on public interfaces and docstrings on public functions.
- Readable names; no unexplained magic numbers — put constants in `config.py`.
- One normalisation function, used everywhere (answers, evidence checks, superseded
  demotion). Per the contract: uppercase, strip whitespace, delete `-` `.` `·` `_`,
  and ignore units/degree signs.
- Tests must be runnable without a GPU, without network, and without the real corpus.
- Run `python3 -I` when executing scripts that read untrusted input from the corpus.
