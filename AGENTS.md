# AGENTS.md — persistent instructions for this repository

This file holds **durable rules**, not progress notes. Progress lives in the
plan's ledger (`.superpowers/sdd/<plan>/progress.md`). Update this file only
when a project rule changes.

## What this project is

**Verity** — a document question-answering product. A user uploads documents,
asks a question in plain language, and gets an answer that cites the exact file
and location it came from.

It is not a chatbot and not a summariser. The answer to a good question already
exists as text in the uploaded documents; the work is finding it, saying where
it came from, and admitting when it is not there.

The product is **local-first**: one shared workspace, no auth yet. It runs on
the developer's machine via Docker Compose.

## Source of truth

When this file, a comment, or an agent's memory disagrees with these documents,
the documents win:

1. `docs/superpowers/specs/` — the design specs. **The spec is the authority.**
2. `docs/superpowers/plans/` — the implementation plan for the work in flight.
   The plan argues from the spec; a conflict between them resolves to the spec.
3. The tests — they record what the system actually does.

`legacy/` holds the AMD AI League MC3 challenge material (`CONTRACT.md`, the
brief, the sample corpus, the starter Dockerfile). It is **not authoritative**.
It is kept because the engine was ported from it and the reasoning behind some
of the parsing code is only written down there.

## The invariants

These are rules, not accidents. Violating any of them is a bug. They are the
ones most likely to be broken by a well-meaning change.

- **I1 — Retrieval unit ≠ citation unit.** Retrieve at chunk level; the model
  may cite only chunks it was actually shown. Small-to-big context is permitted;
  citing a parent that was not shown is not.
- **I2 — The model never sees text it cannot cite.** Every chunk in the prompt
  carries an ID the model may reference. No silently injected context.
- **I3 — Clarifying is never a way to avoid answering.** The loop is capped; at
  the cap it produces a best-effort answer, or refuses only when evidence
  genuinely does not exist.
- **I4 — Memory shapes the question, never the evidence.** Every turn
  re-retrieves. The model may not answer from remembered context and cite a
  document not retrieved this turn.
- **I5 — A document is `ready` only when every chunk is embedded.** Partial
  embedding must fail loudly; a document that looks ready with holes in its
  index is the worst available bug.

## Rules that must never be broken

1. **Never fabricate.** No invented document contents, no invented citations, no
   claimed test results, no claimed model capabilities. If it was not read or
   not run, it does not go in the answer, the code, or a report.
2. **Never cite a file we did not successfully read.** An unreadable, encrypted
   or unsupported file is not a source. (Invariant I2.)
3. **Never let one bad file abort the ingest.** Catch per file, record the
   failure, continue. A corpus of a thousand documents must not be lost to one
   malformed one.
4. **Never refuse defensively.** A refusal and a wrong guess are not equally
   bad, but an unnecessary refusal is still a failure. Abstain only when
   evidence genuinely fails — never as a hedge.
5. **Never invent an expected answer in a test.** Fixtures are test data, not
   lookup tables. A test asserts what the code does with known input; it must
   never encode a product answer.
6. **Never change a public contract to make internal design easier.** The API
   response shape is a contract with the frontend. Changing it is a deliberate
   decision, made in the spec, not a side effect of a refactor.
7. **Never call the model provider from the frontend.** The NVIDIA endpoint
   sends no CORS headers, so a browser cannot call it. The backend proxy is a
   **security requirement**, not a performance choice: it is what keeps the API
   key off the client.
8. **Never put a key in the repository or in a conversation.** Keys live in
   `.env`, which is git-ignored. A key pasted into a chat is a leaked key and
   must be rotated. `.env.example` documents the variable names only.
9. **Never download large model weights or start a large build without asking
   first.**
10. **Never implement a whole stage in one uncontrolled change.** Small,
    verified steps.

## Repository layout

One installable package, `verity`, with one subpackage per process:

```
backend/
  pyproject.toml
  verity/
    engine/     parsers, chunking, retrieval   ← ported from the challenge
    api/        FastAPI
    worker/     ingestion
    db/         schema + migrations
  tests/
frontend/       React (Vite)
docker-compose.yml
docs/           specs, plans, architecture, learning notes
legacy/         challenge material — kept for reference, not authoritative
```

The engine has **no third-party dependencies**. PDF, DOCX and XLSX parsing are
hand-written; that is deliberate, and it is why `verity.engine` can be imported
and tested with nothing installed. Keep it that way — a parsing library added to
the engine is a decision for the spec, not for a task.

`backend/tests/` is the only test tree. `backend/pyproject.toml` is the pytest
config; run pytest from `backend/`.

## Local environment facts (verified)

- The project lives at `/mnt/d/rag`, a **9p/drvfs mount of Windows `D:`**.
  - **`chmod` is a no-op here** — verified experimentally. A test that depends
    on mode bits cannot run in place; copy the data to a native filesystem
    (e.g. `/tmp`) for that. The `hazard_corpus` fixture detects this and says so.
  - File I/O is metadata-slow. Keep build contexts and virtualenvs off this mount.
- **`TMPDIR=/tmp` is required when running pytest here.** Without it, pytest
  puts `tmp_path` on the 9p mount and reports "no tests ran" with a misleading
  `FileNotFoundError`. Passing `TMPDIR=/tmp` also means `chmod` *is* enforced in
  fixtures, so the unreadable-file tests run for real instead of skipping.
- Laptop Python is **3.12.3**. A dependency that installs locally may have no
  wheel for whatever the container pins — verify in the container before
  committing to a dependency.
- Working venv: `~/venvs/mc3` (native ext4). Activate with
  `source ~/venvs/mc3/bin/activate`.
- There is no GPU in the evaluation path any more. Everything model-shaped is a
  hosted API call.

## Conventions

- Python with type hints on public interfaces and docstrings on public
  functions. Readable names; no unexplained magic numbers — put constants in
  `config.py`.
- Comments explain *why*, not *what*. A comment that restates the code is noise;
  a comment that records a trap, a trade-off or a verified fact is the reason
  the next person does not re-break it.
- Tests must be runnable without a network, without the real corpus, and without
  any provider key. Anything that calls a model is behind an interface with a
  fake behind it in tests.
- One normalisation function, used everywhere (answers, evidence checks,
  superseded-revision demotion).
- Run `python3 -I` when executing scripts that read untrusted input.
