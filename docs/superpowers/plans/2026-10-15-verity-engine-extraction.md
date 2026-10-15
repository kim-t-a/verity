# Verity Engine Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `starter/app/` into an installable, properly-imported `verity.engine` package, delete the challenge-only code, and leave a green test suite that only tests what still exists.

**Architecture:** A pure refactor. The engine's behaviour must not change: every test that survives must pass unmodified in what it asserts, so that the only thing that moved is *where the code lives* and *how it imports its neighbours*. Deletions are separate tasks from moves, so a behaviour regression and a deletion mistake can never be confused for one another. No new features land in this plan.

**Tech Stack:** Python 3.12+, setuptools, pytest 9.1.1. The engine is **pure standard library** (verified — PDF, DOCX and XLSX are hand-parsed against `zipfile` and `xml.etree`). No runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-document-qa-product-design.md` (Phase 1, §14; deletions per §16)

## Global Constraints

- **Python `>=3.12`.** The container/Python-3.14 constraint died with the challenge.
- **Package name `verity`.** Import root `verity`, engine subpackage `verity.engine`.
- **Engine has zero third-party runtime dependencies.** Adding one is a spec change, not a task detail.
- **Every `pytest` command must be run as `TMPDIR=/tmp pytest …`.** The project lives on a 9p/drvfs mount of Windows `D:`; with the inherited `TMPDIR`, pytest reports "no tests ran" alongside a misleading `FileNotFoundError`.
- **Use `git mv`, never delete-and-recreate.** Blame history on the parsers is the reason this pivot is cheap; a delete+add loses it.
- **Tests never import by bare name again.** After this plan, `from models import Chunk` must be impossible, not merely unused.
- **Do not change behaviour.** If a surviving test fails, the refactor is wrong — fix the refactor, not the test. The one exception is Task 7, which adds tests without changing code; if one of those fails, stop and report rather than silently fixing.

## Review Focus

Input classes and failure modes the spec implies that no existing test exercises, most likely to bite a real user first. Each gets a test in the task that owns the code.

1. **A UTF-16 text file.** **Verified, and it is a live defect.** `config.TEXT_ENCODINGS` is `("utf-8-sig", "cp1252", "latin-1")`. A UTF-16 file is not valid UTF-8, so it falls through to cp1252, which decodes every byte — including the NUL half of each character — and `looks_like_text` then rejects the file as "more than 10% of the decoded bytes are control characters". The file is reported `UNREADABLE`. Windows Notepad's default "Unicode" option writes UTF-16, so a user uploads a perfectly readable document and it silently becomes unusable. Fixed in Task 8.
2. **A document far larger than the sample corpus.** The challenge corpus is small and text-shaped; users upload 300-page PDFs and 50 MB spreadsheets. The text path caps reads at `MAX_TEXT_BYTES` (32 MB) *before* reading, via a `stat`, but a reader that inflates its input another way has no equivalent ceiling. Expected: a bounded read or a clean `UNREADABLE`, never memory exhaustion.
3. **A zip-based document that expands enormously** (DOCX/XLSX decompression bomb). `MAX_ZIP_MEMBER_BYTES` (256 MB) caps a single member; expected: refusal with a reason, not an OOM.
4. **A zero-byte file of every supported type.** A 0-byte `.pdf`/`.docx`/`.xlsx` is malformed, not empty, and a hand-written parser can easily raise on it. Expected: `EMPTY` or `UNREADABLE` — a value, never an exception.
5. **A file that disappears between discovery and read.** Verified: `readers.read_file()` catches `Exception` and returns `ReadResult.failure(...)`, so the *shipping path is already correct* — `app.py:128` reads through it. The gap is that the per-module contract ("a reader never raises") holds only because of the dispatcher; a direct `text.read(...)` on a missing path raises `FileNotFoundError`. Expected: the dispatcher stays the only entry point, and that guarantee gets a test so a future refactor cannot quietly route around it.

---

### Task 1: Package skeleton and build metadata

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/verity/__init__.py`
- Create: `backend/verity/engine/__init__.py`
- Create: `backend/tests/test_package_imports.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the import root `verity`, the subpackage `verity.engine`, and `verity.__version__: str`. Every later task imports through these names.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_package_imports.py`:

```python
"""The package must be importable by name, from anywhere.

This is the opposite of the test it replaces. The old suite put
``starter/app`` on ``sys.path`` so that modules could import each other by
bare name, reproducing how the container flattened the tree. That hack is
gone: ``verity`` is an installed package, so an import either resolves
through the package or fails loudly.
"""

from __future__ import annotations

import verity
import verity.engine


def test_package_imports() -> None:
    assert verity.engine is not None


def test_version_is_declared() -> None:
    assert isinstance(verity.__version__, str)
    assert verity.__version__.count(".") == 2, "expected a semantic version"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_package_imports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verity'`

- [ ] **Step 3: Write the build metadata**

Create `backend/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "verity"
version = "0.1.0"
description = "Ask your documents. Answers with verifiable citations."
requires-python = ">=3.12"
# The engine is pure standard library on purpose: hand-written parsers for
# PDF, DOCX and XLSX mean there is no dependency graph to untangle, and no
# transitive package that can change what a citation means.
dependencies = []

[project.optional-dependencies]
dev = ["pytest==9.1.1"]

[tool.setuptools.packages.find]
include = ["verity*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Carried over from the root pytest.ini this replaces. A test that skipped --
# for instance because the filesystem does not enforce chmod -- must be
# visible rather than silently counted as a pass.
addopts = "-ra"
```

- [ ] **Step 4: Create the package files**

Create `backend/verity/__init__.py`:

```python
"""Verity — ask your documents, get answers you can check."""

__version__ = "0.1.0"
```

Create `backend/verity/engine/__init__.py`:

```python
"""The retrieval engine: discover files, read them, chunk them, search them.

Nothing in here knows about HTTP, the database, or a language model. It turns
a directory of documents into searchable chunks with provenance, and answers
retrieval queries against them.
"""
```

- [ ] **Step 5: Install the package in editable mode**

Run: `cd backend && ~/venvs/mc3/bin/python -m pip install -e ".[dev]"`
Expected: `Successfully installed verity-0.1.0`

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_package_imports.py -v`
Expected: PASS — 2 passed

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/verity/ backend/tests/test_package_imports.py
git commit -m "Add the verity package skeleton and build metadata"
```

---

### Task 2: Move the engine modules and rewrite the import graph

This is the one atomic task in the plan. A module graph cannot be half-moved: partway through, every module is either unreachable or imported twice under two names. It is mechanical, but it is not small.

**Files:**
- Move (via `git mv`): `starter/app/config.py`, `models.py`, `normalisation.py`, `tokenisation.py`, `chunking.py` → `backend/verity/engine/`
- Move: `starter/app/ingestion/` → `backend/verity/engine/ingestion/`
- Move: `starter/app/retrieval/` → `backend/verity/engine/retrieval/`
- Move: `starter/app/embeddings/` → `backend/verity/engine/embeddings/`
- Move: `tests/` → `backend/tests/`
- Create: `backend/tests/test_no_bare_imports.py`

**Interfaces:**
- Consumes: `verity.engine` from Task 1.
- Produces: `verity.engine.config`, `verity.engine.models` (exports `Chunk`, `DiscoveredFile`, `ReadResult`, `ReadStatus`, `TextSpan`, `MediaType`, `SkipReason`, `SkippedEntry`, `DiscoveryResult`), `verity.engine.normalisation.normalise`, `verity.engine.chunking.chunks_for` / `chunks_for_all`, `verity.engine.ingestion.walker.discover` / `readers.read_file` / `readers.reader_for` / `filetypes`, `verity.engine.retrieval.hybrid` / `keyword` / `vector`. Later tasks and plans import only these paths.

- [ ] **Step 1: Move the files**

```bash
cd /mnt/d/rag
mkdir -p backend/verity/engine
git mv starter/app/config.py        backend/verity/engine/config.py
git mv starter/app/models.py        backend/verity/engine/models.py
git mv starter/app/normalisation.py backend/verity/engine/normalisation.py
git mv starter/app/tokenisation.py  backend/verity/engine/tokenisation.py
git mv starter/app/chunking.py      backend/verity/engine/chunking.py
git mv starter/app/chunkstore.py    backend/verity/engine/chunkstore.py
git mv starter/app/ingestion        backend/verity/engine/ingestion
git mv starter/app/retrieval        backend/verity/engine/retrieval
git mv starter/app/embeddings       backend/verity/engine/embeddings
# tests/ moves per entry, NOT as a directory: backend/tests/ already exists
# (Task 1 put test_package_imports.py there), so `git mv tests backend/tests`
# would nest it as backend/tests/tests.
git mv tests/*.py      backend/tests/
git mv tests/.gitkeep  backend/tests/.gitkeep
git mv tests/fixtures  backend/tests/fixtures
rm -rf tests/__pycache__ && rmdir tests
```

`chunkstore.py` moves with the engine because `retrieval/keyword.py` and
`retrieval/vector.py` both import `ChunkIndex` from it, and `ChunkIndex` is the
*input type* of both retrievers rather than challenge scaffolding. Only its disk
persistence is challenge-specific, and the spec assigns replacing that to Phase 2
(`spec:527`, `spec:600`) — not to this plan. Task 4 therefore leaves it alone.

Leave `starter/app/app.py`, `starter/app/answering/` and
`starter/app/requirements.txt` where they are for now — they are deleted in
Task 4, and moving them first would mean rewriting imports for code that is
about to disappear.

- [ ] **Step 2: Rewrite the imports mechanically**

Every internal import becomes absolute. Apply these rewrites across `backend/verity/engine/**/*.py`:

| From | To |
|---|---|
| `from config import` | `from verity.engine.config import` |
| `from models import` | `from verity.engine.models import` |
| `from normalisation import` | `from verity.engine.normalisation import` |
| `from tokenisation import` | `from verity.engine.tokenisation import` |
| `from chunking import` | `from verity.engine.chunking import` |
| `from chunkstore import` | `from verity.engine.chunkstore import` |
| `from ingestion import` / `from ingestion.` | `from verity.engine.ingestion import` / `from verity.engine.ingestion.` |
| `from retrieval import` / `from retrieval.` | `from verity.engine.retrieval import` / `from verity.engine.retrieval.` |
| `from embeddings import` / `from embeddings.` | `from verity.engine.embeddings import` / `from verity.engine.embeddings.` |
| `import config` | `from verity.engine import config` |

Verified against the tree: these are the only bare-name forms in use. Relative imports *inside* a subpackage (`from . import tabular, xmlpart`) are already correct and must be left alone.

The rewrite covers `backend/tests/*.py` as well as the engine — 26 test files import by bare name, and pytest cannot collect a single one of them after the move. It deliberately does **not** cover `answering` or `app`: those stay in `starter/app/` until Task 4, so their importers are expected to fail collection until then.

- [ ] **Step 3: Prove no bare import survives**

Create `backend/tests/test_no_bare_imports.py`:

```python
"""No engine module may import a neighbour by bare name.

``from models import Chunk`` only ever worked because Docker flattened
``starter/app/`` to ``/app/``, which put the directory on ``sys.path``. Now
that ``verity`` is a real package, a bare import is a bug that would resolve
only by accident -- from whichever directory happened to be the working
directory. This test makes the old style impossible rather than merely unused.
"""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "verity" / "engine"

BARE_NAMES = {
    "config",
    "models",
    "normalisation",
    "tokenisation",
    "chunking",
    "chunkstore",
    "ingestion",
    "retrieval",
    "embeddings",
    "answering",
    "app",
}


def imported_roots(source: Path) -> list[str]:
    """Every top-level module name ``source`` imports.

    Public rather than private so the detector itself can be exercised against
    a tree known to contain bare imports -- a guard nobody has watched fail is
    a guard nobody knows works.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is legitimate inside a package.
            if node.level == 0 and node.module:
                roots.append(node.module.split(".")[0])
    return roots


def bare_import_offenders(root: Path) -> list[str]:
    """Files under ``root`` that import a neighbour by bare name."""
    offenders: list[str] = []
    for source in sorted(root.rglob("*.py")):
        for name in imported_roots(source):
            if name in BARE_NAMES:
                offenders.append(f"{source.relative_to(root)} imports {name!r}")
    return offenders


def test_no_engine_module_imports_a_bare_neighbour() -> None:
    offenders = bare_import_offenders(ENGINE)
    assert offenders == [], f"bare imports in the engine: {offenders}"
```

`imported_roots` is public and `bare_import_offenders` is extracted so the
detector can be pointed at a tree that is *known* to contain bare imports — in
Task 2, `starter/app/`, which still does. A guard nobody has watched fail is a
guard nobody knows works.

The zero-third-party-dependency test does **not** belong here. `__future__` was
missing from its stdlib allowlist, and more importantly the engine still imports
`torch` and `transformers` from `embeddings/local_model.py` — which Task 3
deletes. A test added here could not pass until Task 3, and a task whose suite is
red at its own boundary is not done. It is created in Task 3, where the
dependency actually goes away.

- [ ] **Step 4: Run the full surviving suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q --continue-on-collection-errors`

`--continue-on-collection-errors` is required, not optional: without it the six
`test_answering_*.py` modules below abort the run during collection and you get
no pass/fail counts at all.

Expected: `41 failed, 552 passed, 6 errors`, every one of the 47 accounted for by
a contract this plan removes on purpose:

| Result | Cause | Removed by |
|---|---|---|
| 32 failures in `test_import_graph.py` | asserts the bare-name graph Step 2 abolished | Task 6 replaces it |
| 9 failures in `test_index_cli.py` | drives `starter/app/app.py`, whose imports Step 2 deliberately skips | Task 5 deletes it |
| 6 collection errors in `test_answering_*.py` | `answering/` still imports `config`/`models` by bare name | Task 5 deletes them |

A failure outside these three groups is a refactor mistake: fix the refactor, do
not touch the assertion. Before the move the tree collected 681 tests; 593 ran
here, the difference being the tests inside the six modules that now cannot be
collected.

- [ ] **Step 5: Confirm the old location is empty**

Run: `ls starter/app/`
Expected: only `app.py`, `answering/`, `requirements.txt` (plus `__pycache__`). `chunkstore.py` is *not* there — Step 1 moved it into the engine.

- [ ] **Step 6: Commit**

```bash
git add -A backend starter/app
git commit -m "Move the engine into backend/verity/engine with absolute imports

A module graph cannot be half-moved, so this is one atomic change. Files
moved with git mv to keep blame on the parsers, which is the code worth
keeping and the reason this pivot is cheap.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Reduce the embeddings package to its interface

The local model, the hashing fallback and the rule-7 CPU guard were all challenge machinery. Only the `EmbeddingProvider` Protocol survives, because the NVIDIA provider will satisfy it and tests will use a fake.

**Files:**
- Delete: `backend/verity/engine/embeddings/local_model.py`
- Delete: `backend/verity/engine/embeddings/hashing.py`
- Delete: `backend/verity/engine/embeddings/__init__.py` (replaced by a module)
- Create: `backend/verity/engine/embeddings.py`
- Modify: `backend/tests/test_embeddings.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `verity.engine.embeddings.EmbeddingProvider`, a `runtime_checkable` Protocol with `model_id: str`, `dimension: int`, and `embed(texts: Sequence[str]) -> list[list[float]]`. Later plans implement it with `NvidiaProvider` and `FakeProvider`.

- [ ] **Step 1: Read the Protocol you are keeping**

Run: `cat backend/verity/engine/embeddings/__init__.py`
Keep the `EmbeddingProvider` Protocol definition verbatim — including its docstrings, which explain why `model_id` and `dimension` are part of the interface. Everything else in the file goes.

- [ ] **Step 2: Replace the package with a single module**

```bash
git rm -r backend/verity/engine/embeddings
```

Then create `backend/verity/engine/embeddings.py` containing only the Protocol, with this module docstring:

```python
"""The embedding interface, and nothing that implements it.

Kept deliberately separate from any implementation: retrieval is written
against this Protocol, so the provider can be swapped -- NVIDIA at runtime, a
deterministic fake in tests -- without touching retrieval. The challenge-era
local model and its CPU fallback lived here and are gone; the guard that made
a CPU path unselectable went with them, because nothing here runs on a GPU
any more.
"""
```

- [ ] **Step 3: Prune the embedding tests**

Run: `TMPDIR=/tmp python -m pytest backend/tests/test_embeddings.py -q`
Delete every test that names `HashingEmbedder`, `local_model`, `provider_from_environment`, or a GPU/CPU guard. Keep the tests that assert the Protocol's shape and any test using a hand-written fake.

- [ ] **Step 4: Prove the engine imports nothing third-party**

`embeddings/local_model.py` was the engine's only importer of `torch` and
`transformers`, so deleting it is what makes the zero-dependency property true.
That property is load-bearing — it is why the engine has no dependency graph to
untangle and no transitive package that can change what a citation means — so it
is tested. Append to `backend/tests/test_no_bare_imports.py`:

```python
def test_engine_imports_stranger_modules_only_from_the_standard_library() -> None:
    """The engine's zero-dependency property is load-bearing, so it is tested."""
    allowed = {"verity", *BARE_NAMES}
    stdlib = {
        "__future__", "array", "collections", "csv", "dataclasses", "datetime",
        "enum", "hashlib", "io", "json", "math", "os", "pathlib", "re", "shlex",
        "shutil", "stat", "subprocess", "sys", "time", "typing", "unicodedata",
        "xml", "zipfile", "zlib",
    }
    offenders: list[str] = []
    for source in sorted(ENGINE.rglob("*.py")):
        for root in imported_roots(source):
            if root not in stdlib and root not in allowed:
                offenders.append(f"{source.relative_to(ENGINE)} imports {root!r}")
    assert offenders == [], f"third-party imports in the engine: {offenders}"
```

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_no_bare_imports.py -v`
Expected: PASS. If it names `torch` or `transformers`, something still imports
`local_model` — that import is the thing to fix, not the test. `__future__` is in
the allowlist because it is a stdlib module that every file here imports; leaving
it out produced 28 false positives when this test was first run.

- [ ] **Step 5: Run the suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: PASS. A failure here means something in `retrieval/` was importing a deleted name — that import is the thing to fix.

- [ ] **Step 6: Commit**

```bash
git add -A backend/verity/engine/embeddings.py backend/verity/engine/embeddings backend/tests
git commit -m "Reduce embeddings to the provider Protocol

The local ROCm model, the hashing fallback and the CPU-path guard were all
challenge machinery. Only the interface survives, because the NVIDIA provider
and the test fake will both satisfy it.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: Delete the challenge-only modules

**Files:**
- Delete: `starter/app/app.py`
- Delete: `starter/app/answering/` (entire directory)
- Delete: `starter/app/requirements.txt`, `starter/app/` (now empty)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. This task removes code; nothing later may import these names.

`chunkstore.py` is **not** deleted here. It moved into the engine in Task 2
because `ChunkIndex` is the input type of both retrievers, and the spec assigns
replacing its disk persistence to Phase 2 (`spec:527`, `spec:600`). Deleting it
now would leave the engine unimportable with no task in this plan to repair it.

- [ ] **Step 1: Confirm nothing in the engine imports what you are deleting**

Run: `grep -rn "answering\|^from app\|import app" backend/verity/engine/`
Expected: no matches. A match means a real dependency survived Task 2 — investigate before deleting. Do **not** add `chunkstore` to this pattern: the engine legitimately imports it, and every such match is expected.

- [ ] **Step 2: Delete**

```bash
cd /mnt/d/rag
git rm -r starter/app/answering
git rm starter/app/app.py starter/app/requirements.txt
```

- [ ] **Step 3: Run the suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: collection errors naming the deleted modules. That is correct and expected — Task 5 removes those tests. No *other* error is acceptable.

- [ ] **Step 4: Commit**

```bash
git add -A starter/app
git commit -m "Delete the challenge-only CLI, index store and answering layer

app.py and all of answering/ existed to satisfy a submission contract that no
longer applies. answering/ in particular found literal values by classifying a
question's expected answer shape and ranking candidates lexicographically --
machinery that exists only because the original system had no language model at
query time. A model makes it obsolete, so it is deleted rather than adapted.

chunkstore.py stays: it is the retrievers' input type, and the spec replaces it
with Postgres in Phase 2 rather than here.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Prune the tests for deleted contracts, and retire the sys.path hack

**Files:**
- Delete: `backend/tests/test_index_cli.py`
- Delete: `backend/tests/test_answering_candidates.py`, `test_answering_citations.py`, `test_answering_extract.py`, `test_answering_refusal.py`, `test_answering_shapes.py`, `test_answering_supersession.py`
- Delete: `backend/tests/test_embeddings_local.py`
- Rewrite: `backend/tests/conftest.py`
- Modify: `backend/tests/test_embeddings.py` (it imports `APP_DIR`, which Step 2 removes)
- Modify: `backend/tests/document_fixtures.py` (only if it imports deleted names)

**Interfaces:**
- Consumes: `verity.engine.*` paths from Task 2.
- Produces: `conftest.py` fixtures `hazard_corpus`, `build_corpus`, and constants `REPO_ROOT`, `CORPUS_DIR` (redefined in Task 9 when the corpus moves to `legacy/`).

`test_chunkstore.py` is **not** deleted: `chunkstore.py` survives this plan (see
Task 4), so the tests for it do too. It goes with the module in Phase 2.

`APP_DIR` is removed in Step 2, but `test_embeddings.py:22` does
`from conftest import APP_DIR` and uses it at line 247 to run an isolated
subprocess probe. Whoever executes Step 2 must repoint that probe at the
installed package in the same commit — otherwise removing `APP_DIR` is a
collection error in a file Step 1 does not touch.

- [ ] **Step 1: Delete the tests for deleted code**

```bash
cd /mnt/d/rag
git rm backend/tests/test_index_cli.py backend/tests/test_embeddings_local.py
git rm backend/tests/test_answering_candidates.py backend/tests/test_answering_citations.py
git rm backend/tests/test_answering_extract.py backend/tests/test_answering_refusal.py
git rm backend/tests/test_answering_shapes.py backend/tests/test_answering_supersession.py
```

These test contracts deliberately removed — a CLI that no longer exists, an index format replaced by Postgres, and regex extraction replaced by a model. **Losing them is correct, not a regression.** Their value was in what they proved about code that is gone.

- [ ] **Step 2: Rewrite conftest.py's path setup**

Replace the entire `sys.path` block (lines 22–36 of the original) — the `REPO_ROOT`/`APP_DIR` constants and the `sys.path.insert` — with:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "mc3-corpus"

from verity.engine.ingestion import walker  # noqa: E402
from verity.engine.models import DiscoveredFile  # noqa: E402
```

`verity` is installed, so no path manipulation is needed. The module docstring's first numbered job ("Put `starter/app` on `sys.path`") must be rewritten to say the package is installed instead — leaving that comment in place would describe a mechanism that no longer exists.

Note `REPO_ROOT` moves up one level: conftest is now at `backend/tests/`, not `tests/`.

- [ ] **Step 3: Run the suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: PASS, with a collection count well below the original 681. Record the new number in the commit message.

- [ ] **Step 4: Remove the superseded configuration**

The root `pytest.ini` sets `testpaths = tests` and `addopts = -ra` for a suite that has moved. Its `addopts` now lives in `backend/pyproject.toml`; its `testpaths` would point at a directory that no longer exists. `requirements-dev.txt` is likewise superseded by the `dev` extra in `pyproject.toml`.

```bash
git rm pytest.ini requirements-dev.txt
```

- [ ] **Step 5: Confirm nothing skipped the permission tests silently**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q -rs | grep -i "skip"`
Expected: permission tests skip *only* with the stated reason about drvfs. On `/mnt/d/rag` they will skip — that is the documented behaviour, and `chmod` being a no-op there is verified. To actually run them, copy the corpus to a native filesystem.

- [ ] **Step 6: Commit**

```bash
git add -A backend/tests pytest.ini requirements-dev.txt
git commit -m "Prune tests for deleted contracts and retire the sys.path hack

conftest.py put starter/app on sys.path to reproduce how the container
flattened the tree. verity is an installed package now, so that hack is
gone and an import either resolves through the package or fails loudly.

Deleted tests covered the CLI contract, the JSONL chunk store, the local
embedding model and the whole regex answering layer. All four are
deliberately removed, so the tests go with them. The root pytest.ini and
requirements-dev.txt are superseded by backend/pyproject.toml.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: Replace the import-graph guard

The old `test_import_graph.py` asserted that modules import by *bare* name, and that `app.answer()` returns a 3-tuple. Both assertions are now inverted or obsolete. Task 2 already added `test_no_bare_imports.py` covering the first; this task retires the old file so the suite does not assert two contradictory things.

**Files:**
- Delete: `backend/tests/test_import_graph.py`

**Interfaces:**
- Consumes: `test_no_bare_imports.py` from Task 2.
- Produces: nothing.

- [ ] **Step 1: Confirm the replacement covers the old file's job**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_no_bare_imports.py -v`
Expected: 2 passed. If these do not pass, Task 2 is incomplete — return to it rather than deleting the old guard.

- [ ] **Step 2: Delete the old guard**

```bash
git rm backend/tests/test_import_graph.py
```

Its four tests asserted: that `APP_DIR` is on `sys.path` (now false and undesirable), that 30 modules import by bare name (now forbidden), that `app.index`/`app.answer`/`app.main` exist (the module is deleted), and that `MC3_INDEX_DIR`/`MC2_OUTPUT_DIR` override paths (challenge env vars, gone). Not one survives the pivot.

- [ ] **Step 3: Run the suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A backend/tests
git commit -m "Retire the import-graph guard it was built to serve

It asserted the bare-name import graph that test_no_bare_imports.py now
forbids, plus the CLI entry points of a deleted module. Keeping both would
have the suite assert two opposite things.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: Pin the engine's behaviour on hostile input

A refactor's risk is silent drift. These tests characterise behaviour that must survive the rest of the pivot — they assert what the engine does *today*, and they must pass **without changing any code**. If one fails, stop and report: it means either the refactor broke something, or an assumption in this plan is wrong. Do not "fix" it by editing the engine.

**Files:**
- Create: `backend/tests/test_engine_robustness.py`

**Interfaces:**
- Consumes: `verity.engine.ingestion.walker.discover`, `verity.engine.ingestion.readers.read_file`, `verity.engine.ingestion.readers.reader_for`, `verity.engine.models.MediaType`/`ReadStatus`, and the `build_corpus` fixture from `conftest.py`.
- Produces: nothing importable; this is a guard.

- [ ] **Step 1: Write the tests**

Create `backend/tests/test_engine_robustness.py`. The `build_corpus` fixture takes a mapping of path to bytes, writes them to a temporary directory, and hands them to the **real walker** — so each case below travels the same identification path the pipeline uses.

```python
"""Behaviour on the inputs a real user's documents will contain.

Every test here asserts something the engine already does. They exist because
the corpus the tests were written against is small, tidy and shaped like the
challenge, whereas uploaded documents are not: they are huge, malformed,
misnamed, and occasionally hostile. None of these may raise -- ``read_file``
converts any exception to a failure value, and that guarantee is what stops
one bad document from costing a whole ingest batch.
"""

from __future__ import annotations

import pytest

from verity.engine.ingestion.readers import read_file, reader_for
from verity.engine.models import MediaType, ReadStatus


@pytest.mark.parametrize(
    "rel_path,media_type",
    [
        ("empty.txt", MediaType.TEXT),
        ("empty.csv", MediaType.CSV),
        ("empty.pdf", MediaType.PDF),
        ("empty.docx", MediaType.DOCX),
        ("empty.xlsx", MediaType.XLSX),
    ],
)
def test_a_zero_byte_file_is_a_value_not_an_exception(
    build_corpus, rel_path: str, media_type: MediaType
) -> None:
    """A 0-byte document is malformed, not empty, and hand-written parsers raise on it."""
    files = build_corpus({rel_path: b""})
    result = read_file(files[rel_path])

    assert result.status in {ReadStatus.EMPTY, ReadStatus.UNREADABLE}
    assert result.is_usable is False


def test_a_file_that_vanishes_before_it_is_read_is_a_failure_value(build_corpus, tmp_path) -> None:
    """The delete-while-ingesting case, and the only reason read_file catches broadly."""
    files = build_corpus({"gone.txt": b"text that will not be there\n"})
    discovered = files["gone.txt"]
    discovered.path.unlink()

    result = read_file(discovered)

    assert result.status is ReadStatus.UNREADABLE
    assert "FileNotFoundError" in result.detail


def test_a_document_claiming_a_huge_size_is_capped_before_reading(build_corpus) -> None:
    """MAX_TEXT_BYTES is checked from stat before the read, so this costs a stat."""
    files = build_corpus({"big.txt": b"x" * (40 * 1024 * 1024)})

    result = read_file(files["big.txt"])

    assert result.status is ReadStatus.OK
    assert "truncated" in result.detail


def test_a_zip_member_larger_than_the_cap_is_refused(build_corpus) -> None:
    """A DOCX is a zip; a member past MAX_ZIP_MEMBER_BYTES must not be inflated."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Highly compressible, so the archive stays small while the member does not.
        archive.writestr("word/document.xml", b"<w:t>" + b"a" * (300 * 1024 * 1024) + b"</w:t>")
    files = build_corpus({"bomb.docx": buffer.getvalue()})

    result = read_file(files["bomb.docx"])

    assert result.is_usable is False
    assert result.status in {ReadStatus.UNREADABLE, ReadStatus.EMPTY}


def test_every_media_type_with_a_reader_survives_garbage_bytes(build_corpus) -> None:
    """Nothing in the reader table may raise, whatever bytes it is handed."""
    payload = bytes(range(256)) * 64
    cases = {
        "junk.txt": MediaType.TEXT,
        "junk.csv": MediaType.CSV,
        "junk.pdf": MediaType.PDF,
        "junk.docx": MediaType.DOCX,
        "junk.xlsx": MediaType.XLSX,
    }
    files = build_corpus({name: payload for name in cases})

    for name, media_type in cases.items():
        assert reader_for(media_type) is not None, media_type
        result = read_file(files[name])  # must not raise
        assert result.rel_path == name
```

- [ ] **Step 2: Run the tests**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_engine_robustness.py -v`

Expected: all pass. **If any fails, stop and report it** — the plan's claims about existing behaviour are then wrong, and the correct response is to understand why, not to adjust the assertion until it goes green.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_engine_robustness.py
git commit -m "Characterise engine behaviour on hostile input

The sample corpus is small, tidy and challenge-shaped; uploaded documents
are huge, malformed and occasionally hostile. These tests pin behaviour
that already exists so the rest of the pivot cannot erode it silently.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: Read UTF-16 text files

The one defect this plan repairs. Verified: a UTF-16 file is reported `UNREADABLE` with the detail "more than 10% of the decoded bytes are control characters". Windows Notepad's default "Unicode" encoding is UTF-16LE, so this is an ordinary user's ordinary file being silently discarded.

This is a behaviour change, which is why it is a separate task from the move: with the refactor committed and green first, a regression here cannot be mistaken for a moving mistake.

**Files:**
- Modify: `backend/verity/engine/config.py`
- Modify: `backend/verity/engine/ingestion/readers/source.py`
- Create: `backend/tests/test_readers_text_encoding.py`

**Interfaces:**
- Consumes: `verity.engine.config` (adding `UTF16_BOMS`), `source.decode_text(raw: bytes) -> tuple[str, str]`.
- Produces: `verity.engine.config.UTF16_BOMS: tuple[tuple[bytes, str], ...]`. `decode_text`'s signature and totality are unchanged — it still cannot fail.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_readers_text_encoding.py`:

```python
"""Text files whose encoding is not UTF-8.

The candidate-encoding list in ``config`` ends with latin-1, which maps all
256 byte values, so decoding never fails -- it just sometimes succeeds at
producing something that is not text. A UTF-16 file is the case that matters:
every byte decodes as cp1252, so the text comes back interleaved with NULs and
is then rejected wholesale as "not text".
"""

from __future__ import annotations

import codecs

from verity.engine.ingestion.readers import read_file
from verity.engine.models import ReadStatus

BODY = "Section 4.2 Termination\nNotice period is 30 days.\n"


def test_a_utf16_le_file_is_read(build_corpus) -> None:
    """What Notepad writes when saving as "Unicode"."""
    files = build_corpus({"notes.txt": BODY.encode("utf-16")})  # includes a BOM

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "30 days" in result.text


def test_a_utf16_be_file_is_read(build_corpus) -> None:
    """Big-endian, with an explicit BOM: str.encode("utf-16-be") adds none."""
    raw = codecs.BOM_UTF16_BE + BODY.encode("utf-16-be")
    files = build_corpus({"notes.txt": raw})

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "30 days" in result.text


def test_a_cp1252_file_is_still_read_as_cp1252(build_corpus) -> None:
    """The guard on the fix: BOM detection must not swallow ordinary files.

    0x93 and 0x94 are curly quotes in cp1252. This file has no BOM, so it must
    still take the candidate-encoding path rather than being forced through a
    UTF-16 decoder that would happily produce garbage of the right length.
    """
    files = build_corpus({"notes.txt": b"\x93Termination\x94 notice is 30 days.\n"})

    result = read_file(files["notes.txt"])

    assert result.status is ReadStatus.OK
    assert "Termination" in result.text
    assert "30 days" in result.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_readers_text_encoding.py -v`
Expected: the two UTF-16 tests FAIL with `assert ReadStatus.UNREADABLE is ReadStatus.OK`. The cp1252 test PASSES already.

- [ ] **Step 3: Add the byte-order marks to config**

In `backend/verity/engine/config.py`, beside `TEXT_ENCODINGS`:

```python
#: Byte-order marks that identify a UTF-16 file, and the codec each implies.
#:
#: Checked *before* the candidate encodings in ``TEXT_ENCODINGS``, not appended
#: to them. Appending would not work: latin-1 maps all 256 byte values, so a
#: UTF-16 file already "decodes successfully" -- the bytes just come back
#: interleaved with NULs. Detection therefore has to happen before decoding,
#: because afterwards the evidence is gone.
#:
#: Detection is by BOM rather than by sniffing, so that a cp1252 file of even
#: length is never forced through a UTF-16 decoder that would produce
#: plausible-looking garbage.
UTF16_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)
```

- [ ] **Step 4: Check for a byte-order mark before decoding**

In `backend/verity/engine/ingestion/readers/source.py`, import `UTF16_BOMS` alongside the other names from `config`, then change `decode_text`:

```python
def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode bytes to text, and say which encoding won.

    Returns ``(text, encoding)``. This cannot fail: Latin-1 maps all 256 byte
    values, so the final candidate always succeeds. An undecodable file
    therefore yields mojibake rather than an error, which is the right
    trade-off at this layer -- mojibake still tokenises and can still be
    matched against a query. Whether the result is *text at all* is a separate
    question, answered by :func:`looks_like_text`.

    A byte-order mark is honoured first. Without that check a UTF-16 file --
    which is what Notepad writes when told to save as "Unicode" -- decodes as
    cp1252 into text interleaved with NULs, and is then rejected as not text.
    """
    for bom, encoding in UTF16_BOMS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(encoding), encoding

    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # Unreachable while Latin-1 is in TEXT_ENCODINGS; kept so the function is
    # total even if that list is ever trimmed.
    return raw.decode("utf-8", "replace"), "utf-8 (with replacement)"
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && TMPDIR=/tmp python -m pytest tests/test_readers_text_encoding.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 6: Run the whole suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: PASS. Any other failure means a file that used to decode a particular way now decodes differently — investigate which encoding it actually is before changing anything, because the candidate list is ordered by specificity for a reason.

- [ ] **Step 7: Commit**

```bash
git add backend/verity/engine/config.py \
        backend/verity/engine/ingestion/readers/source.py \
        backend/tests/test_readers_text_encoding.py
git commit -m "Read UTF-16 text files instead of discarding them

TEXT_ENCODINGS ends with latin-1, which maps all 256 byte values, so a
UTF-16 file decoded "successfully" as cp1252 into text interleaved with
NULs and was then rejected wholesale as not text. Notepad writes UTF-16
when told to save as Unicode, so this discarded an ordinary user's
ordinary document.

Detection is by byte-order mark, before the candidate encodings rather
than among them: once decoding has happened the evidence is gone. BOM
detection rather than sniffing means a cp1252 file of even length is
never forced through a UTF-16 decoder that would produce plausible
garbage, and a test pins that.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: Rewrite AGENTS.md for the product

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the spec's decisions and hard invariants I1–I4.
- Produces: nothing importable.

- [ ] **Step 1: Rewrite the file**

`AGENTS.md` currently declares the challenge contract as this repository's source of truth, names a base image and a VRAM floor that no longer apply, and states a citation-path convention for a submission that will never be made. Retarget it at the product. Keep, verbatim in spirit:

- **Rule: never fabricate.** No invented document contents, no invented citations, no claimed test results.
- **Rule: never cite a file we did not successfully read.** An unreadable or encrypted file is not a source. This survives as invariant I2.
- **Rule: never implement a whole stage in one uncontrolled change.** Small, verified steps.
- **Rule: never download large model weights or start a large build without asking first.**
- **The local-environment facts**, which are still true and still cost time when forgotten: the 9p mount and `chmod` being a no-op, `TMPDIR=/tmp` for pytest, Python 3.12 locally, the `~/venvs/mc3` venv.

New content:

- **Source of truth** becomes the spec at `docs/superpowers/specs/`, with `docs/superpowers/plans/` for the work in flight. `CONTRACT.md`, the brief and `mc3-corpus/` move to `legacy/` and stop being authoritative.
- **What the project is**: a document Q&A product. Users upload documents, ask questions in plain language, and get answers citing the exact file and location.
- **The four invariants** I1–I4 from the spec, quoted exactly — they are the rules most likely to be violated by a well-meaning change.
- **The repository layout** from the spec's §14.

- [ ] **Step 2: Verify no stale claim survives**

Run: `grep -n "ROCm\|48 GiB\|60 GiB\|VRAM\|/app/\|600 s\|query_01\|rocm/pytorch" AGENTS.md`
Expected: no matches. Every one of those belongs to the challenge.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "Rewrite AGENTS.md for the product

It declared the challenge contract as this repository's source of truth,
which is now false and actively misleading: it names a mandated base image,
a VRAM floor and a citation-path convention for a submission that will never
be made.

The rules that survive the pivot are kept -- never fabricate, never cite a
file we did not read, never implement a stage in one uncontrolled change --
along with the local-environment facts, which are still true.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: Move challenge material to legacy/

**Files:**
- Move: `CONTRACT.md`, `AMD AI League — Match 3 Challenge Brief.pdf`, `README.txt`, `sample-questions.json`, `selfcheck.py`, `setup.sh`, `mc3-starter-kit (1).zip`, `IMPLEMENTATION_PROGRESS.md`, `mc3-corpus/`, `scripts/`, `starter/` → `legacy/`
- Modify: `backend/tests/conftest.py` (`CORPUS_DIR`)

**Interfaces:**
- Consumes: `CORPUS_DIR` and `REPO_ROOT` from `conftest.py`.
- Produces: `legacy/mc3-corpus/`, still reachable by the tests that use it as a realistic corpus.

- [ ] **Step 1: Move**

```bash
cd /mnt/d/rag
mkdir -p legacy
git mv CONTRACT.md legacy/CONTRACT.md
git mv "AMD AI League — Match 3 Challenge Brief.pdf" "legacy/AMD AI League — Match 3 Challenge Brief.pdf"
git mv README.txt legacy/README.txt
git mv sample-questions.json legacy/sample-questions.json
git mv selfcheck.py legacy/selfcheck.py
git mv setup.sh legacy/setup.sh
git mv "mc3-starter-kit (1).zip" "legacy/mc3-starter-kit (1).zip"
git mv IMPLEMENTATION_PROGRESS.md legacy/IMPLEMENTATION_PROGRESS.md
git mv scripts legacy/scripts
git mv mc3-corpus legacy/mc3-corpus
git mv starter/Dockerfile legacy/Dockerfile
```

Each of these belongs to the submission: the contract and brief define it, `selfcheck.py` is the harness's own checks, `sample-questions.json` is its worked examples, `setup.sh` and the starter zip are its scaffolding, `scripts/fetch_model.py` downloads a model nothing loads any more, `scripts/score_sample.py` scores against answers nothing produces, and `IMPLEMENTATION_PROGRESS.md` tracks stages of a challenge that closed.

Then remove whatever remains of `starter/` — every module under `starter/app/` has moved or been deleted by now.

- [ ] **Step 2: Repoint the corpus constant**

In `backend/tests/conftest.py`:

```python
CORPUS_DIR = REPO_ROOT / "legacy" / "mc3-corpus"
```

- [ ] **Step 3: Run the suite**

Run: `cd backend && TMPDIR=/tmp python -m pytest -q`
Expected: PASS, including `test_readers_real_corpus.py` and `test_walker.py`. A failure naming a missing path means a test hardcoded `mc3-corpus` — repoint it through `CORPUS_DIR`.

- [ ] **Step 4: Confirm the corpus was not modified**

Run: `git status --short legacy/mc3-corpus/`
Expected: only renames (`R`). The corpus is test data; a content change is a bug.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Move the challenge material to legacy/

Kept rather than deleted: it documents why the engine looks the way it does,
and the sample corpus remains a genuinely useful robustness fixture -- it is
the only corpus here containing an encrypted PDF, an unreadable file, a
superseded revision, an unknown binary and OCR images. Only its authority
changes, not its usefulness.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Plan Self-Review

**Spec coverage.** Spec §14 Phase 1 asks for: name the package (Task 1), kill bare-name imports (Tasks 2, 6), delete the CLI/GPU guard/local model/container paths (Tasks 3, 4), keep parsers/chunking/normalisation/BM25/RRF (Task 2 moves all of them), prune tests and state each deletion (Tasks 5, 6 — each deletion names what it covered), and restructure in place keeping git history (Task 2 uses `git mv`). Spec §16's deletion table maps one-to-one onto Tasks 3–5. `AGENTS.md` (D16) is Task 9; `legacy/` is Task 10. Task 8 fixes a defect found while verifying the plan's own claims.

**Deliberately not covered here.** The chunking merge step (spec §7) is absent on purpose. Mixing a behaviour change into a refactor is how a plan loses the ability to tell a regression from an edit, and the merge changes `chunks_for`'s output — so it belongs in the next plan, against a green suite. The token-ceiling measurement (spec Phase 0) is likewise its own work.

**Placeholder scan.** No "TBD", "handle edge cases", or "similar to Task N". Every code step carries the code.

**Type consistency.** `ReadResult.failure(...)` and `ReadResult.of(...)` are used as `models.py` defines them (`status`, `detail`, `spans`, `rel_path`, `media_type`). `ReadStatus.EMPTY`/`UNREADABLE`/`OK`, `MediaType.*`, `read_file`, `reader_for`, `walker.discover`, and `chunks_for` are all named as they exist in the moved modules. `EmbeddingProvider` keeps the property names `model_id` and `dimension`, which the spec's `LLMProvider` does not share — deliberate, they are different interfaces. `decode_text` keeps its `(str, str)` return type and its totality across Task 8.

**Corrections made during this review.**

1. An earlier draft claimed `FileNotFoundError` escaping a reader was a live bug. It is not: `readers.read_file()` catches `Exception` and returns a failure value, and `app.py:128` reads through it. The claim came from calling `text.read()` directly, bypassing the dispatcher. Review Focus item 5 now states what is actually true, and Task 7 tests the guarantee rather than repairing a defect that does not exist.
2. The draft's Review Focus included a speculative filename-safety item that belongs to the upload boundary (Phase 3/4), not the engine. Replaced with the UTF-16 defect below, which is engine-owned and verified.
3. **UTF-16 text files are discarded, and this is verified.** `"Section 4.2 Termination\nNotice period is 30 days.\n".encode("utf-16")` read through `read_file` returns `ReadStatus.UNREADABLE` with the detail "more than 10% of the decoded bytes are control characters". `TEXT_ENCODINGS` has no UTF-16 entry, and latin-1 maps all 256 byte values, so the file decodes "successfully" into text interleaved with NULs and is rejected afterwards. Task 8 fixes it, with a BOM-detection approach and a regression test that a cp1252 file is unaffected.
4. The draft's Task 9 move list omitted `sample-questions.json`, `selfcheck.py`, `setup.sh`, `scripts/`, `IMPLEMENTATION_PROGRESS.md` and the starter zip, and missed a root `pytest.ini` whose `testpaths = tests` would point at a directory that no longer exists. Both fixed — the config move is in Task 5, the rest in Task 10.
