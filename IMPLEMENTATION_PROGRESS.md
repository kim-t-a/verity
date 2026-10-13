# IMPLEMENTATION_PROGRESS.md

Living progress log. Updated at the end of every stage. Durable rules live in
`AGENTS.md`; this file records what was actually done, run, and left undone.

**Last updated:** 2026-10-13

---

## Stage checklist

| Stage | Name | Status |
|---|---|---|
| 0 | Inspect and plan | ✅ **complete** |
| 1a | Corpus discovery and format detection | ✅ **complete** |
| 1b | Document readers (pdf, docx, xlsx, csv, text, code) | ⬜ not started |
| 1c | Image reader interface + normalised chunks | ⬜ not started |
| 2 | Persistent indexing (chunk store, embedding interface, fake provider) | ⬜ not started |
| 3 | Keyword retrieval | ⬜ not started |
| 4 | Vector retrieval | ⬜ not started |
| 5 | Hybrid retrieval | ⬜ not started |
| 6 | Answer extraction and citations | ⬜ not started |
| 7 | Image and vision support | ⬜ not started |
| 8 | Local integration (self-check, sample scoring) | ⬜ not started |
| 9 | AMD GPU validation (notebook, ~3 h) | ⬜ not started |

Stage 1 was split into three steps rather than done in one change, per the rule
against uncontrolled changes: discovery (1a), text readers (1b), and the image
interface plus chunker (1c).

---

## Stage 0 — Inspect and plan ✅

**Goal:** understand the challenge, the starter kit and the environment before
writing any code.

### Findings

- `/mnt/d/rag` contained **only** the brief PDF and `mc3-starter-kit (1).zip`. No code
  had been extracted. It was **not a git repository**.
- The starter kit's CLI layer (`starter/app/app.py`) is **already correct**: argparse
  handles `--index` / `--corpus` / `--query-id` / `--query`, and the output JSON is
  written to the right path with the right schema. `index()` and `answer()` are stubs;
  `answer()` returns `("", [], 0.0)`, which is a contract-valid *refusal* and therefore
  scores 0/10 but never crashes.
- `AGENTS.md`, `IMPLEMENTATION_PROGRESS.md` and `docs/` did not exist — they are
  outputs of this project, not inputs.
- Consulted documents: `CONTRACT.md`, `README.txt`, `starter/Dockerfile`,
  `starter/app/app.py`, `starter/app/requirements.txt`, `selfcheck.py`, `setup.sh`,
  `sample-questions.json`, every file in `mc3-corpus/`, and the full 8-page brief PDF.

### Environment facts established by experiment

| Check | Result |
|---|---|
| Project filesystem | `/mnt/d` is **9p/drvfs** (Windows `D:`) |
| `chmod 000` on `/mnt/d` | **No-op — read still succeeds.** Verified by experiment. |
| `chmod 000` on native fs (`/tmp`) | Correctly raises `Permission denied` |
| Laptop Python | 3.12.3 (image is 3.14 — wheel-availability risk) |
| System `pip` | absent, **but** `python3 -m venv` bootstraps `pip 24.0` offline ✅ |
| `git` | present |
| `docker` | present (Docker Desktop for Windows, invoked from WSL) |
| `unzip`, `tesseract` | **absent** |
| Third-party Python packages | none installed |

### Decisions taken

1. **Project root is `/mnt/d/rag`**, kit extracted **flattened** (kit root = repo
   root), so paths match the brief's layout and `setup.sh`'s relative paths.
2. **`setup.sh` was NOT run.** The committed corpus is pristine; the unreadable-file
   and empty-directory conditions are created by test fixtures on a native filesystem.
3. **venv lives at `~/venvs/mc3`** (native ext4), outside the repo, because the repo
   is on a slow 9p mount.
4. **The unreadable-file test must run against a native-filesystem copy** of the
   corpus. It cannot be exercised in place.
5. **Citations will be corpus-relative**, despite the brief saying absolute is also
   accepted, because `CONTRACT.md` and `selfcheck.py` both steer that way.
6. Architecture is **deterministic-first, model-assisted**: locate evidence, lift the
   exact span, prove the citations. A model is needed for images and disambiguation,
   not as the primary engine.

### Files created or modified

Extracted from `mc3-starter-kit (1).zip` (21 files, 8 directories, byte-for-byte
verified), plus:

- `.gitignore`
- `AGENTS.md`
- `IMPLEMENTATION_PROGRESS.md`

### Commands actually run

```bash
# Inspect zip structure and check for path traversal
python3 -I -c '...zipfile listing...'

# Extract, flattened, refusing to overwrite
python3 -I /tmp/mc3-tools/extract_kit.py "/mnt/d/rag/mc3-starter-kit (1).zip" /mnt/d/rag

# Verify byte-for-byte fidelity
python3 -I /tmp/mc3-tools/verify_kit.py "/mnt/d/rag/mc3-starter-kit (1).zip" /mnt/d/rag
```

### Test results

| # | Test | Result |
|---|---|---|
| 1 | Extracted tree matches the zip (CRC-32 + size, 21 files / 8 dirs) | ✅ pass |
| 2 | `python3 -I starter/app/app.py` with no args → argparse error | ✅ pass |
| 3 | `--index` runs and creates the index dir | ✅ pass |
| 4 | `--query` writes a schema-valid refusal JSON | ✅ pass |
| 5 | `mc3-corpus/vendor/internal_audit.txt` readable (setup.sh not applied) | ✅ pass |
| 6 | `git log` shows the pristine initial commit; tree clean | ✅ pass |
| 7 | `~/venvs/mc3/bin/python -m pip --version` → pip 24.0 | ✅ pass |
| 8 | `pytest --collect-only` | ⏸ **not run** — pytest not yet installed (needs a small network download, awaiting approval) |

### Known bugs / unresolved risks

- **Python 3.12 (laptop) vs 3.14 (image).** No dependency may be considered safe
  until it installs inside the mandated image. Resolve in the first 30 minutes of
  the GPU session.
- **Image size budget.** Base is 29.2 GiB of 60 GiB, leaving ~30 GiB for weights and
  dependencies — roughly a 7–8B model at bf16. The binding limit is disk, not VRAM.
- **Per-question process ⇒ model reload × 10.** Must be solved with a resident worker
  behind a unix socket, with a fail-fast fallback.
- **"Section 2 of *All About the Challenges*"** is referenced by `CONTRACT.md` as
  authoritative for container rules but **was not supplied**. Worth requesting.
- **Citation paths discrepancy** between the brief and `CONTRACT.md` (see AGENTS.md).

### Work that requires the AMD GPU notebook

Everything that depends on real ROCm behaviour: base-image pull, wheel availability on
Python 3.14, `torch.__version__` staying `+rocm`, model load time, VRAM peak, real
vision accuracy, uncompressed image size, and `selfcheck.py`.

### Next step

**Stage 1b — Text document readers.** See the Stage 1a section below.

---

## Stage 1a — Corpus discovery and format detection ✅

**Goal:** enumerate every readable file in the corpus, identify its format, and —
critically — prove that one bad file cannot cost us the rest of the corpus.

### Files created

| File | Responsibility |
|---|---|
| `starter/app/config.py` | Central constants (`PROBE_BYTES`) |
| `starter/app/models.py` | `MediaType`, `SkipReason`, `DiscoveredFile`, `SkippedEntry`, `DiscoveryResult` |
| `starter/app/ingestion/__init__.py` | Package marker |
| `starter/app/ingestion/filetypes.py` | Format detection from leading bytes + extension |
| `starter/app/ingestion/walker.py` | Defensive traversal + failure ledger |
| `tests/conftest.py` | Puts `starter/app` on `sys.path`; builds the hazard corpus fixture |
| `tests/test_filetypes.py` | 25 unit tests |
| `tests/test_walker.py` | 17 traversal tests |
| `tests/test_import_graph.py` | 10 tests guarding the container import graph and the CLI contract |
| `pytest.ini` | Test discovery config |
| `requirements-dev.txt` | Dev-only dependencies (never shipped in the image) |

### Key design decisions

1. **Readability is proven by opening the file, not by checking permissions.**
   `os.access()` consults mode bits that root ignores and that mounted Windows
   drives fabricate without enforcing. Opening the file is the only test that
   cannot lie. The same probe read captures the leading bytes for format detection,
   so the two checks cost one syscall.
2. **Format comes from the bytes; the extension is only a hint.** `%PDF-`, PNG,
   JPEG and ZIP signatures are decisive. DOCX and XLSX are both ZIP containers and
   cannot be told apart from their header, so the extension resolves those two.
   Plain text has no signature, so there the name is all the evidence there is.
3. **A file we cannot identify is reported `UNKNOWN`, never guessed.** The corpus
   deliberately contains one; discovering it correctly means labelling it honestly.
4. **Traversal order is forced to be deterministic** by sorting directory and
   filename lists in place as `os.walk` descends, and by sorting both output lists.
   This is what makes "the unreadable file sorts first" a testable claim.
5. **Directory symlinks are recorded, not followed** — they can leave the corpus or
   form a cycle that never terminates.
6. **A missing corpus root is a recorded failure, not an empty corpus.** Otherwise
   a configuration error would look identical to a corpus that happened to be empty.
7. **The container's import graph is asserted, not assumed.** The tests import
   `config`, `models` and `ingestion.walker` by *bare name*, exactly as `/app/app.py`
   does, and a guard rejects any module that reaches for a `starter.` prefix which
   does not exist at runtime. Without this, the suite could pass locally while the
   container died on its first import.

### Commands actually run

```bash
/home/alfin/venvs/mc3/bin/python -m pip install pytest     # pytest 9.1.1
cd /mnt/d/rag && /home/alfin/venvs/mc3/bin/python -m pytest -v
```

### Test results

```
52 passed in 5.51s
```

**No tests skipped.** That matters: the permission tests contain a guard that skips
when the filesystem does not enforce `chmod 000`, and they did *not* skip — so the
unreadable-file case was genuinely exercised, not silently passed over.

Two checks were run to prove the new guard tests are not vacuous:

- The `starter.`-prefix guard scans **6 real modules** (`app.py`, `config.py`,
  `models.py`, `ingestion/__init__.py`, `ingestion/filetypes.py`,
  `ingestion/walker.py`) — it is not iterating an empty list.
- Mutating a throwaway copy with `from starter import config` made the guard
  **fire** (`['bad_module.py']`), so the predicate detects the failure it claims to.
  The mutation was done in `/tmp`, never inside `starter/app/`.

The fixture corpus puts the unreadable file at `vendor/a_unreadable.txt`, which sorts
**first** in its directory. A walk that aborted on it would lose `b_encrypted.pdf` and
`c_unknown.dat`; the test asserts both survive.

### Observation on the real sample corpus

```
13 readable file(s) [code=1, csv=1, docx=1, jpg=1, log=1, pdf=3, png=1,
                     text=2, unknown=1, xlsx=1], 0 skipped, 8 directories, 1 empty
```

- All 13 files discovered; `archive/` correctly reported as the empty directory.
- `vendor/telemetry_capture.dat` correctly classified `unknown`.
- `vendor/supplier_agreement_ENCRYPTED.pdf` classified `pdf` — *discovery* identifies
  the format; deciding it cannot be read is the reader's job (Stage 1b).
- Every identified file's bytes agree with its extension, so extension-based routing
  is safe for this corpus.

### Verified environment finding

Running the permission tests here **passed**, which confirms the earlier finding from
the other direction: `chmod 000` is enforced on native Linux (where pytest's
`tmp_path` lives) but is a **no-op on `/mnt/d`** (the drvfs mount holding the project).
The guard in `conftest.py` detects this automatically, so the suite behaves honestly
in either location.

### Known bugs / unresolved risks

- None new. The Stage 0 risks all still stand (Python 3.12 vs 3.14, image size budget,
  per-question model reload).

### Next step

**Stage 1b — text document readers**: `ingestion/readers/` for PDF, DOCX (paragraphs
**and** tables), XLSX (all worksheets), CSV, TXT/LOG and Python source, each attaching
provenance (page / sheet / row / line) to the text it extracts. The PDF reader must
detect `/Encrypt` and report the file as unreadable rather than attempting to open it.
