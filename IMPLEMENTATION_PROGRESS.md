# IMPLEMENTATION_PROGRESS.md

Living progress log. Updated at the end of every stage. Durable rules live in
`AGENTS.md`; this file records what was actually done, run, and left undone.

**Last updated:** 2026-10-14

---

## Stage checklist

| Stage | Name | Status |
|---|---|---|
| 0 | Inspect and plan | ✅ **complete** |
| 1a | Corpus discovery and format detection | ✅ **complete** |
| 1b | Document readers (pdf, docx, xlsx, csv, text, code) | ✅ **complete** |
| 1c | Image reader interface + normalised chunks | ✅ **complete** |
| 2 | Persistent indexing (chunk store, embedding interface, fake provider) | ✅ **complete** |
| 3 | Keyword retrieval | ✅ **complete** |
| 4 | Vector retrieval | ✅ **complete** |
| 4b | The embedding model and the real backend | ✅ **complete** |
| 5 | Hybrid retrieval | ✅ **complete** |
| 6 | Answer extraction, citations and refusal | ✅ **complete** |
| 7 | Image and vision support | 🟡 **reader wired and reading; engine not yet in the image** |
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

---

## Stage 1b — Text document readers ✅

**Goal:** turn each discoverable file into a sequence of `TextSpan`s that carry
their own provenance, so that a later stage can both *find* a value and *prove*
which file it came from. Nothing here does retrieval or answering; this stage
only converts bytes into located text.

### Files created

| File | Responsibility |
|---|---|
| `starter/app/ingestion/readers/__init__.py` | Dispatcher: `reader_for(media_type)`, `read_file(discovered)`; the never-raises boundary |
| `starter/app/ingestion/readers/source.py` | Bounded reads, encoding fallback, the not-text guard |
| `starter/app/ingestion/readers/text.py` | `.txt` / `.log` / source: one span per non-blank line |
| `starter/app/ingestion/readers/tabular.py` | Shared table rendering for CSV, XLSX and DOCX tables |
| `starter/app/ingestion/readers/delimited.py` | CSV with delimiter sniffing |
| `starter/app/ingestion/readers/xmlpart.py` | Safe OOXML part loading (DOCTYPE/ENTITY refusal, size caps) |
| `starter/app/ingestion/readers/docx.py` | Paragraphs **and** tables from `word/document.xml` |
| `starter/app/ingestion/readers/xlsx.py` | Every worksheet; both string storage methods |
| `starter/app/ingestion/readers/pdf.py` | `/Encrypt` detection first, then text extraction with ToUnicode CMaps |
| `tests/document_fixtures.py` | Builders for synthetic DOCX/XLSX/PDF |
| `tests/test_readers_common.py`, `test_readers_text.py`, `test_readers_tabular.py`, `test_readers_docx.py`, `test_readers_pdf.py`, `test_readers_real_corpus.py` | 113 new tests |

### Files modified

| File | Change |
|---|---|
| `starter/app/models.py` | Added `ReadStatus`, `TextSpan`, `ReadResult` |
| `starter/app/config.py` | Added `MAX_TEXT_BYTES`, `MAX_ZIP_MEMBER_BYTES`, `TEXT_ENCODINGS`, `UNREMARKABLE_ENCODINGS`, `MAX_CONTROL_CHARACTER_RATIO`, `CELL_SEPARATOR` |
| `tests/conftest.py` | Added the `build_corpus` fixture (real walker over `tmp_path`) |

### Key design decisions

1. **A reader never raises; failure is a value.** Every reader returns a
   `ReadResult` whose `status` is `ok`, `empty`, `encrypted`, `unreadable` or
   `unsupported`. `read_file` still wraps each call in `except Exception` as a
   backstop. This is rule 9 ("never let one bad file abort the corpus walk")
   expressed at the type level rather than as a convention — the caller cannot
   forget to handle a failure, because the failure *is* the return value.

2. **`ok` and `empty` are deliberately distinct.** A file that opened and
   parsed but held no text is `empty`; a file whose reader crashed is
   `unreadable`. Collapsing them would make a reader bug look identical to a
   genuinely blank document, and the corpus hazards include both.

3. **`/Encrypt` is checked before any extraction.** A naive extractor on the
   sample's AES-256 PDF returns a *silently empty page* — no exception, no
   text. That is the worst possible failure (rule 11: refuse only when evidence
   genuinely fails; here the evidence is present but locked). The reader
   therefore tests for `/Encrypt` first and reports `encrypted`, so question 10's
   refusal rests on a positively-identified cause.

4. **Provenance is structural, not a token window.** One span per row, line or
   paragraph, each carrying `location` (`page 3 line 7`, `sheet 'RMA Parts' row 2`,
   `line 42`). Chunking by a fixed token window would cut rows in half and
   detach a value from the label that gives it meaning.

5. **Table rows are rendered with their column names attached**
   (`fixed_in=4.3.2`). A query asking what release *fixed* a bug has the word
   "fixed" only in the header; a row rendered as bare values leaves that word
   with nothing to match. **Labels are added; cell values are never rewritten,
   substituted or paraphrased.**

6. **A first row is only a header if it is at least 3 columns wide and fully
   populated.** A two-column table is far more likely key/value — reading
   `Maximum junction temperature | 94 C` as a header would render the next row
   as `Maximum junction temperature=Board power`, inventing a relationship
   between two unrelated measurements.

7. **Sparse spreadsheet cells are filled to their declared column.** A row
   storing only cell `C2` still means column C; reading cells in the order they
   appear would place that value under column A — a value silently attributed
   to the wrong label, which is worse than a miss.

8. **Both XLSX string storage methods are supported.** The sample corpus uses
   inline strings and has no `sharedStrings.xml`; Excel-authored workbooks
   normally use the shared table. A reader supporting one only returns an empty
   column for the other *without erroring*.

9. **`utf-8-sig` leads the encoding list, not plain `utf-8`.** A BOM is valid
   UTF-8, so plain `utf-8` succeeds on a BOM'd file and leaves U+FEFF glued to
   the first token on line 1 — which is exactly where a part number or a value
   lives. `utf-8-sig` strips a mark when present and behaves identically when
   not, so it dominates.

10. **OOXML parsing refuses DOCTYPE and ENTITY.** `xmlpart` rejects a document
    type definition in the first 4 KiB and checks each member's *declared*
    uncompressed size before decompressing, defeating billion-laughs and
    zip-bomb shapes. The corpus is untrusted input.

11. **`looks_like_text` rejects binaries that happen to be named `.txt`.**
    Indexing mojibake puts noise into retrieval that no query can match. The
    threshold is a control-character *ratio* (`10%`), not a single-byte check,
    because real text files contain the occasional stray control byte.

### Commands actually run

```bash
# Full suite, native ext4 venv, isolated interpreter
/home/alfin/venvs/mc3/bin/python -I -m pytest tests/ -q

# Reader results over the real sample corpus
/home/alfin/venvs/mc3/bin/python -I -c "..."   # walker.discover -> read_file, per file

# Isolate the PDF operator scan on a synthetic document
/home/alfin/venvs/mc3/bin/python -I -c "..."   # dump _CONTENT_OPERATORS matches
```

### Test results

```
165 passed in 11.77s
```

The last run includes 113 reader tests. Before the fixes below the suite stood
at 5 failed / 160 passed; both remaining causes were diagnosed and resolved
rather than papered over.

**Real sample corpus, every file read through the real walker:**

| Status | Spans | Chars | File |
|---|---:|---:|---|
| ok | 11 | 508 | `engineering/ingest_service.py` |
| ok | 6 | 380 | `engineering/meridian_release_notes.txt` |
| ok | 7 | 728 | `logs/prod_inference_2026-09-02.log` |
| ok | 8 | 562 | `planning/roadmap_fy27.docx` |
| unsupported | 0 | 0 | `specs/backplane_pinout.png` |
| ok | 8 | 321 | `specs/tq40_datasheet_r1_WITHDRAWN.pdf` |
| ok | 13 | 518 | `specs/tq40_datasheet_r2.pdf` |
| unsupported | 0 | 0 | `support/asset_label.jpg` |
| ok | 6 | 743 | `support/bug_database.csv` |
| ok | 6 | 672 | `support/rma_parts.xlsx` |
| ok | 2 | 108 | `vendor/internal_audit.txt` |
| encrypted | 0 | 0 | `vendor/supplier_agreement_ENCRYPTED.pdf` |
| unsupported | 0 | 0 | `vendor/telemetry_capture.dat` |

67 spans, no `unreadable` — every reader succeeded on the input it claimed.

### Two real bugs the synthetic fixtures caught

Both were invisible in the sample corpus. This is the argument for building
documents in the tests rather than testing only against the kit's files.

**1. A BOM survived decoding (real bug, silent).** `TEXT_ENCODINGS` originally
led with `utf-8`. A BOM is valid UTF-8, so the plain codec succeeded and left
U+FEFF attached to the first token of line 1. Fixed by leading with
`utf-8-sig`. The corpus could not have shown this — no sample file has a BOM.

**2. The PDF `Tj` matcher never matched an ordinary string (real bug, latent).**

```python
# before -- requires a literal "(" at the start of every repetition
rb"|(?P<single>(?:\([^()\\]|\\.)*\)|<[0-9A-Fa-f\s]*>)\s*Tj"
# after
rb"|(?P<single>\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>)\s*Tj"
```

The old form could only ever succeed by skipping the repetition entirely and
matching a bare `)` before `Tj`, so a `(text) Tj` show operator produced **no
text at all** — confirmed by tracing the scan, which yielded `single=b')'` and
`lines: []` while the content stream and font map were both correct. Both
sample-corpus PDFs use `TJ` *arrays*, so the `Tj` branch was never exercised
and the corpus stayed green. `document_fixtures.pdf_bytes` emits `Tj`, which
exposed it. With the fix the same trace yields
`single=b'(Maximum junction temperature 94 C)'`, and the corpus PDFs (arrays)
and the synthetic PDF (`Tj`) now cover both display operators.

**One stale assertion, not a code bug:** `test_utf8_is_preferred` asserted
`encoding == "utf-8"`, which the deliberate `utf-8-sig`-first ordering made
false. The assertion was updated to pin the new behaviour; the property it
tests — that UTF-8 is not mis-decoded as cp1252 — is unchanged and still
asserted on the decoded text.

### Decoy integrity verified over the corpus

Extraction must not damage the near-misses that make the corpus hard:

- withdrawn r1 retains `105`; r2 holds `94` and not `105`
- release notes still lack `4.3.2` (the decoy file)
- the bug database still carries `fixed_in=4.3.2`
- the legacy `ORR-FAN-1990-A` row and the real `ORR-FAN-2214-B` row both survive
- `DEFAULT_BATCH_TIMEOUT_S` reads `180`; the `60` appears only in the comment
- the encrypted file's value does not leak from any readable file

### Known limitations (accepted for now)

- **DOCX:** headers, footers, footnotes, endnotes and comments are not read —
  only `word/document.xml`. Revisit if the graded corpus puts values there.
- **XLSX:** date-formatted cells are read as their underlying serial number, not
  as a date; formulas are read as their cached result, not their expression.
- **PDF:** form-field values, annotations and text emitted as vector outlines
  (rather than as text operators) are not recovered. Scanned PDFs contain no
  text operators at all and will come back `empty` — a vision-path candidate.
- **PDF CMaps:** ToUnicode is honoured where present; the sample PDFs have none
  and decode as Latin-1, which is correct for them but not in general.
- **CSV:** a single-column file is rendered positionally, and ragged rows keep
  extra cells under generated `column_N` names.

### Next step

**Stage 1c — image reader interface and the normalised chunker.** Define the
interface images will satisfy (so rule 8 holds: the vision path reports itself
as unavailable rather than inventing text), and convert `TextSpan`s into the
retrieval chunks that Stage 2 will index.

---

## Stage 1c — Image reader interface and normalised chunks ✅

**Goal:** two things Stage 2 depends on. First, make the image gap *visible and
named* rather than an absence (rule 8). Second, provide the one normalisation
the contract defines and turn read results into the chunks the index will store.

Neither part does any retrieval. This stage decides what retrieval will be able
to see.

### Files created

| File | Responsibility |
|---|---|
| `starter/app/normalisation.py` | **The** normalisation — `normalise()` and `contains()`. Used everywhere, defined once. |
| `starter/app/chunking.py` | `ReadResult` → `Chunk`s; content filtering and splitting |
| `starter/app/ingestion/readers/image.py` | PNG/JPEG header parsing; reports `VISION_UNAVAILABLE` |
| `tests/test_normalisation.py` | 25 tests |
| `tests/test_chunking.py` | 25 tests |
| `tests/test_readers_image.py` | 23 tests |

### Files modified

| File | Change |
|---|---|
| `starter/app/models.py` | Added `ReadStatus.VISION_UNAVAILABLE` and `Chunk` |
| `starter/app/config.py` | Added `MAX_IMAGE_HEADER_BYTES`, `MAX_CHUNK_CHARS`, `CHUNK_BREAK_LOOKBACK` |
| `starter/app/ingestion/readers/__init__.py` | Registered the image reader for PNG and JPG |
| `tests/document_fixtures.py` | Added `png_bytes`, `jpeg_bytes` |
| `tests/test_readers_common.py`, `tests/test_readers_real_corpus.py` | Updated for the new status |
| `tests/test_import_graph.py` | Guard widened to the new top-level modules and the readers package |

### Key design decisions

1. **A recognised image is `VISION_UNAVAILABLE`, not `UNSUPPORTED`.** An unknown
   `.dat` and a `.png` are both unreadable today, but only one is a gap we are
   *required* to close. Collapsing them would let that obligation hide inside a
   general-purpose failure state: "three unsupported files" reads as a corpus
   of odd formats, while "one unknown binary, two images with no vision
   backend" names the unmet requirement — and keeps naming it after Stage 7
   lands, when absence and presence must stay distinguishable.

2. **The image reader reports what it *did* establish.** It parses the header
   and states the real format and pixel dimensions. "A valid 1000x560 PNG we
   cannot read" is a different finding from "this file claims to be a PNG and
   is not", and only the first is a vision gap — the second is `UNREADABLE` and
   blames nothing on the missing model.

3. **Image metadata is never emitted as spans.** Dimensions go in `detail`
   only. `1400` is a fact about a file, not something the corpus says; putting
   it in a span would let an unrelated question match that file and cite it,
   and every extra citation scores zero. **A file may be cited only once we
   have read what it says.** The test asserts `spans == ()` on every path.

4. **No vision backend interface yet.** Stage 7 knows which model, which
   resolutions, whether a caption suffices. An interface designed without those
   answers would be replaced rather than implemented. The single seam is marked
   in the code.

5. **One normalisation, in one place.** `AGENTS.md` requires it and the reason
   is failure mode, not tidiness: two implementations that drift produce a
   system that matches on one path and misses on another, and both paths look
   correct read in isolation. `contains()` exists so callers cannot normalise
   one side and forget the other — a half-normalised comparison looks almost
   right.

6. **Characters are deleted, not replaced.** `4.3.2`, `4-3-2` and `432` all
   normalise to `432`, as the contract specifies. This is why normalisation is
   a *comparison* form and never a display form: an answer reported to the
   grader is always the value exactly as the corpus spells it, which is why
   `Chunk.text` is stored verbatim and `Chunk.normalised` is derived.

7. **The step order inside `normalise` is load-bearing**, in two places:
   - Degree signs are deleted **before** NFKC, because NFKC folds `º` to the
     letter `o` — deleting first leaves nothing to fold.
   - Separators are deleted **after** NFKC, so a full-width `．` becomes `.`
     first and is then removed along with the ASCII form.

8. **`contains` never matches an empty needle.** Normalisation reduces `---` and
   `""` to the empty string, and an empty needle is a substring of everything —
   which would report a value that was never there. This matters most for
   refusals, where the expected answer is empty by definition.

9. **Chunk content is tested on the normalised form.** A span of `-----`
   normalises to nothing and is not content; `4.3.2` normalises to `432` and
   is. Testing the raw text would keep the first and, in another format, might
   drop the second.

10. **`chunks_for` takes a `ReadResult`, not spans.** `is_usable` is the single
    gate for "may this file be treated as evidence"; routing through it means
    an encrypted or unreadable file cannot reach the index because a caller
    forgot to check. A test asserts that a failure *carrying* spans still
    yields nothing.

11. **Splitting trims to a space when one is near, and hard-slices when not.**
    The lookback is bounded so a minified file does not cause a scan of the
    whole prefix for a break that is not there. Each piece is labelled
    `part N of M`, or a diagnostic pointing at "line 42" would be ambiguous
    about which part held the value.

### Commands actually run

```bash
/home/alfin/venvs/mc3/bin/python -I -m pytest tests/ -q
# walker -> read_file -> chunks_for over mc3-corpus, printing status/chunks/spans
# independent dimension check with file(1) and a second standalone parser
```

### Test results

```
243 passed in 12.13s
```

### A third real bug, caught by the JPEG fixture

The `SOF` parser read height and width from the wrong offsets, because I placed
them as offsets into the *payload* while the segment walk leaves the cursor at
the **length** field:

```
+0 .. +1   length (where the cursor is)
+2         precision
+3 .. +4   height     <- was read at +5
+5 .. +6   width      <- was read at +7
+7         component count
```

The symptom was a 1200x800 image reported as `768x1200` — a swapped pair of
plausible numbers, no exception, no empty result. Exactly the silent-degradation
shape as the two Stage 1b bugs. Fixed to `+3`/`+5`, with the table written into
the code so the next reader does not have to re-derive it.

Verified independently rather than against my own fixture, using `file(1)` and a
second standalone parser on the real corpus images:

```
specs/backplane_pinout.png: PNG image data, 1000 x 560, 8-bit/color RGB
support/asset_label.jpg:    JPEG image data, ... baseline, precision 8, 760x440
```

The reader reports `1000x560` and `760x440`. A real JPEG from a real producer
now parses correctly — evidence that does not depend on the fixture I wrote.

### The pipeline over the real corpus

| Status | Chunks | Spans | File |
|---|---:|---:|---|
| ok | 11 | 11 | `engineering/ingest_service.py` |
| ok | 6 | 6 | `engineering/meridian_release_notes.txt` |
| ok | 7 | 7 | `logs/prod_inference_2026-09-02.log` |
| ok | 8 | 8 | `planning/roadmap_fy27.docx` |
| vision_unavailable | 0 | 0 | `specs/backplane_pinout.png` → *PNG image, 1000x560 pixels* |
| ok | 8 | 8 | `specs/tq40_datasheet_r1_WITHDRAWN.pdf` |
| ok | 13 | 13 | `specs/tq40_datasheet_r2.pdf` |
| vision_unavailable | 0 | 0 | `support/asset_label.jpg` → *JPG image, 760x440 pixels* |
| ok | 6 | 6 | `support/bug_database.csv` |
| ok | 6 | 6 | `support/rma_parts.xlsx` |
| ok | 2 | 2 | `vendor/internal_audit.txt` |
| encrypted | 0 | 0 | `vendor/supplier_agreement_ENCRYPTED.pdf` |
| unsupported | 0 | 0 | `vendor/telemetry_capture.dat` |

67 spans, 67 chunks — no span was dropped by the content filter, which is the
expected result for a corpus of real documents and confirms the filter is not
over-eager. The three unreadable files contribute nothing to the index by three
distinct, separately-reported routes.

### Known limitations

- **Image dimensions are all we have.** No OCR, no captioning. The two sample
  image questions (7 and 8) remain unanswerable, which is correct and asserted.
- **The chunker preserves reader boundaries.** It does not merge a sentence
  split across two PDF lines, nor split a table into one chunk per row-set. If
  retrieval recall proves weak in Stage 5, this is the first place to revisit.
- Carried from Stage 1b: DOCX headers/footers, XLSX date serials, PDF form
  fields and scanned pages.

### Next step

**Stage 2 — persistent indexing.** A chunk store written by `--index` and read
by each `--query` process, plus the embedding interface with a fake provider, so
the index can be built and exercised with no model and no network.

---

## Stage 2 — Persistent indexing ✅

**Goal:** the channel between the two invocations. `--index` runs once and each
of the ten questions runs in a **new process**, so nothing held in memory during
indexing survives to answering. The index is therefore not an optimisation — it
is the only thing that crosses the boundary, and everything the pipeline will
later need (chunks, provenance, and vectors when a real model lands) has to fit
through it.

### Files created

| File | Purpose |
|---|---|
| `starter/app/chunkstore.py` | The persisted index: manifest, chunks, vectors; save, load, staleness check |
| `starter/app/embeddings/__init__.py` | The provider protocol and the rule-7 guard around the CPU backend |
| `starter/app/embeddings/hashing.py` | The CPU-only stand-in embedder (not a submission candidate) |
| `tests/test_chunkstore.py` | 32 tests — round trips and, mainly, the refusal paths |
| `tests/test_embeddings.py` | 19 tests — the guard, the protocol, cross-process determinism |
| `tests/test_index_cli.py` | 11 tests — the real CLI as subprocesses, both entry points |

### Files modified

- `starter/app/config.py` — index layout constants, embedding-provider names,
  `INDEX_FORMAT_VERSION`, `HASHING_DIMENSION`.
- `starter/app/app.py` — `index()` now walks → reads → chunks → optionally
  embeds → saves; `answer()` loads and verifies the index, then refuses, because
  retrieval does not exist yet. A `_report_index()` writes what the pass found to
  stderr.
- `tests/test_import_graph.py` — three modules added to the bare-name guard
  (`chunkstore`, `embeddings`, `embeddings.hashing`); 15 → 18 tests.

### Key design decisions

**Three index files, and the manifest is written last.** `chunks.jsonl` and
`vectors.bin` land first, `manifest.json` last. The manifest is then the commit
marker: an interrupted `--index` leaves a directory with no manifest, which reads
as "no usable index" rather than as a half-built one that looks complete. Three
files rather than one because they have different failure modes — a damaged
chunk record costs one line, not the whole store, and JSONL stays inspectable
with `head` and `grep` during development.

**The fingerprint is names and sizes, not contents.** It must be computable at
query time in milliseconds, before anything is read; hashing file contents would
spend the per-question budget re-doing the work `--index` already did. Names and
sizes catch what it is for — an index built over one corpus and queried against
another — and the digest is taken over sorted `(rel_path, size)` pairs so
directory listing order cannot make an unchanged corpus look stale. Fields are
separated by NUL and newline so `("ab","c")` and `("a","bc")` cannot collide;
that case has its own test.

**A stale index raises rather than warns.** A mismatch means the answers would be
drawn from text that is not in the corpus being asked about, and they would
arrive with citations that look right. `StaleIndexError` subclasses
`IndexIncompatible` so callers that only handle the general case still catch it.
Refusing costs a startup pass (`--index` is ungraded); a confident wrong answer
costs the question.

**Every load is validated, not trusted.** Format version, chunk count against
the manifest, and vector buffer size against `count × dimension × 4`. The size
check is the one that matters most: a truncated buffer misaligns every vector
after the cut, so each one silently belongs to the wrong chunk. All of these are
tested as expected exceptions.

**Text is stored raw, never normalised.** The store keeps what the corpus says;
`Chunk.normalised` is derived at comparison time. A change to the normaliser then
needs no rebuild, and an answer can never be quoted in a form the corpus does not
use. There is a test asserting exactly this.

**The vector/text pairing lives in one object.** `ChunkIndex` holds both, because
read as two separate files they can be paired wrongly — and a vector matched to
the wrong chunk yields a citation that looks entirely plausible and is false.

**The rule-7 guard has three independent defences.** The default provider is the
real one, so an empty environment can never select the CPU path. Naming the
stand-in is not enough — a second, separately-named variable must also be `1`,
and only the exact string `"1"` counts, because `"true"` and `"yes"` are what a
person reaches for next. And the index records its `provider_id`, so a stand-in
index announces itself in `manifest.json`.

**The stand-in uses `blake2b`, not `hash()`.** Python salts string hashing per
interpreter, so a `hash()`-based embedder would produce unrelated vectors in the
`--index` process and in each of the ten `--query` processes. Nothing would
raise. This is the same silent-wrong-output shape as the three bugs Stage 1
found, so it is tested directly rather than merely avoided.

### Commands actually run

```bash
# the guard, and cross-process determinism
python -m pytest tests/test_embeddings.py tests/test_chunkstore.py -q
# the CLI as the harness runs it
python -m pytest tests/test_index_cli.py -q
# everything
python -m pytest -q

# a real index over the sample corpus, then a query in a separate process
MC3_INDEX_DIR=/tmp/mc3-index MC2_OUTPUT_DIR=/tmp/mc3-out \
  python3 starter/app/app.py --index mc3-corpus
MC3_INDEX_DIR=/tmp/mc3-index MC2_OUTPUT_DIR=/tmp/mc3-out \
  python3 starter/app/app.py --corpus mc3-corpus --query-id query_01 --query "..."
```

### Test results

```
tests/test_chunkstore.py   32 passed
tests/test_embeddings.py   19 passed
tests/test_index_cli.py    11 passed
------------------------------------------
whole suite               308 passed in 43.53s
```

308 − 243 = **65 added** here: 32 chunk store, 19 embeddings, 11 CLI, and 3
widening the import guard to cover the three new modules.

### A fourth bug, and it was in my own test

The CLI test originally ran the app under `python3 -I`, borrowing the isolation
discipline from `AGENTS.md`. That invocation **cannot work**:

```
$ python3 -I starter/app/app.py --index mc3-corpus
ModuleNotFoundError: No module named 'chunking'
```

`-I` implies `-P`, which drops the *script's own directory* from `sys.path`, and
the Dockerfile copies `starter/app/` to `/app/` precisely so the application can
import its own modules by bare name. The harness runs `python3 /app/app.py`
without `-I`.

The test was the bug, not the code — but the shape is the familiar one: it
looked like a strictness improvement and would have made the suite test an
invocation that does not exist, while the real one went unexercised. The fix is
to run what the harness runs and replace the child's environment wholesale
instead, which is the isolation `-I` was being reached for. The reason is now in
the helper's docstring so it is not "helpfully" re-added. Note this is a
different thing from the `AGENTS.md` rule, which is about ad-hoc scripts reading
untrusted corpus files — not about the application, whose import graph depends on
the script directory being present.

### Verified on the real sample corpus

Index pass (0.5 s, well inside the 600 s startup budget):

```
indexed mc3-corpus: 13 files, 67 chunks, embeddings=none (lexical only), 0.5s
  file status: encrypted=1, ok=9, unsupported=1, vision_unavailable=2
  not indexed -> encrypted: vendor/supplier_agreement_ENCRYPTED.pdf
  not indexed -> unsupported: vendor/telemetry_capture.dat
  not indexed -> vision_unavailable: specs/backplane_pinout.png
  not indexed -> vision_unavailable: support/asset_label.jpg
```

`manifest.json`:

```json
{
  "format_version": 1,
  "created_utc": "2026-10-13T01:24:55+00:00",
  "corpus_fingerprint": "004a59f59adcacae8215e2ee25670b4c8a13afb095e896bdaa6fad455eaca7af",
  "chunk_count": 67,
  "file_count": 13,
  "provider_id": null,
  "dimension": null,
  "status_counts": {"encrypted": 1, "ok": 9, "unsupported": 1, "vision_unavailable": 2},
  "skipped_count": 0
}
```

The four unreadable files are named individually rather than folded into a
single failure count, which is what makes the vision gap stay visible instead of
blending into a general-purpose "some files failed".

Cross-process round trip — `--query` in a **separate process** from `--index`,
writing the contract's three keys:

```
$ ... python3 starter/app/app.py --corpus mc3-corpus --query-id query_01 --query "..."
exit=0
{"answer": "", "citations": [], "confidence": 0.0}
```

The empty answer is honest: retrieval arrives in Stages 3–6, and the alternative
— a stub returning something that looks like an answer — would be a fabricated
result. With the vector backend enabled by both switches, `vectors.bin` came out
at **68 608 bytes**, exactly `67 × 256 × 4`.

Query-side load cost, warmed, measured 5× each:

| Index | Load |
|---|---|
| 67 chunks, lexical only | 0.5–0.6 ms |
| 67 chunks + 256-dim vectors | 0.9–1.4 ms |

Under a millisecond and a half against a 30 s per-question budget — the load is
not a design constraint at this corpus size, which is worth knowing before
optimising it.

### Proof the determinism test can fail

A test that cannot fail is worse than no test, so the `hash()` trap was
reproduced deliberately with a throwaway embedder built on the built-in:

```
PYTHONHASHSEED=0     → [0,0,0,1,1,0,0,0,1,0, ...]   token "junction" in bucket 3
PYTHONHASHSEED=12345 → [0,0,0,0,0,0,0,0,0,0, ...]   token "junction" in bucket 19
```

Different seeds, different buckets, no error. That is precisely what the shipped
embedder must not do, and `test_vectors_are_identical_across_processes` compares
two seeded subprocesses so it would catch a regression to `hash()`.

### Known limitations

- **The index is lexical-only right now.** `provider_id` is `null` and no
  `vectors.bin` is written, because no GPU embedding backend exists yet. Keyword
  retrieval (Stage 3) can be built against this index without waiting for one.
- **`--index` is not incremental.** It rebuilds from scratch. At 0.5 s over 13
  files that is free; the graded corpus is larger, and this is worth re-measuring
  before it is worth fixing.
- **`corpus_fingerprint` cannot see an edit that preserves size.** A file changed
  in place with the same byte count fingerprints identically. Accepted
  deliberately: the alternative is reading every file per query, which is the
  work the index exists to avoid.
- The stand-in embedder ships in the image. Kept on purpose — it is what makes
  the pipeline smoke-testable end to end inside the container with no model
  download — and the three-defence guard is what stops it being *selected*.

### Next step

**Stage 3 — keyword retrieval.** Given a query, score the 67 chunks and return a
ranked list with provenance, so the answering stage has evidence to work from.
This is the first stage that produces something a question can be answered from,
and it needs no model and no GPU.


---

## Stage 3 — Keyword retrieval ✅

### Goal

Turn a question into a ranked list of chunks, each carrying the provenance
needed to cite it. No model, no GPU, no answer extraction — this stage decides
**what to look at**, not what the answer is.

Retrieval is judged differently from the final answer. A retrieval miss cannot
be recovered later, because nothing downstream can read a chunk that was never
returned. So the failure to guard against is silent: a ranking that looks
reasonable and has the right chunk at position eleven.

### Files created or modified

| File | Change |
|---|---|
| `starter/app/tokenisation.py` | **created** — the two-tier tokenizer |
| `starter/app/retrieval/__init__.py` | **created** — `RetrievedChunk`, `Retriever` protocol, `rank()` |
| `starter/app/retrieval/keyword.py` | **created** — two-field BM25 `KeywordRetriever` |
| `starter/app/config.py` | modified — `MIN_TOKEN_LENGTH`, `BM25_K1`, `BM25_B`, `RETRIEVAL_LIMIT`, `PATH_FIELD_WEIGHT` |
| `starter/app/app.py` | modified — `answer()` loads the index and runs retrieval |
| `tests/test_tokenisation.py` | **created** — 27 tests |
| `tests/test_retrieval_keyword.py` | **created** — 41 tests |
| `tests/test_import_graph.py` | modified — `tokenisation`, `retrieval`, `retrieval.keyword` added to the bare-name guard |

### Key design decisions

**The tokenizer is two-tier, and that is the load-bearing decision of the stage.**
A conventional tokenizer splits on every non-alphanumeric character, which
destroys exactly the strings this challenge asks for: `4.3.2` becomes `4`, `3`,
`2` — three of the most common terms in any technical corpus, matching hundreds
of chunks about equally and identifying none. So each run of text emits:

1. its **parts** (`ORR-FAN-2214-B` → `orr`, `fan`, `2214`), always, and
2. its **joined form** (`orrfan2214b`), but *only when the run contains a digit*.

Both rules are needed and neither subsumes the other. Parts alone lose `4.3.2`
entirely, since its parts are single characters and get dropped. The joined form
alone would fuse `field-replaceable` into `fieldreplaceable`, making the corpus's
"field replaceable" — written with a space in `support/rma_parts.xlsx` —
mutually invisible with a question that hyphenates it. The digit test is what
keeps the two apart: `4.3.2` has digits and gets both forms; `dual-rotor` does
not, so no word is invented that neither corpus nor question uses.

**BM25 rather than counting shared words.** Two mechanisms, both measured below.
*Inverse document frequency* weights a rare term far above a common one, which is
what lets a question be matched on its one distinctive term. *Term-frequency
saturation with length normalisation* stops a chunk winning by repeating itself
or by being long. The smoothed idf form is `ln(1 + (N - df + 0.5)/(df + 0.5))`;
the `+1` inside the logarithm is not cosmetic, because without it a term
appearing in every chunk would score a negative weight — a chunk *penalised* for
containing a query word.

**The file path is a second field, not concatenated onto the text.** Questions
name documents by kind — "the production log", "the datasheet" — and a log file's
lines say `ERROR E7731: thermal throttle engaged` and never the word "log".
Concatenating would inflate every chunk by its path length, and paths vary a
lot, which is a length bias with nothing to do with content. As a field of its
own it gets its own length statistics and its own weight.

**Results carry `matched_terms`, with `path:`-prefixed entries for path hits.**
A chunk found only by its filename says nothing about the question; the same
score from content terms means it does. That distinction is what makes a
surprising ranking debuggable.

**Deliberately absent: stemming, synonyms, phrase matching.** Each trades
precision for recall, and here the wrong answer and the missing answer cost
exactly the same — so a change that finds more chunks but ranks the right one
lower is a loss.

### Commands actually run

```bash
# Full suite, native ext4 venv
python -m pytest tests/ -q
# End to end over the real sample corpus
MC3_INDEX_DIR=/tmp/mc3-idx MC2_OUTPUT_DIR=/tmp/mc3-out \
  python3 starter/app/app.py --index mc3-corpus
MC3_INDEX_DIR=/tmp/mc3-idx MC2_OUTPUT_DIR=/tmp/mc3-out \
  python3 starter/app/app.py --corpus mc3-corpus --query-id query_01 \
    --query "What is the maximum junction temperature of the TQ-40?"
```

### Test results

```
379 passed in 40.56s
```

Up from 308 at the end of Stage 2: +27 tokenizer, +41 retrieval, +3 widening the
import guard.

### Verified behaviour on the real sample corpus

Index pass: 13 files, 67 chunks, 0.2 s, status counts
`encrypted=1, ok=9, unsupported=1, vision_unavailable=2` — all four unreadable
files named individually. Query 1:

```
retrieved 10 chunks for 'What is the maximum junction temperature of the TQ-40?'
   1. specs/tq40_datasheet_r2.pdf          [page 1 line 4] score=12.57  matched=junction, maximum, path:tq40, temperature
   2. specs/tq40_datasheet_r1_WITHDRAWN.pdf [page 1 line 5] score=12.53  matched=junction, maximum, path:tq40, temperature
   3. specs/tq40_datasheet_r2.pdf          [page 1 line 1] score=8.38   matched=40, path:tq40, tq, tq40
```

**The 0.04 margin between rank 1 and rank 2 is the supersession trap, live.**
The current revision wins, but by almost nothing — and it wins partly by
accident, because `_WITHDRAWN` is a longer filename and BM25's length
normalisation penalises it. This is **not** supersession handling and must never
be presented as such. Retrieval is right to return both: both chunks genuinely
match, and a retriever that quietly dropped one would be hiding a judgement it
is not equipped to make. **Stage 6 must demote withdrawn revisions explicitly.**

### What was measured, and what could not be

This is the part worth reading carefully, because most of it is negative.

**Baseline rank of each expected citation** (top 10, six answerable single-hop
questions):

| Question | BM25 | Naive word-overlap |
|---|---|---|
| 1 maximum junction temperature | **1** | 5 |
| 2 TQ-60 customer sampling quarter | 1 | 1 |
| 3 fan assembly part number | 1 | 1 |
| 4 firmware version for ORR-1847 | 1 | 1 |
| 5 error code when throttle engages | 1 | 1 |
| 6 default batch timeout | 1 | 1 |
| 9 multi-hop (log + CSV) | 6 | 6 |

**The honest headline: on this corpus, BM25's measured advantage over the naive
scorer is one question.** A scorer that just counts shared words also puts six of
these seven files at rank 1 and all seven in the top ten. So the sample corpus
cannot demonstrate the value of idf or length normalisation — it is too small
and its questions name their answers too directly.

BM25 is still the right choice, for a reason this measurement cannot show: the
graded corpus is **larger**, and both mechanisms pay off with scale. With 13
files a common word appears in a handful of chunks and the damage is small; with
hundreds, a common word matches hundreds of chunks about equally and idf is the
only thing separating the one that matters. That is reasoning, not evidence, and
it is labelled as such.

**`PATH_FIELD_WEIGHT = 0.5` is a stated judgement, not a fitted value.** A sweep
over 0.0 / 0.25 / 0.5 / 1.0 / 2.0 produced differences indistinguishable from
noise on seven questions — the parameter is not identifiable from this sample.
Its measurable contribution is exactly one rank position: with the field
disabled, question 1's citation drops from rank 1 to rank 2 and question 9's
from 6 to 7. Half a content match is a defensible reading of "a filename is
weaker evidence than the contents", and it is documented as a judgement so a
later reader does not mistake it for a tuned constant.

**Two candidate improvements were tried and rejected on measurement:**

- *df-cutoff (corpus-derived stopwords):* changes nothing at all. Not added.
- *Plural folding:* changes nothing at all, ranks identical. Not added.

Both were rejected rather than kept "just in case", because a mechanism that
cannot be shown to help is a mechanism that can only break later.

### The two regression tests, and one of them is weak

`test_every_answerable_sample_question_reaches_its_citation` asserts each
expected citation appears in the top 10. **This is a weak test and is labelled as
one in its docstring** — the naive baseline passes it too. It is kept as the
floor beneath the sharper test, because it catches the failures that actually
happen: an empty index, a tokenizer that stopped matching, a reader that started
returning nothing, a file dropped from the index.

`test_the_right_file_comes_first_for_the_sample` asserts rank 1. It is fitted to
six questions and is framed as a **canary, not a requirement**: the challenge
needs the file within reach, not first, but a weak retriever and a strong one are
otherwise indistinguishable here. Its docstring says explicitly that when a
later stage legitimately reorders these, the response is to look at the new order
and update the expectation deliberately — never to tune a constant until it goes
green, which would destroy the only signal it carries.

Both are in `tests/`, reading the sample corpus the way
`tests/test_readers_real_corpus.py` already does. Neither is reachable from
answering code: the graded corpus shares none of these paths.

### Two bugs found and fixed before committing

1. **A zero-weight field still returned hits.** `score_into` multiplied the
   score by `weight`, so `path_weight=0.0` produced a score of exactly nothing —
   but it had already registered the chunk as *matched*, so it came back ranked
   last, occupying a slot in the top ten that a real match could have used. Fixed
   by dropping zero-weight fields at construction: a field carrying no weight is
   not searched, rather than searched and multiplied by zero.

2. **`test_vocabulary_size_counts_both_fields` expected 4, actual 5.** I had not
   traced `gamma/delta.txt` through the tokenizer. The `/` separates (three
   tokens: `gamma`, `delta`, `txt`), giving 5 across both fields. The
   expectation was wrong, not the code.

### Known limitations

- **Question 9's citation sits at rank 6, not 1.** The question says "production
  log" and "incident"; the log's own lines contain neither word, and "incident"
  instead matches a comment in `ingest_service.py`. This is the known weak spot
  and it is **not** to be fixed by tuning retrieval — Stage 6's multi-hop
  traversal is what closes it, by reading the log first and then searching for
  what it names.
- **Superseded revisions are not demoted.** Both `tq40_datasheet_r2.pdf` and
  `tq40_datasheet_r1_WITHDRAWN.pdf` are returned, 0.04 apart. Stage 6's job.
- **Length-1 tokens are dropped entirely**, so a question containing a bare `b`
  or a single-digit value cannot match. Judged acceptable: no value in this
  corpus is one character.
- **The `Retriever` protocol is not yet exercised by a second implementation.**
  It exists so the vector and hybrid retrievers can satisfy it without
  inheriting; until Stage 4 writes one, that is design intent rather than a
  verified property. `isinstance` against the runtime-checkable protocol passes.

### Next step

**Stage 4 — vector retrieval.** Embed the chunks (through the provider interface
already built in Stage 2), embed the query, return cosine-nearest chunks behind
the same `Retriever` protocol. This is where the GPU is first actually used, and
where the local hashing embedder stops being adequate for anything but plumbing
tests.

---

## Stage 4 — Vector retrieval ✅

### Goal

Rank chunks by *meaning* rather than by shared words: embed the question, embed
every chunk, and compare the angles. Keyword retrieval cannot connect "how hot
may this part get" to a chunk saying "maximum junction temperature", because no
word is shared.

### Files created or modified

| File | Change |
|---|---|
| `starter/app/retrieval/vector.py` | **created** — `VectorRetriever` and `VectorUnavailable` |
| `starter/app/config.py` | modified — `MIN_VECTOR_SIMILARITY` |
| `starter/app/app.py` | modified — `answer()` runs both retrievers and reports each |
| `tests/test_retrieval_vector.py` | **created** — 26 tests |
| `tests/test_import_graph.py` | modified — `retrieval.vector` added to the bare-name guard |

### Key design decisions

**The central guard: the question must be embedded by the same backend that
embedded the chunks.** Two vectors are only comparable if one model produced
both. A question embedded by model A and chunks embedded by model B land in
unrelated coordinate systems, and the cosine between them is a meaningless
number that looks exactly like a score. Nothing raises. Retrieval returns ten
chunks in a confident order and every one is arbitrary.

This is the same failure shape as the index-format and rule-7 guards -- code
that runs cleanly and produces quietly wrong output -- so `VectorRetriever`
refuses to be *constructed* unless `provider.model_id` and `provider.dimension`
match what the manifest recorded. Refusing loudly costs a keyword-only fallback;
mixing silently costs every question.

**Normalising chunk vectors once, at construction.** Cosine is a dot product
divided by both lengths, so scaling every vector to unit length up front removes
the division from the per-query path. It also means a backend that returns
unnormalised vectors is handled correctly rather than being trusted. `HashingEmbedder`
already normalises, so for it the operation is idempotent -- but a real model may
not, and assuming it did would rank by magnitude, letting a long chunk beat a
relevant one.

**`MIN_VECTOR_SIMILARITY = 0.0`.** Cosine zero means right angles -- no
relationship -- and a negative value means *less* related than a random vector.
Returning those would pad the result list with noise, and a padded list is what
tempts an answering stage into citing something unsupported. Zero is chosen
rather than fitted because it is the only threshold that means something on its
own; a tuned cutoff would need measurements from a real model, and fitting one to
the hashing stand-in would be fitting a number to a model nobody will use.

**A zero vector is not an error.** A chunk with no tokens embeds to all zeros,
which has no direction and therefore no angle with anything. Its similarity is
zero and it never scores. "Normalising" it would be a division by zero.

**Cheap, silent-wrong-answer guard: the query vector's width is checked.** Zipping
two vectors of different lengths in Python stops at the shorter one, so a backend
returning the wrong width would produce a *plausible* cosine from whichever
dimensions happen to line up, and report it as a full match.

**`matched_terms` is empty, deliberately.** Dense retrieval can report *how
similar* but never *why*. A BM25 result says which terms matched; a cosine says
only that a number is larger. That is a real loss of debuggability, and the test
asserting the empty tuple exists so nobody mistakes it for a bug and "fixes" it
by inventing terms that never matched.

### Commands actually run

```bash
python -m pytest tests/ -q
# lexical-only index, then a query against it
MC3_INDEX_DIR=/tmp/mc3-idx MC2_OUTPUT_DIR=/tmp/mc3-out python3 starter/app/app.py --index mc3-corpus
# vector index via the permitted local stand-in, then a query
MC3_EMBEDDING_PROVIDER=hashing MC3_ALLOW_LOCAL_BACKEND=1 ... --index mc3-corpus
```

### Test results

```
406 passed in 44.15s
```

Up from 379: +26 vector retrieval, +1 widening the import guard.

### Verified behaviour

**A lexical-only index reports why it has no vectors, and still answers.**

```
$ ... --index mc3-corpus
indexed mc3-corpus: 13 files, 67 chunks, embeddings=none (lexical only), 0.5s
$ ... --query "What is the maximum junction temperature of the TQ-40?"
  vectors unavailable: no embedding backend is configured, so this index is lexical-only
keyword: 10 chunks for 'What is the maximum junction temperature of the TQ-40?'
   1. specs/tq40_datasheet_r2.pdf [page 1 line 4] score=12.5660
   2. specs/tq40_datasheet_r1_WITHDRAWN.pdf [page 1 line 5] score=12.5287
```

**With vectors, both retrievers run and both are reported.**

```
vector: 10 chunks for 'What is the maximum junction temperature of the TQ-40?'
   1. specs/tq40_datasheet_r1_WITHDRAWN.pdf [page 1 line 5] score=0.4070
   2. specs/tq40_datasheet_r2.pdf          [page 1 line 4] score=0.4070
   3. planning/roadmap_fy27.docx           [paragraph 6]  score=0.3048
```

### All three mismatch paths, exercised through the real CLI

| Situation | Result |
|---|---|
| Vector index, no backend configured for the query | `vectors unavailable: no embedding backend is configured...` → keyword only |
| Lexical index, backend configured for the query | `vectors unavailable: the index holds no vectors...` → keyword only |
| Local backend named without the second switch | `vectors unavailable: ... also requires MC3_ALLOW_LOCAL_BACKEND=1` → keyword only |

Every case is loud on stderr, none crashes, none silently mixes vectors. The
output contract holds (`{"answer": "", "citations": [], "confidence": 0.0}`).

### The finding that matters most from this stage

**The vector path ranks the withdrawn revision *first*.** Both datasheets contain
identical text, so the stand-in embedder gives them identical vectors -- tied at
`0.4070` -- and the corpus-position tie-break then puts
`tq40_datasheet_r1_WITHDRAWN.pdf` ahead of `tq40_datasheet_r2.pdf`, because
`r1` sorts before `r2`.

BM25 accidentally escaped this: length normalisation penalised the longer
`_WITHDRAWN` filename, so the current revision won by 0.04. The vector path has
no such accident and no supersession awareness whatsoever. **This makes
Stage 6's explicit demotion of withdrawn revisions more urgent, not less** -- and
it is a concrete reason not to let vector results displace keyword results
before that demotion exists.

### Measured: the pure-Python scan, and why no numpy

The cosine scan is a plain Python loop. Measured over three runs each, taking
the minimum:

| Chunks | Dimension | Search (s) |
|---|---|---|
| 67 | 256 | 0.001 |
| 670 | 768 | 0.022 |
| 6 700 | 768 | 0.143 |
| 20 000 | 768 | 0.422 |
| 20 000 | 1024 | 0.564 |

At 20 000 chunks of 768 dimensions the scan costs **0.42 s against a 30 s
per-question budget** -- 1.4% of it. The graded corpus would have to be an order
of magnitude larger than any plausible reading of "larger" for this to matter,
so numpy is not added. That is a measurement, not a guess, and it is the number
that should be re-checked before reaching for a dependency.

### What this stage cannot show

**Nothing here demonstrates that dense retrieval helps.** The only embedding
backend that exists is the CPU-only hashing stand-in, which has no notion of
meaning -- it is the hashing trick over word tokens wearing a vector's clothes.
So the tests verify the *mechanism*: the arithmetic, the ordering, the
determinism, and above all the refusals. Retrieval *quality* is a property of the
real model and is unmeasured until one is on the device.

That is the next decision, and it is a decision rather than a task because it
requires downloading model weights, which rule 10 forbids doing unasked.

### Known limitations

- **No real embedding backend exists.** `provider_from_environment()` returns
  `None` for the default `"gpu"`, so a container run today builds a lexical-only
  index. This is the honest state, and the reason `--index` still succeeds.
- **Both retrievers run but nothing fuses them.** Choosing between them, or
  combining them, is Stage 5. Running both now is what keeps the vector path
  exercised on every real query instead of only in unit tests.
- **The query-side cost of a real model is unmeasured**, and it is the single
  biggest risk in the project: each question is a new process, so a naive
  implementation loads the model ten times inside ten 30-second budgets.
- **`embeddings.provider_from_environment()` was not changed by this stage.**
  Adding the real backend is a change to that factory plus a new module under
  `embeddings/`, and it is deliberately not bundled with the retrieval
  mechanism.

### Next step

**Two things, and the first is a decision for you.**

1. **Choose the embedding model** (and confirm the download). It must be
   small enough to ship inside the 60 GiB image and to load fast enough that ten
   fresh processes each fit in 30 s -- or be paired with a resident worker.
   Nothing else in the project can be measured on the GPU until this is settled,
   which makes it the highest-value thing to resolve before the notebook session.
2. **Stage 5 — hybrid retrieval:** fuse keyword and vector rankings behind the
   one `Retriever` interface. Its shape depends on (1), because how much weight
   the vector half deserves depends on what is doing the embedding.

---

## Stage 4b, part 1 — the embedding model, fetched and verified

### Goal

Put real model weights inside the Docker build context, so that the image can
carry them. The container has no network at evaluation time, so a model that is
downloaded at run time hangs and scores zero. Fetching has to happen now, on the
laptop, and the result has to ship in the image.

### Files

| File | Change |
|---|---|
| `scripts/fetch_model.py` | new — a stdlib-only downloader with structural verification |
| `starter/models/bge-small-en-v1.5/` | new — 129 MB of weights and tokenizer, git-ignored, Docker-visible |

No file under `starter/app/` changed. The shipped payload is still stdlib-only.

### Design decisions

**`BAAI/bge-small-en-v1.5`, pinned to a commit.** 33.4 M parameters, 384
dimensions, float32, 129 MB on disk. Small enough that the image budget is not
in question and fast enough that a per-process load has a chance of fitting the
30 s budget. Pinned to commit `5c38ec7c…` rather than to `main`: `main` is a
moving target, and a re-upload upstream would silently change the weights under
a build that is supposed to be reproducible.

**`model.safetensors`, never `pytorch_model.bin`.** The repository publishes both,
holding identical weights. A `.bin` file is a Python pickle, and loading a pickle
executes whatever code it names; `safetensors` is a JSON header plus a flat
buffer of numbers, with no code path at all. The choice costs nothing and removes
a class of risk permanently, rather than trusting a file to stay benign.

**Stdlib `urllib`, not `huggingface_hub`.** Six GET requests do not justify
`requests` + `tqdm` + `filelock` + `fsspec`, every one of which would also need a
Python 3.14 wheel in the image. `scripts/fetch_model.py` imports nothing outside
the standard library.

**An explicit file list, not a wildcard.** A downloader that mirrors whatever is
upstream cannot distinguish a needed config from a test fixture — and
`pytorch_model.bin` is precisely the file a wildcard would sweep in.

**Verification without torch.** `check_safetensors()` parses the header with
`struct` + `json` and cross-checks the weight shapes against `config.json`. A
check that requires the very dependency it is validating proves nothing.

### Commands

```
python3 -I scripts/fetch_model.py --destination starter/models/bge-small-en-v1.5
```

### Results

```
  hidden size: 384
  vocabulary:  30522 tokens
  tensors:     200
  max tokens:  512
model ready at starter/models/bge-small-en-v1.5
```

Confirmed on disk: 129 MB total, `model.safetensors` 133,466,304 bytes, and
`find` for `*.bin`/`*.pt`/`*.pth` returns nothing. The float32 dtype checks out
arithmetically — 30,522 × 384 embedding rows plus 12 transformer layers is about
33.4 M parameters, and 33.4 M × 4 bytes ≈ 133 MB, matching the file exactly.

### What this does NOT establish

**The numbers are unverified.** The header parses and the shapes agree with the
config; nothing here shows the weights produce *correct* embeddings. That needs a
forward pass, which needs torch, which is not installed. Structural soundness and
correct behaviour are different claims and only the first has been earned.

### A finding that changes Stage 4b

`1_Pooling/config.json` states:

```
"pooling_mode_cls_token":  true,
"pooling_mode_mean_tokens": false
```

**This model pools at `[CLS]`, not by averaging token vectors.** Mean pooling is
the habit most tutorials teach for turning a BERT into a sentence embedder, and
using it here would produce 384-dimension unit vectors, a working retriever, and
quietly worse rankings — the silent-degradation shape, not an error.

`modules.json` confirms the full declared pipeline:
`Transformer → Pooling(CLS) → Normalize`. The trailing `Normalize` is consistent
with `VectorRetriever`, which normalises defensively rather than trusting a
backend to have done it, so there is no double-normalisation hazard.

Two smaller facts from the same read:

- **`do_lower_case: true`** — the tokenizer lowercases, matching `tokenise()`.
  No casing mismatch between the lexical and dense halves.
- **`max_position_embeddings: 512`** makes `MAX_CHUNK_CHARS = 2000` *checkable*,
  and the existing comment claiming chunks embed "whole without silent
  truncation" is optimistic rather than verified: 2000 characters of dense
  technical prose is plausibly 450–550 wordpiece tokens. **To be measured against
  the real tokenizer, not assumed.**

### Corrections to earlier notes

`starter/Dockerfile` states that the mandated base image supports **both gfx942
(MI300X, CDNA) and gfx1100 (Navi 31, RDNA3)**. The RDNA3 concern recorded earlier
is therefore narrower than written: torch itself is covered. flash-attention,
bitsandbytes and vLLM remain separately unverified on this card.

### Known limitations

- The Dockerfile's `COPY models/ /models/` is still commented out, and
  `requirements.txt` still declares nothing, so the image does not yet carry the
  model. Both are Stage 4b's remaining work.
- There is no `.dockerignore` under `starter/`. Harmless today; worth adding
  before the build context grows further.
- The model has never been executed.

### Next step

Stage 4b, part 2 — the real embedding backend. Blocked on one decision: whether
to install CPU-only torch locally (~2.5 GB) so the backend can actually be run
and the CLS-versus-mean question settled empirically, or to write it unverified
and debug it on the notebook inside a three-hour window. Rule 10 requires asking
before a download of that size.

---

## Stage 4b, part 2 — the real embedding backend

### Goal

Make the default provider a real model on the GPU, so that a container run with
no environment at all stops being silently lexical-only. This closes the gap
Stage 4 named and left open.

### Files

| File | Change |
|---|---|
| `starter/app/embeddings/local_model.py` | new — `LocalModelEmbedder` and `LocalModelUnavailable` |
| `starter/app/embeddings/__init__.py` | `provider_from_environment()` returns the real backend for the default |
| `starter/app/config.py` | model directory, device, pooling, dimension, recipe version, batch size |
| `starter/app/app.py` | `index()` degrades instead of crashing when a backend will not start |
| `starter/app/requirements.txt` | `transformers~=5.19.0`, `safetensors~=0.8.0` |
| `starter/Dockerfile` | `COPY models/ /models/` enabled, placed before the code copy |
| `starter/.dockerignore` | new — excludes `__pycache__`; `models/` deliberately kept |
| `tests/test_embeddings_local.py` | new — 19 tests |
| `tests/test_embeddings.py` | the bare-environment guard restated as a property |
| `tests/test_import_graph.py` | `embeddings.local_model` added |

### Design decisions

**No device fallback.** If the model cannot be placed on the requested device,
`LocalModelUnavailable` is raised; the code never continues on the CPU. A CPU run
is *rejected* by the challenge, so a silent fallback converts a loud failure into
an invisible one. Development without a GPU sets `MC3_EMBEDDING_DEVICE=cpu`
explicitly, making the CPU path something a person typed.

**CLS pooling, matching the checkpoint.** See part 1: the measurement could not
separate the poolings, so this follows the author's declared intent and is
recorded here as a stated judgement rather than a validated one.

**The provider id is derived, not written out.** `EMBEDDING_MODEL_ID` is
assembled from the model name, pooling, dimension and recipe version, so
changing the pooling *cannot* leave the id describing vectors it no longer
produces. `VectorRetriever` compares that id against the manifest, which is what
stops an old index being searched with new vectors.

**`torch` and `transformers` are imported inside `__init__`.** The container
starts ten fresh processes and the import-graph guard imports every module in the
payload; a module-level import would charge every one of them several seconds for
a library it may not call.

**`import embeddings` does not import torch.** This falls out of the above and is
worth stating: it is what keeps the import-graph test fast, and it also avoids a
circular import, since `embeddings/__init__.py` defines the protocol the backend
implements.

### A bug this stage found in existing code

`index()` called `provider_from_environment()` unguarded while `answer()` already
caught `EmbeddingError`, printed the reason and continued keyword-only. The two
paths disagreed, and `index()` had the worse behaviour: a backend that could not
start would abort the whole pass, leaving no index, so all ten questions would
refuse and the submission would score nothing.

Fixed by degrading in `index()` the same way `answer()` does. The failure is
printed and recorded — the manifest's `provider_id` is null — so a lexical-only
index announces itself in two places and is never confused with an intended one.
The rule-7 guard is untouched: the hashing stand-in still requires both of its
switches, and degrading to *no vectors* is not the same as degrading to the CPU
stand-in.

### Commands

```
python3 -m pytest tests/ -q
MC3_EMBEDDING_MODEL_DIR=$PWD/starter/models/bge-small-en-v1.5 \
MC3_EMBEDDING_DEVICE=cpu MC3_INDEX_DIR=/tmp/mc3idx MC2_OUTPUT_DIR=/tmp/mc3out \
  python3 starter/app/app.py --index mc3-corpus
```

### Results

**426 tests pass**, up from 406. Nineteen new, one for the import guard.

Index built with the real model:

```
indexed mc3-corpus: 13 files, 67 chunks, embeddings=bge-small-en-v1.5-cls-d384-v1, 8.1s
  file status: encrypted=1, ok=9, unsupported=1, vision_unavailable=2
```

Every hazard behaved: the encrypted PDF and the unknown binary are skipped, and
both images report `vision_unavailable` rather than disappearing — which is rule
8 holding, and which is why the two image questions currently cannot rank.

Query 1 through the real CLI, a fresh process as the harness runs it. Keyword
puts the correct revision first by 0.037; the vector path puts the **withdrawn**
revision first:

```
keyword: 1. specs/tq40_datasheet_r2.pdf            score=12.5660
         2. specs/tq40_datasheet_r1_WITHDRAWN.pdf  score=12.5287
vector:  1. specs/tq40_datasheet_r1_WITHDRAWN.pdf  score=0.8503
         2. specs/tq40_datasheet_r2.pdf            score=0.8320
```

This is the Stage 3 accidental escape and the Stage 4 stand-in finding,
reproduced together, with the real model, in the real pipeline. Stage 6's
explicit demotion of withdrawn revisions is now a measured requirement.

### The per-question budget, measured

| step | cost |
|---|---|
| `import torch` | 1.39 s |
| `import transformers` | 1.03 s |
| load tokenizer + weights | ~2.5 s |
| whole query process | 8.99 s |

Against a 30 s per-question limit and a 600 s run limit, ten questions cost
roughly 90 s. **The resident worker is not needed**, which retires the largest
architectural risk carried since Stage 4.

This is measured with the weights on the 9p mount, which inflates file reads; the
image's own filesystem should be faster. The conclusion holds either way, since
even a doubling stays inside the budget. It depends on the embedding model being
small and on no generative model being in the answer path — if Stage 6 adds one,
this must be re-measured.

### Known limitations

- **No answer is produced.** The JSON is still `{"answer": "", "citations": [], "confidence": 0.0}`,
  correctly shaped but empty; turning chunks into values is Stage 6.
- **Nothing here runs on a GPU.** Every measurement is CPU. The device guard is
  tested in the direction that is reachable without one — that `cuda` is refused
  when absent — and the positive direction is untested until the notebook.
- **`transformers~=5.19.0` was verified on Python 3.12, not 3.14.** The
  dependency *does* have 3.14-compatible wheels (checked against PyPI), but the
  library was never executed on 3.14. This is the first thing to verify in the
  built image.
- **Dense retrieval has still not been shown to beat keyword retrieval.** The
  one question where they differ, they disagree *wrongly*.
- **The `2_Normalize` module in `modules.json` is not separately applied.** This
  backend normalises in `embed()` and `VectorRetriever` normalises again, so the
  effect is present; the declared module itself is unused, which is fine because
  this backend does not load sentence-transformers.

### Next step

**Stage 5 — hybrid retrieval.** Fuse the two rankings behind the one `Retriever`
interface. The shape is now constrained by measurement rather than by taste: the
vector half currently *loses* the one question where the two disagree, and loses
it to a superseded file, so fusion cannot simply average them and hope.

---

## Stage 5 — hybrid retrieval

**Status: done.** 451 tests pass (up from 426). Fuses the keyword and vector
rankings behind the one `Retriever` interface, using reciprocal rank fusion.

### The files

| File | Change |
|---|---|
| `starter/app/retrieval/hybrid.py` | **new** — `HybridRetriever`, `reciprocal_rank_fusion`, `describe_sources` |
| `starter/app/retrieval/__init__.py` | `RetrievedChunk.sources` field; `rank(..., origins=)` |
| `starter/app/config.py` | `RRF_K = 60`, `FUSION_CANDIDATE_DEPTH = 3` |
| `starter/app/app.py` | `answer()` now runs one fused retriever, not two reported side by side |
| `tests/test_retrieval_hybrid.py` | **new** — 24 tests |
| `tests/test_import_graph.py` | `retrieval.hybrid` added to the bare-name import guard |

### Why RRF and not a weighted sum of scores

The two retrievers are scored on scales with no common unit. Measured on this
corpus: the best BM25 score is about **12.5** and the best cosine is about
**0.85**. Adding those is adding metres to kilograms, and the result would be
decided by BM25 every time. Fixing that needs normalisation and a weight, and
both would be numbers fitted to whichever questions happened to be at hand —
the graded corpus is different and larger.

RRF discards the scores and keeps the positions: a chunk's fused score is the
sum over the retrievers that returned it of `1 / (RRF_K + rank)`. Ranks carry no
units, so there is nothing to normalise and no scale to get wrong. It adds
exactly one constant, kept at the published 60.

### Measured: the disagreement is a dead heat, to the bit

Stage 4b measured the two halves disagreeing on question 1. Fused, the top two
results are **exactly equal**:

```
1. specs/tq40_datasheet_r2.pdf            score=0.03252247488101534  via=keyword#1+vector#2
2. specs/tq40_datasheet_r1_WITHDRAWN.pdf  score=0.03252247488101534  via=keyword#2+vector#1
   exactly equal: True       equals 1/61 + 1/62 by hand: True
```

Each chunk collects one first place and one second place, so the two fused
scores are the same two terms added in the other order. **Fusion does not
resolve this and cannot.** What breaks the tie is corpus position, which the
hybrid assigns in component-registration order — so listing `vector` first
prints the *withdrawn* revision at rank 1 instead. Both directions are pinned in
`test_the_dead_heat_is_broken_by_registration_order_and_not_by_evidence`.

This is the measured requirement on Stage 6: **superseded revisions must be
demoted by an explicit answering-stage rule.** A fusion that quietly favoured
one half would be hiding a judgement while pretending to make none.

### Measured: what fusion actually bought

Recall@10 is useless here — ten results out of 67 chunks is a sixth of the
corpus, so all three configurations score **7/9** and the two misses are the
image questions, whose files produce no chunks to rank. Every retriever fails
those correctly.

The sensitive measure is the rank of the first chunk belonging to an expected
citation — the chunk the answering stage reads first:

| question | keyword | vector | hybrid |
|---|---|---|---|
| q1 (r2 vs r1_WITHDRAWN) | 1 | 2 | **1** |
| q9 (two files) | 5 | 1 | **2** |
| other five measurable | 1 | 1 | 1 |
| **sum of ranks** | **11** | **8** | **8** |

Neither half is good at both. Keyword nails the supersession question and is
*fifth* on the two-file one; vector is exactly the reverse. Fusion is the only
configuration not bad at either — a **robustness** argument, not an accuracy one.

**Two caveats, stated because they bound the claim.** The q1 rank-1 is *not*
fusion merit: those two chunks tie exactly, so it is the positional tie-break,
and keyword happens to be registered first. Only the q9 result is the consensus
effect working as designed, where a file both halves liked beats chunks only one
half did. And with seven of nine questions already tied at rank 1, the entire
signal comes from two questions — this corpus cannot establish that fusion helps
on the graded one.

So the Stage 4b open risk is **narrowed, not closed**: dense retrieval still has
not been shown to beat keyword outright, but the combination has been shown not
to be *worse* than either half on any measurable question.

### Measured: RRF_K has a wide plateau

| k | recall@10 | top-1 differs from k=60 on |
|---|---|---|
| 0 | 7/9 | q8, q9 |
| 1 | 7/9 | q8, q9 |
| 10 | 7/9 | none |
| 60 | 7/9 | — |
| 100 | 7/9 | none |
| 1000 | 7/9 | none |

The top result is identical for every k from 10 to 1000; only 0 and 1 move it.
The *tail* does vary — q1 shows three distinct top-ten sets across the six
values, q2 shows four — so the constant is not inert, it just has a plateau, and
60 sits well inside it. That is a reason to keep the published value rather than
fit one.

### Design decisions worth naming

- **`RetrievedChunk.sources`** — which retrievers returned a chunk and where
  each ranked it. A fused score is on a scale of its own (0.0325 here, neither a
  cosine nor a term weight), so it cannot be read without knowing whether it came
  from one strong vote or two weak ones. It is the fusion analogue of
  `matched_terms`, and the diagnostics print it as `via=keyword#1+vector#2`.
- **`matched_terms` survive fusion** from whichever half has them. Keyword can
  explain itself; cosine cannot. Dropping the terms at the fusion step would
  throw away the only explanation the system has exactly where the ranking
  becomes hardest to read.
- **Components are asked for `limit × 3` candidates.** Fusion can only choose
  between chunks it has been shown; asking for exactly ten would mean a chunk
  both halves placed eleventh could never be rescued.
- **An empty hybrid is refused.** A retriever returning nothing for every
  question is indistinguishable in the graded output from a corpus that lacks
  the answer — the silent-failure shape this project keeps meeting.
- **One retriever down preserves its order.** With a single component,
  `1/(k+rank)` still decreases with rank, so the fused order equals the input
  order. Asserted rather than assumed, because a fusion that reordered here would
  silently degrade every lexical-only run.

### A trap found while running this

`python -I starter/app/app.py` **fails** with `ModuleNotFoundError: No module
named 'chunking'`. Isolated mode removes the script's directory from
`sys.path`, and `app.py` imports its siblings by bare name. The harness runs
`python3 /app/app.py` *without* `-I`, so this is not a live bug — but
AGENTS.md's "run `python3 -I` on scripts reading untrusted corpus data" must
never be applied to `app.py` itself. It applies to *our* tooling, not to the
program under test.

### What is still not verified

- **Nothing has run on a GPU.** Every measurement above is CPU.
- `transformers` was executed here on **Python 3.12**, still not 3.14.
- The fused ranking has **not** been shown to help on a corpus larger than 13
  files; the measurement's entire signal is two questions.

### Next step

**Stage 6 — answer extraction and citation selection.** Retrieval now returns a
fused ranking and `answer()` still refuses, which is why the output JSON is
still `{"answer": "", "citations": [], "confidence": 0.0}`.

Two things about Stage 6 are now *measured requirements* rather than design
preferences:

1. **Superseded revisions must be demoted by an explicit rule.** Fusion was
   shown to produce an exact tie between `tq40_datasheet_r2.pdf` and
   `tq40_datasheet_r1_WITHDRAWN.pdf`, broken only by which component was
   registered first. Nothing in retrieval separates them.
2. **Citations are a necessity judgement, not a ranking.** The rule is "would
   removing this file make the answer impossible" — extra citations score zero,
   exactly like missing ones. The fused ranking gives candidates; it does not
   give the set.

Also still open and carried forward: the two image questions (q7, q8) cannot be
answered by any amount of retrieval work, because the vision path does not
exist yet. `ingestion/readers/image.py` reports `vision_unavailable` for both,
so rule 8 holds — it refuses rather than inventing text — but those two
questions will be refused until Stage 7.

---

# Stage 6a — answer value extraction

**Status: complete and measured.** `answer()` now returns a real answer value
for every question whose evidence is readable text. Citations are still
**provisional** — see below — and Stage 6b replaces them.

## Files

| File | Change |
|---|---|
| `starter/app/answering/__init__.py` | new — package docstring, public API |
| `starter/app/answering/shapes.py` | new — `AnswerShape`, `ValueSpan`, `question_shapes`, `value_spans`, `content_terms` |
| `starter/app/answering/supersession.py` | new — `is_superseded`, `superseded_paths` |
| `starter/app/answering/candidates.py` | new — `Candidate`, `candidates_from` |
| `starter/app/answering/extract.py` | new — `Extraction`, `extract`, the lexicographic ordering |
| `starter/app/app.py` | `answer()` calls `extract()`, returns value + provisional citation; new `_report_extraction` |
| `tests/test_answering_{shapes,supersession,candidates,extract}.py` | new — 66 tests |
| `tests/test_import_graph.py` | 5 new modules in the parametrise list |

## The design decision that matters: lexicographic, not weighted

Six signals, compared in priority order; the first that separates two candidates
decides. No weights, because every weight would be a number invented here with
nothing to fit it against, and the graded corpus is different and larger.

```
1. superseded          a replaced revision never wins
2. wanted shape        the kind of value the question asked for
3. overlap             share of the question's content words the chunk explains
4. follows_equals      a field's value beats a mention of one
5. retrieval_rank      the fused ranking's own opinion
6. position            (location, rank) — deterministic last resort
```

**Supersession is first, and it has to be**, because it is the one signal that
answers a question the other five cannot see: `94` in r2 and `105` in r1 match
the question identically and differ only in which revision is current.

## Three defects the tests caught before the corpus did

Each was silent — no exception, no warning, just a wrong or empty result.

1. **The content supersession markers were dead code.** `config` holds them as a
   person would write them (`"IS WITHDRAWN"`, with spaces) and they were compared
   against `normalise(text)`, which *deletes all whitespace*. `"IS WITHDRAWN" in
   "THISREVISIONISWITHDRAWN"` is false for every document in every corpus, so the
   whole content half of the rule never fired. Fixed by normalising the markers
   once at import (`normalise(marker)`), with the reason recorded at the
   constant.
2. **A value at the end of a sentence matched nothing.** The trailing guard on
   the number and version patterns was `(?![\w.])`, which forbids *any* following
   full stop — so `is 94.` and `Version 4.3.2.` produced no span at all, while
   `is 94 C.` worked. The guard now forbids a following word character and a full
   stop only when a digit follows it, which keeps the `4.2x`-is-not-a-version
   rejection and admits the sentence.
3. **Hints matched inside words.** Hints were substring-matched against the
   whitespace-deleted question, so `identify` voted for a *quarter* (it ends in
   `fy`) and `rebuild` voted for a *version* (it contains `build`). A hint now
   matches a whole word or a run of consecutive words. The `FY` hint was then
   dropped entirely: a question that merely *names* a fiscal year ("What was the
   FY27 revenue?") wants a number, and the question that wants a quarter says
   "quarter" or "fiscal" — which the sample set's phrasing does.

## Two defects the measurement caught

4. **Hyphenated English was an identifier.** The pattern required a letter but
   never a digit, so `follow-on` and `field-replaceable` were offered as
   candidate part numbers — and `follow-on` won q8 outright, returning a word
   that is not a value. Now both a letter and a digit are required, which is also
   the rule the tokeniser already applies (a run only gets a joined spelling when
   it contains a digit). q8 still fails, because it is a vision question.
5. **A plausible reordering, refuted by measurement.** `follows_equals` (a field
   value beats a prose mention) looked like it should outrank overlap — my own
   docstring claimed it existed to decide one decoy row. Promoting it cost
   **two questions**: q1 returned `212` (a datasheet field) over the `94` asked
   about, and q9 returned `4.2.0` (a *different* ticket's `fixed_in`) over the
   right row's `4.3.2`. `key=value` is everywhere in this corpus, and only
   topical overlap can say *which* of several identically-shaped rows is meant.
   Reverted, and the docstring's false claim corrected.

## Measured result — 10 sample questions, real CLI, one process each

| | |
|---|---|
| answer correct | **7 / 10** |
| citation set exact | **6 / 10** |
| per question | 9.0–10.6 s (budget 30 s) |
| index pass | 8.4 s, 13 files, 67 chunks, 2 unreadable, 2 vision-unavailable |

| n | got | expected | why |
|---|---|---|---|
| 1 | `94` | `94` | ✓ — supersession demoted r1; fusion alone could not |
| 2 | `Q3 FY27` | `Q3 FY27` | ✓ |
| 3 | `ORR-FAN-2214-B` | `ORR-FAN-2214-B` | ✓ — `follows_equals` had to *not* trump overlap here |
| 4 | `4.3.2` | `4.3.2` | ✓ |
| 5 | `E7731` | `E7731` | ✓ |
| 6 | `180` | `180` | ✓ — beat the `60` named in a comment |
| 7 | `E7731` | `B14` | ✗ — vision question, file is `backplane_pinout.png` |
| 8 | `TQ-40` | `REV-C2` | ✗ — vision question, file is `asset_label.jpg` |
| 9 | `4.3.2` | `4.3.2` | answer ✓, **citation set wrong** — cited 1 of 2 files |
| 10 | `7` | `""` | ✗ — refusal case; `7` is a lead-time field, not a price |

**All three wrong answers are stages not yet built**, not extraction failures:
q7 and q8 need vision (Stage 7), q10 needs the refusal judgement and q9 the
multi-hop chain (both Stage 6b). Baseline before this stage was 0/10 answers.

## What is provisional, and what replaces it

`answer()` returns the file the winning value was read out of as its **only**
citation. That is right for a single-document question and wrong for q9, which
needs two files. It ships because a citation set is scored as an exact set, so
one correct citation and an empty one score identically on a two-file question —
the provisional version can only lose where the question was already lost, and
wins outright on the single-document ones.

**Stage 6b must replace it.** The necessity judgement ("would removing this file
make the answer impossible"), the multi-hop chain, and the refusal rule are one
piece of reasoning and belong together.

## What is still not verified

- **Nothing has run on a GPU.** Every measurement is CPU.
- The vision path still reports `vision_unavailable`, so q7 and q8 refuse-shape
  but cannot be answered.
- No threshold exists anywhere in extraction: it always returns its best
  candidate when one exists. Whether that is right for the refusal question is
  Stage 6b's to decide, and q10 is the evidence that it currently is not.

---

# Stage 6b, part 1 — the citation set

**Status: complete and measured.** `answer()` no longer returns the provisional
citation. It returns the set of files that were **necessary**, and it gets q9 —
the two-file question — right for the first time. Part 2, the refusal rule, is
still open and is written up at the end of this section.

## Files

| File | Change |
|---|---|
| `starter/app/answering/citations.py` | new — `cited_paths`, `_uses_a_key`, `_keys_around`, `_forms` |
| `starter/app/app.py` | `answer()` calls `cited_paths(winner, query, index.chunks)`; new `_report_citations`; `cited_paths` added to the `answering` import |
| `starter/app/answering/__init__.py` | `cited_paths` re-exported; docstring gains a fifth step and loses the "does not decide citations" caveat |
| `tests/test_answering_citations.py` | new — 14 tests |
| `tests/test_import_graph.py` | `answering.citations` added to the parametrise list |

## The rule

The contract does not ask which files the answer was read out of. It asks which
files were **necessary to produce it**, and it scores the answer as an exact set,
so a citation too many costs exactly what a citation too few costs. The rule is
therefore the definition itself:

> A file is necessary when removing it would make the answer impossible.

For the seven single-document questions that is the winner's file and nothing
else, and no search is needed to say so. The multi-hop question is where the work
is:

* the log records the incident and names the ticket it was logged against;
* the bug database maps that ticket to the release that closed it.

The value lives in the database. The database alone does not say *which row* the
question is about, and the question names neither the ticket nor the release.
What identifies the row is its key, and the log supplies it — so the log is
load-bearing despite holding no part of the answer.

Four restrictions, each measured against the alternative:

1. **Only identifiers and codes can be keys.** A number names no particular
   thing. Measured: the log line holding `E7731` also says "clocks reduced to
   60%", and `60` occurs in `engineering/ingest_service.py` and again in the
   roadmap, so letting numbers bridge cites three files for a question answerable
   from one.
2. **Keys match by joined spelling, never by fragment.** `ORR-1847` tokenises to
   `orr`, `1847`, `orr1847`; matching on `orr` matches every `ORR-FAN-*`,
   `ORR-HSK-*` and `ORR-CBL-*` part number in the spreadsheet.
3. **The search is over the whole corpus, not the retrieved chunks.** Measured
   below; this is the difference between 6/10 and 7/10.
4. **A file that merely names the key is not necessary.** The bridging chunk must
   also account for at least one of the question's own words.

## Measured: corpus-wide search, not retrieved-only

| Key search | Citation sets exact |
|---|---|
| over the retrieved chunks | 6/10 |
| over the whole corpus | **7/10** |

The question it gains is q9, and the reason is not a matter of taste: the log
line naming the ticket is **not** among the ten chunks that question retrieves.
A rule that looked only at what retrieval happened to fetch names one file of the
two that are necessary, and an exact-set score gives that zero either way.

## Measured: keys from the whole file, not the winning chunk

| Key gathering | Citation sets exact |
|---|---|
| keys from the winner's own chunk only | 6/10 |
| keys from every chunk of the winner's file presenting the value | **7/10** |

This one was found by reading the q9 diagnostic rather than by reasoning, and the
finding is uncomfortable enough to record plainly. q9's answer is `4.3.2` and the
extractor reaches it through the **decoy** row — `ORR-1904`, whose *summary*
mentions `4.3.2` while its `fixed_in` field is empty — because that row's prose
accounts for three of the question's words and the real row's fields account for
one. Both rows present the same value, so both contribute keys, and the
`ORR-1847` that makes the log necessary comes from the row the extractor did not
pick. Reading only the winning chunk loses the bridge.

Two things follow, and they should not be blurred together:

- the citation set is right, for a reason the docstrings now state;
- **q9's answer is right by coincidence.** A near-miss row that named a different
  version would have made the answer wrong, and nothing in the citation rule
  would have caught it. That is an extraction fragility, not a citation one, and
  it is the "comments naming old values next to the real one" hazard landing on a
  case where the decoy happens to name the *same* value.

## Two defects the tests caught

**A test asserting the right answer for the wrong reason.** The first version of
`test_the_value_is_not_its_own_key` passed, and it passed because the other file
happened not to be about the question — not because the rule under test was
doing anything. Every case where a file is excluded now has to be excluded by the
restriction the test names, which is why four of these tests come in pairs that
differ in exactly one thing.

**A rule with a real hole, found by the same test.** The first version of the
bridging rule cited any file containing the key. The sample's parts spreadsheet
carries the ticket number, so it got cited — and it should not be, because naming
a ticket is not connecting it to anything. That is what added the fourth
restriction above.

## Results — 10 sample questions, real CLI, one process each

```
q 1 rc=0  10.0s  ans OK  cites OK  '94'            ['specs/tq40_datasheet_r2.pdf']
q 2 rc=0   9.6s  ans OK  cites OK  'Q3 FY27'       ['planning/roadmap_fy27.docx']
q 3 rc=0   9.5s  ans OK  cites OK  'ORR-FAN-2214-B' ['support/rma_parts.xlsx']
q 4 rc=0   9.4s  ans OK  cites OK  '4.3.2'         ['support/bug_database.csv']
q 5 rc=0   9.5s  ans OK  cites OK  'E7731'         ['logs/prod_inference_2026-09-02.log']
q 6 rc=0   9.1s  ans OK  cites OK  '180'           ['engineering/ingest_service.py']
q 7 rc=0   9.6s  ans NO  cites NO  'E7731'         ['logs/prod_inference_2026-09-02.log']
q 8 rc=0   9.7s  ans NO  cites NO  'TQ-40'         ['planning/roadmap_fy27.docx']
q 9 rc=0   9.1s  ans OK  cites OK  '4.3.2'         ['logs/…log', 'support/bug_database.csv']
q10 rc=0   9.1s  ans NO  cites NO  '7'             ['support/rma_parts.xlsx']
```

**Answer correct 7/10, citation set exact 7/10.** q1–q6 and q9 are exact. Per
question 9.1–10.0 s against a 30 s budget; the citation rule itself costs 0.9 s
for all ten questions including model load.

Full suite **539 passed** (524 before, +14 citations tests, +1 import-graph row).

The three failures are exactly the three known gaps, and no others: q7 and q8 are
the image questions (Stage 7), q10 is the refusal question (Stage 6b part 2).

## What is still not verified

- **Nothing has run on a GPU.** Every measurement is CPU.
- The refusal rule does not exist, so q10 still produces a confident wrong answer
  rather than a refusal.
- The citation rule has never been exercised by a corpus with a **superseded file
  that also bridges**. Whether a withdrawn revision should be able to make itself
  necessary is unresolved, and the sample corpus does not ask.
- The bridging rule assumes the question shares at least one word with the
  passage that connects it to the key. True on q9 (`incident`) and plausible in
  general — the bridge exists to connect the question's subject to the key — but
  unmeasured beyond one example.

## Next step

Stage 6b part 2 — the refusal rule, which is the last piece standing between q10
and a correct answer. It is flagged in advance as the weakest part of the system:
every structural rule tried so far either still answers q10 wrongly or needs a
rarity threshold fitted to a single negative example.

---

# Stage 6b, part 2 — refusal

**Status: complete and measured.** q10 is now refused correctly, and the rule
touches nothing else. Stage 6b is closed.

## Files

| File | Change |
|---|---|
| `starter/app/answering/refusal.py` | new — `refusal`, `_field_label`, `_label_ending_at`, `_content_forms` |
| `starter/app/answering/shapes.py` | new `term_forms` — the joined-form helper, promoted out of `citations.py` so both rules share one normalisation |
| `starter/app/answering/citations.py` | uses the shared `term_forms` rather than its own copy |
| `starter/app/answering/__init__.py` | `refusal` re-exported; docstring gains a sixth step |
| `starter/app/app.py` | `answer()` calls `refusal()` before computing citations; returns `"", [], 0.0` on a refusal |
| `tests/test_answering_refusal.py` | new — 10 tests |
| `tests/test_import_graph.py` | `answering.refusal` added to the parametrise list |

## What the question actually is

q10 asks for the unit price of the TQ-40. The price is real and it is in the
corpus — inside `supplier_agreement_ENCRYPTED.pdf`, which cannot be opened, so it
is not a source. Correct behaviour is an empty answer with no citations, and the
sample's own note is blunt about why: a submission returning the real number has
either cracked the file or hallucinated.

That makes refusal the one rule in the system with an **asymmetric cost**.
Refusing an answerable question and answering it wrongly score identically, so
the rule gains nothing from caution in the abstract — but it can only ever *gain*
on genuinely unanswerable questions, while every false refusal spends a point that
was otherwise won. One negative example, nine positives.

## Three signals measured and refuted

**Confidence — the signal is not weak, it is backwards.**

| Question | Confidence | Outcome |
|---|---|---|
| q3 (part number) | 1.00 | correct |
| q2, q4 | 0.67 | correct |
| **q10 (must refuse)** | **0.60** | — |
| q1 | 0.60 | correct |
| q5, q6 | 0.50 | correct |
| **q9 (two-hop)** | **0.27** | correct |
| q7, q8 | 0.00 | vision pending |

q10 scores third-highest of the ten and the correctly-answered two-hop question
scores second-lowest. **Any threshold that refuses q10 also refuses q9, q1, q5,
q6, q7 and q8** — seven questions to catch one.

**Absence of the question's subject from the readable corpus.** The word `price`
never appears in it, which looks decisive until the same test runs on the
questions that *are* answered: q9 has four content words absent (`log`, `shows`,
`underlying`, `defect`), q7 has three, q8 has three, q10 has **one**. Paraphrase
is normal here, so absence of a word is not absence of a subject.

**Rarity of the matched terms.** This one *does* separate the ten — every
answered question matches a term occurring in ≤3 chunks, q10's rarest match occurs
in 5 — but the margin is one chunk, calibrated on a single negative example, and
it is a statistic rather than a fact about the text. Rejected on those grounds: on
a larger corpus the document-frequency distribution moves and the threshold would
start silently refusing answerable questions.

## The rule that survived

The refusal question's winner is `lead_time_days=7`. The corpus does not merely
*fail to contain* a price — it **states** that this value is a lead time. A value
the corpus explicitly labels as something else cannot be the thing asked for:

> Refuse when the winning value is presented as a field's value, and the field's
> name shares no word with the question.

No fitted constant: the test is "does this field name share a word with the
question", which is a structural fact about the text. Measured, it fires on four
of the ten — three answered correctly (`part_number`, `fixed_in`,
`DEFAULT_BATCH_TIMEOUT_S`, each matching its question's own words) and q10 — and
is **silent on the other six**, including both image questions. Silence is the
point: it can only refuse a value the corpus has labelled as being something
else, so it never refuses on a hunch.

## What it deliberately does not do

**It does not treat a colon as a field separator.** `ERROR E7731: thermal throttle
engaged` would give the value the field name `ERROR` under such a rule, and
`ERROR` shares no word with a question about an error code — so a colon-accepting
rule would refuse q5, a question the corpus answers plainly. Only `=` identifies a
field here, and a value with no label gets no verdict.

**It abstains whenever it cannot prove a mismatch:** unknown chunk, value not
found in it, no `=` before the value, a question with no content words, or a field
name made entirely of stopwords. Every one of those is a test.

## Where it could be wrong

The rule assumes field names are written in words a question would also use.
Measured, all **15 distinct field names** in the readable corpus are —
`fixed_in`, `unit_cost_usd`, `lead_time_days`, `part_number`, `compatible_with`,
none cryptic. A corpus using `pn` for part number would make a correct answer
unreachable. That is the risk this rule takes, stated plainly because it is the
one that cannot be measured here.

## Results — 10 sample questions, real CLI, one process each

```
q 1   7.3s  ans OK  cites OK  '94'             ['specs/tq40_datasheet_r2.pdf']
q 2   7.4s  ans OK  cites OK  'Q3 FY27'        ['planning/roadmap_fy27.docx']
q 3   7.6s  ans OK  cites OK  'ORR-FAN-2214-B' ['support/rma_parts.xlsx']
q 4   7.5s  ans OK  cites OK  '4.3.2'          ['support/bug_database.csv']
q 5   7.7s  ans OK  cites OK  'E7731'          ['logs/prod_inference_2026-09-02.log']
q 6   7.5s  ans OK  cites OK  '180'            ['engineering/ingest_service.py']
q 7   7.9s  ans NO  cites NO  'E7731'
q 8   8.1s  ans NO  cites NO  'TQ-40'
q 9   7.5s  ans OK  cites OK  '4.3.2'          ['logs/…log', 'support/bug_database.csv']
q10   7.5s  ans OK  cites OK  ''               []
```

**Answer correct 8/10, citation set exact 8/10.** The refusal diagnostic fires on
q10 and on nothing else. Full suite **549 passed**. Per question 7.3–8.1 s against
a 30 s budget.

Every remaining failure is now the *vision* gap: q7 and q8 are the two image
questions, and no amount of retrieval or answering work can reach them.

## What is still not verified

- **Nothing has run on a GPU.** Every measurement is CPU.
- The refusal rule has never met a question whose answer genuinely *is* absent
  rather than sitting in an unreadable file. The corpus hazard list does not
  promise such a question exists; q10 is not one.
- The refusal rule's one identified failure mode — a cryptic field name — is
  unmeasured, because this corpus has no cryptic field names.
- Stage 6's answer and citation layers have never been run against a corpus
  containing a superseded file that also bridges, or near-miss rows at a scale the
  graded corpus is described as having.

## Next step

Stage 7 — vision support, for q7 (`specs/backplane_pinout.png`) and q8
(`support/asset_label.jpg`). The ingestion path already reports
`vision_unavailable` for both rather than inventing text, so rule 8 holds; what is
missing is the reader itself.

---

# Stage 7 — pre-implementation investigation

**Status: investigation only.** No application code, Dockerfile, requirements or
test file was created, modified or removed. Nothing is decided. No package, model
or wheel was downloaded.

The investigation was commissioned to answer three questions before committing to
an OCR-based vision path: whether the candidate OCR engines can exist in the
mandated image at all, how OCR output becomes `TextSpan`s without corrupting the
existing pipeline, and whether the current workload satisfies the challenge's
1 GiB VRAM floor.

## What was examined

`CONTRACT.md`; the full 8-page brief, re-extracted page by page with this
project's own `ingestion.readers.pdf` (the laptop has no `poppler`, so the brief
PDF cannot be rendered as an image); `selfcheck.py`; `starter/Dockerfile`;
`starter/app/requirements.txt`; `config.py`; `models.py`; `chunking.py`;
`ingestion/readers/image.py`; `embeddings/local_model.py`;
`tests/test_readers_image.py`; the two corpus images, opened and viewed directly;
and PyPI package metadata.

Local environment facts re-checked on 2026-10-14: the laptop is Ubuntu 24.04 /
Python 3.12.3, and **no Python 3.14 exists locally**; the working venv holds
`torch 2.14.1+cpu`, `transformers 5.19.0`, `numpy 2.5.4`, `tokenizers 0.23.3`,
`safetensors 0.8.0`; `tesseract` is **not installed**; `apt-cache` offers
`tesseract-ocr 5.3.4-1build5` from Ubuntu 24.04 `universe`; and **`docker` is not
reachable from this WSL distro** (Docker Desktop WSL integration is currently
off), so the base image could not be inspected.

The two images were opened and read visually. This was the single most useful step
in the investigation, and it is worth recording what they actually contain,
because the design depends on it:

- `specs/backplane_pinout.png` (1000x560) — a title line, then **six pin boxes in
  a row** (`B11`..`B16`), each with a short vertical leader line dropping to a
  text label beneath it: `GND`, `SDA`, `SCL`, `THERM_ALERT#`, `PRSNT#`, `GND`.
  Below the row sit two prose paragraphs, the first of which also mentions
  `THERM_ALERT#`. Note `GND` appears **twice**, as a genuine duplicate in the
  source.
- `support/asset_label.jpg` (760x440) — a title block reading `Orrery Systems`,
  `MODEL TQ-40`, `BOARD REVISION REV-C2`, `SN OS4-118823-77`, `MADE IN MALAYSIA`,
  plus **a barcode of vertical stripes** on the right.

## Verified facts

**Python 3.14 wheel availability — checked against PyPI, not assumed.** The
brief's own trap 2 makes this the decisive question, and it produced the
investigation's one outright blocker:

| Package | Newest | Python 3.14 on Linux x86_64 |
|---|---|---|
| `rapidocr-onnxruntime` | 1.4.4 | ❌ **blocked** — metadata declares `requires_python = "<3.13,>=3.6"`; pip refuses it on 3.14 |
| `rapidocr` (v3 successor) | 3.10.0 | ✅ `py3-none-any`, `requires_python = "<4,>=3.8"` |
| `onnxruntime` | 1.31.0 | ✅ `cp314-cp314-manylinux_2_28_x86_64` (present since 1.27.0) |
| `opencv-python-headless` | 5.0.0.93 | ✅ `cp37-abi3` stable-ABI wheel, so it covers 3.14 |
| `numpy` | 2.5.4 | ✅ `cp314` manylinux x86_64 |
| `pyclipper` | 1.4.0 | ✅ `cp314` manylinux x86_64 |
| `shapely` | 2.2.0 | ✅ `cp314` manylinux x86_64 |
| `pytesseract` | 0.3.13 | ✅ `py3-none-any` (pure Python) |
| `onnxruntime-rocm` | 1.22.2.post3 | ✅ `cp314` manylinux x86_64 exists — but see risks |

The headline is that the *familiar* package name is the one that fails:
`rapidocr-onnxruntime` is what every tutorial names, and pip would refuse it on
Python 3.14. Only the renamed `rapidocr` 3.x line is a candidate.

**Metadata-level availability is not installation.** Every ✅ above says a wheel
*exists* for the right interpreter and platform. None of it says the package
imports, links, or runs on this image. That distinction is why both candidates
must be exercised inside the container before either is committed to.

**Tesseract in Ubuntu 26.04 (Resolute)** — documented externally, not verified in
the image: `tesseract-ocr` 5.5.0 is in `universe`, and language data ships as
separate packages under `tesseract-lang` (`tesseract-ocr-eng`). Two consequences:
the base image's apt sources must have `universe` enabled, and the `.traineddata`
must be present in the image, because evaluation has no network.

**The VRAM mechanism, from the kit rather than guessed.** `selfcheck.py` holds
`VRAM_FLOOR_GIB = 1` and `VRAM_CEILING_GIB = 48`, and its check 6 names the tool
the harness uses:

```
watch -n1 'amd-smi metric --mem | grep USED_VRAM'
```

`CONTRACT.md` states the same band with 1% tolerance on the upper bound, that VRAM
is "sampled continuously and the peak is what counts", that under the floor means
CPU and is rejected, and that pinning a dummy buffer is detected separately.
Because the *peak* is what counts, a transient spike during `--index` or during
one `--query` is sufficient — the model does not have to be resident.

**The embedding model's weight footprint, by arithmetic.** `config.json` declares
`torch_dtype: float32` and `model.safetensors` is 133,466,304 bytes, which matches
33.4 M parameters at 4 bytes each. **That is 0.13 GiB — roughly an eighth of the
floor.** Everything else the run occupies is runtime overhead.

## Unverified assumptions and risks

1. **The current workload may sit below the VRAM floor.** The weights are
   0.13 GiB. Whether the process peaks above 1 GiB depends entirely on ROCm/HIP
   context and torch allocator overhead, which **cannot be predicted from
   parameter count** and has not been measured — there is no AMD GPU on the
   laptop. This is the largest structural risk found.
2. **The exact sampling mechanism is not specified.** `selfcheck.py` names
   `amd-smi metric --mem` but automates nothing; the sampling interval, and
   whether the floor is evaluated over the container's whole lifetime or only
   during graded phases, are unknown. The phrase "sampled continuously" plus a
   `sleep infinity` container suggests the whole lifetime, which is why the *peak*
   framing matters.
3. **`--index` timing under OCR is unmeasured.** OCR runs once per image during the
   index pass (10 min budget), not per query, so it is affordable — but the graded
   corpus is larger and its image count is unknown.
4. **OCR accuracy on the pinout is unproven.** The label is clean printed text and
   should be easy. The pinout is a *diagram*: Tesseract's page-layout analysis may
   reorder or merge the row. `THERM_ALERT#` ending in `#` is a specific
   misrecognition risk.
5. **`onnxruntime-rocm` is a poor bet for GPU OCR.** It is a separate package from
   `onnxruntime` (both provide the `onnxruntime` module), it bundles its own ROCm
   runtime libraries alongside the base image's, its newest release (1.22.x)
   trails the CPU package (1.31.0), and gfx1100/RDNA3 support for the ROCm
   execution provider is unverified.
6. **RapidOCR 3.x downloads its models on first use.** `requests` is a declared
   dependency and no engine is declared at all, so the ONNX runtime must be named
   explicitly and the model files pre-baked into the image. This is inferred from
   metadata plus documented behaviour, **not verified**.
7. **Barcode noise is a real hazard for the label.** Vertical stripes are exactly
   the shape that produces spurious low-confidence glyph runs. Unfiltered, they
   would enter the index as text.
8. **A cross-stage gap this stage does not close.** q7's expected answer is `B14`
   and q8's is `REV-C2`. `answering/shapes.py` derives wanted value shapes from
   question words; there is no pin-designator shape today, and Stage 6b measured
   q8 currently answering `TQ-40` — which is *also printed on the label*, as
   `MODEL TQ-40`. Making the image readable is therefore necessary but may not be
   sufficient for q7/q8; extraction shapes may need extending too.

## Tesseract versus RapidOCR

| | Tesseract | RapidOCR |
|---|---|---|
| Engine | C++ binary, installed by `apt` | Python + ONNX inference |
| New Python deps required | **none**, if invoked as a subprocess | onnxruntime, opencv-python-headless, numpy, pyclipper, shapely, omegaconf, PyYAML, Pillow, six, tqdm, requests, colorlog |
| Python 3.14 status | not applicable — no Python package involved | `rapidocr-onnxruntime` **refused** by pip; `rapidocr` 3.x installable |
| Coordinates | word-level boxes via TSV output (`left/top/width/height/conf`) | text-line quads plus a confidence score |
| Approximate size | tens of MB (binary, libtesseract, leptonica, `eng.traineddata`) | ~150–300 MB (onnxruntime, OpenCV, numpy, models) |
| Offline | fully offline once installed | fully offline **only if models are pre-baked** |
| GPU | none (CPU only) | CPU by default; ROCm EP is a separate, unverified package |
| pip / torch-replacement risk | **zero** — nothing is pip-installed | real — several installs; mitigate with `--no-deps`, an explicit list, and a post-build `torch.__version__` check |
| Determinism | deterministic | deterministic |

**Recommend which to test first: Tesseract.** The deciding argument is not
accuracy, it is blast radius. Tesseract can be driven by an `apt` package and a
`subprocess` call, which means the shipped payload stays exactly what it is today
— standard library only — and the brief's trap 2, in which a pip install silently
swaps ROCm torch for a CUDA build, is never even approached. It also removes the
Python 3.14 question entirely, because no Python package is involved. Its TSV
output already carries the word-level boxes the geometry test needs.

RapidOCR is the better *fallback*, not the better first test: it is the stronger
engine on messy text, but it is reached only through `rapidocr` 3.x, and it brings
a dependency chain whose every compiled member had to be individually checked —
exactly the kind of risk to spend only if Tesseract proves inadequate.

## The image-to-TextSpan path

**The one structural fact that shapes everything: `TextSpan` has no coordinate
field.** It carries `text`, `location` and `ordinal`. Adding a bounding box would
change `models.py`, the chunk store and the index format version. It is also
unnecessary: geometry is an *internal* concern of the image reader — it is the
thing that lets the reader work out *which* text belongs with *which* pin. What
crosses the boundary is text plus a human-readable location, exactly as every
other reader already does.

So the path is:

```
image bytes
  → OCR (external engine, behind a seam)
  → Detections: (text, confidence, left, top, width, height)      [pure data]
  → line spans:      one span per detected line, location "text line N at (x, y)"
  → association:     pure geometry function over Detections
  → association spans: location "pin region (x, y)-(x2, y2)"
  → ReadResult.of(...) → chunking.py → index     [all unchanged]
```

Nothing new is added to `chunking.py`, the chunk store, retrieval or answering.
The reader keeps its contract: it never raises, failure stays a value, and
`ReadResult.of` still downgrades a text-less result to `EMPTY` rather than `OK`.

**The load-bearing design decision is that the geometry is a pure function over
detections, not over pixels.** Two things follow. The whole association rule — the
part that can be wrong in interesting ways — is unit-testable with synthetic
detection lists, with no OCR engine, no image file and no GPU, exactly as
`probe_from_bytes` let the header parsers be tested without files. And the OCR
engine sits behind one narrow seam, so Tesseract and RapidOCR are swappable
without touching the reasoning.

The deterministic association test for the pinout, as the images show it to be:

1. **Classify** each detection as an *upper* candidate (a short, separator-free
   alphanumeric token, `B14`) or a *lower* candidate (a signal label,
   `THERM_ALERT#`). Classification is structural — short token above another short
   token — **not** a regex fitted to `B\d\d`, because the graded corpus will use
   different designators. Long lines, such as the title and the prose paragraphs,
   are excluded by width, which is what stops the title pairing with the first pin
   box.
2. **Pair** each upper token with the lower token that is (a) strictly below it,
   (b) within a bounded vertical gap, and (c) horizontally centred on it, the
   centre distance being the quantity minimised. Nearest-centre matching is what
   resolves the two genuine `GND` labels, because each is nearest its own pin.
3. **Refuse when ambiguous.** Emit no association, and record the reason, when
   there is no candidate, when the best and second-best candidates are within a
   margin of each other, when two upper tokens claim the same lower token, or when
   the lower token is wide enough to overlap two columns. A diagram the rule cannot
   read must yield *no* relation, never a confident guessed one — an invented
   pin-to-signal pairing is indistinguishable downstream from a real one and would
   be cited as evidence.
4. **Never synthesise unread text.** Every span's text is either a verbatim OCR
   line or the joining of two verbatim OCR strings. No word is added that the image
   does not contain.

**One open design question, deliberately not settled here: the direction of the
rendered association.** `B14 | THERM_ALERT#` and `THERM_ALERT# | B14` are the same
fact but they behave differently downstream. Stage 6b's refusal rule fires when the
winning value sits under a field name sharing no word with the question; if the
association renders as `field=value`, then `THERM_ALERT#=B14` lets the extractor
treat `B14` as the value and the question's own words as the label, while the
reverse invites `THERM_ALERT#` to be lifted as the answer. The right resolution is
measurement on the real pipeline in Stage 7, not a decision taken now on taste.

**Tests the design calls for**, beyond the existing header tests:

- association exercised entirely through synthetic detection lists — arbitrary
  designators and arbitrary signals, never `B14`/`THERM_ALERT#` keyed to the
  sample, so the rule cannot be a lookup table wearing a geometry costume;
- a layout shaped like the real one (six boxes, one duplicate `GND`) pairs
  correctly, **including** the duplicate;
- an asset-label layout yields one span per printed line, and **barcode noise is
  suppressed** — a run of stripe detections at low confidence must not become text;
- OCR returns nothing → `EMPTY` with no spans, and specifically **not** `OK`;
- the engine binary is absent → `VISION_UNAVAILABLE`, so rule 8 still holds and the
  silence is still named;
- the engine exits non-zero, times out, or emits malformed output → `UNREADABLE`
  with no spans, never a traceback;
- duplicate/overlapping detections of the same text are collapsed before pairing;
- ambiguous column pairing and many-to-one conflicts produce **no** association;
- the invariant: no path produces a span whose text is absent from the OCR output.

## GPU and VRAM validation plan

**What the contract actually requires** (both documents agree, and `selfcheck.py`
independently repeats it): peak VRAM between **1 and 48 GiB**, sampled
continuously, peak rather than average; below the floor is judged CPU-only and
rejected; a deliberate dummy buffer is detected separately. Because the peak is
what counts, the requirement is met by any genuine moment above 1 GiB.

**What is known about the current workload:** weights are 0.13 GiB by arithmetic
from the file size; nothing else is known, and **no measurement is invented here**.
Parameter count does not predict peak VRAM, so the honest position is that the
current peak is unknown and the floor's safety margin is unknown with it.

**Measurement plan, for the notebook session:**

1. Sample device memory continuously during a full run, writing timestamps and
   values to a log, using the tool `selfcheck.py` names:
   `while true; do amd-smi metric --mem; sleep 1; done > vram.log`. Record the
   `amd-smi` version, the GPU name and the ROCm version alongside it.
2. Capture the three phases separately, because the peak may occur in only one:
   container idle after start; the `--index` pass; a single `--query`.
3. Record **two independent views** and report both, because they answer different
   questions: the device-level `USED_VRAM` peak (what the harness sees, and it
   includes the HIP context), and the process-level
   `torch.cuda.max_memory_allocated()` / `memory_reserved()` (what our code asked
   for). The gap between them is the runtime overhead, and knowing its size is what
   makes the result interpretable rather than a single number.
4. Run it against the real sample corpus and against a synthetic corpus with many
   more chunks, so the index-pass peak is exercised at scale.
5. Evidence to keep: the raw `vram.log`, the peak value with the phase it occurred
   in, and the exact commands.

**Open questions to flag rather than assume:** the sampling interval, and whether
the floor is evaluated across the container's whole lifetime or only during graded
phases. If either turns out to be stricter than assumed, the answer changes.

**If the peak is below 1 GiB — options, and what they cost:**

1. **Put the vision model on the GPU.** This is the option to prefer, because it is
   *required work* rather than padding: q7 and q8 cannot be answered without it. A
   small torch-based model runs on the ROCm torch the image already ships, adding
   weights but no new compiled dependency. It solves two problems with one
   decision.
2. **Use a larger embedding model.** A genuine quality trade, and bge-large at
   fp32 would clear the floor on weights alone — but it lengthens every query
   process (measured today: 8.99 s of a 30 s budget with the small model) and the
   sample corpus cannot demonstrate a retrieval gain.
3. **Raise the embedding batch size.** Real work, modest effect on activation
   memory only. Cheap to try, unlikely to be sufficient alone.

**What must not be done:** allocating a buffer to clear the floor. The contract
names it explicitly as separately detected, and it is a disqualification, not a
tuning choice.

## Recommended next action

1. **Build and fully test the geometry module first, with no OCR at all.** It is a
   pure function over detections, so every rule above — pairing, the duplicate
   `GND`, the ambiguity refusals, the barcode filter — can be written and proven
   under the existing suite with no engine, no image and no GPU. This needs no
   permission and no download, and it is the part most likely to hide a silent bug.
2. **Then test Tesseract on the two real sample images**, which needs one approval
   this investigation could not take for itself: installing `tesseract-ocr` locally
   (a small `apt` download) to see what the engine actually returns for
   `THERM_ALERT#` and for the pin row. Until that is run, every accuracy claim here
   is an expectation.
3. **Settle the VRAM question on the notebook before choosing a vision
   architecture**, because it decides whether the vision model is merely needed for
   q7/q8 or is also load-bearing for the floor.

Nothing was implemented, and nothing will be until this is approved.

---

# Stage 7A — the image layout geometry

**Status: implemented and tested. Geometry only.** No OCR engine was installed, no
model was downloaded, no Docker or dependency file was touched, and nothing about
answer extraction, citation selection or GPU behaviour changed. The two images in the
sample kit were *not* run through anything: there is still no engine to run them
through.

`starter/app/ingestion/readers/image.py` is **byte-identical to before this stage**.
The seam exists in the design but is not wired, because there is nothing behind it
yet.

## What was built, and why this piece first

The Stage 7 investigation recommended building the geometry before the engine, and
that recommendation is the whole shape of this stage. The part of image support most
likely to be *silently* wrong is not the OCR call — that either returns text or
doesn't — it is the rule that decides which piece of text belongs with which. A
designator and a signal name on a connector diagram are two separate detections, and
the fact that they are related is stated by nothing but their positions.

That rule is now a pure function over boxes:

```python
layout_from(detections, *, image_width=None) -> Layout
```

It reads no file, calls no engine, opens no pixel, and is deterministic. So every
tolerance in it can be tested with invented coordinates, the way the header parsers
in `image.py` are tested with invented bytes.

## The interface

Three frozen dataclasses in, three out, and one function:

| Type | Role |
|---|---|
| `Detection` | one engine finding: `text`, `confidence`, `left`, `top`, `width`, `height` |
| `TextLine` | one line of recognised text, with an `ordinal` (reading position, from 1) and a `location` |
| `Association` | two lines that belong together, plus the `location` of the pair |
| `Rejection` | an association geometry could have made and declined, with a `RejectionReason` |
| `Layout` | everything one image yielded: `lines`, `associations`, `rejections`, `dropped_detections`, `duplicate_detections` |

Field names are the ones OCR engines use, so the eventual adapter is a transcription
rather than a translation.

**`TextSpan` was not changed.** Inspection confirmed it does not need to be. It is
`(text, location, ordinal)` with no coordinate field, and the geometry needs
coordinates only internally — what crosses the boundary is text plus a
human-readable location. Adding a box to `TextSpan` would change the shared schema
that the chunk store and the index format are written against, to buy nothing: the
`location` string carries the provenance, in the shape the other readers already use.

```
"text line 4 at (63, 150)"                              # one line
"text lines 4 and 5 at region (63, 150)-(180, 232)"     # one association
```

That is the same convention `image.py`'s own docstring anticipated
(`"region (120, 240)-(410, 300)"`), and it is enough for a reader to draw the boxes
back onto the image and check a pairing against what the engine claimed. One
`TextSpan` will be emitted per line and one per association, so the association text
enters chunking and the index through exactly the path ordinary text does.

## Files

| File | Change |
|---|---|
| `starter/app/ingestion/readers/image_geometry.py` | **new**, 643 lines — the module |
| `starter/app/config.py` | +11 constants under a `# --- Image layout ---` block |
| `tests/test_readers_image_geometry.py` | **new**, 828 lines, 60 tests |
| `tests/test_import_graph.py` | +1 parametrised row for the new module |
| `starter/app/ingestion/readers/image.py` | **unchanged, byte for byte** |

## The rules, and what each one protects

Each tolerance in `config.py` is documented in place with the failure it guards
against. Summarised:

1. **Lower-confidence-than-`MIN_OCR_CONFIDENCE` detections are not text.** Barcode
   stripes and ruling lines come back as text-shaped detections at near-zero
   confidence. Indexing them would put words into the corpus that are not in it.
2. **Same text *and* same box is one detection** (`DUPLICATE_BOX_OVERLAP`, measured
   against the smaller box so a nested box still counts). Same text at a *different*
   place is a real second occurrence and is kept — a diagram with two `GND` pins
   depends on it.
3. **Line grouping needs vertical overlap *and* a horizontal gap limit**
   (`LINE_VERTICAL_OVERLAP`, `LINE_MAX_HORIZONTAL_GAP_FACTOR`). This is the expensive
   one in both directions: without the horizontal half, every label on a row shares a
   baseline and the whole row fuses into one long line, which is then too long to be
   a candidate — losing *every* pairing on the image at once.
4. **An identifier is one word** (`MAX_IDENTIFIER_WORDS = 1`). This is the strongest
   rule and it does most of the work. The most dangerous layout in an image is a
   title block: a stack of evenly spaced centred lines whose spacing falls inside the
   association band, and whose shape is almost exactly a diagram's. No line of a title
   block is a single token.
5. **A label is at most four words** (`MAX_LABEL_WORDS`), and a heading may not span
   more than half the canvas (`MAX_ASSOCIATION_WIDTH_FRACTION`) when the width is
   known.
6. **The gap between the two boxes must be between 2 and 12 text heights**
   (`ASSOCIATION_MIN_GAP_FACTOR`, `ASSOCIATION_MAX_GAP_FACTOR`). The floor is what
   stops ordinary line spacing inside a text block from reading as a field-and-value
   relationship; the ceiling stops a column-aligned coincidence far down the page.
7. **The centres must line up** (`ASSOCIATION_MAX_CENTRE_OFFSET_FRACTION`), which is
   what a leader line means.
8. **Ambiguity is refused, not resolved** (`ASSOCIATION_AMBIGUITY_FACTOR`). If the two
   best candidates are within one text height of each other, no association is
   emitted and a `Rejection(TOO_CLOSE_TO_CALL)` is recorded. The other two refusals
   are `SHARED_LABEL` (two identifiers claimed one label) and `CHAINED` (a token sits
   on both sides of an association, which a stack of three tokens produces).

Refusals are recorded **only when there was something plausible to refuse**. A token
with no neighbour is ordinary text, and recording every lonely token would bury the
real refusals in noise.

## Tolerances measured, not guessed

Two constants would have been wrong if they had been left as first written, and both
were caught by measuring the real image rather than by reasoning about it.

There is no PIL in the local venv and nothing may be installed, so the sample PNG was
decoded with the standard library (`zlib` plus the five PNG scanline filters) and its
ink rows profiled, using throwaway scripts kept deliberately **outside the
repository** in `/tmp/mc3-tools/`. What came back:

- six pin boxes **117 px wide** on a **150 px column pitch**
- designators **16 px tall**, tops at **y=150**
- signal labels **10 px tall**, tops at **y=222**

So the real gap is 56 px against a 10 px unit — **5.6 text heights**.

- The ceiling had been written as `6.0`. The real pairing sits 7% under it: a
  slightly tighter OCR box would have silently killed the one pairing the whole
  exercise exists to find. Raised to **12.0**.
- The floor had been written as `1.0`, and the adjacent lines of the sample label are
  about 1.3 text heights apart — *above* the floor, so it protected nothing. Raised to
  **2.0**, and the one-word identifier rule was added as the real title-block guard.

Both measurements are re-asserted as tests in `TestTheMeasuredShape`, which replays
the real geometry with invented text, so the next person to loosen a tolerance gets a
failing test rather than a silent loss.

## Tests

60 tests in `tests/test_readers_image_geometry.py`, all synthetic. Every designator
and signal in the file is invented — the sample's real names appear nowhere — because
a test written around the sample's own two strings would pass just as happily against
a lookup table keyed on the filename.

Covering, per the request: several columns with arbitrary identifiers and labels;
duplicate labels in separate columns; duplicate and nested and differently-spelled
detections; missing upper or lower text; long headings; ambiguous column positions;
two identifiers competing for one label; no fabricated text; empty detections and
invalid coordinates (zero, negative, NaN, infinite, boolean, blank, punctuation-only).

Three of them are worth naming:

- `test_a_diagram_sharing_nothing_with_the_sample_still_works` — a designator shape,
  a two-word label and a punctuated lower-case signal that appear in no sample file.
- `test_every_association_is_two_recognised_lines_and_nothing_else` — the
  never-invent invariant stated as an equality: an association's text must split into
  exactly two parts on `CELL_SEPARATOR` and both must be lines the engine reported. A
  connecting word or a rewritten spelling would fail it.
- `TestTheMeasuredShape` — the real geometry, invented text.

## Results

**Geometry tests: 60 passed, 0 failed** (`pytest tests/test_readers_image_geometry.py`).

**Full suite: 610 passed, 0 failed, in 66.75 s.** That is 549 (the Stage 6b figure) plus
the 60 geometry tests plus one import-graph row.

**The first run failed 8 of 54 and is recorded here rather than quietly fixed.** Seven
were my fixtures or helpers being wrong — a two-word "identifier" that the one-word
rule rightly rejected, a helper missing a `width` argument, a 20 px gap that is not a
realistic diagram gap, a reading-order expectation that contradicted the module's own
left-to-right rule. The eighth was a **genuine defect**, and it is the reason
`_lines_from` has a merge pass:

> With detections sorted by `(top, left)`, `REV-C2` at top 100 is visited before
> `REVISION` at top 104, so one line of text split into two groups — and the greedy
> first-match pass never merged them. The fragments were `"BOARD REVISION"` and
> `"REV-C2"`, both short enough to look like candidates.

That is exactly the failure mode the module exists to avoid, so the fix was verified
rather than assumed: with the merge pass disabled, **2 tests fail and 58 pass**; with
it restored, **60 pass**. The regression is pinned by
`test_a_word_arriving_late_still_joins_the_two_halves_it_bridges`, whose fixture
(`NORTH CROSSING SOUTH`, with the middle word reported last) fails as
`['NORTH CROSSING', 'SOUTH']` without the fix.

## What this stage deliberately does not do

- **No OCR engine.** `tesseract-ocr` and `rapidocr` are both still uninstalled, as
  instructed. Nothing here has been run against a real image.
- **No dependency or Docker change.** The Dockerfile and requirements files are
  untouched.
- **The seam is not wired.** `image.py` still reports the vision path as unavailable
  and does not import the geometry module. Rule 8 of `AGENTS.md` — report
  unavailability, never invent text — is still satisfied by the existing code.
- **Nothing downstream changed.** No answer extraction, no citation selection, no GPU
  behaviour. The geometry module is currently reachable only from its tests.

## Still outstanding — recorded so it is not lost

### 1. The GPU memory floor is still unmeasured on the target hardware

Nothing in this stage touches it, and it is still the largest unknown in the project.
The contract samples VRAM continuously and rejects a peak below **1 GiB** as a run on
CPU, with a dummy allocation detected separately. What is not known is whether the
embedding model alone clears the floor: its weights measure 133,466,304 bytes ≈
**0.13 GiB**, which is suggestive and is not the same as a measured peak, because the
figure that matters is peak *device* usage across the whole run.

To be settled on the notebook by running the real pipeline under
`watch -n1 'amd-smi metric --mem | grep USED_VRAM'` and recording the peak.

**Solving it by allocating a dummy buffer is explicitly not on the table.** It is a
disqualification, not a tuning choice, and it would also be dishonest.

### 2. Real-image OCR — measured on 2026-10-14, results in the next section

This was recorded here as an expectation ("that `THERM_ALERT#` comes back with its
`#`, that the six pin designators are read as six separate words"). Both hold, but
only under some page-segmentation modes. The measurement is below and it changed the
engine plan, so the original expectation is kept here only to show it was a guess
until it was run.

## Next step

Adapt Tesseract's TSV to `Detection` and place `layout_from`'s result into the reader
as `TextSpan`s. The geometry and the engine are now both measured against the real
sample images; what remains is the seam between them.

---

# Stage 7B — the engine, measured before it is wired in

Tesseract **5.3.4** was installed without root, entirely on `D:` (see the wrapper's
own comments at `/mnt/d/mc3-disk/tesseract/tesseract`). It is *not* in the image yet
and *not* in any dependency file. It exists so the engine's real behaviour could be
measured, because Stage 7A's two tolerance constants were wrong until measured and
there was no reason to expect the engine to be better behaved.

## What the engine actually returns

Both sample images were run through every useful page-segmentation mode. The result
is the single most important finding in this stage:

**No single PSM mode reads the whole pinout diagram.**

| Mode | Designators recovered from `specs/backplane_pinout.png` |
|---|---|
| `--psm 3` (auto) | `B14`, `B15` — only the two right-hand ones |
| `--psm 4` (single column) | `B14` |
| `--psm 6` (uniform block) | **none at all** |
| `--psm 11` (sparse) | `B11`–`B16` — all six |
| `--psm 12` (sparse + OSD) | `B11`–`B16` — all six |

q7's answer is `B14` paired with `THERM_ALERT#`. **`--psm 6`, the mode the Stage 7
investigation assumed, would have made q7 unanswerable** — it returns the six signal
labels and no designators whatsoever, so the association `B14 | THERM_ALERT#` cannot
be formed at any tolerance. psm 3 and psm 4 recover *part* of the row, which is worse
than none: a partial row looks like a diagram that is merely incomplete.

`--psm 12` was chosen as the primary mode. It reads both sample images completely and
`--psm 11` produced byte-identical results on both, so the choice is not delicate.

## The geometry, on real engine output

Real TSV rows were parsed to `Detection` and passed to `layout_from` unchanged. On
`specs/backplane_pinout.png`, `--psm 12`, `image_width=1000`:

```
layout: 16 lines, 6 associations, 0 rejections, 1 dropped, 0 duplicates
ASSOC 'B14 | THERM_ALERT#'  <- text lines 5 and 11 at region (515, 150)-(624, 233)
```

All six pairings are correct, one per column, with no rejection. The measured table is
regular to about one pixel: designators at `top=150` height 16, labels at `top=222`
height 10, so the vertical gap is 56 px = 5.6 label heights, comfortably inside the
`ASSOCIATION_MIN_GAP_FACTOR = 2.0` / `MAX = 12.0` window that Stage 7A retuned from a
guess of 1.0/6.0. Centre offsets run 0.5–1.0 px, against an allowance of 0.75 x the
label width. The widest label, `THERM_ALERT#` at 109 px, is 0.109 of the image width
against a 0.50 ceiling.

On `support/asset_label.jpg` the geometry correctly finds **zero associations** — a
flat label is not a diagram — and returns the line `'BOARD REVISION REV-C2'` intact,
which contains q8's answer. Every PSM mode agrees on that line.

## Adapter facts this measurement established

These are the things that would otherwise have been guessed wrong:

1. **The confidence column is a float, not an integer.** Real values are
   `91.944946`, not `92`. `int()` raises `ValueError` on it — this was hit during the
   measurement, not predicted. It is a percentage and must be divided by 100.
2. **Diagram rules arrive as words.** Tesseract reports drawn borders and table rules
   as level-5 rows whose text is a single space `" "` — not an empty string, so a
   naive `if not text` check passes them through. On the pinout at `--psm 3`, 15 of 55
   rows are like this.
3. **The geometry already handles (2).** `_is_usable` strips and rejects them, and
   `layout_from` counted all 15 as dropped. The adapter does **not** need its own
   blank filter; duplicating the rule would be a second place to get it wrong.
4. **Punctuation-only tokens are dropped too.** The `-` in the title is discarded by
   the same `any(character.isalnum())` test — correct here, and worth remembering on
   a page where a lone symbol is the answer.
5. **The geometry renders lines better than Tesseract's own text output.** At
   `--psm 6` the plain-text rendering returns `MADE IN MALAYSIA AN`, gluing the logo
   onto a real line; as separate boxes the geometry keeps them apart. The logo reads
   `ii` or `AN` at confidence 0.52, only just above `MIN_OCR_CONFIDENCE = 0.50`.

## Timing

Per image per mode, including library load: **0.46–0.85 s** (cold start 0.80 s versus
0.84 s warm, so the load is not a factor). A three-mode union over a few dozen images
costs well under a minute against the 600 s index budget. Cost is not the constraint
that decides the mode; correctness is, and correctness says `--psm 12`.

## What this stage still does not do

- **The engine is not in the image.** The Dockerfile, requirements files and
  `starter/app/` are untouched by this measurement; nothing under `starter/app/`
  imports Tesseract.
- **The seam is not wired.** `image.py` still reports the vision path unavailable.
  Rule 8 is still satisfied by the existing code.
- **The adapter does not exist yet.** It is deliberately unwritten: it is now a
  transcription of five measured facts rather than of a guess, and it should be
  written as one small verified step.

---

# Stage 7C — the TSV adapter

`starter/app/ingestion/readers/ocr_tsv.py` (new) and
`tests/test_readers_ocr_tsv.py` (new, **44 tests**). The last of the deferred steps:
the pure function that turns Tesseract's stdout into `Detection`s. It is the only
place in the project that knows the engine's output format.

**Where the line falls between it and the geometry.** The adapter drops a row only
when the *format* says it is not a word — the wrong level, the wrong field count, a
coordinate that is not a number. It deliberately does **not** decide that a space or
a lone `-` is not text, because `_is_usable` in the geometry already makes that
decision for its own documented reason. A second copy of the rule would be a second
place for it to drift. Measured on the real pinout at `--psm 3`: the adapter passes
all 15 rule rows through and the geometry drops all 15. Two tests assert both halves
of that contract.

The other deliberate refusal is repair. A row with the wrong number of fields is
counted and skipped, never fixed up: rejoining a corrupt row's tail into the text
field is exactly how a value that appears nowhere in the image would reach the index.
`skipped_rows` counts only rows that *claimed* to be words, so the large and
uninteresting number of page/block/paragraph rows does not bury real corruption.

`TsvWords.header_recognised` exists because "no detections" has two causes that must
not look alike — the engine produced no TSV at all (an error message, a wrong command
line) versus the image was read and had no words. Collapsing them would let a broken
invocation pass for a blank page.

## Verified against the real engine

The synthetic fixtures pin the format. The real check is the adapter fed Tesseract's
actual stdout for both sample images, unmodified:

| Image | psm | header | words | skipped | → lines | assoc | reject |
|---|---|---|---|---|---|---|---|
| `backplane_pinout.png` | 12 | ✓ | 48 | 0 | 16 | **6** | 0 |
| `backplane_pinout.png` | 11 | ✓ | 48 | 0 | 16 | **6** | 0 |
| `backplane_pinout.png` | 6 | ✓ | 42 | 0 | 10 | 0 | 0 |
| `asset_label.jpg` | 12 | ✓ | 13 | 0 | 6 | 0 | 0 |
| `asset_label.jpg` | 11 | ✓ | 13 | 0 | 6 | 0 | 0 |
| `asset_label.jpg` | 6 | ✓ | 13 | 0 | 6 | 0 | 0 |

The six associations at psm 12/11 are `B11 | GND`, `B12 | SDA`, `B13 | SCL`,
**`B14 | THERM_ALERT#`**, `B15 | PRSNT#`, `B16 | GND` — identical to the result
produced earlier by a hand-written parser, which is the point of replacing it. The
`skipped=0` column is itself evidence: the real output has no malformed rows, so
nothing is being silently discarded.

The psm 6 row is the one worth staring at: 42 words read, `0 associations`. The
engine succeeded, the geometry succeeded, and q7's answer is still unreachable. A
pipeline can be green on every file and still be unable to answer.

## Test results

- `tests/test_readers_ocr_tsv.py` — **44 passed**.
- Full suite — **654 passed in 69.44s** (was 610; +44).

One test failed on first run and it was the *fixture*, not the code: invalid UTF-8
was appended to the end of a line, so the undecodable bytes landed inside the text
field and `'SENTINEL��'` came back. That is the faithful answer — the
adapter kept the row, replaced the bytes and did not raise, which is exactly the
walker's contract. The assertion was corrected to expect the damage rather than
pretend it away, and a second test now covers a whole damaged row not costing its
neighbours.

## What this stage still does not do

- **The module is not referenced by the application.** Nothing under `starter/app/`
  imports `ocr_tsv`; it is reachable only from its tests, exactly as `image_geometry`
  was after Stage 7A. `tests/test_import_graph.py` gained one line so the module is
  still proven to import by bare name the way the container does.
- **No subprocess and no engine.** The adapter takes a string. Nothing in it shells
  out, and nothing has been added to the Dockerfile or any requirements file.
- **`image.py` is still byte-identical.** Images still report `VISION_UNAVAILABLE`.
- **Nothing downstream changed.** No extraction, citation or GPU change.

## Next step

Wire the two halves into `image.read`: invoke the engine, parse with
`words_from_tsv`, call `layout_from`, and emit the result as `TextSpan`s. That step
is where behaviour changes — images stop reporting `VISION_UNAVAILABLE` — so it needs
the engine in the image first, which is a Docker change and therefore needs approval.

The open question for that step is what the reader does when the engine is absent:
the honest options are to keep reporting `VISION_UNAVAILABLE` (this build cannot read
images) or to treat a missing binary as a failed read, and that choice should be made
deliberately rather than falling out of whichever `except` runs first.

---

# Stage 7D — the reader wired, and images now read

New: `starter/app/ingestion/readers/ocr_engine.py`, and `image.read` rewritten.
`tests/test_readers_ocr_engine.py` (new, 25 tests); `tests/test_readers_image.py`
amended. **The vision path is now live** — an image with an engine present reads
successfully and produces spans.

## The decision left open at the end of 7C, made deliberately

**A missing engine reports `VISION_UNAVAILABLE`; a present engine that fails reports
`UNREADABLE`.** The two are properties of different things — one of the build, one of
the file — and they call for different responses from whoever reads the report.
`find_engine` therefore returns `None` rather than raising, and `EngineFailed` is
raised only once an engine has been found. Rule 8 is satisfied on the absent path by
the same message it always had, so a machine without Tesseract behaves exactly as
before this stage.

## The engine setting is a command, not a path

Split with `shlex.split`, so it may carry arguments. Not decoration: the container
runs `tesseract` off `PATH`, while this laptop's rootless install needs
`sh /mnt/d/mc3-disk/tesseract/tesseract`. Treating it as one opaque path makes the
second case unexpressible. It also makes the whole chain testable with no engine
installed — the stand-in is `sys.executable` running a short script, which needs no
executable bit and no binary.

The image path is passed as one element of an argument vector. **No shell is
involved**, so a corpus file named `; rm -rf / #.png` is a filename, not a command,
and there is no quoting rule to get wrong.

## psm 11, not 12 — a trap the measurement caught

`--psm 12` produced byte-identical output to `--psm 11` on both sample images, and
was the obvious pick. It is the wrong one: mode 12 requests orientation detection and
so needs `osd.traineddata`. Verified directly by running 12 with only
`eng.traineddata` present — the engine emits `Failed loading language 'osd'` and
`Tesseract couldn't load any languages!` before continuing. Depending on a second
data file to obtain the same bytes is a liability with no upside, so the setting is
**11**. This also means the image needs only `tesseract-ocr`, not
`tesseract-ocr-osd`, when the Dockerfile is written.

## End to end, on the real corpus, with the real engine

`image.read` run against both sample files with Tesseract 5.3.4:

| File | status | spans | notes |
|---|---|---|---|
| `specs/backplane_pinout.png` | `ok` | 22 | 16 lines + 6 associations |
| `support/asset_label.jpg` | `ok` | 6 | 6 lines, 0 associations |

`B14 \| THERM_ALERT#` is span 18, located `text lines 5 and 11 at region (515, 150)-(624, 233)`
— both source boxes named, so the pairing can be checked against the image.
`BOARD REVISION REV-C2` is span 3 of the label. Neither value was hardcoded anywhere;
both were read.

The pinout's detail line reads `16 text lines and 6 associations; discarded 1
detections that were not text` — the discarded one is the lone `-` in the title,
dropped by the geometry's alphanumeric rule as designed in 7A.

## A boundary checked before it could bite

The spans are now what retrieval searches, so the question is whether their tokens
survive tokenisation. Measured:

```
q7 tokens          : [which, backplane, pin, carries, therm, alert, on, the, tq, 40, tq40]
'B14 | THERM_ALERT#' -> [b14, therm, alert]      overlap: alert, therm
q8 tokens          : [what, board, revision, is, printed, on, the, asset, label]
'BOARD REVISION REV-C2' -> [board, revision, rev, c2, revc2]   overlap: board, revision
```

`normalise` keeps `#` (the contract deletes only `-` `.` `·` `_`), and it is the same
function applied to answers, so question and corpus text normalise alike. Each
question's distinctive tokens are present in the span the reader produced. **This is
token overlap, not a retrieval run** — the full pipeline is Stage 8 — but it is the
specific failure that would have made the vision work pointless, and it does not
happen.

## Test results

- `tests/test_readers_ocr_engine.py` — **25 passed**.
- Full suite — **681 passed in 73.40s** (was 655).

Three things had to be fixed, and one of them was a real defect in the suite rather
than in the code:

1. **`test_readers_image.py` was host-dependent.** Its 22 tests passed only because
   this shell has no `tesseract` on `PATH`. In the container they would have taken a
   different path. An autouse fixture now points `MC3_TESSERACT` at a path that does
   not resolve, so "this build has no engine" is pinned rather than inherited from
   the machine. A test that means different things in different places is worse than
   no test.
2. **`test_dimensions_are_not_emitted_as_text` could not fail** where it was. With no
   engine every read returns no spans, so the assertion held for the wrong reason. It
   now lives in the engine test file, where a read can succeed and the claim has
   teeth — a successful read of a 1400x900 PNG must contain neither number.
3. `test_the_detail_states_what_was_recognised` asserted the substring `"vision"`,
   which the new message no longer contains. The assertion now names the actual
   cause, `"no OCR engine is installed in this build"`, which is the stronger claim:
   it pins that the build is blamed rather than the file.

## What this stage still does not do

- **The engine is still not in the image.** `starter/Dockerfile` and every
  requirements file are untouched. The reader works on this laptop only because
  `MC3_TESSERACT` is set by hand; with nothing set it reports `VISION_UNAVAILABLE`
  exactly as before. **Until the Dockerfile changes, the container will not read
  images.**
- **No full-pipeline run.** The spans have not been indexed, retrieved or extracted
  from. Stage 8 is where that happens.
- **The known noise is still noise.** The asset label's logo reads as `ii` at
  confidence 0.52, just above `MIN_OCR_CONFIDENCE = 0.50`, and enters the index as a
  two-character span. Raising the floor would drop genuine text too — real lines sit
  at 0.51–0.55 — so it is left alone and recorded here rather than tuned away.

## Next step

Add Tesseract to the image. That is a Docker change and needs approval: it means a
line in `starter/Dockerfile` (`apt-get install -y --no-install-recommends
tesseract-ocr tesseract-ocr-eng`), roughly 8 MB of packages, and a rebuild. The
command to check afterwards is `python3 /app/app.py --query-id query_07` with a bare
environment, confirming the reader finds the engine without `MC3_TESSERACT` set.


# Stage 7E — the engine goes into the image

## The change

`starter/Dockerfile` gains one layer, placed immediately after `WORKDIR /app` and
*before* `COPY models/ /models/`:

```dockerfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
 && tesseract --version \
 && rm -rf /var/lib/apt/lists/*
```

No Python code changed. The reader already looks for `tesseract` through `PATH`
(`DEFAULT_OCR_ENGINE`), so putting the binary on `PATH` *is* the wiring — there is
no second place to keep in sync.

## No environment variable is needed, and that was the point

`ocr_engine.py` resolves `DEFAULT_OCR_ENGINE = "tesseract"` with `shutil.which`.
Because the default is the name apt installs, the container needs no
`MC3_TESSERACT` at all. That matters for more than tidiness: an image that
depended on an env var would report `VISION_UNAVAILABLE` the moment the harness
invoked `app.py` with a clean environment, and the failure would look like a
missing engine rather than a missing setting. Leaving the variable unset in the
image also keeps it free for what it was designed for — a development machine
with the engine unpacked in an unusual place.

## Why this position in the file

Docker caches a layer until one of its inputs changes, and the ordering here is
deliberate: the engine install depends on neither the weights nor the application
code, so it sits above both. A Python edit reuses this layer, and the 129 MB
weights layer beneath it. Placing it after `COPY app/` instead would re-run apt on
every source change — the slowest possible arrangement, since apt is the only step
that needs the network.

## Why the version check is inside the layer

`tesseract --version` is not decoration. Without it, a package that installed but
could not execute — a missing shared library, the usual cause — would produce a
*successful build* and a corpus-wide `VISION_UNAVAILABLE` discovered only at
scoring time. With it, the same fault fails the build, where it is cheap and
attributable. This is the rule-8 obligation ("never silently skip image support")
expressed as a build-time assertion rather than a runtime hope.

`--no-install-recommends` keeps out the X11, additional image-format and extra
language packages the default set pulls in; `-eng` is the only language the corpus
needs (`OCR_LANGUAGE`). The apt lists are dropped in the same `RUN`, because
removing them in a later layer would leave them in the image anyway.

## Not yet verified

Everything below is a plan, not a result. **The build has not run.**

- **The base image is not fully pulled yet.** See "The base image" below; the
  rebuild cannot start until it lands.
- **The layer's installed size is unmeasured.** Expected to be single-digit MB
  (the engine plus English data), but that is an expectation, not a measurement —
  it must be read off the built image against the 60 GiB limit.
- **`apt` on the base has not been exercised.** The image is `ubuntu26.04`; that
  its sources resolve and that `tesseract-ocr` is available there is untested.
- **The bare-environment check has not been run.** The verification is
  `docker run --rm --network none <image> python3 /app/app.py --corpus /app/corpus
  --query-id query_07 --query "..."` with **no** `MC3_TESSERACT` — confirming the
  engine is found and the image-bearing question is answerable end to end.
- **Whether the corpus report changes.** Once the engine is found, image files
  stop reporting `VISION_UNAVAILABLE` and start contributing spans. That will move
  retrieval results and must be re-measured, not assumed.

## The base image

The rebuild needs `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
(~29 GiB unpacked, not the ~48 GB obtainable by assuming containerd keeps both the
compressed blob and the unpacked copy — measured, it keeps only the unpacked one).
The pull had downloaded 5 of 7 layers before stalling out earlier; at the time of
writing it is fetching the remaining 18.6 GiB at ~1.03 MiB/s, which is a
**local Wi-Fi limit, not a registry one**: the adapter is on 2.4 GHz 802.11n at 97%
signal, four parallel connections do not beat one, and Docker Hub's CDN and
Cloudflare test at the same rate. The container is unaffected; the consequence is
wall-clock, and it is why this stage is written up before it is built.

A third-party registry mirror was considered and rejected: the base image is
checked by **layer identity**, so a mirror that served a byte-different image
would disqualify the submission rather than speed it up.


# Stage 8A — the whole pipeline, run and scored

## Result

**10/10 exact** on the sample set: every answer matches and every citation set is
exactly right, with the refusal on query 10 correctly returning an empty answer
and no citations. Index 8.0s, queries 6.0–9.3s each.

```bash
cd /mnt/d/rag && source ~/venvs/mc3/bin/activate
export MC3_TESSERACT=/mnt/d/mc3-disk/tesseract/tesseract
python3 scripts/score_sample.py --device cpu --model-dir starter/models/bge-small-en-v1.5
```

This is the first time the pipeline has been run end to end over all ten
questions. Stages 3–7 tested each part; nothing had run the whole thing.

## The vision path is load-bearing, not decorative

Queries 7 and 8 answer `B14` and `REV-C2`, values that appear in **no text file** —
they exist only inside `specs/backplane_pinout.png` and `support/asset_label.jpg`.
Both were answered correctly from the index. The same corpus indexed with
`MC3_TESSERACT` unset reports `vision_unavailable=2`, drops from 95 chunks to 67,
and those two questions become unanswerable. The engine is the difference between
8/10 and 10/10, which is what Stage 7E exists to ship.

## Lexical-only is not a safe fallback — it changes answers

The first run scored **9/10**, and the missing point was query 10. The cause was
not the refusal logic: the index had silently degraded to lexical-only, because
`MC3_EMBEDDING_DEVICE='cuda'` was set by default and this laptop has no GPU. The
application refused to fall back to the CPU on its own — correctly, per rule 7 —
and indexed without vectors.

Re-run in true hybrid mode, query 10 does not merely improve, it produces a
*different answer*: lexical-only returns `212` citing `support/rma_parts.xlsx`
(a confident wrong answer to the unanswerable question), hybrid returns nothing.
So the vector retriever is not a quality knob that can be dropped under pressure;
**the two configurations disagree about correctness**, and any future fallback
that lands in lexical-only must be treated as a behaviour change, not a
degradation in degree.

**The refusal rule is principled, but its word-overlap test can be defeated — measured.**
The rule (`answering/refusal.py`) refuses when the winning value is presented as a
field's value *and* the field's name shares no word with the question. Between the
two runs **the rule did not change; the winner did**, and that is the whole of the
difference:

- **Hybrid refuses.** The winning candidate is labelled `lead_time_days`, which
  shares no word with a question about unit price, so the rule fires.
- **Lexical-only answers `212`.** The winner comes from the row
  `part_number=ORR-HSK-1180-A | description=Heatsink, vapour chamber |
  compatible_with=TQ-40 | unit_cost_usd=212 | lead_time_days=35`. The field name
  `unit_cost_usd` shares the word **"unit"** with "unit price", so the rule does
  not fire and the value is answered.

So hybrid gets this question right partly for an incidental reason: it happened to
rank a differently-labelled candidate first. The rule did **not** identify the
price question and reject it — the word-overlap test was defeated by a generic
word. An earlier draft of this note said the lexical-only run "correctly stays
silent"; that was inference, and checking it against the index showed the
opposite.

**Residual risk, stated plainly.** `unit_cost_usd` is a genuine near-miss — cost
is not price, and the real price exists only in the encrypted file — and it passes
the rule. On a larger corpus, if the top candidate for such a question is a
`*_cost` or `*_usd` field, the rule will answer it and spend the point. The
module's docstring already names the assumption this rests on ("field names are
written in words a question would also use") and the risk of cryptic names making
a correct answer unreachable; this is that same assumption failing in the
*opposite* direction, by overlapping too easily. It is recorded rather than
patched because the module's docstring also documents the three alternatives that
were measured and rejected, including one that separated all ten samples and was
rejected for being a statistic fitted to a single negative example — and
tightening this one on the same evidence would repeat that mistake.

## New tooling: `scripts/score_sample.py`

`AGENTS.md` listed this file as existing. It did not — `scripts/` held only
`fetch_model.py` — so the number above was first obtained from a throwaway shell
loop and is now produced by a committed tool.

Two bugs were found in it by running it, both of which produced a *plausible wrong
number* rather than an error, and both are fixed:

- **Relative paths resolved against the wrong directory.** `app.py` is invoked with
  its own directory as the working directory, so `--model-dir
  starter/models/bge-small-en-v1.5` missed, and the run quietly degraded to
  lexical-only and scored 9/10. Every path argument is now resolved before it
  reaches a child process.
- **The index report was discarded.** `app.py` reports on **stderr**, not stdout.
  The scorer echoed stdout only, so the one line announcing
  "no embeddings, indexing lexical-only" — the exact degradation above — was
  invisible. Both streams are now echoed.

That second bug is the more instructive one: the scorer was hiding the failure it
had been written to expose.

## The scorer deliberately does not hold the answers

Every expected value is read from `sample-questions.json` at run time; nothing is
hardcoded, so rule 1 holds. Answer comparison imports the application's own
`normalisation.normalise` rather than reimplementing it, because `AGENTS.md`
requires one comparison form and a drifted copy would score answers differently
from the pipeline that produced them. Comparison is exact against the expected
value or a declared alias — a substring match would let `194` pass as `94` — and a
substring-only match is labelled "close" without being credited. Citations are
compared as exact sets, since an extra citation scores zero exactly like a missing
one. The file lives in `scripts/`, which the Dockerfile does not copy.

## An environment trap worth knowing

`pytest` in this repo fails with **"no tests ran"** and a `FileNotFoundError` from
its capture plugin, because `~/.zshrc` exports `TMPDIR` onto the 9p mount at
`/mnt/d`. It reads as a collection failure and is not one. With `TMPDIR=/tmp` the
same tree gives **681 passed**. Use `TMPDIR=/tmp python3 -m pytest tests/ -q`.

## What this stage does not prove

- **The sample set is not the graded set.** It is 13 files and 95 chunks; the
  graded corpus is different and larger. 10/10 here is evidence, not a score.
- **Nothing here ran in the container.** Every number is from the laptop, on CPU.
  The image, `selfcheck.py` and the GPU are all still unexercised — Stage 7E is
  written but not built.
- **The timings are CPU timings.** 9.3s against a 30s budget has real headroom,
  but ROCm context initialisation per process is not free and is unmeasured. A
  per-query regression there is the likeliest way this budget is lost.
- **`docs/` is empty.** `AGENTS.md` describes it as holding architecture, a
  learning guide and testing notes; none exist yet.
- **Confidence is unvalidated.** It ranges from 0.27 (query 9) to 1.0 (query 3)
  while correctness is uniform, so it is not currently a usable signal of whether
  an answer is right.

## Next step

Build the image once the base finishes pulling, then run
`selfcheck.py` against the sample corpus — that is the first check of the
container contract itself (startup, `--index`, one query, output shape, relative
citations), and it is the nearest thing available to the real harness.


# Stage 8B — the corpus hazards, actually present

## The sample corpus has been lying to us on this machine

`README.txt` specifies that the sample corpus contains **`mc3-corpus/archive/` as
an empty directory** and **`vendor/internal_audit.txt` as chmod 000**. The
directory is there. The unreadable file is not unreadable — it is an ordinary
readable file, because the project lives on a 9p/drvfs mount where `chmod` is a
documented no-op (`AGENTS.md`, verified previously). Every local run so far,
including the 10/10 in Stage 8A, was therefore measured on a corpus with **both of
those hazards silently absent**.

That is the failure mode `README.txt` warns about, in its own words: a submission
"will pass your own testing and fail ours". The number was right and the
configuration was wrong.

## Verified properly, on a native filesystem

The corpus was copied to `/tmp` (ext4), `chmod 000` applied, and the copy
confirmed genuinely unreadable (`cat` is refused):

```
indexed /tmp/mc3-hazard: 12 files, 93 chunks
  file status: encrypted=1, ok=10, unsupported=1
  skipped during walk: 1
```

The unreadable file is skipped, **the walk completes**, and no other file is lost
— 13 files becomes 12 with `ok` dropping 11 to 10, exactly the one file. **Score
unchanged at 10/10.** Rule 9 ("never let one bad file abort the corpus walk") holds
end to end, not merely in the walker's unit tests.

The unit level was checked separately rather than assumed: with `TMPDIR=/tmp`,
`tests/conftest.py`'s probe finds permissions genuinely enforced and
`test_walker.py`'s 17 tests **run** rather than skip — including
`test_unreadable_file_does_not_abort_the_walk` and
`test_empty_directory_is_recorded_not_an_error`. On the 9p mount those same tests
skip with a stated reason, which is the honest behaviour and also why the skip
count must never be read as a pass.

## The habit this establishes

Local green results describe **the corpus as this filesystem presents it**, not
the corpus as it is. Any hazard whose mechanism is a mode bit, a permission, or a
mount option has to be reproduced on native storage before it can be claimed as
covered. `archive/` and the encrypted PDF and the unknown binary are
filesystem-independent and were already exercised; the permission case was not.

## Next step

Unchanged: build the image and run `selfcheck.py`. Note that `selfcheck.py` copies
the corpus into the container with `docker cp`, which **does not preserve mode
bits** either — so the unreadable-file hazard will not be exercised by selfcheck
on this machine regardless. The real harness's `--cap-drop DAC_OVERRIDE` behaviour
remains untestable here and must be checked on the notebook.


# Stage 8C — the container test sequence, ready but not yet run

Everything here is blocked on one thing: the base image finishing its pull. The
Dockerfile is written (7E), the local pipeline is verified (8A/8B), and the
scorer can now read container output directly (`--reuse`), so the moment the base
lands this is mechanical rather than improvised.

```bash
cd /mnt/d/rag

# 1. build
docker build -t mc3:local starter/

# 2. the harness's own checks, in the harness's own order
python3 selfcheck.py mc3:local mc3-corpus

# 3. all ten questions, inside the container, then scored by the same comparator
docker run -d --network none --name mc3-test mc3:local
docker exec mc3-test mkdir -p /app/corpus /app/output
docker cp mc3-corpus/. mc3-test:/app/corpus
docker exec mc3-test python3 /app/app.py --index /app/corpus
python3 - <<'PY'          # drives the ten execs from the sample file
import json, subprocess
for q in json.load(open('sample-questions.json'))['queries']:
    qid = f"query_{q['n']:02d}"
    subprocess.run(["docker","exec","mc3-test","python3","/app/app.py",
                    "--corpus","/app/corpus","--query-id",qid,"--query",q['query']],
                   check=False)
PY
docker cp mc3-test:/app/output/. /tmp/mc3-container-out
python3 scripts/score_sample.py --reuse --output-dir /tmp/mc3-container-out
```

## What each step is actually testing

- **`--network none` on the build run.** Not decoration: it is how the harness
  runs, and it is the only way to catch a runtime download. The model must come
  from `/models/`, which the Dockerfile copies.
- **`selfcheck.py`** checks the *contract*: startup, the `--index` pass, one query
  inside 30s, the output file at `/app/output/<query-id>_output.json`, the JSON
  shape, and that citations are corpus-relative. It runs **one** query, so it
  cannot tell us whether the answers are right.
- **step 3** is the one that scores correctness inside the container, which is the
  combination nothing has yet produced: every number so far is from the laptop.

## What this sequence still cannot prove

- **The GPU path.** No AMD GPU on this machine, so the container will report
  `MC3_EMBEDDING_DEVICE='cuda' was requested but torch cannot see a GPU` and index
  **lexical-only**. The embedding model, the ROCm build of torch, the VRAM floor
  and ceiling, and per-query ROCm init are all unexercised. Expect the container
  score to read **9/10**, matching the lexicon-only result in 8A, and *not* to be
  alarmed by it — it is the expected CPU outcome, not a regression.
- **The unreadable-file hazard.** `docker cp` does not carry mode bits, so the
  copied-in corpus will have a *readable* `internal_audit.txt` whatever the source
  looked like.
- **The 60 GiB size limit.** `docker history` sums layers; the real check is
  uncompressed image size, and it needs the built image.
- **Timing headroom.** CPU timings (6–9s) are not ROCm timings.

Stage 9 (the notebook) is where the GPU half gets answered.


# Dependency closure verified for Python 3.14

`AGENTS.md` states the risk: the laptop is Python 3.12.3 while the image is
Python 3.14, and "a package that installs locally may have no 3.14 wheel. Verify
in the image before committing to a dependency." A compiled dependency with no
3.14 wheel would need a source build and a toolchain the image does not have, and
that failure would only appear *after* the base image finished downloading — the
most expensive possible moment to discover it.

`requirements.txt` declares only two packages, and its comment claims the abi3
question was already checked against PyPI. **That claim was re-verified
independently rather than trusted**, and extended past the two named packages to
the whole unconditional closure, since a transitive compiled dependency is exactly
what would break a build:

- **transformers 5.19.0** — `py3-none-any`, pure Python, `requires_python >=3.10.0`.
- **safetensors 0.8.0** — `cp310-abi3-manylinux_2_17_x86_64`, stable ABI, so it
  installs on 3.14 without a build.
- **tokenizers 0.23.3** — publishes both abi3 *and* cp314 wheels.
- The other six — `huggingface-hub 2.2.0`, `numpy 2.5.4`, `packaging 26.3`,
  `pyyaml 6.0.3`, `regex 2026.9.29`, `typer 0.27.3`, `tqdm 4.70.1` — all carry a
  3.14-compatible wheel.

transformers declares 257 requirements, but only **9 are unconditional**; the rest
are extras, which a plain `pip install -r` never pulls.

**What this does and does not settle.** It settles wheel *availability*, which is
the part that cannot be fixed after the fact. It does not verify that pip resolves
to these versions inside the image, nor that the install succeeds there — that
still needs the image, and it is the first thing to watch in the build log. Note
also that `torch` is deliberately absent from `requirements.txt`: naming it would
let pip replace the ROCm build with a CPU or CUDA one and quietly cost the run its
GPU, which is the same failure rule 7 forbids.

## `docs/` is no longer empty

`AGENTS.md` line 110 says `docs/` holds "architecture, learning guide, testing
notes". It held a `.gitkeep`. All three are now written, and the reason to do it
now rather than after the container run is that writing them down is what caught
the mistake in the next section.

- **`docs/architecture.md`** — the pipeline end to end, the two-phase shape and
  why it is forced, the module map, the budgets, and the layout rule.
- **`docs/testing.md`** — how to run the suite, what each file covers, the two
  tests that are unlike the others (`test_index_cli.py` runs the real command
  line; `scripts/score_sample.py` is deliberately outside the suite), the
  measured results, and an explicit list of **where the suite's guarantees stop**.
- **`docs/learning-guide.md`** — the concepts, each grounded in the file it lives
  in, for someone who has not built a retrieval system before.

Nothing under `starter/app/` was touched, so this adds nothing to the image.

## A robustness gap that turned out not to be one

While listing what remained, I found that `app.py`'s `main()` has **no top-level
exception guard**, and said so as a gap: any unexpected exception, or an
uncreatable output directory, kills the process and writes **nothing**. On a
question that is answered by refusal, an empty response and a crash would score
the same, so the fix looked obvious — catch broadly and still write
`{"answer": "", "citations": [], "confidence": 0.0}`.

**That fix would have been wrong, and the contract says so explicitly**
(`CONTRACT.md` lines 81-82):

> A missing `answer` or a missing `citations` is **malformed and scores zero** —
> a crashed writer and a considered refusal must not look the same.

Writing an empty answer on an unexpected exception is precisely "making a crashed
writer look like a considered refusal". On the question that has no answer in the
corpus it would *gain* a point by claiming a judgement the system never made.

Checking further, the behaviour is already correct and deliberate:

- `answer()` catches its **expected** failures — `IndexUnavailable`,
  `IndexIncompatible`, `HybridUnavailable` — and returns `("", [], 0.0)`, each
  with a stderr line naming the cause. The comment on the first of these already
  cites the contract clause above.
- `tests/test_index_cli.py` pins the contract on the failure paths as well as the
  happy one: three required keys, filename from `--query-id`, a missing index, a
  stale index, a missing `--query-id`. Its module docstring quotes the same clause.
- An **unexpected** exception is left to crash. That asymmetry is the design.

So `main()` stays unguarded. The residual is real but different from what I first
said: the only failures that reach it are unforeseen bugs, and those *should* be
loud. Correcting this mattered more than the original claim, because "add a catch-all
so we always write a file" is a tempting change that a future reader — or I — could
make in good faith and quietly lose a point to.

**Lesson for the notes:** when an output contract has a term for a failure mode,
read what it says that term is *for* before writing code that produces it.

## Baseline re-verified, and one more reason `TMPDIR` matters

`TMPDIR=/tmp ~/venvs/mc3/bin/python -m pytest tests/ -q` → **681 passed, 0
skipped, 61.9 s**, on an unchanged tree.

Zero skips is itself the result. `tests/conftest.py` probes whether `chmod 000`
genuinely denies reads in the temp location before running the permission tests,
and skips with a stated reason when it does not — because a test that cannot fail
reports safety that does not exist. Measured directly:

| location | filesystem | `chmod 000` enforced |
|---|---|---|
| `/tmp` | ext4 | **yes** |
| `/mnt/d/mc3-disk/tmp` | drvfs/9p | no |
| `/mnt/d/rag` | drvfs/9p | no |

So `TMPDIR=/tmp` is load-bearing for **two independent reasons**, and the second
was not previously recorded: besides avoiding the capture-plugin teardown crash
described in [[pytest-tmpdir-9p]], it is what decides whether the unreadable-file
tests test anything at all. Both are now in `docs/testing.md`.

## Container phase still blocked

The base image is still pulling. Progress is measured by the growth of the data
disk rather than by the pull log, because `docker pull`'s progress lines are not
reaching the log file (only the discrete state transitions are):

- 7 layers, 5 already present, `b5d678c07f41` complete;
- the remaining layer is `1e78e706bbca` at 19032.4 MiB — the one that matters;
- internal `/var/lib/docker` usage at last sample **8.91 GiB**, growing.

Nothing in the container phase — the build, `selfcheck.py`, the ten questions in
the image, Stage 9's GPU and VRAM work — can start until it lands, and at the
measured rate it is a multi-hour wait on an ~11 Mbit/s link that nothing local can
improve.

**Rate measured properly**, rather than inferred: sampled the data disk's internal
usage 326 s apart — grew **412 MiB, i.e. 1.26 MiB/s**, with
`/var/lib/docker` at **8.72 GiB of a ~29 GiB target**. That puts the remaining wait
at roughly **4–5 hours**. The earlier figure of 1.23 MiB/s was the whole Wi-Fi
adapter, so the pull is still taking effectively the entire link.

## The local 10/10 needs an environment variable, and drops to 8/10 without it

Re-scoring end to end for fresh timings produced **8/10**, failing queries 7 and 8,
and the index report said why without being asked:

```
file status: encrypted=1, ok=9, unsupported=1, vision_unavailable=2
not indexed -> vision_unavailable: specs/backplane_pinout.png
not indexed -> vision_unavailable: support/asset_label.jpg
```

**The OCR engine was not found.** A fresh shell does not export `MC3_TESSERACT`,
and the local engine is a hand-unpacked tesseract 5.3.4 at
`/mnt/d/mc3-disk/tesseract/tesseract` (a wrapper that sets `TESSDATA_PREFIX`).
`tesseract` is not on `PATH` either, so `which tesseract` reports it missing and
reads as a broken install rather than an unset variable. With it supplied:

```
MC3_TESSERACT=/mnt/d/mc3-disk/tesseract/tesseract \
TMPDIR=/tmp ~/venvs/mc3/bin/python scripts/score_sample.py \
    --corpus mc3-corpus --model-dir starter/models/bge-small-en-v1.5 --device cpu
```

→ **10/10 exact.** Index 7.7 s, 95 chunks, 13 files (11 ok, 1 encrypted,
1 unsupported); queries **6.7–10.9 s**.

**This is a development-machine trap only, and worth being precise about why.**
In the container tesseract is installed system-wide by `apt-get` and the reader
finds it with no environment variable set — that is the property the Dockerfile
asserts at build time with `tesseract --version`. Nothing in the shipped image
depends on `MC3_TESSERACT` being set. But on this laptop the difference between
10/10 and 8/10 is one unexported variable, and the failure is *quiet*: two
plausible wrong answers rather than a named missing capability. That is the same
class of silent degradation the lexical-only row already demonstrated.

The per-query times are the useful number for the budget: **6.7–10.9 s on CPU,
model load included**, against a 30 s limit — and model load is inside each query
because every query is a fresh process. The container should be faster still on
the GPU, but this is the first evidence that the per-question budget has real
headroom rather than being met by luck.

One gotcha found while chasing this: `score_sample.py` defaults `--output-dir` and
`--index-dir` to `tempfile.gettempdir()`, **which follows `TMPDIR`**. A run with
`TMPDIR=/tmp` writes to `/tmp/mc3-score-output`, leaving older files in
`/mnt/d/mc3-disk/tmp/mc3-score-output` looking current. Pass the flags explicitly
when comparing runs. All of the above is now written up in `docs/testing.md`.



