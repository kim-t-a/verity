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
| 1b | Document readers (pdf, docx, xlsx, csv, text, code) | ✅ **complete** |
| 1c | Image reader interface + normalised chunks | ✅ **complete** |
| 2 | Persistent indexing (chunk store, embedding interface, fake provider) | ✅ **complete** |
| 3 | Keyword retrieval | ✅ **complete** |
| 4 | Vector retrieval | ✅ **complete** |
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
