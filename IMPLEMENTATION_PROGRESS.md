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
