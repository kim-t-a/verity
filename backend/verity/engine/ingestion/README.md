# `verity.engine.ingestion` — from a directory to readable text

| Module | Responsibility |
|---|---|
| `walker.py` | Walk a directory defensively. Returns discovered files and skipped entries; never raises on one bad path. |
| `filetypes.py` | Decide a file's **real** format from its leading bytes, not just its name. A `.txt` that is really a PDF is read as a PDF; an unknown binary is honestly `UNSUPPORTED`. |
| `readers/` | One module per format. Each exposes `read(discovered) -> ReadResult`. |

## The reader contract

Every reader returns a value; **no reader raises**. A malformed file costs that
file and nothing else. `read_file()` in `readers/__init__.py` is the only entry
point — it dispatches by media type and converts any unexpected exception into
an `UNREADABLE` result.

A `ReadResult` carries:

- the extracted text,
- a status (`OK`, `EMPTY`, `UNREADABLE`, `UNSUPPORTED`, `VISION_UNAVAILABLE`),
- a list of `TextSpan`s saying where each piece came from (page, line, cell).

Those spans become citations. That is why readers record locations rather than
returning bare text.

## Formats

| Format | Reader | Provenance unit |
|---|---|---|
| PDF | `readers/pdf.py` | page |
| DOCX | `readers/docx.py` | paragraph |
| XLSX | `readers/xlsx.py` | sheet + row |
| CSV / delimited | `readers/delimited.py` | row |
| TXT / LOG / source code | `readers/text.py` | line |
| PNG / JPG | `readers/image.py` (+ `ocr_engine.py`, `ocr_tsv.py`, `image_geometry.py`) | OCR box / region |

See `readers/README.md` for how to add one.
