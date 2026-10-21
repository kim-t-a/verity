# `verity.engine.ingestion.readers` — one module per format

Each reader exposes the same function:

```python
def read(discovered: DiscoveredFile) -> ReadResult
```

The registry lives in `__init__.py` (`_READERS`). Adding a format means adding
one module and one line to that table — not editing a chain of conditionals.

## Rules every reader follows

1. **Never raise.** Return a `ReadResult` with a failure status instead.
2. **Cap the read.** A file is not trusted to be the size its name suggests.
3. **Record provenance.** Every piece of text gets a `TextSpan`; that span is
   what a citation points at.
4. **Never invent text.** If an image cannot be OCR'd, report
   `VISION_UNAVAILABLE`; do not describe the image from its filename.
5. **Prefer stdlib.** PDF, DOCX, and XLSX are hand-parsed against `zipfile`
   and `xml.etree`. Adding a parsing dependency is a spec decision, not a
   reader decision.

## The modules

| Module | Role |
|---|---|
| `pdf.py` | Hand-written PDF parser: objects, streams, text extraction per page. |
| `docx.py` | Reads `word/document.xml` from the ZIP; ignores formatting parts. |
| `xlsx.py` | Reads every worksheet via the shared-strings and sheet XML parts. |
| `delimited.py` | Sniffs the delimiter; renders each row with its header. |
| `text.py` | Bounded read, encoding detection, line provenance. |
| `source.py` | Shared safe-read and decode helpers for text-shaped formats. |
| `tabular.py` | Renders a grid of cells as labelled rows (used by CSV and XLSX). |
| `xmlpart.py` | Loads OOXML XML with entity-expansion protection. |
| `image.py` | Image header + OCR when an engine is available. |
| `ocr_engine.py` | The only process boundary: finds and runs Tesseract. |
| `ocr_tsv.py` | Tesseract TSV → `Detection` boxes. |
| `image_geometry.py` | Groups detections into readable regions from position alone. |

## Adding a reader

1. Create `<format>.py` with a `read(discovered)` function that returns a
   `ReadResult`.
2. Add the new `MediaType` to `engine/models.py` if it is a genuinely new kind.
3. Teach `engine/ingestion/filetypes.py` how to recognise the format from its
   bytes.
4. Add one line to `_READERS` in `__init__.py`.
5. Write tests: a valid file, a truncated file, and a zero-byte file. The last
   two must return a status, never raise.
