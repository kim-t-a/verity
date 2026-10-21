# `backend/tests` — the only test tree

Run from `backend/`:

```bash
TMPDIR=/tmp python -m pytest -q
```

`TMPDIR=/tmp` is mandatory here. The repository is on a 9p/drvfs mount of
Windows `D:`; with the inherited `TMPDIR`, pytest places `tmp_path` on that
mount and reports "no tests ran" with a misleading `FileNotFoundError`.
`/tmp` is a native filesystem, so permission tests run for real instead of
skipping.

## Ground rules

- **No network, no real corpus, no provider key.** Anything model-shaped is
  behind an interface with a fake in `verity/llm/provider.py`.
- **No invented expected answers.** Fixtures are test data, not lookup tables.
  A test asserts what the code does with known input.
- **Watch the test fail first.** A test that has never failed has not been
  observed to test anything.

## Useful fixtures

| Fixture | Purpose |
|---|---|
| `tmp_path` | Per-test scratch directory (kept off the 9p mount by `TMPDIR`). |
| `hazard_corpus` | Builds unreadable, encrypted, empty, and hostile files; detects whether the filesystem enforces `chmod`. |
| `document_fixtures.py` | Builds DOCX/XLSX/PDF bytes in memory, with no external files. |
| `db_conn` (in DB tests) | A real PostgreSQL connection; these tests skip or fail clearly when the DB is absent. |

## Naming

Test files mirror the module they exercise:

- `test_index_types.py` → `engine/index_types.py`
- `test_chunk_repository.py` → `db/chunk_repository.py`
- `test_embedding_provider.py` → `engine/embedding_provider.py`
- `test_nvidia_embedding_provider.py` → `engine/nvidia_embedding_provider.py`

## Adding a test

1. Write the smallest test that expresses the desired behaviour.
2. Run it and read the failure.
3. Make it pass with the smallest change.
4. Run the whole suite, not just the new file.
