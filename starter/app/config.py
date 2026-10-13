"""Central configuration for the MC3 RAG system.

Every tunable constant lives here rather than inline, so behaviour can be
reviewed in one place. Anything that would otherwise be an unexplained number
in the middle of a function belongs in this module.

This file grows as later stages need it; it deliberately holds only what the
code in the repository actually uses today.
"""

from __future__ import annotations

# --- Ingestion --------------------------------------------------------------

#: How many bytes of each file the walker reads. That single read does two
#: jobs: it proves the file is genuinely readable (the only reliable test --
#: permission checks can lie under root and on mounted Windows drives), and it
#: captures the leading bytes used to identify the file's true format.
#: 4 KiB is far more than any format signature needs, and small enough that
#: probing even a large corpus costs almost nothing.
PROBE_BYTES = 4096

#: The largest text file we will read whole. Anything bigger is read up to this
#: many bytes and reported as truncated. A cap exists so that one pathological
#: file cannot exhaust memory and take the whole run down with it; the value is
#: far above any real document in a corpus like this one.
MAX_TEXT_BYTES = 32 * 1024 * 1024

#: The largest single member we will decompress out of a ZIP-based document
#: (``word/document.xml``, a worksheet, ...). A ZIP can claim a huge
#: uncompressed size from a tiny compressed one, so this is checked against the
#: declared size *before* decompressing rather than after.
MAX_ZIP_MEMBER_BYTES = 256 * 1024 * 1024

#: Encodings tried, in order, when reading a text file.
#:
#: ``utf-8-sig`` comes first, ahead of plain ``utf-8``, and that ordering is
#: load-bearing. A byte-order mark is *valid* UTF-8 -- it decodes to U+FEFF --
#: so plain UTF-8 wins on a file that has one and leaves the mark glued to the
#: first token of the first line, where it silently breaks any comparison
#: against that token. ``utf-8-sig`` strips a leading mark if present and
#: behaves exactly like UTF-8 if not, so it dominates.
#:
#: CP1252 follows because it is what Windows-authored files actually use.
#: Latin-1 is last because it maps all 256 byte values and therefore cannot
#: fail, which makes it the candidate that always terminates the search.
TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

#: Encodings whose use is unremarkable and needs no note in diagnostics.
UNREMARKABLE_ENCODINGS = frozenset({"utf-8", "utf-8-sig"})

#: The share of a decoded file that may be control characters before we stop
#: believing it is text at all.
#:
#: This exists because any byte sequence decodes as *something* once Latin-1 is
#: in the list. A binary file named ``.txt`` would therefore decode into a
#: screenful of mojibake and be indexed as though it were prose. Real text --
#: including logs, source code and files with smart quotes -- sits at
#: essentially zero; the sample of every byte value sits around a quarter.
MAX_CONTROL_CHARACTER_RATIO = 0.10

#: How a table cell separator is rendered when a row is flattened into text.
#: Chosen to be visually unambiguous and unlikely to appear inside a cell.
CELL_SEPARATOR = " | "

# --- Images -----------------------------------------------------------------

#: How many bytes of an image are read to identify it and find its dimensions.
#:
#: Only the header is ever needed, never the pixels, so this stays small. It is
#: larger than ``PROBE_BYTES`` because a JPEG's frame header is not at a fixed
#: offset: it comes after whatever metadata segments the producer wrote, and an
#: embedded colour profile can push it well past 4 KiB.
MAX_IMAGE_HEADER_BYTES = 64 * 1024

# --- Chunking ---------------------------------------------------------------

#: The largest chunk the index will store.
#:
#: A chunk is the unit a query is matched against, and both retrieval methods
#: degrade on very large ones: a lexical score is diluted by every unrelated
#: word sharing the chunk, and an embedding compresses the whole thing into one
#: vector, so a single specific value inside a page of prose stops being
#: findable. Splitting bounds that damage.
#:
#: The value is a little under the token limit of the embedding models we are
#: considering, so a chunk can be embedded whole without silent truncation.
MAX_CHUNK_CHARS = 2000

#: How much of the text before the cut is searched for a clean break.
#:
#: A chunk split mid-token would store half a value, which can never be matched
#: and can never be quoted as an answer. Looking back a short distance for a
#: space costs a little and usually avoids that.
CHUNK_BREAK_LOOKBACK = 200

# --- The index --------------------------------------------------------------

#: Layout version of the persisted index.
#:
#: Bumped whenever the on-disk shape changes in a way an older reader would
#: misinterpret. A reader that finds a version it does not know must refuse the
#: index rather than guess: the alternative is answering questions from a
#: misread chunk store, which produces confident wrong answers and cites files
#: that do not support them. An index is cheap to rebuild -- ``--index`` is
#: ungraded -- so refusing costs one startup pass and no score.
INDEX_FORMAT_VERSION = 1

#: The three files an index directory holds.
#:
#: Three files rather than one because they have different shapes and different
#: failure modes. The manifest is small, human-readable and acts as the commit
#: marker: it is written **last**, so an interrupted ``--index`` leaves a
#: directory with no manifest, which reads as "no usable index" instead of as a
#: half-built one. The chunks are one JSON object per line, so a damaged record
#: costs one chunk rather than the whole file, and the store stays inspectable
#: with ``head`` and ``grep`` during development. The vectors are a raw float
#: buffer, because base64 in JSON would cost a third more bytes for no benefit.
MANIFEST_FILENAME = "manifest.json"
CHUNKS_FILENAME = "chunks.jsonl"
VECTORS_FILENAME = "vectors.bin"

# --- Embedding backends -----------------------------------------------------

#: Which embedding backend to use. Read at both ``--index`` and ``--query``,
#: and it must agree between them: an index built with one backend cannot be
#: searched with another, because the vectors are not comparable.
EMBEDDING_PROVIDER_ENV = "MC3_EMBEDDING_PROVIDER"

#: The backend used when nothing is configured.
#:
#: The default is the *real* one, never the local stand-in. That is the whole
#: safety property: a container that sets no environment variables at all must
#: not be able to end up on the CPU-only path, because a run that never uses
#: the GPU is rejected outright.
DEFAULT_EMBEDDING_PROVIDER = "gpu"

#: The CPU-only stand-in used for development and tests.
LOCAL_EMBEDDING_PROVIDER = "hashing"

#: The second key required to reach the local backend.
#:
#: Requesting it by name is not enough -- this must also be set to ``1``. Two
#: independent switches, because the failure they guard against is silent: a
#: stray environment variable left over in a shell profile would otherwise
#: build a CPU-only index that grades as a rejected run, with nothing in the
#: output saying why.
ALLOW_LOCAL_BACKEND_ENV = "MC3_ALLOW_LOCAL_BACKEND"

#: Bucket count for the local hashing embedder. Small on purpose -- it is a
#: plumbing stand-in, not a retrieval model.
HASHING_DIMENSION = 256
