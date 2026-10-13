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

# --- The real embedding backend ---------------------------------------------

#: Where the model's files live inside the image.
#:
#: The Dockerfile copies ``starter/models/`` to ``/models/``, so the weights ship
#: in the image rather than being fetched at run time -- the container has no
#: outbound network during evaluation, so a model downloaded on demand would
#: hang and then time out.
#:
#: Overridable because the repository's own copy sits at
#: ``starter/models/bge-small-en-v1.5`` during development, and the tests and
#: local CLI runs read it from there.
EMBEDDING_MODEL_DIR_ENV = "MC3_EMBEDDING_MODEL_DIR"
DEFAULT_EMBEDDING_MODEL_DIR = "/models/bge-small-en-v1.5"

#: Which device the model is placed on, and the reason this is a named setting
#: rather than a guess made at run time.
#:
#: The default is ``cuda`` -- which is also how ROCm presents itself to torch,
#: so this single default covers both the CUDA and the ROCm build. The container
#: must genuinely occupy VRAM; a run that peaks below 1 GiB is judged to have run
#: on the CPU and rejected. A backend that quietly fell back to the CPU when the
#: device was missing would therefore produce a run that looks successful and is
#: disqualified, which is the worst possible failure shape.
#:
#: So there is no automatic fallback. Development on a machine without a GPU
#: sets this to ``cpu`` explicitly, which makes the CPU path a deliberate choice
#: that someone typed, never something a run drifts into.
EMBEDDING_DEVICE_ENV = "MC3_EMBEDDING_DEVICE"
DEFAULT_EMBEDDING_DEVICE = "cuda"

#: The model's name, as it appears in its provider id.
EMBEDDING_MODEL_NAME = "bge-small-en-v1.5"

#: How token vectors are collapsed into one vector per chunk.
#:
#: ``1_Pooling/config.json`` in the model directory declares CLS pooling
#: (``pooling_mode_cls_token: true``, ``pooling_mode_mean_tokens: false``).
#: That matters because mean pooling is the habit most tutorials teach for
#: turning a BERT into a sentence embedder, and using it here would still
#: produce 384-dimension unit vectors, still rank, and be quietly worse.
#:
#: Measured on the sample corpus, the two poolings produce vectors that differ
#: for all 67 chunks (cosine 0.85-0.98) yet rank all nine answerable questions
#: identically -- so the sample corpus *cannot* show which is better, and this
#: setting follows the checkpoint author's declared intent rather than a fitted
#: measurement. It is a stated judgement, not a validated one.
EMBEDDING_POOLING = "cls"

#: Output width. Fixed by the model's ``hidden_size``, and checked against the
#: weights at startup rather than trusted.
EMBEDDING_DIMENSION = 384

#: Version of the embedding *recipe* -- model, pooling, normalisation.
#:
#: Bump this whenever any change would alter the vectors produced for the same
#: text. It exists so the provider id changes with the behaviour: an index built
#: with one recipe and searched with another would yield cosines between
#: unrelated coordinate systems, which is a number rather than an error, and
#: every citation drawn from it would be unsupported.
EMBEDDING_RECIPE_VERSION = 1

#: The identity written into the index manifest.
#:
#: Derived rather than written out, so that changing the pooling, the dimension
#: or the recipe version *cannot* leave the id behind. A hand-maintained id is
#: one careless edit away from describing vectors it no longer produces.
EMBEDDING_MODEL_ID = (
    f"{EMBEDDING_MODEL_NAME}-{EMBEDDING_POOLING}"
    f"-d{EMBEDDING_DIMENSION}-v{EMBEDDING_RECIPE_VERSION}"
)

#: The longest input the model will read, in wordpiece tokens.
#:
#: ``max_position_embeddings`` in the model's config is 512, and anything past
#: it is discarded silently -- the chunk embeds as though its tail did not
#: exist. Measured over the sample corpus the longest chunk is 53 tokens, so
#: the ceiling is far away here; it is enforced rather than assumed because the
#: graded corpus is larger, and because silent truncation is invisible in the
#: result.
MAX_SEQUENCE_TOKENS = 512

#: How many chunks are embedded in one forward pass.
#:
#: Pure throughput -- it cannot change the vectors, only how fast they are
#: produced. Large enough that the per-call overhead disappears, small enough
#: that a batch of long chunks cannot exhaust memory on a busy device.
EMBEDDING_BATCH_SIZE = 32

#: The files the model directory must contain before it is used.
#:
#: Checked up front so a missing or half-copied model fails with a sentence
#: naming the file, rather than a stack trace from deep inside a library that
#: was handed a directory it could not make sense of.
EMBEDDING_MODEL_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")

# --- Tokenisation -----------------------------------------------------------

#: The shortest term retrieval will index or match.
#:
#: Length one is dropped because a single character is never a value in this
#: corpus and is almost never a word: ``tq-40`` yields the parts ``tq`` and
#: ``40``, and ``ORR-FAN-2214-B`` yields a trailing ``b``. Keeping those would
#: put ``b`` in the index of every part number and let it match any question
#: containing the letter. Two is the shortest thing that can still be a real
#: term -- ``94``, ``r2``, ``E7731``.
MIN_TOKEN_LENGTH = 2

# --- Keyword retrieval (BM25) -----------------------------------------------

#: BM25 term-frequency saturation. A term occurring ten times in a chunk is
#: worth more than one occurring once, but not ten times more -- without this,
#: a long chunk that repeats a word would dominate every ranking.
BM25_K1 = 1.2

#: BM25 length normalisation, from 0 (ignore length) to 1 (fully normalise).
#:
#: Set below 1 because chunks here are not uniformly sized, and the short ones
#: are often the informative ones: a spreadsheet row or a line of source code
#: states a value outright, while a page of datasheet prose mentions it once
#: among hundreds of words. Full normalisation would over-reward the long
#: chunks; ignoring length would let them win on sheer term count.
BM25_B = 0.75

#: How many chunks retrieval hands to the answering stage.
#:
#: The answering stage needs the best few chunks and, for a multi-hop question,
#: the material with which to start a second search. A larger list costs nothing
#: at query time but invites the answering stage to cite something weak, and
#: extra citations score the same as missing ones.
RETRIEVAL_LIMIT = 10

#: How much a match in a chunk's **file path** counts relative to one in its text.
#:
#: The path is searched as a field of its own rather than pasted onto the text,
#: so that it gets its own length statistics. Concatenating would inflate every
#: chunk by the length of its path, and paths vary a lot in length -- a chunk
#: from ``vendor/x.dat`` would be artificially short and one from a deep
#: directory artificially long, which is a length bias with nothing to do with
#: the content.
#:
#: It exists because questions name documents by *kind*: "the production log",
#: "the datasheet", "the roadmap". Those words are often nowhere in the chunk --
#: a log file's lines say ``ERROR E7731: thermal throttle engaged`` and never
#: the word "log" -- so without the path the only route to the file is the
#: vocabulary it happens to share with the question.
#:
#: Below 1.0 because the filename is weaker evidence than the contents: a chunk
#: that merely sits in a file called ``logs/...`` should not outrank one that
#: actually discusses the thing asked about.
PATH_FIELD_WEIGHT = 0.5

# --- Vector retrieval -------------------------------------------------------

#: The lowest cosine similarity a chunk may have and still be returned.
#:
#: Cosine measures the angle between two vectors, so zero means "at right
#: angles" -- no relationship at all -- and a negative value means *less*
#: related than a vector chosen at random. Returning those would fill the result
#: list with noise, and noise in the list is an invitation for the answering
#: stage to cite something that does not support its answer.
#:
#: Zero is chosen rather than a fitted value because it is the only threshold
#: that means something on its own: everything above it is at least weakly
#: related, everything below it is not. A tuned cutoff would need measurements
#: from a real embedding model, which does not exist yet -- and inventing one
#: from the hashing stand-in would be fitting a number to a model that is not
#: going to be used.
MIN_VECTOR_SIMILARITY = 0.0

# --- Hybrid retrieval (fusing keyword and vector) ---------------------------

#: Reciprocal-rank-fusion smoothing, from Cormack, Clarke and Buettcher (2009).
#:
#: Fusion combines two rankings that are scored on scales with no common unit:
#: a BM25 score is an unbounded sum of term weights and the best chunk on this
#: corpus scores around 12.5, while a cosine similarity is bounded by 1 and the
#: best chunk scores around 0.85. Adding those two numbers is adding metres to
#: kilograms, and the result is a ranking dominated by whichever scale happens
#: to be larger -- here, always BM25.
#:
#: RRF sidesteps the units entirely by throwing the scores away and keeping only
#: the *positions*. A chunk's fused score is the sum, over the retrievers that
#: returned it, of ``1 / (RRF_K + rank)``. Ranks are unitless and comparable, so
#: nothing needs normalising and there is no scale to get wrong.
#:
#: ``RRF_K`` sets how much the top of each list is favoured. At the conventional
#: 60 the term barely varies down a ten-item list: rank 1 contributes 1/61 and
#: rank 10 contributes 1/70, a difference of 13%. What dominates instead is
#: *how many* retrievers returned a chunk, so agreement between the two
#: becomes the signal. That is the behaviour wanted here -- a chunk both halves
#: of the system found is far better evidence than one that only the keyword
#: half liked -- and 60 is kept because it is the published default rather than
#: a number fitted to the sample questions, which the graded corpus would not
#: honour. Whether a smaller value ranks better was measured on the sample
#: corpus rather than assumed, and the answer is that it barely matters: the
#: top result is identical at every k from 10 to 1000, and only k=0 and k=1
#: change it (on two questions). Recall is 7 of 9 at every k tested. What does
#: move is the *tail* of the list -- some questions have three different
#: top-ten sets across the six values tried -- so the constant is not inert, it
#: simply has a wide plateau, and 60 sits comfortably inside it. That is a
#: reason to keep the published value rather than fit one; see the Stage 5 notes
#: in ``IMPLEMENTATION_PROGRESS.md``.
RRF_K = 60

#: How many candidates each component retriever is asked for, per result wanted.
#:
#: Fusion can only choose between chunks it has been shown. Asking each
#: retriever for exactly ``RETRIEVAL_LIMIT`` items means the fused list can
#: never contain a chunk that only one retriever placed 11th, which is the
#: whole point of running two. Three is a compromise: deep enough that a chunk
#: ranked just outside one list still gets its second vote, shallow enough that
#: the union stays small.
FUSION_CANDIDATE_DEPTH = 3


# --- Answer extraction ------------------------------------------------------

#: Words carrying no discriminating power, removed before any overlap is measured.
#:
#: Retrieval never needs this list -- BM25's inverse document frequency already
#: discounts a term that appears everywhere, so ``the`` costs nothing there. The
#: extractor is doing a different job: it compares one candidate *against one
#: question* and counts how many of the question's words the candidate's
#: surroundings account for. With no corpus statistics behind that count, a
#: question's ``the`` and ``of`` would each contribute exactly as much as its
#: ``junction``, and every candidate would score the same on them.
#:
#: Deliberately short. A long list starts removing words that matter in this
#: corpus -- ``in`` is stoplisted but ``ingest`` is not, and ``for`` is
#: stoplisted while ``firmware`` is not -- so the list holds only the words that
#: are function words in any English question.
STOPWORDS = frozenset(
    """
    a an and any are as at be been but by can could did do does for from had has
    have how if in into is it its me much no not of on or please should so tell
    than that the their them then there these they this those to us was were
    what when where which who whom whose why will with would you your
    """.split()
)

#: A file whose *name* says it has been replaced. Matched on the normalised path.
#:
#: Kept separate from the content markers below because the two are not equally
#: trustworthy. A filename that says WITHDRAWN was named deliberately by whoever
#: retired the document; a sentence saying "superseded" may be describing some
#: *other* document, and this corpus contains exactly that trap -- the revision
#: history of ``tq40_datasheet_r2.pdf`` says "Revision 1 is withdrawn", which is
#: a statement *about* r1 sitting inside r2. Reading that as evidence that r2 is
#: withdrawn would invert the answer.
SUPERSEDED_PATH_MARKERS = ("WITHDRAWN", "SUPERSEDED", "OBSOLETE", "DEPRECATED")

#: A phrase in a file's own text saying the file itself is no longer current.
#:
#: Only counted when the document is also *naming itself*, which is what the
#: first-person forms here encode: "this document is superseded", "do not design
#: to these numbers". See :mod:`answering.supersession` for how the self-
#: reference is actually checked -- a bare "superseded" substring is not enough,
#: and treating it as enough is the trap described above.
SUPERSEDED_TEXT_MARKERS = (
    "IS WITHDRAWN",
    "IS SUPERSEDED",
    "HAS BEEN SUPERSEDED",
    "SUPERSEDED BY",
    "IS OBSOLETE",
    "NO LONGER CURRENT",
    "DO NOT DESIGN TO",
    "DO NOT USE THIS",
)
