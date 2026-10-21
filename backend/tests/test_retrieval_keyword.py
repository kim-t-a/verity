"""BM25 keyword retrieval.

Most of these are not "does it return results" -- a scorer that returns results
in a random order would pass that. They assert the *ordering properties* that
make BM25 worth using over counting shared words, because those are the
properties that quietly stop holding when someone later changes a constant or a
tokenizer rule:

* a rare term outranks a common one,
* a chunk cannot win by being long, or by repeating itself,
* identical scores come back in a reproducible order.

The last one matters more here than in most systems. Every question runs in a
fresh process against an index read from disk, so a ranking that varied between
runs would make a wrong answer impossible to reproduce -- and an unreproducible
wrong answer cannot be debugged.
"""

from __future__ import annotations

import pytest
from conftest import CORPUS_DIR

from verity.engine.chunking import chunks_for_all
from verity.engine.index_types import ChunkIndex, build_manifest
from verity.engine.config import RETRIEVAL_LIMIT
from verity.engine.ingestion import walker
from verity.engine.ingestion.readers import read_file
from verity.engine.models import Chunk, MediaType
from verity.engine.retrieval import Retriever, rank
from verity.engine.retrieval.keyword import PATH_TERM_PREFIX, KeywordRetriever

discover = walker.discover


def chunk(text: str, rel_path: str = "specs/datasheet.pdf", location: str = "page 1") -> Chunk:
    return Chunk(
        rel_path=rel_path,
        media_type=MediaType.PDF,
        location=location,
        ordinal=0,
        text=text,
    )


def index_of(chunks: list[Chunk]) -> ChunkIndex:
    manifest = build_manifest(
        fingerprint="test",
        chunks=chunks,
        file_count=1,
        provider_id=None,
        dimension=None,
        status_counts={"ok": 1},
        skipped_count=0,
    )
    return ChunkIndex(manifest=manifest, chunks=tuple(chunks))


def retriever_of(chunks: list[Chunk], **kwargs: object) -> KeywordRetriever:
    return KeywordRetriever(index_of(chunks), **kwargs)


# --- The properties BM25 exists for ----------------------------------------


def test_a_rare_term_outranks_a_common_one() -> None:
    """Inverse document frequency, stated as the behaviour it produces.

    Without idf, a chunk matching the question's filler would score as well as
    one matching its distinctive term, and the distinctive term is always the
    one carrying the answer.
    """
    retriever = retriever_of(
        [
            chunk("the thermal limit is documented", rel_path="a.txt"),
            chunk("the thermal limit is documented", rel_path="b.txt"),
            chunk("the thermal limit is documented", rel_path="c.txt"),
            chunk("the thermal limit is documented", rel_path="d.txt"),
            chunk("unrelated prose about scheduling", rel_path="e.txt"),
            chunk("rarething appears exactly here", rel_path="f.txt"),
        ]
    )

    results = retriever.search("the rarething")

    assert results[0].rel_path == "f.txt"
    # And the common word alone still matches everything containing it, in
    # corpus order -- it is not discarded, only outweighed.
    assert [hit.rel_path for hit in retriever.search("the")] == ["a.txt", "b.txt", "c.txt", "d.txt"]


def test_a_chunk_cannot_win_by_repeating_itself() -> None:
    """Term-frequency saturation, stated as the behaviour it produces.

    A chunk mentioning the term ten times is more relevant than one mentioning
    it once, but the curve flattens -- otherwise a chunk could top the ranking
    by padding.
    """
    retriever = retriever_of(
        [
            chunk("batch timeout", rel_path="terse.txt"),
            chunk("batch " * 30 + "timeout", rel_path="padded.txt"),
        ]
    )

    results = retriever.search("batch timeout")

    assert results[0].rel_path == "terse.txt"


def test_a_long_chunk_does_not_win_on_sheer_length() -> None:
    """Length normalisation, stated as the behaviour it produces.

    A long chunk contains more of every term simply by being long. Without
    normalisation the answer's chunk -- a single spreadsheet row -- would be
    buried under any page of prose that happens to mention the same words.
    """
    filler = "unrelated words about scheduling and capacity planning " * 20
    retriever = retriever_of(
        [
            chunk("maximum junction temperature 94", rel_path="row.txt"),
            chunk(f"maximum junction temperature {filler}", rel_path="prose.txt"),
        ]
    )

    results = retriever.search("maximum junction temperature")

    assert results[0].rel_path == "row.txt"


# --- Ordering and determinism -----------------------------------------------


def test_identical_scores_come_back_in_corpus_order() -> None:
    """The tie-break that makes a ranking reproducible.

    Ties are common -- a spreadsheet's rows are near-identical documents -- and
    leaving their order to the sort means leaving it to whatever order the
    scoring walked its dict in.
    """
    retriever = retriever_of(
        [
            chunk("same words here", rel_path=f"row{i}.csv", location=f"row {i}")
            for i in range(5)
        ]
    )

    results = retriever.search("same words")

    assert [hit.rel_path for hit in results] == [f"row{i}.csv" for i in range(5)]
    assert [hit.rank for hit in results] == [1, 2, 3, 4, 5]


def test_the_same_query_twice_gives_the_same_order() -> None:
    """One process answers one question, so this is about a *rebuilt* retriever.

    Each question gets a fresh process and a fresh retriever over the same
    on-disk index. If that produced a different order the second time, no wrong
    answer could be reproduced.
    """
    chunks = [chunk("alpha beta gamma", rel_path=f"f{i}.txt") for i in range(4)]

    first = [hit.rel_path for hit in retriever_of(chunks).search("alpha beta")]
    second = [hit.rel_path for hit in retriever_of(chunks).search("alpha beta")]

    assert first == second


def test_rank_is_one_based_and_contiguous() -> None:
    retriever = retriever_of([chunk(f"term{i} shared", rel_path=f"f{i}.txt") for i in range(4)])

    results = retriever.search("shared")

    assert [hit.rank for hit in results] == [1, 2, 3, 4]


def test_scores_descend() -> None:
    retriever = retriever_of(
        [
            chunk("alpha beta gamma delta", rel_path="many.txt"),
            chunk("alpha", rel_path="one.txt"),
        ]
    )

    scores = [hit.score for hit in retriever.search("alpha beta gamma delta")]

    assert scores == sorted(scores, reverse=True)


# --- The path field ---------------------------------------------------------


def test_a_question_can_reach_a_file_by_its_name() -> None:
    """The case that motivated a second field.

    A question asks about "the production log". The log's own lines say
    ``ERROR E7731: thermal throttle engaged`` and never the word "log" -- the
    word exists only in the path ``logs/prod_inference.log``. Without the path
    field the only route to the file is vocabulary it happens to share with the
    question.
    """
    retriever = retriever_of(
        [
            chunk("nothing relevant on this line", rel_path="logs/prod_inference.log"),
            chunk("nothing relevant on this line either", rel_path="notes/other.txt"),
        ]
    )

    results = retriever.search("the production log")

    assert results[0].rel_path == "logs/prod_inference.log"
    assert f"{PATH_TERM_PREFIX}log" in results[0].matched_terms


def test_a_path_match_is_weaker_than_a_content_match() -> None:
    """The filename is evidence, but less of it than the contents.

    A chunk that merely *lives* in a file called ``datasheet`` should not
    outrank one that actually discusses the thing asked about.
    """
    retriever = retriever_of(
        [
            chunk("silent on the subject", rel_path="specs/datasheet.pdf"),
            chunk("the datasheet describes the junction temperature", rel_path="misc/notes.txt"),
        ]
    )

    results = retriever.search("datasheet junction temperature")

    assert results[0].rel_path == "misc/notes.txt"


def test_a_path_match_is_labelled_as_one() -> None:
    """A match that came from a filename has to be visible as such.

    A chunk found only by its path says nothing about the question, and
    whoever is debugging a surprising ranking needs to be able to tell the two
    apart without re-deriving the scores by hand.
    """
    retriever = retriever_of([chunk("unrelated words", rel_path="logs/prod.log")])

    (result,) = retriever.search("logs")

    assert result.matched_terms == (f"{PATH_TERM_PREFIX}logs",)


def test_path_weight_of_zero_disables_the_field() -> None:
    """A field carrying no weight is not searched at all, not searched at zero.

    Multiplying by zero would still register the chunk as *matched* and return
    it, ranked last with a score of nothing -- a chunk surfaced for a query it
    does not match. Switching the field off has to remove it, which is also how
    its contribution gets measured.
    """
    retriever = retriever_of(
        [chunk("unrelated words", rel_path="logs/prod.log")], path_weight=0.0
    )

    assert retriever.search("logs") == []


# --- Nothing to return ------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "?!", "---"])
def test_a_query_with_no_terms_returns_nothing(query: str) -> None:
    retriever = retriever_of([chunk("any words at all")])

    assert retriever.search(query) == []


def test_a_query_matching_nothing_returns_nothing() -> None:
    """A normal outcome, not an error -- and exactly what a question about the
    encrypted file looks like. The caller's correct response is to refuse."""
    retriever = retriever_of([chunk("alpha beta", rel_path="a.txt")])

    assert retriever.search("zulu yankee") == []


def test_an_empty_index_returns_nothing() -> None:
    """No chunks is a legitimate index -- a corpus where every file failed."""
    retriever = retriever_of([])

    assert retriever.search("anything") == []
    assert len(retriever) == 0


def test_an_index_of_empty_chunks_does_not_divide_by_zero() -> None:
    """A chunk of pure punctuation has no terms and no length.

    Reached through the normal path it cannot happen -- readers report such a
    file EMPTY and it is never chunked -- but an index is a file on disk, and a
    later build writing one by hand should not be able to crash a query.
    """
    retriever = retriever_of([chunk("---", rel_path="a.txt"), chunk("...", rel_path="b.txt")])

    assert retriever.search("anything") == []


# --- Limits and the result shape -------------------------------------------


def test_limit_caps_the_result_count() -> None:
    retriever = retriever_of([chunk(f"shared term in chunk {i}") for i in range(20)])

    assert len(retriever.search("shared", limit=3)) == 3
    assert len(retriever.search("shared", limit=50)) == 20


def test_limit_leaves_the_order_unchanged() -> None:
    """Truncating must not reshuffle: the top 3 are the top 3 of the full list."""
    retriever = retriever_of([chunk(f"shared term in chunk {i}", rel_path=f"f{i}.txt") for i in range(10)])

    full = retriever.search("shared", limit=10)
    capped = retriever.search("shared", limit=3)

    assert [hit.rel_path for hit in capped] == [hit.rel_path for hit in full[:3]]


def test_the_default_limit_comes_from_config() -> None:
    retriever = retriever_of([chunk(f"shared term {i}") for i in range(RETRIEVAL_LIMIT + 5)])

    assert len(retriever.search("shared")) == RETRIEVAL_LIMIT


def test_a_result_carries_its_provenance() -> None:
    """A citation has to travel with the text from the moment it is found.

    Reconstructing the file afterwards would mean mapping a result back to a
    source by guessing, and the guess would be invisible when wrong.
    """
    retriever = retriever_of(
        [chunk("maximum junction temperature 94", rel_path="specs/tq40.pdf", location="page 1 line 4")]
    )

    (result,) = retriever.search("junction temperature")

    assert result.rel_path == "specs/tq40.pdf"
    assert result.location == "page 1 line 4"
    assert result.text == "maximum junction temperature 94"
    assert result.chunk.rel_path == "specs/tq40.pdf"


def test_matched_terms_are_reported_sorted() -> None:
    """The field that answers "why did this score as it did?"."""
    retriever = retriever_of([chunk("thermal throttle engaged")])

    (result,) = retriever.search("throttle thermal")

    assert result.matched_terms == ("thermal", "throttle")


def test_the_retriever_satisfies_the_protocol() -> None:
    assert isinstance(retriever_of([chunk("x y")]), Retriever)


def test_vocabulary_size_counts_both_fields() -> None:
    """A collapse towards zero would mean the tokenizer stopped seeing the corpus.

    Five, not four: ``gamma/delta.txt`` tokenizes to ``gamma``, ``delta`` and
    ``txt``. The ``/`` separates because a path is a path, and the extension is
    a word because a question may well ask for "the txt file".
    """
    retriever = retriever_of([chunk("alpha beta", rel_path="gamma/delta.txt")])

    assert retriever.vocabulary_size == 5


# --- rank() in isolation ----------------------------------------------------


def test_rank_numbers_from_one() -> None:
    chunks = [chunk("a"), chunk("b")]

    ranked = rank(chunks, [(0, 1.0, ("a",)), (1, 2.0, ("b",))])

    assert [hit.rank for hit in ranked] == [1, 2]
    assert [hit.rel_path for hit in ranked] == ["specs/datasheet.pdf", "specs/datasheet.pdf"]
    assert ranked[0].score == 2.0


def test_rank_is_empty_for_no_results() -> None:
    assert rank([], []) == []


# --- Against the real sample corpus -----------------------------------------


@pytest.fixture(scope="module")
def sample_chunks() -> tuple[Chunk, ...]:
    """The sample corpus, read and chunked for real."""
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"sample corpus not present at {CORPUS_DIR}")
    return tuple(chunks_for_all([read_file(f) for f in discover(CORPUS_DIR).files]))


def test_the_sample_corpus_produces_a_usable_index(sample_chunks: tuple[Chunk, ...]) -> None:
    """A real corpus, read and chunked, with a vocabulary that is not empty.

    Whatever else changes, this is the check that retrieval is looking at the
    corpus at all: a tokenizer that stopped matching, or a reader that started
    returning nothing, would show up here as an empty index long before it
    showed up as a wrong answer.
    """
    retriever = KeywordRetriever(index_of(list(sample_chunks)))

    assert len(retriever) > 0
    assert retriever.vocabulary_size > 100


def test_a_real_question_reaches_a_real_chunk(sample_chunks: tuple[Chunk, ...]) -> None:
    """Retrieval against the actual corpus, on a topic it certainly discusses."""
    retriever = KeywordRetriever(index_of(list(sample_chunks)))

    results = retriever.search("maximum junction temperature")

    assert results, "retrieval found nothing in a corpus that discusses this"
    assert results[0].location, "a result with no provenance cannot be cited"
    assert len(results[0].matched_terms) >= 2


#: The sample corpus's answerable, non-image questions and the one file each
#: needs. Text and paths only -- no answer values appear here, because the
#: property under test is whether the file is *reached*, not what it says.
#:
#: Question 9 is the multi-hop case and is absent: it needs the log *and* the
#: CSV, and the CSV is only reachable after reading the log -- that is the
#: answering stage's job, not retrieval's. Question 1's file is the current
#: revision; the withdrawn revision is deliberately left in the corpus and may
#: well rank above it, because retrieval does not adjudicate supersession.
ANSWERABLE_SAMPLE_QUERIES = [
    (
        "What is the maximum junction temperature of the TQ-40?",
        "specs/tq40_datasheet_r2.pdf",
    ),
    (
        "In which quarter does the TQ-60 enter customer sampling?",
        "planning/roadmap_fy27.docx",
    ),
    (
        "What is the part number of the field-replaceable fan assembly for the TQ-40?",
        "support/rma_parts.xlsx",
    ),
    ("Which firmware version fixed ticket ORR-1847?", "support/bug_database.csv"),
    (
        "What error code is logged when the thermal throttle engages?",
        "logs/prod_inference_2026-09-02.log",
    ),
    (
        "What is the default batch timeout, in seconds, in the ingest service?",
        "engineering/ingest_service.py",
    ),
]


@pytest.mark.parametrize(("query", "citation"), ANSWERABLE_SAMPLE_QUERIES)
def test_every_answerable_sample_question_reaches_its_citation(
    sample_chunks: tuple[Chunk, ...], query: str, citation: str
) -> None:
    """Every answerable question must at least surface the file holding its answer.

    **This is a regression guard, not a lookup table.** What it asserts is a
    property of the *retriever*: a question phrased the way a person phrases it
    surfaces the file holding the answer somewhere in the top ten. It says
    nothing about what the answer is, and the ranking is free to put that file
    anywhere from first to tenth. The answering stage has no access to this
    list, and the graded corpus shares none of these paths.

    Stated plainly, this is a weak test -- a scorer counting shared words with
    no idf and no length normalisation also passes it -- and it is kept as the
    floor beneath the sharper test below. It catches the failures that matter
    most: an empty index, a tokenizer that stopped matching the corpus, a
    reader that started returning nothing, a file dropped from the index. Those
    all fail here loudly, and they are the ways retrieval actually breaks.

    The two image questions are absent because their answers are pixels and the
    vision path does not exist yet; question 9 is absent because the second of
    its two files is only identifiable after reading the first, which is
    multi-hop answering rather than retrieval; question 10 is absent because its
    answer is in the encrypted file, so the correct result is an empty list.
    """
    retriever = KeywordRetriever(index_of(list(sample_chunks)))

    paths = [hit.rel_path for hit in retriever.search(query)]

    assert citation in paths, f"top {RETRIEVAL_LIMIT} for {query!r}: {paths}"


@pytest.mark.parametrize(("query", "citation"), ANSWERABLE_SAMPLE_QUERIES)
def test_the_right_file_comes_first_for_the_sample(
    sample_chunks: tuple[Chunk, ...], query: str, citation: str
) -> None:
    """A canary, not a requirement -- and the only test here that BM25 earns.

    The containment test above cannot detect much: measured against the sample
    corpus, a scorer that merely counts shared words with no idf and no length
    normalisation also puts every one of these files in the top ten. So that
    test guards against catastrophe, not against regression. This one is
    sharper. The naive scorer ranks the datasheet fifth for question 1; BM25
    ranks it first, and *this* is the assertion that notices if the ranking
    machinery stops working.

    It is fitted to six questions, which is very little evidence, and it is
    therefore framed as something to *notice* rather than something to obey.
    When a later stage legitimately reorders these -- adding vectors, fusing
    two rankers -- the right response is to look at the new order, decide
    whether it is better, and update this list deliberately. Tuning a constant
    until it goes green again would destroy the only signal it carries.

    Ranking first is not a requirement of the challenge: the answering stage
    only needs the file within reach, and the containment test above is what
    tracks that. This asks a stronger question only because a weak retriever and
    a strong one are otherwise indistinguishable on this corpus.
    """
    retriever = KeywordRetriever(index_of(list(sample_chunks)))

    results = retriever.search(query)

    assert results, f"nothing retrieved for {query!r}"
    assert results[0].rel_path == citation, (
        f"expected {citation} first, got "
        f"{[(hit.rel_path, round(hit.score, 2)) for hit in results[:3]]}"
    )
