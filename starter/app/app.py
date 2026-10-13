#!/usr/bin/env python3
"""Reference submission skeleton for AMD Mini-Challenge 3 (RAG).

Unlike MC-2, the harness invokes this script in TWO different ways, inside your
already-running container.

1. ONCE, before any graded clock starts, to build whatever index you want:

       python3 /app/app.py --index /app/corpus

   This pass is ungraded and charged to your startup budget, not to a query.
   It is where you should pay for parsing PDFs, spreadsheets and images.

2. ONCE PER QUESTION, ten times:

       python3 /app/app.py --corpus /app/corpus \\
                           --query-id query_01 \\
                           --query "What is the maximum junction temperature?"

   and then it reads the file you wrote:

       /app/output/query_01_output.json

   The query id is GIVEN to you, not derived -- a question has no filename to
   take a stem from. Write to exactly that id plus _output.json.

The output is a JSON object with three keys:

    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

  * answer     -- the VALUE only. "94", not "The maximum junction temperature
                  is 94 C". Qualifiers that are part of the value stay:
                  "Q3 FY27", not "Q3". Units and degree signs are normalised
                  away for you, so 94, "94 C" and "94°C" are all accepted.
  * citations  -- corpus-relative paths, compared as an EXACT SET. The test is
                  necessity, not relevance: cite a file only if REMOVING it
                  would make your answer impossible. Dumping your whole
                  retrieval fails almost every question even when the answer is
                  right. Some questions need two files; cite both.
  * confidence -- recorded, never scored.

Missing "answer" or missing "citations" is malformed and scores zero, on
purpose: a crashed writer and a considered refusal must not look the same.

If the corpus does not contain the answer, return an empty answer AND an empty
citation list. "I don't know" is prose, and prose is graded as a wrong answer.
Guessing is wrong, and so is answering from what the model already knows --
the products in this corpus are fictional, so anything recalled rather than
retrieved is wrong by construction.

Three things that will cost you points no matter how good your model is:

  * Loading the model inside answer(). The harness starts a NEW PROCESS for
    every question. A naive implementation loads ten times and blows the
    budget. Persist your index in --index, and keep the model resident behind
    a unix socket, or make loading cheap enough not to matter.
  * Letting one bad file kill the walk. The corpus deliberately contains an
    unreadable file, an encrypted PDF, an unknown binary type and an empty
    directory. A walk that raises on the first unreadable file indexes nothing
    after it -- and because walk order follows the directory listing, WHICH
    files you lose depends on filename order. That is how a submission tests
    clean locally and grades badly here.
  * Running on the CPU. VRAM is sampled continuously and a run that never uses
    the GPU is rejected, so place your model on the device explicitly rather
    than letting it fall back.

Replace index() and answer() below. Leave the plumbing alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from answering import AnswerShape, Extraction, cited_paths, extract, refusal
from answering.supersession import superseded_paths
from chunking import chunks_for_all
from chunkstore import (
    ChunkIndex,
    IndexIncompatible,
    IndexUnavailable,
    build_manifest,
    corpus_fingerprint,
    load,
    save,
    verify_against,
)
from embeddings import EmbeddingError, provider_from_environment
from ingestion import walker
from ingestion.readers import read_file
from models import ReadStatus
from retrieval import RetrievedChunk, Retriever
from retrieval.hybrid import HybridRetriever, HybridUnavailable, describe_sources
from retrieval.keyword import KeywordRetriever
from retrieval.vector import VectorRetriever, VectorUnavailable

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def index(corpus: Path) -> None:
    """Build and PERSIST whatever you need to answer questions later.

    Called once, before any question, with the corpus root. Anything you write
    under INDEX_DIR survives into the per-question invocations; anything you
    keep only in memory does not, because each question is a new process.

    The corpus is a directory tree of mixed types -- pdf, docx, xlsx, csv, txt,
    log, py, png, jpg -- and EVERY type holds at least one graded answer, so
    skipping a format silently costs you the questions that depend on it. Two
    of the ten questions are answerable only by looking at an image.

    Walk defensively: catch per-file, keep going, and record what you could not
    read rather than aborting. See the third bullet in the module docstring.

    This is also the only place any of the expensive work happens. Everything
    paid for here -- walking, decoding, extracting, chunking, embedding -- is
    paid once, because each of the ten questions runs in a fresh process and
    would otherwise repeat all of it inside a 30-second budget.
    """
    started = time.monotonic()

    discovered = walker.discover(corpus)
    results = [read_file(discovered_file) for discovered_file in discovered.files]
    chunks = chunks_for_all(results)

    try:
        provider = provider_from_environment()
    except EmbeddingError as exc:
        # Degraded, not raised -- and this mirrors what answer() already does.
        #
        # A backend that cannot start is one component's fault, and letting it
        # abort the whole pass would leave no index at all, so all ten questions
        # would refuse and the submission would score nothing. Continuing
        # lexical-only produces a partial system instead of none.
        #
        # The failure is printed rather than swallowed, which is what keeps this
        # from being the silent-degradation shape this project keeps meeting: a
        # lexical-only index announces itself here *and* in the manifest, where
        # provider_id is null. It is never a silent switch to the hashing
        # stand-in -- that still requires both of its own environment switches.
        print(f"mc3: no embeddings, indexing lexical-only: {exc}", file=sys.stderr)
        provider = None

    vectors: list[list[float]] = []
    if provider is not None and chunks:
        vectors = provider.embed([chunk.text for chunk in chunks])

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[str(result.status)] = status_counts.get(str(result.status), 0) + 1

    manifest = build_manifest(
        fingerprint=corpus_fingerprint(discovered.files),
        chunks=chunks,
        file_count=discovered.total_files,
        provider_id=provider.model_id if provider is not None else None,
        dimension=provider.dimension if provider is not None else None,
        status_counts=status_counts,
        skipped_count=discovered.total_skipped,
    )
    save(INDEX_DIR, ChunkIndex(manifest=manifest, chunks=chunks, vectors=tuple(vectors)))

    unreadable = sorted(
        f"{result.status}: {result.rel_path}"
        for result in results
        if not result.is_usable and result.status != ReadStatus.EMPTY
    )
    _report_index(corpus, manifest, unreadable, time.monotonic() - started)


def _report_index(corpus, manifest, unreadable, elapsed: float) -> None:
    """Print what the index pass found.

    Written to stderr so it cannot be mistaken for the program's output, and
    printed at all because ``--index`` is the one pass whose behaviour is
    invisible in the graded result: if a format silently stops being read, the
    only symptom is later questions answering wrongly.
    """
    print(
        f"indexed {corpus}: {manifest.file_count} files, "
        f"{manifest.chunk_count} chunks, "
        f"embeddings={manifest.provider_id or 'none (lexical only)'}, "
        f"{elapsed:.1f}s",
        file=sys.stderr,
    )
    print(
        "  file status: "
        + ", ".join(f"{name}={count}" for name, count in manifest.status_counts.items()),
        file=sys.stderr,
    )
    if manifest.skipped_count:
        print(f"  skipped during walk: {manifest.skipped_count}", file=sys.stderr)
    for line in unreadable:
        print(f"  not indexed -> {line}", file=sys.stderr)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Return (answer, citations, confidence) for ONE question.

    THIS is what you replace. `citations` are paths relative to `corpus`, using
    forward slashes -- "specs/tq40_datasheet_r2.pdf", not an absolute path and
    not a bare filename.

    Return ("", [], 0.0) to refuse. Refusing correctly scores full marks on the
    question that has no answer in the corpus, and refusing on a question that
    does have one scores the same as a wrong guess -- so refusal is safe to use
    honestly and useless to use defensively.
    """
    try:
        verify_against(INDEX_DIR, corpus)
        index = load(INDEX_DIR)
    except (IndexUnavailable, IndexIncompatible) as exc:
        # Refusing rather than raising: the output contract requires an answer
        # and a citation list on *every* response, including errors, because a
        # writer that crashed and a considered refusal must not look alike.
        print(f"mc3: cannot use the index: {exc}", file=sys.stderr)
        return "", [], 0.0

    # One fused ranking is what retrieval now produces. Until Stage 5 the two
    # halves were run and reported side by side, because combining them was an
    # open question and picking a winner early would have been a claim this
    # project had not earned. The measurement that settled it is in Stage 5's
    # notes: the halves disagree on exactly one sample question, and the
    # disagreement is a dead heat under fusion, so the fused order is the honest
    # answer and the tie is left for Stage 6 to break by rule.
    try:
        retriever = _retriever_for(index)
    except HybridUnavailable as exc:
        print(f"mc3: no retriever available: {exc}", file=sys.stderr)
        return "", [], 0.0

    results = retriever.search(query)
    _report_retrieval(retriever, query, results)

    # Supersession is judged over the WHOLE corpus, not over this result list.
    # The verdict for one file can rest on text held in another -- r2 says what
    # it replaced -- and a set computed once is a set every question agrees on.
    superseded = superseded_paths(index.chunks)
    extraction = extract(results, query, superseded)
    _report_extraction(extraction, superseded)

    if extraction.winner is None:
        # No candidate value anywhere in the retrieved chunks. For a question
        # whose evidence is a file that could not be read, this is the correct
        # outcome rather than a failure, and refusing is what the contract asks
        # for -- an empty answer with an empty citation list.
        return "", [], 0.0

    winner = extraction.winner

    # Refusal comes before citations, because a question the readable corpus does
    # not answer is empty on *both* counts: no answer and no citations. Deciding
    # it here keeps the two consistent, and it is deliberately the narrowest rule
    # in the system -- see `answering.refusal` for the signals that were measured
    # and rejected before this one, including the one that scored backwards.
    reason = refusal(winner, query, index.chunks)
    if reason is not None:
        print(f"  refusing: {reason}", file=sys.stderr)
        return "", [], 0.0

    # Necessity, not provenance: which files the answer could not have been
    # produced without. That is the contract's question and it is a different
    # one from "which file was the value read out of" -- seven of the ten sample
    # questions have the same answer either way, and the eighth needs a second
    # file that the value never appears in.
    #
    # The whole corpus is passed, not `results`, and the distinction is measured
    # rather than stylistic: on the sample's two-hop question the chunk that
    # supplies the key is not among the ten chunks retrieval returns, so a rule
    # that searched only what was retrieved would name one file of the two that
    # are necessary -- and citation sets are scored exactly, so that costs the
    # same as naming none.
    citations = cited_paths(winner, query, index.chunks)
    _report_citations(citations, winner.rel_path)

    return extraction.answer, list(citations), extraction.confidence


def _report_citations(citations: tuple[str, ...], winner_path: str) -> None:
    """Print the citation set, and which of it retrieval had nothing to do with.

    A citation the winner's own file does not cover came from somewhere the
    ranking did not look, and that is the single most surprising thing this
    stage does -- so it is named rather than left to be inferred from a path
    that appears in the output with no visible origin.
    """
    extra = [path for path in citations if path != winner_path]
    print(f"  citations {list(citations)}", file=sys.stderr)
    if extra:
        print(f"  also necessary (not the winner's file): {extra}", file=sys.stderr)



def _report_extraction(extraction: Extraction, superseded: frozenset[str]) -> None:
    """Print what was extracted, and what it beat, to stderr.

    The winning candidate alone would be a poor diagnostic: knowing that
    ``94`` won says nothing about whether it won by a mile or by a hair, and the
    interesting failures are the near-misses. So the fields that decided it are
    printed rather than just the value -- which is how a wrong answer is traced
    to the specific signal that outranked the right one, instead of being
    re-derived by hand.

    The corpus-wide supersession verdict is printed too, because it is invisible
    in the graded JSON whenever it agrees with retrieval, and it is the signal
    that decides the question when it does not.
    """
    if extraction.winner is None:
        print(
            f"  no candidate value among {extraction.considered} candidates "
            f"(shapes wanted: {_shapes(extraction.wanted_shapes)})",
            file=sys.stderr,
        )
        return

    winner = extraction.winner
    print(
        f"  answer {winner.value!r} from {winner.rel_path} [{winner.location}] "
        f"shape={winner.shape.value} "
        f"superseded={winner.superseded} "
        f"after_equals={winner.follows_equals} "
        f"context={' '.join(winner.context_terms) or '-'} "
        f"of {extraction.considered} candidates "
        f"(shapes wanted: {_shapes(extraction.wanted_shapes)})",
        file=sys.stderr,
    )
    if superseded:
        # A count of paths, not of chunks -- the two are different quantities
        # and printing them side by side would invite reading one as the other.
        print(
            f"  superseded in the corpus: {', '.join(sorted(superseded))}",
            file=sys.stderr,
        )


def _shapes(shapes: tuple[AnswerShape, ...]) -> str:
    """The shape names the question was read as wanting, for the diagnostics."""
    return ", ".join(shape.value for shape in shapes) or "none"


def _retriever_for(index: ChunkIndex) -> HybridRetriever:
    """Every retriever this index and environment can support, fused into one.

    Keyword is always available. The vector half is added only if a backend is
    configured, it matches the one that built the index, and the index actually
    holds vectors -- and when it is missing that is *reported*, because a
    lexical-only hybrid answers every question perfectly plausibly while having
    lost the recall the second retriever existed to provide. Nothing in the
    graded JSON would show it.
    """
    components: list[tuple[str, Retriever]] = [("keyword", KeywordRetriever(index))]

    try:
        provider = provider_from_environment()
    except EmbeddingError as exc:
        print(f"  vectors unavailable: {exc}", file=sys.stderr)
        provider = None

    if provider is None:
        print(
            "  vectors unavailable: no embedding backend is configured, so this "
            "index is lexical-only",
            file=sys.stderr,
        )
    else:
        try:
            components.append(("vector", VectorRetriever(index, provider)))
        except VectorUnavailable as exc:
            print(f"  vectors unavailable: {exc}", file=sys.stderr)

    hybrid = HybridRetriever(components)
    print(f"  fusing: {'+'.join(hybrid.components)} (RRF k={hybrid.rrf_k})", file=sys.stderr)
    return hybrid


def _report_retrieval(
    retriever: Retriever, query: str, results: list[RetrievedChunk]
) -> None:
    """Print what the fused retriever found, to stderr.

    The one window into a stage whose output never reaches the graded JSON:
    ``answer`` is still a refusal, so without this a retrieval regression would
    be invisible until Stage 6 and would then look like an answering bug.

    Each line carries its origins -- which retrievers wanted this chunk and
    where each ranked it -- because a fused score is on a scale of its own and
    cannot be read without knowing whether it came from one strong vote or two
    weak ones.
    """
    print(f"{len(results)} chunks for {query!r}", file=sys.stderr)
    for result in results:
        terms = ", ".join(result.matched_terms) or "-"
        print(
            f"  {result.rank:>2}. {result.rel_path} [{result.location}] "
            f"score={result.score:.4f} via={describe_sources(result)} matched={terms}",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    # Written whole rather than streamed: a partially-written file that the
    # harness reads mid-flush parses as invalid JSON and scores the query zero.
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
