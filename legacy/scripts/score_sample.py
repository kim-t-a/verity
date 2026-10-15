#!/usr/bin/env python3
"""Run the pipeline over the sample corpus and score it against the sample set.

    python3 scripts/score_sample.py
    python3 scripts/score_sample.py --corpus mc3-corpus --device cpu

This is developer tooling. It lives in ``scripts/``, which the Dockerfile does
not copy, so nothing here can run during evaluation -- and that separation is
deliberate rather than incidental. ``AGENTS.md`` rule 1 forbids hardcoding a
sample answer, a sample question id, or an expected citation set, and the risk
that rule guards against is precisely this file: a scorer is the one piece of
code with the answers in front of it, and the temptation is to let it leak into
the thing being scored.

It hardcodes none of them. Every expected value -- the queries, the answers, the
aliases, the citation sets, which questions are refusals -- is read from
``sample-questions.json`` at run time. Point ``--questions`` at a different file
and the scorer scores that instead, with no edit here.

**Answer comparison goes through the application's own normaliser.** Not a copy:
``starter/app/normalisation.py`` is imported directly, because a second
implementation that drifted would score answers differently from the pipeline
that produced them. The contract's comparison form (uppercase, no separators, no
units) is what makes ``REV-C2`` and ``rev c2`` the same answer, and matching must
happen on exactly the form the application already uses internally.

**An answer is compared exactly, not loosely.** It must equal the expected value
or one of its declared aliases once normalised -- a substring match would let
``194`` pass as ``94`` and hide a real failure. Where an answer is only a
*substring* match, the row is reported as close rather than passed, so the
information is visible without being silently credited.

**Citations are compared as exact sets.** Not as overlaps: the brief is explicit
that extra citations score the same as missing ones, zero, so a run that cites
the right file plus one more is a failure and is scored as one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: The repository root, derived from this file's location rather than from the
#: current directory, so the script behaves the same from anywhere.
_REPO = Path(__file__).resolve().parent.parent

#: The application's normaliser, imported rather than reimplemented. The path
#: insert is what makes it importable without packaging ``starter/app``.
sys.path.insert(0, str(_REPO / "starter" / "app"))

from normalisation import normalise  # noqa: E402 - path insert must run first

#: Wall-clock budgets from the contract, reported against so a regression is
#: visible here rather than in the graded run.
INDEX_BUDGET_SECONDS = 600.0
PER_QUERY_BUDGET_SECONDS = 30.0


def answer_matches(got: str, expected: str, aliases: list[str]) -> bool:
    """Whether ``got`` is the expected answer under the contract's comparison form.

    Args:
        got: The answer the pipeline produced.
        expected: The expected value, exactly as the sample file spells it.
        aliases: Other sanctioned spellings of the same value.

    Returns:
        ``True`` when the normalised answer equals the normalised expected value
        or any normalised alias. An expected empty string matches only an empty
        answer, which is how a refusal is scored.
    """
    candidates = {normalise(expected)} | {normalise(alias) for alias in aliases}
    return normalise(got) in candidates


def is_close(got: str, expected: str, aliases: list[str]) -> bool:
    """Whether ``got`` contains or is contained by an accepted form.

    Only used to annotate a row that already failed. Never turns a failure into
    a pass: a value that is *part* of the right answer is not the right answer,
    and the contract grades the exact value.
    """
    forms = [normalise(expected), *(normalise(alias) for alias in aliases)]
    normalised_got = normalise(got)
    if not normalised_got:
        return False
    return any(
        form and (form in normalised_got or normalised_got in form) for form in forms
    )


def run_app(app: Path, args: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess:
    """Invoke ``app.py`` as its own process, the way the harness does.

    A separate process per query is the contract, not a convenience: it is what
    makes model load time count against the 30 second per-question budget, so
    scoring must not quietly reuse an in-process model and report a time the
    harness will never see.

    Raises:
        subprocess.TimeoutExpired: The invocation exceeded ``timeout``.
    """
    return subprocess.run(
        [sys.executable, str(app), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(app.parent),
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=_REPO / "mc3-corpus")
    parser.add_argument("--questions", type=Path, default=_REPO / "sample-questions.json")
    parser.add_argument("--app", type=Path, default=_REPO / "starter" / "app" / "app.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mc3-score-output",
        help="where per-query JSON files are written and read back from",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mc3-score-index",
        help="where the index pass writes, standing in for /app/index",
    )
    parser.add_argument(
        "--device",
        help="sets MC3_EMBEDDING_DEVICE. Use 'cpu' on a development machine "
        "without a GPU; the application refuses to fall back on its own.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="sets MC3_EMBEDDING_MODEL_DIR, standing in for the image's /models/.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="score the existing index instead of rebuilding it",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="score JSON files already sitting in --output-dir without running "
        "anything. This is how container output is scored: the container writes "
        "the files, they are copied out, and the same comparison runs over them "
        "-- one implementation, so a local pass and a container pass mean the "
        "same thing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each query's stdout/stderr when it produces no usable output",
    )
    args = parser.parse_args()

    # Resolve every path before it reaches a child process. ``app.py`` is
    # invoked with its own directory as the working directory, so a path left
    # relative here is resolved against the wrong place -- and the failure is
    # quiet rather than loud: a ``--model-dir`` that misses simply produces a
    # lexical-only run that still scores, which reads as a pipeline result
    # instead of the typo it is.
    args.corpus = args.corpus.resolve()
    args.questions = args.questions.resolve()
    args.app = args.app.resolve()
    args.output_dir = args.output_dir.resolve()
    args.index_dir = args.index_dir.resolve()
    if args.model_dir:
        args.model_dir = args.model_dir.resolve()

    for label, path in (("corpus", args.corpus), ("questions", args.questions), ("app", args.app)):
        if not path.exists():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["queries"]

    env = dict(os.environ)
    env["MC3_INDEX_DIR"] = str(args.index_dir)
    env["MC2_OUTPUT_DIR"] = str(args.output_dir)
    if args.device:
        env["MC3_EMBEDDING_DEVICE"] = args.device
    if args.model_dir:
        env["MC3_EMBEDDING_MODEL_DIR"] = str(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_index and not args.reuse:
        print(f"indexing {args.corpus} ...", flush=True)
        started = time.time()
        try:
            index = run_app(
                args.app, ["--index", str(args.corpus)], env, INDEX_BUDGET_SECONDS
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL: --index exceeded {INDEX_BUDGET_SECONDS:.0f}s", file=sys.stderr)
            return 1
        elapsed = time.time() - started
        if index.returncode != 0:
            print(f"FAIL: --index exited {index.returncode}", file=sys.stderr)
            print((index.stderr or index.stdout).strip()[:2000], file=sys.stderr)
            return 1
        print(f"  indexed in {elapsed:.1f}s (budget {INDEX_BUDGET_SECONDS:.0f}s)")
        # The index pass reports on stderr, not stdout, and that report is the
        # only place a degradation is announced -- "no embeddings, indexing
        # lexical-only" and the per-file status counts both live there. It is
        # echoed rather than discarded, because a run that quietly dropped the
        # vector retriever produces plausible numbers that are not the numbers
        # the shipped pipeline produces.
        for stream in (index.stderr, index.stdout):
            for line in (stream or "").strip().splitlines():
                print(f"  {line}")

    print(f"\n{'':3}{'query':<10}{'time':>7}  {'answer':<20}{'expected':<20}result")

    passed = 0
    for question in questions:
        query_id = f"query_{question['n']:02d}"
        expected = question["expected_answer"]
        aliases = question.get("answer_aliases", [])
        want_citations = question["expected_citations"]

        elapsed: float | None = None
        if not args.reuse:
            started = time.time()
            try:
                proc = run_app(
                    args.app,
                    [
                        "--corpus",
                        str(args.corpus),
                        "--query-id",
                        query_id,
                        "--query",
                        question["query"],
                    ],
                    env,
                    PER_QUERY_BUDGET_SECONDS,
                )
                elapsed = time.time() - started
            except subprocess.TimeoutExpired:
                print(f"XX {query_id:<10}{PER_QUERY_BUDGET_SECONDS:>6.0f}s  TIMEOUT")
                continue

        # No duration is shown when reusing. Those files were written by another
        # process and this run has no idea how long it took; printing a zero or
        # a guess would read as a comfortably fast pass.
        time_column = "-" if elapsed is None else f"{elapsed:.1f}s"

        output_path = args.output_dir / f"{query_id}_output.json"
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"XX {query_id:<10}{time_column:>7}  NO USABLE OUTPUT: {exc}")
            if args.verbose:
                print(f"     rc={proc.returncode}")
                print(f"     stdout: {proc.stdout.strip()[:800]}")
                print(f"     stderr: {proc.stderr.strip()[:800]}")
            continue

        # Shape first: a missing key is malformed and scores zero regardless of
        # what the values would have been.
        answer = payload.get("answer")
        citations = payload.get("citations")
        if not isinstance(answer, str) or not isinstance(citations, list):
            print(f"XX {query_id:<10}{time_column:>7}  MALFORMED: answer/citations missing")
            continue

        answer_ok = answer_matches(answer, expected, aliases)
        citations_ok = sorted(citations) == sorted(want_citations)
        ok = answer_ok and citations_ok
        passed += ok

        shown = answer if len(answer) <= 18 else answer[:17] + "…"
        print(
            f"{'OK ' if ok else 'XX '}{query_id:<10}{time_column:>7}  "
            f"{shown!r:<20}{expected!r:<20}{'pass' if ok else 'FAIL'}"
        )
        if not answer_ok:
            note = " (close, but not equal)" if is_close(answer, expected, aliases) else ""
            print(f"      answer: got {answer!r} want {expected!r}{note}")
        if not citations_ok:
            print(f"      citations: got {sorted(citations)}")
            print(f"                 want {sorted(want_citations)}")
        if elapsed is not None and elapsed > PER_QUERY_BUDGET_SECONDS:
            print(f"      over the {PER_QUERY_BUDGET_SECONDS:.0f}s budget")

    total = len(questions)
    print(f"\n{passed}/{total} exact")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
