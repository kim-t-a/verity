#!/usr/bin/env python3
"""Download the embedding model into the Docker build context.

Run this once, on a machine that has network access. The container has none at
evaluation time, so the weights have to be *inside* the image -- which means
inside ``starter/``, the Docker build context.

Why not ``huggingface_hub``: it pulls ``requests``, ``tqdm``, ``filelock`` and
``fsspec`` to do six GET requests. Those packages would also have to exist for
Python 3.14 in the image, which is one more unverified wheel than necessary.
This module uses the standard library and nothing else.

Why not ``pytorch_model.bin``: the repository publishes both a ``.bin`` and a
``.safetensors`` copy of the same weights. ``.bin`` is a Python pickle, and
loading a pickle executes whatever code it names; ``.safetensors`` is a header
of JSON plus a flat buffer of numbers, with no code path at all. The two files
hold the same weights, so the choice is free -- and it removes a class of risk
for good rather than trusting the file to stay benign.

Usage:

    python3 scripts/fetch_model.py --destination starter/models/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Repository to fetch from, and the revision to pin.
#:
#: Pinned to a commit rather than to ``main`` because ``main`` is a moving
#: target: a re-upload upstream would silently change the weights under a build
#: that is supposed to be reproducible. The commit hash is the identity.
MODEL_REPO = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

#: The files to fetch, and why each one is needed.
#:
#: Deliberately an explicit list rather than "everything in the repository".
#: A downloader that mirrors whatever is upstream cannot tell a needed file from
#: a test fixture or an extra format, and ``pytorch_model.bin`` above is exactly
#: the kind of file that gets swept in by a wildcard.
WANTED_FILES = (
    "config.json",                      # model shape: layers, hidden size, activation
    "model.safetensors",                # the weights, ~130 MB
    "tokenizer.json",                   # the fast tokenizer's vocabulary and merge rules
    "tokenizer_config.json",            # tokenizer class and special-token wiring
    "special_tokens_map.json",          # which token plays which role ([CLS], [SEP], ...)
    "vocab.txt",                        # the plain vocabulary, for the slow tokenizer
    "modules.json",                     # sentence-transformers pipeline description
    "sentence_bert_config.json",        # max sequence length
    "config_sentence_transformers.json",  # pooling/similarity defaults recorded by the author
    "1_Pooling/config.json",            # the pooling mode that turns token vectors into one
)

RESOLVE_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/"

#: Enough to notice a stalled connection on a 130 MB transfer without failing a
#: merely slow one.
TIMEOUT_SECONDS = 60

#: Files below this size are configuration and are read back and parsed; the
#: weights are only structurally checked. Kept as a named constant because the
#: verification path branches on it.
WEIGHT_FILE_THRESHOLD_BYTES = 1024 * 1024


def fetch(url: str, destination: Path) -> int:
    """Download ``url`` to ``destination``, returning the byte count.

    Written to a sibling ``.part`` file and renamed on success, so that an
    interrupted download cannot leave a truncated file that looks complete.
    A half-downloaded ``model.safetensors`` would fail later, at load time in
    the container, far from the cause.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mc3-fetch-model"})
    written = 0

    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        with partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                written += len(block)

    partial.replace(destination)
    return written


def check_safetensors(path: Path) -> dict:
    """Parse a ``.safetensors`` header. Returns the tensor metadata.

    The format is simple enough to verify without torch, which matters: torch is
    not installed here, and a check that needs the very dependency it is
    validating proves nothing. Eight bytes of little-endian length, that many
    bytes of JSON, then raw tensors.

    This confirms the file is structurally a safetensors file and reports what
    is in it. It does not confirm the numbers are the right numbers -- only a
    forward pass can do that.
    """
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length: {header_length}")
        header = json.loads(handle.read(header_length).decode("utf-8"))

    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError("the safetensors header names no tensors")
    return tensors


def check_json(path: Path) -> dict:
    """Parse a JSON config file, so a truncated or HTML error page is caught here."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify(destination: Path) -> None:
    """Verify what was downloaded, and raise if anything is wrong.

    Every file is checked, not just the big one. A configuration file that
    silently 404'd into an HTML error page is the more dangerous failure: the
    weights load fine, the tokenizer falls back to a default, and the model
    quietly produces embeddings that are subtly wrong.
    """
    tensors = check_safetensors(destination / "model.safetensors")
    hidden = tensors.get("embeddings.word_embeddings.weight")
    if hidden is None:
        raise ValueError("the weights hold no word embedding table")
    rows, columns = hidden["shape"]

    config = check_json(destination / "config.json")
    declared = config.get("hidden_size")
    if declared is not None and declared != columns:
        raise ValueError(
            f"config.json declares hidden_size={declared} but the weights have "
            f"{columns} columns per embedding"
        )
    vocab_from_config = config.get("vocab_size")
    if vocab_from_config is not None and vocab_from_config != rows:
        raise ValueError(
            f"config.json declares vocab_size={vocab_from_config} but the weights "
            f"have {rows} embedding rows"
        )

    for name in WANTED_FILES:
        if name.endswith(".json") and name != "model.safetensors":
            check_json(destination / name)

    print(f"  hidden size: {columns}")
    print(f"  vocabulary:  {rows} tokens")
    print(f"  tensors:     {len(tensors)}")
    print(f"  max tokens:  {check_json(destination / 'sentence_bert_config.json').get('max_seq_length')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="directory to fetch into; created if absent",
    )
    parser.add_argument(
        "--repo",
        default=MODEL_REPO,
        help=f"model repository (default: {MODEL_REPO})",
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="commit hash or branch to pin to (default: the pinned commit)",
    )
    args = parser.parse_args(argv)

    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    destination: Path = args.destination

    if (destination / "model.safetensors").exists():
        print(f"{destination} already holds a model; verifying it instead of refetching.")
        verify(destination)
        return 0

    destination.mkdir(parents=True, exist_ok=True)

    for name in WANTED_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {name} ...", flush=True)
        try:
            size = fetch(base + name, target)
        except urllib.error.HTTPError as exc:
            print(f"  failed: HTTP {exc.code} for {name}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"  failed: {exc.reason} for {name}", file=sys.stderr)
            return 1
        print(f"  {size:,} bytes")

    print("verifying ...")
    try:
        verify(destination)
    except (ValueError, KeyError, struct.error, json.JSONDecodeError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nmodel ready at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
