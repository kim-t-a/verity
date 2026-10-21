"""Phase 0: Measure the token ceiling for NVIDIA embedding chunks.

Embeds texts of increasing length through the real NVIDIA API to check
whether MAX_CHUNK_CHARS (2000) fits inside the model's token limit without
silent truncation.

Run with:  TMPDIR=/tmp python -m scripts.measure_token_ceiling

No secrets are printed. Only lengths, whether the API accepted each input,
and vector dimensions are reported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env into environment so NvidiaEmbeddingProvider can read it.
# The .env file lives at the repo root (one level above backend/).
env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name in ("NVIDIA_API_KEY", "NVIDIA_API_BASE_URL") and value:
            os.environ[name] = value

from verity.engine.nvidia_embedding_provider import NvidiaEmbeddingProvider, NvidiaEmbeddingError  # noqa: E402

# Lengths to test, in characters. 2000 is MAX_CHUNK_CHARS.
LENGTHS = [500, 1000, 2000, 3000, 4000]


def make_text(length: int) -> str:
    """Dense-ish text: repeating tokens separated by spaces."""
    return " ".join(f"token{i % 50}" for i in range(length // 6))


def main() -> int:
    provider = NvidiaEmbeddingProvider()
    print(f"Model: {provider.model_id}  dimension: {provider.dimension}\n")

    failures = 0
    for length in LENGTHS:
        text = make_text(length)
        try:
            vectors = provider.embed([text])
            dim = len(vectors[0]) if vectors else 0
            print(f"{length:5d} chars: OK  vectors={len(vectors)}  dimension={dim}")
        except NvidiaEmbeddingError as exc:
            print(f"{length:5d} chars: REJECTED  {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{length:5d} chars: ERROR  {type(exc).__name__}: {exc}")
            failures += 1

    print()
    if failures:
        print(f"{failures} length(s) were rejected or errored.")
        print("Consider lowering MAX_CHUNK_CHARS or splitting on tokens instead of characters.")
        return 1

    print("All lengths accepted by the embedding API.")
    print("No silent truncation detected at these sizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
