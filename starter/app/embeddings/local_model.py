"""The real embedding backend: a local BGE model on the GPU.

This is the module that satisfies two separate challenge requirements at once.
It puts a model on the device -- which is what keeps the run from being judged
CPU-only -- and it gives the retrieval half of the system a notion of *meaning*,
which keyword matching cannot supply: a question asking about "customer
sampling" should reach a document that says "customer samples ship in", and
BM25 alone will not get there.

Three decisions in here are worth reading before changing anything.

**Pooling.** A BERT produces one vector *per token*. A sentence embedding is one
vector per *input*, so the token vectors have to be collapsed. This model
declares CLS pooling -- take the vector at the ``[CLS]`` position -- and that is
not the default anyone reaches for. Mean pooling is the tutorial standard and
would produce a working retriever that ranks quietly worse, which is why
``config.EMBEDDING_POOLING`` records the choice and the provider id encodes it.

**No device fallback.** If the model cannot be placed on the requested device,
this raises. It does not quietly continue on the CPU. A CPU run is *rejected* by
the challenge, so a silent fallback converts a loud failure into an invisible
one -- and the invisible version wastes the entire evaluation budget before
anyone finds out.

**Import cost.** ``torch`` and ``transformers`` are imported inside
``__init__``, never at module level. The container starts a fresh process for
each of the ten questions, and the import-graph test imports every module in the
payload; paying several seconds to import a library that a given process may
never call is a cost with nothing to show for it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from config import (
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE_ENV,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_DIR_ENV,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REQUIRED_FILES,
    EMBEDDING_POOLING,
    MAX_SEQUENCE_TOKENS,
)
from embeddings import EmbeddingError


class LocalModelUnavailable(EmbeddingError):
    """The model, or the device to run it on, could not be used.

    Distinct from :class:`EmbeddingError` so a caller can tell "this backend
    cannot start here" from "this backend failed mid-run", which are different
    problems with different responses.
    """


def _pool_cls(hidden: Any, attention_mask: Any) -> Any:
    """Take the ``[CLS]`` position, which BERT pre-trains as an input summary.

    ``attention_mask`` is accepted and unused. That is deliberate rather than
    sloppy: every pooling function has the same signature so that choosing one
    is a single assignment, and the mask is what the mean variant needs, not
    something this one forgot.
    """
    return hidden[:, 0]


def _pool_mean(hidden: Any, attention_mask: Any) -> Any:
    """Average the token vectors, excluding padding.

    Present but *not* selected -- see the module docstring. It is kept because
    the comparison between the two is a measurement that had to be made, and a
    reader who wants to re-make it should not have to write it from scratch.
    """
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)


_POOLING_FUNCTIONS = {"cls": _pool_cls, "mean": _pool_mean}


class LocalModelEmbedder:
    """Embeds text with a local transformer, on a device that must be available.

    Satisfies :class:`embeddings.EmbeddingProvider`. Constructing it loads the
    weights, which takes under a second for this model and is the only
    meaningful cost the answering process pays for retrieval.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        device: str | None = None,
        pooling: str | None = None,
    ) -> None:
        """Load the model and place it on the device.

        Raises:
            LocalModelUnavailable: the directory is missing files, the pooling
                name is unknown, or the requested device is not usable. Every
                one of these is raised here, at construction, rather than
                surfacing later as a wrong number.
        """
        self._directory = Path(
            model_dir
            if model_dir is not None
            else os.environ.get(EMBEDDING_MODEL_DIR_ENV, DEFAULT_EMBEDDING_MODEL_DIR)
        )
        self._device_name = device or os.environ.get(
            EMBEDDING_DEVICE_ENV, DEFAULT_EMBEDDING_DEVICE
        )
        self._pooling_name = pooling or EMBEDDING_POOLING

        if self._pooling_name not in _POOLING_FUNCTIONS:
            raise LocalModelUnavailable(
                f"unknown pooling {self._pooling_name!r}; "
                f"expected one of {sorted(_POOLING_FUNCTIONS)}"
            )

        self._check_directory()

        # Imported here, not at module level -- see the module docstring.
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LocalModelUnavailable(
                f"the model libraries are not installed: {exc}. "
                "The image needs transformers; the base image already ships torch."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._directory)
            self._model = AutoModel.from_pretrained(self._directory)
        except (OSError, ValueError) as exc:
            raise LocalModelUnavailable(
                f"could not load a model from {self._directory}: {exc}"
            ) from exc

        self._model.eval()
        self._model.to(self._device)

        self._verify_width()

    # --- Construction helpers ----------------------------------------------

    def _check_directory(self) -> None:
        """Fail with the missing filename, before a library can fail obscurely."""
        if not self._directory.is_dir():
            raise LocalModelUnavailable(
                f"model directory {self._directory} does not exist. In the image this "
                "is /models/..., populated by the Dockerfile's COPY of starter/models/. "
                f"Set {EMBEDDING_MODEL_DIR_ENV} to point elsewhere."
            )

        missing = [
            name
            for name in EMBEDDING_MODEL_REQUIRED_FILES
            if not (self._directory / name).is_file()
        ]
        if missing:
            raise LocalModelUnavailable(
                f"model directory {self._directory} is missing {', '.join(missing)}. "
                "A partially copied model would otherwise fail much later, inside a "
                "library, with an error that does not name the real cause."
            )

    def _resolve_device(self, torch: Any) -> Any:
        """Turn the configured device name into a torch device, or refuse.

        No fallback. See the module docstring: a CPU run is rejected, so
        continuing on the CPU silently is worse than stopping loudly.
        """
        if self._device_name.startswith("cuda") and not torch.cuda.is_available():
            raise LocalModelUnavailable(
                f"{EMBEDDING_DEVICE_ENV}={self._device_name!r} was requested but torch "
                "cannot see a GPU. Not falling back to the CPU: a run that never uses "
                "the GPU is rejected, so a silent fallback would turn a visible error "
                f"into an invisible disqualification. Set {EMBEDDING_DEVICE_ENV}=cpu "
                "only for development on a machine without a GPU."
            )
        return torch.device(self._device_name)

    def _verify_width(self) -> None:
        """Check the loaded model really is as wide as config claims.

        The dimension is written into the manifest and used to validate the
        vector buffer on load, so a model that disagreed with it would produce
        an index that misreads itself.
        """
        declared = getattr(self._model.config, "hidden_size", None)
        if declared is not None and declared != EMBEDDING_DIMENSION:
            raise LocalModelUnavailable(
                f"config.EMBEDDING_DIMENSION is {EMBEDDING_DIMENSION} but the model's "
                f"hidden_size is {declared}"
            )

    # --- The provider contract ---------------------------------------------

    @property
    def model_id(self) -> str:
        """Recorded in the index manifest; encodes model, pooling and width."""
        return EMBEDDING_MODEL_ID

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    @property
    def device(self) -> str:
        """Where the model actually sits. Read by diagnostics, not by retrieval."""
        return str(self._device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One 384-dimension unit vector per input, in the same order.

        Returns plain Python floats rather than tensors: the caller writes these
        into a binary float buffer, and a tensor that outlives its graph would
        pin device memory for the life of the process.
        """
        if not texts:
            return []

        pooling = _POOLING_FUNCTIONS[self._pooling_name]
        vectors: list[list[float]] = []

        with self._torch.inference_mode():
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQUENCE_TOKENS,
                    return_tensors="pt",
                ).to(self._device)

                hidden = self._model(**encoded).last_hidden_state
                pooled = pooling(hidden, encoded["attention_mask"])

                # Normalised here, at the boundary, so that every consumer gets
                # unit vectors and a dot product *is* the cosine. VectorRetriever
                # normalises again defensively; doing it twice is harmless, and
                # relying on the caller to do it would not be.
                unit = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(row.tolist() for row in unit.cpu())

        return vectors
