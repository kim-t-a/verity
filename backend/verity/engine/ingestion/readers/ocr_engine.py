"""Finding the OCR engine, and running it on one image.

The narrowest seam in the project. Everything either side of it is a pure
function -- :mod:`ingestion.readers.ocr_tsv` turns a string into detections and
:mod:`ingestion.readers.image_geometry` turns detections into relationships --
and this module is the part that cannot be either, because it starts a process.
Keeping it separate is what lets the two pure halves be tested exhaustively
while the untestable part stays small enough to read in one sitting.

**It never runs a shell.** The image path comes from the corpus, which is
untrusted input, and it is passed as one element of an argument vector rather
than interpolated into a command line. A file called ``; rm -rf /`` is a
filename here, not a command, and there is no quoting rule to get wrong because
there is no quoting.

**The engine is a command, not a path.** The value is split with
:func:`shlex.split`, so it may carry arguments as well as a program name. That
is not decoration: the container runs ``tesseract`` from ``PATH``, while a
development machine may only have a copy unpacked somewhere with its libraries
beside it and need ``sh /path/to/wrapper`` to start it. Treating the setting as
one opaque path would make the second case unexpressible, and would also make
this module awkward to test without an engine installed -- which is the property
that keeps the tests honest.

**Absence is not failure.** :func:`find_engine` returning ``None`` means this
build has no engine: the image has not been read, and the reader says so with
``VISION_UNAVAILABLE``. An engine that is *found* and then fails on a particular
file is a different finding and raises :class:`EngineFailed`, because those two
must not be reported the same way -- one is a property of the build, the other
is a property of the file.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from verity.engine.config import (
    DEFAULT_OCR_ENGINE,
    OCR_ENGINE_ENV,
    OCR_LANGUAGE,
    OCR_PAGE_SEGMENTATION_MODE,
    OCR_TIMEOUT_SECONDS,
)

#: Tesseract's output base meaning "write the result to standard output".
#:
#: Using it means no temporary file is created, so there is nothing to clean up
#: on the failure paths and nothing to leave behind in the image.
_STDOUT = "stdout"

#: The configuration name that selects TSV rather than plain text.
_TSV_CONFIG = "tsv"

#: How much of the engine's error output is kept for the diagnostic. Enough to
#: name the problem, bounded so that one verbose failure cannot bloat the
#: corpus report.
_MAX_ERROR_CHARS = 200


def _split_engine_command(configured: str) -> list[str]:
    """Split the engine setting into an argument vector.

    On a POSIX host the setting follows shlex rules, so it may quote a path
    with spaces and carry arguments. On Windows the POSIX rules are wrong:
    the default splitter treats ``\\`` as an escape character, so it eats
    every backslash in a Windows path and ``D:\\mc3-disk\\...`` arrives as
    ``D:mc3-disk...``, which never resolves. There the split is done with
    POSIX splitting off -- which preserves backslashes, the only change that
    matters -- and each part is unwrapped of the surrounding quotes it keeps.
    """
    if os.name == "nt":
        parts = shlex.split(configured, posix=False)
        return [
            part[1:-1]
            if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'"
            else part
            for part in parts
        ]
    return shlex.split(configured)


class EngineFailed(Exception):
    """The engine was found and could not read this file.

    Distinct from the engine being absent, which is reported by
    :func:`find_engine` returning ``None`` rather than by raising. The reader
    maps this to ``UNREADABLE`` and absence to ``VISION_UNAVAILABLE``, because
    "this file is broken" and "this build cannot read images" call for different
    responses from whoever reads the report.
    """


def find_engine() -> list[str] | None:
    """The command that starts the OCR engine, or ``None`` when there is none.

    Returns:
        An argument-vector prefix, ready to have the image path appended. The
        first element is resolved through ``PATH`` so that a bare name works,
        and any further elements are passed through unchanged.

        ``None`` when nothing resolves -- which is a real answer, not an error.
        A container that has not had the engine installed reports every image as
        ``VISION_UNAVAILABLE``, and that is the honest description of it.
    """
    configured = os.environ.get(OCR_ENGINE_ENV, "").strip() or DEFAULT_OCR_ENGINE

    try:
        parts = _split_engine_command(configured)
    except ValueError:
        # Unbalanced quotes in the environment. There is no engine to run, and
        # the alternative -- raising -- would abort the whole indexing pass over
        # a typo in a shell profile.
        return None

    if not parts:
        return None

    resolved = shutil.which(parts[0])
    if resolved is None:
        return None

    return [resolved, *parts[1:]]


def recognition_argv(
    engine: Sequence[str],
    path: Path,
    *,
    mode: int = OCR_PAGE_SEGMENTATION_MODE,
    language: str = OCR_LANGUAGE,
) -> list[str]:
    """The exact command line that recognises one image.

    Split out from :func:`recognise` so the argument order can be asserted
    without starting a process. It is worth asserting: Tesseract takes the image
    and the output base as the first two positional arguments and only then its
    options, so an option placed before the output base is read as the output
    base, and the failure is a run that writes nothing and reports success.

    Args:
        engine: The prefix from :func:`find_engine`.
        path: The image to recognise.
        mode: Page-segmentation mode. See ``OCR_PAGE_SEGMENTATION_MODE``.
        language: Recognition language.

    Returns:
        The argument vector, with ``path`` as a single element rather than part
        of a command string.
    """
    return [
        *engine,
        str(path),
        _STDOUT,
        "-l",
        language,
        "--psm",
        str(mode),
        _TSV_CONFIG,
    ]


def recognise(
    engine: Sequence[str],
    path: Path,
    *,
    timeout: float = OCR_TIMEOUT_SECONDS,
) -> bytes:
    """Run the engine over one image and return the TSV it wrote.

    The bytes are returned undecoded. Tesseract writes UTF-8, but it is an
    external program's output over an untrusted file and need not be valid
    UTF-8; leaving the decode to
    :func:`ingestion.readers.ocr_tsv.words_from_tsv` means a damaged byte costs
    one character instead of raising here, where there is no detection to
    attribute it to.

    Args:
        engine: The prefix from :func:`find_engine`.
        path: The image to recognise.
        timeout: Seconds to allow before killing the child.

    Returns:
        Whatever the engine wrote to standard output.

    Raises:
        EngineFailed: The engine could not be started, was killed for running
            too long, or exited non-zero. The message names which, because a
            missing shared library and a corrupt image are fixed by different
            people.
    """
    argv = recognition_argv(engine, path)

    try:
        completed = subprocess.run(  # noqa: S603 - argv, not a shell string
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
            # The child must not be able to read this process's standard input.
            # An engine that decided to prompt would otherwise block until the
            # harness killed the run.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngineFailed(f"the OCR engine did not finish within {timeout:g}s") from exc
    except OSError as exc:
        # The engine was on PATH but could not be executed: a missing shared
        # library, a bad interpreter, a permissions problem.
        raise EngineFailed(f"could not start the OCR engine: {exc}") from exc

    if completed.returncode != 0:
        raise EngineFailed(
            f"the OCR engine exited {completed.returncode}: {_first_error(completed.stderr)}"
        )

    return completed.stdout


def _first_error(stderr: bytes) -> str:
    """One line of the engine's error output, for the diagnostic."""
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return "no error output"
    first = text.splitlines()[0].strip()
    if len(first) > _MAX_ERROR_CHARS:
        return first[:_MAX_ERROR_CHARS] + "..."
    return first
