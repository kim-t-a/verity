"""Reading Tesseract's TSV: from an engine's own output to :class:`Detection`.

The geometry in :mod:`ingestion.readers.image_geometry` decides what belongs
with what, and it does so from boxes alone. This module is the other half of
that seam: it turns the thing Tesseract actually prints into those boxes. It is
the only place in the project that knows the engine's output format, and it is
a pure function of a string -- no subprocess, no file, no engine -- so it is
tested against recorded output rather than against an installation.

**Every rule below was measured, not assumed.** Tesseract 5.3.4 was run over
both sample images during Stage 7B and the output inspected directly, because
Stage 7A had already shown that two of its tolerance constants were wrong until
a real image contradicted the guess. Three findings changed what this module
has to do, and all three are the kind that produce silent, plausible-looking
wrong answers rather than errors:

* **The confidence column is a float.** Real values are ``91.944946``, not
  ``92``. The column is documented as an integer percentage, and ``int()`` on
  it raises -- which is the good outcome. A version that rounded to ``92``
  would have been worse, because nothing would have failed.
* **Diagram rules are reported as words.** A drawn border or table rule comes
  back as a full word-level row whose text is a single space. It is not an
  empty string, so a ``if not text`` check passes it straight through, and the
  pinout at ``--psm 3`` emits fifteen of them out of fifty-five rows.
* **Only level 5 is a word.** Tesseract's TSV is a hierarchy, and levels 1 to 4
  repeat the text of the words beneath them as page, block, paragraph and line
  aggregates. Keeping them would put every word in the index three or four
  times over and quietly wreck the term statistics.

**Where the two halves divide.** This module drops a row only when the *format*
says it is not a word -- the wrong level, the wrong number of fields, a
coordinate that is not a number. It deliberately does **not** decide that a
space or a lone ``-`` is not text: ``_is_usable`` in the geometry already makes
that decision, for its own stated reason, and a second copy of the rule here
would be a second place for it to drift. Measured on the real pinout, all
fifteen rule rows reach the geometry and all fifteen are dropped there.

**What it never does.** It never repairs a row. A line with the wrong number of
fields is counted and skipped rather than having its tail rejoined into the
text field, because guessing how a corrupt row was meant to be shaped is how a
value that appears nowhere in the image ends up in the index. Nothing enters a
:class:`Detection` that the engine did not print.
"""

from __future__ import annotations

from dataclasses import dataclass

from .image_geometry import Detection

#: The columns Tesseract's TSV writer emits, in the order it emits them. The
#: adapter looks each one up by name rather than by position, so a build that
#: adds a column shifts nothing, and a reader of this file can see which fields
#: are actually load-bearing.
TSV_HEADER = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)

#: The columns this adapter cannot do without. Deliberately a subset of
#: :data:`TSV_HEADER`: the block, paragraph and line numbering are the engine's
#: internal bookkeeping and nothing downstream reads them, so a format that
#: stopped emitting them would still be readable here.
_REQUIRED_COLUMNS = frozenset({"level", "left", "top", "width", "height", "conf", "text"})

#: The only level whose rows are words. See the module docstring: levels 1 to 4
#: restate the words beneath them, so admitting them multiplies the corpus.
_WORD_LEVEL = 5

#: Tesseract reports confidence as a percentage on 0..100. ``Detection`` is
#: defined on 0..1, so the one conversion happens here and nowhere else.
#:
#: A confidence of ``-1`` is the engine declining to answer, which it writes for
#: things that are not text. It is scaled and passed through rather than dropped
#: here, for the reason in the module docstring: low confidence is a judgement
#: about content, and that judgement belongs to ``MIN_OCR_CONFIDENCE`` in the
#: geometry, which is where it is already made.
_CONFIDENCE_SCALE = 100.0


@dataclass(frozen=True)
class TsvWords:
    """What one TSV blob yielded, and what could not be read from it.

    The two counts exist because ``no detections`` has two very different
    causes and they must not look alike. ``header_recognised`` false means the
    engine did not produce TSV at all -- an error message, a truncated write, a
    wrong command line -- and the image has *not* been read. It true with no
    detections means the image was read and had no words in it. The first is a
    fault to report; the second is a legitimate empty result. Collapsing them
    would let a broken invocation masquerade as a blank image.
    """

    detections: tuple[Detection, ...] = ()
    """The word-level rows, in the order the engine printed them. Order is not
    meaningful: the geometry sorts by position and assigns reading order
    itself."""

    skipped_rows: int = 0
    """Word-level rows that could not be read as words -- the wrong number of
    fields, or a coordinate that is not a number.

    Counts only rows that claimed to be words. The page, block and paragraph
    rows are not counted, because they are not skipped: they are not words, and
    a large and uninteresting number here would bury the real corruption."""

    header_recognised: bool = False
    """Whether the first line was a TSV header this adapter understands."""


def _decode(raw: str | bytes) -> str:
    """Tesseract writes UTF-8; anything else is replaced rather than raised on.

    A byte sequence that is not valid UTF-8 has to cost at most the character
    it appears in. Letting ``UnicodeDecodeError`` escape would abort the
    indexing pass over one damaged glyph, which the walker's contract forbids.
    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def words_from_tsv(raw: str | bytes) -> TsvWords:
    """Turn one Tesseract TSV blob into the words it contains.

    Pure and total: it reads no file, calls no engine, and returns a value for
    every input including input that is not TSV at all. Malformed rows are
    counted and skipped, never repaired and never raised on -- an engine's
    output is untrusted input, and one bad row must not cost the whole image.

    Args:
        raw: The engine's stdout, as text or as the bytes it was written in.

    Returns:
        A :class:`TsvWords`. Callers should check ``header_recognised`` before
        treating an empty ``detections`` as an empty image.
    """
    lines = _decode(raw).splitlines()
    if not lines:
        return TsvWords()

    header = lines[0].rstrip("\r").split("\t")
    position = {name: index for index, name in enumerate(header)}
    if not _REQUIRED_COLUMNS.issubset(position):
        return TsvWords()

    column = {name: position[name] for name in _REQUIRED_COLUMNS}
    width = len(header)

    detections: list[Detection] = []
    skipped = 0

    for line in lines[1:]:
        fields = line.rstrip("\r").split("\t")
        if len(fields) != width:
            # Cannot tell which field is missing, so the text column cannot be
            # trusted either. Count it and move on; see the module docstring.
            skipped += 1
            continue
        try:
            # Compared as a float so that a writer emitting ``5.0`` still reads
            # as a word rather than being counted as corrupt.
            if float(fields[column["level"]]) != _WORD_LEVEL:
                continue
            detection = Detection(
                text=fields[column["text"]],
                confidence=float(fields[column["conf"]]) / _CONFIDENCE_SCALE,
                left=float(fields[column["left"]]),
                top=float(fields[column["top"]]),
                width=float(fields[column["width"]]),
                height=float(fields[column["height"]]),
            )
        except ValueError:
            # Only reachable from a word-level row, since the level itself has
            # to parse before any other column is read.
            skipped += 1
            continue
        detections.append(detection)

    return TsvWords(
        detections=tuple(detections),
        skipped_rows=skipped,
        header_recognised=True,
    )
