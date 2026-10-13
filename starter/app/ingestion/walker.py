"""Defensive corpus traversal.

**The one rule that matters here.** A single unreadable file must never cost us
the rest of the corpus. The graded corpus deliberately contains a file with no
read permission, an encrypted PDF, a file of an unknown binary type and an
empty directory. A walk that raises on the first of these indexes nothing after
it -- and because traversal follows the directory listing, *which* files you
lose depends on filename order. That is precisely how a submission passes
locally and grades badly.

So every failure is caught per entry, recorded in a ledger, and the walk
continues. Nothing in this module raises because of something found in the
corpus.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import config
from ingestion import filetypes
from models import DiscoveredFile, DiscoveryResult, SkipReason, SkippedEntry


def discover(
    root: Path | str,
    probe_bytes: int = config.PROBE_BYTES,
) -> DiscoveryResult:
    """Walk ``root`` and report every file we could and could not use.

    Args:
        root: The corpus root directory.
        probe_bytes: How many bytes to read from each file to prove it is
            readable and to identify its format.

    Returns:
        A :class:`~models.DiscoveryResult` whose ``files`` list is sorted by
        ``rel_path`` and whose ``skipped`` list is the failure ledger.

    The result is independent of the order the filesystem returns entries in:
    directories and filenames are sorted as we descend, and both output lists
    are sorted before returning. That independence is what makes it safe to
    reason about "what did we lose?" without knowing the listing order.
    """
    root_path = Path(root).resolve()
    result = DiscoveryResult(root=root_path)

    if not root_path.is_dir():
        # A missing corpus root is a configuration problem, not a corpus
        # hazard. Recording it keeps the "never raise" promise while refusing
        # to look like a corpus that simply had no files in it.
        result.skipped.append(
            SkippedEntry(
                rel_path=".",
                reason=SkipReason.DIRECTORY_UNREADABLE,
                detail=f"corpus root is not a directory: {root_path}",
            )
        )
        return result

    def on_walk_error(error: OSError) -> None:
        # os.walk calls this when it cannot list a directory. Without the
        # callback it would skip that directory silently, and a whole subtree
        # would vanish from the index with no record of why.
        result.skipped.append(
            SkippedEntry(
                rel_path=_relative(root_path, error.filename),
                reason=SkipReason.DIRECTORY_UNREADABLE,
                detail=str(error),
            )
        )

    for dirpath, dirnames, filenames in os.walk(
        root_path, followlinks=False, onerror=on_walk_error
    ):
        result.directory_count += 1

        # Sorting these lists in place is the documented way to make os.walk
        # descend deterministically: it uses the list object we mutate.
        dirnames.sort()
        filenames.sort()

        _drop_symlinked_directories(root_path, dirpath, dirnames, result)

        if not dirnames and not filenames:
            result.empty_directories.append(_relative(root_path, dirpath))

        for name in filenames:
            _consider_file(root_path, Path(dirpath) / name, result, probe_bytes)

    result.files.sort(key=lambda discovered: discovered.rel_path)
    result.skipped.sort(key=lambda entry: entry.rel_path)
    result.empty_directories.sort()
    return result


def _drop_symlinked_directories(
    root: Path,
    dirpath: str,
    dirnames: list[str],
    result: DiscoveryResult,
) -> None:
    """Remove symlinked directories from the walk, recording each one.

    Following a directory symlink can leave the corpus entirely or form a
    cycle that never terminates. Neither is worth the risk, so they are
    reported rather than followed. Symlinked *files* are still considered --
    a symlink to a real file inside the corpus is harmless.
    """
    for name in list(dirnames):
        candidate = Path(dirpath) / name
        if candidate.is_symlink():
            dirnames.remove(name)
            result.skipped.append(
                SkippedEntry(
                    rel_path=_relative(root, candidate),
                    reason=SkipReason.SYMLINKED_DIRECTORY,
                    detail="not followed: a symlink may leave the corpus or form a loop",
                )
            )


def _consider_file(
    root: Path,
    path: Path,
    result: DiscoveryResult,
    probe_bytes: int,
) -> None:
    """Classify one file, recording a ledger entry instead of raising."""
    rel_path = _relative(root, path)

    try:
        info = path.stat()
    except OSError as exc:
        result.skipped.append(SkippedEntry(rel_path, SkipReason.STAT_FAILED, str(exc)))
        return

    if not stat.S_ISREG(info.st_mode):
        # Sockets, FIFOs and device nodes live on filesystems, not in corpora,
        # but stat() is the only way to know, and reading one could block.
        result.skipped.append(
            SkippedEntry(
                rel_path,
                SkipReason.NOT_A_REGULAR_FILE,
                f"file mode {stat.filemode(info.st_mode)}",
            )
        )
        return

    try:
        # Opening is the only trustworthy readability test. os.access() consults
        # mode bits that root ignores, and mounted Windows drives report modes
        # they do not enforce -- both would report "readable" for a file that
        # cannot actually be read.
        with open(path, "rb") as handle:
            head = handle.read(probe_bytes)
    except PermissionError as exc:
        result.skipped.append(SkippedEntry(rel_path, SkipReason.PERMISSION_DENIED, str(exc)))
        return
    except OSError as exc:
        result.skipped.append(SkippedEntry(rel_path, SkipReason.UNREADABLE, str(exc)))
        return

    verdict = filetypes.sniff(head, path.suffix)
    result.files.append(
        DiscoveredFile(
            path=path,
            rel_path=rel_path,
            media_type=verdict.media_type,
            extension=path.suffix.lower(),
            size_bytes=info.st_size,
            magic_agrees=verdict.agrees,
        )
    )


def _relative(root: Path, target: os.PathLike[str] | str | None) -> str:
    """Express ``target`` as a corpus-relative POSIX path.

    This is the citation form the contract grades: ``specs/tq40_datasheet_r2.pdf``,
    never an absolute path and never a bare filename. Both ``root`` and
    ``target`` come from the same walk, so the relative computation succeeds;
    the fallback exists only so a diagnostic about an unexpected path cannot
    itself raise.
    """
    if target is None:
        return ""
    path = Path(target)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
