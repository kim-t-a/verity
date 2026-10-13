"""Shared pytest configuration and fixtures.

Two jobs:

1. Put ``starter/app`` on ``sys.path`` so tests import the application the same
   way the container does. See the comment below -- this is not a convenience,
   it is what makes a green test meaningful.
2. Provide a miniature corpus containing every hazard the brief warns about,
   built fresh in a temporary directory for each test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "starter" / "app"
CORPUS_DIR = REPO_ROOT / "mc3-corpus"

# The Dockerfile copies `starter/app/` to `/app/`, and the harness runs
# `python3 /app/app.py`, which places /app on sys.path. The application's modules
# therefore import each other by bare name (`import config`), and we reproduce
# that here rather than importing them as `starter.app.*`. Testing a differently
# wired import graph than the one that runs in the container is how a suite goes
# green while the container fails.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@dataclass
class HazardCorpus:
    """A synthetic corpus plus what a correct walk should find in it."""

    root: Path
    unreadable: Path
    unreadable_rel: str
    permissions_enforced: bool
    readable: list[str] = field(default_factory=list)


def _permissions_are_enforced(directory: Path) -> bool:
    """True only when ``chmod 000`` genuinely denies reads in this location.

    It does not on the Windows-mounted project drive: drvfs fabricates POSIX
    modes without enforcing them, so a file set to mode 000 still reads
    normally. It also never does for root, which bypasses mode bits entirely.

    Detecting this lets the permission tests *skip with a stated reason* rather
    than pass while proving nothing -- a test that cannot fail is worse than no
    test, because it reports safety that does not exist.
    """
    probe = directory / ".permission-probe"
    probe.write_text("probe", encoding="utf-8")
    try:
        probe.chmod(0o000)
        try:
            probe.read_text(encoding="utf-8")
        except PermissionError:
            return True
        return False
    finally:
        probe.chmod(0o600)
        probe.unlink(missing_ok=True)


@pytest.fixture
def hazard_corpus(tmp_path: Path) -> HazardCorpus:
    """Build a corpus containing every hazard the challenge warns about.

    The structure deliberately mirrors the real one: an empty directory, an
    unreadable file that sorts **first** within its directory (so a walk that
    aborts on it would lose everything after it), an encrypted PDF, and a file
    of an unknown binary type.
    """
    root = tmp_path / "corpus"
    (root / "archive").mkdir(parents=True)  # deliberately empty
    (root / "engineering").mkdir()
    (root / "specs").mkdir()
    (root / "support").mkdir()
    (root / "vendor").mkdir()

    unreadable_rel = "vendor/a_unreadable.txt"

    contents: dict[str, bytes] = {
        "engineering/service.py": b"DEFAULT_BATCH_TIMEOUT_S = 180\n",
        "engineering/notes.txt": b"Nothing in this file is graded.\n",
        "specs/datasheet.pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\n",
        # JPEG files begin FF D8 FF; the trailing byte makes the sample explicit.
        "support/asset.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        "support/parts.csv": b"part,note\nORR-FAN-2214-B,dual rotor\n",
        unreadable_rel: b"This file exists to be unreadable.\n",
        "vendor/b_encrypted.pdf": b"%PDF-1.4\n% /Encrypt present in the trailer\n",
        # An unknown binary type: every byte value, no recognisable signature.
        "vendor/c_unknown.dat": bytes(range(256)) * 4,
    }
    for rel_path, payload in contents.items():
        (root / rel_path).write_bytes(payload)

    unreadable = root / unreadable_rel
    enforced = _permissions_are_enforced(tmp_path)
    if enforced:
        unreadable.chmod(0o000)

    readable = sorted(rel for rel in contents if rel != unreadable_rel)
    if not enforced:
        # Without enforcement the file stays readable, so a correct walk finds it.
        readable = sorted(contents)

    corpus = HazardCorpus(
        root=root,
        unreadable=unreadable,
        unreadable_rel=unreadable_rel,
        permissions_enforced=enforced,
        readable=readable,
    )
    yield corpus

    # tmp_path cleanup can remove a mode-000 file (unlink needs write permission
    # on the directory, not the file), but restoring the mode keeps the
    # directory readable for anything that inspects it afterwards.
    if enforced:
        unreadable.chmod(0o600)
