"""The repository's promise that secrets never enter git.

``AGENTS.md`` tells a contributor to put the provider key in ``.env`` and says
that file is git-ignored, and the product plan's first phase does exactly that.
A promise written in a document is not a guarantee, so this asserts the ignore
rules that make it true.

It is worth a test rather than a comment because the failure mode is not
recoverable the way most are: a key committed once is in the history for good,
and the only remedy is rotating it. The test is cheap, and it fails the moment
someone drops the pattern while tidying ``.gitignore``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def is_ignored(rel_path: str) -> bool:
    """Whether git would refuse to add ``rel_path``.

    ``--no-index`` is deliberately not passed: the question is about paths git
    tracks, and answering it must not depend on whether the file happens to
    exist yet -- the whole point is to catch the rule before a key is written.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def test_a_dotenv_file_is_ignored_wherever_it_sits() -> None:
    """At the root, next to the backend package, and next to the frontend."""
    assert is_ignored(".env")
    assert is_ignored("backend/.env")
    assert is_ignored("frontend/.env")
    assert is_ignored("backend/.env.local")


def test_the_example_file_is_not_ignored() -> None:
    """Names must be shareable; only values are secret.

    If the example were ignored too, the documented variable names would exist
    on one machine and nowhere else.
    """
    assert not is_ignored(".env.example")
