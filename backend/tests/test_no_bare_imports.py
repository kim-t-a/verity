"""No engine module may import a neighbour by bare name.

``from models import Chunk`` only ever worked because Docker flattened
``starter/app/`` to ``/app/``, which put the directory on ``sys.path``. Now
that ``verity`` is a real package, a bare import is a bug that would resolve
only by accident -- from whichever directory happened to be the working
directory. This test makes the old style impossible rather than merely unused.
"""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "verity" / "engine"

BARE_NAMES = {
    "config",
    "models",
    "normalisation",
    "tokenisation",
    "chunking",
    "chunkstore",
    "ingestion",
    "retrieval",
    "embeddings",
    "answering",
    "app",
}


def imported_roots(source: Path) -> list[str]:
    """Every top-level module name ``source`` imports.

    Public rather than private so the detector itself can be exercised against
    a tree known to contain bare imports -- a guard nobody has watched fail is
    a guard nobody knows works.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is legitimate inside a package.
            if node.level == 0 and node.module:
                roots.append(node.module.split(".")[0])
    return roots


def bare_import_offenders(root: Path) -> list[str]:
    """Files under ``root`` that import a neighbour by bare name."""
    offenders: list[str] = []
    for source in sorted(root.rglob("*.py")):
        for name in imported_roots(source):
            if name in BARE_NAMES:
                offenders.append(f"{source.relative_to(root)} imports {name!r}")
    return offenders


def test_no_engine_module_imports_a_bare_neighbour() -> None:
    offenders = bare_import_offenders(ENGINE)
    assert offenders == [], f"bare imports in the engine: {offenders}"
